from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..model.config import Config
from ..model.model import Model
from ..modules import Embedding, Linear, GatedResidual
from ..modules.module import Module
from ..modules.arch_specific.qwen4_exp_mtp import Qwen4ExpMTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qwen4_exp import Qwen4ExpConfig

"""
MTP (multi-token prediction) draft head for Qwen3.8-Flash-Next: input combine over the trunk's
PRE-collapse hyper-connection stream stack (exported by the trunk's final mixer) plus the next
token's embedding, one full qwen4_exp decoder block (QSA attention, MoE, gated-residual sites),
and its own combine-less mixer. Shares the trunk's embedding and lm_head.

No reference implementation exists for this head; the input-combine stream handling
(Qwen4ExpMTPInputLayer.stream_tap) is a semantic guess that must be confirmed by acceptance
rate on the full model.
"""


class Qwen4ExpMTPStackOut(Module):
    """
    Terminal module of the MTP draft chain: passes the decoder block's stream stack through
    FLATTENED (bsz, seq, hc_mult * hidden) instead of collapsing it, so the model's forward
    output can feed the next drafting step's target_hidden (symmetric with the trunk's pre-mixer
    stack export). The mixer is owned here as a submodule (so it loads with the model) and is
    applied by sample_from_state() before the shared lm_head.
    """

    def __init__(self, config, key: str, mixer: GatedResidual):
        super().__init__(config, key, None)
        self.mixer = mixer
        self.register_submodule(mixer)

    def optimizer_targets(self):
        return []

    # The compile step collects a top-level module's output tensors by ITS key prefix; this
    # module's own key ("mtp_stack_out") names no tensors, so it must hand the collection to
    # the owned mixer (prefix "mtp.hyper_connection_mixer."), or the mixer's tensors stay in
    # the qtensors files and never reach the compiled shards
    def get_compile_sizes(self, stc):
        return self.mixer.get_compile_sizes(stc)

    def get_compile_tensors(self, stc):
        return self.mixer.get_compile_tensors(stc)

    def forward(self, x, params, out_dtype = None):
        return x.flatten(-2).half()


class Qwen4ExpMTPModel(Model):

    def __init__(
        self,
        config: Qwen4ExpConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        from .qwen4_exp import build_qwen4_block

        self.input_layer = Qwen4ExpMTPInputLayer(
            config = config,
            key = "mtp",
            hidden_size = config.hidden_size,
            hc_mult = config.hc_mult,
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        )
        self.modules = [self.input_layer]
        self.first_block_idx = len(self.modules)

        for idx in range(config.mtp_num_hidden_layers):
            self.modules.append(
                build_qwen4_block(
                    config,
                    f"mtp.layers.{idx}",
                    idx,
                    "full_attention",
                    qbits_key = "mtp_bits",
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # The draft chain's output is the flattened PRE-mixer stream stack (it feeds the next
        # drafting step's target_hidden); sample_from_state applies the mixer + shared lm_head
        self.stack_out = Qwen4ExpMTPStackOut(
            config,
            "mtp_stack_out",
            GatedResidual(
                config = config,
                key = "mtp.hyper_connection_mixer",
                hc_mult = config.hc_mult,
                hidden_size = config.hidden_size,
                rms_norm_eps = config.rms_norm_eps,
                use_combine = False,
                out_dtype = torch.half,
            ),
        )
        self.modules.append(self.stack_out)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 4,
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None
        # Populated by attach_to() under TP: a full lm_head replica on the output device
        self.local_head = None

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")

    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and have the trunk's final mixer
        export the pre-collapse stream stack as the draft input state.
        """
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)

        if target.loaded_tp:
            # Co-located draft under TP: bind the borrows to loaded tensors in this process so
            # the drafting loop pays no producer round trip per token. The output-device inline
            # worker runs in the main process and its module list holds a loaded replica of the
            # replicated embedding. The sharded lm_head cannot be borrowed that way, so a full
            # replica is loaded onto the output device from the stc (~0.48 GB; measured
            # 2.69 GiB free on GPU 0 under the full TP4+MTP stack).
            import torch as _torch
            inline = target.mp_parent_conn[target.tp_output_device]
            embed_replica = inline.local_context["modules"][0]
            assert isinstance(embed_replica, Embedding), \
                "Inline worker's module 0 is not the Embedding replica"
            self.input_layer.local_embed = embed_replica
            self.target_embed = weakref.ref(embed_replica)

            free_b, _ = _torch.cuda.mem_get_info(target.tp_output_device)
            head = Linear(
                config = target.config,
                key = "lm_head",
                qbits_key = "head_bits",
                in_features = target.config.hidden_size,
                out_features = target.config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True},
            )
            head.load(_torch.device(target.tp_output_device))
            self.local_head = head
            free_a, _ = _torch.cuda.mem_get_info(target.tp_output_device)
            print(f" -- MTP co-location: local embedding bound, lm_head replica loaded "
                  f"({(free_b - free_a)/1024**3:.2f} GiB on device {target.tp_output_device})",
                  flush = True)
        else:
            target_embed = None
            for m in target.modules:
                if isinstance(m, Embedding):
                    target_embed = m
                    break
            assert target_embed is not None, "Could not locate target's Embedding module"
            self.target_embed = weakref.ref(target_embed)
            self.local_head = None

        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        target_mixer = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_mixer, GatedResidual) and not target_mixer.use_combine, \
            "Expected the trunk's combine-less mixer immediately before lm_head"
        self.draft_verifier_params.update({
            "export_state_norm_keys": {target_mixer.key},
        })

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, max_chunk_size):
        return {}

    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        # state is the flattened pre-mixer stream stack; collapse it before the shared head
        mixer = self.stack_out.mixer
        bsz, seq, _ = state.shape
        stack = state.to(mixer.device).view(bsz, seq, mixer.hc_mult, mixer.hidden_size)
        state = mixer.forward(stack, params)
        if self.local_head is not None:
            # Co-located head replica (attach_to under TP): collapse, project and argmax
            # locally on the output device -- no producer round trip per drafted token
            logits = self.local_head.prepare_for_device(state, params)
            logits = self.local_head.forward(logits, params)
            logits = logits[..., :self.config.vocab_size]
            if params.get("export_draft_conf"):
                conf, ids = torch.max(logits, dim = -1)
                params["draft_conf"] = conf
                return ids
            return torch.argmax(logits, dim = -1)

        target = self.attached_model()
        if target.loaded_tp:
            # The shared lm_head is sharded across ranks: ship the collapsed state to the
            # workers and take the sharded argmax (qwen3_5 MTP pattern). With draft
            # confidence enabled, also return the winning logit value.
            vocab_size = target.config.vocab_size
            sent = target.tp_producer.send(state)
            if params.get("export_draft_conf"):
                argmax, max_vals = target.tp_dispatch_lm_head_argmax(
                    (sent, {}), return_values = True, vocab_size = vocab_size)
                params["draft_conf"] = max_vals
                return argmax
            return target.tp_dispatch_lm_head_argmax((sent, {}), vocab_size = vocab_size)
        ll = target.logit_layer_idx
        lm = target.modules[ll]
        logits = lm.prepare_for_device(state, params)
        logits = lm.forward(logits, params)
        if params.get("export_draft_conf"):
            logits = logits[..., :target.config.vocab_size]
            conf, ids = torch.max(logits, dim = -1)
            params["draft_conf"] = conf
            return ids
        return torch.argmax(logits, dim = -1)
