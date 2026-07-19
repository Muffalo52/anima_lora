# Shared network base for additional network modules (like LyCORIS-family modules: LoHa, LoKr, etc).
# Provides architecture detection and a generic AdditionalNetwork class.

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from library.log import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArchConfig:
    unet_target_modules: List[str]
    te_target_modules: List[str]
    unet_prefix: str
    te_prefixes: List[str]
    default_excludes: List[str] = field(default_factory=list)
    adapter_target_modules: List[str] = field(default_factory=list)
    unet_conv_target_modules: List[str] = field(default_factory=list)


def detect_arch_config(unet, text_encoders) -> ArchConfig:
    # Check Anima: look for Block class in named_modules
    module_class_names = set()
    if unet is not None:
        for module in unet.modules():
            module_class_names.add(type(module).__name__)

    if "Block" in module_class_names:
        return ArchConfig(
            unet_target_modules=["Block", "PatchEmbed", "TimestepEmbedding", "FinalLayer"],
            te_target_modules=["Qwen3Attention", "Qwen3MLP", "Qwen3SdpaAttention", "Qwen3FlashAttention2"],
            unet_prefix="lora_unet",
            te_prefixes=["lora_te"],
            default_excludes=[r".*(_modulation|_norm|_embedder|final_layer).*"],
            adapter_target_modules=["LLMAdapterTransformerBlock"],
        )

    raise ValueError(f"Cannot auto-detect architecture for LyCORIS. Module classes found: {sorted(module_class_names)}")


def _parse_kv_pairs(kv_pair_str: str, is_int: bool) -> Dict[str, Union[int, float]]:
    """Parse a string of key-value pairs separated by commas."""
    pairs = {}
    for pair in kv_pair_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            logger.warning(f"Invalid format: {pair}, expected 'key=value'")
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            pairs[key] = int(value) if is_int else float(value)
        except ValueError:
            logger.warning(f"Invalid value for {key}: {value}")
    return pairs


# Top-level model component prefixes whose parameters are NOT "hidden"
# layers (embeddings, input/output projections, timestep conditioning).
# Checked against ``original_name`` which is the dotted path into the
# root model (e.g. ``time_embedding.linear_1``).
_NON_HIDDEN_NAME_PREFIXES = (
    'time_embedding', 'time_in', 'timestep_embedding',
    'vector_in', 'guidance_in',
    'img_in', 'txt_in',
    'conv_in', 'conv_out',
    'final_layer',
    'x_embedder', 'pos_embedder', 'patch_embed', 'context_embedder',
)


def tag_lora_module_params(lora_module):
    """Tag a LoRA/OFT module's ``nn.Parameter`` objects with optimizer-relevant
    attributes so that Advanced_Optimizers can identify each parameter's role.

    Sets the following attributes on the appropriate parameters:

    * ``_is_dora_scale``  — DoRA magnitude scale
    * ``_is_oft``         — OFT skew-symmetric blocks
    * ``_is_lora_A``      — LoRA down/A factor
    * ``_is_lora_B``      — LoRA up/B factor
    * ``is_hidden``       — 2D hidden-layer weight (determined via
      ``original_name`` heuristic)
    * ``is_vector``       — logically-vector parameter (multi-dim)

    ``is_hidden`` is determined by checking ``original_name`` (the dotted
    path into the root model) against ``_NON_HIDDEN_NAME_PREFIXES`` — a set
    of known top-level component prefixes for embeddings, input/output
    projections, and timestep conditioning layers.

    Args:
        lora_module: A ``torch.nn.Module`` representing a single adapter
            module (e.g. ``LoRAModule``, ``OFTModule``).
    """
    import torch.nn as nn

    # Determine if this module targets a hidden layer by checking
    # original_name against known non-hidden top-level components.
    original_name = getattr(lora_module, 'original_name', None) or ''
    is_hidden = not any(original_name.startswith(pfx)
                        for pfx in _NON_HIDDEN_NAME_PREFIXES)

    # --- OFT blocks ---
    oft_blocks = getattr(lora_module, 'oft_blocks', None)
    if isinstance(oft_blocks, nn.Parameter):
        oft_blocks._is_oft = True
    # OFTModule in oft_flux.py stores oft_blocks as a ParameterList
    if isinstance(oft_blocks, nn.ParameterList):
        for p in oft_blocks:
            if isinstance(p, nn.Parameter):
                p._is_oft = True

    # --- DoRA scale ---
    dora_scale = getattr(lora_module, 'dora_scale', None)
    if isinstance(dora_scale, nn.Parameter):
        dora_scale._is_dora_scale = True
        dora_scale.is_vector = True

    # --- Standard LoRA down/up ---
    for attr, tag in (
        ('lora_down', '_is_lora_A'),
        ('lora_up', '_is_lora_B'),
    ):
        sub = getattr(lora_module, attr, None)
        # Single module (Linear/Conv2d)
        if isinstance(sub, nn.Module) and hasattr(sub, 'weight'):
            w = sub.weight
            if isinstance(w, nn.Parameter):
                setattr(w, tag, True)
                w.is_hidden = is_hidden
        # ModuleList (split-dims in lora_flux.py / lora_lumina.py)
        if isinstance(sub, nn.ModuleList):
            for mod in sub:
                if hasattr(mod, 'weight') and isinstance(mod.weight, nn.Parameter):
                    setattr(mod.weight, tag, True)
                    mod.weight.is_hidden = is_hidden

    # --- Fallback: tag all remaining 2D trainable params ---
    for p in lora_module.parameters(recurse=False):
        if not isinstance(p, nn.Parameter):
            continue
        if p.ndim < 2:
            continue
        if getattr(p, '_is_oft', False):
            continue
        if getattr(p, '_is_lora_A', False) or getattr(p, '_is_lora_B', False):
            continue
        p.is_hidden = is_hidden


def _tag_all_network_params(network):
    """Tag all adapter modules' parameters in a network with optimizer-relevant
    attributes.  Call this from ``prepare_optimizer_params()`` and
    ``prepare_grad_etc()`` so attributes survive device moves.

    Args:
        network: A network object with ``text_encoder_loras`` and/or
            ``unet_loras`` lists (or a ``loras`` list for the generic
            ``AdditionalNetwork``).
    """
    loras = getattr(network, 'text_encoder_loras', []) + getattr(network, 'unet_loras', [])
    if not loras:
        # Fallback: AdditionalNetwork stores everything in a single 'loras' list
        loras = getattr(network, 'loras', [])
    for lora in loras:
        tag_lora_module_params(lora)


class AdditionalNetwork(torch.nn.Module):
    """Generic Additional network that supports LoHa, LoKr, and similar module types.

    Constructed with a module_class parameter to inject the specific module type.
    Based on the lora_anima.py LoRANetwork, generalized for multiple architectures.
    """

    def __init__(
        self,
        text_encoders: list,
        unet,
        arch_config: ArchConfig,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[torch.nn.Module] = None,
        module_kwargs: Optional[Dict] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        reg_dims: Optional[Dict[str, int]] = None,
        reg_lrs: Optional[Dict[str, float]] = None,
        train_llm_adapter: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        assert module_class is not None, "module_class must be specified"

        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.train_llm_adapter = train_llm_adapter
        self.reg_dims = reg_dims
        self.reg_lrs = reg_lrs
        self.arch_config = arch_config

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if module_kwargs is None:
            module_kwargs = {}

        if modules_dim is not None:
            logger.info(f"create {module_class.__name__} network from weights")
        else:
            logger.info(f"create {module_class.__name__} network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )

        # compile regular expressions
        def str_to_re_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
            re_patterns = []
            if patterns is not None:
                for pattern in patterns:
                    try:
                        re_pattern = re.compile(pattern)
                    except re.error as e:
                        logger.error(f"Invalid pattern '{pattern}': {e}")
                        continue
                    re_patterns.append(re_pattern)
            return re_patterns

        exclude_re_patterns = str_to_re_patterns(exclude_patterns)
        include_re_patterns = str_to_re_patterns(include_patterns)

        # create module instances
        def create_modules(
            prefix: str,
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            default_dim: Optional[int] = None,
        ) -> Tuple[List[torch.nn.Module], List[str]]:
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_modules is None or module.__class__.__name__ in target_replace_modules:
                    if target_replace_modules is None:
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            lora_name = f"{prefix}.{original_name}".replace(".", "_")

                            # exclude/include filter
                            excluded = any(pattern.fullmatch(original_name) for pattern in exclude_re_patterns)
                            included = any(pattern.fullmatch(original_name) for pattern in include_re_patterns)
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha_val = modules_alpha[lora_name]
                            else:
                                if self.reg_dims is not None:
                                    for reg, d in self.reg_dims.items():
                                        if re.fullmatch(reg, original_name):
                                            dim = d
                                            alpha_val = self.alpha
                                            logger.info(f"Module {original_name} matched with regex '{reg}' -> dim: {dim}")
                                            break
                                # fallback to default dim
                                if dim is None:
                                    if is_linear or is_conv2d_1x1:
                                        dim = default_dim if default_dim is not None else self.lora_dim
                                        alpha_val = self.alpha
                                    elif is_conv2d and self.conv_lora_dim is not None:
                                        dim = self.conv_lora_dim
                                        alpha_val = self.conv_alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha_val,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                **module_kwargs,
                            )
                            lora.original_name = original_name
                            loras.append(lora)

                    if target_replace_modules is None:
                        break
            return loras, skipped

        # Create modules for text encoders
        self.text_encoder_loras: List[torch.nn.Module] = []
        skipped_te = []
        if text_encoders is not None:
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue

                # Determine prefix for this text encoder
                if i < len(arch_config.te_prefixes):
                    te_prefix = arch_config.te_prefixes[i]
                else:
                    te_prefix = arch_config.te_prefixes[0]

                logger.info(f"create {module_class.__name__} for Text Encoder {i+1} (prefix={te_prefix}):")
                te_loras, te_skipped = create_modules(te_prefix, text_encoder, arch_config.te_target_modules)
                logger.info(f"create {module_class.__name__} for Text Encoder {i+1}: {len(te_loras)} modules.")
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # Create modules for UNet/DiT
        target_modules = list(arch_config.unet_target_modules)
        if modules_dim is not None or (conv_lora_dim is not None and conv_lora_dim != 0):
            target_modules.extend(arch_config.unet_conv_target_modules)
        if train_llm_adapter and arch_config.adapter_target_modules:
            target_modules.extend(arch_config.adapter_target_modules)

        self.unet_loras: List[torch.nn.Module]
        self.unet_loras, skipped_un = create_modules(arch_config.unet_prefix, unet, target_modules)
        logger.info(f"create {module_class.__name__} for UNet/DiT: {len(self.unet_loras)} modules.")

        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        # assertion: no duplicate names
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable modules for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable modules for UNet/DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        te_prefixes = self.arch_config.te_prefixes
        unet_prefix = self.arch_config.unet_prefix

        for key in weights_sd.keys():
            if any(key.startswith(p) for p in te_prefixes):
                apply_text_encoder = True
            elif key.startswith(unet_prefix):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable modules for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable modules for UNet/DiT")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            pass  # already a list with one element

        self.requires_grad_(True)
        _tag_all_network_params(self)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            reg_groups = {}
            reg_lrs_list = list(self.reg_lrs.items()) if self.reg_lrs is not None else []

            for lora in loras:
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    if re.fullmatch(regex_str, lora.original_name):
                        matched_reg_lr = (i, reg_lr)
                        logger.info(f"Module {lora.original_name} matched regex '{regex_str}' -> LR {reg_lr}")
                        break

                for name, param in lora.named_parameters():
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {"lora": {}, "plus": {}, "lr": reg_lr}
                        # LoRA+ detection: check for "up" weight parameters
                        if loraplus_ratio is not None and self._is_plus_param(name):
                            reg_groups[group_key]["plus"][f"{lora.lora_name}.{name}"] = param
                        else:
                            reg_groups[group_key]["lora"][f"{lora.lora_name}.{name}"] = param
                        continue

                    if loraplus_ratio is not None and self._is_plus_param(name):
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = reg_lr * loraplus_ratio if loraplus_ratio is not None else reg_lr
                    else:
                        param_data["lr"] = reg_lr
                    if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                        logger.info("NO LR skipping!")
                        continue
                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    descriptions.append(desc + (" plus" if key == "plus" else ""))

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr
                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")
            return params, descriptions

        if self.text_encoder_loras:
            loraplus_ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            # Group TE loras by prefix
            for te_idx, te_prefix in enumerate(self.arch_config.te_prefixes):
                te_loras = [lora for lora in self.text_encoder_loras if lora.lora_name.startswith(te_prefix)]
                if len(te_loras) > 0:
                    te_lr = text_encoder_lr[te_idx] if te_idx < len(text_encoder_lr) else text_encoder_lr[0]
                    logger.info(f"Text Encoder {te_idx+1} ({te_prefix}): {len(te_loras)} modules, LR {te_lr}")
                    params, descriptions = assemble_params(te_loras, te_lr, loraplus_ratio)
                    all_params.extend(params)
                    lr_descriptions.extend([f"textencoder {te_idx+1}" + (" " + d if d else "") for d in descriptions])

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def _is_plus_param(self, name: str) -> bool:
        """Check if a parameter name corresponds to a 'plus' (higher LR) param for LoRA+.

        For LoRA: lora_up. For LoHa: hada_w2_a (the second pair). For LoKr: lokr_w1 (the scale factor).
        Override in subclass if needed. Default: check for common 'up' patterns.
        """
        return "lora_up" in name or "hada_w2_a" in name or "lokr_w1" in name

    def enable_gradient_checkpointing(self):
        pass  # not supported

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)
        _tag_all_network_params(self)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False
