"""Train-bank loss helpers aligned with ``src/train`` ce_saliency training.

Default hypers match the common training CLI::

    --loss_mode ce_saliency
    --saliency_loss_type contrastive
    --saliency_lambda 1.5
    --saliency_alpha 1.0
    --saliency_margin_plus 2.0
    --saliency_layer -1
    --saliency_neg_sample_k 64

ce_only     -> CE
ce_saliency -> CE + λ * contrastive saliency (negatives subsampled with k=64)
"""
from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


# Defaults aligned with train.py CLI used for ce_saliency + contrastive runs.
_DEFAULT_SALIENCY_LOSS_TYPE = "contrastive"
_DEFAULT_SALIENCY_LAMBDA = 1.5
_DEFAULT_SALIENCY_ALPHA = 1.0
_DEFAULT_MARGIN_PLUS = 2.0
_DEFAULT_SALIENCY_LAYER = -1
_DEFAULT_NEG_SAMPLE_K = 64


@dataclass
class BankLossConfig:
    loss_mode: str = "ce_only"  # ce_only | ce_saliency
    saliency_loss_type: str = _DEFAULT_SALIENCY_LOSS_TYPE
    saliency_lambda: float = _DEFAULT_SALIENCY_LAMBDA
    alpha: float = _DEFAULT_SALIENCY_ALPHA
    eps: float = 1e-8
    margin_plus: float = _DEFAULT_MARGIN_PLUS
    neg_sample_k: int = _DEFAULT_NEG_SAMPLE_K
    saliency_layer: int = _DEFAULT_SALIENCY_LAYER
    exclude_sink_prefix: int = 0
    exclude_special_tokens: bool = False

    @property
    def cache_tag(self) -> str:
        if self.loss_mode == "ce_only":
            return "ce"
        return (
            f"cesal_{self.saliency_loss_type}_lam{self.saliency_lambda:g}"
            f"_m{self.margin_plus:g}_k{self.neg_sample_k}"
        )


def infer_bank_loss_mode(model_path: str | None, model_tag: str) -> str:
    """Prefer adapter-local saliency_training_config.json; else infer from tag.

    ``saliency`` / ``contrastive`` in the path wins over a coincidental
    ``ce_only`` substring (e.g. ``.../ce_saliency``).
    """
    if model_path:
        cfg_path = Path(model_path) / "saliency_training_config.json"
        if cfg_path.is_file():
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                mode = str(raw.get("loss_mode") or "").strip()
                if mode in ("ce_only", "ce_saliency", "saliency_only"):
                    return "ce_saliency" if mode != "ce_only" else "ce_only"
            except Exception:
                pass
    tag = (model_tag or "").lower()
    # Prefer saliency markers first so paths like ".../ce_saliency" are correct.
    if "saliency" in tag or "contrastive" in tag or "cesal" in tag:
        return "ce_saliency"
    if "ce_only" in tag or tag.endswith("_ce") or tag == "ce":
        return "ce_only"
    return "ce_only"


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_bank_loss_config(
    model_path: str | None,
    model_tag: str,
    *,
    loss_mode_override: str | None = None,
) -> BankLossConfig:
    """Load bank objective; defaults match train contrastive + k=64.

    Override sources (later wins only within each field via json/env):
      - ``BankLossConfig`` train-aligned defaults (incl. neg_sample_k=64)
      - adapter ``saliency_training_config.json`` when present
      - env: ``EIF_BANK_LOSS_MODE``, ``EIF_BANK_SALIENCY_LAMBDA``,
        ``EIF_BANK_SALIENCY_NEG_SAMPLE_K``, …
    """
    raw: dict = {}
    if model_path:
        cfg_path = Path(model_path) / "saliency_training_config.json"
        if cfg_path.is_file():
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                raw = {}

    override = (loss_mode_override or "").strip().lower()
    if not override:
        override = (os.environ.get("EIF_BANK_LOSS_MODE") or "").strip().lower()
    if override in ("", "auto", "none"):
        mode = infer_bank_loss_mode(model_path, model_tag)
    elif override in ("ce_only", "ce_saliency"):
        mode = override
    elif override == "saliency_only":
        mode = "ce_saliency"
    else:
        raise ValueError(
            f"Unknown bank loss_mode_override={loss_mode_override!r}; "
            "use auto|ce_only|ce_saliency"
        )

    loss_type = str(
        raw.get("saliency_loss_type")
        or (os.environ.get("EIF_BANK_SALIENCY_LOSS_TYPE") or "").strip()
        or _DEFAULT_SALIENCY_LOSS_TYPE
    )
    lam = float(
        raw["saliency_lambda"]
        if "saliency_lambda" in raw
        else _env_float("EIF_BANK_SALIENCY_LAMBDA", _DEFAULT_SALIENCY_LAMBDA)
    )
    if "saliency_temperature_tau" in raw:
        alpha = float(raw["saliency_temperature_tau"])
    elif "saliency_alpha" in raw:
        alpha = float(raw["saliency_alpha"])
    else:
        alpha = _env_float("EIF_BANK_SALIENCY_ALPHA", _DEFAULT_SALIENCY_ALPHA)
    margin_plus = float(
        raw["saliency_margin_plus"]
        if "saliency_margin_plus" in raw
        else _env_float("EIF_BANK_SALIENCY_MARGIN_PLUS", _DEFAULT_MARGIN_PLUS)
    )
    if "saliency_neg_sample_k" in raw:
        neg_k = int(raw.get("saliency_neg_sample_k") or 0)
    elif (os.environ.get("EIF_BANK_SALIENCY_NEG_SAMPLE_K") or "").strip() != "":
        neg_k = _env_int("EIF_BANK_SALIENCY_NEG_SAMPLE_K", _DEFAULT_NEG_SAMPLE_K)
    else:
        neg_k = _DEFAULT_NEG_SAMPLE_K
    layer = int(
        raw["saliency_layer"]
        if "saliency_layer" in raw
        else _env_int("EIF_BANK_SALIENCY_LAYER", _DEFAULT_SALIENCY_LAYER)
    )

    cfg = BankLossConfig(
        loss_mode=mode,
        saliency_loss_type=loss_type,
        saliency_lambda=lam,
        alpha=alpha,
        eps=float(raw.get("saliency_eps_num", 1e-8)),
        margin_plus=margin_plus,
        neg_sample_k=neg_k,
        saliency_layer=layer,
        exclude_sink_prefix=int(raw.get("saliency_exclude_sink_prefix", 0) or 0),
        exclude_special_tokens=bool(raw.get("saliency_exclude_special_tokens", False)),
    )
    print(
        f"[bank-loss] mode={cfg.loss_mode} type={cfg.saliency_loss_type} "
        f"λ={cfg.saliency_lambda} α={cfg.alpha} margin+={cfg.margin_plus} "
        f"neg_k={cfg.neg_sample_k} layer={cfg.saliency_layer} tag={cfg.cache_tag}",
        flush=True,
    )
    return cfg


def _import_saliency_loss_from_outputs():
    """Load contrastive saliency loss used by ce_saliency train-bank grads.

    Prefers the vendored copy at ``src/saliency_loss.py`` (moved out of
    code-corr-annotation). Falls back to the old CCA path if present.
    """
    try:
        from src.saliency_loss import saliency_loss_from_outputs
        return saliency_loss_from_outputs
    except ImportError:
        pass

    # Fallback: legacy code-corr-annotation/src/train/loss.py
    import sys

    here = Path(__file__).resolve().parent
    root = here.parent
    candidates = [
        root / "code-corr-annotation" / "src" / "train",
        root.parent / "code-corr-annotation" / "src" / "train",
    ]
    for train_dir in candidates:
        if (train_dir / "loss.py").is_file():
            train_dir_s = str(train_dir)
            if train_dir_s not in sys.path:
                sys.path.insert(0, train_dir_s)
            from loss import saliency_loss_from_outputs  # type: ignore
            return saliency_loss_from_outputs

    raise ImportError(
        "Cannot import saliency_loss_from_outputs. Expected src/saliency_loss.py "
        "(or legacy code-corr-annotation/src/train/loss.py)."
    )


def _annot_pairs_from_edges(
    edges, n_tokens: int, device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (pairs [N,2], weights [N]) for one sample; weight defaults to 1.0."""
    pairs: list[list[int]] = []
    weights: list[float] = []
    for e in edges or []:
        try:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                a, b = int(e[0]), int(e[1])
                w = float(e[2]) if len(e) >= 3 else 1.0
            else:
                a = int(e.get("src", e.get("source", -1)))
                b = int(e.get("dst", e.get("target", -1)))
                try:
                    w = float(e.get("weight", 1.0))
                except (TypeError, ValueError):
                    w = 1.0
        except (TypeError, ValueError, AttributeError):
            continue
        qi, qj = (a, b) if a < b else (b, a)
        if 0 <= qi < qj < n_tokens:
            pairs.append([qi, qj])
            weights.append(max(1.0, w))
    if not pairs:
        empty_p = torch.zeros(0, 2, dtype=torch.long, device=device)
        empty_w = torch.zeros(0, dtype=torch.float32, device=device)
        return [empty_p], [empty_w]
    return (
        [torch.tensor(pairs, dtype=torch.long, device=device)],
        [torch.tensor(weights, dtype=torch.float32, device=device)],
    )


def _sdpa_context():
    """Prefer flash / mem-efficient SDPA; avoid MATH which materializes HxTxT per layer.

    With ``enable_input_require_grads`` + long ChatML, MATH SDPA saves ~H·T² per
    decoder layer for backward and OOMs around seq≈2k on 80–96GB GPUs.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
        ])
    except Exception:
        return contextlib.nullcontext()


def _capture_layer_input_hidden(model, layer_index: int):
    """Forward-pre-hook to keep only the saliency layer's input (not all hidden_states)."""
    from src.saliency_loss import _unwrap_to_decoder_stack

    decoder = _unwrap_to_decoder_stack(model)
    n_layers = len(decoder.layers)
    li = int(layer_index) if int(layer_index) >= 0 else n_layers + int(layer_index)
    li = max(0, min(li, n_layers - 1))
    holder: dict[str, torch.Tensor | None] = {"hid": None, "li": li, "n_layers": n_layers}

    def _pre_hook(_module, args):
        # Decoder layer forward(hidden_states, ...)
        if args and torch.is_tensor(args[0]):
            holder["hid"] = args[0]

    handle = decoder.layers[li].register_forward_pre_hook(_pre_hook)
    return holder, handle


def compute_bank_loss(
    model,
    batch,
    *,
    device,
    cfg: BankLossConfig,
    edges=None,
    special_ids: set[int] | None = None,
):
    """Scalar training objective for one bank example (CE or CE+saliency)."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    inputs = {"input_ids": input_ids, "labels": labels}
    if "attention_mask" in batch:
        inputs["attention_mask"] = batch["attention_mask"].to(device)

    need_saliency = cfg.loss_mode == "ce_saliency" and bool(edges)
    if not need_saliency:
        with _sdpa_context():
            outputs = model(**inputs, use_cache=False, return_dict=True)
        loss = outputs.loss
        if loss is None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss, "ce_only"

    # CRITICAL: do NOT set output_attentions=True — materializes HxTxT for every layer.
    # Also avoid output_hidden_states=True (keeps all layer states); hook one layer instead.
    # Prefer flash/mem-efficient SDPA so training does not fall back to MATH TxT.
    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    print(
        f"[bank-loss] ce_saliency forward attn_impl={attn_impl!r} "
        f"seq={int(input_ids.size(1))} (single-layer attn recompute; no all-hidden)",
        flush=True,
    )

    li_req = int(cfg.saliency_layer)
    holder, hook = _capture_layer_input_hidden(model, li_req)
    try:
        with _sdpa_context():
            outputs = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        hook.remove()

    ce = outputs.loss
    if ce is None:
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    if ce.dim() > 0:
        ce = ce.mean()
    # Drop the big vocab tensor from the live namespace (graph may still retain it for CE).
    try:
        outputs.logits = None
    except Exception:
        pass

    hid_in = holder.get("hid")
    if hid_in is None:
        raise RuntimeError(
            "Failed to capture saliency-layer input hidden state via forward hook."
        )
    li = int(holder["li"])
    n_layers = int(holder["n_layers"])

    from types import SimpleNamespace
    from src.loss import _recompute_layer_attn_probs

    saliency_loss_from_outputs = _import_saliency_loss_from_outputs()
    n_tokens = int(input_ids.size(1))
    attn_sel = _recompute_layer_attn_probs(model, hid_in, layer_index=li)
    slim_hidden = [None] * (n_layers + 1)
    slim_hidden[li] = hid_in
    fake_attns = [None] * n_layers
    fake_attns[li] = attn_sel
    sal_outputs = SimpleNamespace(
        attentions=tuple(fake_attns),
        hidden_states=tuple(slim_hidden),
        loss=ce,
        logits=None,
    )

    annot_pairs, annot_weights = _annot_pairs_from_edges(edges, n_tokens, device)
    exclude = None
    if cfg.exclude_sink_prefix > 0 or (cfg.exclude_special_tokens and special_ids):
        em = torch.zeros_like(input_ids, dtype=torch.bool)
        if cfg.exclude_sink_prefix > 0:
            em[:, : cfg.exclude_sink_prefix] = True
        if cfg.exclude_special_tokens and special_ids:
            special = torch.tensor(sorted(special_ids), device=input_ids.device, dtype=input_ids.dtype)
            em = em | torch.isin(input_ids, special)
        exclude = em

    diag = saliency_loss_from_outputs(
        model,
        sal_outputs,
        annot_pairs,
        annot_weights=annot_weights,
        saliency_layer=li,
        exclude_source_mask=exclude,
        alpha=cfg.alpha,
        eps=cfg.eps,
        floor_eps=0.0,
        floor_eps_mode="fixed",
        floor_eps_step=0,
        floor_eps_warmup_steps=0,
        floor_logit_eps=None,
        loss_type=cfg.saliency_loss_type,
        margin_plus=cfg.margin_plus,
        neg_sample_k=cfg.neg_sample_k,
    )
    return ce + float(cfg.saliency_lambda) * diag.loss, "ce_saliency"
