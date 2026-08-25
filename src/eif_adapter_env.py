"""Resolve which adapter to load from report location / env.

Reports live under::

    correlation_matching_results/ce/*.json
    correlation_matching_results/saliency/*.json

Env (see ``eif_api.env.example``)::

    EIF_ADAPTER_PATH_CE
    EIF_ADAPTER_PATH_SALIENCY
    EIF_BASE_MODEL_PATH
    EIF_ADAPTER_PATH          # legacy single-adapter fallback
    EIF_SALIENCY_BANK_PATH_CE / _SALIENCY / legacy EIF_SALIENCY_BANK_PATH
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

ReportFamily = Literal["ce", "saliency", "unknown"]

_FAMILY_DIRS = ("ce", "saliency")


def normalize_report_relpath(report_file_name: str) -> str:
    """Normalize ``ce/foo.json`` / ``ce\\foo.json`` → posix relative path."""
    raw = (report_file_name or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.lstrip("/")


def infer_report_family(
    report_file_name: str | None = None,
    report: dict[str, Any] | None = None,
) -> ReportFamily:
    """Infer adapter family from report path or stamped meta."""
    if report:
        meta = report.get("experiment_meta") or {}
        stamped = str(meta.get("report_family") or meta.get("adapter_family") or "").strip().lower()
        if stamped in ("ce", "ce_only"):
            return "ce"
        if stamped in ("saliency", "ce_saliency"):
            return "saliency"
        name = str(meta.get("report_file") or meta.get("fileName") or "").strip()
        if name:
            report_file_name = name

    rel = normalize_report_relpath(report_file_name or "")
    if not rel:
        return "unknown"
    top = rel.split("/", 1)[0].lower()
    if top == "ce":
        return "ce"
    if top == "saliency":
        return "saliency"

    # Legacy flat filenames: guess from stem tags.
    low = rel.lower()
    if "ce_only" in low or "/ce/" in f"/{low}" or low.startswith("ce_"):
        return "ce"
    if "saliency" in low or "cesal" in low:
        return "saliency"
    return "unknown"


def resolve_report_json_path(corr_results_dir: Path, report_file_name: str) -> Path | None:
    """Resolve a report path under ``correlation_matching_results`` (safe)."""
    rel = normalize_report_relpath(report_file_name)
    if not rel or ".." in rel.split("/"):
        return None
    parts = rel.split("/")
    if len(parts) == 1:
        # Legacy: flat file in corr root.
        cand = (corr_results_dir / parts[0]).resolve()
    elif len(parts) == 2 and parts[0].lower() in (*_FAMILY_DIRS, "raw"):
        cand = (corr_results_dir / parts[0].lower() / parts[1]).resolve()
    else:
        return None
    try:
        cand.relative_to(corr_results_dir.resolve())
    except ValueError:
        return None
    return cand if cand.is_file() else None


def _env_path(*keys: str) -> str:
    for key in keys:
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    return ""


# After continue-train, live probes (token probs / learn) can use the new
# adapter without rewriting eif_api.env. Recover clears this override.
_ACTIVE_ADAPTER_OVERRIDE: str | None = None
_ACTIVE_ADAPTER_SOURCE: str | None = None  # e.g. "continue"


def get_active_adapter_override() -> str | None:
    return _ACTIVE_ADAPTER_OVERRIDE


def get_active_adapter_status() -> dict[str, Any]:
    env_saliency = _env_path("EIF_ADAPTER_PATH_SALIENCY", "EIF_ADAPTER_PATH", "EIF_MODEL_PATH")
    return {
        "overrideActive": bool(_ACTIVE_ADAPTER_OVERRIDE),
        "overridePath": _ACTIVE_ADAPTER_OVERRIDE,
        "source": _ACTIVE_ADAPTER_SOURCE,
        "envAdapterPath": env_saliency or None,
    }


def set_active_adapter_override(
    path: str | None,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Set/clear process-wide live adapter override (does not edit eif_api.env)."""
    global _ACTIVE_ADAPTER_OVERRIDE, _ACTIVE_ADAPTER_SOURCE
    if path is None or not str(path).strip():
        _ACTIVE_ADAPTER_OVERRIDE = None
        _ACTIVE_ADAPTER_SOURCE = None
    else:
        resolved = str(Path(path).expanduser().resolve())
        _ACTIVE_ADAPTER_OVERRIDE = resolved
        _ACTIVE_ADAPTER_SOURCE = source or "override"
    return get_active_adapter_status()


def adapter_path_for_family(family: ReportFamily) -> str:
    """Pick adapter directory for a report family.

    If a continue-train (or other) override is active, prefer that path so
    token-probs / learn / gold share the continued weights until recover.
    """
    if _ACTIVE_ADAPTER_OVERRIDE:
        return _ACTIVE_ADAPTER_OVERRIDE
    return env_adapter_path_for_family(family)


def env_adapter_path_for_family(family: ReportFamily | str) -> str:
    """Pinned CE / saliency path from eif_api.env (ignores continue-train override)."""
    if family == "ce":
        return _env_path("EIF_ADAPTER_PATH_CE", "EIF_ADAPTER_PATH", "EIF_MODEL_PATH")
    if family == "saliency":
        return _env_path("EIF_ADAPTER_PATH_SALIENCY", "EIF_ADAPTER_PATH", "EIF_MODEL_PATH")
    return _env_path("EIF_ADAPTER_PATH", "EIF_MODEL_PATH")


def base_model_path_from_env() -> str:
    return _env_path("EIF_BASE_MODEL_PATH")


def normalize_view_family(raw: str | None) -> str:
    """Normalize UI/API adapter-view ids: live | ce | saliency | base."""
    s = (raw or "live").strip().lower()
    if s in ("", "live", "current", "active", "model"):
        return "live"
    if s in ("ce", "ce_only"):
        return "ce"
    if s in ("saliency", "ce_saliency", "sal"):
        return "saliency"
    if s in ("base", "pretrained"):
        return "base"
    raise ValueError(f"Unknown viewFamily={raw!r} (use live|ce|saliency|base)")


def list_compare_views(live_family: ReportFamily | str | None = None) -> list[dict[str, Any]]:
    """Tabs for the next-token panel: 当前 + env adapters + base."""
    live = (live_family or "unknown")
    if live not in ("ce", "saliency"):
        live = "unknown"
    views: list[dict[str, Any]] = [
        {"id": "live", "label": "当前", "family": live},
    ]
    ce = env_adapter_path_for_family("ce")
    sal = env_adapter_path_for_family("saliency")
    base = base_model_path_from_env()
    if ce:
        views.append({"id": "ce", "label": "CE", "family": "ce", "path": ce})
    if sal:
        views.append({"id": "saliency", "label": "Saliency", "family": "saliency", "path": sal})
    if base:
        views.append({"id": "base", "label": "Base", "family": "base", "path": base})
    return views


def bank_loss_mode_for_family(family: ReportFamily) -> str | None:
    """Default bank objective when ``EIF_BANK_LOSS_MODE`` is unset/auto."""
    override = (os.environ.get("EIF_BANK_LOSS_MODE") or "").strip().lower()
    if override and override not in ("", "auto", "none"):
        return None  # caller keeps explicit override
    if family == "ce":
        return "ce_only"
    if family == "saliency":
        return "ce_saliency"
    return None


def bank_path_for_family(family: ReportFamily) -> str:
    """Optional pinned ``.pt`` bank path for a report family."""
    if family == "ce":
        return _env_path("EIF_SALIENCY_BANK_PATH_CE", "EIF_SALIENCY_BANK_PATH")
    if family == "saliency":
        return _env_path("EIF_SALIENCY_BANK_PATH_SALIENCY", "EIF_SALIENCY_BANK_PATH")
    return _env_path("EIF_SALIENCY_BANK_PATH")


def resolve_bank_file(raw: str, *, repo_root: Path | None = None) -> Path | None:
    """Resolve a bank path (absolute, cwd-relative, or repo-relative)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        if repo_root is not None:
            candidates.append(repo_root / p)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def stamp_report_family(report: dict[str, Any], report_file_name: str) -> ReportFamily:
    """Write ``experiment_meta.report_family`` / ``report_file`` for downstream code."""
    family = infer_report_family(report_file_name, report)
    meta = dict(report.get("experiment_meta") or {})
    meta["report_file"] = normalize_report_relpath(report_file_name)
    if family != "unknown":
        meta["report_family"] = family
    report["experiment_meta"] = meta
    return family
