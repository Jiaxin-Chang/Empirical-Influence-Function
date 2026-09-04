"""Propose a new semantic-prompt version from human logs and eval on hold-out.

Workflow:
  1. Group human add/bump/delete events by sample (full prompt+response stored).
  2. Hash-split samples into train / hold-out.
  3. Train → few-shot pack (text edges, never cross-sample indices).
  4. Hold-out → run full-sample LLM annotate, compare precision/recall.
  5. Activate the new version only if hold-out F1 is strictly better.

Query family (``query_name`` / ``query_expression``) is recorded on each log
event so later packs can be clustered per retrieval boolean. The first iterate
pass builds one mixed default pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VIEWER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(VIEWER_ROOT) not in sys.path:
    sys.path.insert(0, str(VIEWER_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(REPO_ROOT / "eif_api.env")

from server.human_annot_log import (  # noqa: E402
    gold_edges_for_sample,
    gold_index_edges_for_sample,
    group_by_sample,
    load_events,
    log_path,
    sample_key,
)
from server.semantic_prompt import (  # noqa: E402
    default_bundle,
    load_active,
    next_version_id,
    save_version,
    set_active,
)

MIN_TRAIN = 2
MIN_HOLD_OUT = 2
HOLD_OUT_MOD = 5  # ~20% hold-out


def _norm(text: Any) -> str:
    return str(text or "").replace("Ġ", " ").replace("▁", " ").strip()


def _is_hold_out(key: str) -> bool:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % HOLD_OUT_MOD == 0


def split_sample_keys(keys: list[str]) -> tuple[list[str], list[str]]:
    train: list[str] = []
    hold: list[str] = []
    for k in keys:
        (hold if _is_hold_out(k) else train).append(k)
    return train, hold


def _latest_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    return events[-1] if events else {}


def _sample_record(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    gold_idx = gold_index_edges_for_sample(events)
    gold_txt = gold_edges_for_sample(events)
    if not gold_idx and not gold_txt:
        return None
    last = _latest_event(events)
    prompt = str(last.get("prompt") or "")
    response = str(last.get("response") or "")
    if not prompt.strip() or not response.strip():
        return None
    return {
        "key": sample_key(last),
        "prompt": prompt,
        "response": response,
        "language": str(last.get("language") or "go"),
        "query_name": str(last.get("query_name") or ""),
        "query_expression": str(last.get("query_expression") or ""),
        "n_tokens": last.get("n_tokens"),
        "gold_idx": sorted(gold_idx),
        "gold_txt": sorted(gold_txt),
        "events": events,
    }


def usable_samples(events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    grouped = group_by_sample(events if events is not None else load_events())
    out: list[dict[str, Any]] = []
    for evs in grouped.values():
        rec = _sample_record(evs)
        if rec:
            out.append(rec)
    return out


def _prompt_excerpt(prompt: str, *, limit: int = 1200) -> str:
    text = str(prompt or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n…\n" + text[-(limit // 2) :]


def few_shots_from_train(
    records: list[dict[str, Any]],
    *,
    max_shots: int = 4,
) -> list[dict[str, Any]]:
    """One shot per retrieval family when possible; text edges only."""
    ranked = sorted(records, key=lambda r: len(r.get("gold_txt") or []), reverse=True)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for rec in ranked:
        fam = str(rec.get("query_name") or rec.get("query_expression") or "").strip() or "ungrouped"
        by_family.setdefault(fam, []).append(rec)

    picked: list[dict[str, Any]] = []
    used: set[str] = set()
    # Round-robin across families so one bool query cannot fill the pack.
    families = list(by_family.keys())
    while len(picked) < max_shots:
        progressed = False
        for fam in families:
            bucket = by_family[fam]
            if not bucket:
                continue
            rec = bucket.pop(0)
            key = str(rec.get("key") or "")
            if key in used:
                continue
            used.add(key)
            edges = [
                {"src_text": src, "dst_text": dst}
                for src, dst in (rec.get("gold_txt") or [])[:8]
            ]
            if not edges:
                continue
            picked.append({
                "prompt_excerpt": _prompt_excerpt(str(rec.get("prompt") or "")),
                "response": str(rec.get("response") or "")[:400],
                "edges": edges,
                "query_name": str(rec.get("query_name") or ""),
                "query_expression": str(rec.get("query_expression") or ""),
            })
            progressed = True
            if len(picked) >= max_shots:
                break
        if not progressed:
            break
    return picked


def propose_bundle(
    *,
    events: list[dict[str, Any]] | None = None,
    max_shots: int = 4,
) -> dict[str, Any]:
    samples = usable_samples(events)
    keys = [str(s["key"]) for s in samples]
    train_keys, hold_keys = split_sample_keys(keys)
    by_key = {str(s["key"]): s for s in samples}
    train_recs = [by_key[k] for k in train_keys if k in by_key]
    parent = load_active()
    shots = few_shots_from_train(train_recs, max_shots=max_shots)
    vid = next_version_id()
    return {
        "id": vid,
        "system": str(parent.get("system") or default_bundle()["system"]),
        "few_shots": shots,
        "parent_id": parent.get("id") or "default",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": None,
        "split": {
            "n_usable": len(samples),
            "n_train": len(train_keys),
            "n_hold_out": len(hold_keys),
            "train_keys": train_keys,
            "hold_out_keys": hold_keys,
        },
    }


def _pr(pred: set[Any], gold: set[Any]) -> dict[str, float]:
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "pred": 0, "gold": 0}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "pred": len(pred),
        "gold": len(gold),
    }


def _match_sets(
    pred_edges: list[dict[str, Any]],
    rec: dict[str, Any],
    tokens: list[str],
) -> tuple[set[Any], set[Any], str]:
    gold_idx = {(int(a), int(b)) for a, b in (rec.get("gold_idx") or [])}
    gold_txt = {(_norm(a), _norm(b)) for a, b in (rec.get("gold_txt") or [])}
    pred_idx = set()
    pred_txt = set()
    n = len(tokens)
    for e in pred_edges:
        try:
            src, dst = int(e["src"]), int(e["dst"])
        except (KeyError, TypeError, ValueError):
            continue
        pred_idx.add((src, dst))
        if 0 <= src < n and 0 <= dst < n:
            pred_txt.add((_norm(tokens[src]), _norm(tokens[dst])))
    stored_n = rec.get("n_tokens")
    if gold_idx and stored_n is not None and int(stored_n) == n:
        return pred_idx, gold_idx, "index"
    if gold_txt:
        return pred_txt, gold_txt, "text"
    return pred_idx, gold_idx, "index"


def evaluate_bundle(
    bundle: dict[str, Any],
    hold_records: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    from server.llm_semantic_annotate import annotate_prompt_response_semantic

    per: list[dict[str, Any]] = []
    for rec in hold_records:
        key = str(rec.get("key") or "")
        try:
            annotated = annotate_prompt_response_semantic(
                str(rec.get("prompt") or ""),
                str(rec.get("response") or ""),
                tokenizer,
                language=str(rec.get("language") or "go"),
                prompt_bundle=bundle,
            )
        except Exception as exc:
            per.append({"key": key, "error": str(exc)})
            continue
        tokens = list(annotated.get("tokens") or [])
        pred_edges = list(annotated.get("attention_edges") or [])
        pred, gold, mode = _match_sets(pred_edges, rec, tokens)
        stats = _pr(pred, gold)
        stats.update({"key": key, "match": mode, "n_pred_edges": len(pred_edges)})
        per.append(stats)

    scored = [s for s in per if "f1" in s]
    if scored:
        macro = {
            "precision": round(sum(s["precision"] for s in scored) / len(scored), 4),
            "recall": round(sum(s["recall"] for s in scored) / len(scored), 4),
            "f1": round(sum(s["f1"] for s in scored) / len(scored), 4),
            "n_eval": len(scored),
            "n_failed": len(per) - len(scored),
        }
    else:
        macro = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_eval": 0,
            "n_failed": len(per),
        }
    return {"macro": macro, "per_sample": per, "prompt_id": bundle.get("id")}


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    path = (
        (os.environ.get("EIF_BASE_MODEL_PATH") or "").strip()
        or (os.environ.get("ANNOTATION_TOKENIZER") or "").strip()
    )
    if not path or not Path(path).expanduser().exists():
        raise FileNotFoundError(
            "tokenizer not found — set EIF_BASE_MODEL_PATH in eif_api.env"
        )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def iterate(
    *,
    activate_if_better: bool = False,
    propose_only: bool = False,
    max_shots: int = 4,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    events = load_events()
    samples = usable_samples(events)
    keys = [str(s["key"]) for s in samples]
    train_keys, hold_keys = split_sample_keys(keys)
    by_key = {str(s["key"]): s for s in samples}
    train_recs = [by_key[k] for k in train_keys if k in by_key]
    hold_recs = [by_key[k] for k in hold_keys if k in by_key]

    candidate = propose_bundle(events=events, max_shots=max_shots)
    save_version(candidate)
    out: dict[str, Any] = {
        "ok": True,
        "candidate_id": candidate["id"],
        "parent_id": candidate.get("parent_id"),
        "log_path": str(log_path()),
        "n_events": len(events),
        "n_usable_samples": len(samples),
        "n_train": len(train_keys),
        "n_hold_out": len(hold_keys),
        "n_few_shots": len(candidate.get("few_shots") or []),
        "activated": False,
        "reason": "",
    }

    if propose_only:
        out["reason"] = "propose_only — version saved, not evaluated"
        return out
    if len(train_recs) < MIN_TRAIN:
        out["reason"] = (
            f"need ≥{MIN_TRAIN} train samples with net human edges "
            f"(have {len(train_recs)}); version saved, not activated"
        )
        return out
    if len(hold_recs) < MIN_HOLD_OUT:
        out["reason"] = (
            f"need ≥{MIN_HOLD_OUT} hold-out samples "
            f"(have {len(hold_recs)}); version saved, not activated"
        )
        return out

    tok = tokenizer if tokenizer is not None else _load_tokenizer()
    active = load_active()
    cand_eval = evaluate_bundle(candidate, hold_recs, tok)
    base_eval = evaluate_bundle(active, hold_recs, tok)
    candidate["metrics"] = {
        "candidate": cand_eval["macro"],
        "active": base_eval["macro"],
        "per_sample_candidate": cand_eval["per_sample"],
        "per_sample_active": base_eval["per_sample"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_version(candidate)
    out["candidate_metrics"] = cand_eval["macro"]
    out["active_metrics"] = base_eval["macro"]
    out["active_id"] = active.get("id") or "default"

    cand_f1 = float(cand_eval["macro"].get("f1") or 0)
    base_f1 = float(base_eval["macro"].get("f1") or 0)
    if activate_if_better and cand_f1 > base_f1:
        set_active(str(candidate["id"]))
        out["activated"] = True
        out["reason"] = (
            f"hold-out F1 {cand_f1:.4f} > active {base_f1:.4f} — now default"
        )
    elif activate_if_better:
        out["reason"] = (
            f"hold-out F1 {cand_f1:.4f} ≯ active {base_f1:.4f} — kept "
            f"{active.get('id') or 'default'}"
        )
    else:
        out["reason"] = (
            f"evaluated F1 candidate={cand_f1:.4f} active={base_f1:.4f}; "
            "pass --activate-if-better to replace default on improvement"
        )
    return out


def prompt_status() -> dict[str, Any]:
    events = load_events()
    samples = usable_samples(events)
    keys = [str(s["key"]) for s in samples]
    train_keys, hold_keys = split_sample_keys(keys)
    active = load_active()
    families: dict[str, int] = {}
    for rec in samples:
        fam = str(rec.get("query_name") or rec.get("query_expression") or "").strip() or "(no query)"
        families[fam] = families.get(fam, 0) + 1
    return {
        "ok": True,
        "active_id": active.get("id") or "default",
        "log_path": str(log_path()),
        "n_events": len(events),
        "n_usable_samples": len(samples),
        "n_train": len(train_keys),
        "n_hold_out": len(hold_keys),
        "query_families": families,
        "versions": _list_versions_safe(),
    }


def _list_versions_safe() -> list[str]:
    try:
        from server.semantic_prompt import list_version_ids

        return list_version_ids()
    except Exception:
        return ["default"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterate semantic annotate prompts from human logs")
    parser.add_argument("--activate-if-better", action="store_true")
    parser.add_argument("--propose-only", action="store_true")
    parser.add_argument("--max-shots", type=int, default=4)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(prompt_status(), ensure_ascii=False, indent=2))
        return
    result = iterate(
        activate_if_better=args.activate_if_better,
        propose_only=args.propose_only,
        max_shots=max(1, min(8, int(args.max_shots))),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
