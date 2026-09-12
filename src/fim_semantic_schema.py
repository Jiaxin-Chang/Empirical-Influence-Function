"""Structured FIM semantic representation (error query and train sample share one schema).

Boolean substring queries remain a baseline. This module normalizes the
structured fields and can rerank lexical hits by relation/pattern overlap
until a full embedding index exists.
"""

from __future__ import annotations

from typing import Any

SEMANTIC_LIST_KEYS = (
    "domain",
    "pattern",
    "entities",
    "operations",
    "conditions",
)

_FIELD_LIMITS = {
    "domain": 4,
    "pattern": 3,
    "entities": 8,
    "operations": 4,
    "conditions": 3,
}

# Coarse-to-fine weights: role/pattern/relations matter more than a bare entity name.
_FIELD_WEIGHTS = {
    "domain": 0.6,
    "pattern": 2.2,
    "entities": 1.0,
    "operations": 1.6,
    "conditions": 1.6,
}
_ROLE_WEIGHT = 2.4
_REL_BOTH = 3.2
_REL_ONE = 0.9


def _as_str_list(value: Any, *, limit: int = 12) -> list[str]:
    if isinstance(value, str):
        t = value.strip()
        return [t] if t else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, dict):
            text = str(
                item.get("name")
                or item.get("text")
                or item.get("value")
                or item.get("label")
                or ""
            ).strip()
        else:
            text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


RELATION_TYPES = (
    "dataflow",
    "control",
    "api",
    "transform",
    "init",
    "error",
    "config",
    "semantic_dependency",
    "other",
)

_RELATION_TYPE_ALIASES = {
    "data_flow": "dataflow",
    "flow": "dataflow",
    "uses": "dataflow",
    "use": "dataflow",
    "condition": "control",
    "guard": "control",
    "precondition": "control",
    "branch": "control",
    "call": "api",
    "invoke": "api",
    "conversion": "transform",
    "encode": "transform",
    "decode": "transform",
    "serialize": "transform",
    "hash": "transform",
    "setup": "init",
    "initialize": "init",
    "initialization": "init",
    "create": "init",
    "exception": "error",
    "err": "error",
    "failure": "error",
    "configuration": "config",
    "setting": "config",
    "depends": "semantic_dependency",
    "dependency": "semantic_dependency",
    "order": "semantic_dependency",
    "ordering": "semantic_dependency",
    "temporal": "semantic_dependency",
    "semantic": "other",
}


def canonicalize_relation_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in RELATION_TYPES:
        return raw
    return _RELATION_TYPE_ALIASES.get(raw, "other")


def _parse_relation(item: Any) -> dict[str, str] | None:
    if isinstance(item, dict):
        src = str(item.get("source") or item.get("src") or "").strip()
        dst = str(item.get("target") or item.get("dst") or item.get("to") or "").strip()
        typ = canonicalize_relation_type(item.get("type") or item.get("relation"))
        if not src or not dst:
            return None
        return {"source": src, "target": dst, "type": typ}
    text = str(item or "").strip()
    if not text:
        return None
    for sep in ("->", "→", "=>"):
        if sep in text:
            left, right = text.split(sep, 1)
            src, dst = left.strip(), right.strip()
            if src and dst:
                return {"source": src, "target": dst, "type": "other"}
    return None


def normalize_relations(value: Any, *, limit: int = 10) -> list[dict[str, str]]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        rel = _parse_relation(item)
        if rel is None:
            continue
        key = (rel["source"].lower(), rel["target"].lower(), rel["type"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(rel)
        if len(out) >= limit:
            break
    return out


def normalize_semantic_repr(obj: dict[str, Any] | None) -> dict[str, Any]:
    """Accept nested ``semantic`` or top-level fields; always return the canonical dict."""
    if not isinstance(obj, dict):
        return empty_semantic_repr()
    src = obj.get("semantic") if isinstance(obj.get("semantic"), dict) else obj
    if not isinstance(src, dict):
        src = {}
    summary = str(src.get("summary") or obj.get("gold_pattern_summary") or "").strip()
    role = str(src.get("role") or "").strip()
    out = {
        "role": role,
        "domain": _as_str_list(src.get("domain"), limit=_FIELD_LIMITS["domain"]),
        "pattern": _as_str_list(src.get("pattern"), limit=_FIELD_LIMITS["pattern"]),
        "entities": _as_str_list(src.get("entities"), limit=_FIELD_LIMITS["entities"]),
        "operations": _as_str_list(src.get("operations"), limit=_FIELD_LIMITS["operations"]),
        "conditions": _as_str_list(src.get("conditions"), limit=_FIELD_LIMITS["conditions"]),
        "relations": normalize_relations(src.get("relations"), limit=6),
        "summary": summary,
    }
    return out


def empty_semantic_repr() -> dict[str, Any]:
    return {
        "role": "",
        "domain": [],
        "pattern": [],
        "entities": [],
        "operations": [],
        "conditions": [],
        "relations": [],
        "summary": "",
    }


def _norm_term(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _list_overlap(query: list[str], doc: list[str]) -> float:
    """Count query terms that match a doc term (exact or substring)."""
    q_terms = [_norm_term(t) for t in query if _norm_term(t)]
    d_terms = [_norm_term(t) for t in doc if _norm_term(t)]
    if not q_terms or not d_terms:
        return 0.0
    hits = 0.0
    for q in q_terms:
        for d in d_terms:
            if q == d or q in d or d in q:
                hits += 1.0
                break
    return hits / max(len(q_terms), 1)


def _relation_overlap(query_rels: list[dict[str, str]], doc_rels: list[dict[str, str]]) -> float:
    if not query_rels or not doc_rels:
        return 0.0
    hits = 0.0
    for qr in query_rels:
        qs = _norm_term(qr.get("source") or "")
        qt = _norm_term(qr.get("target") or "")
        if not qs or not qt:
            continue
        best = 0.0
        for dr in doc_rels:
            ds = _norm_term(dr.get("source") or "")
            dt = _norm_term(dr.get("target") or "")
            src_ok = bool(qs and ds and (qs == ds or qs in ds or ds in qs))
            dst_ok = bool(qt and dt and (qt == dt or qt in dt or dt in qt))
            if src_ok and dst_ok:
                best = 1.0
                break
            if src_ok or dst_ok:
                best = max(best, 0.35)
        hits += best
    return hits / max(len(query_rels), 1)


def semantic_repr_similarity(
    query: dict[str, Any] | None,
    doc: dict[str, Any] | None,
) -> float:
    """Compare two structured representations. Relations/pattern weigh more than entities."""
    q = normalize_semantic_repr(query if isinstance(query, dict) else {})
    d = normalize_semantic_repr(doc if isinstance(doc, dict) else {})
    score = 0.0
    qrole, drole = _norm_term(q.get("role") or ""), _norm_term(d.get("role") or "")
    if qrole and drole:
        if qrole == drole or qrole in drole or drole in qrole:
            score += _ROLE_WEIGHT
        else:
            qw, dw = set(qrole.split()), set(drole.split())
            if qw and dw:
                score += _ROLE_WEIGHT * (len(qw & dw) / max(len(qw), 1))
    score += _FIELD_WEIGHTS["domain"] * _list_overlap(q["domain"], d["domain"])
    score += _FIELD_WEIGHTS["pattern"] * _list_overlap(q["pattern"], d["pattern"])
    score += _FIELD_WEIGHTS["entities"] * _list_overlap(q["entities"], d["entities"])
    score += _FIELD_WEIGHTS["operations"] * _list_overlap(q["operations"], d["operations"])
    score += _FIELD_WEIGHTS["conditions"] * _list_overlap(q["conditions"], d["conditions"])
    score += _REL_BOTH * _relation_overlap(q["relations"], d["relations"])
    qsum, dsum = _norm_term(q["summary"]), _norm_term(d["summary"])
    if qsum and dsum:
        qw = set(qsum.split())
        dw = set(dsum.split())
        if qw and dw:
            score += 1.2 * (len(qw & dw) / max(len(qw), 1))
    return float(score)


def flatten_semantic_text(sem: dict[str, Any] | None) -> str:
    """Canonical text for embedding / display. Not JSON dump, not summary-only."""
    s = normalize_semantic_repr(sem if isinstance(sem, dict) else {})
    lines: list[str] = []
    if s.get("role"):
        lines.append("Role: " + str(s["role"]).rstrip(".") + ".")
    if s["domain"]:
        lines.append("Domain: " + "; ".join(s["domain"]) + ".")
    if s["pattern"]:
        lines.append("Pattern: " + "; ".join(s["pattern"]) + ".")
    if s["entities"]:
        lines.append("Entities: " + "; ".join(s["entities"]) + ".")
    if s["operations"]:
        lines.append("Operations: " + "; ".join(s["operations"]) + ".")
    if s["conditions"]:
        lines.append("Conditions: " + "; ".join(s["conditions"]) + ".")
    if s["relations"]:
        rels = [
            f'{r["source"]} -> {r["target"]} [{r["type"]}]'
            for r in s["relations"]
        ]
        lines.append("Relations: " + "; ".join(rels) + ".")
    if s["summary"]:
        lines.append("Summary: " + s["summary"])
    return "\n".join(lines)


def _hay_blob(hit: dict[str, Any]) -> str:
    parts = [
        str(hit.get("prompt_preview") or ""),
        str(hit.get("response_preview") or ""),
        str(hit.get("preview") or ""),
        str(hit.get("task_id") or ""),
    ]
    return "\n".join(parts).lower()


def semantic_overlap_score(haystack: str, sem: dict[str, Any] | None) -> float:
    """Lexical overlap against structured fields. Relations need both endpoints."""
    s = normalize_semantic_repr(sem if isinstance(sem, dict) else {})
    h = (haystack or "").lower()
    if not h.strip():
        return 0.0
    score = 0.0
    for key, weight in _FIELD_WEIGHTS.items():
        for term in s.get(key) or []:
            t = str(term).strip().lower()
            if t and t in h:
                score += weight
    for rel in s.get("relations") or []:
        src = str(rel.get("source") or "").strip().lower()
        dst = str(rel.get("target") or "").strip().lower()
        if src and dst and src in h and dst in h:
            score += _REL_BOTH
        elif (src and src in h) or (dst and dst in h):
            score += _REL_ONE
    return float(score)


def rerank_hits_by_semantics(
    hits: list[dict[str, Any]],
    sem: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Stable rerank: higher structured overlap first; original order as tie-break."""
    if not hits:
        return []
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for i, hit in enumerate(hits):
        if not isinstance(hit, dict):
            continue
        sc = semantic_overlap_score(_hay_blob(hit), sem)
        row = dict(hit)
        row["semantic_score"] = round(sc, 3)
        scored.append((sc, i, row))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [row for _, _, row in scored]
