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

# Stage-2 rerank: embedding is only a 0.1 tie-break, not the ranker.
RERANK_WEIGHTS = {
    "relation": 0.40,
    "pattern": 0.22,
    "role": 0.15,
    "operations": 0.13,
    "conditions": 0.07,
    "entities": 0.03,
}
EMBED_TIE_WEIGHT = 0.10
EMBED_TEXT_KIND = "mechanism_v1"

REL_SRC_W = 0.4
REL_TGT_W = 0.4
REL_TYPE_W = 0.2
REL_COS_FLOOR = 0.35

_TYPE_NEAR_PAIRS = {
    frozenset(("dataflow", "transform")),
    frozenset(("dataflow", "api")),
    frozenset(("transform", "api")),
    frozenset(("control", "config")),
    frozenset(("control", "semantic_dependency")),
    frozenset(("init", "api")),
    frozenset(("error", "control")),
    frozenset(("error", "semantic_dependency")),
}


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


def relation_type_soft_match(a: str, b: str) -> float:
    ta = canonicalize_relation_type(a)
    tb = canonicalize_relation_type(b)
    if ta == tb:
        return 1.0
    if frozenset((ta, tb)) in _TYPE_NEAR_PAIRS:
        return 0.5
    return 0.0


_WEAK_SINGLE_ENDPOINTS = frozenset({
    "request", "error", "data", "context", "input", "output", "value",
    "object", "result", "header", "token", "response", "cache", "option",
})


def lexical_endpoint_sim(a: str, b: str) -> float:
    an, bn = _norm_term(a), _norm_term(b)
    if not an or not bn:
        return 0.0
    if an == bn:
        return 1.0
    short, longp = (an, bn) if len(an) <= len(bn) else (bn, an)
    short_toks = short.split()
    weak_single = len(short_toks) == 1 and (
        len(short) < 12 or short in _WEAK_SINGLE_ENDPOINTS
    )
    if not weak_single and (an in bn or bn in an):
        return 1.0
    aw, bw = set(an.split()), set(bn.split())
    if not aw or not bw:
        return 0.0
    jacc = len(aw & bw) / max(len(aw | bw), 1)
    if "request" in aw and "request" in bw and len(aw) == 1:
        return 0.0
    return float(jacc) if jacc >= 0.5 else 0.0


def relation_pair_score(
    qr: dict[str, str],
    dr: dict[str, str],
    endpoint_sim: Any,
) -> float:
    src = float(endpoint_sim(qr.get("source") or "", dr.get("source") or ""))
    tgt = float(endpoint_sim(qr.get("target") or "", dr.get("target") or ""))
    typ = relation_type_soft_match(qr.get("type") or "", dr.get("type") or "")
    src = max(0.0, min(1.0, src))
    tgt = max(0.0, min(1.0, tgt))
    return REL_SRC_W * src + REL_TGT_W * tgt + REL_TYPE_W * typ


def relation_semantic_overlap(
    query_rels: list[dict[str, str]],
    doc_rels: list[dict[str, str]],
    endpoint_sim: Any,
) -> float:
    if not query_rels or not doc_rels:
        return 0.0
    hits = 0.0
    for qr in query_rels:
        if not _norm_term(qr.get("source") or "") or not _norm_term(qr.get("target") or ""):
            continue
        best = 0.0
        for dr in doc_rels:
            best = max(best, relation_pair_score(qr, dr, endpoint_sim))
        hits += best
    return hits / max(len(query_rels), 1)


def collect_relation_endpoints(*sems: dict[str, Any] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sem in sems:
        s = normalize_semantic_repr(sem if isinstance(sem, dict) else {})
        for rel in s.get("relations") or []:
            for key in ("source", "target"):
                phrase = _norm_term(rel.get(key) or "")
                if phrase and phrase not in seen:
                    seen.add(phrase)
                    out.append(phrase)
    return out


def same_mechanism_cluster(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
    endpoint_sim: Any,
    *,
    rel_min: float = 0.55,
    pat_min: float = 0.4,
) -> bool:
    """True when two docs are the same transferable mechanism (for greedy dedup)."""
    sa = normalize_semantic_repr(a if isinstance(a, dict) else {})
    sb = normalize_semantic_repr(b if isinstance(b, dict) else {})
    rel = relation_semantic_overlap(sa["relations"], sb["relations"], endpoint_sim)
    pat = _list_overlap(sa["pattern"], sb["pattern"])
    role = _role_overlap(str(sa.get("role") or ""), str(sb.get("role") or ""))
    return rel >= rel_min and (pat >= pat_min or role >= 0.5)


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


def flatten_semantic_text_for_embedding(sem: dict[str, Any] | None) -> str:
    """Mechanism-heavy canonical text for dense recall.

    Pattern and relations are repeated; domain and summary are omitted so AWS/HTTP
    boilerplate does not dominate the neighborhood.
    """
    s = normalize_semantic_repr(sem if isinstance(sem, dict) else {})
    chunks: list[str] = []
    if s["pattern"]:
        line = "Pattern: " + "; ".join(s["pattern"]) + "."
        chunks.extend([line, line, line])
    if s["relations"]:
        rels = "; ".join(
            f'{r["source"]} -> {r["target"]} [{r["type"]}]' for r in s["relations"]
        )
        line = "Relations: " + rels + "."
        chunks.extend([line, line])
    role = str(s.get("role") or "").strip()
    if role:
        line = "Role: " + role.rstrip(".") + "."
        chunks.extend([line, line])
    if s["operations"]:
        chunks.append("Operations: " + "; ".join(s["operations"]) + ".")
    if s["conditions"]:
        chunks.append("Conditions: " + "; ".join(s["conditions"]) + ".")
    if s["entities"]:
        chunks.append("Entities: " + "; ".join(s["entities"][:4]) + ".")
    return "\n".join(chunks) or "empty semantic representation"


def mechanism_signature(sem: dict[str, Any] | None) -> str:
    """Dedup key: pattern + relation endpoints, ignoring domain/API names."""
    s = normalize_semantic_repr(sem if isinstance(sem, dict) else {})
    pats = tuple(_norm_term(p) for p in (s.get("pattern") or [])[:2] if _norm_term(p))
    if not pats:
        role = _norm_term(s.get("role") or "")
        pats = tuple(role.split()[:6]) if role else ()
    rels = tuple(
        sorted(
            f"{_norm_term(r.get('source') or '')}|{r.get('type')}|{_norm_term(r.get('target') or '')}"
            for r in (s.get("relations") or [])[:3]
            if _norm_term(r.get("source") or "") and _norm_term(r.get("target") or "")
        )
    )
    return f"{pats}||{rels}"


def _role_overlap(qrole: str, drole: str) -> float:
    qn, dn = _norm_term(qrole), _norm_term(drole)
    if not qn or not dn:
        return 0.0
    if qn == dn or qn in dn or dn in qn:
        return 1.0
    qw, dw = set(qn.split()), set(dn.split())
    if not qw:
        return 0.0
    return len(qw & dw) / max(len(qw), 1)


def semantic_struct_components(
    query: dict[str, Any] | None,
    doc: dict[str, Any] | None,
    *,
    endpoint_sim: Any | None = None,
) -> dict[str, float]:
    """Each component in [0, 1]. Domain/summary are not used for rerank."""
    q = normalize_semantic_repr(query if isinstance(query, dict) else {})
    d = normalize_semantic_repr(doc if isinstance(doc, dict) else {})
    if endpoint_sim is not None:
        rel = relation_semantic_overlap(q["relations"], d["relations"], endpoint_sim)
    else:
        rel = _relation_overlap(q["relations"], d["relations"])
    return {
        "relation": rel,
        "pattern": _list_overlap(q["pattern"], d["pattern"]),
        "role": _role_overlap(str(q.get("role") or ""), str(d.get("role") or "")),
        "operations": _list_overlap(q["operations"], d["operations"]),
        "conditions": _list_overlap(q["conditions"], d["conditions"]),
        "entities": _list_overlap(q["entities"], d["entities"]),
    }


def semantic_struct_score(
    query: dict[str, Any] | None,
    doc: dict[str, Any] | None,
    *,
    endpoint_sim: Any | None = None,
) -> float:
    parts = semantic_struct_components(query, doc, endpoint_sim=endpoint_sim)
    return float(sum(RERANK_WEIGHTS[k] * parts[k] for k in RERANK_WEIGHTS))


def semantic_combined_score(
    query: dict[str, Any] | None,
    doc: dict[str, Any] | None,
    *,
    embed_cos: float | None = None,
    embed_weight: float = EMBED_TIE_WEIGHT,
    endpoint_sim: Any | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Final rank = (1-w)*struct + w*max(embed_cos, 0). Embedding is a tie-break."""
    parts = semantic_struct_components(query, doc, endpoint_sim=endpoint_sim)
    struct = float(sum(RERANK_WEIGHTS[k] * parts[k] for k in RERANK_WEIGHTS))
    w = max(0.0, min(0.3, float(embed_weight)))
    if embed_cos is None:
        return struct, struct, parts
    embed = max(0.0, min(1.0, float(embed_cos)))
    combined = (1.0 - w) * struct + w * embed
    return float(combined), struct, parts


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
