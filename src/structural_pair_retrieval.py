"""AST structural (+ text) train-pair retrieval — **no attention_edges subtypes**.

Goal: find unlabeled (context → completion) pairs that are structurally similar
to a selected test/gold source→target edge.

Pipeline
--------
1. Decode each train sequence once; map every token → char span on the *original*
   ChatML string (UI / token indices stay in that space).
2. Extract FIM hole: ``<PRE>…<SUF>…<MID>`` + assistant MID fill. Reconstruct
   ``prefix + mid + suffix`` (put MID back into the hole) and tree-sitter-parse
   *only that* code — ignore long “snippets before” / ChatML wrappers.
3. Attach AST features to tokens that fall in PRE / MID / SUF regions (via
   reconstruct↔original char mapping).
4. **Full enumerate** every (context_i ∈ PRE∪SUF, completion_j ∈ MID).
5. Score vs query pair: 0.8 · AST-pair similarity + 0.2 · endpoint text similarity.
6. Disk-cache per-sample token AST feats so restarts / repeated clicks are cheap.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from heapq import nlargest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache" / "structural_ast_index"
# Bump when parse geometry changes so stale pkls are ignored.
AST_INDEX_VERSION = "fim_recon_v1"

DEFAULT_STRUCT_WEIGHT = 0.8
DEFAULT_TEXT_WEIGHT = 0.2

FIM_PRE = "<PRE>"
FIM_SUF = "<SUF>"
FIM_MID = "<MID>"
_IM_END = "<|im_end|>"

_TS_PACKAGE = {
    "go": "tree_sitter_go",
    "python": "tree_sitter_python",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
}

_LANG_CACHE: dict[str, Any] = {}
_TOKENIZER = None
_INDEX_MEM: dict[str, Any] | None = None


def _resolve_path(raw: str | None) -> Path | None:
    if not raw or not str(raw).strip():
        return None
    p = Path(str(raw).strip()).expanduser()
    if p.is_absolute():
        return p if p.is_file() else None
    for root in (Path.cwd(), REPO_ROOT):
        cand = (root / p).resolve()
        if cand.is_file():
            return cand
    return None


def _train_jsonl_path() -> Path | None:
    for key in ("EIF_TRAIN_DATA", "ANNOTATION_TRAIN_DATA"):
        found = _resolve_path(os.environ.get(key))
        if found is not None:
            return found
    return None


def normalize_surface(tok: str | None) -> str:
    s = (tok or "").replace("Ġ", " ").replace("▁", " ").strip()
    s = re.sub(r"\s+", "", s)
    return s.casefold()


def text_similarity(a: str | None, b: str | None) -> float:
    na, nb = normalize_surface(a), normalize_surface(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
        return 0.55 + 0.45 * (len(shorter) / max(1, len(longer)))
    return float(SequenceMatcher(None, na, nb).ratio())


def pair_text_similarity(
    q_src: str, q_dst: str, t_src: str, t_dst: str,
) -> float:
    return 0.5 * (
        text_similarity(q_src, t_src) + text_similarity(q_dst, t_dst)
    )


def detect_language(text: str, default: str = "go") -> str:
    raw = (os.environ.get("EIF_STRUCTURAL_LANG") or "").strip().lower()
    if raw in _TS_PACKAGE:
        return raw
    low = text[:4000].lower()
    if "```go" in low or "package " in low or "func " in low:
        return "go"
    if "```python" in low or "def " in low:
        return "python"
    if "```rust" in low or "fn " in low:
        return "rust"
    if "```java" in low:
        return "java"
    return default


def _load_ts_language(lang: str):
    if lang in _LANG_CACHE:
        return _LANG_CACHE[lang]
    pkg_name = _TS_PACKAGE.get(lang)
    if not pkg_name:
        raise ValueError(f"Unsupported language for structural AST: {lang}")
    try:
        from tree_sitter import Language
        import importlib
        pkg = importlib.import_module(pkg_name)
        lang_obj = Language(pkg.language())
    except Exception as exc:
        raise ImportError(
            f"Need tree-sitter + {pkg_name} for structural retrieval "
            f"(lang={lang}): {exc}"
        ) from exc
    _LANG_CACHE[lang] = lang_obj
    return lang_obj


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    raw = (os.environ.get("EIF_BASE_MODEL_PATH") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        return None
    try:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        return _TOKENIZER
    except Exception as exc:
        print(f"[structural-ast] tokenizer load failed: {exc}", flush=True)
        return None


@dataclass
class TokenAstFeat:
    """AST attachment for one token index (may be empty if outside code / failed)."""
    node_type: str = ""
    path: tuple[str, ...] = ()
    start: int = -1
    end: int = -1
    node_id: int = -1  # id(node) within one parse; only valid inside one sample


@dataclass
class PairAstFeat:
    src_type: str = ""
    dst_type: str = ""
    src_path: tuple[str, ...] = ()
    dst_path: tuple[str, ...] = ()
    lca_type: str = ""
    relation: str = "none"  # ancestor|descendant|sibling|cousin|none


@dataclass
class SampleAstIndex:
    train_sample_id: int
    tokens: list[str]
    answer_start: int
    language: str
    feats: list[TokenAstFeat]
    # For LCA: parent pointer by node_id within this sample parse
    parent: dict[int, int] = field(default_factory=dict)
    node_type_by_id: dict[int, str] = field(default_factory=dict)
    # FIM region token indices on the *original* ChatML sequence
    context_indices: list[int] = field(default_factory=list)  # PRE ∪ SUF
    completion_indices: list[int] = field(default_factory=list)  # MID fill
    parse_mode: str = "fim"  # fim | fallback_full


@dataclass
class _ReconSegment:
    """One contiguous slice of reconstructed code, mapped back to original chars."""

    recon_lo: int
    recon_hi: int
    orig_lo: int

    @property
    def orig_hi(self) -> int:
        return self.orig_lo + (self.recon_hi - self.recon_lo)


def _find_fim_prefix_suffix(text: str) -> tuple[str, str, int, int, int, int] | None:
    """Return (prefix, suffix, pre_body_lo, pre_body_hi, suf_body_lo, suf_body_hi)."""
    pre_i = text.rfind(FIM_PRE)
    if pre_i < 0:
        return None
    suf_i = text.find(FIM_SUF, pre_i + len(FIM_PRE))
    if suf_i < 0:
        return None
    mid_i = text.find(FIM_MID, suf_i + len(FIM_SUF))
    if mid_i < 0:
        return None
    pre_lo = pre_i + len(FIM_PRE)
    suf_lo = suf_i + len(FIM_SUF)
    return (
        text[pre_lo:suf_i],
        text[suf_lo:mid_i],
        pre_lo,
        suf_i,
        suf_lo,
        mid_i,
    )


def _assistant_mid_from_tokens(
    text: str,
    tokens: list[str],
    spans: list[tuple[int, int]],
    answer_start: int,
) -> tuple[str, int, int, list[int]]:
    """MID fill = assistant body tokens (answer_start … before <|im_end|>)."""
    n = len(tokens)
    if answer_start < 0 or answer_start >= n:
        return "", -1, -1, []
    end = n
    for j in range(answer_start, n):
        surf = tokens[j] if isinstance(tokens[j], str) else str(tokens[j])
        if _IM_END in surf:
            end = j
            break
    if end <= answer_start:
        end = n
    indices = list(range(answer_start, end))
    if not indices:
        return "", -1, -1, []
    lo = spans[indices[0]][0]
    hi = spans[indices[-1]][1]
    mid = text[lo:hi]
    # Drop trailing chat footer if somehow included.
    cut = mid.find(_IM_END)
    if cut >= 0:
        mid = mid[:cut]
        hi = lo + len(mid)
    return mid, lo, hi, indices


def _tokens_overlapping(
    spans: list[tuple[int, int]],
    lo: int,
    hi: int,
) -> list[int]:
    out: list[int] = []
    if lo < 0 or hi <= lo:
        return out
    for ti, (a, b) in enumerate(spans):
        if b > lo and a < hi:
            out.append(ti)
    return out


def _orig_char_to_recon(oc: int, segments: list[_ReconSegment]) -> int | None:
    for seg in segments:
        if seg.orig_lo <= oc < seg.orig_hi:
            return seg.recon_lo + (oc - seg.orig_lo)
    return None


def _build_token_ast_feats_on_recon(
    code: str,
    orig_spans: list[tuple[int, int]],
    segments: list[_ReconSegment],
    language: str,
    active_token_indices: set[int],
) -> tuple[list[TokenAstFeat], dict[int, int], dict[int, str]]:
    """Parse reconstructed ``code``; attach feats only for tokens in FIM regions."""
    n = len(orig_spans)
    feats = [TokenAstFeat() for _ in range(n)]
    parent: dict[int, int] = {}
    node_type_by_id: dict[int, str] = {}
    if not code.strip() or not active_token_indices:
        return feats, parent, node_type_by_id

    try:
        from tree_sitter import Parser
        lang = _load_ts_language(language)
        parser = Parser(lang)
        raw = code.encode("utf-8", errors="surrogatepass")
        tree = parser.parse(raw)
    except Exception as exc:
        print(f"[structural-ast] parse failed ({language}): {exc}", flush=True)
        return feats, parent, node_type_by_id

    root = tree.root_node
    stack = [root]
    while stack:
        node = stack.pop()
        nid = id(node)
        node_type_by_id[nid] = node.type
        for child in node.children:
            parent[id(child)] = nid
            stack.append(child)

    def covering_node(start: int, end: int):
        if start >= end:
            return None
        node = root
        changed = True
        while changed:
            changed = False
            for child in node.children:
                if child.start_byte <= start and end <= child.end_byte:
                    node = child
                    changed = True
                    break
        return node

    code_len = len(code)
    for ti in active_token_indices:
        if ti < 0 or ti >= n:
            continue
        a, b = orig_spans[ti]
        if a >= b:
            continue
        mid_o = (a + b) // 2
        r_mid = _orig_char_to_recon(mid_o, segments)
        if r_mid is None:
            # try any char in the token span
            r_mid = None
            for oc in range(a, b):
                r_mid = _orig_char_to_recon(oc, segments)
                if r_mid is not None:
                    break
        if r_mid is None or r_mid >= code_len:
            continue
        r_end = min(code_len, r_mid + max(1, b - a))
        node = covering_node(r_mid, r_end)
        if node is None or node.type in ("ERROR",):
            node = covering_node(r_mid, r_mid + 1)
        if node is None:
            continue
        path_types: list[str] = []
        cid: int | None = id(node)
        g2 = 0
        while cid is not None and g2 < 64:
            path_types.append(node_type_by_id.get(cid, "?"))
            cid = parent.get(cid)
            g2 += 1
        feats[ti] = TokenAstFeat(
            node_type=node.type,
            path=tuple(path_types),
            start=int(node.start_byte),
            end=int(node.end_byte),
            node_id=id(node),
        )

    return feats, parent, node_type_by_id


def build_fim_recon_index(
    tokens: list[str],
    answer_start: int,
    *,
    language: str | None = None,
) -> tuple[
    list[TokenAstFeat],
    dict[int, int],
    dict[int, str],
    list[int],
    list[int],
    str,
    str,
]:
    """Parse PRE+MID+SUF only; return feats + context/completion token index lists."""
    text, spans = tokens_and_char_spans(tokens)
    lang = language or detect_language(text)
    fim = _find_fim_prefix_suffix(text)
    mid_text, mid_lo, mid_hi, mid_toks = _assistant_mid_from_tokens(
        text, tokens, spans, answer_start,
    )

    if fim is not None and mid_lo >= 0:
        prefix, suffix, pre_lo, pre_hi, suf_lo, suf_hi = fim
        code = prefix + mid_text + suffix
        pre_len = len(prefix)
        mid_len = len(mid_text)
        segments = [
            _ReconSegment(0, pre_len, pre_lo),
            _ReconSegment(pre_len, pre_len + mid_len, mid_lo),
            _ReconSegment(pre_len + mid_len, pre_len + mid_len + len(suffix), suf_lo),
        ]
        ctx = _tokens_overlapping(spans, pre_lo, pre_hi) + _tokens_overlapping(
            spans, suf_lo, suf_hi,
        )
        # stable unique
        seen: set[int] = set()
        context_indices: list[int] = []
        for i in ctx:
            if i not in seen and i not in set(mid_toks):
                seen.add(i)
                context_indices.append(i)
        completion_indices = list(mid_toks)
        active = set(context_indices) | set(completion_indices)
        feats, parent, node_type_by_id = _build_token_ast_feats_on_recon(
            code, spans, segments, lang, active,
        )
        return (
            feats, parent, node_type_by_id,
            context_indices, completion_indices, lang, "fim",
        )

    # Fallback: whole sequence (old behavior) — rare if data is proper FIM ChatML.
    feats, parent, node_type_by_id = _build_token_ast_feats(text, spans, lang)
    n = len(tokens)
    ans = max(1, min(int(answer_start), n - 1)) if n else 0
    return (
        feats, parent, node_type_by_id,
        list(range(0, ans)), list(range(ans, n)), lang, "fallback_full",
    )


def _answer_start_from_obj(obj: dict[str, Any], n: int) -> int:
    for key in ("answer_start_index", "response_start", "prompt_len"):
        v = obj.get(key)
        if isinstance(v, int) and 0 <= v < n:
            return v
    labels = obj.get("labels") or obj.get("label")
    if isinstance(labels, list) and len(labels) == n:
        for i, lab in enumerate(labels):
            try:
                if int(lab) != -100:
                    return i
            except (TypeError, ValueError):
                continue
    # Fallback: last third as "completion" so we still enumerate something.
    return max(1, (2 * n) // 3)


def tokens_and_char_spans(
    tokens: list[str],
) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate token surfaces; return full text + per-token [start, end)."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for t in tokens:
        s = t if isinstance(t, str) else str(t)
        # Keep tokenizer spaces (Ġ) as space so offsets stay aligned with decode.
        s = s.replace("Ġ", " ").replace("▁", " ")
        start = pos
        parts.append(s)
        pos += len(s)
        spans.append((start, pos))
    return "".join(parts), spans


def _build_token_ast_feats(
    text: str,
    spans: list[tuple[int, int]],
    language: str,
) -> tuple[list[TokenAstFeat], dict[int, int], dict[int, str]]:
    """Parse ``text`` and map each token span → smallest covering AST node."""
    n = len(spans)
    feats = [TokenAstFeat() for _ in range(n)]
    parent: dict[int, int] = {}
    node_type_by_id: dict[int, str] = {}
    if not text.strip():
        return feats, parent, node_type_by_id

    try:
        from tree_sitter import Parser
        lang = _load_ts_language(language)
        parser = Parser(lang)
        # tree-sitter expects bytes; for ASCII-heavy Go this matches str indices.
        raw = text.encode("utf-8", errors="surrogatepass")
        tree = parser.parse(raw)
    except Exception as exc:
        print(f"[structural-ast] parse failed ({language}): {exc}", flush=True)
        return feats, parent, node_type_by_id

    root = tree.root_node

    # Parent map via walk.
    stack = [root]
    while stack:
        node = stack.pop()
        nid = id(node)
        node_type_by_id[nid] = node.type
        for child in node.children:
            parent[id(child)] = nid
            stack.append(child)

    # Char → token index (first covering token).
    offset_to_tok: dict[int, int] = {}
    for ti, (a, b) in enumerate(spans):
        for p in range(a, b):
            if p not in offset_to_tok:
                offset_to_tok[p] = ti

    # For each token, find smallest node that covers its span center / range.
    def covering_node(start: int, end: int):
        if start >= end:
            return None
        node = root
        # Descend while a child fully covers [start, end).
        changed = True
        while changed:
            changed = False
            for child in node.children:
                if child.start_byte <= start and end <= child.end_byte:
                    node = child
                    changed = True
                    break
        return node

    for ti, (a, b) in enumerate(spans):
        if a >= b:
            continue
        # Prefer center point for punctuations with empty-ish spans after strip.
        mid = (a + b) // 2
        node = covering_node(a, b)
        if node is None or node.type in ("ERROR",):
            # try mid point
            node = covering_node(mid, mid + 1) if mid < len(text) else None
        if node is None:
            continue
        # Path-to-root via parent ids.
        path_types: list[str] = []
        cid: int | None = id(node)
        g2 = 0
        while cid is not None and g2 < 64:
            path_types.append(node_type_by_id.get(cid, "?"))
            cid = parent.get(cid)
            g2 += 1
        feats[ti] = TokenAstFeat(
            node_type=node.type,
            path=tuple(path_types),
            start=int(node.start_byte),
            end=int(node.end_byte),
            node_id=id(node),
        )

    return feats, parent, node_type_by_id


def build_pair_ast_feat(
    feats: list[TokenAstFeat],
    parent: dict[int, int],
    node_type_by_id: dict[int, str],
    src: int,
    dst: int,
) -> PairAstFeat:
    fs = feats[src] if 0 <= src < len(feats) else TokenAstFeat()
    fd = feats[dst] if 0 <= dst < len(feats) else TokenAstFeat()
    out = PairAstFeat(
        src_type=fs.node_type,
        dst_type=fd.node_type,
        src_path=fs.path,
        dst_path=fd.path,
    )
    if fs.node_id < 0 or fd.node_id < 0:
        return out

    # Ancestors set for src.
    src_anc: list[int] = []
    cid = fs.node_id
    seen = set()
    while cid is not None and cid not in seen:
        seen.add(cid)
        src_anc.append(cid)
        cid = parent.get(cid)

    # Climb dst until hit src ancestor.
    lca = None
    dst_anc: list[int] = []
    cid = fd.node_id
    seen = set()
    while cid is not None and cid not in seen:
        seen.add(cid)
        dst_anc.append(cid)
        if cid in set(src_anc):
            lca = cid
            break
        cid = parent.get(cid)

    if lca is not None:
        out.lca_type = node_type_by_id.get(lca, "")
        if lca == fs.node_id:
            out.relation = "ancestor"  # src is ancestor of dst
        elif lca == fd.node_id:
            out.relation = "descendant"  # src under dst
        elif parent.get(fs.node_id) == parent.get(fd.node_id) == lca or (
            parent.get(fs.node_id) is not None
            and parent.get(fs.node_id) == parent.get(fd.node_id)
        ):
            out.relation = "sibling"
        else:
            out.relation = "cousin"
    return out


def pair_ast_similarity(a: PairAstFeat, b: PairAstFeat) -> float:
    """Structure similarity in [0, 1] from AST pair signatures (no subtype labels)."""
    if not a.src_type and not a.dst_type and not b.src_type and not b.dst_type:
        return 0.0
    score = 0.0
    if a.lca_type and a.lca_type == b.lca_type:
        score += 0.30
    if a.src_type and a.src_type == b.src_type:
        score += 0.15
    if a.dst_type and a.dst_type == b.dst_type:
        score += 0.15
    if a.relation != "none" and a.relation == b.relation:
        score += 0.15
    # Path bag overlap (rootward types).
    if a.src_path and b.src_path:
        sa, sb = set(a.src_path[:8]), set(b.src_path[:8])
        score += 0.125 * (len(sa & sb) / max(1, len(sa | sb)))
    if a.dst_path and b.dst_path:
        sa, sb = set(a.dst_path[:8]), set(b.dst_path[:8])
        score += 0.125 * (len(sa & sb) / max(1, len(sa | sb)))
    return float(min(1.0, score))


def _cache_path(train_path: Path, sample_id: int, language: str, sig: str) -> Path:
    h = hashlib.sha1(
        f"{train_path.resolve()}|{sample_id}|{language}|{sig}".encode("utf-8")
    ).hexdigest()[:16]
    return CACHE_DIR / f"sample_{sample_id}_{h}.pkl"


def _sample_sig(tokens: list[str], answer_start: int) -> str:
    head = "".join(tokens[:32])
    tail = "".join(tokens[-32:]) if len(tokens) > 32 else ""
    return f"{AST_INDEX_VERSION}|{len(tokens)}|{answer_start}|{hash(head)}|{hash(tail)}"


def index_one_sample(
    sample_id: int,
    tokens: list[str],
    answer_start: int,
    *,
    language: str | None = None,
    train_path: Path | None = None,
    use_disk_cache: bool = True,
) -> SampleAstIndex:
    lang_hint = language or detect_language("".join(
        (t if isinstance(t, str) else str(t)) for t in tokens[:80]
    ))
    sig = _sample_sig(tokens, answer_start)
    if use_disk_cache and train_path is not None:
        cpath = _cache_path(train_path, sample_id, lang_hint, sig)
        if cpath.is_file():
            try:
                with cpath.open("rb") as f:
                    obj = pickle.load(f)
                if (
                    isinstance(obj, SampleAstIndex)
                    and len(obj.feats) == len(tokens)
                    and getattr(obj, "parse_mode", "") in ("fim", "fallback_full")
                    and hasattr(obj, "context_indices")
                ):
                    return obj
            except Exception:
                pass

    (
        feats, parent, node_type_by_id,
        context_indices, completion_indices, lang, parse_mode,
    ) = build_fim_recon_index(tokens, answer_start, language=language)
    sample = SampleAstIndex(
        train_sample_id=sample_id,
        tokens=list(tokens),
        answer_start=int(answer_start),
        language=lang,
        feats=feats,
        parent=parent,
        node_type_by_id=node_type_by_id,
        context_indices=context_indices,
        completion_indices=completion_indices,
        parse_mode=parse_mode,
    )

    if use_disk_cache and train_path is not None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cpath = _cache_path(train_path, sample_id, lang, sig)
            with cpath.open("wb") as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"[structural-ast] cache write failed: {exc}", flush=True)
    return sample


def _tokens_from_train_obj(obj: dict[str, Any], tokenizer) -> list[str]:
    qt = obj.get("qwen_tokens")
    if isinstance(qt, list) and qt:
        return [str(x) for x in qt]
    ids = obj.get("input_ids")
    if not isinstance(ids, list) or not ids:
        return []
    if tokenizer is None:
        return [f"<{int(i)}>" for i in ids]
    return [tokenizer.decode([int(i)], skip_special_tokens=False) for i in ids]


def build_or_load_train_ast_index(
    *,
    force: bool = False,
    max_samples: int | None = None,
) -> tuple[list[SampleAstIndex], str]:
    global _INDEX_MEM
    path = _train_jsonl_path()
    if path is None:
        raise FileNotFoundError(
            "No train JSONL. Set EIF_TRAIN_DATA / ANNOTATION_TRAIN_DATA."
        )
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    if (
        not force
        and _INDEX_MEM is not None
        and _INDEX_MEM.get("path") == key
        and _INDEX_MEM.get("mtime") == mtime
    ):
        return _INDEX_MEM["samples"], key

    tokenizer = _get_tokenizer()
    samples: list[SampleAstIndex] = []
    t0 = time.time()
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_samples is not None and idx >= max_samples:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            tokens = _tokens_from_train_obj(obj, tokenizer)
            if len(tokens) < 2:
                continue
            ans = _answer_start_from_obj(obj, len(tokens))
            if ans <= 0 or ans >= len(tokens):
                continue
            sample = index_one_sample(
                idx, tokens, ans, train_path=path, use_disk_cache=True,
            )
            samples.append(sample)
            if (idx + 1) % 20 == 0:
                print(
                    f"[structural-ast] indexed {idx + 1} samples "
                    f"({time.time() - t0:.1f}s)",
                    flush=True,
                )

    n_fim = sum(1 for s in samples if s.parse_mode == "fim")
    n_pairs = sum(
        max(0, len(s.context_indices)) * max(0, len(s.completion_indices))
        for s in samples
    )
    print(
        f"[structural-ast] ready samples={len(samples)} fim={n_fim} "
        f"enumerable_pairs≈{n_pairs} (PRE∪SUF)×MID cache={CACHE_DIR} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    _INDEX_MEM = {"path": key, "mtime": mtime, "samples": samples}
    return samples, key


def index_query_tokens(
    tokens: list[str],
    prompt_len: int,
    *,
    language: str | None = None,
) -> SampleAstIndex:
    ans = max(1, min(int(prompt_len), len(tokens) - 1)) if tokens else 0
    return index_one_sample(
        sample_id=-1,
        tokens=tokens,
        answer_start=ans,
        language=language,
        train_path=None,
        use_disk_cache=False,
    )


def retrieve_structural_pairs(
    *,
    query_src_token: str,
    query_dst_token: str,
    query_src_index: int,
    query_dst_index: int,
    query_tokens: list[str] | None = None,
    query_prompt_len: int | None = None,
    query_subtype: str | None = None,  # ignored — kept for API compat
    top_k: int = 40,
    struct_weight: float = DEFAULT_STRUCT_WEIGHT,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
    min_score: float = 0.02,
    max_train_samples: int | None = None,
) -> dict[str, Any]:
    """Full-enumerate context×completion pairs; rank by AST+text vs query edge."""
    _ = query_subtype  # explicitly unused
    sw = float(struct_weight)
    tw = float(text_weight)
    z = sw + tw
    if z <= 0:
        sw, tw = DEFAULT_STRUCT_WEIGHT, DEFAULT_TEXT_WEIGHT
        z = sw + tw
    sw, tw = sw / z, tw / z

    samples, train_path = build_or_load_train_ast_index(max_samples=max_train_samples)

    # Query AST pair signature.
    if query_tokens and len(query_tokens) > max(query_src_index, query_dst_index):
        q_index = index_query_tokens(
            query_tokens,
            int(query_prompt_len or max(1, query_src_index)),
        )
        q_pair = build_pair_ast_feat(
            q_index.feats, q_index.parent, q_index.node_type_by_id,
            int(query_src_index), int(query_dst_index),
        )
        q_lang = q_index.language
    else:
        q_pair = PairAstFeat()
        q_lang = detect_language(query_src_token + query_dst_token)

    scored: list[tuple[float, dict[str, Any]]] = []
    n_scored = 0
    n_skipped_empty = 0
    t0 = time.time()

    for sample in samples:
        ans = sample.answer_start
        ctx_idxs = sample.context_indices or list(range(0, max(0, ans)))
        cmp_idxs = sample.completion_indices or list(
            range(ans, len(sample.tokens))
        )
        if not ctx_idxs or not cmp_idxs:
            continue
        # Full enumeration over FIM regions only: (PRE∪SUF) × MID.
        for j in cmp_idxs:
            t_dst = sample.tokens[j]
            for i in ctx_idxs:
                t_src = sample.tokens[i]
                if not normalize_surface(t_src) and not normalize_surface(t_dst):
                    n_skipped_empty += 1
                    continue

                t_pair = build_pair_ast_feat(
                    sample.feats, sample.parent, sample.node_type_by_id, i, j,
                )
                s_struct = pair_ast_similarity(q_pair, t_pair)
                s_text = pair_text_similarity(
                    query_src_token, query_dst_token, t_src, t_dst,
                )
                score = sw * s_struct + tw * s_text
                n_scored += 1
                if score < min_score:
                    continue
                scored.append((score, {
                    "id": f"ast_t{sample.train_sample_id}_s{i}_d{j}",
                    "train_sample_id": int(sample.train_sample_id),
                    "cos_sim": float(score),
                    "coarse_cos_sim": float(score),
                    "struct_score": round(float(s_struct), 6),
                    "text_score": round(float(s_text), 6),
                    "score": round(float(score), 6),
                    "subtype": "",  # intentionally empty — not using edge labels
                    "retrieval": "structural_ast",
                    "parse_mode": sample.parse_mode,
                    "ast": {
                        "train_src_type": t_pair.src_type,
                        "train_dst_type": t_pair.dst_type,
                        "train_lca": t_pair.lca_type,
                        "train_relation": t_pair.relation,
                        "query_src_type": q_pair.src_type,
                        "query_dst_type": q_pair.dst_type,
                        "query_lca": q_pair.lca_type,
                        "query_relation": q_pair.relation,
                    },
                    "test_correlation": {
                        "source_token_index": int(query_src_index),
                        "target_token_index": int(query_dst_index),
                        "source_token": query_src_token,
                        "target_token": query_dst_token,
                        "saliency_score": 0.0,
                    },
                    "train_correlation": {
                        "source_token_index": int(i),
                        "target_token_index": int(j),
                        "source_token": t_src,
                        "target_token": t_dst,
                        "response_token_offset": int(j - ans),
                    },
                    "train_context": {
                        "source_context": [],
                        "target_context": [],
                    },
                    "annotation": None,
                }))

    top_rows = [row for _, row in nlargest(max(1, int(top_k)), scored, key=lambda x: x[0])]
    print(
        f"[structural-ast] scored={n_scored} kept≥{min_score}:{len(scored)} "
        f"empty_skip={n_skipped_empty} top={len(top_rows)} "
        f"query_ast=({q_pair.src_type}->{q_pair.dst_type} lca={q_pair.lca_type}/{q_pair.relation}) "
        f"elapsed={time.time() - t0:.2f}s",
        flush=True,
    )
    return {
        "status": "success",
        "retrieval": "structural_ast",
        "trainPath": train_path,
        "nTrainSamples": len(samples),
        "nScoredPairs": n_scored,
        "nCandidates": len(scored),
        "cacheDir": str(CACHE_DIR),
        "query": {
            "sourceToken": query_src_token,
            "targetToken": query_dst_token,
            "sourceIndex": query_src_index,
            "targetIndex": query_dst_index,
            "language": q_lang,
            "structWeight": sw,
            "textWeight": tw,
            "ast": {
                "src_type": q_pair.src_type,
                "dst_type": q_pair.dst_type,
                "lca": q_pair.lca_type,
                "relation": q_pair.relation,
            },
        },
        "pairs": top_rows,
    }
