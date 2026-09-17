"""Structured semantic FIM retrieval — independent of Boolean query retrieval.

This module never emits hole_relation / sibling_line / corpus_search_expressions.
Query and train preprocessing share one four-field schema::

    {
      "role": "...",
      "pattern": [...],
      "operations": [...],
      "relations": [{"source": "...", "target": "...", "type": "..."}]
    }

Role locates *overall semantic role / duty of the hole in the surrounding logic*;
pattern locates *which transferable mechanism*; operations describe *what the
missing span does semantically*; relations explain *how it connects to context
and later code*.

Canonical display text is ``flatten_semantic_text``. Dense recall embeds
``flatten_semantic_text_for_embedding`` (pattern/relations repeated).
Retrieval is two-stage when ``EIF_LLM_SEMANTIC_EMBEDDINGS`` (or the jsonl sidecar
npz) exists: embedding coarse recall, then structured rerank. Otherwise schema scan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.fim_semantic_schema import (
    flatten_semantic_text,
    normalize_semantic_repr,
    semantic_export_repr,
    semantic_repr_similarity,
)
from src.llm_train_retrieval import (
    _build_openai_client,
    _chat_complete_json,
    _env,
    _extra_body,
    prepare_llm_train_query,
)


def build_llm_semantic_messages(
    *,
    fim_prompt: str,
    gold_completion: str,
    language: str | None = None,
) -> list[dict[str, str]]:
    prepared = prepare_llm_train_query(fim_prompt, gold_completion)
    problem = prepared["fim_problem_surface"]
    gold = prepared["gold_mid_completion"]
    lang = (language or "code").strip() or "code"
    system = (
        "You analyze Fill-in-the-Middle (FIM) code completion samples. "
        "The hole may be marked <MID>, <FIM>, or [MASK]. ChatML wrappers are already stripped.\n\n"
        "Infer the semantic meaning of the MISSING SPAN from prefix, suffix, and gold fill. "
        "Use exactly four fields. They form one closed loop and MUST NOT repeat each other:\n"
        "  Role      = ROLE     — overall semantic role / responsibility of the missing span "
        "in the current code logic (not a so-that purpose clause, not a whole-function summary)\n"
        "  Pattern   = WHICH    — which transferable code mechanism this is\n"
        "  Operation = WHAT     — what semantic action the missing code performs\n"
        "  Relation  = CONNECTS — semantic links to surrounding context and later code\n\n"
        "Loop:\n"
        "  Role (semantic role / duty) → Pattern (mechanism) → "
        "Operation (semantic action of the missing span) → Relation (links to context and later code).\n"
        "If two fields would say the same thing, rewrite until they answer different questions.\n\n"
        "Do NOT emit domain, entities, conditions, summary, hole_relation, sibling_line, "
        "Boolean queries, or keyword-search strings.\n\n"
        "IMPORTANT:\n"
        "- Center every field on the missing span, not the entire function.\n"
        "- Use the gold fill ONLY to infer semantics. Do NOT copy gold source, API names, "
        "string literals, or identifiers. Generalize into concepts "
        "(security token, HTTP request, creation error).\n"
        "- Shared vocabulary, different questions: role is overall semantic role / duty, "
        "pattern is mechanism class, operations are semantic acts, relations are links.\n"
        "- Consider BOTH the prefix (what the hole depends on) AND the suffix "
        "(what the hole enables, guards, or changes) — not only the immediately next line.\n\n"
        "Return STRICT JSON (no markdown) with exactly these keys:\n"
        "{\n"
        '  "role": "overall semantic role / duty of this hole in the surrounding logic",\n'
        '  "pattern": ["one primary mechanism class; at most 3 if truly distinct"],\n'
        '  "operations": ["1-4 semantic operations the missing span performs"],\n'
        '  "relations": [\n'
        '    {"source": "semantic concept", "target": "semantic concept", '
        '"type": "dataflow | control | semantic_dependency | transform | init | error | config | api | <invented if none fit>"}\n'
        "  ]\n"
        "}\n\n"
        "Worked examples (pick the matching kind; do not copy unless the sample is the same):\n\n"
        "1) Credential / setup before a later step:\n"
        "func addSecurityToken(req, token) { <MID> signRequest(req) }\n"
        "{\n"
        '  "role": "request authentication preparation",\n'
        '  "pattern": ["credential propagation before signing"],\n'
        '  "operations": ["attach security token to request metadata"],\n'
        '  "relations": [\n'
        '    {"source": "security token", "target": "request metadata", "type": "dataflow"},\n'
        '    {"source": "request authentication", "target": "request signing", "type": "semantic_dependency"}\n'
        "  ]\n"
        "}\n\n"
        "2) Return the already-built result (plain return hole):\n"
        "func PutParams() map[string]string { m := map[string]string{...}; <MID> }\n"
        "{\n"
        '  "role": "return of assembled result",\n'
        '  "pattern": ["hand back constructed value"],\n'
        '  "operations": ["return the prepared map"],\n'
        '  "relations": [\n'
        '    {"source": "assembled parameter map", "target": "function result", "type": "dataflow"}\n'
        "  ]\n"
        "}\n\n"
        "3) Assignment that captures a computed value:\n"
        "n := <MID>\\nuse(n)\n"
        "{\n"
        '  "role": "bind computed value for later use",\n'
        '  "pattern": ["capture result then consume"],\n'
        '  "operations": ["assign computed length to a local"],\n'
        '  "relations": [\n'
        '    {"source": "computed value", "target": "local binding", "type": "dataflow"},\n'
        '    {"source": "local binding", "target": "later use", "type": "semantic_dependency"}\n'
        "  ]\n"
        "}\n\n"
        "4) Error-driven early exit:\n"
        "x, err := Open(); if err != nil { <MID> }; use(x)\n"
        "{\n"
        '  "role": "failed-resource guard before later use",\n'
        '  "pattern": ["error-driven early return"],\n'
        '  "operations": ["abort and surface the open error"],\n'
        '  "relations": [\n'
        '    {"source": "open error", "target": "hole", "type": "control"},\n'
        '    {"source": "early failure", "target": "later resource use", "type": "semantic_dependency"}\n'
        "  ]\n"
        "}\n\n"
        "role (ROLE):\n"
        "- Answer: what semantic role does this missing position play in the current code logic?\n"
        "- A short responsibility / positioning phrase for this hole in the local logic.\n"
        "- You MAY summarize the hole's overall duty. Do NOT summarize the whole function. "
        "Do NOT restate Operation. Do NOT write so that / in order to / 以便 clauses.\n"
        '- Good: "request authentication preparation"; "return of assembled result"; '
        '"bind computed value for later use"; "failed-resource guard before later use"\n'
        '- Bad: "so a present credential can be used when the request is later signed" '
        '(purpose clause); "attach security token to request metadata" (that is Operation); '
        '"return m"; "complete addSecurityToken"; "if statement"\n\n'
        "pattern (WHICH):\n"
        "- One PRIMARY transferable mechanism / contextual behavior pattern. "
        "Add a 2nd/3rd only if they are genuinely distinct.\n"
        "- Never more than 3. Do not list operations as extra patterns.\n"
        '- Good: "credential propagation before signing"; "hand back constructed value"; '
        '"capture result then consume"; "error-driven early return"; '
        '"resource initialization before use".\n'
        '- Bad: "if statement"; "header set"; "function call"; copying role or operations.\n'
        "- Do not force the sample into the mechanisms shown in the examples. "
        "Discover the most appropriate transferable mechanism from the given code.\n\n"
        "operations (WHAT):\n"
        "- Semantic behavior of the gold / missing span: what it does as meaning, "
        "not API spelling or syntax.\n"
        "- 1-4 items, in order if there are several steps.\n"
        '- Good: "attach security token to request metadata"; "return the prepared map"; '
        '"assign computed length to a local"; "abort and surface the open error".\n'
        '- Bad: repeating role; "call Header.Set"; "write some code"; "return m".\n\n'
        "relations (CONNECTS):\n"
        "- Semantic links between contextual entities and the hole, AND the hole's "
        "semantic effect on later code.\n"
        "- Prefix: what the hole depends on. Suffix: the most important later semantic "
        "dependence or effect — not necessarily the next statement "
        "(skip over unrelated lines if the real dependence is farther).\n"
        "- Emit 1-3 meaningful relations. Do not invent relations just to reach the minimum. "
        "source/target MUST be semantic concepts, not identifiers.\n"
        "- type: pick from this list first — dataflow, control, semantic_dependency, "
        "transform, init, error, config, api. "
        "Invent a new short label ONLY if none of those eight fit. Do not leave type empty.\n"
    )
    user = (
        f"Language: {lang}\n\n"
        "Prefix + suffix (hole in the middle):\n"
        f"{problem}\n\n"
        "Gold fill for the hole:\n"
        f"{gold}\n\n"
        "Fill the four-field loop. Do not repeat the same claim in two fields:\n"
        "Role = ROLE (overall semantic role / duty of this hole, not a so-that clause, "
        "not a whole-function summary) → Pattern = WHICH (one primary mechanism) → "
        "Operation = WHAT (semantic action of the missing span) → Relation = CONNECTS "
        "(context + later-code links, not only the next line).\n"
        "Return only the required JSON object."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm_semantic_analyze(
    *,
    fim_prompt: str,
    gold_completion: str,
    model: str | None = None,
    max_tokens: int | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    client = _build_openai_client()
    model_name = model or _env("ANNOTATE_MODEL") or _env("LLM_RETRIEVE_MODEL") or "qwen-plus"
    mt = max_tokens or int(_env("ANNOTATE_MAX_TOKENS") or "4096")
    messages = build_llm_semantic_messages(
        fim_prompt=fim_prompt,
        gold_completion=gold_completion,
        language=language,
    )
    extra = _extra_body()
    print(
        f"[llm-semantic] model={model_name} surface_chars={len(fim_prompt)} "
        f"gold_chars={len(gold_completion)}",
        flush=True,
    )
    raw, parsed = _chat_complete_json(
        client,
        model=model_name,
        messages=messages,
        max_tokens=mt,
        extra_body=extra or None,
        temperature=0.2,
        log_prefix="[llm-semantic]",
    )
    sem = semantic_export_repr(parsed)
    return {
        "model": model_name,
        "raw": raw,
        "semantic": sem,
        "semantic_flat_text": flatten_semantic_text(sem),
        "messages": messages,
    }


def search_semantic_corpus_jsonl(
    corpus_path: str,
    query_sem: dict[str, Any] | None,
    *,
    top_k: int = 10,
    max_scan: int | None = None,
    min_score: float = 0.01,
    recall_k: int | None = None,
) -> list[dict[str, Any]]:
    """Two-stage retrieve when embeddings exist; else full schema scan."""
    from src.fim_semantic_index import embedding_npz_path, search_embed_then_rerank

    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"semantic corpus not found: {corpus_path}")
    npz = embedding_npz_path(path)
    if npz.is_file():
        try:
            hits = search_embed_then_rerank(
                str(path),
                query_sem,
                top_k=top_k,
                recall_k=recall_k,
            )
            print(
                f"[llm-semantic] pipeline=embed_recall+struct_rerank "
                f"hits={len(hits)} npz={npz} "
                f"relation_align={hits[0].get('relation_align') if hits else ''}",
                flush=True,
            )
            return hits
        except Exception as exc:
            print(
                f"[llm-semantic] embed recall failed ({exc}); fallback schema scan",
                flush=True,
            )
    return _search_semantic_schema_scan(
        path,
        query_sem,
        top_k=top_k,
        max_scan=max_scan,
        min_score=min_score,
    )


def _search_semantic_schema_scan(
    path: Path,
    query_sem: dict[str, Any] | None,
    *,
    top_k: int = 10,
    max_scan: int | None = None,
    min_score: float = 0.01,
) -> list[dict[str, Any]]:
    q = normalize_semantic_repr(query_sem if isinstance(query_sem, dict) else {})
    scored: list[tuple[float, int, dict[str, Any]]] = []
    scanned = 0
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            scanned = idx + 1
            if max_scan is not None and idx >= max_scan:
                break
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            doc = normalize_semantic_repr(row)
            score = semantic_repr_similarity(q, doc)
            if score < min_score:
                continue
            source_line = row.get("source_line")
            try:
                source_line_i = int(source_line)
            except (TypeError, ValueError):
                source_line_i = idx
            hit = {
                "line": source_line_i,
                "semantic_line": idx,
                "task_id": str(row.get("task_id") or row.get("id") or ""),
                "semantic_score": round(float(score), 3),
                "match_region": "semantic",
                "prompt_preview": str(row.get("prompt_preview") or "")[:280],
                "response_preview": str(row.get("response_preview") or "")[:200],
                "summary": str(doc.get("summary") or "")[:240],
                "pattern": list(doc.get("pattern") or [])[:6],
                "retrieval": "schema_scan",
            }
            scored.append((score, idx, hit))
    scored.sort(key=lambda x: (-x[0], x[1]))
    hits = [row for _, _, row in scored[: max(1, top_k)]]
    for h in hits:
        h["scanned_rows"] = scanned
    print(f"[llm-semantic] pipeline=schema_scan hits={len(hits)} scanned={scanned}", flush=True)
    return hits


def retrieve_llm_semantic_samples(
    *,
    fim_prompt: str,
    gold_completion: str,
    corpus_path: str | None = None,
    top_k: int = 10,
    max_corpus_scan: int | None = None,
    run_corpus_search: bool = True,
    language: str | None = None,
) -> dict[str, Any]:
    if not (fim_prompt or "").strip():
        raise ValueError("fim_prompt is required")
    if not (gold_completion or "").strip():
        raise ValueError("gold_completion is required")

    prepared = prepare_llm_train_query(fim_prompt, gold_completion)
    llm_out = call_llm_semantic_analyze(
        fim_prompt=prepared["fim_problem_surface"],
        gold_completion=prepared["gold_mid_completion"],
        language=language,
    )
    sem = semantic_export_repr(llm_out.get("semantic") or {})
    semantic_corpus = (
        (corpus_path or "").strip()
        or _env("EIF_LLM_SEMANTIC_CORPUS")
    )
    raw_corpus = _env("EIF_LLM_TRAIN_CORPUS") or _env("EIF_TRAIN_CORPUS") or ""
    entry: dict[str, Any] = {
        "name": "structured_semantic",
        "expression": flatten_semantic_text(sem)[:500],
        "why": "embed recall + relations/pattern rerank on EIF_LLM_SEMANTIC_CORPUS",
        "match_in": "semantic",
        "corpus_hits": [],
        "local_bank_hits": [],
        "retrieval": "semantic",
    }
    if run_corpus_search:
        if semantic_corpus and Path(semantic_corpus).is_file():
            try:
                hits = search_semantic_corpus_jsonl(
                    semantic_corpus,
                    sem,
                    top_k=top_k,
                    max_scan=max_corpus_scan,
                )
                entry["corpus_hits"] = hits
                entry["semantic_corpus_path"] = semantic_corpus
                pipe = ""
                if hits:
                    pipe = str(hits[0].get("retrieval") or "")
                if pipe:
                    entry["retrieval"] = pipe
                    if pipe == "schema_scan":
                        entry["why"] = (
                            "schema-to-schema scan (no EIF_LLM_SEMANTIC_EMBEDDINGS / embed failed). "
                            "Run: python -m src.fim_semantic_preprocess --embed-only"
                        )
            except Exception as exc:
                entry["corpus_error"] = str(exc)
        else:
            entry["corpus_error"] = (
                "Set EIF_LLM_SEMANTIC_CORPUS to the jsonl from "
                "python -m src.fim_semantic_preprocess, and "
                "EIF_LLM_SEMANTIC_EMBEDDINGS to the npz from --embed-only. "
                "Keep EIF_LLM_TRAIN_CORPUS as the raw FIM file."
            )
    return {
        "status": "success",
        "retrieve_mode": "semantic",
        "semantic": sem,
        "semantic_flat_text": flatten_semantic_text(sem),
        "analysis": {"semantic": sem},
        "search_results": [entry],
        "corpus_path": raw_corpus or None,
        "semantic_corpus_path": semantic_corpus or None,
        "llm": {
            "model": llm_out.get("model"),
            "raw": llm_out.get("raw"),
        },
        "query": {
            "fim_prompt_chars": len(fim_prompt),
            "gold_completion_chars": len(gold_completion),
            "gold_preview": prepared["gold_mid_completion"][:400],
            "fim_surface_preview": prepared["fim_problem_surface"][:600],
        },
        "prepared": prepared,
    }
