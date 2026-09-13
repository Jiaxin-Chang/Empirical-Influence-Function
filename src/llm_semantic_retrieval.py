"""Structured semantic FIM retrieval — independent of Boolean query retrieval.

This module never emits hole_relation / sibling_line / corpus_search_expressions.
Query and train preprocessing share one schema::

    {
      "role": "...",
      "domain": [...],
      "pattern": [...],
      "entities": [...],
      "operations": [...],
      "conditions": [...],
      "relations": [...],
      "summary": "..."
    }

Structured fields are the retrieval signal. ``summary`` is for display.
Canonical display text is ``flatten_semantic_text``. Dense recall embeds
``flatten_semantic_text_for_embedding`` (pattern/relations repeated; no domain/summary).
Retrieval is two-stage when ``*.embeddings.npz`` exists: embedding coarse recall,
then structured rerank. Otherwise falls back to full schema scan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.fim_semantic_schema import (
    flatten_semantic_text,
    normalize_semantic_repr,
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
        "Your task is to infer the semantic role of the missing span from the prefix, suffix, "
        "and gold fill, and represent that role using a normalized, structured semantic representation "
        "that is stable enough for large-scale retrieval.\n\n"
        "IMPORTANT:\n"
        "- Center every field on the missing span and its relationship with the surrounding code.\n"
        "- Use the gold fill ONLY to infer the semantic role of the missing span.\n"
        "- Do NOT reproduce or paraphrase the implementation line-by-line.\n"
        "- Do NOT copy the gold source code, exact code fragments, or implementation-specific literals.\n"
        "- Do not use the exact API names, string literals, identifiers, or code fragments from the gold "
        "unless they represent an essential domain concept. Generalize implementation-specific details "
        "into semantic concepts.\n"
        "- Preserve domain concepts when they are semantically important "
        "(e.g. security token, HTTP request), but generalize variable names "
        "(k.SecurityToken, r.Header) and implementation details (X-Amz-Security-Token, Header.Set).\n"
        "- Focus on transferable code mechanisms, data dependencies, control conditions, "
        "API effects, and semantic relationships.\n"
        "- The representation should retrieve semantically related FIM samples written with "
        "different variable names, APIs, or implementation details.\n"
        "- Prefer a shared vocabulary across fields: if an entity is \"security token\", "
        "conditions/relations/operations should reuse that same phrase, not SecurityToken / sessionToken / tok.\n"
        "- Do not paste the same phrase into both pattern and operations. "
        "pattern = transferable mechanism; operations = actions the hole performs.\n"
        "- Prefer lowercase multi-word phrases (2-6 words) for role, pattern, entities, and operations.\n"
        "- Look at the code AFTER the hole. If the hole feeds, enables, or guards a later step, "
        "that later step MUST appear in relations (and usually in role).\n"
        "- role, pattern, and operations must be different: role = this hole's job; "
        "pattern = mechanism class reusable across programs; operations = concrete actions.\n\n"
        "Return STRICT JSON (no markdown) with exactly these keys:\n"
        "{\n"
        '  "role": "short semantic role of the missing span",\n'
        '  "domain": ["2-4 coarse technical or functional domains"],\n'
        '  "pattern": ["1-3 transferable behavioral mechanisms, NOT syntax"],\n'
        '  "entities": ["3-8 important semantic entities directly related to the hole"],\n'
        '  "operations": ["1-4 important semantic operations"],\n'
        '  "conditions": ["0-3 conditions under which the missing behavior occurs"],\n'
        '  "relations": [\n'
        '    {"source": "semantic entity or concept", "target": "semantic entity or concept", '
        '"type": "dataflow | control | api | transform | init | error | config | semantic_dependency | other"}\n'
        "  ],\n"
        '  "summary": "one concise English sentence describing the core semantic behavior of the missing span"\n'
        "}\n\n"
        "Field requirements:\n\n"
        "role:\n"
        "- Describe what the missing span contributes to the surrounding code (a phrase, not a full function summary).\n"
        "- Keep it implementation-independent.\n"
        '- Example: "propagate a credential into request metadata before request signing"\n'
        '- Other examples: "error handling and early exit"; "initialize a required dependency before use"\n\n'
        "domain:\n"
        "- 2-4 coarse domains of the HOLE itself, not every topic in the whole function.\n"
        "- Valid: HTTP, authentication, AWS, database, concurrency, file I/O.\n"
        "- Do not add authentication just because an earlier permission check exists.\n"
        "- Do not use overly specific labels such as \"AWS Signature Version 4 authentication\".\n"
        "- Avoid filler domains such as \"API calls\" or \"data handling\" unless that is the hole's actual domain.\n\n"
        "pattern:\n"
        "- A mechanism class, NOT a restatement of role or operations.\n"
        '- Valid: "conditional credential propagation", "request metadata injection", '
        '"consume stream then parse", "error-driven early return", '
        '"resource initialization before use".\n'
        '- Invalid: "if statement", "header set", "function call", '
        '"read request body into variable for processing" (that is a role, not a pattern), '
        '"set header field" (too syntactic).\n'
        "- If a credential/token is injected into a request, prefer "
        "\"conditional credential propagation\" over generic \"conditional header injection\".\n\n"
        "entities:\n"
        "- 3-8 semantically important entities related to the hole.\n"
        '- Prefer concepts such as "security token", "HTTP request", "request body", "parsed form".\n'
        "- Not entities: local variables, Go type names (Keys struct), or processes "
        "(\"error handling\", \"request context\") unless they are real objects.\n\n"
        "operations:\n"
        "- 1-4 actions of the hole, plus one following action when the suffix depends on it.\n"
        '- Prefer "propagate credential to request header" over "set header field" or "call Header.Set".\n'
        "- Do not copy role into operations.\n\n"
        "conditions:\n"
        "- 0-3 real guards. Empty is better than tautology.\n"
        '- Good: "security token is present"; "database create failed".\n'
        '- Bad: "request body needs to be processed"; "inside an if statement".\n\n'
        "relations:\n"
        "- This is the most important retrieval field.\n"
        "- Emit 1-3 relations. Prefer: (1) data into/out of the hole, "
        "(2) how the hole affects the NEXT statement in the suffix.\n"
        "- Example pair for signing: "
        '{"source":"security token","target":"HTTP request header","type":"dataflow"} and '
        '{"source":"header injection","target":"request signing","type":"semantic_dependency"}.\n'
        "- source and target MUST be semantic concepts, not identifiers (not k.SecurityToken / r.Header).\n"
        '- Example: {"source":"security token","target":"HTTP request header","type":"dataflow"}\n'
        "- type MUST be one of:\n"
        "  dataflow: value/state moves or is copied (credential -> request header).\n"
        "  control: a condition gates whether the hole runs (token present -> header write).\n"
        "  api: the hole's effect is an API/semantic operation (request -> set header).\n"
        "  transform: data is converted (payload -> HMAC digest; object -> JSON).\n"
        "  init: create/setup a dependency before later use (nil cache -> initialize cache).\n"
        "  error: error/failure path (error -> wrap and return).\n"
        "  config: configuration or flag drives behavior (option -> request modification).\n"
        "  semantic_dependency: required ordering or functional prerequisite "
        "(header injection -> request signing).\n"
        "  other: none of the above; do not invent extra type names.\n\n"
        "summary:\n"
        "- One concise English sentence for humans and auxiliary rerank, NOT the primary retrieval key.\n"
        "- Answer: what does the missing code do, under what condition, using what data, "
        "and with what effect on the surrounding computation?\n"
        "- Do not simply summarize the whole function.\n"
        "- Forbidden in summary: header names, string literals, API identifiers from gold "
        "(no X-Amz-Security-Token, Header.Set, ioutil.ReadAll).\n\n"
        "Do NOT output:\n"
        "- hole_relation, sibling_line, Boolean queries, corpus_search_expressions, "
        "keyword-search strings, or source code from the gold fill."
    )
    user = (
        f"Language: {lang}\n\n"
        "Prefix + suffix (hole in the middle):\n"
        f"{problem}\n\n"
        "Gold fill for the hole:\n"
        f"{gold}\n\n"
        "Infer the semantic role of the missing span from the prefix, suffix, and gold fill.\n\n"
        "Then return the structured semantic representation using the required JSON schema.\n\n"
        "The representation should describe the missing span at a transferable semantic level "
        "so that it can be matched against semantically related code samples with different "
        "variable names, APIs, or implementations."
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
    sem = normalize_semantic_repr(parsed)
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
    top_k: int = 20,
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
    top_k: int = 20,
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
    top_k: int = 15,
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
    sem = llm_out.get("semantic") or normalize_semantic_repr({})
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
                            "schema-to-schema scan (no embeddings.npz / embed failed). "
                            "Run: python -m src.fim_semantic_preprocess --embed-only"
                        )
            except Exception as exc:
                entry["corpus_error"] = str(exc)
        else:
            entry["corpus_error"] = (
                "Set EIF_LLM_SEMANTIC_CORPUS to the jsonl from "
                "python -m src.fim_semantic_preprocess. "
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
