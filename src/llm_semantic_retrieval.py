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
npz) exists: embedding coarse recall, then structured rerank. Otherwise Stage-2
runs on the full semantic jsonl (no Top-500 cutoff).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.fim_semantic_schema import (
    collect_relation_endpoints,
    flatten_semantic_text,
    lexical_endpoint_sim,
    normalize_semantic_repr,
    semantic_combined_score,
    semantic_export_repr,
)
from src.llm_train_retrieval import (
    _build_openai_client,
    _chat_complete_json,
    _env,
    _extra_body,
    clean_gold_mid_completion,
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


def _fills_equivalent(a: str, b: str) -> bool:
    """True when two hole fills differ only by whitespace."""
    return " ".join((a or "").split()) == " ".join((b or "").split())


def build_valid_semantic_messages(
    *,
    fim_prompt: str,
    gold_completion: str,
    model_prediction: str,
    language: str | None = None,
) -> list[dict[str, str]]:
    """Valid-side prompt: four fields describe the gold-vs-prediction residual.

    Train-corpus preprocess keeps ``build_llm_semantic_messages`` (full gold span).
    """
    prepared = prepare_llm_train_query(fim_prompt, gold_completion)
    problem = prepared["fim_problem_surface"]
    gold = prepared["gold_mid_completion"]
    pred = clean_gold_mid_completion(model_prediction) or (model_prediction or "").strip()
    lang = (language or "code").strip() or "code"
    system = (
        "You analyze a Fill-in-the-Middle (FIM) sample the model answered incorrectly. "
        "The hole may be marked <MID>, <FIM>, or [MASK]. ChatML wrappers are already stripped.\n\n"
        "You receive the prefix and suffix, the GOLD fill, and the MODEL PREDICTION for the same hole. "
        "Describe only the semantic residual: the association the prediction failed to realize "
        "and the gold still requires. Do not describe behavior the prediction already got right.\n\n"
        "How to locate the residual:\n"
        "- Compare the prediction to the gold fill. Ignore pure whitespace and indentation.\n"
        "- The residual is the gold content that is missing or wrong in the prediction, "
        "plus the link that content must have to the surrounding code "
        "(including code the prediction already wrote, and the suffix).\n"
        "- If the prediction is empty, unrelated, or shares no real structure with the gold, "
        "the residual is the whole gold span. Describe that whole span.\n"
        "- If the prediction already matches the gold, there is no residual. "
        "Describe the gold span as a whole.\n\n"
        "Use exactly four fields. They form one closed loop and MUST NOT repeat each other:\n"
        "  Role      = ROLE of the residual only — its duty in the local logic\n"
        "  Pattern   = WHICH transferable mechanism the residual is\n"
        "  Operation = WHAT semantic action the residual performs\n"
        "  Relation  = CONNECTS — the semantic links the model still needs to learn. "
        "This is the primary field.\n\n"
        "Relation rules (primary):\n"
        "- Anchor every relation on the residual, not on steps the prediction already got right.\n"
        "- Emit 1-3 relations. Each one is a link the model failed to make.\n"
        "- Typical shape: what the missing piece consumes → what it produces, and "
        "what it produces → the later use the prediction skipped "
        "(the lookup, the suffix, or a binding the model wrote with the wrong input).\n"
        "- source/target are semantic concepts, not raw identifiers or API spellings.\n"
        "- Keep the concept specific to the missing mechanism. "
        "If the gold case-folds a key, say normalized key / lowercased key, "
        "not a generic transformed key, and not a different mechanism such as "
        "encoding a key for storage.\n"
        "- type: pick from dataflow, control, semantic_dependency, transform, init, "
        "error, config, api. Invent a short label ONLY if none of those fit.\n\n"
        "Role: a short duty of the residual inside the local logic. "
        "Not a whole-function summary. Not a so-that / in order to clause. "
        "Do not restate an operation the prediction already did.\n"
        "Pattern: one primary mechanism of the residual. Add a second only if it is "
        "truly distinct. Specific enough that a different mechanism with the same "
        "cartoon shape would not use the same phrase.\n"
        "Operations: 1-4 semantic acts of the residual only, in order. "
        "Do not list acts the prediction already performed correctly.\n"
        "Do NOT copy gold source, API names, string literals, or identifiers. "
        "Generalize into concepts, but do not generalize away the missing mechanism.\n\n"
        "Do NOT emit domain, entities, conditions, summary, hole_relation, sibling_line, "
        "Boolean queries, or keyword-search strings.\n\n"
        "Return STRICT JSON (no markdown) with exactly these keys:\n"
        "{\n"
        '  "role": "duty of the residual in the surrounding logic",\n'
        '  "pattern": ["one primary mechanism of the residual"],\n'
        '  "operations": ["1-4 semantic acts of the residual only"],\n'
        '  "relations": [\n'
        '    {"source": "semantic concept", "target": "semantic concept", '
        '"type": "dataflow | control | semantic_dependency | transform | init | error | config | api"}\n'
        "  ]\n"
        "}\n\n"
        "Worked example (near-miss; do not copy unless the sample is the same):\n"
        "Context: read a map, then type-switch on the value when the lookup succeeds.\n"
        "Prediction: val, ok := values[key]\n"
        "Gold: key = strings.ToLower(key), then val, ok := values[key]\n"
        "The prediction already reads the map. It skipped case-folding the key and "
        "still indexed the map with the original key.\n"
        "{\n"
        '  "role": "case-fold the lookup key before the map read",\n'
        '  "pattern": ["key normalization before lookup"],\n'
        '  "operations": ["convert key to lowercase"],\n'
        '  "relations": [\n'
        '    {"source": "original key", "target": "normalized key", "type": "transform"},\n'
        '    {"source": "normalized key", "target": "map lookup key", "type": "dataflow"}\n'
        "  ]\n"
        "}\n\n"
        "Worked example (prediction shares no structure with the gold; describe the whole span):\n"
        "Prediction: return nil\n"
        "Gold: an error-driven early return that surfaces the open error before later use.\n"
        "Role, pattern, operations, and relations then describe that whole guard."
    )
    user = (
        f"Language: {lang}\n\n"
        "Prefix + suffix (hole in the middle):\n"
        f"{problem}\n\n"
        "Gold fill for the hole:\n"
        f"{gold}\n\n"
        "Model prediction for the hole (incorrect):\n"
        f"{pred}\n\n"
        "Describe the residual only. Relations are the links the model still needs to learn, "
        "anchored on what the gold has and the prediction lacks. "
        "If the prediction shares no real structure with the gold, describe the whole gold span instead.\n"
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
    model_prediction: str | None = None,
) -> dict[str, Any]:
    client = _build_openai_client()
    model_name = model or _env("ANNOTATE_MODEL") or _env("LLM_RETRIEVE_MODEL") or "qwen-plus"
    mt = max_tokens or int(_env("ANNOTATE_MAX_TOKENS") or "4096")
    pred = (model_prediction or "").strip()
    residual = bool(pred) and not _fills_equivalent(gold_completion, pred)
    if residual:
        messages = build_valid_semantic_messages(
            fim_prompt=fim_prompt,
            gold_completion=gold_completion,
            model_prediction=pred,
            language=language,
        )
    else:
        messages = build_llm_semantic_messages(
            fim_prompt=fim_prompt,
            gold_completion=gold_completion,
            language=language,
        )
    extra = _extra_body()
    print(
        f"[llm-semantic] model={model_name} mode={'residual' if residual else 'full_span'} "
        f"surface_chars={len(fim_prompt)} gold_chars={len(gold_completion)} "
        f"pred_chars={len(pred)}",
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
        "query_mode": "residual" if residual else "full_span",
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
    """Two-stage retrieve when embeddings exist; else Stage-2 on the full jsonl."""
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
                f"[llm-semantic] embed recall failed ({exc}); "
                "fallback full-corpus Stage-2",
                flush=True,
            )
    return _search_stage2_full_corpus(
        path,
        query_sem,
        top_k=top_k,
        max_scan=max_scan,
        min_score=min_score,
    )


def _search_stage2_full_corpus(
    path: Path,
    query_sem: dict[str, Any] | None,
    *,
    top_k: int = 10,
    max_scan: int | None = None,
    min_score: float = 0.01,
) -> list[dict[str, Any]]:
    """Stage-2 struct score on every semantic jsonl row (no embedding Top-500)."""
    from src.fim_semantic_index import (
        _hit_from_row,
        build_endpoint_sim,
        diversify_mechanism_hits,
        embed_model_id,
    )

    q = normalize_semantic_repr(query_sem if isinstance(query_sem, dict) else {})
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
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
            row = dict(row)
            row["_semantic_line"] = idx
            rows.append(row)

    rel_mode = "lexical"
    endpoint_sim = lexical_endpoint_sim
    model = embed_model_id()
    if model:
        try:
            phrases = collect_relation_endpoints(q, *rows)
            endpoint_sim = build_endpoint_sim(phrases, model)
            rel_mode = "endpoint_embed"
        except Exception as exc:
            print(
                f"[llm-semantic] full-corpus endpoint embed failed ({exc}); "
                "lexical relation endpoints",
                flush=True,
            )
            endpoint_sim = lexical_endpoint_sim
            rel_mode = "lexical"

    ranked_items: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for rank, row in enumerate(rows, start=1):
        combined, struct, parts = semantic_combined_score(
            q,
            row,
            embed_cos=None,
            endpoint_sim=endpoint_sim,
        )
        if combined < min_score:
            continue
        hit = _hit_from_row(
            row,
            score=combined,
            struct_score=struct,
            embed_score=0.0,
            recall_rank=rank,
            parts=parts,
        )
        ranked_items.append((combined, hit, row))

    ranked_items.sort(key=lambda x: (-x[0], x[1].get("line") or 0))
    hits = diversify_mechanism_hits(ranked_items, max(1, top_k), endpoint_sim)
    scanned = len(rows)
    for h in hits:
        h["scanned_rows"] = scanned
        h["recall_k"] = scanned
        h["retrieval"] = "struct_rerank_full"
        h["relation_align"] = rel_mode
        h["embed_score"] = 0.0
    print(
        f"[llm-semantic] pipeline=struct_rerank_full hits={len(hits)} "
        f"scanned={scanned} relation_align={rel_mode}",
        flush=True,
    )
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
    model_prediction: str | None = None,
) -> dict[str, Any]:
    if not (fim_prompt or "").strip():
        raise ValueError("fim_prompt is required")
    if not (gold_completion or "").strip():
        raise ValueError("gold_completion is required")

    prepared = prepare_llm_train_query(fim_prompt, gold_completion)
    pred = clean_gold_mid_completion(model_prediction or "") or (model_prediction or "").strip()
    llm_out = call_llm_semantic_analyze(
        fim_prompt=prepared["fim_problem_surface"],
        gold_completion=prepared["gold_mid_completion"],
        language=language,
        model_prediction=pred,
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
                    if pipe == "struct_rerank_full":
                        entry["why"] = (
                            "full-corpus Stage-2 (no EIF_LLM_SEMANTIC_EMBEDDINGS / sidecar npz). "
                            "Same relation/pattern/role/operations score as two-stage rerank; "
                            "no embedding Top-500."
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
            "model_prediction_preview": pred[:400],
            "query_mode": llm_out.get("query_mode") or "full_span",
        },
        "prepared": prepared,
    }


def _hydrate_env() -> None:
    try:
        from src.gold_live_attribution import _hydrate_eif_env
        _hydrate_eif_env(force_file=True)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    """Search the semantic corpus with a ready-made 4-field JSON (no LLM analyze)."""
    import argparse
    import sys

    _hydrate_env()
    p = argparse.ArgumentParser(
        description="Search EIF_LLM_SEMANTIC_CORPUS with a 4-field semantic JSON. "
        "Does not call the generative LLM. Loads repo-root eif_api.env.",
    )
    p.add_argument(
        "--query-json",
        default="-",
        help="path to query JSON, or '-' for stdin (default: stdin)",
    )
    p.add_argument(
        "--corpus",
        default="",
        help="semantic jsonl (default: EIF_LLM_SEMANTIC_CORPUS)",
    )
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args(argv)

    raw = sys.stdin.read() if args.query_json.strip() == "-" else Path(args.query_json).read_text(encoding="utf-8")
    try:
        query = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"invalid query JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(query, dict):
        print("query JSON must be an object", file=sys.stderr)
        return 2
    sem = semantic_export_repr(query)
    corpus = (args.corpus or "").strip() or _env("EIF_LLM_SEMANTIC_CORPUS")
    if not corpus:
        print("set EIF_LLM_SEMANTIC_CORPUS in eif_api.env or pass --corpus", file=sys.stderr)
        return 2

    print(f"corpus={corpus}", flush=True)
    print(f"query={json.dumps(sem, ensure_ascii=False)}", flush=True)
    print(f"flat=\n{flatten_semantic_text(sem)}", flush=True)
    hits = search_semantic_corpus_jsonl(corpus, sem, top_k=max(1, int(args.top_k)))
    print(f"hits={len(hits)}", flush=True)
    for i, h in enumerate(hits):
        print(
            json.dumps(
                {
                    "rank": i,
                    "line": h.get("line"),
                    "semantic_score": h.get("semantic_score"),
                    "struct_score": h.get("struct_score"),
                    "embed_score": h.get("embed_score"),
                    "task_id": h.get("task_id"),
                    "pattern": h.get("pattern"),
                    "prompt_preview": h.get("prompt_preview"),
                    "response_preview": h.get("response_preview"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
