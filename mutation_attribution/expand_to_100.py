#!/usr/bin/env python3
"""Build 100 ground-truth queries whose parents are rows of csn_go_train_fim_10k.jsonl.

Rows already in go_fim_mutations.jsonl are kept when their task_id is in that
10k file and the parent text matches. ``ground_truth_line`` is rewritten to the
0-based index in the 10k file. The remaining slots are new rewrites: every user
identifier is renamed off the parent vocabulary, string literals are split, and
if/else chains are turned into switches (or the reverse).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_go_fim_mutations import (  # noqa: E402
    STOP,
    TRAIN,
    _parser,
    ast_hists,
    content_tokens,
    cosine,
    full_code,
    hist,
    make_prompt,
    rank_of,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "go_fim_mutations.jsonl"
TRAIN_NAME = TRAIN.name
# Parents that already existed in the full-corpus set and also sit in the 10k file.
# Later rows are marked origin=10k and are kept on rerun.
SETTLED_LINES = {
    7985, 3875, 1288, 942, 2916, 3516, 2928, 77, 7655, 20, 8833, 9278, 336, 3408, 4472, 1679,
}
MARK_A = "/*__HOLE_A__*/"
MARK_B = "/*__HOLE_B__*/"
TARGET = 100

KEEP = set(STOP) | {"delete", "clear", "min", "max"}

_SYL = [
    "bri", "cal", "dor", "fen", "gal", "hin", "jor", "kel", "lun", "mar",
    "nel", "orin", "pel", "quin", "rin", "sol", "tar", "vin", "wex", "yor",
    "zed", "axo", "bem", "cor", "dax", "elm", "fay", "gor", "hel", "ivy",
    "jun", "koa", "lex", "mio", "nox", "oak", "pia", "qua", "rue", "sen",
]
_WORDS = [a + b for a in _SYL for b in _SYL if a != b]

BAD_REPO = (
    "aws/aws-sdk-go",
    "kubernetes/kubernetes",
    "openshift/origin",
    "prometheus/",
    "etcd-io/",
    "Azure/azure-sdk",
    "google/go-github",
    "hashicorp/terraform",
)
TRIVIAL_MID = {
    "return",
    "return nil",
    "return err",
    "return nil, err",
    "return nil, nil",
    "return 0, nil",
    "return false",
    "return true",
}


def _lex(src: str):
    """Yield (kind, start, end). kind is ident, string, raw, rune, comment, other."""
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            yield "comment", i, j
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            yield "comment", i, j
            i = j
            continue
        if ch == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            yield "string", i, j
            i = j
            continue
        if ch == "`":
            j = src.find("`", i + 1)
            j = n if j < 0 else j + 1
            yield "raw", i, j
            i = j
            continue
        if ch == "'":
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "'":
                    j += 1
                    break
                j += 1
            yield "rune", i, j
            i = j
            continue
        if ch == "_" or ch.isalpha():
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            yield "ident", i, j
            i = j
            continue
        yield "other", i, i + 1
        i += 1


def _rename(src: str, banned: set[str]) -> str:
    idents = []
    for kind, a, b in _lex(src):
        if kind != "ident":
            continue
        word = src[a:b]
        if word == "_" or word in KEEP or word.casefold() in KEEP:
            continue
        idents.append(word)
    mapping: dict[str, str] = {}
    pool = iter(w for w in _WORDS if w not in banned and w not in KEEP)
    for word in idents:
        if word in mapping:
            continue
        nxt = next(pool)
        while nxt.casefold() in banned or nxt in mapping.values():
            nxt = next(pool)
        mapping[word] = nxt
    out = []
    for kind, a, b in _lex(src):
        if kind == "ident" and src[a:b] in mapping:
            out.append(mapping[src[a:b]])
        else:
            out.append(src[a:b])
    return "".join(out)


def _split_string_token(tok: str) -> str:
    if tok.startswith("`"):
        inner = tok[1:-1]
        if len(inner) < 2:
            return tok
        h = max(1, len(inner) // 2)
        return "`" + inner[:h] + "` + `" + inner[h:] + "`"
    body = tok[1:-1]
    if len(body) < 2:
        return tok
    h = max(1, len(body) // 2)
    if h < len(body) and body[h - 1] == "\\":
        h += 1
    if h >= len(body):
        return tok
    return '"' + body[:h] + '" + "' + body[h:] + '"'


def _split_strings(src: str, banned: set[str]) -> str:
    import re

    ident = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    def dirty(tok: str) -> bool:
        if tok[:1] not in {'"', "`"}:
            return False
        body = tok[1:-1]
        if len(body) >= 4:
            return True
        return any(
            m.group(0).casefold() in banned and len(m.group(0)) > 1
            for m in ident.finditer(body)
        )

    cur = src
    for _ in range(8):
        out = []
        changed = False
        for kind, a, b in _lex(cur):
            tok = cur[a:b]
            if kind in ("string", "raw") and dirty(tok):
                nxt = _split_string_token(tok)
                changed = changed or nxt != tok
                out.append(nxt)
            else:
                out.append(tok)
        cur = "".join(out)
        if not changed:
            break
    return cur


def _cond_has_simple_stmt(cond: str) -> bool:
    depth = 0
    i = 0
    n = len(cond)
    while i < n:
        ch = cond[i]
        if cond.startswith("//", i):
            j = cond.find("\n", i)
            i = n if j < 0 else j
            continue
        if cond.startswith("/*", i):
            j = cond.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if ch in "\"'`":
            if ch == "`":
                j = cond.find("`", i + 1)
                i = n if j < 0 else j + 1
                continue
            j = i + 1
            while j < n:
                if cond[j] == "\\":
                    j += 2
                    continue
                if cond[j] == ch:
                    j += 1
                    break
                j += 1
            i = j
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth == 0:
            return True
        i += 1
    return False


def _if_chain(node, src: str):
    kids = list(node.children)
    body_i = next(i for i, c in enumerate(kids) if c.type == "block")
    cond = src[kids[0].end_byte: kids[body_i].start_byte].strip()
    block = kids[body_i]
    body = src[block.start_byte + 1: block.end_byte - 1]
    rest = kids[body_i + 1:]
    if len(rest) >= 2 and rest[0].type == "else" and rest[1].type == "block":
        else_body = src[rest[1].start_byte + 1: rest[1].end_byte - 1]
        return [(cond, body)], else_body
    if len(rest) >= 2 and rest[0].type == "else" and rest[1].type == "if_statement":
        more, else_body = _if_chain(rest[1], src)
        return [(cond, body)] + more, else_body
    return [(cond, body)], None


def _switch_text(chain, else_body) -> str:
    parts = ["switch {"]
    for cond, body in chain:
        parts.append(f"case {cond}:")
        parts.append(body)
    if else_body is not None:
        parts.append("default:")
        parts.append(else_body)
    parts.append("}")
    return "\n".join(parts)


def _case_body(case, src: str) -> str:
    colon = next(c for c in case.children if c.type == ":")
    return src[colon.end_byte: case.end_byte]


def _case_expr(case, src: str) -> str:
    for child in case.children:
        if child.type == "expression_list":
            return src[child.start_byte: child.end_byte]
    return ""


def _switch_to_if(node, src: str) -> str | None:
    if "fallthrough" in src[node.start_byte: node.end_byte]:
        return None
    kids = list(node.children)
    brace = next(c for c in kids if c.type == "{")
    tag = src[kids[0].end_byte: brace.start_byte].strip()
    cases = [c for c in kids if c.type in ("expression_case", "default_case")]
    if not cases:
        return None
    ordinary = []
    default_body = None
    for case in cases:
        body = _case_body(case, src)
        if case.type == "default_case":
            default_body = body
            continue
        expr = _case_expr(case, src).strip()
        if tag:
            bits = [p.strip() for p in expr.split(",") if p.strip()]
            if not bits:
                return None
            cond = " || ".join(f"({tag} == ({bit}))" for bit in bits)
        else:
            cond = expr
        if not cond:
            return None
        ordinary.append((cond, body))
    if not ordinary:
        return None
    lines = []
    for i, (cond, body) in enumerate(ordinary):
        lines.append(("if" if i == 0 else "} else if") + f" {cond} {{")
        lines.append(body)
    if default_body is not None:
        lines.append("} else {")
        lines.append(default_body)
    lines.append("}")
    return "\n".join(lines)


def _outer_nodes(tree, type_name: str):
    found = []

    def walk(node, parent_type: str):
        if node.type == type_name and parent_type != type_name:
            found.append(node)
            return
        for child in node.children:
            walk(child, node.type)

    walk(tree.root_node, "")
    return found


def _apply_spans(src: str, spans: list[tuple[int, int, str]]) -> str:
    spans = sorted(spans, key=lambda item: item[0], reverse=True)
    for start, end, text in spans:
        src = src[:start] + text + src[end:]
    return src


def _rewrite_control(src: str, parser) -> tuple[str, str]:
    tree = parser.parse(src.encode("utf-8"))
    if tree.root_node.has_error:
        return src, "rename-only"
    spans = []
    kind = []
    for node in _outer_nodes(tree, "if_statement"):
        try:
            chain, else_body = _if_chain(node, src)
        except (StopIteration, IndexError):
            continue
        if any(_cond_has_simple_stmt(cond) for cond, _body in chain):
            continue
        if any(not cond.strip() for cond, _body in chain):
            continue
        spans.append((node.start_byte, node.end_byte, _switch_text(chain, else_body)))
        kind.append("if-to-switch")
    if not spans:
        for node in _outer_nodes(tree, "expression_switch_statement"):
            text = _switch_to_if(node, src)
            if not text:
                continue
            spans.append((node.start_byte, node.end_byte, text))
            kind.append("switch-to-if")
    if spans:
        return _apply_spans(src, spans), "+".join(sorted(set(kind)))
    wrapped = _wrap_body(src, parser)
    if wrapped != src:
        return wrapped, "body-switch"
    return src, "rename-only"


def _wrap_body(src: str, parser) -> str:
    tree = parser.parse(src.encode("utf-8"))
    func = None

    def walk(node):
        nonlocal func
        if func is None and node.type == "function_declaration":
            func = node
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    if func is None:
        return src
    block = next((c for c in func.children if c.type == "block"), None)
    if block is None or block.end_byte - block.start_byte < 2:
        return src
    inner = src[block.start_byte + 1: block.end_byte - 1]
    wrapped = "\nswitch {\ndefault:\n" + inner + "\n}\n"
    return src[: block.start_byte + 1] + wrapped + src[block.end_byte - 1:]


def _strip_comments(src: str) -> str:
    out = []
    for kind, a, b in _lex(src):
        if kind == "comment":
            out.append("\n" if "\n" in src[a:b] else " ")
        else:
            out.append(src[a:b])
    return "".join(out)


def rewrite(prefix: str, middle: str, suffix: str, parser) -> tuple[str, str, str, str] | None:
    marked = f"{prefix}{MARK_A}{middle}{MARK_B}{suffix}"
    banned = {tok for tok in content_tokens(prefix + middle + suffix)}
    renamed = _rename(marked, banned)
    renamed = _split_strings(renamed, banned)
    rewritten, kind = _rewrite_control(renamed, parser)
    if MARK_A not in rewritten or MARK_B not in rewritten:
        return None
    pre, rest = rewritten.split(MARK_A, 1)
    mid, suf = rest.split(MARK_B, 1)
    pre, mid, suf = _strip_comments(pre), _strip_comments(mid), _strip_comments(suf)
    if not mid.strip():
        return None
    return pre, mid, suf, kind


def _load_existing() -> tuple[list[dict], set[int], set[str]]:
    rows = []
    lines: set[int] = set()
    repos: set[str] = set()
    if OUT.is_file():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            if "ground_truth_line" in row:
                lines.add(int(row["ground_truth_line"]))
                repos.add(str(row.get("ground_truth_repo") or ""))
    return rows, lines, repos


def _stride(items: list, count: int) -> list:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[min(len(items) - 1, int(i * step))] for i in range(count)]


def _pick_parents(used_lines: set[int], used_repos: set[str], limit: int) -> list[tuple[int, dict]]:
    """One function per repo, spread across the whole 10k file."""
    primary: list[tuple[int, dict]] = []
    secondary: list[tuple[int, dict]] = []
    seen_repo: set[str] = set()
    with TRAIN.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx in used_lines:
                continue
            row = json.loads(line)
            repo = str(row.get("repository_name") or "")
            if any(bad in repo for bad in BAD_REPO):
                continue
            pre = row.get("prefix") or ""
            mid = row.get("middle") or ""
            suf = row.get("suffix") or ""
            if not (180 <= len(pre) + len(mid) + len(suf) <= 1100):
                continue
            if len(mid.strip()) < 8:
                continue
            if " ".join(mid.split()) in TRIVIAL_MID:
                continue
            if not str(pre).lstrip().startswith("func"):
                continue
            item = (idx, row)
            if repo not in seen_repo and repo not in used_repos:
                seen_repo.add(repo)
                primary.append(item)
            else:
                secondary.append(item)
    # First `limit` candidates already span the file. Extras are only used
    # when a rewrite fails to parse or still shares a content token.
    preferred = _stride(primary, limit)
    have = {idx for idx, _row in preferred}
    extras = _stride([item for item in primary if item[0] not in have], limit)
    picked = preferred + extras
    if len(picked) < limit:
        for item in _stride(secondary, limit):
            if item[0] in have:
                continue
            picked.append(item)
            have.add(item[0])
    return picked


def _semantic(kind: str, middle: str) -> dict:
    return {
        "role": "same hole duty as the parent sample",
        "pattern": [f"{kind} paraphrase of the parent mechanism"],
        "operations": [" ".join(middle.split())[:220]],
        "relations": [
            {
                "source": "rewritten hole",
                "target": "the same surrounding mechanism as the parent",
                "type": "semantic_dependency",
            }
        ],
    }


def _score(mutant: str, parent: str, parser, pool_tokens, pool_ast) -> dict:
    q_toks = content_tokens(mutant)
    p_toks = content_tokens(parent)
    shared = sorted(set(q_toks) & set(p_toks))
    q_hist = hist(q_toks)
    p_cos = cosine(q_hist, hist(p_toks))
    other_cos = [cosine(q_hist, hist(toks)) for _idx, toks in pool_tokens]
    q_types, q_edges, bad = ast_hists(parser, mutant)
    p_types, p_edges, _pbad = ast_hists(parser, parent)
    ast_parent = 0.5 * (cosine(q_types, p_types) + cosine(q_edges, p_edges))
    ast_others = [
        0.5 * (cosine(q_types, types) + cosine(q_edges, edges))
        for _idx, types, edges in pool_ast
    ]
    return {
        "parse_error": bad,
        "shared_content_tokens": shared,
        "token_cosine_to_parent": round(p_cos, 4),
        "token_rank_in_251": rank_of(p_cos, other_cos),
        "token_median_other": round(sorted(other_cos)[len(other_cos) // 2], 4),
        "ast_score_to_parent": round(ast_parent, 4),
        "ast_rank_in_251": rank_of(ast_parent, ast_others),
        "ast_median_other": round(sorted(ast_others)[len(ast_others) // 2], 4),
    }


def _pool(parser, used_lines: set[int]):
    random.seed(0)
    reservoir: list[tuple[int, str]] = []
    seen = 0
    with TRAIN.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx in used_lines:
                continue
            seen += 1
            code = full_code(json.loads(line))
            if len(code) < 40:
                continue
            item = (idx, code)
            if len(reservoir) < 250:
                reservoir.append(item)
            else:
                j = random.randrange(seen)
                if j < 250:
                    reservoir[j] = item
    pool_tokens = [(idx, content_tokens(code)) for idx, code in reservoir]
    pool_ast = []
    for idx, code in reservoir:
        types, edges, _bad = ast_hists(parser, code)
        pool_ast.append((idx, types, edges))
    return pool_tokens, pool_ast


def _mutation_id(n: int) -> str:
    return f"m{n:02d}" if n < 100 else f"m{n}"


def _is_handwritten(rec: dict) -> bool:
    role = str((rec.get("shared_semantic") or {}).get("role") or "")
    return role != "same hole duty as the parent sample"


def _index_train() -> dict[str, tuple[int, dict]]:
    found: dict[str, tuple[int, dict]] = {}
    with TRAIN.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            found[str(row.get("task_id") or "")] = (idx, row)
    return found


def _keep_10k(existing: list[dict], index: dict[str, tuple[int, dict]]) -> tuple[list[dict], list[str]]:
    """Keep queries whose parent text is the 10k row with the same task_id."""
    kept: list[dict] = []
    dropped: list[str] = []
    for rec in existing:
        hit = index.get(str(rec.get("ground_truth_task_id") or ""))
        if hit is None:
            dropped.append(str(rec.get("mutation_id")))
            continue
        idx, row = hit
        same = (
            (row.get("prefix") or "") == (rec.get("parent_prefix") or "")
            and (row.get("middle") or "") == (rec.get("parent_middle") or "")
            and (row.get("suffix") or "") == (rec.get("parent_suffix") or "")
        )
        if not same:
            dropped.append(str(rec.get("mutation_id")))
            continue
        rec = dict(rec)
        rec["ground_truth_line"] = idx
        rec["ground_truth_uid"] = row.get("uid")
        rec["ground_truth_func"] = row.get("func_name")
        rec["ground_truth_repo"] = row.get("repository_name")
        rec["train_file"] = TRAIN_NAME
        kept.append(rec)
    def _sort_key(rec: dict) -> tuple:
        mid = str(rec.get("mutation_id") or "")
        number = int(mid[1:]) if mid.startswith("m") and mid[1:].isdigit() else 10**9
        return (0 if _is_handwritten(rec) else 1, number)

    kept.sort(key=_sort_key)
    return kept, dropped


def main() -> None:
    existing, _old_lines, _old_repos = _load_existing()
    index = _index_train()
    kept, dropped = _keep_10k(existing, index)
    fresh = []
    for rec in kept:
        line_no = int(rec["ground_truth_line"])
        if rec.get("origin") == "10k" or line_no in SETTLED_LINES:
            fresh.append(rec)
        else:
            dropped.append(str(rec.get("mutation_id")))
    kept = fresh
    print(f"loaded={len(existing)} kept_in_10k={len(kept)} dropped={len(dropped)}", flush=True)
    if dropped:
        print("dropped", ", ".join(dropped), flush=True)
    used_lines = {int(rec["ground_truth_line"]) for rec in kept}
    used_repos = {str(rec.get("ground_truth_repo") or "") for rec in kept}
    need = TARGET - len(kept)
    if need < 0:
        raise SystemExit(f"kept {len(kept)} which is already above {TARGET}")
    parser = _parser()
    made: list[tuple] = []
    if need == 0:
        print("already 100 parents inside the 10k file")
    else:
        parents = _pick_parents(used_lines, used_repos, need)
        print(f"need={need} candidates={len(parents)}", flush=True)
        tried = 0
        for idx, row in parents:
            if len(made) >= need:
                break
            tried += 1
            parent_code = full_code(row)
            _types, _edges, bad_parent = ast_hists(parser, parent_code)
            if bad_parent:
                continue
            try:
                rewritten = rewrite(
                    row.get("prefix") or "",
                    row.get("middle") or "",
                    row.get("suffix") or "",
                    parser,
                )
            except StopIteration:
                continue
            if rewritten is None:
                continue
            pre, mid, suf, kind = rewritten
            mutant = pre + mid + suf
            _mt, _me, bad = ast_hists(parser, mutant)
            if bad:
                continue
            shared = set(content_tokens(mutant)) & set(content_tokens(parent_code))
            if shared:
                continue
            made.append((idx, row, pre, mid, suf, kind))
            if len(made) % 10 == 0:
                print(f"  accepted {len(made)}", flush=True)
        print(f"accepted={len(made)} tried={tried}", flush=True)
        if len(made) < need:
            raise SystemExit(f"only produced {len(made)} of {need}")

    records = list(kept)
    for idx, row, pre, mid, suf, kind in made:
        records.append(
            {
                "mutation_id": "",
                "split": "mutation_valid",
                "language": "go",
                "ground_truth_line": idx,
                "ground_truth_task_id": row.get("task_id"),
                "ground_truth_uid": row.get("uid"),
                "ground_truth_func": row.get("func_name"),
                "ground_truth_repo": row.get("repository_name"),
                "train_file": TRAIN_NAME,
                "origin": "10k",
                "rewrite_note": (
                    f"{kind}; every user identifier renamed off the parent vocabulary; "
                    "string literals split. The hole still performs the parent operation."
                ),
                "shared_semantic": _semantic(kind, mid),
                "prefix": pre,
                "middle": mid,
                "suffix": suf,
                "response": mid,
                "prompt": make_prompt(pre, suf),
                "parent_prefix": row.get("prefix"),
                "parent_middle": row.get("middle"),
                "parent_suffix": row.get("suffix"),
            }
        )

    score_lines = {int(rec["ground_truth_line"]) for rec in records}
    pool_tokens, pool_ast = _pool(parser, score_lines)
    print(f"{'id':<6}{'prior':<8}{'line':<8}{'astR':<6}parent")
    for n, rec in enumerate(records, start=1):
        rec["mutation_id"] = _mutation_id(n)
        rec["train_file"] = TRAIN_NAME
        parent_code = full_code(
            {
                "prefix": rec.get("parent_prefix"),
                "middle": rec.get("parent_middle"),
                "suffix": rec.get("parent_suffix"),
            }
        )
        rec["gap"] = _score(rec["prefix"] + rec["middle"] + rec["suffix"], parent_code, parser, pool_tokens, pool_ast)
        print(
            f"{rec['mutation_id']:<6}{str(rec.get('prior_mutation_id') or '-'):<8}"
            f"{rec['ground_truth_line']:<8}{rec['gap']['ast_rank_in_251']:<6}{rec.get('ground_truth_func')}"
        )

    OUT.write_text(
        "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records),
        encoding="utf-8",
    )
    lines = [int(rec["ground_truth_line"]) for rec in records]
    n_share = sum(1 for rec in records if rec["gap"]["shared_content_tokens"])
    n_err = sum(1 for rec in records if rec["gap"]["parse_error"])
    n_ast = sum(1 for rec in records if rec["gap"]["ast_rank_in_251"] <= 5)
    print(
        f"wrote {OUT} n={len(records)} train={TRAIN_NAME} "
        f"line {min(lines)}..{max(lines)} shared={n_share} parse_err={n_err} ast_top5={n_ast}"
    )


if __name__ == "__main__":
    main()
