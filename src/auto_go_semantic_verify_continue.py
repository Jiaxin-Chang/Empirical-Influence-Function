#!/usr/bin/env python3
"""Go 错测样本：semantic 归因 → 逐条验证 → 通过后写入续训小集。

对 predictions JSONL 里 label≠predict 的每一条：

  1. ``/api/llm-semantic-retrieve`` 取 Top-K 训练样本（默认 10）
  2. 按分数从高到低依次验证 Top-``--max-verify``（默认前 3 条；检索仍取 Top-10）
  3. 每条候选：MID 对齐 → LLM 语义标注（附带 **测试样本** 的 4 字段 semantic）
  4. 把标注后的 **这一条** 复制 20 份写进独立 trial JSONL（不叠加其它续训样本）
  5. 用这 20 条做 CE+saliency 续训，看当前 test 的 gold CE 是否下降
  6. 恢复原始 adapter 后，再用同一 20 条做纯 CE 续训
  7. 若 saliency 下降量 > 0 **且** 纯 CE 下降量 < saliency 下降量 →
     把这 20 份追加进 ``--accepted-path``，进入下一条测试
  8. 第 1 条不满足就试第 2 条，第 3 条仍不行则跳过该测试

续训通过 HTTP 打 ``ttav_bundle_api``（``trainData`` 显式指向 trial 文件，
不会读 viewer 上已经攒下来的续训小集）。每次续训后 ``/api/continue-adapter-recover``，
下一次从 ``EIF_ADAPTER_PATH_CE`` 再出发。

Prerequisites（``eif_api.env`` 已加载，两端 API 已起）::

    python -m src.ttav_bundle_api

    cd tools/annotation-viewer && python -m server.main \\
      --continue-data /mnt/md124/jiaxin/Empirical-Influence-Function/go_continue_verify_scratch.jsonl

viewer 的 continue JSONL 必须是 **scratch**（标注落盘用），不能是
``--accepted-path``。验证通过的 20 份只由本脚本追加到 accepted 文件。

Example::

    python -m src.auto_go_semantic_verify_continue

    python -m src.auto_go_semantic_verify_continue --max-tests 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.auto_raw_ce_to_continue import (
    PipelineClient,
    _collect_semantic_hits,
    _continue_paths_match,
    _fim_and_gold,
    _http_json,
    _is_mismatch,
    _load_raw_rows,
    _mid_decision,
    _row_language,
    _viewer_health,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PREDICTIONS = "/mnt/md124/jiaxin/go_ce_outputs/qwen3-8b-go.predictions.jsonl"
DEFAULT_CORPUS = "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.jsonl"
DEFAULT_SEMANTIC_CORPUS = (
    "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.semantic.jsonl"
)
DEFAULT_SCRATCH = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/go_continue_verify_scratch.jsonl"
)
DEFAULT_ACCEPTED = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/go_continue_verified_subset.jsonl"
)


@dataclass
class VerifyState:
    copies: int
    max_tests: int
    accepted: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    processed_test_lines: list[int] = field(default_factory=list)
    used_corpus_lines: set[int] = field(default_factory=set)

    @property
    def n_ok_tests(self) -> int:
        return len(self.accepted)

    @property
    def n_continue_rows(self) -> int:
        return sum(int(r.get("continue_rows") or 0) for r in self.accepted)

    def done(self) -> bool:
        return self.max_tests > 0 and self.n_ok_tests >= self.max_tests


def _save_state(path: Path | None, state: VerifyState) -> None:
    if path is None:
        return
    payload = {
        "copies": state.copies,
        "max_tests": state.max_tests,
        "n_ok_tests": state.n_ok_tests,
        "n_continue_rows": state.n_continue_rows,
        "accepted": state.accepted,
        "skipped": state.skipped[-400:],
        "processed_test_lines": state.processed_test_lines,
        "used_corpus_lines": sorted(state.used_corpus_lines),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state(
    path: Path | None,
    *,
    copies: int,
    max_tests: int,
) -> VerifyState:
    state = VerifyState(copies=copies, max_tests=max_tests)
    if path is None or not path.is_file():
        return state
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return state
    for row in obj.get("accepted") or []:
        if isinstance(row, dict):
            state.accepted.append(row)
            line = row.get("corpus_line")
            if line is not None:
                state.used_corpus_lines.add(int(line))
    for line in obj.get("processed_test_lines") or []:
        try:
            state.processed_test_lines.append(int(line))
        except (TypeError, ValueError):
            pass
    for line in obj.get("used_corpus_lines") or []:
        try:
            state.used_corpus_lines.add(int(line))
        except (TypeError, ValueError):
            pass
    print(
        f"[resume] loaded ok_tests={state.n_ok_tests} "
        f"continue_rows≈{state.n_continue_rows} from {path}",
        flush=True,
    )
    return state


def _compact_ok(row: dict[str, Any]) -> bool:
    ids = row.get("input_ids")
    labs = row.get("label", row.get("labels"))
    return (
        isinstance(ids, list)
        and isinstance(labs, list)
        and bool(ids)
        and len(labs) == len(ids)
    )


def _n_continue_edges(row: dict[str, Any]) -> int:
    n = 0
    for e in row.get("attention_edges") or []:
        if not isinstance(e, dict):
            continue
        contrib = str(e.get("contrib") or "")
        if contrib in ("user_add", "user_bump", "llm_auto"):
            n += 1
        elif "src" in e and "dst" in e:
            n += 1
    return n


def _read_continue_row(path: Path, corpus_line: int) -> dict[str, Any] | None:
    """Last compact continue row whose ``source_corpus_line`` matches."""
    if not path.is_file():
        return None
    found: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            try:
                line = int(obj.get("source_corpus_line"))
            except (TypeError, ValueError):
                continue
            if line != int(corpus_line):
                continue
            if _compact_ok(obj):
                found = obj
    return found


def _write_copies(row: dict[str, Any], dest: Path, copies: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    base_uid = str(row.get("uid") or row.get("task_id") or row.get("raw_id") or "sample")
    with dest.open("w", encoding="utf-8") as handle:
        for i in range(max(1, int(copies))):
            obj = json.loads(json.dumps(row, ensure_ascii=False))
            obj["uid"] = f"{base_uid}::v{i:02d}_{uuid.uuid4().hex[:8]}"
            obj["raw_id"] = obj["uid"]
            obj["duplicate_of"] = base_uid
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return max(1, int(copies))


def _append_jsonl(path: Path, src: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as handle, path.open("a", encoding="utf-8") as out:
        for raw in handle:
            if not raw.strip():
                continue
            out.write(raw if raw.endswith("\n") else raw + "\n")
            n += 1
    return n


def _write_one_test(path: Path, *, task_id: str, prompt: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"task_id": task_id, "prompt": prompt, "label": label},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _loss_pair(current_test: dict[str, Any] | None) -> tuple[float | None, float | None, float | None]:
    cur = current_test if isinstance(current_test, dict) else {}
    before = cur.get("loss_before")
    after = cur.get("loss_after")
    delta = cur.get("loss_delta")
    try:
        b = float(before) if before is not None else None
    except (TypeError, ValueError):
        b = None
    try:
        a = float(after) if after is not None else None
    except (TypeError, ValueError):
        a = None
    try:
        d = float(delta) if delta is not None else None
    except (TypeError, ValueError):
        d = None
    if d is None and b is not None and a is not None:
        d = round(a - b, 6)
    return b, a, d


def _wait_continue_job(
    eif_url: str,
    job_id: str,
    *,
    timeout: float,
    poll: float,
) -> dict[str, Any]:
    t0 = time.time()
    last_msg = ""
    while True:
        st = _http_json(
            "GET",
            f"{eif_url.rstrip('/')}/api/continue-train-eval-status"
            f"?jobId={job_id}",
            timeout=30.0,
        )
        stage = str(st.get("stage") or "")
        msg = str(st.get("message") or stage)
        if msg != last_msg:
            print(f"       job {job_id} [{stage}] {msg}", flush=True)
            last_msg = msg
        if stage == "completed":
            result = st.get("result")
            return result if isinstance(result, dict) else {}
        if stage == "error" or st.get("error") is True:
            raise RuntimeError(msg or f"continue-train job {job_id} failed")
        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"continue-train job {job_id} timed out after {timeout:.0f}s "
                f"(last: {msg})"
            )
        time.sleep(max(1.0, poll))


def _recover_adapter(eif_url: str) -> None:
    try:
        out = _http_json(
            "POST",
            f"{eif_url.rstrip('/')}/api/continue-adapter-recover",
            timeout=180.0,
        )
        print(
            f"       recover adapter: {out.get('message') or 'ok'}",
            flush=True,
        )
    except Exception as exc:
        print(f"       [warn] adapter recover failed: {exc}", flush=True)


def _run_continue_train(
    eif_url: str,
    *,
    train_data: Path,
    test_data: Path,
    output_dir: Path,
    loss_mode: str,
    current_test: dict[str, Any],
    max_steps: int,
    learning_rate: float,
    adapter_family: str,
    timeout: float,
    poll: float,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "mode": "train",
        "trainData": str(train_data),
        "testData": str(test_data),
        "outputDir": str(output_dir),
        "maxSteps": int(max_steps),
        "learningRate": float(learning_rate),
        "lossMode": loss_mode,
        "evalBefore": False,
        "evalAfterFull": False,
        "reportFamily": adapter_family,
        "adapterFamily": adapter_family,
        "currentTest": current_test,
    }
    t0 = time.time()
    while True:
        try:
            resp = _http_json(
                "POST",
                f"{eif_url.rstrip('/')}/api/continue-train-eval",
                body,
                timeout=60.0,
            )
            break
        except RuntimeError as exc:
            if "HTTP 409" not in str(exc):
                raise
            if time.time() - t0 > timeout:
                raise TimeoutError(f"continue-train still busy: {exc}") from exc
            print(f"       wait for other train job: {exc}", flush=True)
            time.sleep(max(5.0, poll))
    job_id = str(resp.get("jobId") or "").strip()
    if not job_id:
        raise RuntimeError(f"continue-train missing jobId: {resp}")
    print(f"       started {loss_mode} job={job_id}", flush=True)
    return _wait_continue_job(eif_url, job_id, timeout=timeout, poll=poll)


def _annotate_hit(
    client: PipelineClient,
    *,
    gold: str,
    language: str,
    hit: dict[str, Any],
    expression: str,
    hit_corpus: str | None,
    target_semantic: dict[str, Any] | None,
    scratch_path: Path,
) -> tuple[dict[str, Any] | None, str]:
    cline = hit.get("line")
    if cline is None:
        return None, "hit missing corpus line"
    cline = int(cline)
    region = str(hit.get("match_region") or "")
    try:
        prep = client.mid_rewrite_prep(
            line=cline,
            gold=gold,
            expression=expression,
            corpus_path=hit_corpus,
        )
    except Exception as exc:
        return None, f"mid_prep_error: {exc}"

    ok, rewrite_id, note = _mid_decision(
        prep,
        region,
        keep_original_if_unrewritten=True,
    )
    if not ok:
        return None, note
    print(f"       MID ok: {note}", flush=True)

    if client.dry_run:
        return {"dry_run": True, "corpus_line": cline}, f"dry_run ({note})"

    try:
        client.bind_rewrite(
            line=cline,
            rewrite_id=rewrite_id,
            corpus_path=hit_corpus,
        )
        accept = client.annotate_and_accept(
            line=cline,
            corpus_path=hit_corpus,
            language=language,
            target_semantic=target_semantic,
        )
    except Exception as exc:
        return None, f"annotate_failed: {exc}"

    n_edges = int(accept.get("n_continue_edges") or 0)
    continue_path = Path(str(accept.get("continue_path") or scratch_path))
    row = _read_continue_row(continue_path, cline)
    if row is None:
        return None, (
            f"annotated but compact continue row not found "
            f"(line={cline} path={continue_path} edges={n_edges})"
        )
    if _n_continue_edges(row) <= 0:
        return None, f"annotate produced 0 continue edges (line={cline})"
    row["_verify_meta"] = {
        "corpus_line": cline,
        "rewrite_id": rewrite_id,
        "mid_note": note,
        "n_continue_edges": _n_continue_edges(row),
        "accept_edges": n_edges,
        "match_region": region,
        "semantic_score": hit.get("semantic_score"),
        "task_id": hit.get("task_id"),
    }
    return row, note


def _verify_candidate(
    eif_url: str,
    *,
    trial_path: Path,
    test_path: Path,
    workdir: Path,
    test_line: int,
    rank: int,
    current_test: dict[str, Any],
    max_steps: int,
    learning_rate: float,
    adapter_family: str,
    min_drop: float,
    timeout: float,
    poll: float,
) -> tuple[bool, dict[str, Any]]:
    rec: dict[str, Any] = {"rank": rank, "passed": False}
    sal_out_dir = workdir / "runs" / f"t{test_line}_r{rank}_ce_saliency"
    ce_out_dir = workdir / "runs" / f"t{test_line}_r{rank}_ce_only"

    print("       CE+saliency continue-train on trial 20 copies only…", flush=True)
    try:
        sal_result = _run_continue_train(
            eif_url,
            train_data=trial_path,
            test_data=test_path,
            output_dir=sal_out_dir,
            loss_mode="ce_saliency",
            current_test=current_test,
            max_steps=max_steps,
            learning_rate=learning_rate,
            adapter_family=adapter_family,
            timeout=timeout,
            poll=poll,
        )
    except Exception as exc:
        rec["saliency_error"] = str(exc)
        return False, rec
    finally:
        _recover_adapter(eif_url)

    sal_b, sal_a, sal_d = _loss_pair(sal_result.get("currentTest"))
    rec["saliency"] = {
        "loss_before": sal_b,
        "loss_after": sal_a,
        "loss_delta": sal_d,
        "output_dir": str(sal_result.get("outputDir") or sal_out_dir),
    }
    sal_drop = None if sal_b is None or sal_a is None else (sal_b - sal_a)
    rec["saliency_drop"] = sal_drop
    print(
        f"       saliency gold CE {sal_b} → {sal_a} drop={sal_drop}",
        flush=True,
    )
    if sal_drop is None or sal_drop <= float(min_drop):
        rec["fail_reason"] = (
            f"saliency drop {sal_drop} not > min_drop={min_drop}"
        )
        print(f"       fail: {rec['fail_reason']}", flush=True)
        return False, rec

    print("       CE-only continue-train on the same 20 copies…", flush=True)
    try:
        ce_result = _run_continue_train(
            eif_url,
            train_data=trial_path,
            test_data=test_path,
            output_dir=ce_out_dir,
            loss_mode="ce_only",
            current_test=current_test,
            max_steps=max_steps,
            learning_rate=learning_rate,
            adapter_family=adapter_family,
            timeout=timeout,
            poll=poll,
        )
    except Exception as exc:
        rec["ce_error"] = str(exc)
        return False, rec
    finally:
        _recover_adapter(eif_url)

    ce_b, ce_a, ce_d = _loss_pair(ce_result.get("currentTest"))
    rec["ce"] = {
        "loss_before": ce_b,
        "loss_after": ce_a,
        "loss_delta": ce_d,
        "output_dir": str(ce_result.get("outputDir") or ce_out_dir),
    }
    ce_drop = None if ce_b is None or ce_a is None else (ce_b - ce_a)
    rec["ce_drop"] = ce_drop
    print(
        f"       ce-only gold CE {ce_b} → {ce_a} drop={ce_drop}",
        flush=True,
    )
    if ce_drop is None:
        rec["fail_reason"] = "ce-only gold CE missing"
        print(f"       fail: {rec['fail_reason']}", flush=True)
        return False, rec
    if not (ce_drop < sal_drop):
        rec["fail_reason"] = (
            f"ce drop {ce_drop} is not < saliency drop {sal_drop}"
        )
        print(f"       fail: {rec['fail_reason']}", flush=True)
        return False, rec

    rec["passed"] = True
    print(
        f"       PASS: saliency_drop={sal_drop} > ce_drop={ce_drop} "
        f"and saliency_drop > {min_drop}",
        flush=True,
    )
    return True, rec


def _process_test(
    client: PipelineClient,
    state: VerifyState,
    *,
    test_line: int,
    row: dict[str, Any],
    language: str,
    scratch_path: Path,
    accepted_path: Path,
    workdir: Path,
    max_verify: int,
    min_drop: float,
    max_steps: int,
    learning_rate: float,
    adapter_family: str,
    timeout: float,
    poll: float,
    state_path: Path | None,
) -> None:
    task_id = str(row.get("task_id") or f"row_{test_line}")
    fim, gold = _fim_and_gold(row)
    if not fim or not gold:
        print(f"[test {test_line}] skip: missing prompt/gold ({task_id})", flush=True)
        state.skipped.append({
            "test_line": test_line,
            "task_id": task_id,
            "reason": "missing_prompt_or_gold",
        })
        state.processed_test_lines.append(test_line)
        _save_state(state_path, state)
        return

    print(
        f"\n=== test line={test_line} task_id={task_id} "
        f"(ok_tests={state.n_ok_tests}, accepted_rows≈{state.n_continue_rows}) ===",
        flush=True,
    )
    lang = _row_language(row, language)

    try:
        retrieve = client.llm_semantic_retrieve(fim, gold, language=lang or None)
    except Exception as exc:
        print(f"  [fail] llm-semantic-retrieve: {exc}", flush=True)
        state.skipped.append({
            "test_line": test_line,
            "task_id": task_id,
            "reason": f"semantic_retrieve_failed: {exc}",
        })
        state.processed_test_lines.append(test_line)
        _save_state(state_path, state)
        return

    if str(retrieve.get("status") or "") not in ("success", "ok", ""):
        print(
            f"  [fail] semantic retrieve status={retrieve.get('status')}: "
            f"{retrieve.get('message')}",
            flush=True,
        )
        state.skipped.append({
            "test_line": test_line,
            "task_id": task_id,
            "reason": "semantic_retrieve_bad_status",
            "detail": retrieve.get("message"),
        })
        state.processed_test_lines.append(test_line)
        _save_state(state_path, state)
        return

    sr0 = (retrieve.get("search_results") or [{}])[0]
    if isinstance(sr0, dict) and sr0.get("corpus_error"):
        print(f"  [fail] semantic corpus: {sr0.get('corpus_error')}", flush=True)
        state.skipped.append({
            "test_line": test_line,
            "task_id": task_id,
            "reason": "semantic_corpus_error",
            "detail": sr0.get("corpus_error"),
        })
        state.processed_test_lines.append(test_line)
        _save_state(state_path, state)
        return

    hits = _collect_semantic_hits(retrieve)
    expression = str(
        retrieve.get("semantic_flat_text")
        or (sr0.get("expression") if isinstance(sr0, dict) else "")
        or ""
    )
    corpus_path = (
        str(retrieve.get("corpus_path") or "").strip()
        or client.corpus_path
    )
    target_semantic = (
        retrieve.get("semantic")
        if isinstance(retrieve.get("semantic"), dict)
        else None
    )
    print(
        f"  semantic hits={len(hits)} target_semantic={bool(target_semantic)} "
        f"raw_corpus={corpus_path or '-'}",
        flush=True,
    )
    for i, h in enumerate(hits[:10]):
        print(
            f"    [{i}] line={h.get('line')} sem={h.get('semantic_score')} "
            f"task={h.get('task_id') or '-'}",
            flush=True,
        )
    if not hits:
        state.skipped.append({
            "test_line": test_line,
            "task_id": task_id,
            "reason": "no_semantic_hits",
        })
        state.processed_test_lines.append(test_line)
        _save_state(state_path, state)
        return

    test_path = workdir / "current_test.jsonl"
    _write_one_test(test_path, task_id=task_id, prompt=fim, label=gold)
    current_test = {"taskId": task_id, "prompt": fim, "label": gold}

    trials: list[dict[str, Any]] = []
    candidates = hits[: max(1, int(max_verify))]
    for rank, hit in enumerate(candidates, start=1):
        cline = hit.get("line")
        try:
            cline_i = int(cline)
        except (TypeError, ValueError):
            continue
        if cline_i in state.used_corpus_lines:
            print(
                f"  [skip hit {rank}] corpus line={cline_i} already accepted",
                flush=True,
            )
            continue

        print(
            f"  try hit rank={rank}/{len(candidates)} (of {len(hits)} retrieved) "
            f"line={cline_i} sem={hit.get('semantic_score')}",
            flush=True,
        )
        compact, note = _annotate_hit(
            client,
            gold=gold,
            language=lang,
            hit=hit,
            expression=expression,
            hit_corpus=corpus_path,
            target_semantic=target_semantic,
            scratch_path=scratch_path,
        )
        if compact is None:
            print(f"       annotate skip: {note}", flush=True)
            trials.append({
                "rank": rank,
                "corpus_line": cline_i,
                "stage": "annotate",
                "reason": note,
            })
            continue

        if client.dry_run:
            print("       dry-run: skip continue-train", flush=True)
            trials.append({
                "rank": rank,
                "corpus_line": cline_i,
                "stage": "dry_run",
                "reason": note,
            })
            continue

        trial_path = workdir / "trials" / f"t{test_line}_r{rank}.jsonl"
        n_written = _write_copies(compact, trial_path, state.copies)
        print(
            f"       wrote {n_written} isolated copies → {trial_path}",
            flush=True,
        )
        passed, rec = _verify_candidate(
            client.eif_url,
            trial_path=trial_path,
            test_path=test_path,
            workdir=workdir,
            test_line=test_line,
            rank=rank,
            current_test=current_test,
            max_steps=max_steps,
            learning_rate=learning_rate,
            adapter_family=adapter_family,
            min_drop=min_drop,
            timeout=timeout,
            poll=poll,
        )
        rec["corpus_line"] = cline_i
        rec["semantic_score"] = hit.get("semantic_score")
        rec["mid_note"] = note
        rec["n_continue_edges"] = (compact.get("_verify_meta") or {}).get(
            "n_continue_edges"
        )
        trials.append(rec)
        if not passed:
            continue

        appended = _append_jsonl(accepted_path, trial_path)
        meta = compact.get("_verify_meta") or {}
        rec_ok = {
            "test_line": test_line,
            "task_id": task_id,
            "corpus_line": cline_i,
            "corpus_path": corpus_path,
            "rank": rank,
            "semantic_score": hit.get("semantic_score"),
            "mid_note": note,
            "rewrite_id": meta.get("rewrite_id"),
            "n_continue_edges": meta.get("n_continue_edges"),
            "continue_rows": appended,
            "accepted_path": str(accepted_path),
            "trial_path": str(trial_path),
            "saliency": rec.get("saliency"),
            "ce": rec.get("ce"),
            "saliency_drop": rec.get("saliency_drop"),
            "ce_drop": rec.get("ce_drop"),
            "trials": trials,
        }
        state.accepted.append(rec_ok)
        state.used_corpus_lines.add(cline_i)
        state.processed_test_lines.append(test_line)
        print(
            f"  ✓ accepted rank={rank} line={cline_i} "
            f"+{appended} rows → {accepted_path} "
            f"(ok_tests={state.n_ok_tests})",
            flush=True,
        )
        _save_state(state_path, state)
        return

    state.processed_test_lines.append(test_line)
    state.skipped.append({
        "test_line": test_line,
        "task_id": task_id,
        "reason": (
            f"none of top-{len(candidates)} hits passed "
            f"(retrieved {len(hits)})"
        ),
        "trials": trials,
    })
    print(
        f"  [skip] test line={test_line}: none of top-{len(candidates)} hits passed",
        flush=True,
    )
    _save_state(state_path, state)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Semantic retrieve Top-K → annotate with test semantic → "
            "isolated 20-copy CE+saliency vs CE continue-train → keep if saliency helps more"
        ),
    )
    p.add_argument("--input", default=DEFAULT_PREDICTIONS)
    p.add_argument("--corpus-path", default=DEFAULT_CORPUS)
    p.add_argument("--semantic-corpus-path", default=DEFAULT_SEMANTIC_CORPUS)
    p.add_argument(
        "--continue-path",
        default=DEFAULT_SCRATCH,
        help="viewer scratch JSONL (must match annotation-viewer --continue-data)",
    )
    p.add_argument(
        "--accepted-path",
        default=DEFAULT_ACCEPTED,
        help="verified continue-train set (20 copies appended per passing test)",
    )
    p.add_argument(
        "--workdir",
        default="",
        help="trial jsonl + continue-train output dirs (default: beside accepted-path)",
    )
    p.add_argument("--eif-url", default="http://127.0.0.1:8766")
    p.add_argument("--viewer-url", default="http://127.0.0.1:8765")
    p.add_argument("--language", default="go")
    p.add_argument("--copies", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10, help="semantic retrieve K")
    p.add_argument(
        "--max-verify",
        type=int,
        default=3,
        help="try at most this many top hits per test (1st, then 2nd, then 3rd)",
    )
    p.add_argument("--max-tests", type=int, default=0, help="0 = all mismatches")
    p.add_argument("--start-line", type=int, default=1)
    p.add_argument("--end-line", type=int, default=0, help="0 = last line")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument(
        "--adapter-family",
        default="ce",
        choices=("ce", "saliency"),
        help="continue-train start adapter (go_ce_outputs → ce)",
    )
    p.add_argument(
        "--min-drop",
        type=float,
        default=0.0,
        help="saliency gold-CE drop must be strictly greater than this",
    )
    p.add_argument("--train-timeout", type=float, default=10800.0)
    p.add_argument("--poll", type=float, default=5.0)
    p.add_argument("--state", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument(
        "--skip-processed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--mismatches-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        alt = REPO_ROOT / args.input
        if alt.is_file():
            input_path = alt
        else:
            print(f"input not found: {args.input}", file=sys.stderr)
            return 2

    scratch_path = Path(args.continue_path).expanduser()
    accepted_path = Path(args.accepted_path).expanduser()
    if scratch_path.resolve() == accepted_path.resolve():
        print(
            "scratch --continue-path must differ from --accepted-path "
            "(viewer would mix failed annotations into the keep set).",
            file=sys.stderr,
        )
        return 2

    if args.workdir:
        workdir = Path(args.workdir).expanduser()
    else:
        workdir = accepted_path.parent / "go_semantic_verify_work"
    workdir.mkdir(parents=True, exist_ok=True)

    if args.state:
        state_path = Path(args.state).expanduser()
        if not state_path.is_absolute():
            cand = REPO_ROOT / state_path
            state_path = cand if not state_path.exists() else state_path
    else:
        state_path = workdir / "verify_state.json"

    copies = max(1, int(args.copies))
    client = PipelineClient(
        eif_url=args.eif_url,
        viewer_url=args.viewer_url,
        corpus_path=str(args.corpus_path or "") or None,
        annotate="llm-semantic",
        top_k=max(1, int(args.top_k)),
        max_scan=None,
        graphsignal_use_llm=None,
        dry_run=bool(args.dry_run),
        retrieve_mode="semantic",
        semantic_corpus_path=str(args.semantic_corpus_path or "").strip() or None,
    )
    client.ping()

    want_continue = str(scratch_path)
    try:
        health = _viewer_health(client.viewer_url)
    except Exception as exc:
        print(
            f"cannot read viewer /api/health at {client.viewer_url}: {exc}\n"
            f"start annotation-viewer with --continue-data {want_continue}",
            file=sys.stderr,
        )
        return 2
    actual = str(health.get("continue_path") or "").strip()
    if not _continue_paths_match(want_continue, actual):
        print(
            f"viewer continue_path mismatch.\n"
            f"  expected scratch: {want_continue}\n"
            f"  actual:           {actual or '(unset)'}\n"
            f"Restart annotation-viewer writing to the scratch file:\n"
            f"  cd tools/annotation-viewer && python -m server.main "
            f"--continue-data {want_continue}",
            file=sys.stderr,
        )
        return 2
    print(f"[scratch] viewer writes to {actual}", flush=True)
    print(f"[accepted] passing 20-copies append to {accepted_path}", flush=True)
    print(f"[workdir] {workdir}", flush=True)

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    if not accepted_path.is_file():
        accepted_path.write_text("", encoding="utf-8")

    rows = _load_raw_rows(input_path)
    print(
        f"[input] {input_path} rows={len(rows)} copies={copies} "
        f"top_k={args.top_k} max_verify={args.max_verify} "
        f"max_tests={args.max_tests or 'all'}",
        flush=True,
    )
    print(
        f"[train] family={args.adapter_family} steps={args.max_steps} "
        f"lr={args.lr} min_drop={args.min_drop} dry_run={args.dry_run}",
        flush=True,
    )

    if args.fresh:
        if state_path.is_file():
            print(f"[fresh] removing checkpoint {state_path}", flush=True)
            try:
                state_path.unlink()
            except OSError as exc:
                print(f"[fresh] could not delete state file: {exc}", file=sys.stderr)
                return 2
        state = VerifyState(copies=copies, max_tests=int(args.max_tests))
        _save_state(state_path, state)
    else:
        state = _load_state(
            state_path,
            copies=copies,
            max_tests=int(args.max_tests),
        )

    processed = (
        set(state.processed_test_lines)
        if args.skip_processed and not args.fresh
        else set()
    )
    end_line = int(args.end_line) or 10**12
    for test_line, row in rows:
        if test_line < int(args.start_line) or test_line > end_line:
            continue
        if state.done():
            break
        if test_line in processed:
            print(f"[skip] test line={test_line} already in state", flush=True)
            continue
        if args.mismatches_only and not _is_mismatch(row):
            print(
                f"[skip] test line={test_line} label==predict "
                f"({row.get('task_id') or ''})",
                flush=True,
            )
            continue
        _process_test(
            client,
            state,
            test_line=test_line,
            row=row,
            language=str(args.language or ""),
            scratch_path=scratch_path,
            accepted_path=accepted_path,
            workdir=workdir,
            max_verify=max(1, int(args.max_verify)),
            min_drop=float(args.min_drop),
            max_steps=max(1, int(args.max_steps)),
            learning_rate=float(args.lr),
            adapter_family=str(args.adapter_family),
            timeout=float(args.train_timeout),
            poll=float(args.poll),
            state_path=state_path,
        )

    _save_state(state_path, state)
    print(
        f"\n=== finished: ok_tests={state.n_ok_tests} "
        f"accepted_rows≈{state.n_continue_rows} "
        f"skipped_events={len(state.skipped)} "
        f"accepted={accepted_path} state={state_path} ===",
        flush=True,
    )
    return 0 if state.n_ok_tests > 0 or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
