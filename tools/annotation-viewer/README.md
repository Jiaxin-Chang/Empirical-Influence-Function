# Train Annotation Viewer

可视化 / 编辑 train 上的 `attention_edges`。源集只读；增删、LLM 标注 **接受后** upsert 到续训小集。

和报告页、续训共用仓库根目录 **`eif_api.env`**。完整闭环（检索 → 标注 → 续训）见仓库根 [README.md](../../README.md)。

```env
ANNOTATION_TRAIN_DATA=/path/to/source_with_edges.jsonl
ANNOTATION_CONTINUE_TRAIN_DATA=/path/to/continue_annotated_subset.jsonl
EIF_BASE_MODEL_PATH=/path/to/Qwen3-8B
```

- 启动时若续训小集文件不存在，会建空 jsonl
- Tokenizer 用 `EIF_BASE_MODEL_PATH`
- 报告页「续训(小集)」读的就是 `ANNOTATION_CONTINUE_TRAIN_DATA` 的**当前磁盘内容**

```bash
cd tools/annotation-viewer
python -m server.main   # API 8765，读 ../../eif_api.env
pnpm dev                # UI http://127.0.0.1:5275 ；占用则失败
```

Deep link：`http://127.0.0.1:5275/?sample=N&target=<idx>&source=<idx>`

大语料：`?corpusLine=L&corpusPath=...&queryExpr=...`  
从 Semantic Top-10 点进来还会带 `targetSem=`（test 四字段 HINT）。

## LLM 语义标注

对当前 FIM **一次 LLM 调用** 出整张 `src→dst` 边表（`subtype=semantic`）。有 `targetSem` 时，test 的 role/pattern/operations/relations 只作为 HINT，模型必须输出 token **整数下标**。

默认 **关 thinking**（与 GraphSignal 的 `ANNOTATE_ENABLE_THINKING` 分开）。要开：`ANNOTATE_SEMANTIC_ENABLE_THINKING=1`（截断且 0 边时会自动无 thinking 再打一次）。

流程：预览 → 接受写入小集 / 拒绝回退。接受前不要重启后端（预览会落盘，但旧进程里未落盘的会 404）。

## 人工日志 → 提示词版本

人手 add / delete / bump（不含 GraphSignal / LLM auto）追加到 `tools/annotation-viewer/human_annot_log.jsonl`（`ANNOTATION_HUMAN_LOG` 可改）。

```bash
cd tools/annotation-viewer
python -m server.eval_semantic_prompt --status
python -m server.eval_semantic_prompt --propose-only
python -m server.eval_semantic_prompt --activate-if-better
```

版本在 `prompts/semantic/versions/`；`active.json` 为当前默认。
