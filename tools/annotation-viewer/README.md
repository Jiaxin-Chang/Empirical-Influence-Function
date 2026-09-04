# Train Annotation Viewer

独立工具：可视化 / 编辑 train JSONL 里的 `attention_edges` 标注。

配置统一放在仓库根目录 **`eif_api.env`**（与 `ttav_bundle_api` 共用），不再使用本目录 `.env`。

```env
ANNOTATION_TRAIN_DATA=/path/to/source_with_edges.jsonl
ANNOTATION_CONTINUE_TRAIN_DATA=/path/to/continue_annotated_subset.jsonl
EIF_BASE_MODEL_PATH=/path/to/Qwen3-8B
```

- 源集只读浏览；增删标注 upsert 到 `ANNOTATION_CONTINUE_TRAIN_DATA`
- 启动后端时若 `ANNOTATION_CONTINUE_TRAIN_DATA` 路径尚无文件，会自动新建空 `.jsonl`
- Tokenizer 用 `EIF_BASE_MODEL_PATH`（无需 `ANNOTATION_TOKENIZER`）
- 续训小集与 `python -m src.continue_train_eval` / 报告页「续训(小集)」共用同一路径

```bash
cd tools/annotation-viewer
python -m server.main   # 自动读 ../../eif_api.env
pnpm dev                # 固定 http://127.0.0.1:5275 ；占用则直接失败
```

Deep link：`http://127.0.0.1:5275/?sample=N&target=<idx>&source=<idx>`

大语料：`?corpusLine=L&corpusPath=...&queryExpr=<bool>&queryName=<family>`

## LLM 语义标注（整样本一次出边）

点击「LLM 语义标注」对当前 FIM 样本 **一次 LLM 调用** 返回整张 `src→dst` 边表（不再逐 token）。

## 人工标注日志 → 提示词版本

人手 add / delete / bump（不含 GraphSignal / LLM auto）会追加到 `tools/annotation-viewer/human_annot_log.jsonl`（可用 `ANNOTATION_HUMAN_LOG` 改路径）。每条事件存：

- 完整 `prompt` + `response`（在整段样本里看关系）
- 边的 **文本**（`src_text` → `dst_text` + 局部上下文）；下标只用于同一样本回放
- `query_expression` / `query_name`（这条样本是哪个检索 bool 召回的，供以后按族挂 few-shot）

攒了一批之后迭代提示词（hold-out 上比 precision/recall，更好才替换默认）：

```bash
cd tools/annotation-viewer
python -m server.eval_semantic_prompt --status
python -m server.eval_semantic_prompt --propose-only
python -m server.eval_semantic_prompt --activate-if-better
```

版本写在 `prompts/semantic/versions/`；`active.json` 指向当前默认。hold-out 样本太少时只存版本、不自动激活。
「按检索族挂 few-shot」是有数据之后的事：现在日志先记下 query，iterate 时会尽量每个检索族抽一条示例。
