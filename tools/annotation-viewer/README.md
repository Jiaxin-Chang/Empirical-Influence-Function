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
