# Empirical Influence Function — 交互工具

这套工具用来对 **FIM 代码补全的错误例子** 做可视化错误归因分析，并据此做小集续训。

模型在 test 上补错时，往往不是整道题都错，而是某个机制（比较分支、默认值短路、错误提前返回等）没学对。工具把这条错误从「看不懂的生成」拆成可检查的链路：

1. **看错在哪**：打开 all-tokens 归因报告，对照 greedy 预测和 gold，点 token 看 saliency / 相关 train。
2. **语义分析当前错题**：用 LLM 抽出挖空位置的四字段语义（role / pattern / operations / relations），描述「这个洞在干什么、和前后文怎么连」。
3. **归因到 train**：按该语义在训练语料里检索机制相近的样本（也可 Boolean 检索），定位可能把模型带偏或本该教会它的那条 train。
4. **语义标注**：打开该 train，按 test 的机制 HINT 标 `src→dst` 注意力边（模型预测 completion token 时该看哪些 context token），把「根因假设」写成可续训的边表。
5. **验证与续训**：接受后写入续训小集，在报告页对当前 test 做 LoRA 续训，用 greedy 预测和 gold CE（teacher-force）看这条归因是否真的压低了错误。

两个界面分工：报告页做可视化归因、语义检索和续训评估；标注页把归因落到具体 train 的边上。配置统一在仓库根目录 `eif_api.env`（从 `eif_api.env.example` 复制）。

| 界面 | 端口 | 作用 |
|------|------|------|
| correlation-report | http://127.0.0.1:5273 | 可视化错误归因；Semantic / Boolean 检索；续训并看 gold CE |
| annotation-viewer | http://127.0.0.1:5275 | 打开被归因的 train；LLM 语义标注；接受后写入续训小集 |

## 配置要点

```bash
cp eif_api.env.example eif_api.env
```


| 变量                                     | 用途                         |
| -------------------------------------- | -------------------------- |
| `EIF_BASE_MODEL_PATH`                  | 基座权重（分词、embedding、续训）      |
| `EIF_ADAPTER_PATH_CE`                  | 当前 CE LoRA                 |
| `EIF_TEST_DATA`                        | 续训评估用的 test jsonl          |
| `ANNOTATION_CONTINUE_TRAIN_DATA`       | **续训小集**（标注接受写入这里；续训也读这里）  |
| `EIF_LLM_TRAIN_CORPUS`                 | 原始 FIM 语料（Boolean / 打开标注页） |
| `EIF_LLM_SEMANTIC_CORPUS`              | 四字段语义 jsonl（Semantic 检索）   |
| `EIF_LLM_SEMANTIC_EMBEDDINGS`          | Stage-1 embedding npz      |
| `DASHSCOPE_API_KEY` / `ANNOTATE_MODEL` | 远程 LLM（分析语义 + 标边）          |


Boolean 检索用原始 FIM；Semantic 检索用 preprocess 产出的 jsonl + npz，**不要**把 `EIF_LLM_TRAIN_CORPUS` 指到 `.semantic.jsonl`。

## 启动（四个终端）

先 `pnpm install` 一次：`tools/annotation-viewer` 和 `tools/correlation-report`。

```powershell
# A  annotation 后端  →  8765
cd tools\annotation-viewer
python -m server.main

# B  annotation 前端  →  5275
cd tools\annotation-viewer
pnpm dev

# C  报告 / 续训 API  →  8766（报告页默认连这个端口）
python -m src.ttav_bundle_api --host 0.0.0.0 --port 8766

# D  报告前端  →  5273
cd tools\correlation-report
pnpm dev
```

Linux / bash 同样四个命令；PowerShell 的 `cd tools\annotation-viewer` 改成 `cd tools/annotation-viewer`。

改了 `eif_api.env` 或 Python 标注代码后，**重启对应后端**。前端 Vite 一般会热更新。

## 怎么用

1. 打开 http://127.0.0.1:5273 ，载入一份 `*_all_tokens.json` 报告，选当前 **预测出错** 的 test。
2. **Semantic 测试**：对这条错题的 FIM 做语义分析（role / pattern / operations / relations），在 `EIF_LLM_SEMANTIC_CORPUS` 上召回机制相近的 Top-10 train（根因候选）。
3. 点一条命中：先做 MID 改写，再打开标注页（URL 带上 test 四字段，作为「这条错题在找什么机制」的 HINT）。
4. 在 http://127.0.0.1:5275 对归因到的 train 做 **LLM 语义标注**（整样本一次出 `src→dst` 边）。预览符合根因假设后再 **接受**，写入 `ANNOTATION_CONTINUE_TRAIN_DATA`。需要多份拷贝时用 duplicate。
5. 回到报告页点 **续训(小集)**，检验这条归因是否有效：看 greedy 预测是否靠近 gold，以及 **gold CE（teacher-force）** 是否下降。续训在点击那一刻 **重新读磁盘上的小集**（不是后端启动时冻住的那份）。

Semantic 检索也可以不经过报告页，直接把四字段 JSON 喂给：

```bash
python -m src.llm_semantic_retrieval --query-json query.json --top-k 10
```

（自动加载 `eif_api.env`，不再调用分析 LLM。）

## 语义语料（Semantic 检索前准备一次）

```bash
python -m src.fim_semantic_preprocess \
  -i "$EIF_LLM_TRAIN_CORPUS" \
  -o "$EIF_LLM_SEMANTIC_CORPUS" \
  --language go

python -m src.fim_semantic_preprocess --embed-only -o "$EIF_LLM_SEMANTIC_CORPUS"
```

第二步用本地 `EIF_BASE_MODEL_PATH` 做 last-token embedding，写出 npz。

