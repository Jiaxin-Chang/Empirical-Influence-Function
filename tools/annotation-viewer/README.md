# Train Annotation Viewer

独立工具：可视化 / 编辑 train JSONL 里的 `attention_edges` 标注（不复用 correlation-report）。

## 功能

1. 打开 `go_single_train_v2_graphsignal_10k_compact.json.bak`（或任意同格式 JSONL）
2. 选择一条训练样本，展示按 `input_ids` 解码的 token 序列
3. 点击 **target token**：
   - **蓝色背景**：ALTI saliency top-6（需启动时加 `--model`）
   - **彩色下划线**：指向该 target 的 annotation sources（按 subtype 着色）
4. 点击带下划线的 token 或边列表中的「删除」→ 从原 JSONL 删除该 edge
5. 「添加标注」模式：选 subtype → 点 source → 点 target → 写回文件

### 下划线颜色

| subtype | 含义 |
|---------|------|
| bracket | 括号/定界符配对 |
| defuse | 变量声明 → 使用点 |
| call | 被调函数 → 实参 |
| return | return → 返回表达式 |
| type | 类型标注 ↔ 变量名等 |
| dataflow | 值流（非同名绑定） |
| semantic | 语法配对关键词 |
| api | 库用法配对 |

编辑主要写入 `attention_edges`（BPE 索引，与 `input_ids` 对齐）；`annotations` 做 best-effort 同步。

## 启动（只看 / 改 annotation，不需要 saliency）

在 `tools/annotation-viewer/.env` 里配置训练数据路径（仓库已带示例）：

```env
ANNOTATION_TRAIN_DATA=../../smoke_train_data.jsonl
```

```bash
cd tools/annotation-viewer
# 终端 1：后端（读 .env 的 ANNOTATION_TRAIN_DATA；只用 tokenizer 解码，不加载 GPU）
python -m server.main

# 终端 2：前端
pnpm dev
```

也可显式传参覆盖 `.env`：

```bash
python -m server.main --data "D:\AAAworks\Empirical-Influence-Function\smoke_train_data.jsonl" --tokenizer "D:\AAAworks\Qwen3-8B"
```

### 从 correlation-report 跳转

报告页点击 pair 上的 `TRAIN #N` 会打开：

`http://127.0.0.1:5174/?sample=N&target=<train_target_idx>`

自动加载该样本并选中对应 target（只展示 annotation，不强制算 saliency）。

## Saliency（蓝底 top-6，可选）

两种方式：

### A. 本机有 GPU / 模型权重
```bash
python -m server.main --data ../../go_single_train_....jsonl \
  --model /path/to/Qwen2.5-Coder-7B-Instruct
```

### B. 在别的 GPU 服务器算完，拷到本机（推荐笔记本）
```bash
# 1) GPU 服务器预计算（例如前 20 条、answer 区所有 target）
python tools/annotation-viewer/scripts/precompute_saliency_cache.py \
  --data go_single_train_v2_graphsignal_10k_compact_resp_edges.jsonl \
  --model /path/to/Qwen2.5-Coder-7B-Instruct \
  --out saliency_cache \
  --indices 0-19 \
  --targets answer

# 2) 把整个 saliency_cache/ 目录拷到本机（可放在 annotation-viewer/ 下）

# 3) 本机启动：不要 --model，改传缓存目录
cd tools/annotation-viewer
python -m server.main \
  --data ../../go_single_train_v2_graphsignal_10k_compact_resp_edges.jsonl \
  --saliency-cache ./saliency_cache
```

缓存文件格式：`saliency_cache/<样本index>.json`，内含该样本各 target 的 top-k source。  
只对缓存里有的 `(样本, target)` 会显示蓝底；未预计算的 target 仍无蓝底。
