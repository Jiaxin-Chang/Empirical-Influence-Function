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

```bash
cd tools/annotation-viewer
# 终端 1：后端（只用 tokenizer 解码 input_ids，不加载 GPU 模型）
python -m server.main \
  --data "D:\AAAworks\annotation\code-corr-annotation\data\annotated\smoke_train_data_oversample_mid_edges.jsonl" \
  --tokenizer "D:\AAAworks\Qwen3-8B"

# 终端 2：前端
npm run dev
```

默认 tokenizer 也是 `D:\AAAworks\Qwen3-8B`；路径存在时可省略 `--tokenizer`。  
**不要**加 `--model`，除非你要本机算 ALTI 蓝底（会占 GPU）。

打开页面后：**点左侧样本** → 右侧出现 token；再点 answer 区 token → 看彩色下划线 annotation。

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
