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

## 启动

```bash
# 终端 1 — API（会索引 ~10k 行，首次约几秒）
cd tools/annotation-viewer
pip install -r server/requirements.txt
python -m server.main --data ../../go_single_train_v2_graphsignal_10k_compact.json.bak

# 可选：启用 saliency（需本机有 Qwen 权重）
python -m server.main --data ../../go_single_train_v2_graphsignal_10k_compact.json.bak \
  --model ../../../code-corr-annotation/models/Qwen2.5-Coder-7B-Instruct

# 终端 2 — 前端
cd tools/annotation-viewer
npm install
npm run dev    # http://127.0.0.1:5174
```

后端默认 `http://127.0.0.1:8765`，Vite 已把 `/api` 代理过去。
