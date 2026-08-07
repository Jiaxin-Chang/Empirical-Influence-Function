# 用归因报告 + 预计算 embedding 看 Open Full Probe

放好两类文件，报告页上点 **Open Full Probe** 就能出图，**约 3 秒**，任意 token 对随点随算。
不加载模型、不需要 GPU、不需要 TTAV 服务器。

对应脚本：[`export_probe_bundle_from_pt.py`](export_probe_bundle_from_pt.py)

---

## 一、文件放哪

两个目录，都在仓库根下：

```
Empirical-Influence-Function/
├── correlation_matching_results/
│   └── correlation_matching_results_{sample_id}_all_tokens.json
└── token_embeddings/
    └── {sample_id}/              ← 目录名必须等于上面的 {sample_id}
        ├── test.pt
        ├── train_0.pt
        └── train_{归因报告里的 train id}.pt
```

实例（当前仓库里就是这样）：

```
correlation_matching_results/correlation_matching_results_ce_saliency_go_csims_44_all_tokens.json
token_embeddings/ce_saliency_go_csims_44/test.pt
token_embeddings/ce_saliency_go_csims_44/train_0.pt … train_4.pt
```

### 三条硬规则

**1. 目录名 = 报告文件名去掉前后缀**

```
correlation_matching_results_{ce_saliency_go_csims_44}_all_tokens.json
                             └──────────────────────┘
                                      目录名
```

这个 id 前后端已经在用（`infer_sample_id()` / `inferSampleIdFromMeta()`），不要另起名字。

**2. 报告文件名必须是纯 ASCII**

`[A-Za-z0-9_-]` 之外的字符（尤其中文）会被前端的 `isSafeSegment` 拒绝。原始文件名里的
`信令开发部` 就是这样被改成 `go_csims` 的。任务本身的中文名在报告内容里保留，不受影响。

**3. 一个目录 = 一个模型变体，里面所有 `.pt` 必须同一个 adapter**

ce_only 和 ce_saliency 各自一个目录，**train 的 embedding 不能共用**——那正是要对比的量。
实测：同 adapter 下相同 token 余弦 **0.9998**，跨 adapter **0.889**（最低 0.109）。混用会让
test 点和 train 点之间的距离掺进模型差异，脚本加载时会直接报错。

### 放不下就换根目录

```bash
export EIF_TOKEN_EMBEDDING_ROOT=/mnt/big-disk/token_embeddings
```

## 二、`.pt` 里要有什么

| 字段 | 类型 | 说明 |
|---|---|---|
| `hidden` | `[n_tokens, dim]` | 最后一层 token embedding |
| `input_ids` | `[n_tokens]` | |
| `labels` | `[n_tokens]` | `-100` = prompt，其余 = 答案区 |
| `token_surfaces` | `list[str]` | 已解码的 token，如 `' are'`（不是 `Ġare`） |
| `layer` | `int` | 取的第几层 |
| `model_name_or_path` | `str` | 基座模型 |
| `adapter_path` | `str` | LoRA 检查点，用于一致性校验 |

## 三、怎么用

1. 启动两个服务（都在这台机器上）：

```bash
# EIF bundle API
python -m src.ttav_bundle_api --host 0.0.0.0 --port 8766

# 报告前端
cd tools/correlation-report && pnpm install && pnpm dev
```

2. 打开 `http://localhost:5273`，选你的报告
3. **"TTAV Jump" 那行选「本页显示」**
4. 点一个 output token → 选一条 source→target 边 → 某个 TRAIN 分组里点 **Open Full Probe**
5. 约 3 秒后画布出现在页面左下角

### 「本页显示」和「新窗口」的区别

| | 本页显示 | 新窗口 |
|---|---|---|
| 画在哪 | 报告页内的浮动画布 | TTAV 网页版 |
| 需要 TTAV 服务器 | **否** | 是 |
| 耗时 | **~3 秒** | ~11 秒（要传 45MB 给 TTAV） |
| 有的功能 | 散点、连线、hover 定位到代码 | 全部（邻居线、refine、时间轴） |

两种模式都保留，随时切换。

## 四、几个用起来的细节

**勾选 pair 再点 Open Full Probe**，图上就只高亮那一对的 4 个端点。不勾选默认全部高亮——
一组有 100+ 对时会亮掉几十个点，很难看出重点。

**展开某个 pair 卡片**，它下面的完整训练样本会自动滚动到这对 pair 的位置，两个 token 用紫色
环标出。上方分组级的完整样本也会同步定位。

**鼠标划过画布上任意一点**，上方 Model Output 或右侧 TRAIN 文本里对应的 token 会高亮并滚动
到可见；反过来划过代码 token 也会点亮它在图上的点。

**画布可拖拽调整大小**（右上角手柄），双击手柄恢复默认。「收起」保留已加载的数据。

**标签太密**：放大即可，屏幕外的点不参与标签布局，放大后附近的点会显示更多标签。

## 五、常见报错

**`lies beyond the aligned test region (first N tokens)`**

报告和 `.pt` 记录的是**同一个 prompt 的两次不同生成**——prompt 完全一致，输出走到某处就分叉了。
分叉之后的位置，`.pt` 里存的是另一串 token，拿报告的下标去索引会取到错误的向量，所以直接拒绝。

**解决**：重导 `.pt`，要与归因报告出自**同一次生成**。最稳妥的做法是在跑归因的那个进程里顺手把
hidden state 存下来；如果必须分开跑，固定 seed 并用贪心解码（`do_sample=False`），否则采样
不确定就必然对不上。

**在补齐之前**：报告里下标小于分叉点的 target 仍然可用，脚本会告诉你边界在哪。

**`adapter mismatch`**

test 和 train 的 `.pt` 来自不同的 LoRA 检查点，见上面第三条硬规则。

**`missing required field(s)`**

`.pt` 缺字段，对照第二节的表。

**点标签前面多一个数字（如 `3588.BP1059`）**

浏览器缓存了旧前端，`Ctrl+Shift+R` 强制刷新。

## 六、也可以命令行单独生成

不经过前端，直接算一个 probe：

```bash
python -m src.export_probe_bundle_from_pt \
    --report   correlation_matching_results/correlation_matching_results_{sample_id}_all_tokens.json \
    --test-pt  token_embeddings/{sample_id}/test.pt \
    --train-pt token_embeddings/{sample_id}/train_0.pt \
    --train-id 0 --test-target 2752 --test-source 2747 \
    --for-frontend
```

`--for-frontend` 把结果写成报告页能直接读的缓存（1.3 MB 的 `projection.json`）。

看报告里有哪些可用组合：

```bash
python -m src.export_probe_bundle_from_pt --report <报告> --list-targets \
    --test-pt x --train-pt x --train-id 0 --test-target 0
```

其他常用参数：

```bash
--projection umap          # 换降维方法，默认 pca
--truncate-to-aligned      # 报告与 .pt 生成分叉时，只保留对齐的前缀
--with-embeddings          # 额外写完整 bundle（~190x 大），只有要用 TTAV 新窗口才需要
--no-upload --out-dir /tmp # 只生成不注册，先看结构
```

---

## 附：只有两段代码、没有归因报告？

那走另一条链路 [`RUN_PAIR_BUNDLE.md`](RUN_PAIR_BUNDLE.md)——只要两个 `.pt` 就能出散点图。

但**画不出 Open Full Probe**：probe 的连线需要 `cos_sim`，那是梯度空间的量，只有归因报告有，
从 embedding 推不出来。详见 `ONBOARDING_visualization.md` 第 4 节「三个容易混淆的相似度」。
