# 归因评测笔记

## 目前遇到的问题

在 feature attribution 的评测中，我们发现原始 ALTI top-k 并不总是等价于“真正支持当前 target token 的 source”：

- 有些 token 的 ALTI saliency 很高，但扰动后 target token 的 logprob 没有明显下降。
- 少数情况下，扰动 top-k source 后 logprob 反而上升，说明这些 source 可能不是支持 target，而是有反向或竞争作用。
- 在 full oracle / perturbation 分析中，也能看到一些对 target 有实际影响的 token 没有进入原始 ALTI top-k。

## 观察到的不准 case

已有 case 里，归因不准大致集中在几类 source：

- 语法或结构 token：例如 `:=`、括号前缀、控制结构附近的 token。这类 token 可能影响当前位置的代码结构，但不一定支持具体生成的 target token。
- 方法/字段访问片段：例如 `.Error`、`.Set`、`.Range`。它们和 target 位置有强信息流，但可能是在约束上下文，而不是直接提高当前 target token 的概率。
- BPE 残片：例如被切碎的标识符片段、后缀片段。这类单 token 看起来 saliency 高或低都可能不稳定，因为真正的语义单位是整个 identifier / function name / span。
- 反向 source：一些高 saliency token 被扰动后，target logprob 反而上升，说明 ALTI 捕捉到了信息流，但没有区分这股信息流是支持还是抑制当前 target token。

这些现象说明，原始 ALTI 更像是在回答“哪些 source 流向了 target 位置”，但不一定总是在回答“哪些 source 支持了当前这个 target token”。

## 基于 case 的改进思路

### 1. signed_clip ranking

引入 target-direction 判断。对 target token `y`，用 LM head 里对应的输出 embedding `W_y` 表示“提高 y 的 logit 的方向”。对 source contextual hidden state `h_i`，计算：

```text
direction(i, y) = cosine(h_i, W_y)
```

`signed_clip` 的排序分数是：

```text
score(i) = ALTI_saliency(i -> y) * max(0, direction(i, y))
```

直觉是：

- ALTI 负责衡量 source 到 target 位置的信息流强度。
- direction score 负责判断这个 source 是否支持具体 target token。
- 如果 direction 为负，就认为它可能是反向/竞争 source，裁成 0 后降权。

### 2. span-level source unit

代码里的很多语义单位会被 BPE 切碎。单 token 归因可能会把一个 identifier / method name / function name 的贡献拆散。

因此加入 span 模式：把相邻的词法 BPE 片段合成一个 source unit，排序和扰动都以 span 为单位。这样更接近代码语义单位，也能减少 BPE 残片造成的噪声。

### 3. 移除 saliency cutoff

之前考虑过 saliency threshold / cumulative-mass cutoff，但 raw ALTI 数值通常很小，额外阈值容易引入误解，也会让实验口径变复杂。

当前主实验不再做 saliency 筛选：保留所有非 trivial source unit，只比较不同排序方式和 source 粒度。

## 当前实验状态

这些改进还在实验中。当前建议在同一批样本上比较：

- `feature-ranking-mode=alti`, `feature-source-unit=token`
- `feature-ranking-mode=signed_clip`, `feature-source-unit=token`
- `feature-ranking-mode=alti`, `feature-source-unit=span`
- `feature-ranking-mode=signed_clip`, `feature-source-unit=span`

主要观察：

- Group@5 / Group@10 effectiveness 是否提升。
- positive rate 是否提升。
- reverse rate 是否下降。
- case 里高 saliency 但反向/无效的 source 是否被降权。
