# 归因评测笔记

## 目标

评测归因方法找到的 source 是否真的会影响当前预测。

## Feature Attribution

当前主要看 effectiveness 指标：

- 先用归因方法给 source unit 排序。
- 扰动方法找到的 top-k source unit。
- 如果扰动后 target token 的 logprob 下降超过阈值 tau，就认为这次归因是有效的。
- 汇总 Group@5 / Group@10 effectiveness、positive rate 和 reverse rate。

## 排序方案

baseline 是原始 ALTI saliency：

```text
score(source) = ALTI_saliency(source -> target)
```

signed ranking 会额外判断 source 是否支持当前 target token。对 target token y，用 LM head 中 y 对应的输出 embedding `W_y` 表示“提高 y 的 logit 的方向”。对每个 source 的 contextual hidden state `h_i`，计算：

```text
direction(i, y) = cosine(h_i, W_y)
```

`signed_clip` 的排序分数是：

```text
score(i) = ALTI_saliency(i -> y) * max(0, direction(i, y))
```

直觉解释：

- ALTI 表示 source 有多少信息流向 target 位置。
- direction score 估计这个 source 是否支持具体的 target token。
- 如果 direction 为负，说明它可能是反向/抑制作用；`signed_clip` 会把这部分裁成 0，让高 ALTI 但反向的 token 降权。

## Source Unit 方案

token 模式：按单个非 trivial BPE token 排序。

span 模式：把相邻的词法 BPE 片段合成一个 source unit，然后整体扰动这个 span。这个设计是为了减少标识符、函数名等语义单元被 BPE 切碎后导致的解释噪声。

## 已移除方案

saliency threshold / cumulative-mass cutoff 已经从主实验中移除。原因是 raw ALTI 数值通常很小，额外阈值会让实验口径变复杂，也容易误解。当前方案保留所有非 trivial source unit，只比较排序方式和 source 粒度。

## 建议对比

在同一批样本上比较：

- `feature-ranking-mode=alti`, `feature-source-unit=token`
- `feature-ranking-mode=signed_clip`, `feature-source-unit=token`
- `feature-ranking-mode=alti`, `feature-source-unit=span`
- `feature-ranking-mode=signed_clip`, `feature-source-unit=span`

重点看 Group@5 / Group@10 effectiveness、positive rate 和 reverse rate。
