# Attribution Evaluation Notes

## Goal

Evaluate whether the attribution method finds sources that are actually effective for the current prediction.

## Feature Attribution

Current primary metric is effectiveness:

- Rank source units by the feature attribution method.
- Perturb the method top-k source units.
- If the target token logprob drops by at least tau, count the attribution as effective.
- Report group effectiveness such as Group@5 / Group@10 and positive-rate variants.

## Ranking Variants

Baseline ranking is plain ALTI saliency:

```text
score(source) = ALTI_saliency(source -> target)
```

Signed ranking adds a target-direction check. For target token y, use the LM head output embedding W_y as the direction that increases y's logit. For each source contextual hidden state h_i, compute:

```text
direction(i, y) = cosine(h_i, W_y)
```

`signed_clip` ranks by:

```text
score(i) = ALTI_saliency(i -> y) * max(0, direction(i, y))
```

Interpretation:

- ALTI measures how much the source flows to the target position.
- The direction score estimates whether that source supports the specific target token.
- Negative-direction sources are clipped to zero, so high-flow but target-opposing tokens are demoted.

## Source Unit Variants

Token mode ranks individual non-trivial BPE tokens.

Span mode merges adjacent lexical BPE fragments into one source unit, then perturbs the whole span. This is meant to reduce cases where a meaningful identifier/function name is split across tokens.

## Removed Idea

The saliency threshold / cumulative-mass cutoff idea is removed from the main experiment. Raw ALTI values are often small, and this extra cutoff makes the evaluation harder to explain. The current plan keeps all non-trivial source units and only changes ranking or source granularity.

## What To Compare

Run the same sample set under:

- `feature-ranking-mode=alti`, `feature-source-unit=token`
- `feature-ranking-mode=signed_clip`, `feature-source-unit=token`
- `feature-ranking-mode=alti`, `feature-source-unit=span`
- `feature-ranking-mode=signed_clip`, `feature-source-unit=span`

Compare Group@5 / Group@10 effectiveness, positive rate, and reverse rate.
