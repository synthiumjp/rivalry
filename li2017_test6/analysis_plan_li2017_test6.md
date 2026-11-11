# A sixth test, with a pre-specified analysis plan: what the gated lengthening of the competitor is, what it depends on, and whether it generalises

**Authors:** Jon-Paul Cacioli, Chris Marmo
**Project:** https://osf.io/kqdgb/
**Code:** https://github.com/synthiumjp/rivalry, folder `li2017_test6/`, at the commit that precedes the run

## 1. Purpose

In five tests in Li et al.'s (2017) model, an increment delivered only while the attended grating is dominant
(the live gate) lengthened the competitor's mean dominance duration, and the same increment delivered throughout
shortened it. An independent referee raised questions that the earlier tests do not answer. This test addresses
them, with each question tied to one or more hypotheses below.

1. **Is the lengthening a shift of the whole distribution, or the loss of short episodes?** A gate that removes brief
   returns of the attended grating, or brief competitor episodes, can raise the mean without lengthening the typical
   episode. We record the full distribution of merged competitor durations (D1, D2).
2. **Does it depend on serial dependence?** If the lengthening works by carrying the attended grating's prolonged
   dominance into the competitor's episode, it should scale with how strongly one duration predicts the next at
   baseline, which is weak in observers. We measure the baseline lag-1 correlation and relate it to the lengthening
   (C1, C2).
3. **Does a bias in the model's own attention layer behave like the input increment?** Li et al.'s model has an
   attention layer whose output multiplies the monocular drives. We deliver a top-down bias toward grating 1 to that
   layer, gated and continuously (AT1 to AT3).
4. **Is the lengthening reproduced by an open-loop replay of the trial's own baseline schedule?** The live gate both
   depends on the state of the present trial and extends itself (by delaying the handover it stays on longer). A
   replay of the same seed's baseline dominance schedule, with the same noise, keeps the schedule's alignment with the
   trial's noise but does not respond to the increment's own effect, and so tests the gate's self-extension (R1). A
   replay from another seed of the same configuration has the same statistics; because a replayed increment can
   entrain the rivalry, its comparison with the self-replay is reported with the agreement between the replayed
   schedule and the run's own dominance, not decided.
5. **How does the lengthening change as the increment is withdrawn earlier before the switch?** The fifth test
   withdrew it just before the switch and found no difference. Here the gate is withdrawn, for the rest of each
   attended period, once grating 1's advantage falls back below one of three graded margins (W1, per unit of time
   on, because earlier withdrawal also reduces the time on).
6. **Does the sign reversal generalise to canonical rate models?** Two continuous-time firing-rate models of the
   Laing and Chow (2002) form analysed by Shpiro et al. (2007, 2009), one with spike-frequency adaptation and one with
   synaptic depression of the cross-inhibition (G1, G2 and D1, D2 in each).

## 2. Models and code

- `li2017.py`, `run_li2017_test.py` and `run_li2017_test3.py` are unchanged from the earlier tests (acceptance,
  dominance, observers, measures).
- `li2017_t6.py` contains a copy of the port's `simulate()` with one option added: `target='attention'` delivers the
  increment to the attention layer, whose drive for orientation 1 becomes (r<sub>1</sub> − r<sub>2</sub>) + *b*·on and
  for orientation 2 its negative. With the default `target='input'` it is identical to `li2017.simulate` (checked bit
  for bit).
- `rate_models.py`: τ d*u*<sub>i</sub>/d*t* = −*u*<sub>i</sub> + *f*(*I*<sub>i</sub> + *s*<sub>i</sub> − β*u*<sub>j</sub>
  − *g a*<sub>i</sub> + *n*<sub>i</sub>), τ<sub>a</sub> d*a*<sub>i</sub>/d*t* = −*a*<sub>i</sub> + *u*<sub>i</sub>
  (spike-frequency adaptation, "sfa"), and τ d*u*<sub>i</sub>/d*t* = −*u*<sub>i</sub> + *f*(*I*<sub>i</sub> +
  *s*<sub>i</sub> − β*d*<sub>j</sub>*u*<sub>j</sub> + *n*<sub>i</sub>), τ<sub>d</sub> d*d*<sub>i</sub>/d*t* = 1 −
  *d*<sub>i</sub> − γ*d*<sub>i</sub>*u*<sub>i</sub> (depression, "dep"); *f*(*x*) = 1/(1 + exp(−(*x* − 0.2)/0.1)),
  τ = 10 ms, Ornstein-Uhlenbeck input noise (τ<sub>n</sub> = 100 ms), forward Euler with *dt* = 0.5 ms.
- `run_li2017_test6.py` (simulation) and `analyse_li2017_test6.py` (this analysis). Checksums in `code_sha256.txt`.

## 3. Samples

- **Li et al.'s model:** 60 configurations drawn with seed 20261111 from the third test's ranges and accepted by its
  rule (at least 10 episodes per seed on average, median coefficient of variation 0.3 to 0.7, 8 seeds of 120 s;
  rejections are not recorded by reason).
- **Rate models:** 60 configurations each (seeds 20261112, sfa; 20261113, dep), drawn uniformly from: input *I*
  0.4 to 1.0; cross-inhibition β 0.6 to 1.4; adaptation strength *g* 0.1 to 0.8 and τ<sub>a</sub> 1 to 6 s (sfa);
  depression rate γ 0.5 to 4 and τ<sub>d</sub> 1 to 6 s (dep); noise sd 0.02 to 0.15. Accepted by the same rule,
  plus a mean dominance duration of at least 500 ms (the rate models' time unit is otherwise free, and shorter
  episodes approach the 50 ms scoring floor). These ranges and the 500 ms rule were set from baseline simulations
  only (yield and mean duration at zero increment, 40 draws per model); no condition with an increment was run.
- Each condition: 32 seeds of 120 s with common random numbers, first 5 s discarded, batches of 10 configurations.

## 4. Conditions

Input increments are 10% of the configuration's input strength, on grating 1 (channel 1).

| Condition | Increment on | Models |
|---|---|---|
| Live gate | while grating 1 is dominant (zero latency) | all |
| Continuous | throughout | all |
| Self-replay | whenever grating 1 was dominant at the same time step of this seed's own baseline run (same noise; open loop) | all |
| Other-seed replay | the same, from the baseline run of another seed of the same configuration | all |
| Withdrawn gate, *k* = 3, 10, 30 | as the live gate, but withdrawn for the rest of the attended period once grating 1's advantage, having exceeded *k* dominance thresholds, falls back below them (re-armed when grating 2 becomes dominant) | Li |
| Attention-layer bias, gated and continuous, *a* = 0.25 and 0.5 | bias *b* = *a* × the configuration's median \|r<sub>1</sub> − r<sub>2</sub>\| at baseline (outside the indeterminate band), on the attention layer, gated as the live gate or continuous | Li |

Recorded for every condition: the third test's measures under every scoring; the pooled merged competitor durations'
5th to 95th percentiles, mean, the fraction shorter than 0.25 *T* and the mean of those of at least 0.25 *T*
(*T*: the configuration's mean baseline dominance duration); the time on. At baseline and under the live gate: the
lag-1 correlations, within each seed, between an attended duration and the following competitor duration
(*r*<sub>ab</sub>) and the reverse, on the alternating episode sequence of the first test's scoring (difference
criterion, episodes under 50 ms dropped, consecutive same-sign episodes merged). For the live and withdrawn gates:
the median time from the increment's withdrawal to the switch. In every condition: the fraction of time at which the
increment's on-state agrees with grating 1's current dominance, and, in Li et al.'s model, the attention layer's
mean response for grating 1 while each grating is dominant. For the live gate, continuous delivery and the gated
attention-layer bias, also: the same distribution summaries for competitor episodes under unmerged scoring, and the
rate of grating-1 runs shorter than 50 ms (absorbed into competitor episodes by merged scoring). For the rate
models the absolute criterion uses the midpoint of the 10th and 90th percentiles of channel 2's activity, because that
activity is bimodal.

## 5. Hypotheses and decision rules

Changes are percentage changes relative to the configuration's baseline. Unless stated, a hypothesis holds when the
lower Wilson 95% bound of the proportion of configurations showing the predicted direction exceeds 0.5, on merged
scoring; mean-based hypotheses are also reported under observer scoring, unmerged scoring and the absolute criterion.

**In each model (Li, sfa, dep):**
- **G1.** The live gate lengthens the competitor (mean).
- **G2.** Continuous delivery shortens the competitor (mean).
- **D1.** The live gate lengthens the median of the pooled competitor durations.
- **D2.** The live gate lengthens the mean of the competitor episodes of at least 0.25 *T*.

**In Li et al.'s model:**
- **R1.** The live gate lengthens the competitor more than self-replay.
- **W1.** Per unit of time on, the gate withdrawn at *k* = 30 lengthens the competitor less than at *k* = 3.
- **AT1.** The gated attention-layer bias (*a* = 0.5) lengthens the competitor.
- **AT2.** The continuous attention-layer bias (*a* = 0.5) shortens the competitor.
- **AT3.** The gated attention-layer bias changes the competitor more (in the lengthening direction) than the
  continuous one.

**Across all 180 configurations:**
- **C1.** Serial dependence predicts the lengthening. For each configuration, *r*<sub>ab</sub> is estimated from
  seeds 0 to 15 (mean of the seeds' Fisher-*z* values, back-transformed) and the live gate's change in the
  competitor's mean from seeds 16 to 31, so that the two do not share noise. Both are converted to ranks within each
  model (average ranks divided by *n* + 1), and the correlation of the pooled within-model ranks must be positive: the
  lower bound of a 95% bootstrap interval (configurations resampled within each model, 10,000 resamples, seed
  20261111) exceeds 0. The correlation within each model is reported.

**In each model:**
- **C2.** Among configurations with baseline *r*<sub>ab</sub> (all seeds) below 0.2, the range reported in observers,
  the live gate lengthens the competitor. Tested only in a model with at least 20 such configurations; otherwise
  reported as not testable.

*Predictions.* We expect G1, G2, D1, D2, R1, W1 and C1 to hold in Li et al.'s model. We are uncertain about AT1 to
AT3, because the attention layer saturates and its bias acts mainly near
the switch; about G1 in the rate models, because an earlier generic sigmoid architecture showed a weak effect
(Section S12); and about C2, which is the case that matters for observers.

*Reported without decisions:* the comparison of self-replay with other-seed replay, with the agreement of each
replayed schedule with the run's own dominance and the time on during each grating's dominance; R1 per unit of time
on; W1 in total; R1, W1 and AT3 as proportions under the other scorings; D2 with a threshold of 0.5 *T*; the rate of
competitor episodes of at least 0.25 *T* per minute; D1 and D2 under unmerged scoring; the rate of grating-1 runs
shorter than 50 ms; the attention layer's response in each condition; replay medians under every scoring; the withdrawal times and time on of the withdrawn
gates; the attended grating's change in every condition; quantile-by-quantile changes, the change in the fraction of
short episodes and in the number of episodes; the correlation C1 within each model; the live gate's effect on
*r*<sub>ab</sub> (as a change from baseline).

## 6. What will be reported

All hypotheses whatever their outcome, under every scoring where defined; the numbers of draws accepted and rejected
by reason; configurations without measurable episodes in a condition. Analyses not specified here are exploratory.

## 7. Status of this plan

The plan was not registered on OSF. It and the code were committed to the repository before the run. Before that, an
independent check reviewed the plan and the code and ran the code with a nonzero increment on one configuration of Li
et al.'s model (4 seeds, 20 s) to test its behaviour. It found that replayed increments entrain the rivalry, that the
first version of the withdrawn gate flickered and was confounded with time on, that pooling three models in C1 and C2
would mix model differences with serial dependence, and that the absolute criterion was unreliable in the rate
models. The withdrawn gate was made hysteretic, W1 was restated per unit of time on, the comparison of the two
replays was moved to the reported measures, C1 was restated within models with split seeds and C2 per model, and
measures of agreement, attention response, unmerged distributions and brief intrusions were added. A dry run with
every increment set to zero (seed 111, 4 configurations per model, 4 seeds, 30 s; `dry6.json`) checked the code
path and confirmed that every condition then reproduces the baseline exactly; the analysis was checked on that output
and on a synthetic copy with imposed effects. No condition with a nonzero increment was run before the run.
