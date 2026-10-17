# A fourth test, with a pre-specified analysis plan, in the attention model of binocular rivalry of Li, Rankin, Rinzel, Carrasco and Heeger (2017): does a linear-response kernel measured from intrinsic noise predict the effects of increments on the competing percept?

**Authors:** Jon-Paul Cacioli, Chris Marmo
**Project:** https://osf.io/kqdgb/ (original study; first, second and third tests in this model; secondary analysis of human data)
**Code:** https://github.com/synthiumjp/rivalry, folder `li2017_test4/`, at the commit that precedes the run

## 1. Background and purpose

Three registered tests in this model found that the timing of an increment to the attended grating's
input sets the sign of its effect on the competing grating's dominance durations. Delivered only while
the attended grating is dominant (*gated*), the increment lengthened the competitor; delivered
throughout (*continuous*), it shortened it. Within the competitor's episode, the increment's effect was
weighted toward the end. Two parts of the account failed, however:

- the registered first-order prediction of how the effect is distributed across four windows tiling the
  competitor's episode (third test, H14);
- in exploratory comparisons, the first-order accounts did not describe the window profile and the
  reversal curves at once.

Human experiments have measured the same kind of weighting with reverse correlation. Lankheet (2006)
let stimulus strength fluctuate randomly during rivalry and averaged the fluctuations around perceptual
switches. This test asks whether the corresponding measurement in the model, a *linear-response kernel*
estimated from the model's own input noise in baseline runs with no increment, predicts the effects of
increments, with nothing fitted to the increment conditions.

**Pilot (disclosed).** An exploratory pilot on the first 10 configurations of the third test's main
sample (code and outputs in `pilot/`) estimated the kernel from 192 seeds of baseline per configuration
and ran windows, gated and continuous delivery at 5%, 10% and 20% with 16 seeds. Medians under merged
scoring:

- continuous delivery: observed −8.7%, −16.1% and −29.3% at 5%, 10% and 20%; kernel prediction about
  −8.9%, −17.8% and −35.6%;
- the four windows and the late window at 5%: observed +0.1%, +0.3%, −2.0%, −5.7% and −7.0%; kernel
  −1.5%, +0.2%, −1.7%, −3.8% and −8.8%;
- the same at 20%: observed −2.3%, −7.1%, −20.1%, −23.8% and −15.7%; kernel −5.9%, +0.8%, −6.8%,
  −15.4% and −35.2%;
- gated delivery: observed +4.5%, +9.9% and +16.9% (9, 10 and 10 of 10 configurations lengthened);
  kernel prediction about −0.9%, −1.7% and −3.4%.

The pilot suggested three things, which this test specifies as hypotheses (K1 to K6) on a new sample: the kernel
predicts continuous delivery; the window profile approaches the kernel's as the increment shrinks, so
that the failure of the first-order profile test reflects the size of the 20% increment; and the gated
lengthening is first order in the increment's size but is not predicted by the kernel, because it
depends on the increment tracking the percept. The pilot's configurations are not reused.

## 2. Model and code

The port `li2017.py` is that of the second and third tests, unchanged (SHA-256 checksums in
`code_sha256.txt`): published parameters; Ornstein-Uhlenbeck input noise with a time constant of 100 ms, added
to each monocular input; an additive increment on grating 1's input (left eye, orientation 1), gated by
a schedule. The third test's script `run_li2017_test3.py` is used unchanged for configuration acceptance,
dominance, the simulated observers, the duration measures, the late-window delay and the early and late
windows. `sim_noise.py` reproduces `li2017.simulate` with no increment and records the noise on grating 1's
input, which is where the increment enters; the run script checks at start-up that its trajectories are
identical to the port's.

## 3. Sample

Sixty configurations are drawn with seed 20261023 from the third test's main ranges (input strength 0.35
to 0.8, opponency weight 0.5 to 0.75, attention weight 0.4 to 0.9, adaptation weight 1.5 to 2.5,
τ<sub>h</sub> 1 to 4 s, noise 0.03 to 0.07) and accepted by the third test's rule (at least ten episodes
per seed and a median coefficient of variation between 0.3 and 0.7, on 8 seeds of 120 s), in batches of 10 configurations. The sample is
smaller than the earlier tests' because each configuration needs 192 seeds for its kernel and 32 seeds
per condition.

## 4. Conditions

Each condition is simulated for 32 seeds of 120 s with the same noise as its baseline (common random
numbers), and the first 5 s are discarded. *T* is the configuration's mean baseline dominance duration as
in the third test. Increments are a fraction of the configuration's input strength:

| Condition | Increment | Schedule |
|---|---|---|
| Gated | 5%, 10%, 20% | on while grating 1 is dominant (zero latency), as in the earlier tests |
| Continuous | 5%, 10%, 20% | on throughout |
| Yoked | 5% | follows the realised gate of the 5% gated condition in the next seed of the same configuration (the last seed follows the first) |
| Four windows | 5%, 20% | on during [0, 0.25), [0.25, 0.5), [0.5, 0.75) and [0.75, 1) *T* after the start of each merged competitor episode, while it lasts |
| Late window | 5%, 20% | on from the third test's matched delay to the end of each competitor episode |

## 5. The kernel

For each configuration, 192 further seeds of 120 s are simulated without increment (separate noise),
with the noise on grating 1's input recorded at 1 ms. Dominance and merged competitor episodes are
defined as under merged scoring. *T*<sub>k</sub> is the mean of the mean merged competitor duration and
the mean attended interval preceding it, and bins are 0.125 *T*<sub>k</sub> wide. Two ordinary
least-squares regressions, each with an intercept:

- **A (aligned to the competitor episode's start).** The duration of each merged competitor episode on
  the mean noise in each bin over [0, 3 *T*<sub>k</sub>) after the episode's start, with the noise value
  just before the start as an additional covariate. The noise is autocorrelated (100 ms), so noise
  shortly after the start is correlated with the noise that brought the switch about; the covariate
  absorbs that dependence and is not used in predictions. Kernel episodes start at the first competitor
  run of at least 50 ms; the window schedules start at the first competitor sample, as in the third test.
- **B (aligned to the start of the preceding attended interval).** The same duration on the bin means of
  noise × 1[grating 1 dominant] and of noise × 1[not dominant], over [0, 4 *T*<sub>k</sub>) after the
  start of the attended interval that precedes the episode (the last qualifying attended run, of at
  least 50 ms, before the episode starts), with the noise value just before that start as a covariate; *f*(bin) is the mean fraction of each bin in
  which grating 1 is dominant.

Predictions, as a percentage of the kernel's mean merged competitor duration, for an increment of size
Δ (fraction × input strength):

- window [*a*, *b*) *T*: 100Δ·Σ β<sub>A</sub>(bin)·overlap(bin, [*aT*, *bT*)) / mean duration, with the
  overlap as a fraction of the bin; the late window uses [*d*<sub>L</sub>, 3 *T*<sub>k</sub>);
- gated: 100Δ·Σ β<sub>B,dominant</sub>(bin)·*f*(bin) / mean duration;
- continuous: 100Δ·Σ [β<sub>B,dominant</sub>(bin)·*f*(bin) + β<sub>B,not</sub>(bin)·(1 − *f*(bin))] / mean
  duration;
- yoked: *q* × the continuous prediction, where *q* is the configuration's fraction of time the 5% gate
  is on, because the yoked increment is on independently of the present trial.

The gated prediction uses when grating 1 is dominant on average, relative to the start of the attended
interval, but not the dependence between the gate and the noise of the present trial. A configuration
whose regressions have fewer than 650 episodes has no kernel and is excluded and counted. Window
predictions need *T* and the late window needs *d*<sub>L</sub>; the gated, continuous and yoked
predictions do not.

## 6. Hypotheses and decision rules

Changes are percentage changes in the mean competitor duration relative to the configuration's baseline,
as in the earlier tests, and "median" is the median across configurations. *R* is the ratio of the median
observed change to the median predicted change. Configurations with a missing value in a comparison are
excluded from it and counted.

- **K1 (primary). The kernel predicts continuous delivery.** *R* lies within [0.75, 1.25] at both 5% and
  10%. Also reported: the number of configurations within 25% of their prediction, and Spearman's
  correlation between predicted and observed changes.
- **K2 (primary). Small increments follow the kernel; large ones do not.** For the [0.5, 0.75) window and
  for the late window, |log *R*| is smaller at 5% than at 20%, and the lower 2.5% bound of a bootstrap
  of the difference (|log *R*<sub>20%</sub>| − |log *R*<sub>5%</sub>|; 2,000 resamples of configurations,
  seed 20261023; lower bound the empirical 2.5% quantile) is above zero. If the medians differ in sign,
  or either is zero, |log *R*| is infinite; a resample (or the sample) in which both are infinite counts
  against the hypothesis. Also reported: the
  observed and predicted medians for every window at both sizes.
- **K3 (primary). The gated lengthening is not predicted by the kernel.** At 5%, the lower Wilson 95%
  bound exceeds 0.5 both for the proportion of configurations in which gated delivery lengthens the
  competitor and for the proportion in which the kernel's gated prediction is less than half the
  observed gated change.
- **K4. Yoked replay behaves as the kernel predicts.** At 5%, the lower Wilson 95% bound exceeds 0.5
  both for the proportion of configurations in which the yoked change is closer to the kernel's yoked
  prediction than to the live gated change, and for the proportion in which the live gated change
  exceeds the yoked change.
- **K5. The gated effect is first order.** Over configurations with both values, the median gated change
  per unit increment at 5% divided by that at 20% lies within [0.67, 1.5]. The 10% value is reported.
- **K6. The kernel is weighted late.** The median, across configurations, of the share of the negative
  (shortening) weight, Σ min(β<sub>A</sub>, 0), that lies in [0.5, 1.5) *T*<sub>k</sub> exceeds 0.5.

## 7. Scorings

Decisions use merged scoring (the second test's primary), on which the kernel is built. Every hypothesis
is also reported under observer scoring (the third test's primary), unmerged scoring (the first test's)
and the absolute criterion, without decisions. The observer's latency makes it unreliable for short
episodes, as the third test showed.

## 8. What will be reported

All hypotheses, whatever their outcome; the kernel profile (median across configurations); per-condition
medians under every scoring; the number of configurations drawn and accepted; any configuration whose
kernel could not be estimated (fewer than 50 episodes). Analyses not specified here will be labelled
exploratory.

## 9. Status of this plan

The code (`run_li2017_test4.py`, `analyse_li2017_test4.py`, `sim_noise.py`) was fixed before any condition
of this sample was run. The run is `python run_li2017_test4.py --workers 2`; the batch size
of 10 configurations is part of the design (it sets the noise streams), and the number of workers does
not affect the results. This plan was not registered on OSF; it and the code were committed to the repository before
the run, and the test is reported as a pre-specified follow-up rather than a registered test. An
independent review of the code and this text before the run led to
the start-noise covariate, the yoked prediction, criterion (b) of H27, the bounds of H29, the
definition in H30 and the minimum number of episodes. A dry run with every increment set to zero, a different seed (104), 4
configurations, 4 seeds per condition, 16 kernel seeds and 30 s runs (`dry4.json`) checked the code path, and the analysis was run on
the pilot's data rearranged into this test's format (`pilot/pilot_as_test4_tests.json`, with the yoked
condition, which the pilot did not run, replaced by the gated one). The pilot is described in Section 1.
No other simulation of this sample has been run.

## Reference

Lankheet, M. J. M. (2006). Unraveling adaptation and mutual inhibition in perceptual rivalry. *Journal of
Vision*, 6(4), 304–310. https://doi.org/10.1167/6.4.1
