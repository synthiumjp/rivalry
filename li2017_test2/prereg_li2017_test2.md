# A second pre-registered test of the delivery-timing account in the attention model of binocular rivalry of Li, Rankin, Rinzel, Carrasco and Heeger (2017): where in the competitor's episode an increment acts

**Authors:** Jon-Paul Cacioli, Chris Marmo
**Related registrations:** https://osf.io/d975z/ (original study); https://osf.io/d865z/ (first test in this model); https://osf.io/3rthv/ (secondary analysis of human data)
**Code:** https://github.com/synthiumjp/rivalry, folder `li2017_test2/`, at the commit recorded with this registration

## 1. Background and purpose

The account under test reduces rivalry dynamics with adaptation to one slow variable that
travels between two switching thresholds. An increment on one channel (the attended grating)
moves a threshold only at a switch at which it is present. Present at the switch that begins the
competitor's episode, it lengthens that episode; present at the switch that ends it, it shortens
it by more, because episodes end close to the slow variable's asymptote. In the first registered
test in Li et al.'s model (osf.io/d865z), dominance-gated delivery lengthened the competitor,
continuous delivery shortened it, and delaying the increment's withdrawal reversed the effect,
whereas delaying its onset did not.

An independent review pointed out that those results do not separate this *handover* account
from a *dose* account: that the competitor's episode shortens in proportion to how much of the
increment is present while the competitor is dominant, wherever in the episode that is. Delayed
withdrawal and delayed onset differ in that dose (duty 0.95 against 0.31 at one mean duration).
This registration tests predictions that separate the two accounts, asks whether the effect
survives Chong, Tadin and Blake's (2005) manipulation as run, and tests the claim that the
lengthening requires a slow memory (adaptation).

**Predictions of the two accounts.** Let *s* be the competitor change under gated delivery (the
increment is present at the switch that begins the competitor's episode), *u* the change under
continuous delivery, *e* = *s* − *u* the effect of presence at the switch that ends it, *D* a
competitor episode's duration, *F* its distribution function and *g*(*d*) = E[min(*D*, *d*)]/E[*D*]
the fraction of competitor time within *d* of the episode's start.

- *Early window* (present from the start of the competitor's episode for a time *w*, absent at
  its end unless the episode is shorter than *w*). Handover account, first order: −*e*·*F*(*w*);
  its second-order effects (more activity and so more adaptation of the suppressed attended
  channel; a lower asymptote of the slow variable) both lengthen the episode. Dose account:
  −*e*·*g*(*w*).
- *Late window* (present from *d*<sub>L</sub> after the start until the end, matched to the early
  window in expected time on). Handover: −*e*·(1 − *F*(*d*<sub>L</sub>)). Dose: −*e*·*g*(*w*), as for
  the early window.
- *Delayed withdrawal by d* (present from the attended channel's dominance until *d* after it ends).
  The increment is present at the end of a competitor episode only if the episode, with the
  increment on from its start, is shorter than *d*; the relevant distribution is therefore that of
  competitor durations under continuous delivery, *F*<sub>u</sub>. Handover: change
  = *s* − *e*·*F*<sub>u</sub>(*d*), reversing where *F*<sub>u</sub>(*d*) = *p*\* = *s*/(*s* − *u*). Dose:
  change = *s* − *e*·*g*<sub>u</sub>(*d*), reversing where *g*<sub>u</sub>(*d*) = *p*\*.

The early-versus-late comparison (H7) rules out only position-invariance: any account in which an
increment matters more near the switch predicts it. The early-window magnitude (H9) and the
reversal delay (H10) are specific to the handover account.

## 2. Model

The port of the first test (li2017.py; validated against the published behaviour, see
osf.io/d865z), with one change: a schedule may return a graded increment level in [0, 1] instead
of on/off, to allow a contrast ramp. For on/off schedules the arithmetic is unchanged
(`inc * on` instead of `np.where(on, inc, 0)`). Published parameters, Ornstein-Uhlenbeck input
noise (time constant 100 ms), forward Euler at 0.5 ms, binocular-summation responses recorded every
1 ms. Schedules read the responses of the previous integration step.

## 3. Samples

**Main sample.** New configurations drawn with seed 20261016 (the first test used 20261007), in
batches of 25, from the first test's ranges: input strength 0.35 to 0.8; w<sub>o</sub> 0.5 to 0.75;
w<sub>a</sub> 0.4 to 0.9; w<sub>h</sub> 1.5 to 2.5; τ<sub>h</sub> 1,000 to 4,000 ms; noise standard
deviation 0.03 to 0.07. Accepted at baseline, before any condition is run, if they produce at least
10 dominance episodes per seed on average and a median CV within [0.3, 0.7]. Sampling stops at 200.

**No-adaptation sample.** w<sub>h</sub> = 0 (no adaptation; τ<sub>h</sub> fixed at 2,000 ms, which is
then irrelevant), noise standard deviation 0.15 to 0.30, other ranges as above, seed 20261017.
Accepted if at least 10 episodes per seed on average (no CV criterion: noise-driven alternation
without adaptation has CV near 1). Sampling stops at 100.

Every condition is simulated for 8 seeds of 120 s, with the noise of its baseline (common random
numbers); report latencies use a separate random stream. With a zero increment every condition
reproduces its baseline exactly.

## 4. Dominance, measures and conditions

**Dominance** (as in the first test): grating 1 (attended, incremented) is dominant while
R<sub>b1</sub> − R<sub>b2</sub> > θ, grating 2 (competitor) while R<sub>b2</sub> − R<sub>b1</sub> > θ,
θ = 5% of the mean baseline binocular response of both gratings, per seed. Absolute criterion:
grating 2 is dominant while R<sub>b2</sub> exceeds the mean across seeds of each seed's median
baseline R<sub>b2</sub>. First 5 s discarded.

**Scoring.** *Merged scoring (primary in this registration):* competitor episodes are merged
across indeterminate spells and grating-1 excursions shorter than 50 ms; a merged competitor
episode runs from the first grating-2 run of at least 50 ms after a grating-1 run of at least 50
ms, to the next grating-1 run of at least 50 ms; the episode before the first grating-1 episode
and the unfinished last one are dropped. *Registered scoring of the first test* (secondary):
episodes shorter than 50 ms discarded, indeterminate spells split episodes, first and last
excluded. *Absolute criterion* (secondary), scored as in the first test. A configuration's
duration is the mean across seeds of per-seed means; change = 100 × (condition − baseline) /
baseline; "lengthens" means change > 0 (ties count as not lengthening). *T* is the configuration's
mean baseline dominance duration (mean of the two gratings, first-test scoring).

**In rivalry.** A configuration is in rivalry in a condition if it has at least 3 merged
competitor episodes per seed on average.

**Distributions.** For each configuration, pooled merged competitor durations are summarised
(1% quantiles; *g* at 0 to 3 *T* in steps of 0.01 *T*) at baseline and under continuous delivery,
for all seeds and for seeds 0 to 3.

**Conditions** (main sample), all at 20% of input strength unless stated:

1. Gated: on while grating 1 is dominant.
2. Ungated: on throughout.
3. Early window: on from the start of a merged competitor episode, tracked online (it starts at
   the first grating-2 sample after a grating-1 run of at least 50 ms, and ends when a grating-1
   run reaches 50 ms), for 0.25 *T*, and never while grating 1 is dominant.
4. Late window: on from *d*<sub>L</sub> after the start of a merged competitor episode until it
   ends, never while grating 1 is dominant. *d*<sub>L</sub> is set per configuration so that
   E[max(*D* − *d*<sub>L</sub>, 0)] = E[min(*D*, 0.25 *T*)], *D* = pooled baseline merged competitor
   durations (all seeds); if fewer than 5 such episodes exist, *d*<sub>L</sub> is undefined and the
   configuration is excluded from H7 to H9.
5. Delayed withdrawal at *d* = 0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.2, 1.6 and 2.0 *T*: on when grating
   1 becomes dominant, off *d* after it stops being dominant (still on if grating 1 regains
   dominance first).
6. Chong et al.'s manipulation as run (Experiment 3): increment 100% (contrast doubled). When grating
   1 is *reported* dominant, the increment rises linearly to 100% over 520 ms; when grating 2 is
   reported, it falls linearly to 0 over 520 ms (from whatever level it has reached; a new report of
   grating 1 restarts the rise from that level). Report: a change of exclusive dominance is reported
   after a latency drawn uniformly from 200 to 450 ms, independently for each switch, if the new
   percept is still the one most recently dominant (indeterminate spells do not cancel a pending
   report; a return to the reported percept does). Inputs up to 0.8 double to 1.6; a sensitivity
   analysis is restricted to configurations with input ≤ 0.5.
7. As 6, with the fall delayed by a further 0.5 *T* and 1.0 *T* after the report of grating 2
   (descriptive).
8. Report-contingent early and late windows: as 3 and 4 but timed from the report of grating 2
   (0.25 *T* from the report; or from *d*<sub>L</sub> after it until the report of grating 1), step
   changes, 20% (descriptive; these run on into grating 1's dominance by the report latency).

No-adaptation sample: baseline, gated 20%, ungated 20%.

For every condition we record the mean increment level during and outside grating-1 episodes
(realised dose), fragments per merged competitor episode, and the number of configurations out
of rivalry.

## 5. Hypotheses and decision rules

Proportions use Wilson 95% intervals across configurations. Merged scoring is primary for every
hypothesis; results under the first test's scoring and the absolute criterion are also reported.
**Primary hypotheses: H7 and H10.**

**H7 (primary).** At matched expected time on, the late window shortens the competitor more than
the early window: early-window change minus late-window change is positive in more than half of
configurations (lower Wilson bound above 0.5). Configurations out of rivalry in either window, or
without *d*<sub>L</sub>, are excluded and counted.

**H8.** The late window shortens the competitor: median change below 0 and upper Wilson bound of the
proportion lengthening below 0.5 (same exclusions as H7).

**H9 (specific to the handover account).** The early window shortens the competitor less than the
dose account predicts: early-window change minus −*e*·*g*(0.25 *T*) is positive in more than half
of configurations (lower Wilson bound above 0.5), with *s*, *u* from all seeds and *g* from the
pooled baseline distribution (all seeds). Configurations with *e* ≤ 0 are excluded. The handover
prediction −*e*·*F*(0.25 *T*) and the late-window predictions are reported per configuration.

**H10 (primary; specific to the handover account).** The reversal delay of delayed withdrawal is
closer to the handover prediction than to the dose prediction.
- Seed split: *s*, *u*, *F*<sub>u</sub> and *g*<sub>u</sub> from seeds 0 to 3; the delay curve
  (gated, then the nine delays) from seeds 4 to 7.
- *p*\* = *s*/(*s* − *u*), defined if *s* > 0 and *u* < 0. *d*<sub>pred</sub> = the *p*\*-quantile of
  *F*<sub>u</sub> (pooled merged competitor durations under continuous delivery, seeds 0 to 3);
  *d*<sub>dose</sub> = the smallest *d* with *g*<sub>u</sub>(*d*) = *p*\* (linear interpolation; undefined if
  *g*<sub>u</sub> never reaches *p*\* by 3 *T*). Both in units of *T*.
- Observed delay: fit a non-increasing function to the delay curve (pool-adjacent-violators, least
  squares), and take its first zero crossing by linear interpolation between delays. If the fitted
  curve is ≤ 0 at delay 0 the configuration is excluded; if it never reaches 0 by 2.0 *T*, the
  observation is right-censored.
- Included: configurations in rivalry in the gated, ungated and all delay conditions, with both
  predictions defined and |ln(*d*<sub>pred</sub>/*d*<sub>dose</sub>)| > ln 1.25 (predictions that
  differ by less than 25% are not informative).
- Handover is closer if |ln(*d*<sub>obs</sub>/*d*<sub>pred</sub>)| < |ln(*d*<sub>obs</sub>/*d*<sub>dose</sub>)|. For
  a censored observation, handover is closer if *d*<sub>pred</sub> > *d*<sub>dose</sub> and
  √(*d*<sub>pred</sub>·*d*<sub>dose</sub>) ≤ 2.0 *T*; if √(*d*<sub>pred</sub>·*d*<sub>dose</sub>) > 2.0 *T*
  the configuration is excluded as ambiguous.
- H10 holds if handover is closer in more than half of included configurations (lower Wilson bound
  above 0.5). All exclusions are counted and reported.
- Descriptive: the partial Spearman correlation of *d*<sub>obs</sub> with *d*<sub>pred</sub> controlling
  for *d*<sub>dose</sub>. Sensitivity analyses: baseline instead of continuous-delivery
  distributions; all seeds without the split; the raw crossing without the monotone fit.

**H11.** Under Chong et al.'s manipulation as run (condition 6), the competitor lengthens: lower
Wilson bound above 0.5, configurations out of rivalry counted as not lengthening.

**H12 (a slow memory is required).** In the no-adaptation sample: continuous delivery shortens the
competitor (upper Wilson bound below 0.5); gated delivery lengthens the attended grating (lower
Wilson bound above 0.5); and gated delivery leaves the competitor nearly unchanged: |median change|
below a quarter of the main sample's median gated change, and smaller than in the main sample
(one-sided Mann-Whitney *p* < .05). Configurations out of rivalry under gated delivery are excluded.

**Reporting.** Every hypothesis is reported whatever its outcome, under all three scorings, with
medians, counts, realised doses and exclusions. If H7 or H10 fails, we will report that the test
does not support the handover account over the dose account in this model. Analyses not listed
here are labelled exploratory. The registered analysis is `analyse_li2017_test2.py`.

## 6. What was done before registration

- **The first registered test** in this model and its exploratory robustness re-analysis
  (osf.io/d865z; repository `li2017_test/`), on a different sample. Its results are known to us and
  bear on this test: gated 20% +23.3%, continuous 20% −32.4% (so *p*\* ≈ 0.42); delayed withdrawal
  +11.4%, +10.6%, −24.9% and −32.4% at 0.25, 0.5, 1.0 and 1.5 *T*; delayed onset positive at every
  delay; the zero-latency gate amplifies brief attended excursions, and merged scoring reduces the
  gated effect at 10% from +16.3% to +8.9%. The delay grid was chosen with these results in view.
- **The Chong-paradigm ramp sweeps** of the original study in the authors' own model (GC-LCA), an
  earlier design note (`chong_paradigm_design.md`), which described the return to baseline as
  immediate. Chong et al. (2005) state that the contrast was "smoothly reduced ... over 520 ms";
  condition 6 follows the paper.
- **Feasibility of alternation without adaptation:** a fixed grid of 20 configurations
  (w<sub>h</sub> = 0; noise 0.05, 0.1, 0.15, 0.2, 0.3; w<sub>a</sub> 0.6, 0.9; input 0.5, 0.8;
  w<sub>o</sub> 0.65), baseline only, one 60 s run each (simulation seed 5). There was no alternation
  at noise 0.05; sparse alternation (3 to 22 episodes) at 0.10; regular alternation from 0.15, with CV
  0.9 to 1.55.
- **The modified port** was checked against the first test using the first test's own schedule
  class: on its first batch of 25 configurations, for the gated 10% and delayed-withdrawal 1.0 *T*
  conditions, durations were reproduced exactly (duty compared to floating-point tolerance).
- **Dry runs** with every increment set to zero, 30 s runs, 4 main and 4 no-adaptation
  configurations drawn with seeds 99 and 100 (not part of either registered sample), of this code
  and of an earlier version: every condition reproduced its baseline exactly.
- **Synthetic validation of the analysis:** the analysis was run on the dry-run structure with
  effects injected as each account predicts. With handover-generated data H7, H9 and H10 held; with
  dose-generated data they did not. (The partial correlation held under both and is therefore
  descriptive only.)
- **An independent pre-freeze review** of this registration and code, whose must-fix points
  (merged-episode timing, the continuous-delivery distribution for H10, censoring and separation
  rules, the Chong ramp-down, rivalry loss) are incorporated.
- No condition of this registration with a non-zero increment has been run, and no configuration
  of either registered sample has been simulated.

## 7. Code

`li2017.py` (port, modified as in Section 2), `run_li2017_test.py` (first test; helpers),
`run_li2017_test2.py` (this test), `analyse_li2017_test2.py` (registered analysis),
`repro_check.py`, `feas_noadapt.py`, `synth_test.py` (checks above). The registered run is
`python run_li2017_test2.py` followed by `python analyse_li2017_test2.py`.
