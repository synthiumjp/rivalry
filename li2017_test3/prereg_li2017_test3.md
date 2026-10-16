# A third pre-registered test of the delivery-timing account in the attention model of binocular rivalry of Li, Rankin, Rinzel, Carrasco and Heeger (2017): how an increment is weighted across the competitor's episode, and what an observer would report

**Authors:** Jon-Paul Cacioli, Chris Marmo
**Project:** https://osf.io/kqdgb/ (original study, first and second tests in this model, secondary analysis of human data)
**Code:** https://github.com/synthiumjp/rivalry, folder `li2017_test3/`, at the commit recorded with this registration

## 1. Background and purpose

The second registered test in this model (osf.io/kqdgb) asked where in the competitor's episode
an increment on the attended grating acts:
- At matched expected time on, an increment late in the competitor's episode shortened it
  (−16.8%) and an early one did not (−0.5%).
- The point at which delayed withdrawal of the increment reversed its effect was closer to the
  prediction of the *handover* account than to that of a *dose* account in 166 of 186
  configurations.
- Two registered hypotheses failed. Chong, Tadin and Blake's (2005) manipulation, run with a
  doubled input, locked the attended grating into exclusive dominance (H11). Without adaptation,
  the zero-latency gate shortened the competitor (−7.7%) instead of leaving it unchanged (H12).
- The location result held under the primary (merged) scoring but not under the first test's
  scoring, and it reversed under the absolute criterion.
- Exploratory follow-ups suggested two explanations: a gate without latency captures brief
  attended excursions, and calibrated increments under Chong et al.'s timing lengthen the
  competitor.

An independent review of the paper raised five points, which this registration tests:
1. *A third account.* The second test separates weighting toward the end of the episode from
   uniform weighting, but not the handover account from a *hazard-multiplier* account. In that
   account the increment multiplies the instantaneous switching rate while it is on, and in a
   system with adaptation the rate rises through the episode.
2. *Scoring.* No observer-like scoring was applied to every condition.
3. *Controls.* The location test was not run without adaptation, the latency gate was not run
   with adaptation, and the no-adaptation sample also differs in noise.
4. *Onset step.* The early window combines presence early in the episode with a step at its
   onset.
5. *Human-relevant conditions.* The predictions for human observers rest on conditions that were
   not simulated at a calibrated increment with report timing.

**Three accounts.** Let *D* be a competitor episode's duration, with density *f*, distribution
function *F* and survival function *S*. Let *m*(*t*) = E[*D* − *t* | *D* > *t*] be the mean
residual duration. Let *e* = *s* − *u* be the effect of the increment's presence at the end of
the competitor's episode, where *s* is the gated change and *u* the continuous change. To first
order, the change produced by an increment present during the window [*a*, *b*) of the
competitor's episode is:
- *handover* (the increment shifts the switch at which it is present by a fixed amount):
  −*e*·[*F*(*b*) − *F*(*a*)];
- *hazard-multiplier* (the switching rate is multiplied by a constant while the increment is on):
  −*e*·∫<sub>*a*</sub><sup>*b*</sup> *f*(*t*)*m*(*t*) d*t* / E[*m*(*D*)];
- *dose* (each unit of time on counts equally): −*e*·∫<sub>*a*</sub><sup>*b*</sup> *S*(*t*) d*t* / E[*D*].

Each is normalised so that presence throughout the episode gives −*e*. All three coincide when the
switching rate is constant (exponential *D*). When the rate rises through the episode, the
hazard-multiplier account weights early windows more than the handover account does. Its relation
to the dose account depends on the distribution: in this model, short merged competitor episodes
can give it a larger early weight than the dose account.

## 2. Model

The port and its schedule interface are those of the second test (li2017.py, unchanged; SHA-256
recorded with this registration):
- published parameters;
- Ornstein-Uhlenbeck input noise (time constant 100 ms);
- forward Euler at 0.5 ms;
- binocular-summation responses recorded every 1 ms.

Schedules read the responses of the previous integration step.

## 3. Samples

**Main sample.** New configurations drawn with seed 20261020 (the first test used 20261007, the
second 20261016), in batches of 25, from the ranges of the first two tests:
- input strength 0.35 to 0.8;
- w<sub>o</sub> 0.5 to 0.75;
- w<sub>a</sub> 0.4 to 0.9;
- w<sub>h</sub> 1.5 to 2.5;
- τ<sub>h</sub> 1,000 to 4,000 ms;
- noise standard deviation 0.03 to 0.07.

A configuration is accepted at baseline if it produces at least 10 dominance episodes per seed on
average and a median CV within [0.3, 0.7]. Sampling stops at 200.

**No-adaptation sample.** w<sub>h</sub> = 0; noise standard deviation 0.15 to 0.30; other ranges as
above; seed 20261021. Accepted if it produces at least 10 episodes per seed on average. Sampling
stops at 100.

**High-noise sample (noise control).** Adaptation as in the main sample (w<sub>h</sub> 1.5 to 2.5,
τ<sub>h</sub> 1,000 to 4,000 ms), with the noise of the no-adaptation sample (0.15 to 0.30); seed
20261022. Accepted if it produces at least 10 episodes per seed on average (no CV criterion).
Sampling stops at 50. Without adaptation the model alternates only with three to six times the
main sample's noise, so the no-adaptation sample differs from the main sample in both. This sample
separates the two.

Every condition is simulated for 8 seeds of 120 s, with the noise of its baseline (common random
numbers within each batch of 25 configurations). Each observer's report latencies are pre-drawn per
instance, seeded by the configuration, the seed and the observer, so they are identical across
conditions and do not depend on the other instances in the batch. With a zero increment, every
condition reproduces its baseline exactly.

## 4. Dominance, scorings, measures and conditions

**Dominance.** As in the first two tests:
- grating 1 (attended, incremented) is dominant while R<sub>b1</sub> − R<sub>b2</sub> > θ;
- grating 2 (competitor) is dominant while R<sub>b2</sub> − R<sub>b1</sub> > θ;
- θ = 5% of the mean baseline binocular response of both gratings, per seed;
- first 5 s discarded.

**Observer scoring (primary for every hypothesis in this registration).** A simulated observer
runs online in every condition, including baseline:
- A change of exclusive dominance is reported after a latency drawn uniformly from 200 to 450 ms,
  independently for each switch, if the new percept is still the one most recently dominant.
- Indeterminate spells do not cancel a pending report; a return to the reported percept does.
  Excursions shorter than the latency are therefore not reported, and mixed periods are attributed
  to the last reported percept.
- A reported duration is the time between successive reports. The time before the first report is
  dropped, and the first and last reported durations are dropped.
- In the conditions timed by report (3 and 10 to 12), these same reports drive the increment, as an
  observer's key presses would.

A second, passive observer with latencies of 300 to 600 ms runs alongside, as a sensitivity
analysis.

**Secondary scorings**, reported for every hypothesis, all as defined in the second test:
- merged scoring (primary in the second test);
- the first test's scoring;
- the absolute criterion (grating 2 dominant while R<sub>b2</sub> exceeds the mean across seeds of
  each seed's median baseline R<sub>b2</sub>).

**Durations and changes.** A configuration's duration is the mean across seeds of per-seed means,
taken over the seeds for which both the baseline and the condition give a finite value.
Change = 100 × (condition − baseline) / baseline. "Lengthens" means change > 0. *T* is the
configuration's mean baseline dominance duration (mean of the two gratings, first-test scoring);
windows and holds are in units of *T*. Configurations with non-finite *T* are excluded from every
hypothesis.

**In rivalry.** A configuration is in rivalry in a condition if it has at least 3 merged and at
least 3 observer-reported competitor episodes per seed on average.

**Distributions.** For each configuration, pooled merged competitor durations are summarised at
baseline and under each continuous condition, for all seeds and for seeds 0 to 3:
- quantiles at 1% steps;
- on 0 to 10 *T* in steps of 0.01 *T*:
  - the dose weight *g*(*d*) = E[min(*D*, *d*)]/E[*D*];
  - the hazard weight *z*(*d*) = E[*m*(*D*); *D* < *d*]/E[*m*(*D*)], with *m*(*D*<sub>*i*</sub>) the mean of
    *D*<sub>*j*</sub> − *D*<sub>*i*</sub> over the pooled durations *D*<sub>*j*</sub> > *D*<sub>*i*</sub>.

**Other measures.**
- The competitor's mean binocular response while it is dominant (difference criterion).
- The realised dose (mean increment level inside and outside grating-1 episodes).
- The realised withdrawal time: the area under the increment level from the start of each merged
  competitor episode until the level first reaches 0 or the episode ends.
- Fragments per merged competitor episode.
- Absolute-criterion episode counts.

**Conditions, main sample.** Merged competitor episodes for windows are tracked online, as in the
second test.

At 20% of input strength:
1. Gated: on while grating 1 is dominant.
2. Latency gate: on while grating 1 has been dominant for at least 100 ms without interruption.
3. Report gate: on while the primary observer's most recent report is grating 1 (step changes).
4. Ungated: on throughout.
5. Windows: on from *a* to *b* after the start of a merged competitor episode, never while
   grating 1 is dominant, for [*a*, *b*) = [0, 0.25), [0.25, 0.5), [0.5, 0.75) and [0.75, 1.0) *T*.
6. Late window: on from *d*<sub>L</sub> after the start of a merged competitor episode until it ends,
   never while grating 1 is dominant. *d*<sub>L</sub> is matched to the [0, 0.25 *T*) window in
   expected time on, as in the second test; if fewer than 5 baseline merged competitor episodes
   exist, *d*<sub>L</sub> is undefined and the configuration is excluded from hypotheses that use it.
7. Ramped early window: the level rises linearly from 0 to 1 over the first 0.125 *T* of a merged
   competitor episode and stays on until 0.3125 *T*, never while grating 1 is dominant. Its time on
   (0.25 *T*) matches the [0, 0.25 *T*) window.

At 10% of input strength (the calibrated increment):

8. Gated.
9. Ungated.
10. Chong et al.'s timing: on the report of grating 1, the increment rises linearly to full over
    520 ms; on the report of grating 2, it falls linearly to 0 over 520 ms, from whatever level it
    has reached. As condition 6 of the second test, but at 10% and driven by the primary observer.
11. As 10, with the fall delayed by a hold of 0.25, 0.5, 0.75, 1.0 or 1.5 *T* after the report of
    grating 2.
12. Report-contingent early and late windows, as in the second test (step changes): on for
    0.25 *T* from the report of grating 2, or from *d*<sub>L</sub> after it until the report of
    grating 1.

**Conditions, no-adaptation and high-noise samples:** baseline; at 20%, gated, latency gate, report
gate, ungated, the [0, 0.25 *T*) window and the late window; at 10%, Chong et al.'s timing.

## 5. Hypotheses and decision rules

Proportions use Wilson 95% intervals across configurations. Observer scoring is primary for every
hypothesis. Every hypothesis is also reported under the passive observer, merged scoring, the first
test's scoring and the absolute criterion.

**Primary hypotheses: H14 and H21** (tests that separate accounts) and **H19** (a confirmatory
replication on a new sample of an exploratory result of the second test, and the basis of a
prediction for human observers).

**Replications and manipulation checks.** H13, H19 and H20 were expected from the second test's
results (Section 6) and are labelled as replications or manipulation checks, not severe tests.

**H13 (replication of the second test's H7 under observer scoring).** The early-window
([0, 0.25 *T*)) change minus the late-window change is positive in more than half of
configurations (lower Wilson bound above 0.5). Configurations out of rivalry in either window, or
without *d*<sub>L</sub>, are excluded and counted.

**H14 (primary; handover against hazard-multiplier).**
- *Weights and predictions.* For each configuration, the weights *w* of each account (Section 1)
  for the five windows (the four fixed windows and the late window) come from the pooled baseline
  distribution, all seeds. The predicted changes are −*e*·*w*, with *e* from the gated and ungated
  20% conditions.
- *Fit.* The observed profile *o* is fitted by *β*·(−*e*·*w*), *β* ≥ 0 (least squares through the
  origin), separately for each account.
- *Comparison.* The handover account is closer if its residual sum of squares is smaller. Ties
  count against it.
- *Inclusion.* A configuration is included if:
  - it is in rivalry in all five windows and in the gated and ungated conditions;
  - *e* > 0;
  - its handover and hazard weight vectors are separated (cosine similarity below 0.98).

  Excluded and non-separated configurations are counted.
- *Why a scaled fit.* The scaled fit is primary because all three accounts are first order: the
  second test's late window shortened the competitor by about half the handover prediction
  (−16.8% against −33.6%), so a comparison of magnitudes (*β* = 1) would mainly test the size of
  *e*, which none of the accounts predicts. What distinguishes the accounts is the shape of the
  profile.
- *Second-order guard.* Second-order effects lengthen the episode when the increment is present
  early, and they favour whichever account predicts the least early shortening. The same
  comparison is therefore also made on the three late windows ([0.5, 0.75) and [0.75, 1.0) *T*,
  and the late window) alone.
- *Decision.* H14 holds if:
  - in the five-window comparison, the handover account is closer in more than half of included
    configurations (lower Wilson bound above 0.5); and
  - the three-late-window comparison does not contradict it (upper Wilson bound above 0.5).
- *Also reported:*
  - the same rule against the dose weights;
  - sensitivity analyses: *β* = 1 for five and three windows, merged scoring, the passive observer,
    and a seed split (weights and *e* from seeds 0 to 3, windows from seeds 4 to 7);
  - each window's median change and each account's median predicted change.
- *If the scaled and unscaled comparisons disagree*, we will report that the profile's shape
  favours one account and that no first-order account captures its magnitude.

**H15 (excursion capture).** In the main sample, the latency gate lengthens the competitor more
than the zero-latency gate: latency-gate change minus gated change is positive in more than half of
configurations (lower Wilson bound above 0.5). Configurations out of rivalry in either condition are
excluded and counted.

**H16 (adaptation, with a gate that ignores brief excursions).** In the no-adaptation sample, all of
the following hold for the latency gate:
- continuous delivery shortens the competitor (upper Wilson bound below 0.5);
- the latency gate lengthens the attended grating (observer scoring; lower Wilson bound above 0.5);
- the latency gate leaves the competitor nearly unchanged: |median change| below a quarter of the
  main sample's median latency-gate change, and smaller than in the main sample (one-sided
  Mann-Whitney *p* < .05).

Configurations out of rivalry in the condition concerned are excluded from each part. The same rule
applied to the report gate (H16b) is reported. The latency gate's 100 ms is shorter than the
observer's latency, so in the no-adaptation sample it still delivers during excursions the observer
does not report, which biases H16 toward failing. The report gate does not. The zero-latency gate is
reported for comparison.

**H17 (location requires a rising switching rate).**
- *Statistic.* Within each sample, the early-window change minus the late-window change is divided
  by *e* (gated minus ungated 20%). Configurations with *e* ≤ 0 are excluded.
- *Decision.* H17 holds if the no-adaptation sample's |median| of this ratio is below half the main
  sample's median, and the ratio is smaller in the no-adaptation sample (one-sided Mann-Whitney
  *p* < .05).
- *Testability.* H17 can be tested only if the main sample's median is positive; otherwise it is
  reported as not supported.
- *Also reported:* the three accounts' predicted ratios from each sample's own distributions, and
  the high-noise sample's ratio.

**H18 (presence, not the onset step).** Three parts, each reported separately; H18 holds if all
three hold:
- (a) The ramped early window's change minus the late-window change is positive in more than half
  of configurations (lower Wilson bound above 0.5). The two windows have the same expected time on.
- (b) Under the absolute criterion, the median change under the [0, 0.25 *T*) step window is
  negative, and the median change under the ramped window, at the same time on, is at least half as
  negative. That is what normalisation of the competitor's response by the increment predicts; a
  shortening caused by the onset step would largely disappear with the ramp.
- (c) The [0, 0.25 *T*) window lowers the competitor's mean response while it is dominant: upper
  Wilson bound of the proportion of configurations with an increase below 0.5.

**H19 (primary; Chong et al.'s timing at the calibrated increment).** Under condition 10, the
competitor lengthens (lower Wilson bound above 0.5). Configurations out of rivalry count as not
lengthening. Positive control, reported alongside: the attended grating lengthens (observer
scoring; lower Wilson bound above 0.5).

**H20 (holding reverses the effect; manipulation check).** With a hold of 1.5 *T*, the competitor
shortens: median change below 0, and upper Wilson bound of the proportion lengthening below 0.5.
Configurations out of rivalry are excluded.

**H21 (primary; where holding reverses the effect).** The hold at which the competitor's change
reverses is closer to the handover prediction than to the dose prediction.
- *Effective delay.* The increment is withdrawn, on average, *d*<sub>0</sub> + *h* after the competitor's
  episode begins, where *h* is the hold and *d*<sub>0</sub> = 585 ms (mean report latency, 325 ms,
  plus half the 520 ms ramp). This is an approximation. If the true *d*<sub>0</sub> is longer, it
  biases the test against the handover account; if shorter, toward it. A sensitivity analysis uses
  each configuration's realised withdrawal time instead (Section 4).
- *Seed split.* From seeds 0 to 3:
  - Δ<sub>0</sub> = the change under condition 10;
  - *e* = gated 10% minus ungated 10%;
  - the continuous-delivery (10%) distribution.

  From seeds 4 to 7: the hold curve (no hold, then the five holds).
- *Handover prediction.* First order: change(*d*) = *s*<sub>c</sub> − *e*·*F*<sub>u</sub>(*d*), with
  *s*<sub>c</sub> = Δ<sub>0</sub> + *e*·*F*<sub>u</sub>(*d*<sub>0</sub>). The predicted reversal
  *d*<sub>pred</sub> is where *F*<sub>u</sub>(*d*) = *F*<sub>u</sub>(*d*<sub>0</sub>) + Δ<sub>0</sub>/*e*.
- *Dose and hazard predictions.* *d*<sub>dose</sub> and *d*<sub>haz</sub> are obtained in the same
  way with *g*<sub>u</sub> and *z*<sub>u</sub>.
- *When predictions are defined.* Predictions are defined if Δ<sub>0</sub> > 0, *e* > 0 and the target
  value is reached by 10 *T*. All are effective delays in units of *T*.
- *Observed reversal.* Fit a non-increasing function to the hold curve (pool-adjacent-violators)
  and take its first zero crossing; *d*<sub>obs</sub> = that hold plus *d*<sub>0</sub>.
  - If the fitted curve is ≤ 0 with no hold, the configuration is excluded.
  - If the curve never reaches 0 by 1.5 *T*, the observation is right-censored.
  - Configurations with a non-finite point on the curve are excluded and counted separately.
- *Inclusion.* A configuration is included if:
  - it is in rivalry in every hold condition and in gated and ungated 10%;
  - |ln(*d*<sub>pred</sub>/*d*<sub>dose</sub>)| > ln 1.25.
- *Closeness and censoring.* The closeness and censoring rules are those of the second test's H10.
- *Decision.* H21 holds if the handover prediction is closer in more than half of included
  configurations (lower Wilson bound above 0.5).
- *What H21 can separate.* H21 separates end-weighted accounts from the dose account. In synthetic
  checks the handover and hazard reversal predictions were within 25% of each other in almost
  every configuration, so the comparison against *d*<sub>haz</sub> is descriptive. The question of
  handover against hazard therefore rests on H14.
- *Also reported:*
  - Spearman correlations of *d*<sub>obs</sub> with each prediction;
  - the proportion within 25% of each prediction;
  - sensitivity analyses: the baseline distribution, all seeds without the split, the raw crossing,
    merged scoring, the passive observer, and the realised withdrawal time.

**H22 (report-contingent windows at the calibrated increment).** The report-contingent early
window's change minus the late window's change is positive in more than half of configurations
(lower Wilson bound above 0.5).

**H23 (Chong et al.'s timing without adaptation).** In the no-adaptation sample, under condition
10, the competitor lengthens in fewer than half of configurations (upper Wilson bound below 0.5).
Configurations out of rivalry are excluded and counted.

**H24 (noise control).** In the high-noise sample, which has adaptation, both of the following
lengthen the competitor (lower Wilson bound above 0.5; configurations out of rivalry count as not
lengthening):
- (a) the latency gate;
- (b) Chong et al.'s timing at 10%.

If H16 and H23 hold and H24 fails, the effect of removing adaptation cannot be separated from that
of raising the noise, and we will say so.

**Reporting.**
- Every hypothesis is reported whatever its outcome, under all five scorings, with medians,
  counts, realised doses and exclusions.
- If H14 fails, we will report that the test does not support the handover account over a
  hazard-multiplier account in this model.
- If H19 or H21 fails, we will report that the corresponding prediction for human observers is
  not supported by the model at a calibrated increment, and withdraw or revise it.
- Analyses not listed here are labelled exploratory.
- The registered analysis is `analyse_li2017_test3.py`.

## 6. What was done before registration

**The first and second registered tests** in this model, their exploratory follow-ups, and their
results are known to us. They were run on different samples, and they bear on several hypotheses
here:

| Hypothesis | Related earlier result (second test unless stated) |
|---|---|
| H13 | H7: 198 of 200 under merged scoring, 101 of 200 under first-test scoring, 7 of 200 under the absolute criterion. Merge thresholds 20 to 200 ms: 24 or 25 of 25. |
| H14 | Computed by the pre-freeze reviewer on the second test's two windows. Median early-window predictions: hazard −8.8%, handover −2.4%, dose −10.2%, observed −0.5%. A scaled fit to the two windows favoured handover over hazard in 133 of 200 configurations; *β* = 1 favoured it in 17 of 200, because the handover account over-predicts the late window. |
| H15, H16 | Exploratory: without adaptation, a 100 ms latency gate gave +1.2% (73 of 100 lengthening) against −7.7% for the zero-latency gate. |
| H19 | Exploratory: under Chong et al.'s timing at 10%, the competitor changed +6.2% (merged), +4.5% (first-test scoring) and +1.4% (absolute criterion); the attended grating changed +25%. |
| H20 | Delayed withdrawal (step, 20%) reversed at a median of 0.72 *T*. |
| H22 | Report-contingent windows at 20%: early minus late positive in 126 of 200 (merged scoring). |
| H23 | Exploratory: Chong et al.'s timing at 20% without adaptation gave −18.4%. |
| H18(b) | Under the absolute criterion, the early window gave −36.5% and the late window −18.3%. |

Observer scoring, the windows other than [0, 0.25 *T*), the ramped window, the report gate, the
latency gate with adaptation, the high-noise sample and the holds at 10% have not been run on any
configuration with a non-zero increment.

**Other checks:**
- **The referee report and technical audit of the paper** that motivated this test.
- **An independent pre-freeze review** of a draft of this registration and its code. It found
  problems with the following, all incorporated:
  - H14's scaling and second-order confound;
  - the time on of the ramped window;
  - the counting of configurations out of rivalry in H23;
  - the scaling of H17;
  - truncation of the distribution summaries at 3 *T*;
  - the noise level of the synthetic check;
  - per-instance observer latencies;
  - paired seeds;
  - the noise confound (high-noise sample added).
- **A dry run** with every increment set to zero: 30 s runs, 4 configurations in each sample drawn
  with seeds 101, 102 and 103 (not part of any registered sample). Every condition reproduced its
  baseline exactly in every duration, count and response measure. The schedules' increment levels
  and the realised withdrawal times were recorded; with a zero increment they do not affect the
  trajectory. In the dry run, the high-noise sample's mean durations were short (0.3 to 0.6 s), so the
  observer's latency is a large fraction of a dominance episode there; this was known when H24 was
  registered.
- **Synthetic validation of the analysis** (`synth_test3.py`):
  - *Structure.* The second test's 200 main and 100 no-adaptation configurations (their baselines,
    *T*, *d*<sub>L</sub> and duration distributions).
  - *Injected effects.* Window profiles and hold curves as each account predicts, scaled by 0.6.
  - *Noise.* A per-configuration baseline error of 2.4% and a per-seed error of 10%.
  - *Results:*

    | Generating account | H14, five windows vs hazard | H14, late windows | H14 | H21 vs dose |
    |---|---|---|---|---|
    | Handover | 118 of 153 | 34 of 47 | holds | 69 of 92 (holds) |
    | Hazard | 40 of 153 | 15 of 47 | fails | 52 of 81 (holds) |
    | Dose | 21 of 153 | 10 of 47 | fails | 21 of 76 (fails) |
    | Handover, +4% on the two earliest windows | 144 of 153 | 29 of 47 | holds | 65 of 81 (holds) |
    | Hazard, +4% on the two earliest windows | 79 of 153 | 14 of 47 | fails | 63 of 81 (holds) |

  - *Other outcomes.* H13 failed under dose-generated data. The handover and hazard reversal
    predictions were separated in at most 6 configurations.

No condition of this registration with a non-zero increment has been run, and no configuration of
any registered sample has been simulated.

## 7. Code

| File | Purpose |
|---|---|
| `li2017.py` | the second test's port, unchanged |
| `README_original_Li2017.ME` | the original README, required by the licence (CC BY-SA 3.0) |
| `run_li2017_test.py` | the first test's script, used for its episode helpers |
| `run_li2017_test3.py` | this test's simulation |
| `analyse_li2017_test3.py` | the registered analysis |
| `synth_test3.py` | the synthetic check (reads the second test's results) |
| `dry3.json`, `dry3_tests.json`, `synth3_summary.json` | the dry run and the synthetic check |
| `code_sha256.txt` | checksums of the code files and this registration |

The registered run is `python run_li2017_test3.py` followed by `python analyse_li2017_test3.py`.
Library versions are recorded in the results file.
