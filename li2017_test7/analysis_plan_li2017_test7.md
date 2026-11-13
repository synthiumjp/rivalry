# A seventh test, with a pre-specified analysis plan: the gated lengthening where successive durations are as weakly correlated as observers'

**Authors:** Jon-Paul Cacioli, Chris Marmo
**Project:** https://osf.io/kqdgb/
**Code:** https://github.com/synthiumjp/rivalry, folder `li2017_test7/`, at the commit that precedes the run

## 1. Purpose

In the sixth test, the live gate's lengthening of the competitor grew with the baseline lag-1 correlation between an
attended duration and the following competitor duration (*r*<sub>ab</sub>). Observers' correlation is about 0.06 on
average (Cao, Pastukhov, Aleshin, Mattia, & Braun, 2021), whereas the sixth test's configurations had medians of 0.27
to 0.31, and too few fell below 0.2 to test the lengthening there. A review of the manuscript raised three
questions, which this test addresses:

1. **Does the live gate lengthen the competitor where successive durations are as weakly correlated as observers',
   and by how much?** Configurations are drawn from ranges with weaker adaptation and accepted only if their baseline
   *r*<sub>ab</sub>, estimated on the acceptance runs, is below 0.1 (G1, G2).
2. **Does the competitor's change shrink relative to the attended grating's where successive durations are weakly
   correlated?** In the sixth test the competitor's change was, roughly, the attended grating's change multiplied by
   *r*<sub>ab</sub> (an exploratory, post hoc analysis). If so, the ratio of the competitor's change to the attended
   grating's should be lower in these configurations than in the sixth test's (P1), and the ratio sets the size of
   the effect to expect in observers.
3. **Does a self-replay that is never on during the competitor's dominance, and that cannot stay on through the delay
   the increment causes, lose the lengthening?** Self-replay of the trial's own baseline schedule was on during part
   of the competitor's dominance (2.9% to 5.7% of its time on in the sixth test), where an increment shortens the
   competitor. A masked self-replay is on only when the baseline schedule says the attended grating was dominant and
   it is dominant now. It is never on during the competitor's dominance, and it is off once the baseline schedule's
   attended period has ended, whereas the live gate stays on for as long as the attended grating stays dominant. The
   masked replay can still prolong the attended grating's dominance beyond the baseline switch, through the state it
   leaves, but it is not on for that extension. The live gate minus the masked replay therefore measures the effect
   of the increment's presence after the unperturbed switch time, together with a difference in time on (M1); the
   attended grating's change and the fraction of its dominance during which each increment is on are reported with
   it.

## 2. Models and code

- `li2017.py`, `li2017_t6.py`, `rate_models.py`, `run_li2017_test.py`, `run_li2017_test3.py` and
  `run_li2017_test6.py` are unchanged copies from the sixth test's folder (acceptance helpers, dominance, observers,
  measures, serial correlation, distributions).
- `run_li2017_test7.py` (simulation) adds a schedule mode, `replay_masked`, and an acceptance rule with a serial
  correlation criterion; `analyse_li2017_test7.py` (this analysis). Checksums in `code_sha256.txt`.

## 3. Samples

- **Ranges.** Li et al.'s model: input 0.35 to 0.8, opponency 0.5 to 0.75, attention weight 0.4 to 0.9, adaptation
  weight *w*<sub>h</sub> 0.3 to 1.2 (sixth test 1.5 to 2.5), τ<sub>h</sub> 1 to 4 s, noise 0.04 to 0.10 (sixth test
  0.03 to 0.07). Adaptation model: *I* 0.4 to 1.0, β 0.6 to 1.4, *g* 0.05 to 0.35 (sixth test 0.1 to 0.8), τ<sub>a</sub>
  1 to 6 s, noise 0.06 to 0.20 (0.02 to 0.15). Depression model: *I* 0.4 to 1.0, β 0.9 to 1.4 (0.6 to 1.4), γ 0.2 to
  1.0 (0.5 to 4), τ<sub>d</sub> 1 to 6 s, noise 0.08 to 0.18 (0.02 to 0.15).
- **Acceptance.** At least 10 episodes per seed on average and a median coefficient of variation between 0.3 and 0.8
  (sixth test 0.3 to 0.7), a mean dominance duration of at least 0.5 s in the rate models (as in the sixth test), and a
  baseline *r*<sub>ab</sub> below 0.1 (8 seeds of 120 s; the first test's scoring; mean of the seeds' Fisher-*z*
  values). Rejections are recorded by reason, and candidates that pass after the target is reached are counted as
  surplus. Sixty configurations per model (seeds 7001001, li; 7001002, sfa;
  7001003, dep), up to 3,000 draws per model; if fewer are accepted, the run continues with those, and the
  hypotheses are tested on them.
- **How the ranges and rule were set.** From baseline simulations only (zero increment): 60 draws per model from
  wide ranges, 60 from the ranges above (80 for a second set of depression-model ranges), recording acceptance and
  *r*<sub>ab</sub>. In the rate models, observer-like serial correlation occurred mainly where noise dominates
  switching and the coefficient of variation exceeds 0.7; the upper bound was raised to 0.8 for this reason (the
  sampling check of Li et al.'s model used 0.2 to 0.8). Expected yields: about a quarter of draws in Li et al.'s
  model and the adaptation model, and a fifth in the depression model. No condition with an increment was run.
- Because *r*<sub>ab</sub> is selected on the acceptance runs (8 seeds, standard error about 0.1), which use
  independent noise, its value in the main run will regress upward; the main-run value is reported. The selection
  changes the population, not only *r*<sub>ab</sub>: it favours weaker adaptation, stronger noise and, in Li et al.'s
  model, long mean durations, and in the rate models the accepted configurations fall into a short-duration,
  high-variability group and a long-duration group. G1 is therefore also reported by mean duration (above and below
  each model's median) and among configurations whose acceptance coefficient of variation is at most 0.7.
- Each condition: 32 seeds of 120 s with common random numbers, first 5 s discarded, batches of 10 configurations.

## 4. Conditions

Increments of 10% of the configuration's input strength, on grating 1, in every model.

| Condition | Increment on |
|---|---|
| Live gate | while grating 1 is dominant |
| Continuous | throughout |
| Self-replay | whenever grating 1 was dominant at the same time step of this seed's own baseline run (same noise; open loop) |
| Masked self-replay | as self-replay, and only while grating 1 is dominant now |

Recorded: the third test's measures under every scoring; the pooled merged competitor durations' percentiles; the
time on, the fraction of the increment's time on that falls in the competitor's dominance (zero by construction for
the live gate and the masked replay), the fraction of grating 1's dominance during which the increment is on, and
the agreement of the increment's on-state with grating 1's current dominance; *r*<sub>ab</sub> at baseline (per seed) and under the live
gate.

## 5. Hypotheses and decision rules

Changes are percentage changes relative to the configuration's baseline. Unless stated, a hypothesis holds when the
lower Wilson 95% bound of the proportion of configurations showing the predicted direction exceeds 0.5, on merged
scoring; each is also reported under observer scoring, unmerged scoring and the absolute criterion.

**In each model (Li, sfa, dep), on the accepted configurations:**
- **G1.** The live gate lengthens the competitor (mean).
- **G2.** Continuous delivery shortens the competitor (mean).
- **M1.** The live gate lengthens the competitor more than the masked self-replay.

- **P1.** The ratio of the live gate's change in the competitor's mean duration (merged scoring) to its change in the
  attended grating's mean duration (unmerged scoring, the sixth test's attended measure), per configuration, is lower
  than in the sixth test's configurations of the same model (`li2017_test6/li2017_test6_results.json`): one-sided
  Mann-Whitney test, *p* < .05. Configurations whose attended change is below 5% are excluded from the ratio in both
  tests, because the ratio is unstable there.

A hypothesis is tested in a model only if at least 10 configurations were accepted; otherwise it is reported as not
testable.

*Predictions.* We expect G2 to hold in every model. We are uncertain about G1, because the sixth test's lengthening
grew with *r*<sub>ab</sub> and its lowest-correlation configurations showed smaller effects; this is the question,
and with per-configuration noise of a few percent a failure of G1 would not show that the effect is absent, so the
median change and its bootstrap interval are reported with it. We expect M1 and P1 to hold.

*Reported without decisions:* the competitor's median change, with a 95% bootstrap interval across configurations
(10,000 resamples, seed 7001001), under every condition and scoring; self-replay against the live gate, and the
proportion of configurations in which each replay lengthens the competitor; for every condition, the attended
grating's change, the ratio of the competitor's change to it, time on, the fraction of the increment's time on in
the competitor's dominance, the fraction of grating 1's dominance during which it is on, and agreement; the main-run
*r*<sub>ab</sub> (median, interquartile range, numbers below 0.1 and 0.2); G1 among configurations whose main-run
*r*<sub>ab</sub> (seeds 0 to 15) is below 0.1, with the change from seeds 16 to 31; G1 by mean duration and among
configurations with an acceptance coefficient of variation of at most 0.7; the change in the median of the pooled
competitor durations; within each model, the rank correlations of the competitor's change (seeds 16 to 31) with the
product of *r*<sub>ab</sub> and the attended change (seeds 0 to 15), with *r*<sub>ab</sub> alone and with the attended
change alone, which are weak tests here because *r*<sub>ab</sub> is near zero and noisily estimated; and the median
ratio of the competitor's change to the attended grating's under the live gate, with its bootstrap interval, and that
ratio multiplied by 29%, Chong, Tadin and Blake's (2005) attended lengthening, as the competitor's change it implies.

## 6. What will be reported

All hypotheses whatever their outcome, under every scoring; the numbers of draws accepted and rejected by reason;
configurations without measurable episodes in a condition. Analyses not specified here are exploratory.

## 7. Status of this plan

The plan was not registered on OSF. It and the code were committed to the repository before the run. A dry run with
every increment set to zero (seeds 7111 to 7113, 4 configurations per model, 4 seeds, 30 s; `dry7.json`) checked the
code path and confirmed that every condition then reproduces the baseline durations exactly, and the analysis was
checked on that output and on a synthetic copy with imposed effects (`synth7.json`, made by `make_synth7.py`), whose
medians and ratios it recovered. Before the commit, the plan and code were reviewed by a separate instance of an AI
model (Claude, Anthropic), not by a person, which also ran the code with a 10% increment on one configuration per model
(4 seeds of 30 s) to test its behaviour; its findings, and the changes they led to, are in `precheck7.md`. It found
that the masked replay prolongs the attended grating's dominance beyond the baseline switch (the plan's first draft
said it could not), that a test of the product of *r*<sub>ab</sub> and the attended change as a predictor would be
driven mainly by noise in *r*<sub>ab</sub> near zero, and that the analysis lacked guards for small samples. The
description of the masked replay was corrected and its reported measures extended, the product test was replaced by
P1 and moved to the reported measures, and guards, duration and variability breakdowns and bootstrap intervals were
added. No other condition with a nonzero increment was run before the run.
