# Deviations and results: registration osf.io/3rthv

**Registered:** 4 October 2026, https://osf.io/3rthv/
**Analysis code and design note committed before outcome data were opened:** commit `8eaee4a`,
https://github.com/synthiumjp/rivalry (file `analysis_einhauser.py`, SHA-256 `ac52e5c4…`)
**Data:** Einhäuser, Sandrock & Schütz (2021), Zenodo record 4575552

## Deviations from the registration

1. **Design branch applied in every phase.** The registration's Design B assumed the post-task phase
   would have no per-trial difficult stimulus and supplied a majority rule for that case. In the data,
   difficulty is assigned trial by trial in every phase, so Design B's own definition (T is the hard
   stimulus on trials where exactly one is hard) was applied throughout and the majority rule was not
   needed. Recorded in the design note before any outcome was opened, and committed in `8eaee4a`.
2. **Two exploratory analyses added before outcomes were opened**: a drift correction using
   symmetric trials, and a task-phase analysis restricted to the period before the button press. Both
   were specified in the design note, committed in `8eaee4a`, and are reported as exploratory.
3. **H2 minimum sample.** The registration does not set a minimum number of observers for H2. The
   committed code skipped any test with fewer than five observers, so it did not compute H2, for which
   four observers met the registered inclusion rule. H2 was therefore computed separately on those
   four, exactly as registered, and is reported below. This threshold was in the committed code but
   not in the registration.
4. **Execution.** The first run was stopped by the computing environment after four of eight analyses.
   The analysis was rerun in full with the same committed code. It is deterministic, and the
   registered results were identical in both runs.

No hypothesis, test, threshold or exclusion rule was changed.

## Registered results

**Precondition for H1: not met.** The registration requires the difficult stimulus's dominance
durations to increase relative to the pre-task blocks in both later phases. They decreased: median
Δ<sub>T,task</sub> = −15.6% and Δ<sub>T,post</sub> = −16.1%. Under the registration H1 is therefore
reported as uninformative, not as a failure of the account.

For completeness, H1 as computed (18 observers): median Δ<sub>O,task</sub> − Δ<sub>O,post</sub> =
−0.1 percentage points [−12.8, +8.8], one-sided Wilcoxon *p* = .71. Secondary: median
Δ<sub>O,task</sub> = −15.3% [−28.5, −8.1]; median Δ<sub>O,post</sub> = −12.7% [−25.1, +7.8].

**H2: four observers met the registered inclusion rule** (Δ<sub>T,task</sub> > 5%). Spearman
ρ = −0.80, one-sided *p* = .10. With four observers the smallest attainable one-sided *p* is .042, so
this test has almost no power and the result is uninformative.

Holm correction was not applied, since neither primary test was interpretable.

## Exploratory results

| Analysis | Precondition | H1: Δ<sub>O,task</sub> − Δ<sub>O,post</sub> | H2 |
|---|---|---|---|
| Gain threshold 0.2 | not met | −4.8 pp [−14.2, +6.0], *p* = .88 | *n* = 6, ρ = −0.71, *p* = .055 |
| Gain threshold 0.5 | not met | +4.0 pp [−17.3, +10.0], *p* = .59 | *n* = 2, not computed |
| Minimum episode 200 ms | not met | +3.4 pp [−12.7, +7.1], *p* = .66 | *n* = 4, not computed |
| Minimum episode 500 ms | not met | +1.9 pp [−7.3, +7.1], *p* = .61 | *n* = 5, ρ = −0.50, *p* = .20 |
| H2 without the 5% floor | not met | as registered | *n* = 20, ρ = +0.06 [−0.41, +0.47], *p* = .60 |
| Drift-corrected by symmetric trials | met (+3.9%, +2.4%) | −9.8 pp [−13.4, +7.3], *p* = .89 | *n* = 10, ρ = +0.07, *p* = .57 |
| Task phase before the button press | not met | −6.8 pp [−29.1, −4.3], *p* = .999 | *n* = 2, not computed |

The H1 statistic is one-sided in the predicted direction, so *p* near 1 means the observed difference
lies in the opposite direction.

## Interpretation, under §7 of the registration

The registered test could not be conducted as intended. In this paradigm, performing the task shortened
dominance durations for both stimuli, by around 15% in the whole task phase and around 30% before the
response, and difficulty did not lengthen the difficult stimulus's dominance relative to the pre-task
baseline. The dataset therefore does not contain the effect the registered test presupposes.

None of the exploratory analyses supports the account. Where the precondition is met (drift-corrected),
H1's estimate lies in the opposite direction but is not significant. In the pre-response analysis, H1
lies significantly in the opposite direction, but in a phase where both stimuli's durations shorten
sharply, so it does not isolate a gated increment either.

A likely reason is that the paradigm differs from the conditions the account addresses: here the task
speeds rivalry as a whole, and difficulty does not act as an increment applied only while one stimulus
is visible. This explanation is offered after seeing the data and is not claimed as support.

The modelling results reported in the manuscript are unaffected. The account's predictions for human
observers remain untested by this analysis.
