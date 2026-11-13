# An eighth test, with a pre-specified analysis plan: Chong et al.'s report-contingent timing where successive durations are as weakly correlated as observers'

**Authors:** Jon-Paul Cacioli, Chris Marmo
**Project:** https://osf.io/kqdgb/
**Code:** https://github.com/synthiumjp/rivalry, folder `li2017_test8/`, at the commit that precedes the run

## 1. Purpose

The manuscript's first prediction for observers concerns an experiment like Chong, Tadin and Blake's (2005): a contrast
increment that follows the observer's reports, calibrated so that the attended grating's reported dominance lengthens
by about 29%. In the third test, with configurations whose successive durations were more correlated than observers',
this timing lengthened the competitor's reported dominance by 8.7%. In the seventh test, in configurations whose baseline
lag-1 correlation between an attended duration and the following competitor duration was below 0.1 (close to
observers' 0.06), the live gate's lengthening of the competitor disappeared under observer scoring. The manuscript
currently sizes the prediction by scaling, not by simulation. This test simulates it directly:

**At observer-like serial correlation, does an increment with Chong et al.'s report-contingent timing, calibrated to
a 29% lengthening of the attended grating's reported dominance, shorten or lengthen the competitor's reported
dominance, and by how much?**

## 2. Models and code

Li et al.'s (2017) model (the port `li2017.py`, with `li2017_t6.py` for input increments) and the two rate models of the
sixth test (`rate_models.py`), unchanged. The simulation, observers, scoring and measures are those of the third,
sixth and seventh tests (`run_li2017_test3.py`, `run_li2017_test6.py`, `run_li2017_test7.py`, unchanged). New code:
`run_li2017_test8.py` (the report-contingent schedule on the sixth test's schedule class, and the run) and
`analyse_li2017_test8.py` (this plan's analysis).

## 3. Sample

The seventh test's accepted configurations: 60 per model, in the order accepted, read from
`../li2017_test7/li2017_test7_results.json`. They are processed in the seventh test's chunks of 10, with its 32 seeds
of 120 s and its tag, so that the noise streams and the simulated observers' latencies are the seventh test's. The
baseline runs therefore reproduce the seventh test's baseline exactly, and every condition is paired, noise for noise,
with the seventh test's conditions. Because this sample's results under other conditions are known (the seventh test),
the test is pre-specified but not blind to the sample. Apart from the pre-run check (`precheck8.md`), which ran the
conditions on the first configuration of each model with 4 seeds and different noise, no result for the conditions
below has been simulated.

## 4. Conditions

Baseline (no increment), and Chong et al.'s timing at five sizes: 2.5%, 5%, 7.5%, 10% and 20% of input strength. The
increment's level rises linearly to 1 over 520 ms while the primary simulated observer (report latency 200 to 450 ms)
reports the attended grating, and falls linearly to 0 over 520 ms otherwise, as in the third test's calibrated condition
without a hold. The increment is the level times the size. The 20% size is there so that calibration does not depend on
a ceiling (Section 5); 7.5% narrows the bracket in which most crossings are expected. In Li et al.'s model the larger
sizes approach lock-in (one grating dominant for most of the run), so their uncalibrated values are reported but not
compared with the third test.

## 5. Analysis and decision rules

Changes are computed per configuration, as in the earlier tests: 100 × (mean over seeds with the increment − mean over
seeds without) / mean over seeds without, over seeds with a finite value in both runs. Primary scoring is observer
scoring (the primary observer's reported durations), as in Chong et al.'s experiment.

*Observer-valid configurations.* Observer scoring is meaningful only where the simulated observer reports most of the
model's episodes. A configuration is *observer-valid* if, at baseline, the mean number of reported competitor episodes
is at least 0.8 of the mean number of merged competitor episodes (mean over seeds). This is decided from the baseline
alone, which reproduces the seventh test's. The criterion was chosen knowing that in the seventh test the configurations
with mean durations below about 1 s, in which the observer reported only 53% to 60% of episodes and the report latency
plus the ramp is about as long as a dominance period, were those in which the live gate shortened the competitor under
observer scoring; the same regime was treated as invalid for observer scoring in the third test's control samples.
**The primary analysis (C2 to C4) uses the calibrated observer-valid configurations.** The same quantities for all
calibrated configurations are reported alongside.

*Calibration.* For each configuration, the attended grating's observer-scored change at 0 (by definition 0) and at each
size is interpolated linearly to find the size at which it first reaches 29%. Every other change (the competitor's
under each scoring, and the attended grating's on the model) is interpolated at that size with the same weights. A
configuration is not calibrated if its attended change does not reach 29% by 20%, or if a value needed is not finite;
both are counted.

- **C1 (manipulation check).** The 10% increment lengthens the attended grating's reported dominance: proportion of
  configurations with a positive change, lower Wilson 95% bound above 0.5. It is expected to hold (the seventh test's
  live gate lengthened the reported attended grating in every configuration) and is reported as a check, not as
  evidence.
- **C2 (the competitor is not shortened).** At the calibrated size, the median competitor change under observer
  scoring, with a percentile bootstrap 95% interval across configurations (10,000 resamples), has a lower bound above
  −2%. The margin is the smallest effect in the manuscript's planning table (Table 6).
- **C3 (the competitor is lengthened).** As C2, with a lower bound above 0.
- **C4 (reported).** Whether the interval's upper bound is below 0 (the competitor is shortened) and below −2% (shortened
  by more than the margin).

Reading of the outcomes: C2 and C3 holding, the competitor lengthens; C2 holding and C3 failing, no shortening beyond
the margin and no lengthening shown; C2 failing with an interval that includes −2%, inconclusive; C4 below −2%, the
competitor shortens. C1 is decided in a model with at least 10 configurations; C2 to C4 require at least 10 calibrated
observer-valid configurations, and otherwise are reported as not tested, with the number available.

## 6. What will be reported

- C1 to C4 per model; the number of observer-valid configurations; calibration coverage by group (calibrated, not
  reaching 29% by 20%, a value not finite) and the proportion reaching 29% at 20% (Wilson 95%); the median calibrated
  size; for uncalibrated configurations, the competitor change at 20%.
- At the calibrated size: the competitor's change under merged scoring, unmerged scoring, the absolute criterion and the
  passive observer (latency 300 to 600 ms), and the attended grating's change on the model (unmerged scoring), so that a
  reported 29% can be compared with the model's own lengthening. All for the observer-valid group and for all
  configurations.
- The competitor's and attended grating's changes at each size and scoring, uncalibrated.
- The difference between the 10% condition and the seventh test's live gate at 10%, configuration by configuration
  (same noise), under each scoring: what the report latency and the ramps cost.
- The competitor's predominance (fraction of reported time) and number of reported competitor episodes, and the
  fraction of the increment's level that falls in the competitor's dominance (difference criterion).
- Whether the baseline reproduces the seventh test's exactly.
- The comparison with the third test's 8.7% uses this test's calibrated value; the third test's value was itself at a
  calibrated size.

## 7. Checks before the run

- A dry run with every increment set to zero (`dry8.json`, `dry8_tests.json`; 4 configurations per model, 4 seeds,
  30 s, chunk 4): every trajectory-based measure of every condition reproduces its baseline exactly. (The level-based
  per-seed measures differ by construction, because the level ramps even at size 0.) Its chunking differs from the
  seventh test's, so `baseline_reproduces_test7` is false there, as expected.
- The first chunk of the seventh test's adaptation-model configurations, at the run's settings, reproduces the seventh
  test's baseline exactly on the machine used to write the code (`repro8_check.py`). The run itself checks the baseline
  of every chunk against the seventh test's and stops if any value differs.
- The analysis on synthetic data with known effects (`make_synth8.py`): a linear scenario in which the competitor
  lengthens and one in which it shortens, recovered exactly; and a scenario with a convex attended response, a
  competitor response not proportional to it, 30% observer-invalid configurations (90% in one model, so that C2 and C3
  are not tested there), about a third of configurations not reaching 29%, and some missing values, recovered to within
  0.2 points of the exact (continuous) solution, with coverage counted and "not tested" reported.
- A review of this plan and the code before the run, by a separate instance of an AI model (Claude), recorded in
  `precheck8.md` with the changes it led to.

## 8. Status of this plan

Not registered on OSF. The plan and code are committed to the repository before the run, and the test is reported as a
pre-specified follow-up. Any departure from this plan will be reported as such.
