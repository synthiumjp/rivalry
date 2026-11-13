# Pre-check of the eighth test (plan, `run_li2017_test8.py`, `analyse_li2017_test8.py`)

Reviewer: a separate instance of Claude (Anthropic). Scripts and outputs are in `precheck/`. No existing file was edited.
Files reviewed: plan 14:44, `run_li2017_test8.py` (md5 e840ab0f…), `analyse_li2017_test8.py` (md5 f9790c1b…),
`make_synth8.py` (md5 f483980a…). The folder changed while the review was running: `_repro8_tmp.py` became
`repro8_check.py` with `repro8_check_sfa.log` added, `README.md` was added, and the synthetic input JSONs were removed.
The three code files above did not change.

## Verdict

The simulation code is correct. `Schedule8`'s `'chong'` mode is bit-for-bit identical to the third test's
`Schedule3('chong')` with no hold. The noise and latency streams are the seventh test's, and the calibration function
does what the plan says.

The design has one substantive problem that should be settled before the run (M-2). About half of the rate-model
configurations (31 of 60 adaptation, 25 of 60 depression) have T < 1 s. In those configurations:

- the simulated observer reports only 53–60% of the model's episodes;
- the observer's report latency plus the 520 ms ramp is about as long as a dominance period;
- most will not reach a 29% attended change by 10%, so the calibration will drop them;
- they are exactly the configurations in which the seventh test's live gate shortened the competitor under observer
  scoring.

So, as specified, C2 and C3 in the rate models mostly measure the configurations with T ≥ 1 s, selected by the 10% size
ceiling. The direction of that selection can be predicted from the seventh test. There is also a small reporting bug
(M-1) and a plan sentence that my behaviour run made false (M-3). With M-1 to M-3 fixed, the run is good to go.

## Must fix before the run

**M-1. A C2 or C3 that is not testable is reported as `False` (fails), not as "not tested".**
`analyse_li2017_test8.py`, lines 148–149: `dec[name] = dict(..., C2=r['C2_not_shortened']['holds'], C3=r['C3_lengthened']['holds'], ...)`.
`holds` is `bool(testable and ...)`, so a model with fewer than 10 calibrated configurations prints `C2: false,
C3: false`. The plan (Section 5) says that C2 and C3 "are not tested in that model and the reason is reported".

I confirmed this with `precheck/edge8.py`: synthetic depression data with 0 calibrated configurations gives
`'dep': {'C1': True, 'C2': False, 'C3': False}`. Replace lines 148–149 with:
```python
    nt = 'not tested (%d calibrated configurations, fewer than %d)' % (r['C2_not_shortened']['n'], MIN_N)
    dec[name] = dict(C1=r['C1_attended_lengthens']['holds'],
                     C2=r['C2_not_shortened']['holds'] if r['C2_not_shortened']['testable'] else nt,
                     C3=r['C3_lengthened']['holds'] if r['C3_lengthened']['testable'] else nt,
                     baseline_identical=r['baseline_reproduces_test7']['identical'])
```

**M-2. Under observer scoring, short-T configurations are outside the regime of Chong et al.'s experiment, and the
calibration's 10% ceiling will select them out. Decide how to handle this before the run.**

The evidence below comes from the seventh test's results file (base and gated at 10%, observer scoring), split by the
configuration's T:

| model, T stratum | n | observer reports / merged model episodes (baseline) | baseline mOA / mA | gated 10%: fraction with mOA change ≥ 29% | gated 10%: median mOB change |
|---|---|---|---|---|---|
| sfa, T < 1 s | 31 | 0.54 | 1.88 | 0.16 | −5.4% |
| sfa, 1–2 s | 4 | 0.80 | 1.26 | 1.00 | +1.7% |
| sfa, ≥ 2 s | 25 | 0.99 | 1.01 | 1.00 | +2.8 to +8.0% |
| dep, T < 1 s | 25 | 0.60 | 1.68 | 0.08 | −6.5% |
| dep, 1–2 s | 10 | 0.80 | 1.26 | 0.80 | +1.3% |
| dep, ≥ 2 s | 25 | 0.98 | 1.02 | 1.00 | +4.5 to +7.6% |
| li, all (T ≥ 1.7 s) | 60 | 0.88–0.95 | 1.16–1.33 | 1.00 | −0.9 to +0.9% |

What this means for the run:

- **Short-T configurations are outside Chong et al.'s regime.** In the T < 1 s configurations the observer misses
  40–46% of episodes. Mean report latency (325 ms) plus the 520 ms ramp is about T itself. In my sfa behaviour
  configuration (T = 0.63 s), 34–45% of the increment's level fell inside the competitor's dominance. Its
  observer-scored attended change at 10% was +66%, against +13% when scored on the model. The "29% reported
  lengthening" there comes mostly from the observer merging through competitor episodes.
- **The project has called this regime invalid before.** It is the regime the manuscript already calls an invalid
  observer scoring: the third test's control samples, in which the observer reported 41% and 31% of episodes.
- **Chong et al.'s timing is weaker than the live gate.** In my depression behaviour configuration (T = 4.5 s), the
  attended change at 10% was +27.6%, so that configuration would not be calibrated. The seventh test's gate gave +43% in
  the same configuration with different noise. Coverage at 10% under Chong et al.'s timing will therefore be below the
  gate's 57–58% in the rate models.
- **The selection works in a direction that is known in advance.** The configurations the 10% ceiling drops are the ones
  in which the gate shortened the competitor. C2 and C3 in the rate models will therefore be estimated on a subset chosen
  by an attribute that predicts the outcome's sign.

Required changes (all pre-specifiable from the seventh test's baseline alone, without any condition outcome):

1. **Add a larger size, so that calibration coverage does not depend on the ceiling.** Set `SIZES = (0.025, 0.05, 0.10,
   0.20)` in `run_li2017_test8.py` line 30 and `analyse_li2017_test8.py` line 14. Update the plan's Sections 4 and 5,
   and C1's choice of size (keep C1 at 10% if you want comparability with the seventh test's gate).
   - Cost: one more simulation per chunk, 6 instead of 5, about +20% run time.
   - Optionally add 0.075 as well, which shortens the 5–10% bracket that most rate-model crossings will fall in (see
     F-3 on why the bracket width matters).
2. **Pre-specify an observer-valid stratum and say which set is primary.** Define it as configurations whose baseline
   ratio of reported to merged competitor episodes, `mean(nOB)/mean(nBm)` over seeds, is at least 0.8. That is close to
   T ≥ 1 s here, and it leaves about 29 sfa, about 35 dep and 60 Li configurations.
   - My recommendation: primary C2/C3 on calibrated observer-valid configurations, because the plan's premise is
     observer scoring as in Chong et al.; all calibrated configurations reported alongside.
   - The reverse is acceptable if it is justified in the plan. What matters is that the choice is made now.
   - Implement it in `model_block`:
     `valid = np.array([np.nanmean([x['nOB'] for x in c]) / np.nanmean([x['nBm'] for x in c]) for c in R['base']]) >= 0.8`,
     then run `calibrate(att[valid], comp[m][valid])` next to the existing call.
3. **Report coverage by stratum.** Report the number calibrated, never reaching 29%, and non-finite, per stratum. For
   uncalibrated configurations, report the competitor change at the largest size.
4. **Disclose the knowledge behind the choice.** The plan should say that the stratum was chosen knowing the seventh
   test's T-dependence of the gate's effect.

**M-3. Plan and README text that is now inaccurate.**

- **Section 3, "no result for the conditions below has been simulated", is no longer true.** This check ran Chong et
  al.'s timing at 2.5%, 5% and 10% on the first accepted configuration of each model, with 4 seeds of 30 s and of 120 s.
  It used tag `precheck8_<model>`, so the noise differs from the run's. Replace the sentence with: "Apart from the
  pre-run check (`precheck8.md`), which ran the conditions on the first configuration of each model with 4 seeds and
  different noise, no result for the conditions below has been simulated."
- **`README.md` lists `code_sha256.txt`, which does not exist.** Generate it after the fixes and before the commit.

## Should fix

- **S-1. C1 is close to trivially true.** In the seventh test, the gate at 10% lengthened the observer-scored attended
  grating in every configuration of every model (180/180; minimum +8%).
  - Keep C1 as a manipulation check, but label it as such.
  - Add the informative quantity: the proportion of configurations reaching 29% at the largest size (Wilson 95%), by
    stratum.
- **S-2. Nothing decides or reports "shortened", although the question is "shorten or lengthen, and by how much".**
  Add a C4 (decided or reported): the upper bound is below 0, and separately below −2%. State how the outcomes read:
  - a C2 failure whose interval straddles −2% is inconclusive;
  - C2 holding with C3 failing means "no shortening beyond the margin, lengthening not shown".
- **S-3. Report the model-scored attended change (mA) at the calibrated size, per stratum.** Where observer scoring and
  model scoring diverge (short T), the reader needs to see that the 29% is a reported, not a perceptual, lengthening.
  Optionally also report the passive observer (`mOA2`, `mOB2`, latency 300–600 ms) at each size as a latency
  sensitivity. It is already simulated and stored.
- **S-4. In Li et al.'s model the uncalibrated 10% condition is near lock-in and partly censored, so do not compare it
  with the third test's 8.7%.**
  - Behaviour configuration (120 s): duty 0.82; competitor predominance 0.49 → 0.18; reported attended +224%, with
    reported attended durations of 26–39 s in a 115 s window.
  - `observer_durations` drops the first and last reported runs, and `change()` drops seeds that are NaN in either run,
    so mOA at 10% is biased downward by censoring.
  - Li configurations will calibrate at about 2.5–5%. The comparison with the third test's 8.7% should use this test's
    calibrated value; its 8.7% was itself at a calibrated size. The 10% values should be labelled as near-lock-in.
- **S-5. Fail fast on baseline reproduction.** As it stands, a mismatch is found only by the analysis after a run of
  about an hour.
  - In `run_model`, after `r = fu.result()`, compare `r['base']` with
    `D7['models'][model]['results']['base'][kk*a.chunk:(kk+1)*a.chunk]` on `mA, mBm, mOA, mOB` and raise if any differ
    (non-dry runs only).
  - Also resolve `SRC7` relative to `__file__` (line 29), so the run does not depend on the working directory.
- **S-6. Plan Section 5 and the code disagree on MIN_N for C1.** The plan says "each [C1–C3] … requires at least 10
  calibrated configurations". The code decides C1 when n_config ≥ 10, regardless of calibration. The code's rule is
  sensible, so reword the plan.
  - Also, "not calibrated and is counted": the code reports only `n_calibrated`. Add the counts by reason (does not
    reach 29%; a needed value is non-finite).
- **S-7. Plan Section 7, dry run.** "Every condition reproduces its baseline exactly" holds for every trajectory
  measure. The level-based per-seed measures (`dose_comp`, `dose_att`, `area_comp`) differ in 138–144 of 816 values per
  model, by construction, because the level ramps even at size 0. Say "every trajectory-based measure".
  - Also note that `baseline_reproduces_test7: identical=false` in `dry8_tests.json` is expected (different settings and
    chunking), so that it is not read as a failure.
- **S-8. Name clash.** `results['dose_comp']` (run_chunk8: the fraction of the level that falls in the competitor's
  dominance, difference criterion) and the per-seed `dose_comp` in `R3.measures` (the mean level outside merged
  attended episodes) are different quantities. Rename the first, for example `level_in_comp`, or define both in the
  plan.
- **S-9. The synthetic check does not exercise the interpolation's failure modes.** Its effects are linear in size, so
  interpolation is exact by construction. Add one scenario with a convex attended response and a competitor response
  that is not proportional to it. Add another with about 30% of configurations uncalibrated and some NaN values, to
  exercise the counting and M-1.

## Fine as is

- **F-1. `Schedule8` reproduces the third test's `'chong'` condition exactly.**
  - In `'chong'` it calls `R3.Schedule3.__call__` once per step, with `self.mode == 'none'` set by `Schedule6`. That call
    steps both observers once, records them at even steps, updates the episode tracker and returns zeros, which are
    unused. It does not call `Schedule6.__call__`, so the observers are never stepped twice.
  - The level then rises by `dt/520` while `obs.rep == 1` and falls otherwise. It is read after the observer's update,
    at the same step as in `Schedule3`. The hold term in `Schedule3` is identically false when `extra=None`, because
    `since_rep < 0` never holds.
  - The bitwise test in `precheck/behav8.py`, for every model with 4 seeds of 30 s and 120 s at 10%, drove `Schedule3('chong')`
    through `li2017.simulate` (Li) or `rate_models.simulate` with the same seed and pools. `rb`, `on`, and both
    observers' records were identical to `run_batch8`'s. This also confirms that `li2017_t6.simulate` equals
    `li2017.simulate` for input increments.
  - Non-`'chong'` modes go through `Schedule6.__call__`, which calls Schedule3 once. The constructor maps
    `args[2]=N, args[3]=nt` correctly.
- **F-2. The noise and latency streams are the seventh test's.**
  - Tag `test7_<model>`, seeds 0–31, chunks of 10 in acceptance order, and `keyt`/`inst` from `sorted(T7.RANGES[model])`
    with `round(·, 6)`. The JSON float round trip is exact. The non-dry run asserts chunk 10, 32 seeds and 120 s.
  - `run_batch8` matches `run_batch7` line for line apart from the schedule class. For `'none'`, `Schedule8` and
    `Schedule7` both defer to `Schedule6.__call__`.
  - The second baseline uses `'none'` where the seventh test used `'none_rec'`. That only adds recording, so the
    trajectory and the observers are the same.
  - `repro8_check_sfa.log` shows a maximum relative difference of 0 on the first chunk. Each condition shares the batch
    seed with the seventh test's `gated`, so the 10%-minus-gate comparison is paired, noise for noise.
- **F-3. `calibrate()` is correct.**
  - It interpolates through (0, 0) and finds the first crossing, whose denominator is always positive because
    a[j−1] < 29 ≤ a[j]. A non-finite value before the crossing leaves the configuration uncalibrated, which is
    conservative.
  - Unit cases in `edge8.py` behaved correctly: monotone, non-monotone with the first crossing taken, dip then cross,
    never, NaN, exactly 29 at a grid point, negative attended change.
  - Because the competitor value uses the same weight w as the attended value, the interpolation is linear in the
    (attended change, competitor change) plane. Convexity in size therefore biases only the reported calibrated size,
    not the calibrated competitor change. The only bias in that change is the curvature of competitor against attended
    within the bracket, which is why a narrower 5–10% bracket (M-2.1) helps.
  - The ratio form adds a Jensen bias of order CV²/n_seeds, well below 0.1%. It shares the baseline denominator across
    sizes, so it does not distort the interpolation.
  - Calibrating on mOA and evaluating mOB is the right primary choice for an analogue of Chong et al., within the
    observer-valid regime (M-2).
- **F-4. C2's −2% margin is defensible.** It equals the size the manuscript gives prediction 1, about 2% according to
  the project notes; I did not check Table 6 itself. With 30–60 configurations and a per-configuration SE of about 2–4%
  under observer scoring, the bootstrap interval of the median will be about ±1–1.5%, so C2 and C3 can both be decided.
- **F-5. The rest of the analysis code is correct.** `change()`, `wilson()`, `boot_median_ci()`, MIN_N on n_config, and
  the predominance and episode-count summaries are correct. Rerunning `analyse_li2017_test8.py` on regenerated
  synthetic data reproduced `synth8_lengthen_tests.json` byte for byte.

## Behaviour run (scale 1.0; first accepted configuration per model; 4 seeds; tag `precheck8_<model>`)

These are changes in % against the same run's baseline. OA/OB are observer-scored attended and competitor durations; A
and Bm are unmerged attended and merged competitor durations on the model. "duty" is the mean level after burn-in. "lev
in comp" is the fraction of the level that falls in the competitor's dominance. "pred" is the competitor's reported
predominance, baseline → condition.

**30 s (as requested).** The Li and dep configurations have T ≈ 9.4 s and 5.2 s, which gives 0.8 and 2 reported
competitor episodes per seed. Their numbers are edge effects and cannot be read as effects (for example, Li's OA is
negative at every size).

| model | size | OA | OB | A | Bm | duty | lev in comp | pred |
|---|---|---|---|---|---|---|---|---|
| li | 2.5 / 5 / 10% | −21.4 / −7.5 / −16.7 | −0.6 / −2.7 / −3.1 | +0.6 / +18.7 / +64.1 | −0.6 / −1.9 / −72.9 | .65 / .70 / .94 | .05 / .05 / .02 | .43 → .35 / .30 / .06 |
| sfa | 2.5 / 5 / 10% | +8.9 / +22.6 / +63.5 | −6.0 / −7.6 / +3.4 | +8.3 / +12.4 / +7.0 | −0.7 / −3.3 / −17.4 | .50 / .53 / .60 | .46 / .41 / .35 | .53 → .50 / .47 / .40 |
| dep | 2.5 / 5 / 10% | +0.1 / +10.4 / +12.1 | −0.1 / +4.2 / +4.7 | +0.1 / +10.3 / +12.0 | −0.1 / +4.4 / +4.9 | .49 / .51 / .56 | .09 / .08 / .09 | .51 → .51 / .49 / .44 |

**120 s (the run's length; same configurations; 5.0 / 49 / 12 reported competitor episodes per seed).**

| model (T) | size | OA | OB | A | Bm | duty | lev in comp | pred |
|---|---|---|---|---|---|---|---|---|
| li (8.7 s) | 2.5 / 5 / 10% | +22.6 / +38.0 / +223.9 | −3.8 / −7.4 / −16.8 | +34.1 / +49.3 / +165.4 | −12.7 / −11.7 / −22.0 | .58 / .61 / .82 | .05 / .04 / .02 | .49 → .42 / .39 / .18 |
| sfa (0.63 s) | 2.5 / 5 / 10% | +7.4 / +18.3 / +66.2 | −1.9 / −0.6 / +2.0 | +4.5 / +7.9 / +13.1 | −3.3 / −7.7 / −16.6 | .51 / .53 / .61 | .45 / .40 / .34 | .51 → .49 / .47 / .39 |
| dep (4.5 s) | 2.5 / 5 / 10% | +7.6 / +13.2 / +27.6 | +1.4 / −0.7 / +5.9 | +7.5 / +13.0 / +25.7 | +1.4 / −0.7 / +3.6 | .54 / .56 / .58 | .11 / .11 / .09 | .47 → .45 / .44 / .42 |

Reading of the 120 s numbers. With 4 seeds, the per-configuration SE of OB is roughly 5–8% (Li), 3% (sfa) and 5%
(dep), so the competitor changes are within noise.

- **Li:** it would calibrate at about 3.5% (interpolated competitor about −5%, noisy).
- **sfa:** about 5.8% (competitor about −0.1%), but in the short-T regime of M-2. OA +66% against A +13%.
- **dep:** it would not calibrate (27.6% at 10%).

**Level trace against reports (instance 0, 120 s, all three models).**

- The level stays within [0, 1], and the largest per-sample step is 0.001923 = 1/520 per ms, as expected.
- It never falls while the primary observer reports grating 1 (fraction 1.000) and never rises otherwise (1.000).
- Aligned on reports of grating 1, the mean level is 0.00 / 0.19 / 0.50 / 1.00 / 1.00 at 0 / 100 / 260 / 520 / 800 ms
  (Li, dep).
- Aligned on reports of grating 2, it is 1.00 / 0.81 / 0.50 / 0.00 / 0.00.
- At the model's switch to the competitor, the level is still 1.0 (Li, dep) or 0.74 (sfa). The increment therefore
  overlaps the start of each competitor episode by the report latency plus the down-ramp, as Chong et al.'s timing
  implies.
- For sfa, where reported periods are often shorter than the 520 ms ramp, the level at a grating-1 report starts at
  0.06 and reaches only 0.90 at 520 ms on average.

The increment behaves as specified.

**Cost.** One configuration × 4 seeds × 120 s, through `run_chunk8` plus two extra batches: Li 241 s, sfa 86 s, dep
89 s, on one core each.

## Scratch files (`precheck/`)

- `behav8.py`, `behav8b.py`: behaviour run and bitwise test against `Schedule3`.
- `behav8_<model>_{30s,120s}.json`, `*.log`: behaviour-run output.
- `trace_<model>_{30s,120s}.npz`: level, reports, responses and threshold for instance 0.
- `edge8.py`, `edge8_tests.json`: M-1 and the calibrate() unit cases.
- `synth8_lengthen_tests_re.json`: re-analysis of the regenerated synthetic data. The roughly 30 MB synthetic inputs
  were deleted after use and are regenerated by `make_synth8.py`.

## Changes made in response (by the authors, before the run)

- **M-1.** Decisions for a model with fewer than 10 calibrated observer-valid configurations are now reported as "not
  tested (n calibrated ...)" rather than as failures (checked on the synthetic 'convex' scenario, model 'dep').
- **M-2.** Sizes 7.5% and 20% added (`SIZES = (0.025, 0.05, 0.075, 0.10, 0.20)`). An observer-valid group (baseline
  reported / merged competitor episodes ≥ 0.8) is pre-specified, and C2 to C4 use it as primary; all calibrated
  configurations are reported alongside. Coverage is reported by group and reason (calibrated, never reaching 29% by
  20%, non-finite), with the proportion reaching 29% at 20% and, for uncalibrated configurations, the competitor change
  at 20%. The plan states that the criterion was chosen knowing the seventh test's dependence on T.
- **M-3.** Plan Section 3 now mentions this check's behaviour runs. `code_sha256.txt` generated after the changes.
- **S-1.** C1 labelled a manipulation check that is expected to hold; the proportion reaching 29% at the largest size is
  reported.
- **S-2.** C4 (reported) added: the interval's upper bound below 0 and below −2%. The plan gives the reading of each
  combination of outcomes.
- **S-3.** At the calibrated size the analysis now reports the attended grating's change on the model (unmerged
  scoring) and the passive observer's competitor change, for both groups.
- **S-4.** The plan says the larger sizes approach lock-in in Li et al.'s model and that the comparison with the third
  test uses the calibrated value.
- **S-5.** The run compares each chunk's baseline with the seventh test's and stops on any difference; `SRC7` is resolved
  relative to the script.
- **S-6.** The plan states the minimum numbers for C1 and for C2 to C4 as the code applies them, and coverage is counted
  by reason.
- **S-7.** The plan's description of the dry run says "trajectory-based measures" and notes that
  `baseline_reproduces_test7` is false in the dry run by design.
- **S-8.** The run's fraction of the level in the competitor's dominance is renamed `level_in_comp`.
- **S-9.** The synthetic check has a third scenario: a convex attended response, a competitor response not proportional
  to it, observer-invalid configurations, uncalibrated configurations and missing values. It is recovered to within
  0.2 points of the exact solution.
- The dry run was repeated with the five sizes (`dry8.json`, `dry8_tests.json`).
