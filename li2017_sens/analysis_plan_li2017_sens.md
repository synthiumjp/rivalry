# Sampling-sensitivity check of the sign reversal in Li et al.'s (2017) model: analysis plan

**Authors:** Jon-Paul Cacioli, Chris Marmo
**Project:** https://osf.io/kqdgb/

## 1. Purpose

In five tests in Li et al.'s model, a gated increment lengthened the competitor in at least 95% of
configurations under merged scoring, and continuous delivery shortened it in all of them. Every test
drew configurations uniformly within the same parameter ranges and accepted only those with a median
coefficient of variation (CV) of dominance durations between 0.3 and 0.7. Reviewers asked whether the
result depends on those choices. This check repeats the gated and continuous comparison in two new
ensembles that change them.

## 2. Ensembles

- **Ensemble L (sampling and bounds).** Each parameter of the third test's ranges (input strength,
  opponency, attention weight, adaptation weight, τ<sub>h</sub>, noise) is drawn log-uniformly from
  its range widened by 25% of its log-width on each side (for example τ<sub>h</sub> from 0.71 to 5.66
  s instead of 1 to 4 s). Acceptance: at least 10 episodes per seed and a median CV between 0.2 and 0.8.
  Seed 20261025.
- **Ensemble U (eligibility).** The third test's ranges, drawn uniformly, with no CV filter: every
  configuration that alternates (at least 10 episodes per seed) is accepted. Seed 20261026.

Acceptance uses 8 seeds of 120 s and the difference criterion, as in the earlier tests. Rejected
draws are counted by reason. Sixty configurations per ensemble, in batches of 10.

## 3. Conditions

Baseline, gated (on while the attended grating is dominant, zero latency) and continuous delivery of
an increment of 10% of input strength, 16 seeds of 120 s per condition with common random numbers,
first 5 s discarded. Code: `run_li2017_sens.py`, which uses `li2017.py` and `run_li2017_test3.py`
unchanged; only the sampler, the acceptance rule, the number of seeds and the conditions are new.

## 4. Hypotheses and decision rules

Changes are percentage changes in the competitor's mean duration relative to baseline, per
configuration. Decisions use merged scoring and are made separately for each ensemble; all are also
reported under observer scoring, unmerged scoring and the absolute criterion.

- **S1.** Gated delivery lengthens the competitor: the lower Wilson 95% bound of the proportion of
  configurations in which it does exceeds 0.75.
- **S2.** Continuous delivery shortens the competitor: the lower Wilson bound of the proportion in
  which it does exceeds 0.75.
- **S3.** The gated change exceeds the continuous change: lower Wilson bound above 0.75.

A configuration in which a condition leaves no measurable competitor episode is excluded from that
condition's comparisons and counted. Also reported: medians, numbers drawn and rejected by reason.

## 5. Status

Not registered. The plan and code were written, and a dry run with every increment set to zero
(seed 107, 4 configurations, 4 seeds, 30 s; `li2017_sens_L_dry.json`, `li2017_sens_U_dry.json`)
checked the code path, before the run; their SHA-256 checksums were recorded in `code_sha256.txt`
before the run started, and the files were committed to the repository with the results.
