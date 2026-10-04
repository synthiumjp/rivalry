# Pre-registration: a test of the dominance-gating account on existing human data

**Type:** secondary data analysis, confirmatory
**Registrants:** Jon-Paul Cacioli, Chris Marmo
**Related registration:** https://osf.io/d975z/ (March 2026). This registration does not amend it. The
March registration covers the original modelling study, whose registered tests were run as specified.
The hypotheses below arose from that study's exploratory analyses and are registered here before being
tested on independent data. Departures of the modelling study from the March registration are recorded
separately in its deviation note.
**Manuscript:** "Why a dominance-gated contrast increment mimics attention in binocular rivalry"
(Sections 4.3 and 4.4 derive the predictions tested here).

---

## 1. Data

Einhäuser, W., Sandrock, A., & Schütz, A. (2021). Data supplementing "Perceptual difficulty persistently
increases dominance in binocular rivalry – even without a task". Zenodo,
https://zenodo.org/records/4575552. Published article: *Perception*, 50(4), 343–366.

The data are public and de-identified. Per the data description: 24 observers × 16 blocks × 16 trials;
dominance measured from optokinetic nystagmus (`OKNrawGain`, sampled at 1 ms); per-trial task difficulty
for each rival stimulus (`diffBlue`, `diffRed`); eye assignment (`eyeBlue`, `eyeRed`); task responses in
blocks 5–12 only; no task in blocks 1–4 and 13–16.

## 2. What we have seen, and what we will not see before analysis

At registration neither registrant has downloaded the data or read the article's Results or Figures. We
have read the title, the Zenodo data description, and citations of the article in other work. Before the
analysis is fixed we will read only (a) the article's Methods section and (b) the authors' posted example
code, solely to establish the design and the sign convention of `OKNrawGain`. We will not inspect
`OKNrawGain`, `idxButton` or `isCorrect` until the analysis code below is written and committed (§6).

## 3. Background and predictions

The modelling study finds that in competitive networks with adaptation, a modulation present only while
the attended stimulus is dominant (*gated*) lengthens the attended stimulus without shortening the
competitor, whereas the same modulation present throughout (*ungated*) shortens the competitor, as
Levelt's second proposition describes. It derives this from the geometry of switching between adaptation
thresholds, and predicts that the gated competitor-to-attended ratio of duration changes falls as mean
dominance duration rises.

In this dataset, task-driven attention to a difficult stimulus can act only while that stimulus is visible,
so during task blocks it is gated. Any bias in favour of that stimulus that persists after the task ends
acts whenever the stimulus is present, so it is ungated. That gives two tests.

**H1 (primary; the sign rule).** The competitor's duration change relative to pre-task blocks is more
positive during the task phase (gated) than in the post-task phase (ungated):
Δ<sub>O,task</sub> − Δ<sub>O,post</sub> > 0.

**H2 (primary; the duration–ratio relationship).** Across observers, the task-phase coupling ratio
r = Δ<sub>O,task</sub> / Δ<sub>T,task</sub> is negatively rank-correlated with the observer's mean
pre-task dominance duration.

**Secondary, descriptive.** H1a: the median Δ<sub>O,task</sub> is not negative. H1b: the median
Δ<sub>O,post</sub> is negative.

## 4. Definitions

**Phases.** Pre: blocks 1–4. Task: blocks 5–12. Post: blocks 13–16. Only trials with OKN data are used
(the description indicates trials 1–8 in pre and post blocks).

**Trained stimulus (T) and other stimulus (O).** Determined from the design variables alone, before any
outcome is inspected:

- *Design A:* if, within an observer, one stimulus is difficult throughout the task phase, T is that
  stimulus for that observer in every phase.
- *Design B:* if difficulty varies across trials, T is defined per task trial as the difficult stimulus
  in trials where exactly one is difficult, and those trials form the task-phase sample. The post phase
  then has no per-trial T; T for the post phase is the stimulus that was difficult in the majority of
  that observer's task trials, and H1 is tested only for observers with a majority of at least 75%.
  Otherwise H1 is reported as untestable in this design.

**Dominance extraction.** Each 1 ms sample is assigned to stimulus *s* when the OKN gain is in *s*'s
direction (sign convention from the authors' code) with |gain| ≥ 0.3, and is otherwise indeterminate.
NaN gaps (fast phases, blinks) shorter than 300 ms are filled with the preceding state. A dominance
episode is a maximal run of one state, allowing indeterminate gaps under 300 ms. Episodes shorter than
300 ms are discarded, as are the first and last episode of each trial, which are truncated.

**Duration measures.** For each observer, phase and stimulus: the mean episode duration. Changes are
Δ<sub>s,phase</sub> = 100 × (D<sub>s,phase</sub> − D<sub>s,pre</sub>) / D<sub>s,pre</sub>. Pre-task mean
duration, for H2, is the mean of both stimuli's pre-phase episodes.

## 5. Analysis

**Exclusions.** A trial is excluded if more than 50% of its samples are NaN. An observer is excluded from
a test if any required phase has fewer than 10 valid episodes for either stimulus.

**Precondition for H1.** H1 is interpretable only if the trained stimulus gains in both phases, as the
dataset's title reports for dominance. We require the group median Δ<sub>T,task</sub> > 0 and
Δ<sub>T,post</sub> > 0. If either fails, H1 is reported as uninformative, not as a failure of the
account.

**H1 test.** One-sided Wilcoxon signed-rank test on Δ<sub>O,task</sub> − Δ<sub>O,post</sub> across
observers.

**H2 test.** One-sided Spearman correlation between r and pre-task mean duration, over observers with
Δ<sub>T,task</sub> > 5%; below that the ratio is unstable because its denominator nears zero.

**Error control.** Holm correction across H1 and H2 at family-wise α = .05.

**Reporting.** We report every test and descriptive statistic named here, with 95% bootstrap intervals
(10,000 resamples over observers), whatever the outcome, and the number of observers entering each.

**Sensitivity, labelled exploratory.** Gain thresholds of 0.2 and 0.5; minimum episode durations of
200 and 500 ms; H2 without the 5% floor.

## 6. Order of work, and how it can be checked

1. This registration is posted on OSF.
2. The data and code are downloaded; the Methods section and example code are read, and only the
   design variables are inspected.
3. The design branch (A or B) and the sign convention are recorded in a dated note.
4. The analysis code implementing §4–5 is committed to https://github.com/synthiumjp/rivalry, and its
   commit hash is added to this registration as an update.
5. Only then is the code run on the outcome variables.

## 7. Interpretation, fixed in advance

- **H1 supported:** the competitor's response changes sign between gated and ungated phases in the
  direction the account predicts. This is reported as confirmatory support.
- **H1 not supported, with Δ<sub>O,post</sub> ≥ 0:** either the account fails for human attention, or the
  persistent post-task bias is not an ungated increase in strength. For instance, it may be a lasting
  change in adaptation, which the account's scope condition (manuscript Section 5.2) excludes. Both
  readings are reported; the result is not claimed as support.
- **H2:** with 24 observers, power to detect ρ = −0.5 one-sided is about 0.81 at α = .05 and 0.71
  at the Holm-adjusted .025, and lower if observers are excluded. A null result is weak
  evidence against the relationship and is reported as such.
- Whatever the outcome, the result is reported in the manuscript.
