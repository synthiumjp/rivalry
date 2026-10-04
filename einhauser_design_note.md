# Design note for registration osf.io/3rthv

**Date:** 4 October 2026
**Status:** written after step 2 of the registered order of work and before step 4. At the time of
writing, `OKNrawGain`, `idxButton` and `isCorrect` have not been loaded or inspected, and the
article's Results and Figures have not been read.

## What was inspected

1. The authors' example code (`figure2.m` to `figure6and7.m`), for the design and the sign convention.
2. The design variables `diffBlue`, `diffRed`, `eyeBlue` and `eyeRed`, loaded on their own.

## Sign convention

The authors' comment in `figure3.m`: in screen coordinates the stimulus shown to the left eye moves in
the positive direction and the stimulus shown to the right eye in the negative direction. Positive OKN
gain therefore means the left-eye stimulus is dominant. For each trial, a stimulus's direction is +1 if
its eye variable is 1 (left) and −1 if it is 2 (right). The two stimuli are in opposite eyes on every
trial that was run.

## Design

Difficulty is assigned trial by trial, fully crossed (blue easy or hard × red easy or hard), in every
phase, balanced for every observer:

| Phase | Blocks | Trials with OKN data | Per difficulty combination | Exactly one stimulus hard |
|---|---|---|---|---|
| Pre | 1–4 | 32 | 8 | 16 |
| Task | 5–12 | 128 | 32 | 64 |
| Post | 13–16 | 32 | 8 | 16 |

Eye of presentation is balanced, with blue in the left eye on half the trials in each phase.

## Branch, and one clarification

**Design B applies**: difficulty varies across trials. The registration's Design B anticipated that
the post phase would have no per-trial difficult stimulus and supplied a majority rule for that case.
The post phase does have one, as does the pre phase. Design B's own definition is therefore applied in
every phase: on trials where exactly one stimulus is hard, T is the hard stimulus and O the easy one.
The majority rule and its 75% threshold are not needed. No hypothesis, test or threshold changes.

All other definitions are as registered, with these details fixed now:

- **Δ** for role R (T or O) in phase X uses that role's mean episode duration on the
  exactly-one-hard trials of phase X, against the same role on the exactly-one-hard trials of the pre
  phase.
- **Pre-task mean duration** for H2 is the mean of all valid episodes, both stimuli, on all pre-phase
  trials, as the registration's wording specifies.
- **Task phase** means whole task trials, as registered. The split at the button press is not used in
  the primary analysis.
- **Order of episode filtering:** the first and last episode of each trial are removed first, since
  they are the ones truncated by the trial boundaries; episodes shorter than 300 ms are then discarded.
- **NaN gaps of 300 ms or longer** are treated as indeterminate and end an episode, since the
  registration fills only shorter gaps.
- **Holm correction** across H1 and H2: the smaller p is compared with .025 and the larger with .05.

## Two matters bearing on interpretation, recorded before the outcome is known

1. **Difficulty is a property of the stimulus, present before any task.** The post phase therefore
   measures how difficulty affects dominance after the task, relative to before it. It does not
   isolate a bias created by the task. Whether difficulty's influence in the no-task phases acts
   during suppression, and so counts as ungated, is not known. This is the case foreseen in §7 of the
   registration, and its interpretation clause applies as written.
2. **Durations may drift across the session.** The post phase comes last, and a general lengthening
   of durations over time would raise Δ<sub>O,post</sub> and so bias against H1. This does not change
   the registered test.

## Additional analyses, specified now and labelled exploratory

These are added before outcome inspection and are reported as exploratory, alongside the registered
sensitivity analyses.

- **Drift-corrected Δ:** each role's duration in a phase divided by the mean duration on that phase's
  symmetric trials (both easy or both hard), before computing change from the pre phase.
- **Pre-response task phase:** the task-phase analysis restricted to samples before the button press,
  where attention to the dominant stimulus is cleanest.
