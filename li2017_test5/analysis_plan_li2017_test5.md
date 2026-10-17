# A fifth test, with a pre-specified analysis plan, in the attention model of binocular rivalry of Li, Rankin, Rinzel, Carrasco and Heeger (2017): does the gated lengthening come from the increment's presence at the handover, and does the kernel's time course hold for brief pulses?

**Authors:** Jon-Paul Cacioli, Chris Marmo
**Project:** https://osf.io/kqdgb/
**Code:** https://github.com/synthiumjp/rivalry, folder `li2017_test5/`, at the commit that precedes the run

## 1. Background and purpose

In the fourth test, a 5% increment delivered only while the attended grating was dominant (the live
gate) lengthened the competitor by 4.9% (57 of 59 configurations). A linear kernel estimated from the
model's input noise predicted no change for it (−0.1%), given when the attended grating tends to be
dominant, although it predicted continuous delivery and yoked replay. External reviews of the
resulting manuscript raised two points that this test addresses:

1. **What about the live gate matters.** The kernel's failure shows that the gate's timing covaries
   with the state of each trial. It does not show which property of that timing produces the
   lengthening. The switching-threshold account predicts that the increment's presence at the
   handover, the moment the attended grating loses dominance, is what lengthens the competitor. The
   yoked control did not isolate this, because the yoked increment was also on during the
   competitor's dominance.
2. **Whether the kernel's time course is right.** The kernel predicted continuous delivery and the
   profile of quarter-length windows, but its weights were never compared with the effects of brief
   pulses.

**Design.** The live gate is split into two halves at a time *c* after the attended grating becomes
dominant, with *c* chosen for each configuration so that the two halves are on for the same time at
baseline. The *early half* is on from the start of the attended period until *c*, and so is off at
most handovers. The *late half* is on from *c* until the handover, and so is on at most handovers.
Both are confined to the attended grating's dominance, so neither is on during the competitor's
dominance. A third arm, the *late half without the transition*, is the late half switched off once
the attended grating's advantage falls below three times the dominance margin, so that it is off
through the switch itself while its time on is almost unchanged. Comparing the late half with the
early half asks whether presence late in the attended period matters; comparing it with the late half
without the transition asks whether presence through the switch itself matters, at nearly matched
time on.

Two cautions are built into the reporting. Under an increment, the late half delays the handover and
so is on for longer than the early half; the comparison of the halves is therefore also reported per
unit of realised time on. And the late half is on while the attended grating is most adapted, so the
halves differ in the adaptation state as well as in their presence at the handover.

*Disclosure.* No pilot was run. An independent check of an earlier version of this code ran three
configurations from the dry-run sample (seed 105) with a 10% increment, 8 seeds of 60 s. The earlier
version did not match the halves' time on; the check found the late half lengthening the competitor
more than the early half in one configuration and less in another. The code was then revised (the
attended-period and competitor-episode trackers, the split point and the transition arm), and the
hypotheses below are the revised ones.

## 2. Model and code

`li2017.py` and `run_li2017_test3.py` are unchanged from the third and fourth tests, and the kernel
estimation, merged episodes and the start-up check of `sim_noise.py` are imported unchanged from
`run_li2017_test4.py`. New code: `run_li2017_test5.py` (the halves and the pulses) and
`analyse_li2017_test5.py`. Checksums are in `code_sha256.txt`.

## 3. Sample

Sixty configurations drawn with seed 20261024 from the third test's main ranges and accepted by its
rule (at least ten episodes per seed and a median coefficient of variation between 0.3 and 0.7, on 8
seeds of 120 s), in batches of 10 configurations. Each condition: 32 seeds of 120 s with common random
numbers; the first 5 s discarded. Kernel: 192 further seeds of 120 s without increment, as in the
fourth test. The run is `python run_li2017_test5.py --workers N`; the batch size is part of the design
and the number of workers does not affect the results.

## 4. Conditions

All increments are 10% of input strength. *T* is the configuration's mean baseline dominance duration,
as in the third test.

| Condition | Increment on |
|---|---|
| Live gate | while grating 1 is dominant (zero latency), as in the earlier tests |
| Early half | while grating 1 is dominant and less than *c* into the attended period |
| Late half | while grating 1 is dominant and at least *c* into the attended period |
| Late half without the transition | as the late half, and only while grating 1's advantage exceeds three times the dominance margin |
| Continuous | throughout |
| Pulses | during [*a*, *a* + 0.125) *T* of each competitor episode, while grating 1 is not dominant, for *a* = 0, 0.25, 0.5, 0.75, 1.0 and 1.25 |

*Episodes and periods.* Defined online as in merged scoring and as in the kernel. A competitor
episode begins when a grating-2 run reaches 50 ms (dated from that run's start) and ends when a
grating-1 run reaches 50 ms. An attended period begins when a grating-1 run reaches 50 ms outside a
competitor episode (dated from that run's start) and ends when the next competitor episode begins.
Brief dominance of grating 1 inside a competitor episode is outside the attended period, so the halves
together are on slightly less than the live gate.

*Split point.* For each configuration, a run with zero increment (identical to the baseline) records,
online, the time into the attended period at every sample at which grating 1 is dominant within one; *c*
is the median of these times over all 32 seeds, so the two halves are on for equal time at baseline. A
configuration with fewer than 1,000 such samples has no split point and is excluded from the
comparisons that need one.

## 5. Hypotheses and decision rules

Changes are percentage changes in the competitor's mean duration relative to the configuration's
baseline. Medians are across configurations. Decisions use merged scoring; every hypothesis is also
reported under observer scoring, unmerged scoring and the absolute criterion.

- **L1 (primary). The late half lengthens the competitor more than the early half.** The lower Wilson
  95% bound of the proportion of configurations with late > early exceeds 0.5.
- **L2 (primary). The late half lengthens the competitor.** The lower Wilson 95% bound of the proportion
  of configurations in which it lengthens the competitor exceeds 0.5.
- **L3 (primary). Presence through the switch itself matters.** The late half lengthens the competitor
  more than the late half without the transition: the lower Wilson 95% bound of the proportion of
  configurations with late > late without the transition exceeds 0.5.
- **L4. The kernel's time course holds for pulses.** Over configurations with a kernel, the ratio of the
  sum of the six median observed pulse effects to the sum of the six median kernel predictions lies
  within [0.67, 1.5], and the Pearson correlation between the two six-point profiles is at least 0.7.

*Kernel predictions for pulses.* As in the fourth test (percent of the kernel's mean merged competitor
duration, increment Δ = 0.1 × input strength): 100Δ·Σ β<sub>A</sub>·overlap(bin, [*aT*, (*a* +
0.125)*T*)).

*Reported without decisions:* the comparison of the halves per unit of realised time on; each
condition's change in the attended grating's mean duration and its time on; the halves' changes as
fractions of the live gate's; and the live gate and continuous delivery with their kernel predictions
(a replication of the fourth test's K1 and K3).

## 6. What will be reported

All hypotheses whatever their outcome, under every scoring; the number of configurations drawn and
accepted; configurations without a split point or a kernel. Analyses not specified here are
exploratory.

## 7. Status of this plan

The plan was not registered on OSF. It and the code were committed to the repository before the run.
A dry run with every increment set to zero (seed 105, 4 configurations, 4 seeds, 16 kernel seeds,
30 s runs; `dry5.json`) checked the code path and confirmed that the two halves are on for equal time
at baseline, and the analysis was run on the dry-run output with four of the fourth test's kernels
substituted (`codecheck5.json`) to check the kernel-prediction code. The independent check described
in Section 1 is the only run of this code with a nonzero increment before the run.
