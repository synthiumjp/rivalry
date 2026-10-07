# A pre-registered test of the delivery-timing account in the attention model of binocular rivalry of Li, Rankin, Rinzel, Carrasco and Heeger (2017)

**Authors:** Jon-Paul Cacioli, Chris Marmo
**Related registrations:** https://osf.io/d975z/ (original study); https://osf.io/3rthv/ (secondary analysis of human data)
**Code:** https://github.com/synthiumjp/rivalry, folder `li2017_test/`, commit [COMMIT HASH]

**Status at registration.** The model has been ported and its baseline behaviour checked against
the published paper (Section 4). The acceptance rate of the sampling procedure was checked on 25
baseline draws, and the full analysis was run once with every increment set to zero to check the
code (Section 4). No condition with a non-zero increment has been run in this model.

## 1. Background

In a leaky competing accumulator with mutual inhibition and adaptation, and in three of four
further generic architectures, we found that an input increment on one channel lengthens the
competing channel's dominance episodes when it is delivered only while the incremented channel is
dominant, and shortens them when it is delivered continuously. A reduction of the dynamics to one
slow variable travelling between two switching thresholds explains the sign: an increment present
at the handover that begins the competitor's episode lengthens it, and one present at the handover
that ends it shortens it by more. Those analyses were exploratory. This registration tests the
account's predictions in a published model it was not developed in: the attention model of
binocular rivalry of Li et al. (2017), which combines divisive normalisation, opponency-based
mutual inhibition, slow divisive adaptation and a stimulus-driven attention layer.

## 2. Model and manipulation

The model is the published MATLAB implementation (archive.nyu.edu/handle/2451/38721, CC BY-SA
3.0), ported line by line to Python: forward Euler with a 0.5 ms step, the same update order, the
smoothed rectifier, and the published parameters (input strength 0.5 per eye, exponent 2,
monocular exponent 1, σ = 0.5, gain 2, adaptation weight w<sub>h</sub> = 2, opponency weight
w<sub>o</sub> = 0.65, attention weight w<sub>a</sub> = 0.6, time constants 5, 20, 150 and 2,000 ms
for the sensory, opponency, attention and adaptation variables, attention suppression constant
0.2, onset transient as published). Two additions are made. First, the published code is
deterministic and alternates with constant durations, so, as described in the paper's Methods, an
Ornstein-Uhlenbeck noise term with a 100 ms time constant is added to each monocular input.
Second, an additive increment can be applied to one grating's input on a schedule.

Two orthogonal gratings, 1 and 2, are presented to the left and right eye respectively. Grating 1
is the attended grating, to which the increment is applied; grating 2 is the competitor.

*Dominance.* Grating 1 is dominant while R<sub>b1</sub> − R<sub>b2</sub> exceeds θ and grating 2
while the reverse holds, where R<sub>b</sub> are the binocular summation responses and θ is 5% of
the run's mean baseline binocular response (the difference criterion). Under the absolute
criterion, grating 2 is dominant while R<sub>b2</sub> exceeds the median of its own baseline
response, averaged over seeds. The first 5 s of each run are discarded, the first and last episode
are excluded, and episodes shorter than 50 ms are not counted.

*The increment.* An increment Δ equal to 10% or 20% of the configuration's input strength is added
to grating 1's input on one of these schedules:

1. Baseline: no increment.
2. Gated: only while grating 1 is dominant.
3. Ungated: at every time step.
4. Yoked: on the dominance time course of grating 1 taken from the baseline run of another seed of the same configuration.
5. Offset delay (20% only): switched on when grating 1 becomes dominant and off a delay *d* after it stops being dominant, with *d* = 0.25, 0.5, 1.0 and 1.5 times the configuration's mean baseline dominance duration.
6. Onset delay (20% only): switched on a delay *d* after grating 1 becomes dominant and off as soon as it stops being dominant, with the same values of *d*.

Every condition for a configuration is run with the same noise realisation as its baseline, so
that with a zero increment every condition reproduces the baseline exactly.

## 3. Configurations and sampling

Configurations are drawn at random (seed 20261007) around the published operating point, each
parameter independently and uniformly: input strength (both eyes) 0.35 to 0.8; opponency weight
w<sub>o</sub> 0.5 to 0.75; attention weight w<sub>a</sub> 0.4 to 0.9; adaptation weight
w<sub>h</sub> 1.5 to 2.5; adaptation time constant τ<sub>h</sub> 1,000 to 4,000 ms; noise standard
deviation 0.03 to 0.07. Each condition is simulated for 8 seeds of 120 s of model time. A
configuration is accepted if, at baseline, it produces on average at least 10 dominance episodes
per seed and the median across seeds of the coefficient of variation of dominance durations lies
within [0.3, 0.7]. Acceptance is decided before any increment condition is run, and sampling stops
at 200 accepted configurations. The published parameter set (noise standard deviation 0.05) is also
run as a single pre-specified configuration and reported descriptively.

## 4. Checks already performed

*Port.* Without noise and at the published parameters, the port reproduces the published
behaviour. With attention (condition 1) the two gratings alternate with a constant dominance
duration of 2.97 s. Without attention (condition 2, w<sub>a</sub> = 0) the responses settle within
10 s to equal activity and rivalry stops, which is the paper's central result. Monocular plaids
(conditions 3 and 4) do not alternate. With input noise of standard deviation 0.05, dominance
durations average 1.8 s with a coefficient of variation of 0.44. At that noise level the port
follows the four modified Levelt propositions as the paper reports: varying grating 1's input from
0.3 to 0.8 with grating 2 at 0.5 raises grating 1's predominance from 0.01 to 0.99 (I); the
stronger grating's mean duration changes more than the weaker one's on both sides of equidominance,
for example +875 ms against −601 ms from 0.5 to 0.6 (II); alternation rate peaks, at 0.52 per
second, at equidominance (III); and raising both inputs together from 0.35 to 0.8 raises
alternation rate from 0.46 to 0.73 per second (IV).

*Sampling.* On 25 baseline draws from the ranges above, 19 met the acceptance criteria.

*Code path.* The full analysis was run on 4 configurations, with 30 s runs and every increment set
to zero. Every condition reproduced its baseline exactly (all changes 0.0%), confirming that the
conditions share their baseline's noise.

## 5. Hypotheses

For each accepted configuration and condition, the mean dominance duration of each grating is
averaged over seeds and expressed as a percentage change from the same configuration's baseline.
Proportions are tested with Wilson 95% intervals across configurations that have a measurable
competitor episode in the condition.

**H1 (primary).** Under the difference criterion, at both increment sizes, gated delivery
lengthens the competitor in more than half of configurations (lower Wilson bound above 0.5), and
ungated delivery lengthens it in fewer than half (upper Wilson bound below 0.5).

**H2.** Under the absolute criterion, the same holds as in H1, at both increment sizes.

**H3.** Gated delivery lengthens the competitor more than yoked delivery: the paired difference in
competitor change (gated minus yoked) is positive in more than half of configurations (lower
Wilson bound above 0.5), under both criteria and at both increment sizes.

**H4.** Delaying the withdrawal of the increment reverses its effect. At the 20% increment, under
the difference criterion, the median competitor change does not increase across offset delays of
0 (gated delivery), 0.25, 0.5, 1.0 and 1.5 mean durations, and is negative at 1.5.

**H5.** Delaying the onset of the increment does not reverse its effect. At the 20% increment,
under the difference criterion, the median competitor change is positive at every onset delay
tested.

**H6 (secondary, scope).** Across configurations, the gated coupling ratio at the 10% increment
(competitor change divided by attended change, under the difference criterion) correlates
positively with exp(−*T*/τ<sub>h</sub>), where *T* is the configuration's mean baseline dominance
duration: Spearman ρ > 0 with a one-sided *p* below .05 against 5,000 permutations of τ<sub>h</sub>
across configurations. The reduction behind this prediction assumes adaptation subtracted from the
drive, whereas in this model adaptation divides it, so H6 tests the reduction's scope; its failure
would narrow the claim rather than contradict the sign rule.

## 6. Decision rules and reporting

H1 is the test of the central claim. If it fails at either increment size, we will report that the
delivery-timing account does not transfer to this model. Every hypothesis is reported whatever its
outcome, together with the medians and counts behind it and the results for the published
parameter set. Analyses not listed here will be labelled exploratory.

## 7. Code

The port (`li2017.py`), the analysis (`run_li2017_test.py`) and the validation scripts are in the
repository folder and commit given at the top of this document, with the original README and
licence as the licence requires. The registered run is `python run_li2017_test.py`.
