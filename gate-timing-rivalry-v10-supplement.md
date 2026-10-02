# Supplementary material

## Why a dominance-gated contrast increment mimics attention in binocular rivalry

Jon-Paul Cacioli and Chris Marmo

Section, table and figure numbers prefixed S refer to this supplement; unprefixed numbers refer
to the main text. References are listed in the main text.

## S1 Pre-registered tests in full

We report the pre-registered study first and compactly, because its two registered
contrasts are **null**, and because what follows depends on that. The substantive
results in Sections 4.3 to 4.5 come from controls and manipulations that were not
registered, and we label them exploratory throughout rather than presenting nulls and
discoveries as though they carried the same evidential weight.

**Dose-response (Prediction 3, supported).** The registered test was a logistic
dose-response fit of switch probability on goal strength. It is supported. Fitting the binomial outcome
directly, 50 seeds at each of six goal levels, the adaptation-dominated regime gives a **median
per-configuration slope of +3.264 with all 12 fittable configurations positive**, and the
inhibition-dominated regime +1.105 with 10 of 13 positive. Only 12 and 13 of 30 configurations
admit a fit at all, because the remainder never switch at any level, which is the same fact
Prediction 4 reports below.

Pooling across configurations instead gives +0.921 [+0.745, +1.097] and +0.241 [+0.092, +0.391],
both reliably positive but roughly a third and a fifth of the per-configuration medians.
Configurations differ in intercept as well as in slope, so a pooled logistic flattens the
dose-response; we report the per-configuration distribution as primary in line with Section S14,
and note this as a further instance of the aggregation hazard catalogued in Section S9. An
independent replicate on 118 configurations gives +1.411 [+1.144, +1.679] and +0.175 [+0.035,
+0.316] pooled.

Two caveats attach to the slopes above. Only 12 and 13 of 30 configurations admit a fit, and
fittability is itself the switching outcome, so the two medians are computed on differently
selected subsets and are not directly comparable. And the 118-configuration replicate is drawn
from the dissociation-retest sample, screened on rivalry only, which does not appear in Section
4.2's table. The "strong criterion" used below, monotonic rise to 1.00
with monotonically falling latency, is a post-hoc summary and is not in the pre-registration;
it is retained because it is the criterion the original analysis applied, and labelled as
unregistered here.

Five of the 30 selected configurations met that criterion: switch rate rising monotonically
to 1.00 at high goal strength with latency falling monotonically. The cleanest (α = 0.08, β = 0.25, κ = 0.04, λ = 0.15) rose
from 0.14 at *G* = 10%λ to 1.00 at 90%λ with latency falling from 366 to 144 timesteps in
the adaptation-dominated regime, against 0.00 at every level in the inhibition-dominated
one. Paired effect sizes against matched baseline rose from *d* = 0.34 to 0.52, declining
to 0.39 at 90%λ through ceiling compression.

**Regime contrast (Prediction 4, contradicted).** The registered 2 × 4 ANOVA on switch success
returned *F* = 0.033, *p* = .992 for the interaction and *F* = 1.26, *p* = .263 for the main
effect, and the regimes barely differ in whether they switch at all: 24% of
adaptation-dominated and 20% of inhibition-dominated regimes produced any switching across 100
randomly drawn configurations. On its own terms the prediction is null.

The registered logistic fit for Prediction 3 contradicts it more directly, and we report that
rather than resting on the ANOVA's null. Prediction 4 states that the goal pulse produces *no*
switches in inhibition-dominated regimes. It produces a reliable positive dose-response there,
median slope +1.105 with 10 of 13 fittable configurations positive and a pooled interval
excluding zero. Both regimes respond to goal strength and differ in slope by roughly threefold
per configuration, eightfold pooled in the replicate. **The predicted categorical difference is
graded**, which is what Section 4.1 concludes independently from the classification analysis:
the regime labels do not classify controllability.

We offered a mechanism for the ANOVA's null and withdrew it. The adaptation-to-leak ratio
classifies controllability in opposite directions between regimes (AUC 0.77 and 0.21), but the
regimes are constructed by multiplying α, β and κ, so within-regime AUC for that ratio is
computed on a truncated range of the same predictor. Under a median split on σ, orthogonal to
the multipliers, the reversal disappears (AUC 0.481 and 0.490).

**Persistence modulation ≠ input gain (Prediction 5, registered contrast null).** **The registered test is null, and the reason is
instructive.** It is a 2 × 2 ANOVA, mechanism by channel, on mean dominance duration at a
mean-matched increment amplitude across 98 configurations. Nothing reaches significance:
mechanism *F*(1, 97) = 2.00, *p* = .161; channel *F*(1, 97) = 2.71, *p* = .103; interaction
*F*(1, 97) = 1.65, *p* = .203. The cell means are large, 321 timesteps for persistence
modulation on the manipulated channel against 73 for the matched increment. The design is
within configuration, so the error term is the standard deviation of the paired differences,
not the marginal spread; those paired standard deviations are 913, 962 and 1,821 for the
mechanism, channel and interaction contrasts respectively, against mean effects of 130, 160 and
236. The differences are therefore genuinely inconsistent across configurations rather than
merely noisy in the margins, which is a stronger statement than a power failure: at *n* = 98
paired, an effect of this size would be detected if it were consistent.

The variance is not caused by outliers: only one configuration shows any winner-take-all.
Duration distributions under persistence modulation are heavy-tailed, so the mean is a poor
summary and an ANOVA on it is the wrong instrument. Two deviations follow and they disagree in
a way that settles the question. Excluding the single winner-take-all configuration gives
mechanism *F*(1, 96) = 10.10, *p* = .002, channel *F* = 9.18, *p* = .003, interaction
*F* = 5.26, *p* = .024. Taking logarithms instead, on all 98, gives mechanism *F*(1, 97) = 74.65
and channel *F* = 40.40, both *p* < .001, but **interaction *F* = 0.01, *p* = .914**.

The two mechanisms therefore differ in the magnitude of their effect, and in how it divides
between the channels additively, but not multiplicatively: on a log scale they scale both
channels by indistinguishable factors. The registered contrast is the interaction, so it is
**scale-dependent and not robust**, and we do not claim it. What survives on every scale is
that the mechanisms differ in magnitude and that both channels respond.

The finding we regard as substantive is not the registered contrast but the *sign* of
inter-channel coupling and its dependence on delivery timing, which is exploratory and
occupies most of what follows. A second signature we initially reported on the same
comparison, in alternation rate, did not survive the registered duration filter and is
withdrawn (Section 4.3.1).

**These null contrasts mask a large effect that a control reveals.** Sections 4.3 to 4.5
report it: whether a modulation is delivered while the competitor is winning determines
the sign of its effect on that competitor, established by manipulating delivery timing
and isolated from duty cycle by a yoked-replay control, and holding across four
architectural families.

**What does classify controllability.** Since the regime labels do not, we report what
does. Floor occupancy (Figure S1), the proportion of post-burn-in timesteps on which the suppressed
channel sits at exactly zero, separates controllable from uncontrollable configurations
at AUC 0.792 and 0.735 under the σ split, on configurations not used to develop the
measure. It is the only predictor tested that survives a stratification which does not
truncate its rival's range, and those figures are defensible; the
pre-registered regime split gives 0.78 and 0.99 but on a truncated range. A single
threshold does not transfer: fitted pooled at 0.002, it gives 87.0% accuracy against a
76.0% majority baseline in one regime and 80.0% against 80.0% in the other.

**It gates rather than grades.** Among the 44 configurations producing any switching,
floor occupancy predicts switch rate not at all (ρ = +0.092 [−0.231, +0.420], LOO
*R*² = −0.084), with median 0.045 among switching and 0.966 among non-switching
configurations. A step function on it beats a linear one (LOO *R*² +0.060 against
+0.006), the signature of a threshold, and adding it to the six log-parameters improves
on them alone (+0.166 against +0.112). What governs the gradient among switchers is
inhibition strength (log β, LOO *R*² +0.257, ρ = −0.610 [−0.776, −0.383]); adaptation
weight has no relationship across the natural parameter distribution (log α, ρ = +0.076,
*p* = .62), though holding everything else fixed and raising κ reduces switch rate from
1.00 to 0.00. Control therefore requires both that the suppressed representation survives
suppression and that competition is weak enough for the amplified residual to prevail,
and the two are dissociable.

**And it damps coupling.** Splitting at the median (0.385), the median absolute change in
competitor duration under the goal signal is 9.6% in the high-floor half against 24.3% in
the low-floor half, while a matched input increment gives 7.4% and 7.7%, both
consequences of *G* · *x* vanishing when *x* does. Section S10 shows the operative
quantity is *G* · *x*<sub>suppressed</sub> relative to the noise scale rather than
exact-zero occupancy, which has no variance under a softened rectifier while control
still fails.

![Figure S1](figures/figure1.pdf)

**Figure S1.** Classification of controllability. Area under the ROC curve for each candidate predictor of whether the goal pulse can produce a switch, by stratification (Section 4.1). The dotted line marks chance.

## S2 Modified Levelt propositions and duration variability

**Proposition I** (raising one eye's strength raises its predominance; Figure S2A) held in 29 of 30
configurations, median ρ = 1.00. Brascamp et al. (2015) describe the modern rendition as close
to tautological, and its recovery here is a structural property of competitive inhibition
rather than a discriminating test.

**Proposition II** (increasing the strength difference primarily increases the *stronger*
stimulus's duration), tested separately either side of equidominance on 100 configurations:
held in **93 of 100** [86.3%, 96.6%] where the varied stimulus was stronger, 60 of 100
[50.2%, 69.1%] where the fixed one was, and both sides in 57 of 100 against a 25% chance
baseline. The asymmetry is not a survivorship artefact of the extraction protocol, that fails
in the opposite direction, since configurations where the weaker-side test *succeeds* show
higher winner-take-all rates at extreme imbalance (0.546 against 0.050, Welch
*p* = 1.5 × 10⁻¹⁰) and eightfold fewer weak-channel episodes. It reflects how strongly a
configuration responds to imbalance at all: where large imbalance drives near-exclusive
dominance the stronger channel's duration rises steeply and the proposition is satisfied. Fewer
surviving episodes also means the weak channel's mean is estimated from a smaller
boundary-filtered sample, which is a different survivorship path rather than the absence of
one, and the two cannot be separated with these data. The weaker-side test therefore partly
measures strength-sensitivity, a limitation that applies equally to empirical implementations.

**Proposition III** (alternation rate peaks at equidominance; Figure S2C), by quadratic fit in
predominance: **25 of 30** showed a concave interior peak and **21 of 30** peaked within 0.15
of equidominance, median vertex 0.495 (IQR [0.460, 0.501]). The small leftward displacement is
reliable (Wilcoxon *p* = .005) and reflects the asymmetry of a sweep varying one eye only.

**Proposition IV** (raising strength in both eyes equally raises alternation rate) **was
violated in 97 of 100 configurations** [91.5%, 99.0%] in the opposite direction, median
ρ = −1.000, none increasing. Across four resamples the violated proportion ranged 0.97 to 1.00
with median ρ exactly −1.000 in every draw; winner-take-all was 0.0000 at every signal level;
and filtering at the 5-timestep boundary leaves it unchanged (29/30 decreasing, median
ρ = −0.994). Not a measurement artefact.

**That failure is a known property of the model class.** Shpiro et al. (2007) showed four
mutual-inhibition models disobey the proposition at low input strength, producing an
increasing-duration regime below the decreasing-duration one it describes; Curtu et al. (2008)
established the generality analytically; Seely and Chow (2011) enumerated biophysically
plausible resolutions; Platonov and Goossens (2013) demonstrated the low-strength reversal
empirically and reconciled contrast and coherence data with a saturating input transfer; and
Brascamp et al. (2015) accordingly qualify the proposition with a near-threshold clause.

What is specific to this model is that it violates the proposition **across the entire tested
range**, from *S* = 0.20 to 0.70 with no crossing into the decreasing-duration regime. The
steady-state relation explains why:

$$x = \frac{S}{\lambda + \beta + \alpha\kappa/\gamma}$$

Activation and adaptation scale linearly with *S* while σ is fixed, so signal-to-noise rises
monotonically with drive and there is no strength at which the boundary is crossed. The
relation is invariant to the semi-implicit update of Equation 2, since *x*(*t*+1) = *x*(*t*) at
steady state, so neither this argument nor the bound of Section S5 depends on that choice.

**Two remedies, one of them ours and one stronger.** Scaling σ with signal
(σ = σ₀ · *S*/0.5, holding the input drive's coefficient of variation constant) removes the
violation entirely: 0 of 30 configurations decreasing at any filtering threshold, 28 of 30
increasing unfiltered (median ρ = +0.955), but recovery weakens under filtering (median
ρ = +0.299 at 5 timesteps), because scaled noise also adds absolute noise at high signal and
inflates the transitional population, so part of the unfiltered recovery is flicker.
Signal-dependent noise is absent from Seely and Chow's list, which operates on the
deterministic dynamics, and is a route to the decreasing-duration regime that does not require
altering the input nonlinearity. The input nonlinearity itself **does not work here**: a
Naka-Rushton transfer (*n* = 2, half-saturation 0.30, normalised at *S* = 0.5) leaves 96 of 100
configurations violating, median ρ = −0.991 against −1.000. Whatever produces the transition in
the models Seely and Chow analysed does not transfer through the input mapping alone.

Section 4.7 reports a second remedy, arrived at while addressing a different failure and
stronger than this one: making adaptation superlinear in activation recovers the proposition in
**200 of 200 configurations at an exponent of 1.75 or above**, median ρ = +1.000 under the
registered filter against 0 of 200 and −1.000 at an exponent of 1. The exponent is fitted to
the phenomena rather than derived, so that section locates the defect rather than supplying a
corrected law. Unlike signal-dependent
noise it survives filtering undiminished and acts on the deterministic dynamics. The two
failures turn out to have the same cause.

**Summary.** The model recovers Propositions I and III, recovers II on the stronger-stimulus
side and partially overall, and sits entirely within the increasing-duration regime, for a
reason we identify, with two remedies demonstrated. Propositions II and IV are the two Brascamp
et al. describe as core; the model passes one partially and fails the other, and the failure is
repairable.

![Figure S2](figures/figure3.pdf)

**Figure S2.** Modified Levelt propositions, 30 configurations. (A) Dominance duration of the varied and fixed channels against the varied channel's input, relative to equidominance, median and interquartile range, log scale (Propositions I and II). (B) Ratio of the stronger to the weaker channel's duration slope above equidominance, log scale; values above zero satisfy Proposition II. (C) Alternation rate against predominance, individual levels and binned medians (Proposition III).

### Duration variability

Across the full grid of rivalry-producing configurations, 931 of 6,814 (13.7%, CI
[12.9%, 14.5%]) achieved a coefficient of variation in the target range [0.35,
0.65] and 1,719 (25.2%) in the wide range [0.25, 0.75], with a median CV of 0.563.
These are the non-circular statistics; figures computed on the 30 configurations
selected for CV proximity to 0.5 are guaranteed by the selection rule and are not
reported as support.

The choice between gamma and Weibull descriptions of dominance durations has an
existing literature which our comparison should engage with rather than treat as internal
(Brascamp et al., 2005; van Ee, 2009). Weibull distributions were preferred over gamma by AIC
in 74.9% of rivalry-producing configurations and maximum-likelihood and method-of-moments
estimates of the gamma shape parameter disagreed by more than a factor of two in
54.9%, validating the registered choice of CV as the primary variability metric.
Duration distributions are bimodal: a transitional population comprising 15% of
episodes with mean duration 1.5 timesteps and Weibull shape 1.83, and a sustained
population comprising 85% with mean 102.2 and shape 3.80. The aggregate CV of 0.491
matches the human-data target but reflects this mixture. The boundary at
approximately 5 timesteps motivates the secondary threshold used throughout, and
the transitional population is directly responsible for one withdrawn result
(Section 4.3.1).

## S3 Decomposition of the gate-timing contrast in full

Six schedules
separate the components (Tables S1 to S3, Figure 3B and 3C), all at matched amplitude and duty cycle, replayed at the same seed as
the live gate they derive from so the noise is identical. A zero-delay replay reproduces the
live condition to 0.00 percentage points, which is the validity check for the comparison. All
figures below are on the 156 of 200 configurations complete on every condition, and the mean
dominance episode in this sample is 30 timesteps.

Values below are from the schedule campaign, 156 configurations complete on every condition.

**Table S1.** Competitor response to six schedules at matched amplitude, duty cycle and dose.

| Schedule | Competitor |
|---|---|
| Continuous, dose-matched | −14.3% |
| Fixed-period bursts, 2 to 200 timesteps | −24.2% to −9.1%, never above |
| Episode-derived burst durations, uncorrelated with this trial (yoked) | +6.3% |
| The same burst and gap lengths in random order (shuffled) | +3.9% |
| Episode-derived, shifted by 1 / 2 / 3 whole episodes | +8.4% / +6.9% / +6.8% |
| **Live gate** | **+18.8%** |

The schedules are not nested, so the terms below are a chain rather than a partition of
independent effects, and they sum to the full span exactly.

**Table S2.** Decomposition of the difference between continuous and gated delivery. The terms are a chain rather than a partition and sum to the full span exactly.

| Step | Contrast | pp | 95% CI |
|---|---|---|---|
| Schedule statistics | yoked − continuous | **+20.5** | [+15.3, +23.8] |
| Ordering of the bursts | shuffled − yoked | −2.3 | [−5.0, +1.0] |
| Contingency with the present trial | live − shuffled | **+14.9** | [+11.4, +17.8] |
| | **full span, live − continuous** | **+33.1** | [+27.7, +36.5] |

Percentage points are reported rather than shares. Two of the three terms are substantial and one
is not distinguishable from zero, so shares would divide an ill-defined total between
non-independent quantities, which is the ratio-summary hazard of Section S9. The summary that
survives is two-part: **schedule statistics flip the sign, and contingency with the present
trial sets the size.** Everything with episode-derived duration statistics that does not track
this trial, whether yoked, shuffled or shifted by one, two or three episodes, lands between
+3.9% and +8.4%; the live gate sits eleven points above the top of that band. The shift series
is flat across one, two and three episodes, so what those conditions capture is not proximity
in time.

**Attrition, and what it does to these intervals.** Forty-four of 200 configurations are
incomplete. Tripling the run length to 60,000 timesteps recovers almost none of them, the live
condition going from 161 to 165 complete configurations, while every estimate above moves by
under a percentage point (live +18.5% against +18.6%, yoked +4.9% against +4.7%, continuous
−17.6% against −17.9%). The missing configurations are therefore not short of run length. A
missing value means that a condition left no measurable competitor episode in any of twelve
seeds, and for the live gate this happens in about 25 of 190 configurations: there the gated
increment does not lengthen the competitor but drives the attended channel to near-exclusive
dominance, a departure from the rivalry regime of the same kind Section 4.3.2 reports for the
anti-gated condition. These configurations are excluded from every gated sign count in this
paper, which should be read as conditional on the gate leaving alternation intact. Dropout is not random with respect to the effect: the
dropped configurations have a median floor occupancy of 0.494 against 0.340 for those retained,
and mean episodes of 134.5 timesteps against 30.2. Since the schedule-statistics term is
fourfold larger in low-floor configurations (below), the complete-case sample is enriched for
configurations in which the effect is large, and the intervals above should be read as
conditional on that.

**Both terms act on the attended channel, and contingency additionally changes how much
transfers.** Recording the attended channel alongside the competitor for each schedule
separates what each term does.

Values below are from the later two-channel campaign, which records both channels and admits
different numbers of configurations per condition, so its competitor values differ from the
table above by one to two percentage points.

**Table S3.** Attended and competitor response together, with their ratio, for the four core schedules.

| Schedule | Attended | Competitor | Ratio |
|---|---|---|---|
| Continuous, dose-matched | +17.8% | −17.9% | −1.01 |
| Yoked | +41.2% | +4.7% | +0.11 |
| Duration-shuffled | +40.4% | +3.9% | +0.10 |
| **Live gate** | **+63.0%** | **+18.6%** | **+0.30** |

The two columns move together and in the same order, which is what Section 4.3.2's
adaptation-recovery account requires: a schedule that lengthens the attended channel more gives
the competitor more time to de-adapt. A schedule with the right duration statistics but no
relationship to the present trial reaches two thirds of the live gate's attended lengthening,
and a correspondingly smaller competitor response. Attended lengthening is therefore doing
the work the account assigns it.

The ratio column shows that this is not the whole story. Contingency raises the
competitor-to-attended ratio from 0.11 to 0.30, so tracking the present trial does not merely
lengthen the attended channel further, it makes each unit of attended lengthening transfer
about three times as effectively. The natural reading is that a contingent gate extends
precisely those attended episodes during which the competitor is suppressed, giving it longer
uninterrupted de-adaptation, whereas a schedule with matching statistics but the wrong alignment
lengthens attended episodes without consistently extending the competitor's suppression windows.
We have not tested that reading, and record it as the obvious next probe.

**Neither intermittency nor timescale is sufficient.** Delivering the same dose at the same
amplitude and duty cycle as a regular square wave reproduces continuous delivery at every
period tested, from 2 to 200 timesteps, which is 0.07 to 6.7 mean episodes in this sample of fast-alternating configurations, with a best value of
−9.1% that never approaches the +6.3% of an irregular schedule. A fixed period matching the
mean episode length does no better than one far below it. What distinguishes the schedules that
work is that their burst durations are drawn from the dominance duration distribution and are
therefore highly variable; shuffling those durations into a random order changes nothing, so
the distribution is sufficient and its sequence is irrelevant.

**Four mechanisms proposed for the schedule-statistics term, and four withdrawn.** We
attributed it first to partial self-gating by the rectifier, a burst arriving while the
attended channel sits near the floor being absorbed. That predicts the term should grow with
floor occupancy; across 170 configurations it correlates at ρ = −0.777, and splitting at the
median gives +29.5 points in low-floor configurations against +6.8 in high-floor ones, a
fourfold difference in the direction the account forbids. We then attributed it to the
adaptation integration window, a schedule modulated faster than 1/γ being invisible to the
adaptation state, which predicts a threshold in burst duration near 1/γ that the fixed-period
series does not show at any duration. Ordering and phase are excluded by the shuffled and
episode-shifted conditions above.

The fourth was the most promising and is the most informative in failing. If alternation in the
operative regime is noise-induced escape (Section S6), dwell time is exponential in barrier
height and therefore convex in a delivered perturbation, so by Jensen's inequality a schedule
with variable burst durations should outperform a fixed one at the same mean. That predicts the
whole table: no variance in a fixed period, hence no gain at any period; shuffling irrelevant,
since the argument depends on the distribution and not the sequence; and the term vanishing
where the channel is floored and there is no barrier to modulate. It fails two tests.

Drawing burst durations from gamma distributions at fixed mean and matched duty cycle, and
sweeping the shape from near-deterministic to strongly skewed, the competitor's response does
rise with variability, from −23.0% at a coefficient of variation of 0.14 to −9.8% at 0.51
(ρ = +0.600 across the sweep). But it **saturates there and never becomes positive**, while the
yoked schedule, whose bursts are actual dominance episodes with a coefficient of variation near
0.5, reaches +5.8%. A schedule matched on mean, duty cycle and variability falls fifteen points
short of one drawn from the system's own episodes. Variability is necessary and not sufficient.

The single-burst response itself was measured by triggering one burst at the onset of every
fourth attended-dominance episode and comparing the episodes that received it with those that
did not in the same run, on 200 configurations. An earlier version, placed at a fixed time and
read as a run-wide mean, averaged one perturbed episode against roughly a hundred unperturbed
ones and returned a flat function; that result was a dilution artefact. Phase-locked, the burst
does move the following competitor episode, by +1.8%, +2.3%, +3.6%, +5.5% and +0.5% for bursts
of 5, 10, 20, 40 and 80 timesteps against a mean episode of 36. The response rises to a peak
near one episode's length and falls beyond it. It is not convex, so the fourth account stays
withdrawn on this evidence too. Even at its peak a single burst produces under a third of the
live gate's effect, so most of the effect accumulates across successive bursts rather than
superposing from independent ones. The containing attended episode is shortened slightly by
bursts shorter than an episode (−0.9% to −4.1%) and lengthened by longer ones (+6.0% and
+10.4%), consistent with a brief increment at onset hastening the attended channel's own
adaptation.

One prediction of the account holds, and it is the one we had nominated as the falsification.
The attended channel's own gain rises monotonically and steeply with burst variability, from
+2.2% to +17.5% across the same sweep (ρ = +0.943), more strongly than the competitor's response
does. Variability acts on the attended channel much as convexity would predict; what fails is
the transfer from that gain to the competitor. **That dissociation is the residue of this
investigation.** Variability in the schedule reliably lengthens the attended percept, only part
of that lengthening reaches the competitor, and the shortfall is largest exactly where the
schedule is synthetic rather than drawn from the system's own dynamics.

The schedule-statistics term therefore has no mechanism at the level of the schedule itself,
though its *route* is established: both it and the contingency term
act by lengthening the attended channel, and the competitor follows (see the table above). What
is unexplained is why a schedule's burst-duration distribution determines how much attended
lengthening it produces, given that mean, duty cycle, dose, ordering and phase are all matched.
Any successful account of that must not be a superposition of single-burst responses; must not depend on burst
ordering or on phase; must scale with variability in burst duration without being exhausted by
it; must distinguish a schedule drawn from the system's own episode distribution from a
synthetic one matched on mean and variability; and must vanish as the suppressed channel
approaches the floor. The fourth has been narrowed further. Our first synthetic schedules tied each gap
deterministically to the burst before it, which the yoked schedule does not. Drawing gaps
independently from their own gamma distribution at matched duty cycle leaves the competitor at
−21.1%, −10.5% and −10.1% for shape parameters of 50, 3.8 and 1.0, against +5.7% for yoked and
−12.8% for continuous delivery. The duration-shuffled schedule already permutes bursts and gaps
independently, drawing both from their empirical distributions, and matches yoked. So what
separates a working schedule from a synthetic one is not the joint structure of bursts and gaps
or their first two moments but the shape of the empirical duration distributions themselves.

## S4 Derivation of the threshold-geometry account

This section derives the sign rule of Section 4.4 and states which of its first-order
consequences hold in simulation and which do not.

### S4.1 Reduction to one slow variable

Write the adaptation imbalance as u = a<sub>A</sub> − a<sub>B</sub>. From Equation 2, in
continuous time,

$$\frac{du}{dt} = -\gamma u + \kappa\,(x_A - x_B).$$

When adaptation is slow relative to activation, the activations sit near one of two
quasi-stationary states: A dominant, with x<sub>A</sub> ≈ x<sub>A</sub><sup>+</sup> and
x<sub>B</sub> ≈ x<sub>B</sub><sup>−</sup>, or the mirror image. Within an episode u therefore
relaxes exponentially towards an asymptote, +U<sub>A</sub> = κ(x<sub>A</sub><sup>+</sup> −
x<sub>B</sub><sup>−</sup>)/γ during A's dominance and −U<sub>B</sub> during B's. A handover
occurs when u reaches a switching threshold: +c<sub>A</sub> for A to B and −c<sub>B</sub> for B
to A. In the oscillatory regime these are the points where the dominant state loses stability;
in the noise-driven regime they are the imbalances at which escape becomes likely, and we take
the median imbalance observed at handover.

An A episode runs from u = −c<sub>B</sub> towards +U<sub>A</sub> and ends at +c<sub>A</sub>; a B
episode runs from +c<sub>A</sub> towards −U<sub>B</sub> and ends at −c<sub>B</sub>. Hence

$$T_A = \frac{1}{\gamma}\ln\frac{U_A + c_B}{U_A - c_A}, \qquad
  T_B = \frac{1}{\gamma}\ln\frac{U_B + c_A}{U_B - c_B}.$$

The asymmetry that drives everything below is visible here. Each episode *ends* close to its
asymptote, where U − c is small and the logarithm is steep, and *starts* far from it, where
U + c is large and the logarithm is shallow.

### S4.2 An increment displaces a threshold, at the handover where it is present

An increment Δ on channel A's input that is present at an A-to-B handover makes A harder to
displace, so B needs a larger imbalance to take over: c<sub>A</sub> → c<sub>A</sub> + δ. One
present at a B-to-A handover makes A easier to restore: c<sub>B</sub> → c<sub>B</sub> − δ. To
first order δ ≈ Δ/α, since adaptation enters the accumulator with weight α and the extra input
must be matched by extra adaptation. What matters is whether the increment is present *at the
handover*, which is where the three schedules differ.

**Gated** (present only while A is dominant, so present at A-to-B handovers and absent at
B-to-A handovers). Only c<sub>A</sub> moves:

$$\Delta T_B = \frac{\delta}{\gamma\,(U_B + c_A)} > 0, \qquad
  \Delta T_A = \frac{\delta}{\gamma\,(U_A - c_A)} > 0.$$

The competitor's episode starts further from its own end point and lengthens. If the shift is large enough that c<sub>A</sub> + δ ≥ U<sub>A</sub>, the imbalance can no longer reach the handover threshold during A's dominance and, in the deterministic limit, A never hands over; with noise, handovers become rare. This predicts which configurations a gated increment removes from rivalry: those with δ large relative to the margin U − c. Section 4.4 tests it (AUC 0.91).

**Ungated** (present at both handovers). Both thresholds move:

$$\Delta T_B = \frac{\delta}{\gamma}\left[\frac{1}{U_B + c_A} - \frac{1}{U_B - c_B}\right] < 0.$$

The start of B's episode moves away from its end point, which lengthens it, and its end point
moves towards its start, which shortens it. Because the episode ends near the asymptote the
second effect is larger, so the competitor shortens.

**Anti-gated** (present only at B-to-A handovers). Only c<sub>B</sub> moves:

$$\Delta T_B = -\frac{\delta}{\gamma\,(U_B - c_B)}, \qquad
  \Delta T_A = -\frac{\delta}{\gamma\,(U_A + c_B)}.$$

Both channels shorten, the competitor by more.

**The sign rule.** A modulation present at the handover that begins the competitor's episode
lengthens it; one present at the handover that ends it shortens it, and the second effect is
the larger because episodes end near their asymptote. That is the whole of the derived claim.
It requires only that episodes are well described as travel between thresholds towards an
asymptote, which Section 4.4 tests directly.

### S4.3 First-order magnitudes

In the symmetric case, U<sub>A</sub> = U<sub>B</sub> = L and c<sub>A</sub> = c<sub>B</sub> = c,
so T<sub>A</sub> = T<sub>B</sub> and percentage changes stand in the same ratio as absolute ones.
Writing ρ = (L − c)/(L + c), the competitor-to-attended ratios are

**Table S4.** First-order competitor-to-attended ratios in the symmetric case.

| Schedule | Ratio | Range |
|---|---|---|
| Gated | ρ | between 0 and 1 |
| Ungated | −1 | |
| Anti-gated | 1/ρ | greater than 1 |

These are first-order statements. They neglect the increment's effect on the asymptote, since
an increment present during A's dominance raises x<sub>A</sub><sup>+</sup> and so U<sub>A</sub>;
they treat switching as deterministic; and they assume the two channels are symmetric. On 199 configurations from the eligible pool (Section 4.4, Table 5) the gated ratio follows ρ
closely, at Spearman +0.83 to +0.87 and 1.1 to 1.2 times ρ. The ungated ratio is about −0.45
rather than −1, and the anti-gated ratio is unrelated to 1/ρ. The asymmetry is expected. The
gated result depends only on the start threshold, where U + c is large and the logarithm nearly
linear, so the first-order expansion is accurate. The other two depend on the end threshold,
where U − c is small and the logarithm steep, so the expansion is least accurate there and
noise-driven switching, which the deterministic treatment ignores, matters most. The sign rule
does not depend on any of this, since it follows from the ordering U − c < U + c alone.

Because T = ln(1/ρ)/γ in the symmetric case, the gated ratio also satisfies ρ ≈ exp(−γT): it
can be written in terms of the mean dominance duration and the adaptation rate alone. Tested
directly, exp(−γT) computed from each configuration's adaptation rate and observed mean
duration predicts the gated ratio at Spearman +0.85 across 195 configurations, the observed ratio
at 1.17 times the prediction, nearly as well as ρ measured from the switching geometry itself.

### S4.4 Why Levelt's second proposition is unreachable, and why superlinear adaptation helps

Levelt's second proposition asks that raising one input leave that channel's duration nearly
unchanged while the competitor shortens, a ratio of large magnitude. A continuous increment is
the ungated case, which to first order trades the two channels off symmetrically, at a ratio
near −1. The proposition is therefore out of reach for this reason rather than for want of the
right parameters.

With adaptation superlinear in activation the adaptation drive is κx<sup>p</sup>, and an
increment that raises x<sub>A</sub><sup>+</sup> raises U<sub>A</sub> by a factor that grows with
p. That speeds the rise of u during A's dominance and cancels the lengthening of A's own
episode, while leaving the competitor's threshold displacement intact, so the magnitude of the
ungated ratio should grow with p. The gate-timing series of Section 4.7 shows exactly this: the
ungated ratio is −0.83, −1.18 and −2.24 at p = 1, 2 and 3.

### S4.5 The ramp-speed crossover

A gate that ramps with time constant τ delivers only part of the increment at the handover that
begins the competitor's episode, a fraction 1 − e<sup>−D/τ</sup> for an attended episode of
length D, and leaves a residue e<sup>−D/τ</sup> at the handover that ends it. Setting the two
first-order effects equal gives the crossover

$$\left(1 - e^{-D/\tau}\right)\rho = e^{-D/\tau}
  \quad\Longrightarrow\quad \frac{\tau}{D} = \frac{1}{\ln(1 + 1/\rho)}.$$

The crossover is therefore of order one, and it falls earlier where ρ is small, which is where
the gated ratio is small. Across the four architecture families of Section 4.5 the family with
by far the smallest gated ratio, the sigmoid family at 0.026, also has by far the earliest
crossover, 0.17, and the family with the largest ratio, input adaptation at 0.749, has the
latest, 0.70. On four points this is suggestive rather than established. Taking each family's gated ratio as
its ρ, the formula gives 0.87, 0.95, 0.27 and 1.18 against observed crossovers of 0.60, 0.45,
0.17 and 0.70: the right order and scale, overestimated by about half. The residue at the ending
handover is exactly the term the first-order analysis handles worst (Section S4.3), which is
the likely source.

### S4.6 Where the reduction's assumptions fail

The reduction treats adaptation as driven by a channel's own output, a<sub>i</sub> tracking
κx<sub>i</sub>, and as entering the accumulator subtractively, with a transfer close enough to
linear that an input increment shifts the switching threshold by about Δ/α. In the four
architectural families of Section 4.5 those assumptions hold for the subtractive and divisive
families, and the prediction ρ ≈ exp(−γT) holds there (Spearman +0.86 and +0.48, both beating a
permutation of γ within the family). They fail for the sigmoid family, whose saturating transfer
compresses the dominant channel's response to an increment and so the threshold shift, and for
the input-adaptation family, where adaptation acts on the input rather than being driven by the
output. The prediction fails in both (+0.55, *p* = .10; +0.16, *p* = .22). The sign rule, which
needs only the ordering U − c < U + c, is not affected by either and is established in three of
the four families.

### S4.7 Relation to earlier work

The reduction is the relaxation-oscillator geometry used to analyse release and escape in
mutually inhibitory pairs (Wang & Rinzel, 1992) and to derive Levelt's fourth proposition and
its violation in rivalry models (Shpiro et al., 2007; Curtu et al., 2008). Its application to
the timing of a modulation relative to the competitor's dominance, and the sign rule that
follows, is new.

## S5 One-parameter tests against published magnitudes

A stronger test is to fit the magnitude to one published number and predict a
second with no remaining degrees of freedom.

**Attention (Chong et al., 2005, Experiment 1).** Fitting the goal signal per
configuration to the observed 50% increase in the manipulated channel, the model
predicts a competitor change of **+8.6%**, IQR [−0.3%, +30.1%]; the observed value
is **+5.0%** (*t*(3) = 0.54, *p* = .63), inside the interval. Fitted goal strength
is a median of 0.35λ. Both intervals should be read together: the model's predicted
interval spans 30 percentage points and the observed value is a null at *n* = 4.
This is a consistency check rather than a prediction, an interval that wide would
accommodate many models.

**A derived bound on attentional effect size.** In **47 of 100** configurations the
model cannot reach a 50% increase at all, the maximum achievable being
approximately 45%. Neither the saturation ceiling nor the constraint on goal
strength is responsible: raising *x*<sub>max</sub> from 5 to 20 and relaxing the
constraint to 0.99λ leaves the count unchanged.

The bound is architectural. Persistence modulation removes at most λ from the
steady-state denominator λ + β + ακ/γ. The maximally boosted activation therefore exceeds
baseline by a factor (λ + β + ακ/γ)/(β + ακ/γ), so the gain relative to baseline is at most
λ / (β + ακ/γ), and the share of the boosted activation attributable to the goal signal is at most
λ / (λ + β + ακ/γ). We report the second. The two are monotone transforms of one another, so
every rank correlation below is identical under either. Earlier single draws gave ρ = +0.774
and +0.722; configurations that reach a 50% increase have a median
bound of 0.245 against 0.150 for those that do not. Across four independent draws of roughly 195 configurations the bound predicts the maximum
achievable duration increase at a median ρ of **+0.641**, range [+0.598, +0.676]. The composite
is not restating that high leak permits large modulation: **λ alone predicts nothing**, at a
median ρ of +0.034 with a range of [−0.067, +0.109] that straddles zero, while the partial
correlation of the composite controlling for λ is **+0.731**, range [+0.716, +0.736], higher
than the raw correlation. λ therefore acts as a suppressor rather than as the driver. The bound
carries a prediction: large attentional effects on dominance duration should
require high-leak, low-inhibition regimes.

**Withdrawn: the same expression does not govern the stimulus-strength arm.** We
initially reported that it did, at an identical ρ = +0.774. That does not
replicate. A purpose-built check returns ρ = −0.351 between the reciprocal of the
denominator and the attended-channel change under an increment. The two predictors
are not collinear (ρ = +0.611), so the coincidence of coefficients was not a
consequence of measuring the same thing; we cannot account for it and we withdraw
the claim. The bound applies to the attentional arm only.

The same check produced a negative correlation between the two outcome measures on one draw, which we reported as a possible parametric counterpart of the timing result and withdraw (Table S5).

**Stimulus strength (Levelt, 1965; Mueller & Blake, 1989).** Fitting the ungated
increment so that the competitor's duration falls by the observed 30%, the model
predicts a manipulated-channel change of **+38.9%**, IQR [+24.7%, +93.5%], on the
56 of 100 configurations for which the criterion is reachable. The observed value is
approximately 0%. **The model fails this test**, overshooting by the full size of
the effect it is asked to predict.

We expected the cause to be the linear input mapping: an increment large enough to
compress the competitor by 30% also raises the manipulated channel's own activation
substantially. **We tested that and it is wrong.** Replacing the linear map with a
Naka-Rushton transfer (*n* = 2, half-saturation 0.30, normalised so that *S* = 0.5
is unchanged) leaves the prediction at **+38.3%** against +39.4% for the linear
map, on 55 configurations. The saturating transfer that Platonov and Goossens
(2013) validated against contrast and coherence data does not close this gap in
this architecture, and we do not have an alternative account. The overshoot is
reported as an unexplained failure.

The same test bears on the security of the paper's main result. Under the saturating
transfer the model still violates Modified Proposition IV in 96 of 100
configurations against 97 of 100 under the linear map, so the input nonlinearity is
not the route out of that regime either. But the sign contrast is unaffected: the
goal signal gives +38.6% and +13.2% under the saturating transfer against +39.3%
and +12.9% under the linear one, and the increment gives +13.2% and −11.0% against
+15.0% and −10.0%. **The same-sign/opposite-sign result is not an artefact of the
input mapping.**

## S6 Attractor geometry, and which regime a configuration occupies

Setting σ = 0 and treating adaptation as a slow variable at steady state, we
computed nullclines and fixed points for three representative configurations. Under
the adaptation-dominated regime with a goal signal, all three showed two stable
fixed points: the symmetric rivalry state and a goal-biased attractor displaced off
the diagonal. Their coexistence is the mechanism of controlled switching, since a
transient pulse moves the system between basins without destroying the symmetric
attractor. Under the inhibition-dominated regime, nullcline intersection identified
four fixed points in all three configurations and **none was stable**, the
deterministic counterpart of the floor-occupancy result.

As goal strength increases the adaptation-dominated interior fixed point shifts
smoothly toward the biased diagonal through a pitchfork-like branching, while in the
inhibition-dominated regime the winner-take-all fixed points diverge and no stable
interior attractor appears at any goal strength tested. Adaptation was held at its
steady-state value, so this is a reduced two-dimensional analysis of a
four-dimensional system: every winner-take-all fixed point sits on the rectifier
boundary where the Jacobian is undefined, no fixed point has complex eigenvalues,
and limit cycles in the full system are not examined.

**The architecture spans two regimes, and the labelling above is wrong for one of
them.** The distinction we arrive at below is established in the literature and we should have
engaged with it from the outset. Moreno-Bote, Rinzel and Rubin (2007) characterised
noise-induced alternations in attractor-network bistability, which is the regime we identify
here; Braun and Mattia (2010) frame attractors and noise as twin drivers of multistability;
and Pastukhov et al. (2013) and Cao, Braun and Mattia (2014) map where in parameter space
adaptation-driven and noise-driven alternation each lives, which is exactly what the split
below measures. Our contribution is not the distinction but the possibility of deriving the
coupling sign from it, which Section 4.3.2 reports attempting and rejecting: the escape-rate argument predicts the schedule-statistics term should scale with burst-duration variance, and it does so for the attended channel but not for the competitor.

At σ = 0 the strongly adapting configurations do not alternate: they settle
onto a fixed point. The weakly adapting ones do, on a clean limit cycle, one
configuration with α = 0.03 gives a coefficient of variation of 0.06 at zero noise,
against a configuration with α = κ = 0.10 which produces no episodes at all. So the
stable interior fixed point we described above as "the symmetric rivalry state" is a
misnomer in the strongly adapting cells. A stable state in which both channels sit
at equal activation and nothing alternates is fusion, not rivalry.

What follows is that alternation in those cells is **noise-induced escape between
basins**, not adaptation-driven oscillation. Adaptation still matters, because it
deepens the basin the system has just left, but the transition itself is a
noise-driven escape and its dwell times are correspondingly near-exponential rather
than near-periodic.

That reading unifies several results reported separately elsewhere in this paper,
and we should have arrived at it earlier. Near-exponential dwell times are what
Section S2's mixture describes, and they are why maximum-likelihood and
method-of-moments estimates of the gamma shape disagree by more than a factor of two
in 54.9% of configurations: a likelihood fit chases the short-dwell tail that the
moments average away. Section S2's observation that signal-to-noise rises
monotonically with drive, and that the model therefore has no decreasing-duration
regime, is the escape-rate account of that confinement rather than a separate fact
about it. Section 4.1's finding that inhibition dominates the gradient while
adaptation does not is what basin geometry rather than adaptation timescale would
predict. Section S10's operative condition, that control fails once
*G* · *x*<sub>suppressed</sub> falls below the noise scale, is a statement about
escape thresholds, which is why it survives a change of rectifier that removes the
floor entirely.

This section identifies which regime a configuration
occupies rather than as a general dynamical account. The cells carrying Section 4.1 are in the escape regime; the nullcline geometry above describes the
alternating regime, which the weakly adapting cells occupy. Adaptation was held at
its steady-state value, so the analysis is a reduced treatment either way, and we do
not claim to have characterised the transition between the two.

## S7 Claims withdrawn, and why

This study reports more withdrawals than is usual, and we consolidate them here rather than
leaving a reader to accumulate them across the Results. Each is documented at the point it was
made; Table S5 is the index. Every one was withdrawn because a control we ran defeated it,
not because a reviewer objected, and in each case the control is named.

**Table S5.** Claims withdrawn in the course of this study, with the control that defeated each.

| Claim | Withdrawn because | Where |
|---|---|---|
| Alternation rate discriminates the five formulations | Applying the registered 5-timestep minimum to the transition count reverses the comparison, because the raw measure counts margin crossings from the transitional population | 4.3.1 |
| The gated and ungated coupling-ratio distributions are non-overlapping | Interquartile ranges do not establish disjoint ranges, and the full ranges intersect near zero | 4.3.5 |
| The model reproduces Chong et al.'s coupling ratio quantitatively | The ratio is 90% predictable from the model's own parameters, and Section 4.5 shows it is not architecture-invariant either | 4.3.5 |
| The anti-gated condition's −76.1% is a coupling magnitude | Its coefficient of variation reaches 1.17 against a baseline of 0.54, outside the window by which every configuration was admitted, so the condition has left the rivalry regime | 4.3.2 |
| One architectural expression governs both the attentional and the stimulus-strength arm | The correlation did not replicate under a purpose-built check | S5 |
| The adaptation-to-leak ratio classifies controllability in opposite directions between regimes | The regimes are constructed by multiplying that ratio's components, so within-regime AUC is computed on a truncated range of the same predictor; a stratification orthogonal to the construction shows no reversal | 4.1 |
| ρ = −0.267 is a parametric counterpart of the timing result | One unregistered draw with no interval | S5 |
| Contingency accounts for the majority of the live-versus-yoked difference | The three-way decomposition attributes it to schedule statistics, with phase not distinguishable from zero | S3 |
| The schedule-statistics term is partial self-gating by the rectifier | It should then grow with floor occupancy; it correlates at ρ = −0.777, the direction the account forbids | S3 |
| The schedule-statistics term is the adaptation integration window | Predicts a threshold in burst duration near 1/γ, which the fixed-period series does not show at any duration | S3 |
| The schedule-statistics term is burst ordering, or phase | Both excluded by the shuffled and episode-shifted conditions | S3 |
| The schedule-statistics term is convexity in burst duration | Variability at fixed mean does not reproduce the yoked value, and the phase-locked single-burst response is non-monotone in burst length rather than convex | S3 |
| The sign reversal is established in all four architectures | A reproducible rerun puts the sigmoid family's gated proportion at 26/40 [50%, 78%], so the reversal is not established there | 4.5 |
| The ramp-speed crossover is architecture-invariant | The reproducible rerun gives 0.60, 0.45 and 0.70 in three families and 0.17 in the sigmoid one | 4.5 |
| Timing determines the coupling sign for any modulation | It does not for modulations carrying persistent state; the operative variable is suppressed-phase activation | 5.2 |

Three of these were reported in earlier versions of this manuscript and are withdrawn here.
The remainder are accounts we proposed and tested in the course of this work; five of them
concern one quantity, the schedule-statistics term of Section 4.3.2, which we report without a
mechanism for exactly that reason.

We note the pattern rather than defend it. A study that proposes an account, tests it, and
withdraws it produces more retractions than one that proposes fewer accounts, and the count is
a measure of how much was checked rather than of how much was wrong. What survived is listed in
Section 7.

## S8 Why the coupling ratio cannot be recovered

Superlinear adaptation repairs both structural failures and displaces one quantitative
agreement: conditional on reproducing Levelt's second proposition, the fraction of
configurations also matching Chong et al.'s coupling ratio falls from four of six to one of
ninety-seven across the exponent range (Section 4.7).

Before asking why the agreement cannot be recovered, one qualification about the repair
itself. The exponent is not otherwise motivated, nothing in the architecture or the
physiology selects a value near 1.75 to 2, and we fit it to the phenomena rather than
deriving it, so the result locates where the defect lives and does not deliver a corrected
model. That one free parameter repairs two failures at once is suggestive of the common cause
identified above, but it is not evidence for the particular law.

Four mechanisms for recovering it were implemented and all four fail.

**Table S6.** Four mechanisms tested for recovering the attentional coupling ratio.

| Mechanism | Effect on the ratio | Why it fails |
|---|---|---|
| Continuous adaptation modulation | negative, wrong sign | falls on the increment's side of the coupling variable (Section 4.3.1) |
| Dominance-gated adaptation modulation | negative, wrong sign | gating does not move it, for the reason in Section 5.2 |
| Dominance-phase self-gain | unmoved at *p* = 2; CV falls to 0.06 | cancelled by the same superlinear adaptation that effects the repair, since both act on the dominant channel's activation |
| Inhibition-driven adaptation | passes through the observed 0.310 and on to +0.12, at *p* = 2 | but drives CV to 1.13 and the sign contrast from 196 of 200 to 6 |

The fourth is diagnostic rather than merely negative, and it answers the question. Blocking
the competitor's de-adaptation drives the ratio down *and* the sign contrast with it, because
the coupling and the ratio are one phenomenon at two magnitudes: the competitor lengthens
because it de-adapts, and the ratio measures how much. **This architecture cannot match the
quantitative ratio while producing the qualitative finding.** The constraint is structural,
not a defect awaiting repair. The first and third mechanisms also fail in opposite
directions, driving the coefficient of variation to 0.06 and to 1.13 respectively, where the
registered window is [0.35, 0.65].

Two failures do not exhaust the space of mechanisms, but they close the two obvious routes:
nothing acting on either channel's dominance-phase gain, or on the composition of adaptation,
will do it. A successor needs the competitor's return governed by something other than
recovery from its own adaptation, a separate slow variable, or alternation that is not
adaptation-driven. Both are outside this model class, which is the honest end point of this
line of work.

## S9 Measurement hazards and the Equation 2 discrepancy

Four hazards arose repeatedly in this study and each reversed a conclusion at least once. Three
are generic to simulation studies with per-configuration statistics, and we have written them up
separately, with the examples, as a standalone note rather than leaving them at the end of a
rivalry discussion where the readers who would use them will not find them. In summary: ratio
statistics should be computed per configuration and aggregated afterwards, since the two
conventions diverged at six points here and changed the conclusion at three; statistics should
be reported per stratum as well as pooled, since a predictor informative in both regimes in
opposite directions reports as chance when pooled; and duration filters must be applied to
transition counts and not only to durations, since the transitional population of Section S2
manufactured an alternation-rate difference that reversed under the registered threshold.

**The fourth is specific to this model and belongs here.** Equation 2 as originally written
specified adaptation driven by *x<sub>i</sub>*(*t*); the executed pipeline drives it from
*x<sub>i</sub>*(*t*+1), advancing activation and then adaptation from the new activation. The
equation has been corrected to match the code (Section 2.1), and no result was recomputed,
because every result here was generated with the code's version and the pipeline is internally
consistent throughout.

The discrepancy was found not by inspection but because an independent reimplementation built
from the published equations disagreed with the frozen grid on the coefficient of variation by
up to a factor of 4.6 while agreeing on mean dominance duration to within 10% for every
configuration with σ ≥ 0.10. That signature, agreement on the first moment and disagreement on
the second concentrated in the low-noise cells, localised the difference to the adaptation
feedback loop, and substituting the one line reproduced every stored digit.

The two forms differ at O(ακ). Quantified over every rivalry-producing configuration at 30 seeds
by 20,000 timesteps with identical seeds, the rank correlation between the two coefficients of
variation is +0.988 and the steady-state relation is identical under both, so Section S2's
account of the increasing-duration regime and the bound of Section S5 are untouched. What is
affected is the composition of the eligible pool: 931 configurations qualify under the pipeline
and 890 under the published form, of which 664 qualify under both, an overlap of 71.3%. The
divergence scales with the adaptation coupling as the O(ακ) argument predicts, the ratio of
median coefficients of variation rising from 1.01 to 1.25 across bins of increasing ακ.

**The central result is invariant to the choice, in selection and in simulation.** Rerunning the
gate-timing series entirely under the published form, with configurations screened that way and
simulated that way, gives the competitor lengthening in 176 of 187 configurations under gated
delivery and in 9 of 189 under ungated, against 182 of 188 and 6 of 189 under the pipeline form.
The discretisation affects which configurations are eligible and their coefficients of
variation; it does not affect the paper's central claim.

**The check that should have come first.** A comparison of moments cannot separate a difference
in the update from a difference in seeding or in the dominance threshold, which is why this took
several rounds to localise. The direct check is to run both implementations on identical
parameters and an identical seed and compare the traces. Doing so, with no manipulation applied
and the corrected update order, the maximum absolute difference between the two implementations
is zero at every one of 20,000 timesteps on both channels; the noise streams match as well as
the update rules. Any reimplementation used to audit a simulation should be compared
trace-against-trace at a shared seed before any statistic is computed from it.

The generalisable lesson is narrow but sharp. Discretisation choices that look like
implementation detail, such as whether a coupled variable is advanced from the old state or the
new one, can be the difference between two dynamical regimes, and they are invisible in the
steady-state algebra where such choices are usually checked. A model specification is not
reproducible unless the update order is stated. We now state it, along with initial conditions
and burn-in, none of which we had reported.

## S10 The rectifier is graded, not a knife edge

Table S7 reports the sweep. Replacing the half-wave rectifier with (1/*k*) log(1 + e^{*kx*}), which converges
to max(0, *x*) as *k* → ∞ and has *f*(0) = log 2 / *k*, the dissociation emerges
continuously and at every goal level tested.

This removes a confound in the original test, which used log(1 + e^{*x*}) alone and
thereby conflated removing the floor with injecting *f*(0) ≈ 0.693 of tonic
activation at every timestep. It also makes the architectural claim more robust:
the effect does not depend on exact-zero arithmetic and should generalise to any
competitive network with a saturating lower bound.

**Table S7.** Switch rate against rectifier sharpness, inhibition-dominated regime.

| *k* | *f*(0) | *G* = 25%λ | 50%λ | 75%λ | 90%λ |
|---|---|---|---|---|---|
| 1 | 0.693 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 0.347 | 0.03 | 0.13 | 0.19 | 0.34 |
| ≥ 5 | ≤ 0.139 | 0.00 | 0.00 | 0.00 | 0.00 |

(Values are switch rate in the inhibition-dominated regime.)

At *k* ≥ 5 the block is complete at every goal level. At *k* = 2 it is partial and goal-dependent, and at *k* = 1 the transition is graded rather than a block: control fails at low goal strength and succeeds at high, with *f*(0) = 0.693 meaning the suppressed channel never reaches zero.

Exact-zero activation is therefore sufficient but not necessary, and floor
occupancy is not the operative variable. The operative condition is that
*G* · *x*<sub>suppressed</sub> falls below the noise scale, which at *k* = 5 gives
a goal contribution near 0.014 against σ = 0.10. Floor occupancy is a proxy for it
that happens to be adequate under a hard rectifier and has zero variance under a
soft one; a scale-free version, *G* · *x*<sub>suppressed</sub>/σ, is the quantity a
confirmatory study should pre-specify.

## S11 Controls and ablations

Removing adaptation produced winner-take-all with zero switches. Removing
inhibition produced approximately 2,900 switches per run of rapid uncorrelated
fluctuation.

**The random goal control, corrected.** The pre-registration predicted that a
randomly varying goal signal would be less effective than a structured one; it was
not, reaching switch rates of 0.82 to 1.00. The original interpretation attributed
this to state-independence and was wrong: a goal signal drawn from Uniform(0, λ) is
still applied as *G* · *x*, and its mean of λ/2 makes it approximately sustained
modulation at 50%λ with multiplicative noise. Adding the missing control, a random
*additive* increment at matched mean, shows randomisation is nearly irrelevant, constant and random behave alike within each mechanism while the mechanisms differ
from each other. Random goal signals behave like structured ones because both
remain contingent on the channel's state, which is the same property the yoked
control isolates in Section 4.3.2.

## S12 Pre-registration audit

Every registered prediction is reported here regardless of outcome (Table S8), with the test
specified in the pre-registration, the sample it was run on, and the result.

**Table S8.** Pre-registration audit: every registered prediction with its test, sample and outcome.

| Prediction | Pre-registered test | Sample | Outcome |
|---|---|---|---|
| Model reproduces modified Levelt propositions | Per-proposition tests | see outcome | **Partial.** I recovered, 29/30 selected; II on the stronger-stimulus side, 93/100 random, both sides 57/100; III recovered, 21/30 selected; **IV violated, 97/100 random, direction reversed.** Note that the registered analysis tested the *original* propositions; the modified set was substituted post hoc when the error was found, and the equal-strength sweep needed for IV was added at that point |
| Duration variability near CV 0.5 | CV per configuration | full grid, 6,814 | **Partial.** The grid median is 0.563, but only 13.7% of configurations fall inside the registered window [0.35, 0.65], and CV was also an eligibility criterion, so the principal sample cannot test this |
| Goal pulse produces switches in adaptation-dominated regimes | Logistic dose-response | 30 selected | **Supported.** Registered logistic fit gives a median per-configuration slope of +3.264 in the adaptation-dominated regime with 12/12 positive, and +1.105 with 10/13 in the inhibition-dominated one (Section 4.1) |
| Goal pulse produces no switches in inhibition-dominated regimes | 2 × 4 ANOVA | 30 selected; classifier on 100 random | **Contradicted.** The registered ANOVA is null (*F* = 0.033, *p* = .992) and the regimes barely differ in whether they switch at all (24% against 20%), but the registered logistic fit for Prediction 3 shows a reliable positive dose-response in the inhibition-dominated regime, so the predicted categorical difference is graded rather than absent. A mechanism offered for the ANOVA's null was withdrawn under a control (Section 4.1) |
| Persistence modulation ≠ input gain | 2 × 2 ANOVA on duration | 100 random | **Null as registered** (interaction *F*(1, 97) = 1.65, *p* = .203), and uninformative on raw durations because duration distributions under persistence modulation are heavy-tailed. Two deviations are reported in Section 4.1 and disagree on the interaction, which is therefore scale-dependent and not claimed. The finding we regard as substantive is not the registered contrast but the sign of inter-channel coupling and its dependence on delivery timing (Section 4.3.2), which is exploratory. A second signature we reported on this comparison, in alternation rate, is withdrawn (Section 4.3.1) |

Exploratory analyses, labelled as such in the text: the gate-timing series and its
three controls; floor occupancy as a gating variable; the gate/gradient
decomposition; the rectifier sharpness sweep; the random additive control; the
noise-scaling test of Modified Proposition IV; the one-parameter fits; the attractor
analysis. The paper's central claim is among them.

Sample designations follow Section 3.1: *selected* denotes the 30 configurations
chosen for proximity of CV to 0.5, *random* the 100 drawn from the remaining
eligible pool. Where both were run, the random sample is primary.

**Withdrawn statistics.** Three quantities reported in earlier versions of this
manuscript are withdrawn here rather than silently dropped, and each is documented
where it was claimed: a difference in alternation rate between mechanisms, which
did not survive the registered duration filter (Section 4.3.1); a link between the
steady-state expression and the input-increment arm, which did not replicate under
a purpose-built check (Section S5); and a quantitative match to Chong et al.'s
competitor-to-attended ratio, which proved to be a statement about parameters
rather than mechanism (Section 4.3.5).

## S13 Corrections to the original pipeline

Four defects in the originally executed analysis were found and corrected before
the results below were generated. We report them because they affected
intermediate outputs.

*Seeding, twice.* The trial seed in the Phase II run was formed by adding offsets
for configuration, regime, goal level and seed. The offsets overlapped, so certain
cells shared noise streams systematically, the 50% goal level in one regime with
the 0% level in the other, among others. A replacement arithmetic scheme was
injective before reduction but exceeded the modulus by a factor of about nineteen,
so distinct cells could still collide after wraparound. All results below derive
trial seeds from `numpy.random.SeedSequence` applied to the index tuple, which has
neither property.

*Monotonicity.* The dose-response monotonicity test accepted any non-decreasing
sequence, which configurations that never switch satisfy vacuously. We report a
guarded test requiring at least one nonzero rate and a strict increase.

*Protocol.* The adaptation-gain sweep applied a sustained goal signal to one
channel from the first timestep and scored whether that channel subsequently
dominated. With no baseline period, no identification of the suppressed channel and
no pulse, it did not measure volitional switching. It is re-run here under the
registered transient protocol.

The simulation kernel was unaffected by any of these and is unchanged.

## S14 Statistical conventions

This is a simulation study and the epistemics of null-hypothesis testing differ
accordingly. Configurations are grid points we selected, not a sample from a
population, and seed counts are under our control, so *p* values index simulation
effort as much as evidence. We report bootstrap confidence intervals on every
correlation and effect size, Wilson intervals on every proportion, and the
proportion of configurations showing a predicted pattern as primary evidence.
Classification accuracies are reported against the majority-class baseline. Where
every configuration falls the same way, we report the count and the effect size and
omit the *p* value, which in that case indexes only the number of configurations
run.

Ratio-valued statistics are computed per configuration and aggregated afterwards.
Where the mean of per-configuration ratios and the ratio of pooled means diverge,
both are reported. This is not a formality: the two conventions disagreed at six
separate points in this study and altered the substantive conclusion at three of
them: the comparison with the published attentional coupling ratio, the
predominance contrast between mechanisms, and the slope ratios for Modified
Propositions II and IV. The other three, the duration preservation ratio, the
choice between coupling ratio and coupling magnitude, and the ramp-speed crossover in
Section 4.5, where the median of per-configuration crossings and the crossing of the
median curve differ by a factor of nearly two, diverged without changing what was
concluded.

Principal statistics are additionally computed on four independent random draws of
100 configurations from the eligible pool, and we report the range across draws
rather than only the bootstrap interval within one. The two sources of variation
differ, and for two statistics the first draw proved to be the most favourable of
the four.

Where a variable is measured within two parameter regimes, statistics are reported
**per regime** as well as pooled. Pooling regimes that differ in the distribution
of a predictor can reverse or eliminate a relationship present in both, and does so
here (Section 4.1).

**Convention.** Classifier performance (AUC) is reported as the **Median across
the four draws with the range in parentheses**. Correlations and effect sizes are
reported from the first draw with bootstrap intervals in square brackets, and their
across-draw range is given where it matters. Square brackets therefore always
denote a bootstrap interval and parentheses an across-draw range.

**Magnitude checking.** Every quantity entering a reported result was checked
against its expected order of magnitude before analysis. The errors caught during
this study were almost entirely of this kind rather than errors of logic: a
reference increment derived from mean dominance duration rather than mean
activation, and coupling tested against effect magnitude rather than the coupling
ratio. Both ran, looked interpretable, and were wrong by orders of magnitude or by
sign.

**Canonical run.** Section 4.3 and its subsections draw on separate simulation
campaigns run at 40 to 50 seeds per condition. Where the same quantity appears in
more than one, values differ by roughly one percentage point through Monte Carlo
variation: the goal signal at 50%λ gives +38.2%, +38.8% and +39.3% on the
manipulated channel across three campaigns, and +13.6%, +13.3% and +12.9% on the
competing one. **We designate the campaign reported in Section 4.3.2 as
canonical**, because it is the one that includes the gated, ungated, anti-gated and
yoked conditions together, and quote its values in the Abstract, the Contributions,
and Sections 4.3 and 5.1. Values from earlier campaigns are identified in place;
Section 4.3.5's table is one such, and its gated 2× row differs from the canonical
values by 0.3 and 0.7 percentage points.

