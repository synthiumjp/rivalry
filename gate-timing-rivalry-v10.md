# Why a dominance-gated contrast increment mimics attention in binocular rivalry
## Suppressed-phase activation, switching thresholds, and the sign of inter-channel coupling

**Jon-Paul Cacioli**<sup>1</sup> and **Chris Marmo**<sup>2</sup>

<sup>1</sup> Independent Researcher, Melbourne, Victoria, Australia. synthiumhotmail.com
<sup>2</sup> [AFFILIATION TO CONFIRM]

**Author contributions (CRediT).** J-PC: conceptualisation, methodology, software, formal
analysis, investigation, writing (original draft), writing (review and editing),
visualisation. CM: [TO SUPPLY: contribution roles. Select from the CRediT taxonomy and agree
with the co-author before submission. Resources and software are the likely categories if the
contribution was computational infrastructure.]

---

## Abstract

Chong, Tadin and Blake (2005) obtained the signature of endogenous attention in binocular
rivalry from a pure contrast increment, by applying it only while the attended stimulus was
dominant: the attended percept lengthened without the competing percept shortening, contrary
to Levelt's second proposition. Why a contrast change should mimic attention has not been
explained. In a goal-conditioned leaky competing accumulator we show that the sign of a
modulation's effect on the competing percept is set by how active the attended channel is while
the competitor is dominant, and we derive why. Reduced to one slow variable, the adaptation
imbalance travelling between two switching thresholds, a modulation present at the handover
that begins the competitor's episode lengthens it and one present at the handover that ends it
shortens it, the second effect being the larger because episodes end near their asymptote. The
reduction predicts episode durations from unmanipulated traces, every sign we observe, and the
size of the gated coupling ratio, which it identifies with how close to the adaptation
asymptote switches occur and which, where adaptation is driven by a channel's own output, it predicts from mean
dominance duration and adaptation rate alone. An increment delivered only during the attended channel's dominance
therefore lengthens the competitor's episodes where the same increment delivered continuously
shortens them. The contrast survives a duty-cycle-matched yoked control, an outcome criterion
that does not reference the attended channel, removal of the eligibility filter and a change of
update rule, and it is established in three of four architectural families. The same geometry
explains why the model cannot reproduce Levelt's second proposition, since a continuous
increment trades the two channels off almost symmetrically, and why making adaptation
superlinear repairs it. The account predicts that lengthening Chong et al.'s contrast ramp far
enough will invert the sign of the effect on the competing percept. Both pre-registered regime
contrasts were null and are reported as specified.

## 1. Introduction

When incompatible images are presented to the two eyes, perception alternates. One image
dominates, the other is suppressed, and dominance switches at irregular intervals while the
physical stimulus does not change. Whether an observer can control this alternation has been
disputed since Helmholtz claimed he could retain whichever image he chose and Breese reported
that he could not, and the dispute has not settled. Observers instructed to hold one rival
percept dominant generally cannot (Blake, 1988; Meng & Tong, 2004), and voluntary control is
weaker for rivalry than for other ambiguous figures (van Ee, van Dam, & Brouwer, 2005). Yet
endogenous attention prolongs dominance of attended stimuli (Chong, Tadin, & Blake, 2005),
biases the percept at onset (Mitchell, Stoner, & Reynolds, 2004), accelerates switching when
cued (Paffen, Alais, & Verstraten, 2006), and after prolonged training can produce
near-exclusive dominance (Dieter, Melnick, & Tadin, 2016). Rivalry ceases altogether when
attention is withdrawn (Zhang et al., 2011; Brascamp & Blake, 2012), so attention is
necessary for the phenomenon while being a poor instrument for steering it.

### 1.1 The gap

Two accounts identify different limiting quantities. Dieter and Tadin (2011) argued from
biased competition that the limit is the hierarchical level at which stimulus conflict
resolves: conflict persisting through many processing stages should be susceptible to
attentional bias, conflict resolved early in interocular interactions should not (see also
Stuit et al., 2014). Hugrass and Crewther (2012) identified a different quantity. Testing
volitional switching between apparent motion, drifting gratings and stationary gratings, they
found observers could generate intentional switches in the two motion conditions but not the
stationary one, verified objectively by corresponding reversals in optokinetic nystagmus
slow-phase direction. Their proposed mechanism was that the dorsal stream *maintains a
representation of the suppressed motion stimulus* which remains accessible to top-down
attention while excluded from awareness. On that account the limit is not where conflict
resolves but whether anything of the suppressed percept survives being suppressed. Two
features matter below: "maintained suppressed representation" has no measurable correlate,
and the dissociation is **categorical** rather than graded, so any formalisation should
predict a boundary.

Existing computational models supply neither, and one gap is worth stating precisely. Li,
Rankin, Rinzel, Carrasco and Heeger (2017) built a model in which rivalry arises from the
interaction of attentional modulation and mutual inhibition, reproducing the
attention-dependence of rivalry, the eye-swap phenomena and Levelt's propositions. **But
every one of its Levelt tests manipulates stimulus strength.** The model is never asked what
signature attention *itself* leaves on the two channels' durations, which is the quantity the
empirical literature disagrees about. Chong et al. (2005) report attention lengthening the
attended stimulus while leaving the unattended one unchanged, contrary to Levelt's second
proposition, though at *n* = 4 their effect on the unattended percept is not distinguishable
from zero and Section 5.3 states what that does and does not license us to claim; Hancock and Andrews (2007) report the reverse under cueing and call it analogous
to a contrast increase. Neither has a mechanism producing both. Section 4.3 shows the sign of
inter-channel coupling distinguishes them, and that what sets that sign is when the
modulation is delivered.

### 1.2 Why input gain is the wrong starting point

The intuitive formalisation of goal-directed attention in an accumulator model is
multiplicative gain on the input: replace *S<sub>i</sub>* with
(1 + *G<sub>i</sub>*)*S<sub>i</sub>*. Under the constant inputs that define steady-state
rivalry this reduces to an additive constant, so the attentional contribution does not depend
on the accumulator's state, and any effect it predicts is also predicted by increasing
stimulus strength, falling under Levelt's propositions rather than constituting a distinct
mechanism of control.

The scope is narrow. This does not apply to normalisation models of attention, where
attention enters both the excitatory and suppressive drive and therefore produces contrast or
response gain depending on the ratio of attention field to stimulus size (Reynolds & Heeger,
2009); under constant input, attention in such models is not equivalent to raising contrast.
Nor does it apply to time-varying inputs. It applies to the constant-input,
single-drive-term case, which is what a simple accumulator account of steady-state rivalry
would use. Section 4.5 shows the result we report does not depend on that exemption: it holds
in a divisive-normalisation architecture too.

### 1.3 Delivery timing, and what a state-dependent term approximates

We therefore locate the goal signal in the accumulator's dynamics rather than its input. In
the GC-LCA the goal signal reduces the effective leak of channel *i* from λ to
λ − *G<sub>i</sub>*, equivalent to adding *G<sub>i</sub>* · *x<sub>i</sub>*(*t*) to the
update: gated recurrent self-excitation amplifying whatever activation the channel holds.

The property that matters is not state-dependence as such. It is that
*G<sub>i</sub>* · *x<sub>i</sub>* is large while channel *i* is dominant and small while it is
suppressed, so the term approximates in continuous form what Chong et al. (2005) imposed by
hand as a binary gate. Section 4.3.1 shows state-dependence is the wrong abstraction, a
state-dependent modulation that keeps the attended channel *high* during suppression produces
the stimulus-strength signature, not the attentional one. What the goal signal shares with
Chong et al.'s gate is delivery timing.

Two predictions follow. **Coupling sign should follow delivery timing rather than
mechanism**: a modulation absent during the competitor's dominance leaves that competitor
facing less competition while it wins, so its episodes should lengthen, and this should hold
for an identical increment redelivered on different schedules (Section 4.3.2). **Control
should fail where the suppressed representation is annihilated**: because *G* · *x* is exactly
zero at the rectifier floor, control should succeed only where the suppressed channel retains
activation above it. This expresses the Hugrass and Crewther hypothesis as a computable
quantity. Section S10 qualifies the strong form, under a softened rectifier control fails
while activation never reaches zero, so the operative condition is that *G* · *x* falls below
the noise scale, making the prediction one of a steep transition rather than a strict
boundary.

### 1.4 Pre-registered predictions

Five predictions were registered before data were generated.<sup>1</sup> (1) With the goal
signal off, the model reproduces the modified Levelt propositions. (2) Dominance durations
show a coefficient of variation near 0.5. (3) A transient goal pulse to the suppressed
channel produces reliable switches in adaptation-dominated regimes, with switch probability
rising and latency falling as goal strength increases. (4) The same pulse produces no
switches in inhibition-dominated regimes. (5) Persistence modulation and mean-matched input
gain produce different rivalry signatures.

Predictions 2 and 3 were supported. Prediction 1 was supported for three of the four modified
propositions and violated for the fourth, for a reason we identify and repair. Prediction 4
failed as a regime contrast (*F* = 0.033, *p* = .992); we offered a mechanism for the null and
withdrew it under a control. Prediction 5's registered contrast, a mechanism by channel
interaction, was null. What that comparison did show
is a difference in the *sign* of the effect on the competing channel, which we did not
anticipate and which is exploratory. A separate
signature we initially reported on that comparison, a difference in alternation rate, did not
survive the registered duration filter and is withdrawn.

<sup>1</sup> Pre-registration: https://osf.io/d975z/ (March 2026, prior to data generation).
Code: https://github.com/synthiumjp/rivalry. All predictions are reported regardless of
outcome; analyses not specified in the pre-registration are labelled exploratory throughout.

Supplementary material, submitted with this paper, reports the pre-registered tests in full,
the decomposition of the gate-timing contrast and the mechanisms tested for it, the derivation
of Section 4.4, the one-parameter fits, the attractor analysis, the Equation 2 forensics, the
statistical conventions and the audit trail, including every claim withdrawn in the course of
the study (Table S5). Nothing reported in earlier versions has been removed; it has been moved.

## 2. The Model

### 2.1 Architecture

The GC-LCA comprises two mutually inhibitory accumulator channels, each receiving
a constant signal input and subject to noise, adaptation, and a half-wave
rectifier nonlinearity. It inherits the four core principles of the standard leaky
competing accumulator (Usher & McClelland, 2001) and adds a fifth,
goal-conditioned persistence modulation.

The activation of channel *i* at time *t* + 1 is

$$x_i(t+1) = \Big[(1 - \lambda + G_i)\,x_i(t) + S_i - \beta\,x_j(t) - \alpha\,a_i(t) + \eta_i(t)\Big]_0^{x_{max}} \tag{1}$$

where the bracket denotes clipping to [0, *x*<sub>max</sub>], *j* ≠ *i* indexes
the competing channel, and η*<sub>i</sub>*(*t*) ~ 𝒩(0, σ²) is independent Gaussian
noise. Each channel maintains a slow adaptation variable,

$$a_i(t+1) = (1 - \gamma)\,a_i(t) + \kappa\,x_i(t+1) \tag{2}$$

where γ is the adaptation decay rate and κ the adaptation gain, independent of α so
that the strength of adaptation-driven switching and the weight of adaptation in
the accumulator update can be varied separately.

**The update is semi-implicit and this matters.** Adaptation is driven by the
post-rectification activation at the *same* timestep, not the previous one: within
an iteration, *x* is advanced by Equation 1 and then *a* is advanced from the new
*x*. Physiologically this is the natural reading, adaptation tracks the activation
that has just occurred, but the choice is not innocuous. An earlier version of
this manuscript stated Equation 2 with *x<sub>i</sub>*(*t*) on the right-hand side,
which is not the update the reported results were generated with. The two forms
differ at O(ακ) in the adaptation feedback loop, and for strongly adapting
configurations they place the model in different dynamical regimes. Section S9
reports the size of the discrepancy and what it does and does not affect; the
deviations list records the correction.

The goal-conditioned gain must satisfy *G<sub>i</sub>* < λ so that the effective
leak remains positive, and all simulations enforce *G<sub>i</sub>* ≤ 0.95λ.
Negative values are permitted and are used in Section 4.3.4: they raise the
effective leak to λ + |*G<sub>i</sub>*| and are bounded only by the far weaker
requirement that the recurrent coefficient exceed −1. With *G<sub>i</sub>* = 0 and
κ absorbed into α, Equations 1 and 2 reduce to the standard LCA with adaptation.

The saturation ceiling *x*<sub>max</sub> = 5.0 was chosen to bound numerical
excursions. Its headroom is narrower than we initially stated. At equidominance the
steady-state relation gives a grid-wide median activation of 0.92 and a maximum of
2.65, so headroom is 1.9-fold. During dominance the inhibitory term drops out and
the maximum rises to 5.62, **above the ceiling**, which binds in 120 of 9,000 grid
configurations (1.3%). Ceiling occupancy is reported for every condition and does
not exceed 5.7% of timesteps in any configuration tested. We initially suspected
the ceiling of producing the model's limit on attentional effect size; Section
S5 shows it does not, and derives that limit from the steady-state denominator
instead.

In terms of effective leak, λ<sub>eff</sub> = λ − *G<sub>i</sub>*, so the added
contribution is *G<sub>i</sub>* · *x<sub>i</sub>*(*t*): zero at the rectifier
floor, proportional to the residual above it. A constant increment contributes
equally regardless of state and can therefore always overcome suppression given
sufficient magnitude.

### 2.2 Dominance extraction and derived measures

Both channels are initialised at *x* = 0.1 with zero adaptation, and the state is
recorded before each update, so the first recorded sample is the initial condition.
The first 500 timesteps of each run are discarded. Channel A is dominant when
*x<sub>A</sub>* − *x<sub>B</sub>* > θ with θ = 0.05, channel B when the reverse,
and intermediate periods are indeterminate. The first and last dominance episode
within each seed are excluded as truncated by the observation window.

The pre-registration specified a minimum-duration filter at the 5th percentile of
per-seed durations. In practice this evaluates to 1 timestep in 29 of 30
configurations and removes nothing. The duration distribution is bimodal, with a
transitional population of 1–2 step margin crossings and a sustained population
separated at approximately 5 timesteps (Section S2). Duration statistics are therefore reported at both thresholds wherever the choice matters. This is a documented
deviation, and it is not cosmetic: applying the 5-timestep threshold reverses one
comparison we initially reported as a result (Section 4.3.1).

Two quantities are computed on the full post-burn-in trace rather than on
boundary-excluded episodes: **Predominance**, the proportion of determinate time in
which a given channel is dominant, and **Alternation rate**, the number of
dominance transitions per timestep. Both are time integrals and are therefore
immune to the survivorship bias that boundary exclusion introduces as a
configuration approaches winner-take-all (Section S2). Alternation rate computed
on raw transitions counts margin crossings from the transitional population and is
reported only under the 5-timestep threshold.

**Floor occupancy** is the proportion of post-burn-in timesteps on which the
lower-activation channel is at exactly zero. It is measured during the baseline
period of each trial, before any goal signal is applied.

---

## 3. Method

### 3.1 Grid and selection

The full factorial grid comprised 4 values of λ (0.08–0.20), 6 of β (0.10–0.35), 5
of α (0.03–0.12), 5 of σ (0.04–0.12), 3 of γ (0.02–0.05), and 5 of κ
(0.50α–1.50α), yielding 9,000 configurations, each simulated for 30 seeds of
20,000 timesteps with *S<sub>A</sub>* = *S<sub>B</sub>* = 0.5. Eligibility
required three criteria: at least 10 dominance switches per seed on average
(rivalry-producing), ρ > 0.7 for predominance against stimulus strength
(Levelt-compliant), and a coefficient of variation of dominance durations within
[0.35, 0.65].

The pre-registration specified a coarse grid followed by a dense sweep. Faster than
anticipated computation allowed the full grid in a single run, which is greater
coverage than planned.

Two samples are used downstream and should not be confused. Thirty configurations
selected for proximity of CV to 0.5 constitute the original Phase II set. A further
**100 configurations were drawn at random from the 732 eligible configurations not
in that set**, and carry the principal analyses. Where the two samples give
different values we report both and treat the random sample as primary, since the
selected set is chosen on a criterion correlated with the outcomes.

### 3.2 Levelt testing

Modified Propositions II and IV are the two uniquely informative members of the
modified set; Modified Proposition I is close to tautological and Modified
Proposition III follows from Modified Proposition II (Brascamp, Klink, & Levelt,
2015). They require different sweeps and we run both.

**Asymmetric sweep.** *S<sub>A</sub>* ∈ {0.25, …, 0.75} with *S<sub>B</sub>* =
0.50 fixed, testing Modified Propositions I, II and III. Modified Proposition II
predicts a sign-dependent pattern about equidominance, so we fit separately on each
side, regressing each channel's mean duration on the absolute strength difference
and normalising by that channel's duration at equidominance.

**Equal-strength sweep.** *S<sub>A</sub>* = *S<sub>B</sub>* ∈ {0.20, …, 0.70},
testing Modified Proposition IV. This condition has no counterpart in the original
pre-registered design and was added when the original Levelt analysis was found to
test the original rather than the modified propositions.

Modified Proposition III predicts that alternation rate peaks at equidominance. We
fit a quadratic in predominance and test the location and concavity of the vertex,
following the plotting convention of Brascamp et al. (2015). A rank correlation
against unilateral stimulus strength cannot detect an inverted U and is not used.

### 3.3 Volitional control protocol

Each configuration was tested in two parameter regimes created by pre-registered
multipliers: high adaptation and low inhibition (α × 1.5, β × 0.6, κ × 1.5) and low
adaptation with high inhibition (α × 0.4, β × 1.5, κ × 0.4). We refer to them throughout as the adaptation-dominated and inhibition-dominated regimes.

Each trial ran 5,000 timesteps at *G* = 0, identified the currently suppressed
channel from mean activation over the preceding 200 timesteps, then applied a goal
pulse to that channel for 500 timesteps. A response window of 800 timesteps from
pulse onset was monitored for a switch, defined as the target channel sustaining
dominance for 50 consecutive timesteps. Floor and ceiling occupancy were logged
during the baseline period of every trial, so the gating variable is measured on
the same trials that produce the switch outcome.

### 3.4 Continuous-modulation protocol

Comparing mechanisms requires full duration distributions and therefore sustained
rather than transient intervention. For each configuration three conditions were
simulated with 100 seeds of 20,000 timesteps: baseline; persistence modulation at
*G* = 50%λ applied continuously to channel A; and mean-matched input gain replacing
the goal signal with a constant increment Δ*S* = *G* × *x̄*<sub>A</sub>, with
*x̄*<sub>A</sub> estimated from 20 baseline seeds per configuration.

### 3.5 Gate-timing protocol

The gate-timing series (Section 4.3.2) delivers one physical manipulation, an
additive increment on channel A's input, under three schedules:

- **Gated**: applied only while *x*<sub>A</sub> − *x*<sub>B</sub> > θ.
- **Ungated**: applied at every timestep.
- **Anti-gated**: applied only while *x*<sub>B</sub> − *x*<sub>A</sub> > θ.

Duty cycle and instantaneous amplitude cannot both be matched across schedules,
since gating necessarily reduces the fraction of timesteps on which the increment
is delivered. The gated and anti-gated schedules therefore run at 2× the
ungated amplitude, which brings total delivered dose to within 30% across the three
(1.08, 1.00 and 0.73 in units of the ungated total), and report the amplitude and
dose of every condition. Delivered dose is itself partly an outcome, a schedule
that lengthens the attended channel's episodes thereby delivers more, so dose is
reported for transparency and is not used as a control. The control on duty cycle
is yoked replay (Section 4.3.2).

**Yoked replay.** The increment is delivered on a schedule extracted from the
dominance time-course of a *different* run of the same configuration. This
preserves the marginal statistics of the schedule, duty cycle, episode-length
distribution, switching frequency, while destroying contingency with the present
trial's state.

**Absolute dominance criterion.** Because the difference criterion defines
dominance using both channels, an effect on the competitor measured under it is
partly definitional. Every condition is therefore recomputed under a criterion that
does not reference the attended channel: the competitor is dominant while
*x*<sub>B</sub> exceeds a fixed threshold. Both sets of values are reported and
labelled throughout.

### 3.6 Corrections, and statistical conventions

Four defects in the originally executed analysis were found and corrected before any result
below was generated: two seeding defects, a monotonicity test that configurations which never
switch satisfied vacuously, and an adaptation-gain sweep that did not implement the registered
transient protocol. All are documented in Section S13; the simulation kernel was unaffected by
any of them.

Statistical conventions are set out in Section S14 and used throughout. This is a
simulation study, so *p* values index simulation effort as much as evidence, and we report
bootstrap intervals on every correlation and effect size, Wilson intervals on every
proportion, and the proportion of configurations showing a predicted pattern as primary
evidence. Ratio-valued statistics are computed per configuration and aggregated afterwards,
with both conventions reported where they diverge, which happened at six points in this study
and altered the conclusion at three (Section S9). Principal statistics are computed on four
independent draws of 100 configurations with the range across draws reported. Statistics are
reported per regime as well as pooled wherever a design contains regimes. Section S14 also
designates the canonical simulation campaign and lists three statistics withdrawn from earlier
versions of this manuscript.

## 4. Results

Section 4.1 reports the pre-registered tests. Sections 4.3 to 4.5 establish what sets the
sign of inter-channel coupling, why, and how general it is. Sections 4.6 and 4.7 concern what
the model cannot do with stimulus strength, and one repair. Analyses beyond the pre-registration
are exploratory throughout.

### 4.1 Pre-registered hypotheses

Five predictions were registered (Section 1.4). Their registered tests are summarised here
and reported in full in Section S1. The results the paper rests on, in Sections 4.3 to 4.7, come
from analyses that were not registered.

Prediction 1, the modified Levelt propositions, is supported for three of the four; the fourth
fails for a reason Section 4.4 derives and Section 4.7 repairs. Prediction 2, a coefficient of
variation near 0.5, is supported at the grid median. Prediction 3, a logistic dose-response of
switch probability on goal strength, is supported: the median per-configuration slope is +3.264
in the adaptation-dominated regime, with all 12 fittable configurations positive, and +1.105 in
the inhibition-dominated regime, with 10 of 13 positive. Pooling across configurations flattens
these to +0.921 and +0.241, an aggregation effect discussed in Section S9.

Prediction 4, that the goal pulse produces no switches in inhibition-dominated regimes, is
contradicted. Its registered ANOVA is null (interaction *F* = 0.033, *p* = .992), but the
logistic fit above finds a reliable dose-response in that regime, so the predicted categorical
difference is graded. Prediction 5, that persistence modulation and a matched input gain produce
different signatures, is null as registered (mechanism by channel interaction *F*(1, 97) = 1.65,
*p* = .203). Durations under persistence modulation are heavy-tailed, and the interaction that
appears when the one winner-take-all configuration is excluded vanishes on a log scale, so we do
not claim it.

What does classify whether a configuration can be switched is floor occupancy, the proportion
of time the suppressed channel sits at zero: AUC 0.792 and 0.735 under a stratification that
does not truncate the comparison predictor (Figure S1). It gates control rather than grading
it, and it damps inter-channel coupling. The regime labels themselves do not classify
controllability.

### 4.2 Parameter space, and the pools used downstream

Of the 9,000 configurations in the grid, 6,814 produce rivalry, and their duration variability
spans a wide range around the registered eligibility window (Figure 1). Three configuration
pools appear in this paper under three sets of criteria (Table 1), and they are not
interchangeable. Every analysis states which it uses.

**Table 1.** Configuration pools used in this paper, with their criteria.

| Pool | *n* | Criteria | Used by |
|---|---|---|---|
| Registered eligible | **762** | rivalry (≥10 switches/seed), Levelt ρ > 0.7 over 11 signal levels at 8 seeds, CV ∈ [0.35, 0.65] | the canonical campaigns behind Tables 2 and 3, Sections 4.6 and S2, and the 100-configuration random draws throughout |
| Exponent-scan eligible | **946** at *p* = 1 | the same three, but Levelt ρ computed over 5 levels at 3 seeds, which is too coarse to discriminate (median ρ = 1.000) | every analysis run with the later pipeline: Sections 4.3.3, 4.4 and 4.7, the sign counts of Section 4.3.5, the window control of Section 4.3.1, and the bound of Section S5 |
| Update-order comparison | **931** | rivalry and CV ∈ [0.35, 0.65]; no Levelt criterion | Section S9 only |

The 946 pool is more permissive than the registered one. Most analyses added after the
pre-registered campaigns draw on it, and their results must not be pooled
with results from the 762. The 931 figure in Section S9 is the pipeline arm of a paired
comparison whose other arm gives 890; neither applies the Levelt criterion, so neither is
comparable to 762. Section 3.1 also refers to **732**, the registered eligible configurations
not in the 30-configuration selected set.

Of 9,000 configurations, 6,814 (75.7%) produced stable rivalry. Among these, 5,530
achieved ρ > 0.7 for predominance against stimulus strength. Applying all three
criteria yielded **762 eligible configurations** (8.5% of the grid); the principal
analyses draw 100 at random from the 732 of these not in the selected set.

![Figure 1](figures/figure2.pdf)

**Figure 1.** Duration variability across the parameter grid. Coefficient of variation of dominance durations for the 6,814 rivalry-producing configurations of 9,000. The shaded band is the registered eligibility window [0.35, 0.65] and the dashed line the median.

### 4.3 What sets the sign of inter-channel coupling

Under sustained modulation, on 100 randomly drawn configurations, a goal signal and
a matched input increment shift the two channels' dominance durations in opposite
directions. At |*G*| = 50%λ the goal signal gives +38.8% on the manipulated channel
and **+13.3%** on the competitor; the increment gives +13.6% and **−10.4%**.
Reversing the sign of the goal signal reverses its effect almost exactly (+38.2%
against −38.1% at 50%λ), the asymmetry at 90%λ arising only because positive
strength is bounded by the 0.95λ constraint and negative strength is not.

The pre-registered summary was a duration preservation ratio, which renders the
contrast as 1.14 against 0.89. Because both lie near unity the ratio invites the
reading that the competitor is "preserved" when it in fact lengthens by 14%, and we
report signed percentage change instead (Section S9). Duration statistics under
boundary exclusion are conditioned on rivalry having persisted; predominance is
computed on full traces. Winner-take-all reaches at most 12.8% under the goal
signal and ceiling occupancy at most 5.7%.

The sections that follow establish what governs the sign of that coupling. Section
4.3.1 shows that it tracks the attended channel's activation during the
*competitor's* dominance episodes, and states why that evidence cannot bear the
claim on its own. Section 4.3.2 sets the same quantity by design, through the
timing of an otherwise identical increment, and is the basis for the claim.

#### 4.3.1 What the coupling sign tracks, and why this is not the evidence

Five formulations of attentional modulation were implemented in the same architecture (Table 2, Figure 2) and
compared outcome-matched and paired, each tuned until the manipulated channel's duration
rose by 30%, on the 44 configurations where all five reach that target. Restricting to the
intersection matters: reachability ranges from 60 to 92 configurations across formulations
and correlates with the parameters that carry the coupling.

**Table 2.** Five formulations of attentional modulation, outcome-matched at +30% on the attended channel, ordered by the attended channel's activation during the competitor's dominance.

| Formulation | Suppressed-phase activation | Competitor | Same sign |
|---|---|---|---|
| Response gain | −19.8% | +12.2% | 35/44 |
| Persistence | −17.7% | +12.7% | 39/44 |
| Inhibitory output | −11.9% | +3.6% | 27/44 |
| Adaptation | +27.4% | −15.6% | 0/44 |
| Input increment | +47.2% | −24.9% | 0/44 |

Rows are ordered by the attended channel's mean activation during episodes in which the
*competitor* is dominant. That variable orders the competitor's response into three groups. It cannot separate the
top pair, which differ by 2.1 points on the predictor and 0.5 in the opposite direction on
the outcome, a gap within sampling variation. The relationship holds within configurations
as well as across formulations, at a median within-configuration ρ of −0.900 [−0.917,
−0.883] with the fitted slope negative in 100 of 100. Averaging the predictor over a fixed window from the onset of each competitor-dominance
episode, rather than over the whole episode, leaves the relationship unchanged: the relationship is monotone in **165 of 165**
configurations at whole-episode, first-10, first-25 and first-50 timesteps alike. We report the
count rather than a rank correlation, since four monotone points saturate the statistic and it
cannot then distinguish strong from perfect. The window-boundary dependence is therefore not what produces the
relationship, though the definitional dependence remains.

**This is not the evidence for the paper's claim, and we do not present it as such.** The
predictor is measured on the same traces as the outcome, and the extraction rule relates
them: dominance is defined as *x*<sub>A</sub> − *x*<sub>B</sub> > θ, so a competitor's
episode ends when *x*<sub>A</sub> rises relative to *x*<sub>B</sub>. Section 4.3.2 sets the
quantity by design instead.

What the ranking does establish is which property matters, and it rules out the one we
started from. Whether a modulation scales with current activation is not the operative
feature: an input increment keeps the attended channel high during suppression because it
is delivered regardless of state, and reducing the attended channel's adaptation does the
same by keeping it higher throughout, so both produce the stimulus-strength signature
despite one being state-dependent. "State-dependence" misclassifies adaptation modulation;
suppressed-phase activation does not. Paired against the input increment on the same 44
configurations the four alternatives give *d* = +2.91, +2.78, +2.52 and +2.26, each higher
in 44 of 44; inhibitory output is the weakest case on coupling sign (27/44, CI [47%, 74%])
and we exclude it from the claim.

We previously reported adaptation accumulated per unit activation (κ/γ) as the governing
predictor at ρ = +0.838. It is subsumed, controlling for suppressed-phase activation it
falls to +0.358, while the reverse holds at −0.797, but returns in Section 4.3.5 as the
strongest single predictor of coupling *magnitude* once timing has set its sign.

**Two claims withdrawn here.** Alternation rate does not discriminate the formulations:
applying the registered 5-timestep minimum to the transition count reverses the comparison
(persistence −7.3% against the increment −8.9%, from −11.9% and −4.2% raw), because the raw
measure counts margin crossings from the transitional population of Section S2. And
generality is narrower than the eligible pool suggests: on 100 configurations drawn from all
6,814 rivalry-producing ones, with no CV or Levelt criterion, the goal signal gives +5.9%
with the same sign in 61 of 100 (Wilson [51%, 70%]) while the increment gives −15.4% with
the same sign in 0 of 100. The increment's opposite-sign coupling is robust across the whole
grid; the goal signal's same-sign coupling is only weakly better than chance outside the
filter, which selects on a property set by the adaptation-noise balance that carries it.

![Figure 2](figures/figure4.pdf)

**Figure 2.** What the coupling sign tracks. (A) Competitor duration change against the change in the attended channel's activation during the competitor's dominance, for five formulations each swept over magnitude. (B) Median competitor change per formulation, ordered by median change in suppressed-phase activation. [TO SUPPLY: the source campaign labels formulations 1 to 5; state which is which, and confirm the ordering matches Table 2.]

#### 4.3.2 Gate timing as an independent variable, and three controls

We set the quantity of Section 4.3.1 by design rather than measuring it. One physical
manipulation, an additive increment on channel A's input, is delivered under three
schedules (Table 3, Figure 3A): only while A is dominant, at every timestep, and only while A is suppressed.

**Table 3.** One increment delivered under three schedules, medians across 100 configurations, under the difference criterion and under an absolute criterion that does not reference the attended channel. Only the sign and the contrasts in Control 1 should be read quantitatively; the absolute-criterion values are the conservative ones.

| Condition | Amplitude | Dose | Attended | Competitor, difference criterion | Competitor, absolute criterion |
|---|---|---|---|---|---|
| **Gated** (attended dominant) | 2× | 1.08 | +61.1% | +16.2% | **+3.8%** |
| **Ungated** | 1× | 1.00 | +14.4% | −10.4% | **−6.5%** |
| **Anti-gated** (attended suppressed) | 2× | 0.73 | −61.7% | −76.1%\* | −54.1%\* |

\*The anti-gated condition leaves the rivalry regime (coefficient of variation 1.17 against a
baseline of 0.54) and its value is a directional control, not a coupling magnitude; see below.

Medians across 100 configurations under the difference criterion. Amplitude and duty cycle
cannot both be matched, and delivered dose is itself partly an outcome, so it is reported for
transparency rather than as a control. The sign flip appears at matched amplitude too: at 1×
the gated schedule gives +9.5% on the competitor and the ungated −10.4%.

**The anti-gated direction is forced, and leaves the rivalry regime.** Raising the attended
channel while its competitor is winning must shorten that competitor's episode, so this
condition is architecturally guaranteed in direction and its function is only to establish
that the manipulation reaches the competitor. It also departs from the regime: both channels'
episodes shorten together and the coefficient of variation rises to 1.17 against a baseline of
0.54, outside the [0.35, 0.65] window by which every configuration here was admitted, while
filtered switch count does not rise (413 against 418), so most of the trace has become
indeterminate. Recomputing with exclusions logged excludes none of 200 and gives the
anti-gated 1× row as −30.5% attended and −54.2% competitor at a coefficient of variation of
0.829, against −69.1% and −85.1% at a coefficient of variation of 1.203 for 2×. The 2×
competitor value of −85.1% here and −76.1% in the table above come from different campaigns,
and we quote the canonical one in the table; the difference does not bear on the conclusion,
which is that the condition has left the rivalry regime at both amplitudes. **−76.1% is therefore not reported as a coupling magnitude.**

**The informative direction is not forced.** Delivering the increment only while the attended
channel is *already dominant* makes the competitor's episodes longer. Nothing in mutual
inhibition requires that, and it is the direction the empirical literature reports. It follows from
adaptation recovery: a lengthened attended episode gives the competitor more time to
de-adapt, so it returns stronger. That mechanism is confirmed by intervention rather than
asserted: adding a term so each channel adapts to the inhibition it receives keeps a
suppressed channel adapted, and drives the gated coupling ratio from +0.882 to +0.118 at an adaptation exponent of *p* = 2,
where that intervention was run; at *p* = 1.75 it reverses the ratio's sign. Section S8
reports what else it removes.

**Control 1: yoked replay.** Gating lengthens the attended channel's episodes by 61%, which by
itself gives the competitor more recovery time, so the control is not the ungated increment but
a schedule with the same duty cycle and no contingency, obtained by replaying a different run's
dominance time-course. Yoking abolishes the gated-versus-anti-gated separation entirely: 92.3
percentage points live, **−0.1** yoked, the two yoked schedules indistinguishable. Against its
own yoked control the live gate exceeds it by **+9.7 pp** at 1× and **+20.4 pp**
at 2× under the absolute criterion, at paired effect sizes of *d* = +1.61 and +1.72. Those
effect sizes are paired across the 100 configurations, not across observers, and their
denominator shrinks as seed count rises, so they index the consistency of the difference
across parameter space and not a subject-level magnitude. Yoked delivery is not inert, it gives +5.2%
and +5.3%, reaching 28% to 46% of the live effect across the exponent scan.

**Control 2: an outcome criterion not referencing the attended channel.** Redefining the
competitor's dominance as *x*<sub>B</sub> exceeding a fixed threshold removes the definitional
dependence between manipulation and outcome. Under it, gated 1× and 2× give +3.6% and +3.8%
against +9.5% and +16.2%; ungated 1× and 2× give −6.5% and −17.1% against −10.4% and −21.9%;
anti-gated 2× gives −54.1% against −76.1%. Attenuation is substantial and not uniform, the
gated arm retains 23–38% against 63–78% for the ungated, so much of the *magnitude* under the
difference criterion is definitional, and the gated arm's dose-response disappears. What
survives is the sign, and the contingency effect: the live-versus-yoked contrast is *larger*
under the absolute criterion (+20.4 pp against +11.0 pp at 2×), the opposite of what a
measurement artefact predicts.

**Control 3: the small-increment limit.** A relationship dominated by the threshold crossing
should diverge as the increment shrinks. Across a sixteenfold range the coupling ratio is
bounded and slowly varying: 0.38 to 0.26 gated, 2.63 to 1.23 anti-gated, −0.60 to −0.94
ungated.

**What remains.** The two channels are coupled by inhibition, so no manipulation of one is
fully independent of the other, and the definitional dependence of "the competitor is not dominant" on "the attended channel
is high" is reduced but not eliminated by the absolute
criterion. That bounds how strongly any claim of this kind can be made in a two-channel model.
What the series establishes is that an experimenter who fixes when an otherwise identical
increment is delivered fixes the sign of the coupling, in the predicted direction, and that
this requires the increment to be structured on the timescale of the alternation and to track
the present trial, not merely to be intermittent. Section 5.2 states the
scope condition: this holds for modulations that leave no trace when withdrawn, and not for
those carrying persistent state.

![Figure 3](figures/figure5.pdf)

**Figure 3.** Gate timing as an independent variable. (A) Competitor duration change for one increment under three schedules, under the difference and absolute criteria (Table 3). (B) Attended and competitor duration change for continuous, yoked, duration-shuffled and live-gated delivery at matched amplitude, duty cycle and dose (Table S3). (C) Competitor change for fixed-period burst schedules at matched dose against burst length, with the yoked and continuous values marked; no fixed period reaches the yoked value.

#### 4.3.3 What the difference between continuous and gated delivery is made of

Separating the components of the +33.1 percentage points between continuous and gated
delivery, at matched amplitude, duty cycle and dose and with the noise held identical, gives two
terms (Section S3, Tables S1 to S3). Intermittency as such contributes nothing: chopping the
increment into regular bursts at any period from 2 to 200 timesteps reproduces continuous
delivery. A schedule whose burst durations are drawn from the system's own dominance episodes,
but which does not track the present trial, contributes +20.5 points [+15.3, +23.8]; shuffling
its bursts into a random order changes nothing, and phase contributes nothing separable.
Contingency on the present trial contributes the remaining +14.9 [+11.4, +17.8].

Both terms act by lengthening the attended channel, and the competitor follows. Continuous,
yoked, shuffled and live-gated delivery lengthen the attended channel by +17.8%, +41.2%, +40.4%
and +63.0%, and change the competitor by −17.9%, +4.7%, +3.9% and +18.6%. That is what the
adaptation-recovery account of Section 4.3.2 requires. Contingency additionally raises the
competitor-to-attended ratio, from 0.11 to 0.30, so tracking the present trial makes each unit
of attended lengthening transfer more effectively.

Why the schedule-statistics term exists is not resolved. Rectifier self-gating, the adaptation
integration window, burst ordering, phase, convexity in burst duration and burst-gap coupling
were each tested and excluded (Sections S3 and S7). What remains is that the shape of the
empirical duration distribution matters beyond its mean and variability. In about 25 of 190
configurations the gate drives the attended channel to near-exclusive dominance and leaves no
measurable competitor episode. Those are excluded from every gated sign count in this paper, and
Section 4.4 shows the switching geometry predicts which configurations they are.

#### 4.3.4 Attentional suppression is the same mechanism with reversed sign

Several studies report *reductions* in the dominance durations of a stimulus that
was task-irrelevant or unattended, with the attended stimulus unaffected (Hancock &
Andrews, 2007; Paffen et al., 2008; Moreno-Sánchez et al., 2019). A model in which
the goal signal is constrained positive cannot address them. The constraint is not
required: positive goal signals must satisfy *G* < λ to keep effective leak
positive, whereas negative ones increase it to λ + |*G*| and are bounded only by
the far weaker requirement that the recurrent coefficient exceed −1. Setting
*G* < 0 needs no change to Equation 1, and the sign symmetry reported in Section
S5 shows it needs no new mechanism.

Paffen et al. (2008) provide the closest test, because their
irrelevant-versus-neutral pairing manipulated one stimulus through five days of
training while its rival was never presented during training. The irrelevant
stimulus's own duration fell (*t*(5) = −2.68, *p* = .02) while the neutral rival was
unaffected (*p* = .86). Their magnitude of roughly −20% lies between the model's
−20.6% at 25%λ and −38.1% at 50%λ. Their rival effect is null at *n* = 6, and the
model predicts −9.1% under a negative goal signal against +6.8% under a negative
increment; both lie within their interval, so the pairing constrains magnitude
without discriminating mechanism.

#### 4.3.5 Chong et al.'s gated-contrast manipulation, reproduced in direction

Chong, Tadin and Blake (2005, Experiment 3) doubled the contrast of the attended grating over
520 ms *whenever observers reported it dominant*, returning it to baseline as soon as
dominance switched away, and obtained the attentional signature rather than the Levelt
signature. Theirs is a physical, binary implementation of state-gating. We implemented it
directly (Table 4, Figure 4): an increment applied to channel A only while *x*<sub>A</sub> − *x*<sub>B</sub> > θ,
approached with a first-order lag of time constant τ standing in for their ramp. Values below
are medians across 100 configurations, from an earlier campaign than the canonical one
(Section S14); the gated 2× row differs from Section 4.3.2 by 0.3 and 0.7 percentage points.

**Table 4.** Chong et al.'s gated manipulation implemented, against their observed values.

| Condition | Manipulated | Competitor |
|---|---|---|
| Goal signal *G*·*x* at 50%λ | +38.8% | **+13.3%** |
| **Increment, gated, 1×, τ = 1** | **+34.4%** | **+9.5%** |
| Increment, gated, 2×, τ = 1 | +60.8% | +16.9% |
| Increment, ungated, 1× | +13.6% | −10.4% |
| Increment, ungated, 2× | +24.0% | −21.9% |
| *Chong Exp 3, observed* | *+29.0%* | *+9.0%* |

**The direction reproduces robustly.** Gated delivery gives a positive
competitor-to-attended ratio in 182 of 188 configurations at 1× and 161 of 162 at 2×; ungated
delivery gives a positive ratio in 6 of 189 and 5 of 166, with medians of +0.527 and +0.462
against −0.483 and −0.450. The denominators fall short of 200 because configurations in which
gated delivery leaves no measurable competitor episode are excluded; Section 4.3.3 shows those
are cases of near-exclusive attended dominance, not competitor responses of either sign, and
Section 4.4 that the switching geometry predicts them. Under the superlinear adaptation law of Section 4.7 the
separation is cleaner still, 200 of 200 against 8 of 200 at *p* = 2, and under the published
form of Equation 2 rather than the executed one it is unchanged, 176 of 187 against 9 of 189
(Section S9). We withdraw a stronger
claim made in an earlier draft that the two distributions are non-overlapping: they are not,
since at *p* = 1 the gated range runs from −0.022 to +1.083 and the ungated from −2.902 to
+0.274. The intersection is a tail artefact, a ratio is unstable wherever its denominator
passes near zero, but that is a reason to report sign counts rather than ranges, which is
what Section S9 recommends generally and what we should have done here.

**The magnitudes are not reproduced, and we withdraw the earlier claim that they were.** We
reported a match to Chong et al.'s ratio of 9/29 = 0.310. Across 100 configurations the gated
ratio spans 0.006 to 0.935 with a median of 0.522; only 6% fall within ±0.06 of the observed
value; and 90% of its variance is predictable from the six model parameters (leave-one-out
*R*² = 0.902, principally adaptation gain at ρ = +0.733 with inhibition at −0.486). The model
can produce almost any ratio in this range, so producing 0.310 constrains its parameters
rather than its mechanism. The comparison is also aggregation-dependent, the ratio of medians
gives 0.279 and the median of ratios 0.522, one of six occasions in this study where that
choice mattered (Section S9). Section 4.5 shows the ratio is not even
architecture-invariant. Timing sets the sign of the coupling and the dynamics set how much transfers, so an observed
ratio characterises the observer rather than testing the gating account. Section 4.4 identifies
which property of the dynamics it measures.

**A boundary condition on ramp speed, in usable units.** Sweeping the lag constant relative to
mean dominance duration locates a sign crossover at **τ / duration ≈ 0.59**. At the fastest
gate (0.028) the competitor changes by +5.6% with 18 of 20 configurations same-signed; at
0.139, +4.0% and 17 of 20; at 1.115, −2.8% and only 5 of 20. A ramp slow relative to episode
duration delivers part of the increment while the target is suppressed, and the gating is
lost. Section 4.5 finds the crossover at 0.45 to 0.70 in three further architectures but 0.17
in a fourth, so the value is architecture-dependent and only its order of magnitude is
structural.

**A falsifiable prediction the existing data do not address.** The gate must complete within a
fraction of a dominance episode for the attentional signature to appear, and lengthening it
inverts the sign. Chong et al.'s gate was contingent on observers' *reports*, which adds a
report latency of roughly 200–450 ms. A latency is a pure delay rather than part of the lag, but
Section 4.5 finds that a pure delay places the crossover almost exactly where a lag of the same
duration does, so the report latency can be added to the ramp: against episodes of two to three
seconds their gate sits at τ / duration of about 0.27 to 0.41, below the crossover in three of
four architectures. A ramp of 1.5 to 2.5 seconds places τ / duration between 0.6 and 1.2, beyond
the crossover in all four. The prediction is a **sign reversal in the competing percept's duration**, which is
the form this paper is best placed to make since the magnitudes are parameter-determined and
the sign is not, and the model fails if the competitor's change remains positive above
τ / duration of unity.

![Figure 4](figures/figure6.pdf)

**Figure 4.** Chong et al.'s gated manipulation reproduced in direction. (A) Competitor-to-attended ratio for gated and ungated delivery across configurations, with Chong et al.'s observed 0.310 marked. (B) Gated ratio against adaptation gain κ.

### 4.4 Why the sign reverses: switching-threshold geometry

The sign rule of Sections 4.3.1 to 4.3.4 can be derived rather than only observed. Write the
adaptation imbalance as *u* = *a*<sub>A</sub> − *a*<sub>B</sub>. When adaptation is slow
relative to activation, each dominance episode is the time *u* takes to travel from one
switching threshold to the other while relaxing towards an asymptote set by the dominant
channel's activation. This is the relaxation-oscillator geometry used to analyse release and
escape in mutually inhibitory pairs (Wang & Rinzel, 1992) and Levelt's fourth proposition in
rivalry models (Shpiro et al., 2007; Curtu et al., 2008). Every episode ends close to its
asymptote, where its duration is most sensitive to where the threshold sits, and starts far
from it, where it is least sensitive.

An increment on the attended channel matters at the handovers. Present when the attended
channel hands over, it raises the imbalance the competitor needs in order to take over, so the
competitor's episode starts further from its own end point. Present when the competitor hands
back, it lowers the imbalance the attended channel needs, so the competitor's episode ends
sooner. **A modulation present at the handover that begins the competitor's episode lengthens
it, and one present at the handover that ends it shortens it, by more.** A dominance-gated
increment is present only at the first; a continuous one at both, where the second wins; an
anti-gated one only at the second. That is the sign reversal of Table 3, derived from the
ordering of two distances along a single variable (Section S4). Table 5 tests the reduction
directly.

**Table 5.** The switching-threshold reduction against simulation, on configurations from the
eligible pool. The threshold *c* and asymptote *L* are measured from each unmanipulated trace,
and ρ = (*L* − *c*)/(*L* + *c*). Increments are 0.25 × λ × baseline activation; at 0.5 × the
counts are 170/187, 4/188 and 0/199 and the gated ratio correlates with ρ at +0.87, 1.13 ρ.

| Test | Prediction | Result |
|---|---|---|
| Episode duration from (*c*, *L*) | tracks observed | Spearman +0.89 on 199 configurations; predicted/observed 1.04 |
| Competitor under gated increment | lengthens | lengthens in 177/195 [86%, 94%] |
| Competitor under ungated increment | shortens | shortens in 182/195; lengthens in 13 [4%, 11%] |
| Competitor under anti-gated increment | shortens | shortens in 199/199; lengthens in 0 [0%, 2%] |
| Gated ratio | ρ | Spearman +0.83 with ρ; median 1.19 ρ |
| Ungated ratio | −1 | median −0.45 |
| Anti-gated ratio | 1/ρ | unrelated to 1/ρ (Spearman +0.14) |
| Gated ratio from duration and adaptation rate alone | exp(−γ*T*) | Spearman +0.85 on 195 configurations; median 1.17 exp(−γ*T*) |
| Configurations the gate drives out of rivalry | those with increment shift δ near margin *L* − *c* | AUC 0.91 for δ/(*L* − *c*); 4 exclusions rising to 12 as the increment doubles |

The reduction describes these dynamics. Switching thresholds and asymptotes measured from
unmanipulated traces predict mean episode duration at Spearman +0.89, and it gives the observed
sign under all three schedules. It also predicts the size of the gated coupling ratio, which it
was not constructed to do: across 199 configurations the ratio follows ρ at Spearman +0.83 and
sits at about 1.2 ρ. It does not predict the other two magnitudes. The ungated ratio is −0.45
rather than −1, and the anti-gated ratio bears no relation to 1/ρ.

That split is what the geometry leads one to expect. A gated increment moves only the threshold
at which the competitor's episode starts, far from the asymptote, where travel time is close to
linear in the threshold and a first-order expansion is accurate. Ungated and anti-gated
increments also move the threshold at which the episode ends, near the asymptote, where the
logarithm is steep, the expansion is least accurate and noise-driven switching matters most.

**What a measured coupling ratio estimates.** Section 4.3.5 withdraws the model's match to Chong
et al.'s ratio of 0.310 because the ratio is set by parameters. The reduction says which property
of the dynamics sets it: the gated ratio estimates ρ, the distance of the switching threshold
from the adaptation asymptote. On this reading their observers switched when the adaptation
imbalance had reached about half its asymptotic value, since *c*/*L* = (1 − ρ)/(1 + ρ) ≈ 0.53.
Because the reduction also gives *T* = ln(1/ρ)/γ, it implies ρ ≈ exp(−*T*/τ<sub>a</sub>) for
mean dominance duration *T* and adaptation time constant τ<sub>a</sub>, and the model bears this
out directly. Computed from each configuration's adaptation rate and observed mean dominance
duration, with no reference to the manipulation, exp(−γ*T*) predicts the gated coupling ratio at
Spearman +0.85 across 195 configurations, the observed ratio sitting at 1.17 times the
prediction (Table 5). Within this architecture most of that power comes from duration alone,
which predicts the ratio at +0.83, because γ takes only three values on the grid. Pairing each
configuration's own γ with its own *T* still matters: permuting γ across configurations drops
the correlation to a null median of +0.73 (95th percentile +0.77, *p* = .0002). Where γ varies
continuously, in the subtractive family of Section 4.5, the combination clearly outperforms both
of its parts, at +0.86 against +0.74 for duration and +0.65 for γ. A gated coupling ratio
measured in an observer should therefore be
predictable, before the attention experiment is run, from that observer's mean dominance
duration and an independent estimate of their adaptation time constant; for episodes of two to
three seconds, 0.310 implies τ<sub>a</sub> of about 1.7 to 2.6 seconds. The median ρ across our
eligible pool happens to be 0.316. We attach no weight to that, since the pool is not a model
of Chong et al.'s observers.

**The geometry also predicts which configurations leave rivalry.** A gated increment raises
the attended channel's handover threshold from *c* towards *c* + δ, with δ ≈ Δ/α. If that
reaches the asymptote the attended channel never hands over. The configurations excluded from
the gated counts should therefore be those in which δ is large relative to the margin *L* − *c*,
and they are. δ/(*L* − *c*) identifies them at AUC 0.91; doubling the increment triples the
exclusions, from 4 to 12; and at the larger increment all twelve have δ/(*L* − *c*) ≥ 0.5,
against none of the 114 configurations below it. The exclusion stated in Sections 4.3.3 and 4.3.5
is a consequence of the mechanism, not an unexplained loss.

Two consequences carry into what follows. A continuous increment trades the two channels off
almost symmetrically, which is why the model cannot reach Levelt's second proposition (Section
4.6), and an adaptation law that cancels the attended channel's own gain breaks that symmetry
(Section 4.7). And the ramp-speed crossover is of order one and falls earlier where ρ is small,
which is the pattern Section 4.5 finds across architectures.

### 4.5 The sign reversal across architectures

Every result above is obtained in one architecture: subtractive mutual inhibition with
adaptation subtracted from the accumulator and driven by each channel's own output. Whether
the gate-timing result belongs to that accumulator or to competitive networks with adaptation
generally determines what the central claim is about. The gate-timing series was therefore reimplemented (Figure 5) in three further architectures (Table 6), each a generic representative
of a family:
**B**, divisive, with the competitor entering the denominator of a normalised drive (Wilson,
2003; Li et al., 2017); **C**, subtractive with a sigmoid transfer on the net input (Laing &
Chow, 2002; Shpiro et al., 2007); and **D**, subtractive with adaptation acting
multiplicatively on the *input*, synaptic depression rather than spike-frequency adaptation.

Each family received an independent random parameter search over wide ranges, and a
configuration entered the test only if it produced rivalry on its own terms: at least ten
switches per seed, a coefficient of variation in [0.30, 0.70], and Levelt's first proposition
satisfied. Acceptance was decided before any gated condition was run. Forty configurations per
family, twelve seeds, 20,000 timesteps.

**Table 6.** The gate-timing series in four architectures, each a generic family representative. Deterministic seeds; 40 configurations per family, 12 seeds, 20,000 timesteps. The last two columns test the prediction of Section 4.4 within each family: the Spearman correlation of the gated coupling ratio with exp(−γ*T*), with the *p* value from permuting γ within the family, and the median ratio of observed to predicted.

| Family | yield | median CV | gated competitor | positive in | ungated competitor | positive in | exp(−γ*T*) vs gated ratio (*p*) | observed / predicted |
|---|---|---|---|---|---|---|---|---|
| A subtractive | 9.0% | 0.457 | **+8.7%** | 35/38 [79%, 97%] | −8.9% | 0/38 [0%, 9%] | +0.86 (.0005) | 1.06 |
| B divisive | 6.2% | 0.490 | **+4.4%** | 37/40 [80%, 97%] | −1.5% | 10/40 [14%, 40%] | +0.48 (.03) | 1.60 |
| C sigmoid | 1.2% | 0.543 | +1.6% | 26/40 [50%, 78%] | −18.3% | 2/40 [1%, 17%] | +0.55 (.10) | 0.22 |
| D input-adapt | 16.3% | 0.563 | **+2.5%** | 35/40 [74%, 95%] | −0.7% | 9/40 [12%, 38%] | +0.16 (.22) | 1.44 |

**The sign reversal is established in three of the four families and directionally consistent
in the fourth.** In the subtractive, divisive and input-adaptation families the gated
proportion's 95% interval lies above chance and the ungated proportion's lies below it, with no
overlap. In the sigmoid family the medians have opposite signs and ungated delivery shortens the
competitor in 38 of 40 configurations, but the gated proportion's interval reaches down to
chance, so the reversal is not established there. An earlier run of the same comparison, whose
seeding was not reproducible, reported the sigmoid family as clearing the bar; we report the
reproducible run.

Three of these bear on earlier claims. The divisive case is the one Section 1.2 exempts from the
input-gain argument: that exemption stands, and the gate-timing result does not depend on it.
The input-adaptation case changes the locus of adaptation rather than the form of suppression,
which is the more demanding variation, since the mechanism of Section 4.3.2 runs through the
competitor's de-adaptation and survives moving adaptation to the input. The sigmoid case is the
family whose analyses Section S2 relies on for the increasing-duration regime, and it is where
the result is weakest: gating lengthens the attended channel there by over 50% but transfers
very little of that to the competitor.

**The magnitudes do not generalise.** The gated coupling ratio is +0.463, +0.537, +0.026 and
+0.749 across the four families, a roughly thirtyfold spread. Section 4.3.5 withdraws the
quantitative match to Chong et al.'s 0.310 because the ratio is 90% predictable from the model's
own parameters; it is not architecture-invariant either, so a measured coupling ratio
constrains an architecture and its parameters jointly and cannot adjudicate between mechanisms.

**Nor does the ramp-speed crossover.** Setting the ramp time constant per configuration as a
fraction of that configuration's own mean duration and locating the downward zero crossing per
configuration gives medians of 0.60, 0.45 and 0.70 in the subtractive, divisive and
input-adaptation families, bracketing the 0.59 of Section 4.3.5, but 0.17 in the sigmoid family,
a spread of 4.1. Interquartile ranges within families are wide, [0.21, 1.27], [0.23, 0.86],
[0.07, 0.32] and [0.22, 1.19]. Replacing the first-order lag with a pure delay, which is what a
report-contingent gate adds, leaves the crossover essentially where it was: 0.59, 0.52, 0.18 and
0.60. The crossover is therefore a magnitude rather than a structural constant, and the
prediction of Section 4.3.5 is directional: lengthening the gate far enough inverts the sign,
somewhere between about a fifth and three-quarters of a dominance episode depending on the
architecture. A test should sweep τ / duration from about 0.1 to 1.5 rather than target a point.

**Where the prediction of Section 4.4 transfers.** Within the subtractive and divisive families,
exp(−γ*T*) predicts the gated coupling ratio at +0.86 and +0.48, both above the level reached by
permuting γ within the family, and with observed ratios at 1.06 and 1.60 times the prediction. It
does not transfer to the other two. In the sigmoid family the correlation of +0.55 does not beat
the permutation (*p* = .10) and the prediction overestimates the ratio fourfold; in the
input-adaptation family the correlation is +0.16 (*p* = .22). The division follows the
reduction's assumptions. It treats adaptation as driven by the channel's own output and
subtracted from its drive, with a transfer close to linear, which describes the subtractive and
divisive families; the sigmoid family saturates, and the input-adaptation family moves
adaptation to the input. Across families the medians put the extremes in the right order, the
sigmoid family lowest and the input-adaptation family highest, but swap the middle two, so the
spread in coupling ratio between architectures is partly accounted for rather than explained.

**Why the crossover is of order one.** A gate with time constant τ continues delivering for about
τ after the attended channel loses dominance, so the contaminated fraction of the competitor's
episode is of order τ / D, and gating is lost as that fraction approaches unity. That fixes the
scale, not the value, which is why the crossover lands between 0.17 and 0.70 rather than at a
single number.

**Caveats.** These are family representatives with parameters found by search, not
reimplementations of published models, so what is established is robustness to architectural
kind rather than reproduction of any author's parameter set; testing the published models
directly would be stronger. The sigmoid family rivalled in only 1.2% of draws at the ranges
used, so its forty configurations sample a narrow corner. The anti-gated condition's
coefficient of variation reaches 0.993 here but stays between 0.568 and 0.625 in the other
three, so the departure from the rivalry regime reported in Section 4.3.2 is specific to this
accumulator rather than general.

![Figure 5](figures/figure8.pdf)

**Figure 5.** The result is architectural. (A) Proportion of configurations in which the competitor lengthens under gated and ungated delivery in four architectural families, with 95% Wilson intervals; the dotted line marks chance. (B) Median gated coupling ratio per family. (C) Median ramp-speed crossover in τ / D per family, with interquartile ranges.

### 4.6 The model cannot reproduce Levelt's second proposition

Against the modified Levelt propositions the model does well on three (Section S2, Figure
S2). Raising one channel's strength raises its predominance in 29 of 30 configurations, the
stronger stimulus's duration rises more steeply in 93 of 100, and alternation rate peaks near
equidominance in 25 of 30. It fails the fourth: raising both inputs equally *lowers* alternation
rate in 97 of 100 configurations, across the whole range tested, so the model sits entirely in
the increasing-duration regime that mutual-inhibition models show at low input strength (Shpiro
et al., 2007; Curtu et al., 2008). Signal-dependent noise removes that violation; a saturating
input transfer does not. The second failure concerns the stimulus-strength side of Chong et
al.'s contrast directly.

The input increment is one arm of the contrast above, and it is quantitatively
wrong. Section S5 reports that fitting it to the observed 30% compression of the
competitor predicts a 39% increase in the manipulated channel where approximately
0% is observed. We searched for configurations in which an ungated
increment reproduces the canonical phenomenology directly, defined as a
manipulated-channel change within ±10% and a competitor change between −15% and
−45%, at any increment magnitude in the swept range.

**There are none.** Across 100 configurations and six magnitudes each, not one
combination reproduces Levelt's second proposition. The failure is structural
rather than a matter of scaling, and it is not remedied by a saturating input
transfer (Section S2). Section 4.7 re-runs the search on 200 configurations from a
more permissive pool and finds three, a rate of 1.5% whose interval overlaps the
zero found here; the two are consistent and the section states the arithmetic.

The proposition requires the competitor-to-attended ratio of duration changes to
diverge: the manipulated stimulus roughly unchanged while its competitor is
substantially compressed. In the
model that ratio is centred at **−0.475** with an interquartile range of [−0.781, −0.215], so
the two channels trade off close to symmetrically, not tightly, since that range spans a
factor of 3.6, but around a value near −0.5 rather than diverging. We initially framed this as a bound on the ratio, requiring
an effectively unbounded value. That framing is wrong and invites the objection that our own
criterion, attended within ±10% and competitor in [−15%, −45%], is satisfied by |ratio| between
roughly 1.5 and 4.5, a range the model does reach: the largest |ratio| observed anywhere in the
sweep is 3.15. **The failure is a joint miss in two dimensions, not a ratio bound.** No
configuration lands inside the box, because the configurations achieving a large |ratio| do so
by compressing the competitor far beyond −45% or by moving the manipulated channel well outside
±10%, not by holding one near zero while the other falls into range.

We do not have a derivation of why the ratio takes this value. The natural
suggestion is that adaptation, being proportional to activation, scales with the
increment-induced gain rather than cancelling it, so that the two channels trade
off symmetrically; but we have not shown that this follows from the model
equations, and an attempt to relate the increment arm to the steady-state
expression λ/(λ + β + ακ/γ) did not replicate (Section S5). The failure is
robustly established and unexplained. Together with the model occupying only the
increasing-duration regime, it locates a systematic defect in how the architecture
converts input change into dominance dynamics, and Section S8 returns to it.

**Why this does not undermine Section 4.3.2.** The obvious worry is that the
central contrast has one broken arm, so that the coupling result reflects the
defect rather than the modulations. It does not, for a structural reason: the
causal series in Section 4.3.2 contains no goal signal and no comparison between
mechanisms. It is one increment delivered under three schedules. A defect in how
the model converts input change into duration change applies identically to all
three conditions and cannot generate the ordering between them. Separately, and
more weakly, adaptation modulation, which is not an input increment, falls on the
increment's side of the coupling variable exactly as Section 4.3.1 predicts.

### 4.7 A superlinear adaptation law repairs both structural failures

Section 4.6 reports that no configuration reproduces Levelt's second proposition at any
increment magnitude, and attributes it to the adaptation law: adaptation proportional to
activation scales with the increment-induced gain rather than cancelling it. That diagnosis
predicts a repair. Replacing κ·*x* in Equation 2 with κ′·*x*<sup>*p*</sup>, renormalising
κ′ = κ / *x̄*<sup>*p*−1</sup> so the baseline adaptation drive is matched, varies the
curvature of the law without changing its level. Eligibility was re-derived under each
exponent rather than inherited (Table 7, Figure 6A and 6B): eligible counts rise from 946 of 6,814 at *p* = 1 to 1,655 at
*p* = 3, while the median CV *of that pool* falls from 0.556 at *p* = 1 to 0.443 before
recovering, distinct from the grid-wide median of 0.563 in Section S2, since superlinear
adaptation makes durations more regular and draws the grid's high-CV tail into the registered
window. Rivalry is retained in 200 of 200 sampled configurations at every exponent.

Proposition IV figures below are from 30 seeds × 20,000 timesteps and are unchanged from a
10-seed run to within three configurations in any cell.

**Table 7.** Levelt's second proposition and Modified Proposition IV against the adaptation exponent.

| *p* | Levelt II reachable | Wilson 95% | Prop IV recovered (filtered) | median ρ |
|---|---|---|---|---|
| 1.00 | 3 / 200 | [0.5%, 4.3%] | 0 / 200 | −1.000 |
| 1.25 | 6 / 200 | [1.4%, 6.4%] | 108 / 200 | +0.086 |
| 1.50 | 9 / 200 | [2.4%, 8.3%] | 194 / 200 | +0.943 |
| 1.75 | 20 / 200 | [6.6%, 14.9%] | **200 / 200** | **+1.000** |
| 2.00 | 44 / 200 | [16.8%, 28.2%] | 200 / 200 | +1.000 |
| 2.50 | 78 / 200 | [32.5%, 45.9%] | 200 / 200 | +1.000 |
| 3.00 | 91 / 200 | [38.7%, 52.4%] | 200 / 200 | +1.000 |

**Both failures are repaired by the same change.** Levelt II reachability is monotone in *p*
with the interval at *p* = 1 disjoint from those at *p* ≥ 1.75. Modified Proposition IV
recovers completely from *p* = 1.75 under the registered 5-timestep filter, against 0 of 200
at *p* = 1, which reproduces Section S2. The mechanism is the one predicted: at matched
increment the attended channel's own gain falls from +41.2% at *p* = 1 to +11.8% at *p* = 3 as
adaptation increasingly cancels the drive increase, while inhibition continues to compress the
competitor. Against the remedy Section S2 demonstrates this one is markedly stronger, signal-dependent noise recovers Proposition IV in 28 of 30 configurations unfiltered but falls
to a median ρ of +0.299 under the same filter, where the superlinear law holds at +1.000, and
it acts on the deterministic dynamics, where the existing literature's modifications live
(Seely & Chow, 2011). The raw transition count also rises with *p*, from 0 to 138 of 200,
though it diverges from the filtered count between *p* = 1.25 and 2.5: higher drive widens
channel separation and suppresses the margin crossings of Section S2, so raw counts fall,
while superlinear adaptation shortens sustained episodes, so filtered counts rise. The
filtered measure is the one corresponding to perceptual alternation, for the reason Section
4.3.1 gives when withdrawing a claim built on raw counts.

The three hits at *p* = 1 are consistent with the null in Section 4.6, obtained on 100
configurations: the Wilson intervals overlap and a true rate of 1.5% yields no hits in 100
draws with probability 0.22. The pool used here is also more permissive (see the deviation
note below).

**The gate-timing result survives the repair and strengthens.** The series was repeated at three exponents (Table 8).

**Table 8.** The gate-timing series under each adaptation exponent.

| *p* | gated 1× | gated 2× | ungated 1× | competitor > 0, gated 2× | live − yoked, difference criterion |
|---|---|---|---|---|---|
| 1.0 | +10.7% | +17.1% | −14.1% | 158/170 | 10.5 pp |
| 2.0 | +16.2% | +26.3% | −8.5% | 195/195 | 14.2 pp |
| 3.0 | +14.2% | +27.2% | −7.4% | 200/200 | 19.7 pp |

The sign contrast is total at *p* ≥ 2 in both directions, the competitor lengthens under
gated delivery in 200 of 200 and shortens under ungated in 194 of 200, it survives the
registered filter with the filtered effect *growing* with the exponent (+4.8%, +7.8%, +9.3%
at 1×), and the live-versus-yoked separation grows likewise. In the regime where the
stimulus-strength arm is quantitatively correct, the coupling result is stronger than where it
is not, which answers the concern of Section 4.6 more directly than the structural argument
given there.

**What the repair costs, and how much of the space satisfies everything.** The gated coupling
ratio moves with the exponent, from 0.318 pooled at *p* = 1 to 0.746 at *p* = 2 and 1.502 at
*p* = 3, while Chong et al.'s observed 0.310 sits on *p* = 1. Conditional on
reproducing Levelt II, the fraction of configurations also matching the ratio within ±0.10
declines monotonically: 4/6, 2/5, 4/13, 3/18, 3/31, 1/71, 1/97 across the exponents above,
with Wilson intervals of [20%, 81%] at *p* = 1 and [0.2%, 5.6%] at *p* = 3 that do not
overlap. The count satisfying both never exceeds 4 of 200 and does not grow as reachability
rises sixteenfold; the marginal count near the observed ratio falls from 23 to 8. No
configuration satisfies both at every exponent and only two of eighteen recur at any second
exponent, so the set is largely reconstituted as the exponent moves. Taking all four criteria
together, eligibility, Proposition IV, Levelt II, and the ratio, **one configuration in 200
satisfies all four**, at an exponent between 1.75 and 2. A Wilson interval on 1 of 200 is
[0.05%, 2.8%], so the honest statement is not that a solution exists but that **no
configuration robustly satisfies all four**, and the count at the ±0.05 and ±0.15 tolerances
is 0 and 1 respectively. We report the count for completeness and draw no conclusion from it.

Two of the three constraints are structural properties the literature treats as diagnostic and
both are repaired; the third is a quantitative match that Section 4.3.5 shows is 90%
predictable from the model's own parameters and Section 4.5 shows is not even
architecture-invariant. This is not an even trade. Section S8 reports
four mechanisms tested for recovering the ratio and why all four fail.

**The effect is behavioural, not a change in which configurations are tested.** Because
eligibility is re-derived at each exponent and the eligible count rises from 946 to 1,655, the
monotone trend above could in principle reflect a shifting population rather than a change in
behaviour at fixed configuration. The same criterion applied to different dynamics selects
different regions of parameter space, so applying it throughout is not by itself a defence. We
therefore repeated the sweep on two sets that do not change with the exponent (Table 9).

**Table 9.** Levelt II reachability on sets that do not change with the exponent.

| *p* | eligibility re-derived | fixed intersection | unfiltered |
|---|---|---|---|
| 1.00 | 3 / 200 | 2 / 200 [0.3%, 3.6%] | **0 / 200** [0.0%, 1.9%] |
| 1.25 | 6 / 200 | 5 / 200 | 1 / 200 |
| 1.50 | 9 / 200 | 8 / 200 | 1 / 200 |
| 1.75 | 20 / 200 | 24 / 200 [8.2%, 17.2%] | 13 / 200 [3.8%, 10.8%] |
| 2.00 | 44 / 200 | 53 / 200 [20.9%, 33.0%] | 28 / 200 [9.9%, 19.5%] |
| 2.50 | 78 / 200 | 90 / 200 | 46 / 200 |
| 3.00 | 91 / 200 | **124 / 200** [55.1%, 68.4%] | **62 / 200** [25.0%, 37.7%] |

The fixed intersection is the 603 configurations eligible at every exponent, of which 200 were
sampled. The unfiltered set is 200 rivalry-producing configurations drawn with no
coefficient-of-variation and no Levelt criterion at all. Reachability rises monotonically on
both, with the interval at *p* = 1 disjoint from those at *p* ≥ 1.75 in each case, and
Modified Proposition IV recovers completely on both (2 to 200 of 200 on the intersection, 23 to
200 on the unfiltered set). The trend is therefore a property of the dynamics at fixed
configuration.

The two fixed sets bracket the re-derived figures in a way worth stating. On the intersection
the effect is *stronger* than with eligibility re-derived, which is unsurprising since that set
is selected on being robust to the exponent and so carries its own filter. The unfiltered set is
the unbiased test and gives the smaller effect, roughly two thirds of the re-derived rate at
every exponent, which is the number to quote when asked how much of the trend survives without
any selection. Its *p* = 1 row also reproduces the zero of Section 4.6 on a pool that section
never used.

**Deviations specific to this analysis.** Eligibility was re-derived with a five-level,
three-seed sweep for the Levelt-ρ criterion rather than the eleven-level, eight-seed sweep of
the original grid, which is too coarse to discriminate (median ρ = 1.000) and yields 946
eligible configurations at *p* = 1 against the registered 762. The same criterion is applied
at every exponent so the across-exponent comparison is internally consistent, but the pool is
more permissive and these results should not be pooled with analyses using the registered one.
The exponent scan was not pre-registered and is exploratory. Because the trend reproduces
on both fixed sets above, the permissive pool affects the absolute rates but not the direction
or the significance of the trend, and the unfiltered figures are the conservative ones.

![Figure 6](figures/figure7.pdf)

**Figure 6.** Superlinear adaptation repairs both structural failures. (A) Configurations of 200 in which Levelt's second proposition is reachable, against the adaptation exponent *p*, with eligibility re-derived at each *p*. (B) Configurations recovering Modified Proposition IV, under the registered 5-timestep filter and on raw transitions. (C) Levelt II reachability on two sets fixed across *p*: the intersection of the eligible sets and an unfiltered rivalry-producing sample (Table 9).

## 5. Discussion

### 5.1 What determines the effect on the competitor

How much a modulation raises the attended channel's activation while the competitor is
winning determines what happens to that competitor, and delivery timing matters because it is
one of the things that controls that quantity. Modulations absent during the competitor's dominance leave it facing less
competition while it wins, so its episodes lengthen; modulations present then shorten them.
Across five formulations matched for effect on the attended channel, the attended channel's
activation during the competitor's dominance orders the competitor's response into three
groups (Section 4.3.1). That ordering is correlational and partly definitional, so the claim
rests instead on setting delivery timing by design: one increment, three schedules, competitor
responses of opposite sign, the ordering surviving a criterion that does not reference the
attended channel. Decomposed at matched amplitude, duty cycle and dose, intermittency as such
contributes nothing, since a regular square wave reproduces continuous delivery at every period
from 0.07 to 6.7 mean episodes. What carries the effect is that the
burst durations are drawn from the dominance duration distribution and are therefore variable,
worth +20.5 percentage points, together with contingency on the present trial, worth a further
+14.9. Schedule statistics flip the sign; contingency sets the size.
Shuffling those durations into a random order changes nothing, and no phase term is
distinguishable from zero. The sign reversal holds in four
architectural families; the magnitudes hold in none (Section 4.5).

Two quantities divide the labour. Timing sets the sign of the coupling. Adaptation gain sets
how much transfers: coupling ratio is predicted principally by κ, with 90% of its variance
recoverable from the six parameters (Section 4.3.5). An observed coupling ratio therefore
measures an observer's adaptation dynamics rather than testing the gating account.

The same architecture bears on whether control is possible at all, though the evidence is
weaker and of a different kind. A goal signal proportional to activation is negligible when
its target is annihilated, so a configuration whose suppressed channel sits near the
rectifier floor offers little to amplify. Section S10 shows the operative quantity is
*G* · *x*<sub>suppressed</sub> relative to the noise scale rather than floor occupancy
itself. The coupling role is supported by direct manipulation; the gating role only by
classification across configurations, with a pre-registered contrast that came out null. We
therefore state the connection as a conjecture that one quantity underlies both.

### 5.2 Why does timing govern an increment but not a change in adaptation?

If delivery timing were the operative property, gating any modulation on dominance should
move it across the ordering of Section 4.3.1. It does not. At an exponent of 2 the
competitor lengthens in 7 of 200 configurations under dominance-gated adaptation modulation
and 3 of 200 under the continuous version, against 199 of 200 for a gated increment and 8 of
200 for the same increment ungated.

The difference is persistence of state. An input increment carries none: switch it off while
the attended channel is suppressed and its contribution there is exactly zero, so the gate
controls suppressed-phase activation directly. A change in adaptation gain carries state.
Reducing the attended channel's adaptation while it is dominant means it accumulates less,
so during the competitor's *subsequent* dominance it is less adapted and therefore more
active, the manipulation is off, but its effect persists through the adaptation variable.
Suppressed-phase activation rises either way.

So there is a second scope condition on timing, alongside the decomposition in Section 4.3.2.
Timing controls the sign for a modulation that leaves no trace when withdrawn; for one carrying
persistent state it does not, because the effect outlives the delivery window. In both cases
the sign follows the attended channel's activation during the competitor's dominance, however
that activation is arrived at, which is why the title names that variable and not the
instrument. The variable of Section 4.3.1 is primary and timing is how an experimenter controls it in
the memoryless case, which is the case Chong et al. instantiated and the case our causal
series manipulates.

### 5.3 Why do the two attention literatures disagree?

Endogenous attention deployed during rivalry prolongs the attended stimulus while leaving the
unattended one unchanged (Chong et al., 2005), contrary to Levelt's second proposition.

**What the human evidence actually establishes, and what it does not.** The target phenomenon
needs stating carefully, because the paper's central result is a *positive* change in the
competitor and the evidence for a positive change is thin. Chong et al.'s Experiment 1 gives
+5.0% on the unattended percept at *t*(3) = 0.54, *p* = .63. Experiment 3 gives +9.0% with no
test reported at *n* = 4. Paffen et al.'s rival effect is null at *n* = 6. What has been
demonstrated across these studies is the **absence of Levelt-type shortening**, which is a real
and consequential finding, since Levelt's second proposition predicts shortening and attention
does not produce it. What has not been demonstrated at conventional power is that the competitor
*lengthens*.

Those are different targets and the second is much harder to hit. A model predicting zero
coupling clears the first. The claim is therefore stated against the first: the model reproduces
the **sign of departure from Levelt**, and its positive coupling magnitude is a prediction about
a branch of the phenomenon that the existing data do not establish. This is also the strongest
argument for the ramp-speed experiment of Section 4.3.5, which would be the first adequately
powered test of whether the positive branch exists at all.
Returning to the disagreement: manipulations of stimulus strength produce the opposite of the
attentional pattern (Levelt, 1965; Mueller & Blake,
1989), and attentional manipulations delivered by cue or prior exposure fall on the strength
side (Hancock & Andrews, 2007; Paffen et al., 2008; Moreno-Sánchez et al., 2019). The field
has described the disagreement without a mechanism producing both.

Chong et al.'s third experiment is the pivot, because it applies one physical manipulation
two ways: doubling contrast *only while the attended grating was dominant* reproduced the
attentional signature, where ungated increments produce the Levelt pattern. Section 4.3.5
implements their gate and reproduces the direction, with the competitor lengthening under
gated delivery in 182 of 188 configurations and shortening under ungated in 183 of 189; the
magnitudes are not informatively reproduced, and Section S8 explains why they cannot be.

The account gives the two literatures a common variable. We do not claim that attention
resembles state-dependent modulation; rather, the one manipulation known to produce the
attentional signature from a pure contrast change *is* a delivery-timing manipulation, and
cue-based and training-based manipulations are not gated on the moment-to-moment state of the
percept, so on this account they should fall on the strength side, which is where they are
observed to fall.

### 5.4 A candidate correlate of the maintained suppressed representation

Hugrass and Crewther (2012) proposed that volitional control succeeds when the dorsal stream
maintains a representation of the suppressed stimulus accessible to attention, a posit with
no measurable correlate. The residual activation of the suppressed channel, scaled by the
noise it must exceed, supplies a candidate, and it is the same family of quantity that
governs inter-channel coupling in Section 4.3.

In the gating role it has the right shape for their result, which is categorical rather than
graded: it classifies whether control is possible while showing no relationship with how much
control there is. Section 4.1 gives AUC 0.792 and 0.735 under a stratification that does not
truncate the comparison predictor's range. Against that, the pre-registered regime contrast
was null and a single threshold does not transfer, so this is a candidate correlate rather
than an established one. The correspondence does sharpen what their optokinetic data could
test: slow-phase pursuit direction gives an objective per-trial index of the suppressed
percept that behavioural report cannot, and the direct test is whether a per-observer index
of residual suppressed activity separates observers who could switch from those who could
not.

**A test using stimulus size.** Violations of Levelt's second proposition cluster with small
rival stimuli and confirmations with large ones (Kang, 2009), a pattern Moreno-Sánchez et al.
(2019) invoke to explain why their result conformed where Chong et al.'s did not. The account predicts this, though by analogy rather than by
simulation, and the analogy is the weak part: residual suppressed activation stands in for
piecemeal dominance in a two-channel model that cannot represent piecemeal dominance at all.
The prediction is therefore extra-model and we label it so. Taking the analogy on its own
terms: small stimuli give near-exclusive dominance and
therefore little residual activation, large stimuli give piecemeal dominance and residual
activation throughout, and Section 4.1 shows the attentional signature attenuated 2.5-fold in
low-residual configurations (Section S1) while the increment signature is unaffected. The same attentional
manipulation should therefore drift from the attentional signature toward the strength
signature as stimulus size increases, with the crossover tracking exclusive visibility. This
is testable against existing data and we have not run it.

### 5.5 What was ruled out, and what the supplement holds

Several accounts proposed in the course of this work were tested and withdrawn, and Table S5
records each with the control that defeated it. Two groups bear on how the main text should be
read. Six candidate mechanisms for the schedule-statistics term of Section 4.3.3 were excluded,
so that term is reported without a mechanism. And four mechanisms for recovering Chong et al.'s
quantitative coupling ratio in the repaired model fail (Section S8). The most informative of
them blocks the competitor's de-adaptation and removes the ratio and the coupling together: in
this architecture the ratio and the qualitative effect are one phenomenon at two magnitudes, so
the ratio cannot be matched without losing the effect.

The published form of Equation 2 differed from the executed code in its update order. Every
result here was generated with the code's form, the equation has been corrected (Section 2.1),
and the central result reproduces under the published form in both configuration selection and
simulation. The analysis kernel and the simulation pipeline agree bit-exactly on a shared seed.
The forensics, and three measurement hazards that each reversed a conclusion during this work,
are in Section S9.

## 6. Limitations

**The stimulus-strength arm is quantitatively wrong.** The model reproduces Levelt's second
proposition in no configuration at any increment magnitude (Section 4.6), and fitted to the
observed 30% compression of the competitor it predicts a 39% increase in the manipulated
channel where approximately 0% is observed (Section S5). A saturating input transfer changes
the prediction by roughly one percentage point. Section 4.7 repairs the failure through the
adaptation law; Section 4.6 gives the structural reason the gate-timing result does not depend
on the arm being correct.

**The adaptation exponent is fitted, and two candidate repairs for the coupling ratio failed.**
Nothing in the architecture or the physiology selects the exponent; it is tuned to phenomena
rather than derived, the value recovering two structural properties displaces a published
quantitative measurement, and eligibility shifts with it. One free parameter repairing two
failures is suggestive of a common cause, but it is not a corrected
model. Of the four mechanisms tested for recovering the ratio (Section S8) and summarised in Table S6, none improves on
its own null across the five criteria, and two fail in opposite directions, driving the
coefficient of variation to 0.06 and to 1.13 against a registered window of [0.35, 0.65]. Two
failures do not exhaust the space, but they close the two obvious routes.

**The coupling variable is entangled with the outcome measure, and the magnitudes largely
are.** The predictor in Section 4.3.1 is measured on the same traces as the outcome and related
to it by the extraction rule. Section 4.3.2 addresses this by setting gate timing
experimentally, and three controls indicate the sign reversal is not a measurement artefact, but the magnitudes are: the gated arm retains only 23–38% of its measured effect under a
criterion that does not reference the attended channel, and its apparent dose-response
disappears there. Only the sign and the live-versus-yoked contrast should be read
quantitatively. In a two-channel mutually inhibitory model no manipulation of one channel is
independent of the other, which bounds any claim of this kind, and the anti-gated condition is
architecturally forced and reported as a control rather than a result.

**Four architectures, and generic ones.** The comparison of five *formulations* in Section
4.3.1 is within a single model, and normalisation models of attention are represented there only
by response gain because that architecture has no divisive term to modulate. The gate-timing
result itself is established in three of four architectures (Section 4.5), including a
divisive-normalisation family and one where adaptation acts on the input, and is directionally
consistent but not established in the fourth, a sigmoid firing-rate family. But those are generic
family representatives with parameters found by search, not reimplementations of Li et al.
(2017) or any other published model, so what is established is robustness to architectural kind
rather than reproduction of a specific model. Running the series inside a published
normalisation model would be stronger and has not been done. The sigmoid family rivalled in
only 1.2% of draws at the ranges used. The quantitative prediction of Section 4.4, that the gated coupling ratio follows exp(−γ*T*), is narrower still: it holds in the two families whose adaptation is driven by a channel's own output through a near-linear transfer and fails in the other two, so it should be read as a property of that class of model rather than of rivalry models generally.

**Generality within the parameter space is limited, in two ways.** The sign contrast is
near-total inside the eligible pool but the attentional side is only weakly better than chance
outside it (61 of 100, CI [51%, 70%]) while the increment side remains absolute (0 of 100). The
eligibility filter selects on coefficient of variation, which is set by the adaptation-noise
balance carrying the coupling, so it is not independent of the effect; nor is it independent of
the discretisation, since the pool defined under the lagged form of Equation 2 overlaps this one
by only 71.3% (Section S9). The discretisation does not affect the central result, which reproduces under the published
form of Equation 2 in both selection and simulation (Section S9). Nor does the eligibility filter. On
200 configurations drawn from all rivalry-producing ones with no coefficient-of-variation and no
Levelt criterion, the gated increment lengthens the competitor in 158 of 168 [89%, 97%] and the
ungated increment in 1 of 164 [0%, 3%]. The gated increment therefore does not share the goal
signal's weakness outside the filter, where the goal signal's same-sign coupling reaches only 61
of 100 (Section 4.3.1). The median magnitude roughly halves, to +5.5% against −13.6% for
ungated delivery, but the sign contrast is intact. A future pre-registration should define eligibility on a quantity
less sensitive to both. Measured residual suppressed activation is the obvious candidate, and it
should assign regimes by measurement rather than by the fixed multipliers used here, which do
not generalise.

**The pre-registered core is null and the substantive results are exploratory.** Both regime
contrasts returned null. The gate-timing series and its controls, the gating analysis, the
rectifier sweep, the one-parameter fits, the exponent scan, the cross-architecture comparison
and the noise-scaling test are all exploratory, and the paper's central claim is among them. We
identify mechanism where we can and withdraw it where a control defeats it rather than
attributing nulls to power, but a confirmatory replication, with delivery timing as the
manipulation and *G* · *x*<sub>suppressed</sub>/σ pre-specified as the gating variable, has not
been run and is what this study warrants next. The empirical mapping in Section 5.3 is likewise
post hoc: the studies compared differ in stimuli, task and observers.

**Measurement and scope.** The gating variable does not transfer as a single global threshold
across regimes and we do not establish how a regime-appropriate cutoff should be set a priori.
The weaker-side test of Proposition II partly measures strength-sensitivity rather than the
proposition. The model should not be used to predict effects of bilateral contrast manipulation.
The architecture is two-channel and cannot address piecemeal dominance or travelling waves. The
attractor analysis is a reduced two-dimensional treatment of a four-dimensional system and does
not describe the strongly adapting cells, which do not alternate in the deterministic limit
(Section S6). Integration is discrete-time and semi-implicit in the adaptation variable,
timesteps are not calibrated to real time so latencies are ordinal only, and the results are
specific to that discretisation to the extent set out in Section S9. The stimulus-size
prediction in Section 5.4 remains untested despite existing data that bear on it.

## 7. Conclusion

Attention and stimulus strength are routinely treated as interchangeable routes to
biasing binocular rivalry. They are not, and what separates them is when the
modulation acts relative to the competitor's dominance. An increment delivered only
while the attended percept is visible lengthens the competitor's episodes; the same
increment delivered throughout shortens them; delivered only while the attended
percept is suppressed it shortens them further, though at the amplitude required that
condition leaves the rivalry regime and is reported as a directional control rather than a
magnitude. The first is not
forced, is the direction the empirical attentional literature reports, and requires
the modulation being structured on the timescale of the alternation and
tracking it: chopping the same dose far below that timescale reproduces continuous delivery,
and a schedule with the right episode structure but tracking a different trial produces about a third of the live
value.

Chong, Tadin and Blake (2005) demonstrated this without a model, by gating a contrast
increment on dominance and obtaining the attentional signature from a pure contrast change.
What the present account adds is why. Reduced to one slow variable travelling between two
switching thresholds, a modulation present at the handover that begins the competitor's episode
lengthens it and one present at the handover that ends it shortens it, by more, because episodes
end near their asymptote. That single ordering gives the sign of all three schedules, why a
continuous increment trades the channels off almost symmetrically and so cannot reproduce
Levelt's second proposition, and why an adaptation law that cancels the attended channel's own
gain repairs it. The reduction predicts episode durations, every sign we observe and the size of the gated
coupling ratio, though not the ungated or anti-gated magnitudes. A gated coupling ratio of the
kind Chong et al. measured therefore estimates how close to their adaptation asymptote an
observer's switches occur, and should be predictable from that observer's dominance durations
and adaptation time constant, as it is in the model's subtractive and divisive forms though not
where adaptation acts on the input or the transfer saturates.

How far this generalises is bounded by what we tested, and that boundary is wider than
we first supposed. The sign reversal is established in three architectures spanning subtractive and
divisive suppression and adaptation applied to a channel's output or to its input, and is
directionally consistent in a fourth with a sigmoid transfer. The magnitudes hold in none of
them, varying roughly thirtyfold across the four. So the claim is about competitive networks with adaptation, and it is a
claim about sign: the direction of the effect is set by an experimental variable rather
than a measured one, and the size of it is set by parameters and architecture jointly and
identifies neither.

What remains specific to the model developed here is its account of stimulus strength,
which is demonstrably wrong, together with the two repairs and the two failed repairs
reported for it. The study's pre-registered contrasts were null, so the results that
survive are exploratory ones that survived controls rather than confirmations of what was
registered. Top-down goals modulate ongoing dynamics; what they do to a competitor depends on how
much they raise the attended channel while that competitor is winning, and delivery timing is
how an experimenter changes that.

---

## Data and code availability

All simulation data are available on the Open Science Framework at
https://osf.io/d975z/. The reproducible pipeline is available at
https://github.com/synthiumjp/rivalry and requires Python 3.12 with NumPy, SciPy,
Numba and Matplotlib. The study design and analysis plan were pre-registered prior
to data generation.

**Deviations from the pre-registered plan.** The full 9,000-configuration grid was
executed in a single run rather than in two stages. The minimum-duration filter
specified at the 5th percentile evaluates to 1 timestep in 29 of 30 configurations;
results are reported at that threshold and at the empirically motivated 5-timestep
boundary. The registered Levelt analysis tested the original rather than the
modified propositions; the modified set was substituted and the equal-strength
sweep testing Modified Proposition IV added when this was found. The signed sweep
(Section 4.3), the gate-timing series and its controls (4.3.2), the gated increment
(4.3.5) and the one-parameter fits (4.7) were all added post hoc, after the
published literature was examined, and are exploratory. No published value was used
to select model parameters; the reference increment in 4.5.5 derives from baseline
activation, not from the data it is compared against. The principal analyses use
100 configurations drawn at random from the eligible pool rather than the 30
originally selected on CV proximity, because the selection criterion correlates
with the outcomes. The duration preservation ratio is a manuscript-stage metric not
specified in the pre-registration. Four defects in the originally executed analysis
are documented in Section S13, and three statistics reported in earlier versions of
this manuscript are withdrawn and documented in place (Section S14). All
exploratory analyses are labelled as such throughout.

**Superseded gate-timing campaign.** An earlier version of this manuscript
reported the gated and anti-gated conditions from a campaign that predates the
first-order lag parameterisation of the gate described in Section 3.5, and whose
stored records carry neither a delivered-dose nor a lag field. Its gate was
therefore effectively instantaneous. Because the competitor's response varies
systematically with ramp speed (Section 4.3.5), that campaign's values are not
comparable with the canonical ones and have been removed rather than reconciled:
the gated 2× competitor response was reported as +23.6% and is +16.2% under the
canonical campaign, and the anti-gated 2× value was reported as −47.7% and is
withdrawn as a magnitude altogether for the reason given in Section 4.3.2. Three
independent campaigns agree on the canonical value at τ = 1 (+16.9%, +16.2% and
+17.1%), so the difference is a change of condition and not Monte Carlo variation.
Every gate-timing number in this manuscript is now quoted from the campaign
designated canonical in Section S14.

**Correction to Equation 2.** Equation 2 as written in an earlier version of this
manuscript specified adaptation driven by *x<sub>i</sub>*(*t*). The executed
pipeline drives it from *x<sub>i</sub>*(*t*+1), advancing activation first and
adaptation from the new activation. The equation has been corrected to match the
code; no result has been recomputed, because every result reported here was
generated with the code's version and the pipeline is internally consistent across
all analyses. The discrepancy was identified by an independent reimplementation
built from the published equations and is characterised in Section S9: it is
O(ακ), leaves the steady-state relation and the bound of Section S5 unchanged,
and alters the composition of the eligible pool by roughly a third. Initial
conditions (*x* = 0.1, zero adaptation) and the recording convention were also
unreported and are now stated in Section 2.2.

---

## References

Blake, R. (1988). Dichoptic reading: The role of meaning in binocular rivalry.
*Perception & Psychophysics*, 44, 133–141.

Brascamp, J. W., & Blake, R. (2012). Inattention abolishes binocular rivalry:
Perceptual evidence. *Psychological Science*, 23, 1159–1167.

Brascamp, J. W., Klink, P. C., & Levelt, W. J. M. (2015). The 'laws' of binocular
rivalry: 50 years of Levelt's propositions. *Vision Research*, 109, 20–37.

Brascamp, J. W., van Ee, R., Pestman, W. R., & van den Berg, A. V. (2005). Distributions
of alternation rates in various forms of bistable perception. *Journal of Vision*, 5(4):1,
287–298.

Braun, J., & Mattia, M. (2010). Attractors and noise: twin drivers of decisions and
multistability. *NeuroImage*, 52, 740–751.

Cao, R., Braun, J., & Mattia, M. (2014). Stochastic accumulation by cortical columns may
explain the scalar property of multistable perception. *Physical Review Letters*, 113, 098103.

Chong, S. C., Tadin, D., & Blake, R. (2005). Endogenous attention prolongs
dominance durations in binocular rivalry. *Journal of Vision*, 5(11):6, 1004–1012.

Curtu, R., Shpiro, A., Rubin, N., & Rinzel, J. (2008). Mechanisms for frequency
control in neuronal competition models. *SIAM Journal on Applied Dynamical
Systems*, 7, 609–649.

Dieter, K. C., & Tadin, D. (2011). Understanding attentional modulation of
binocular rivalry: A framework based on biased competition. *Frontiers in Human
Neuroscience*, 5, 155.

Dieter, K. C., Melnick, M. D., & Tadin, D. (2016). Perceptual training profoundly
alters binocular rivalry through both sensory and attentional enhancements.
*Proceedings of the National Academy of Sciences*, 113, 12874–12879.

Hancock, S., & Andrews, T. J. (2007). The role of voluntary and involuntary
attention in selecting perceptual dominance during binocular rivalry.
*Perception*, 36, 288–298.

Hugrass, L., & Crewther, D. (2012). Willpower and conscious percept: Volitional
switching in binocular rivalry. *PLoS ONE*, 7, e35963.

Kang, M.-S. (2009). Size matters: A study of binocular rivalry dynamics. *Journal
of Vision*, 9(1):17, 1–11.

Levelt, W. J. M. (1965). *On binocular rivalry*. Soesterberg: Institute for
Perception RVO-TNO.

Li, H.-H., Rankin, J., Rinzel, J., Carrasco, M., & Heeger, D. J. (2017). Attention
model of binocular rivalry. *Proceedings of the National Academy of Sciences*, 114,
E6192–E6201.

Meng, M., & Tong, F. (2004). Can attention selectively bias bistable perception?
Differences between binocular rivalry and ambiguous figures. *Journal of Vision*,
4(7):2, 539–551.

Mitchell, J. F., Stoner, G. R., & Reynolds, J. H. (2004). Object-based attention
determines dominance in binocular rivalry. *Nature*, 429, 410–413.

Moreno-Bote, R., Rinzel, J., & Rubin, N. (2007). Noise-induced alternations in an
attractor network model of perceptual bistability. *Journal of Neurophysiology*, 98,
1125–1139.

Moreno-Sánchez, M., Aznar-Casanova, J. A., & Valle-Inclán, F. (2019). Attention to
monocular images bias binocular rivalry. *Frontiers in Systems Neuroscience*, 13,
12.

Mueller, T. J., & Blake, R. (1989). A fresh look at the temporal dynamics of
binocular rivalry. *Biological Cybernetics*, 61, 223–232.

Paffen, C. L. E., Alais, D., & Verstraten, F. A. J. (2006). Attention speeds
binocular rivalry. *Psychological Science*, 17, 752–756.

Paffen, C. L. E., Verstraten, F. A. J., & Vidnyánszky, Z. (2008). Attention-based
perceptual learning increases binocular rivalry suppression of irrelevant visual
features. *Journal of Vision*, 8(4):25, 1–11.

Pastukhov, A., García-Rodríguez, P. E., Haenicke, J., Guillamon, A., Deco, G., &
Braun, J. (2013). Multi-stable perception balances stability and sensitivity. *Frontiers in
Computational Neuroscience*, 7, 17.

Platonov, A., & Goossens, J. (2013). Influence of contrast and coherence on the
temporal dynamics of binocular motion rivalry. *PLoS ONE*, 8, e71931.

Reynolds, J. H., & Heeger, D. J. (2009). The normalization model of attention.
*Neuron*, 61, 168–185.

Seely, J., & Chow, C. C. (2011). Role of mutual inhibition in binocular rivalry.
*Journal of Neurophysiology*, 106, 2136–2150.

Shpiro, A., Curtu, R., Rinzel, J., & Rubin, N. (2007). Dynamical characteristics
common to neuronal competition models. *Journal of Neurophysiology*, 97, 462–473.

Stuit, S. M., Paffen, C. L. E., Van der Smagt, M. J., & Verstraten, F. A. J.
(2014). Suppressed images selectively affect the dominant percept during binocular
rivalry. *Journal of Vision*, 14(5):9, 1–13.

Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: The
leaky, competing accumulator model. *Psychological Review*, 108, 550–592.

van Ee, R. (2009). Stochastic variations in sensory awareness are driven by noisy neuronal
adaptation: evidence from serial correlations in perceptual bistability. *Journal of the
Optical Society of America A*, 26, 2612–2622.

Van Ee, R., van Dam, L. C. J., & Brouwer, G. J. (2005). Voluntary control and the
dynamics of perceptual bi-stability. *Vision Research*, 45, 41–55.

Wang, X.-J., & Rinzel, J. (1992). Alternating and synchronous rhythms in reciprocally inhibitory model neurons. *Neural Computation*, 4, 84–97.

Zhang, P., Jamison, K., Engel, S., He, B., & He, S. (2011). Binocular rivalry
requires visual attention. *Neuron*, 71, 362–369.

## Figures

Figures are placed at the point of first discussion. Supplementary Figures S1 and S2 are in the
supplementary material.

| Figure | Section | What it shows |
|---|---|---|
| 1 | 4.2 | Duration variability across the parameter grid |
| 2 | 4.3.1 | What the coupling sign tracks |
| 3 | 4.3.2 | Gate timing as an independent variable |
| 4 | 4.3.5 | Chong et al.'s manipulation reproduced in direction |
| 5 | 4.5 | The sign reversal across architectures |
| 6 | 4.7 | Superlinear adaptation repairs both structural failures |
