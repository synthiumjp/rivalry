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

The competitor's episode starts further from its own end point and lengthens.

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

| Schedule | Ratio | Range |
|---|---|---|
| Gated | ρ | between 0 and 1 |
| Ungated | −1 | |
| Anti-gated | 1/ρ | greater than 1 |

These are first-order statements. They neglect the increment's effect on the asymptote, since
an increment present during A's dominance raises x<sub>A</sub><sup>+</sup> and so U<sub>A</sub>;
they treat switching as deterministic; and they assume the two channels are symmetric. Section
4.4 reports that the gated ratio is correlated with ρ but smaller than it, the ungated ratio is
about −0.6 rather than −1, and the anti-gated prediction fails. The sign rule does not depend on
any of these, since it follows from the ordering U − c < U + c alone.

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
latest, 0.70. On four points this is suggestive rather than established, and since the gated
ratio falls short of ρ the formula overestimates the crossover by about half.

### S4.6 Relation to earlier work

The reduction is the relaxation-oscillator geometry used to analyse release and escape in
mutually inhibitory pairs (Wang & Rinzel, 1992) and to derive Levelt's fourth proposition and
its violation in rivalry models (Shpiro et al., 2007; Curtu et al., 2008). Its application to
the timing of a modulation relative to the competitor's dominance, and the sign rule that
follows, is new.
