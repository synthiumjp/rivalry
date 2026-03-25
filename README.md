# GC-LCA: Goal-Conditioned Leaky Competing Accumulator

**A computational model of volitional control in binocular rivalry**

[![Pre-registration](https://img.shields.io/badge/OSF-Pre--registered-blue)](https://osf.io/d975z/overview?view_only=b29684da1ef14c47a1994a15ea3f8a37)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

This repository contains the complete reproducible pipeline for:

> Cacioli, J. P. (2026). State-dependent persistence modulation in a goal-conditioned leaky competing accumulator: A computational account of volitional control in binocular rivalry. *Manuscript in preparation for Psychological Review.*

## Overview

The GC-LCA extends the Usher and McClelland (2001) leaky competing accumulator by introducing goal-conditioned persistence modulation via gated recurrent self-excitation. The goal signal reduces the effective leak rate of a target channel, amplifying representations that retain residual activation while having zero effect on representations driven to the rectifier floor by inhibition.

The model provides the first computational account of the Hugrass and Crewther (2012) finding that volitional switching succeeds for motion-defined stimuli (dorsal stream) but fails for stationary stimuli (ventral stream).

**Headline result:** Goal-conditioned modulation extends target dominance while preserving rivalry (DPR ≈ 1.05). Mean-matched signal boost extends target dominance while destroying rivalry (DPR ≈ 0.62). Cohen's *d* = 0.78.

## Repository Structure

```
rivalry/
├── gc_lca_phase1_grid.py           # Phase I: 9,000-config parameter sweep
├── gc_lca_phase2_dissociation.py   # Phase II: 18,000-trial dissociation test
├── paper1_confirmatory_stats.py    # Confirmatory statistics (Blocks 0–7)
├── paper1_supplementary_stats.py   # Supplementary statistics (H3, H4 ANOVA, H2 d)
├── paper1_mechanistic.py           # Phase portraits, sensitivity, κ role (+ Figs 7–9)
├── paper1_figures.py               # Publication figures 1–6
├── convert_figures.py              # PDF → TIFF conversion for submission
│
├── phase1_grid_results.json        # Phase I results (7.6 MB)
├── phase2_dissociation_results.json# Phase II results
├── paper1_stats_results.json       # Confirmatory stats output
├── paper1_supplementary_stats.json # Supplementary stats output
├── paper1_mechanistic_results.json # Mechanistic analysis output
│
├── fig1_model.pdf                  # Figure 1: Model architecture
├── fig2_parameter_space.pdf        # Figure 2: Parameter space
├── fig3_levelt.pdf                 # Figure 3: Levelt propositions
├── fig4_h4_headline.pdf            # Figure 4: H4 discriminant test (headline)
├── fig5_dissociation.pdf           # Figure 5: Volitional control / dissociation
├── fig6_mechanism.pdf              # Figure 6: Rectifier, trajectory, ablations
├── fig7_phase_portrait.pdf         # Figure 7A: Phase portraits
├── fig7b_bifurcation.pdf           # Figure 7B: Bifurcation diagram
├── fig8_sensitivity.pdf            # Figure 8: Sensitivity analysis
├── fig9_kappa_role.pdf             # Figure 9: κ role
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24
scipy>=1.10
numba>=0.57
matplotlib>=3.7
```

For figure conversion (optional):
```
pdf2image>=1.16
Pillow>=9.0
```

Python 3.12 recommended. Numba JIT compilation is used throughout for performance.

### Reproduce Everything

Run the scripts in order. Each script reads the output of the previous one.

```bash
# Phase I: parameter sweep (9,000 configs × 30 seeds × 20,000 steps)
# Runtime: ~5 minutes with Numba JIT on multi-core CPU
python gc_lca_phase1_grid.py

# Phase II: dissociation testing (18,000 trials)
# Runtime: ~2 seconds
python gc_lca_phase2_dissociation.py

# Confirmatory statistics (includes new simulations for H4 and ablations)
# Runtime: ~4 minutes
python paper1_confirmatory_stats.py

# Supplementary statistics (H3 interaction, H4 ANOVA, H2 effect sizes)
# Runtime: <1 second
python paper1_supplementary_stats.py

# Mechanistic analysis (phase portraits, sensitivity, κ role)
# Also generates figures 7–9
# Runtime: ~1 minute
python paper1_mechanistic.py

# Publication figures 1–6
python paper1_figures.py

# Convert figures for submission (optional; requires poppler)
python convert_figures.py --also-jpg
```

Total pipeline runtime: approximately 10–12 minutes on a modern multi-core CPU.

## Model Specification

**Core update equation:**

```
x_i(t+1) = clip[0, x_max]( (1 - λ + G_i)·x_i(t) + S_i - β·x_j(t) - α·a_i(t) + η_i(t) )
```

**Adaptation:**

```
a_i(t+1) = (1 - γ)·a_i(t) + κ·x_i(t)
```

**Stability constraint:** 0 ≤ G_i < λ (enforced as G_i ≤ 0.95λ)

| Parameter | Description | Grid values |
|-----------|-------------|-------------|
| λ | Leak rate | {0.08, 0.10, 0.15, 0.20} |
| β | Mutual inhibition | {0.10, 0.15, 0.20, 0.25, 0.30, 0.35} |
| α | Adaptation weight | {0.03, 0.05, 0.08, 0.10, 0.12} |
| σ | Noise SD | {0.04, 0.06, 0.08, 0.10, 0.12} |
| γ | Adaptation decay | {0.02, 0.03, 0.05} |
| κ | Adaptation gain | {0.50α, 0.75α, 1.00α, 1.25α, 1.50α} |
| G | Goal signal | {0, 10, 25, 50, 75, 90}% of λ |

## Pre-Registration

All confirmatory hypotheses, parameter grids, selection criteria, statistical tests, and ablation conditions were pre-registered on OSF before data collection:

**OSF:** [https://osf.io/d975z/](https://osf.io/d975z/overview?view_only=b29684da1ef14c47a1994a15ea3f8a37)

Deviations from the pre-registered plan are documented in the manuscript's Transparency and Openness section.

## Key Results

| Hypothesis | Outcome | Key statistic |
|------------|---------|---------------|
| V1 (Levelt Props I–IV) | Supported | ρ = 0.97 (Prop I), 29/30 pass |
| H1 (Duration variability) | Supported | 29/30 adequate; Weibull > gamma in 75% |
| H2 (Volitional control) | Partially supported | 5/30 strict; β₁ = 7–15 |
| H3 (Dorsal/ventral dissociation) | Non-significant pooled | F(3,232) = 0.033, p = .992 |
| H4 (G ≠ signal boost) | **Supported (headline)** | DPR: 1.05 vs 0.62; d = 0.78 |
| H8 (Soft rectifier) | **Supported** | 0% → 100% ventral rate |

## Phase I → Phase II Pipeline

```
9,000 configs (Phase I grid)
    → 6,814 rivalry-producing
    → 762 eligible (rivalry + Levelt + CV)
    → 30 selected (best CV proximity to 0.5)
    → 18,000 Phase II trials (30 × 6 G levels × 2 regimes × 50 seeds)
```

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{cacioli2026gclca,
  title={State-dependent persistence modulation in a goal-conditioned leaky 
         competing accumulator: A computational account of volitional control 
         in binocular rivalry},
  author={Cacioli, JP},
  journal={Manuscript in preparation for Psychological Review},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds on the Usher and McClelland (2001) leaky competing accumulator framework and addresses the volitional switching dissociation reported by Hugrass and Crewther (2012).
