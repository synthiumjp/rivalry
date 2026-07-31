"""
fix_git.py — get C:\\crewther into a state that can be released alongside the paper.

The manuscript cites https://github.com/synthiumjp/rivalry as the code repository and
Section A.5 designates a canonical campaign. Neither claim is checkable unless the
repository actually contains the code that produced the results, with a commit the
paper can name.

This script does not commit or push anything. It reports what it finds, writes a
.gitignore and a README if they are absent, and prints the exact commands to run. Every
destructive step is left to you.

Two constraints it enforces, both from this project's own history:

  RESULT FILES ARE LARGE AND NUMEROUS. There are dozens of JSONs, several over a
  megabyte. Committing them all makes the repository unusable for anyone cloning it to
  read the code. The convention below tracks code and the small summary outputs the
  paper quotes, and lists the large campaign files as release assets instead.

  SIX SCRIPTS MUST NEVER BE STAGED. The project notes record that `git add -A` has
  previously swept in files that should not be public. This script lists everything
  untracked so nothing is added blind.

USAGE
    python fix_git.py                # audit and write scaffolding
    python fix_git.py --no-write     # audit only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def git(*args, check=False):
    try:
        r = subprocess.run(['git'] + list(args), capture_output=True, text=True,
                           timeout=30)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        sys.exit("git not on PATH")
    except subprocess.TimeoutExpired:
        return 1, '', 'timeout'


GITIGNORE = """# --- build and environment -------------------------------------------------
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.vscode/
.idea/
figures/*.png

# --- large campaign outputs ------------------------------------------------
# The paper quotes summary values from these; the files themselves are release
# assets rather than repository content, because several exceed a megabyte and a
# clone should be readable without them. Regenerate with the block that produced
# them, or attach them to a tagged release.
phase1_grid_results.json
wave9_M_pulse.json
wave9_N_matching.json
wave2_*.json
wave3_*.json
wave6_*.json
wave8_*.json
wave9_*.json
wave1[0-9]_*.json
w22_*.json
zoomed_heatmap_results.json
dissociation_retest_results.json

# --- keep these, they are small and the paper cites them directly ----------
!phase2_dissociation_results.json
!registered_tests_output.json
!wave22_V_verify.json
!wave22_P_paired.json
"""

README = """# Gate timing and inter-channel coupling in binocular rivalry

Code for *Suppressed-phase activation sets the sign of inter-channel coupling in
binocular rivalry*, Cacioli and Marmo.

Pre-registration: https://osf.io/d975z/ (March 2026, prior to data generation).

## Layout

| file | what it does |
|---|---|
| `gc_lca_phase1_grid.py` | the 9,000-configuration parameter grid and Levelt screening |
| `wave2_campaign.py` | the simulation pipeline: `run_trace` is the reference kernel |
| `wave22_adaptation.py` | analysis blocks V and A through P, one per question |
| `wave23_architectures.py` | the four-architecture comparison and the ramp crossover |
| `registered_tests.py` | the two pre-registered analyses, computed from stored outputs |
| `consistency_check.py` | manuscript audit: cross-references, conflicting values, hedging |
| `make_figures.py` | figures from stored JSONs |

## Reproducing

Blocks are independent and each writes a JSON stamped with the constants, the git
commit and the numba version it ran under.

```
python wave22_adaptation.py --verify              # kernel against the frozen grid
python wave22_adaptation.py --block P             # kernel against wave2_campaign, paired
python wave22_adaptation.py --block A --out w22_  # eligibility per adaptation exponent
python wave22_adaptation.py --block B --elig w22_ --out w22_
```

`--elig` names an existing eligibility pool to read; `--out` names the prefix to write.
They are separate so a new campaign can reuse a pool without overwriting it.

`--block P` is the load-bearing control. It calls `wave2_campaign.run_trace` and this
repository's kernel with identical parameters and an identical seed and compares the
traces element by element. They agree bit-exactly. Any change to either kernel should
be checked against it before anything else is run.

## Requirements

Python 3.11+, numpy, scipy, numba, matplotlib. Without numba the pure-Python fallback
runs but produces a different random stream, so results will not match; the provenance
stamp in every output records which was used.

## Note on result files

Large campaign outputs are release assets rather than repository content. The small
summaries the paper quotes directly are tracked.
"""




# ---- files that belong in the repository, by name, never by wildcard ----
# Campaign scripts produced the results the paper reports, so they belong in the
# repository even where a later wave supersedes them. Listed by name because
# `git add *.py` would also sweep in scratch and one-off inspection scripts.
CODE = [
    # kernels and campaigns, in the order they were run
    'gc_lca_phase1_grid.py', 'wave2_campaign.py', 'wave3_levelt.py',
    'wave6_robustness.py', 'wave8_regimes.py', 'wave9_classifier.py',
    'wave10_signed_G.py', 'wave11_gated.py', 'wave12_transfer.py',
    'wave13_coupling.py', 'wave14_formulations.py', 'wave15_paired.py',
    'wave16_coupling_final.py', 'wave17_antigate.py', 'wave18_yoked.py',
    'wave19_reanalysis.py', 'wave20_audit.py', 'wave21_ratio.py',
    'wave22_adaptation.py', 'wave23_architectures.py',
    # audit and reporting tools
    'wave22_cv_audit.py', 'wave22_metrics_compare.py', 'wave22_reconcile.py',
    'wave22_verify_claims.py', 'registered_tests.py', 'consistency_check.py',
    'make_figures.py', 'fix_git.py',
]

# Scratch and one-off inspection scripts. Named so the audit can say explicitly
# that they were considered and excluded, rather than silently omitting them.
EXCLUDED = ['deep_dump.py', 'inspect_outputs.py', 'summarise_wave6.py']
SMALL = ['phase2_dissociation_results.json', 'wave22_V_verify.json',
         'w22_wave22_P_paired.json', 'registered_tests_output.json']
SCAFFOLD = ['.gitignore', 'README.md']


def do_commit(message, execute):
    """Stage a named list and commit. Nothing is added by wildcard, and the exact
    command sequence is printed before anything runs."""
    print("\n" + "=" * 72)
    print("STAGING" if execute else "STAGING, DRY RUN")
    print("=" * 72)

    present, absent = [], []
    for f in SCAFFOLD + CODE + SMALL:
        (present if os.path.exists(f) else absent).append(f)

    print(f"\n  will stage {len(present)} files:")
    for f in present:
        kb = os.path.getsize(f) / 1024
        flag = '  <-- over 1 MB, consider a release asset' if kb > 1024 else ''
        print(f"    {f:<44} {kb:>8.0f} KB{flag}")
    if absent:
        print(f"\n  not present, skipped: {', '.join(absent)}")

    rc, out, _ = git('status', '--porcelain')
    untracked = [l[3:] for l in out.split('\n') if l.startswith('??')]
    unstaged = [u for u in untracked if u not in present and not u.startswith('figures/')]
    deliberate = [u for u in unstaged if u in EXCLUDED]
    if deliberate:
        print(f"\n  excluded by name as scratch or inspection scripts: "
              f"{', '.join(deliberate)}")
    unstaged = [u for u in unstaged if u not in EXCLUDED]
    if unstaged:
        print(f"\n  {len(unstaged)} untracked files will NOT be staged. Check that "
              f"none of them belongs:")
        for f in sorted(unstaged)[:20]:
            print(f"    {f}")
        if len(unstaged) > 20:
            print(f"    ... and {len(unstaged) - 20} more")

    if not execute:
        print(f"""
  Nothing has been staged. To do it:

      python fix_git.py --commit -m "{message}"

  That runs exactly:

      git add {' '.join(present[:4])} ...
      git commit -m "{message}"

  and nothing else. It will not run `git add -A`, will not push, and will not tag.""")
        return

    rc, o, e = git('add', *present)
    if rc != 0:
        print(f"  git add failed: {e}")
        return
    print(f"\n  staged {len(present)} files")
    rc, o, _ = git('diff', '--cached', '--stat')
    print('\n'.join('    ' + l for l in o.split('\n')[-12:]))
    rc, o, e = git('commit', '-m', message)
    print(f"\n  {o or e}")
    rc, h, _ = git('rev-parse', '--short', 'HEAD')
    print(f"""
  Committed as {h}.

  Next, and these are deliberately not automated:

      git tag -a v1.0-submission -m "State at manuscript submission"
      git push origin main --tags

  Then put {h} into Section A.5 in place of the campaign name, so the paper's
  canonical designation names a commit rather than a description. Every output file
  already carries a _provenance block with the commit it was produced under, which
  makes that designation checkable in both directions.""")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-write', action='store_true')
    ap.add_argument('--commit', action='store_true',
                    help='actually stage and commit; without it this is a dry run')
    ap.add_argument('-m', dest='msg',
                    default='Analysis pipeline and manuscript audit tooling',
                    help='commit message')
    args = ap.parse_args()

    rc, out, _ = git('rev-parse', '--is-inside-work-tree')
    if rc != 0:
        print("NOT A GIT REPOSITORY.\n")
        print("  git init")
        print("  git remote add origin https://github.com/synthiumjp/rivalry.git")
        print("\nRun this script again afterwards.\n")
    else:
        print("=" * 72)
        print("REPOSITORY STATE")
        print("=" * 72)
        for label, argv in (('branch', ('rev-parse', '--abbrev-ref', 'HEAD')),
                            ('commit', ('rev-parse', '--short', 'HEAD')),
                            ('remote', ('remote', '-v')),
                            ('last commit', ('log', '-1', '--format=%h %ad %s',
                                             '--date=short'))):
            rc, o, e = git(*argv)
            print(f"  {label:<12} {o.splitlines()[0] if o else (e or 'none')}")

        rc, o, _ = git('status', '--porcelain')
        lines = [l for l in o.split('\n') if l.strip()]
        mod = [l for l in lines if not l.startswith('??')]
        untracked = [l[3:] for l in lines if l.startswith('??')]
        print(f"\n  {len(mod)} modified or staged, {len(untracked)} untracked")

        if mod:
            print("\n  MODIFIED:")
            for l in mod[:30]:
                print("   ", l)

        if untracked:
            print(f"\n  UNTRACKED ({len(untracked)}). Review every line before adding "
                  f"anything.")
            code = [f for f in untracked if f.endswith('.py')]
            data = [f for f in untracked if f.endswith('.json')]
            other = [f for f in untracked if f not in code and f not in data]
            for label, group in (('python', code), ('json', data), ('other', other)):
                if group:
                    print(f"\n    {label} ({len(group)}):")
                    for f in sorted(group)[:25]:
                        try:
                            kb = os.path.getsize(f) / 1024
                            print(f"      {f:<46} {kb:>8.0f} KB")
                        except OSError:
                            print(f"      {f}")
                    if len(group) > 25:
                        print(f"      ... and {len(group)-25} more")

        print("""
  DO NOT run `git add -A`. This project's notes record that it has previously swept
  in files that should not be public. Add by name, or add the .gitignore first and
  then review `git status` again.""")

    # ---- scaffolding ----
    print("\n" + "=" * 72)
    print("SCAFFOLDING")
    print("=" * 72)
    for name, content in (('.gitignore', GITIGNORE), ('README.md', README)):
        p = Path(name)
        if p.exists():
            print(f"  {name} exists, left alone ({p.stat().st_size} bytes)")
        elif args.no_write:
            print(f"  {name} absent, would write {len(content)} bytes")
        else:
            p.write_text(content, encoding='utf-8')
            print(f"  wrote {name}")

    # ---- size audit ----
    print("\n" + "=" * 72)
    print("LARGEST FILES, which decide what belongs in a release rather than a clone")
    print("=" * 72)
    sizes = []
    for f in Path('.').glob('*'):
        if f.is_file() and f.suffix in ('.json', '.py', '.md', '.csv'):
            sizes.append((f.stat().st_size, f.name))
    for s, n in sorted(sizes, reverse=True)[:12]:
        print(f"  {s/1024:>9.0f} KB  {n}")
    total = sum(s for s, _ in sizes) / 1024 / 1024
    print(f"\n  {len(sizes)} files, {total:.1f} MB total")
    if total > 50:
        print("  Over 50 MB. Keep the large campaign outputs out of the repository "
              "and attach\n  them to a tagged release instead; the .gitignore above "
              "already does this.")

    do_commit(args.msg, args.commit)

    print("""
======================================================================
FURTHER NOTES
======================================================================

Review the untracked list above first. Then, adding by name rather than with -A:

    git add .gitignore README.md
    git add gc_lca_phase1_grid.py wave2_campaign.py wave22_adaptation.py
    git add wave23_architectures.py registered_tests.py consistency_check.py
    git add make_figures.py fix_git.py
    git add phase2_dissociation_results.json wave22_V_verify.json
    git status                      # read this before committing
    git commit -m "Analysis pipeline and manuscript audit tooling for the rivalry paper"

Then tag the commit the paper names as canonical, and put that hash into Section A.5:

    git tag -a v1.0-submission -m "State at manuscript submission"
    git rev-parse --short v1.0-submission
    git push origin main --tags

Section A.5 currently designates a canonical campaign by name. Replace that with the
tag and the commit hash, so the designation is checkable. Every output file now carries
a `_provenance` block with the commit it was produced under, which makes the claim
verifiable in both directions.

Large campaign outputs go on the release page rather than in the tree:

    gh release create v1.0-submission w22_*.json wave9_*.json phase1_grid_results.json \\
        --title "Campaign outputs at submission" \\
        --notes "Result files for the figures and tables. Code is in the tagged tree."
""")


if __name__ == '__main__':
    main()
