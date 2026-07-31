"""
consistency_check.py — audit the manuscript for internal contradiction.

Nine rounds of incremental revision plus seven sections rewritten in one sitting is
exactly the condition under which a document starts disagreeing with itself. This
checks the classes of error that actually occurred during this project rather than a
generic style pass.

  1. STRUCTURE      heading numbering monotone and gapless; no duplicate numbers
  2. REFERENCES     every "Section X" resolves, and points at plausible content
  3. NUMBERS        the same quantity stated twice with different values
  4. WITHDRAWALS    a claim withdrawn in one place and asserted in another
  5. COUNTS         "n of N" figures that disagree across sections
  6. SUPERSEDED     values from campaigns the manuscript designates non-canonical
  7. HEDGES         claims stated flatly in one place and hedged in another

Usage:  python consistency_check.py gate-timing-rivalry-v9.md
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict


def load(path):
    src = open(path, encoding='utf-8').read()
    heads = [(m.group(1), m.group(2), m.group(3), m.start())
             for m in re.finditer(r'^(#{2,4}) ([\dA]+(?:\.\d+)*) (.+)$', src, re.M)]
    # section containing each character offset
    def sec_of(pos):
        cur = None
        for lv, num, title, p in heads:
            if p <= pos:
                cur = num
            else:
                break
        return cur or '(front matter)'
    return src, heads, sec_of


def line_of(src, pos):
    return src[:pos].count('\n') + 1


def check_structure(heads):
    print("=" * 74)
    print("1. STRUCTURE")
    print("=" * 74)
    nums = [h[1] for h in heads]
    dupes = {n for n in nums if nums.count(n) > 1}
    print(f"  {len(heads)} numbered headings")
    if dupes:
        print(f"  ** DUPLICATE numbers: {sorted(dupes)}")
    else:
        print("  no duplicate numbers")
    # gaps within each parent
    by_parent = defaultdict(list)
    for n in nums:
        parts = n.split('.')
        by_parent['.'.join(parts[:-1]) or 'top'].append(parts[-1])
    gaps = []
    for parent, kids in by_parent.items():
        ints = sorted(int(k) for k in kids if k.isdigit())
        if not ints:
            continue
        expect = list(range(ints[0], ints[-1] + 1))
        missing = [i for i in expect if i not in ints]
        if missing:
            gaps.append((parent, missing))
    if gaps:
        for parent, missing in gaps:
            print(f"  ** GAP under {parent}: missing {missing}")
    else:
        print("  no numbering gaps")
    return not dupes and not gaps


def check_refs(src, heads, sec_of):
    print("\n" + "=" * 74)
    print("2. CROSS-REFERENCES")
    print("=" * 74)
    known = {h[1]: h[2] for h in heads}
    bad, selfref = [], []
    # \s+ so line-wrapped references are seen; ranges and bare numbers handled below
    for m in re.finditer(r'Sections?\s+([\dA]+\.\d+(?:\.\d+)?)', src):
        tgt = m.group(1)
        if tgt not in known:
            bad.append((line_of(src, m.start()), tgt))
        elif sec_of(m.start()) == tgt:
            selfref.append((line_of(src, m.start()), tgt))
    # second half of ranges: "Sections 4.5.1 to 4.5.4"
    for m in re.finditer(r'Sections?\s+[\dA]+\.[\d.]+\s+(?:to|and|through)\s+([\dA]+\.[\d.]+)',
                         src):
        t = m.group(1).rstrip('.')
        if t not in known:
            bad.append((line_of(src, m.start()), t + ' (range end)'))
    # bare section numbers with no prefix, e.g. "the increment in 4.5.4"
    for m in re.finditer(r'\b(?:in|of|from|see)\s+([45A]\.\d+(?:\.\d+)?)\b(?![\d.%])', src):
        t = m.group(1)
        if t not in known:
            bad.append((line_of(src, m.start()), t + ' (bare)'))
    print(f"  {len(list(re.finditer(r'Sections?\\s+[\\dA]', src)))} references found")
    if bad:
        for ln, t in bad:
            print(f"  ** DANGLING line {ln}: Section {t}")
    else:
        print("  all references resolve")
    if selfref:
        print(f"  note: {len(selfref)} self-references "
              f"(a section citing itself) at lines "
              f"{[l for l, _ in selfref][:6]}")
    # keyword plausibility: does the target title share a content word?
    STOP = {'the','a','of','and','in','is','not','to','for','on','an','it','its',
            'this','that','what','why','does','do','both','two','one','from','with',
            'as','by','at','are','was','were','be','been','which','their','than'}
    weak = []
    for m in re.finditer(r'([^.]{0,90})Sections? ([\dA]+\.\d+(?:\.\d+)?)', src):
        ctx, tgt = m.group(1).lower(), m.group(2)
        if tgt not in known:
            continue
        title_words = {w.strip('.,:;()*') for w in known[tgt].lower().split()} - STOP
        ctx_words = {w.strip('.,:;()*') for w in ctx.split()} - STOP
        if title_words and not (title_words & ctx_words):
            weak.append((line_of(src, m.start()), tgt, known[tgt][:38]))
    print(f"\n  {len(weak)} references whose surrounding text shares no content word")
    print("  with the target's title (may be fine, worth eyeballing):")
    for ln, t, title in weak[:12]:
        print(f"    line {ln:>5}  -> {t:<7} {title}")
    return not bad


def check_numbers(src, sec_of):
    print("\n" + "=" * 74)
    print("3. REPEATED QUANTITIES WITH DIFFERENT VALUES")
    print("=" * 74)
    # quantities that recur and have been sources of contradiction in this project
    PROBES = {
        'gated 2x competitor':      r'\+16\.2%|\+17\.1%|\+16\.9%|\+23\.6%',
        'ungated 1x competitor':    r'−10\.4%|−10\.5%|-10\.4%',
        'anti-gated 2x competitor': r'−76\.1%|−47\.7%',
        'live-vs-yoked at 2x':      r'\+20\.4|\+11\.0',
        'grid median CV':           r'median (?:CV|coefficient[^.]{0,30})[^.]{0,20}0\.5\d\d',
        'CV window fraction':       r'(?<!\+)13\.7%|(?<!\+)14\.3%',
        'eligible pool':            r'\b762\b|\b946\b|\b732\b',
        'rivalry-producing':        r'\b6,?814\b',
        'Levelt II at p=1':         r'\b3 (?:of|/) ?200\b|none of 100|no configuration',
        'PropIV violated':          r'97 of 100|97/100',
        'Chong ratio':              r'0\.310|9/29',
        'crossover':                r'0\.59|0\.40 to 0\.66',
        'ratio spread':             r'fifteen-fold|15-fold',
        'sign counts gated':        r'180 of 188|178 of 188|33/38',
        'aggregation divergences':  r'six separate points|five separate points|three separate points',
    }
    issues = 0
    for label, pat in PROBES.items():
        hits = defaultdict(list)
        for m in re.finditer(pat, src):
            hits[m.group(0)].append((line_of(src, m.start()), sec_of(m.start())))
        if len(hits) > 1:
            issues += 1
            print(f"\n  {label}: {len(hits)} distinct values")
            for val, locs in sorted(hits.items(), key=lambda kv: -len(kv[1])):
                where = ', '.join(f"{s}@{l}" for l, s in locs[:5])
                print(f"    {val:<22} {len(locs):>2}x  {where}")
    if not issues:
        print("  no probed quantity appears with conflicting values")
    return issues


def check_withdrawals(src, sec_of):
    print("\n" + "=" * 74)
    print("4. WITHDRAWN CLAIMS STILL ASSERTED ELSEWHERE")
    print("=" * 74)
    WITHDRAWN = {
        'alternation-rate signature':
            (r'withdraw|withdrawn', r'alternation rate\s+(?:should|falls|differ)'),
        'non-overlapping distributions':
            (r'withdraw the stronger claim', r'non-overlapping\s+distributions'),
        'Chong ratio quantitative match':
            (r'withdraw', r'match(?:es|ed)? (?:Chong|their) ratio|quantitative match'),
        'AUC reversal as mechanism':
            (r'withdraw', r'classifies in opposite directions'),
        'bound governs both arms':
            (r'did not replicate|withdraw', r'identical ρ|both arms'),
        'anti-gate as magnitude':
            (r'do not report −76\.1%', r'range from \+16\.2% to −76\.1%|to −76\.1%'),
    }
    issues = 0
    for label, (wpat, apat) in WITHDRAWN.items():
        w = [(line_of(src, m.start()), sec_of(m.start()))
             for m in re.finditer(wpat, src, re.I)]
        a = []
        for m in re.finditer(apat, src, re.I):
            window = src[max(0, m.start() - 260):m.start() + 160]
            if re.search(r'withdraw|initially reported|earlier (?:draft|version)|'
                         r'does not replicate|TO SUPPLY|shows is|we do not claim',
                         window, re.I):
                continue
            a.append((line_of(src, m.start()), sec_of(m.start())))
        if a:
            print(f"\n  {label}")
            print(f"    withdrawal language in {len(w)} places")
            print(f"    assertion-shaped text at:")
            for ln, s in a[:6]:
                ctx = src[max(0, src.index('\n', ln) if False else 0):0]
                print(f"      {s}@{ln}")
            issues += 1
    if not issues:
        print("  no withdrawn claim found asserted elsewhere")
    return issues


def check_counts(src, sec_of):
    print("\n" + "=" * 74)
    print("5. 'n OF N' FIGURES SHARING A DENOMINATOR")
    print("=" * 74)
    by_den = defaultdict(set)
    for m in re.finditer(r'(\d+)\s*(?:of|/)\s*(\d{2,3})\b', src):
        num, den = int(m.group(1)), int(m.group(2))
        if num > den:
            print(f"  ** IMPOSSIBLE line {line_of(src, m.start())}: {num} of {den}")
        if den in (30, 38, 40, 44, 100, 162, 166, 170, 188, 189, 195, 196, 200):
            by_den[den].add((num, sec_of(m.start())))
    for den in sorted(by_den):
        vals = sorted(by_den[den])
        print(f"  /{den}: " + ", ".join(f"{n}({s})" for n, s in vals[:14]))
    print("\n  Eyeball for a numerator that should match another and does not.")


def check_superseded(src, sec_of):
    print("\n" + "=" * 74)
    print("6. SUPERSEDED-CAMPAIGN VALUES")
    print("=" * 74)
    SUP = {'+23.6%': 'wave17 gated 2x, pre-lag',
           '−47.7%': 'wave17 anti-gated 2x, pre-lag',
           '+364.5%': 'wave17 suppressed-phase activation',
           '+101.7%': 'wave17 suppressed-phase activation',
           '0.276': 'pre-withdrawal Chong ratio match',
           '0.278': 'pre-withdrawal Chong ratio match'}
    found = 0
    for val, note in SUP.items():
        for m in re.finditer(re.escape(val), src):
            s = sec_of(m.start())
            ctx = src[max(0, m.start()-70):m.start()+18].replace('\n', ' ')
            ok = re.search(r'supersed|withdraw|earlier version|earlier draft|'
                           r'earlier campaign|reported as', ctx, re.I)
            print(f"  {val}  {s}@{line_of(src, m.start())}  "
                  f"[{'flagged as superseded' if ok else '** UNFLAGGED **'}]  {note}")
            found += 1
    if not found:
        print("  no superseded values present")


def check_hedges(src, sec_of):
    print("\n" + "=" * 74)
    print("7. HEDGE CONSISTENCY ON KEY CLAIMS")
    print("=" * 74)
    CLAIMS = {
        'the crossover value':      r'0\.59|0\.40 to 0\.66',
        'architecture generality':  r'four architect|all four|architectural',
        'the exponent repair':      r'superlinear',
        'floor occupancy gating':   r'floor occupancy',
    }
    HEDGE = re.compile(r'\b(may|might|appears|suggests|candidate|preliminary|'
                       r'we do not claim|cannot|only|estimate|order of|'
                       r'exploratory|not established|fitted|not derived|'
                       r'tuned to|generic|representatives|caveat|narrow(?:er)?|'
                       r'restricted|would be stronger|has not been)\b', re.I)
    for label, pat in CLAIMS.items():
        hedged = bare = 0
        for m in re.finditer(pat, src, re.I):
            window = src[max(0, m.start()-220):m.start()+220]
            if HEDGE.search(window):
                hedged += 1
            else:
                bare += 1
        print(f"  {label:<26} {hedged:>3} hedged, {bare:>3} unhedged")
    print("\n  A claim with many unhedged mentions and few hedged ones is stated")
    print("  more confidently in most places than in the one where it is qualified.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'gate-timing-rivalry-v9.md'
    src, heads, sec_of = load(path)
    print(f"{path}: {len(src.split())} words, {len(heads)} numbered sections\n")
    ok_s = check_structure(heads)
    ok_r = check_refs(src, heads, sec_of)
    n_num = check_numbers(src, sec_of)
    n_wd = check_withdrawals(src, sec_of)
    check_counts(src, sec_of)
    check_superseded(src, sec_of)
    check_hedges(src, sec_of)
    print("\n" + "=" * 74)
    print(f"SUMMARY: structure {'ok' if ok_s else 'FAIL'}, "
          f"references {'ok' if ok_r else 'FAIL'}, "
          f"{n_num} conflicting quantities, {n_wd} withdrawal conflicts")
    print("=" * 74)


if __name__ == '__main__':
    main()
