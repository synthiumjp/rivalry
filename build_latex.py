"""
build_latex.py — turn the markdown manuscript into a LaTeX submission package.

Produces:
    build/manuscript.tex     the converted source, editable
    build/manuscript.pdf     compiled
    build/figures/*.pdf      copied
    manuscript-latex.tar.gz  the lot, ready to upload

The markdown carries HTML sub and sup tags for variable subscripts and affiliation
markers, which pandoc passes through as raw HTML and LaTeX then ignores. They are
converted to math mode and \\textsuperscript before conversion. Tables are booktabs.

USAGE
    python build_latex.py                # convert and compile
    python build_latex.py --no-pdf       # convert only
    python build_latex.py --engine xelatex
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

SRC = 'gate-timing-rivalry-v9.md'
BUILD = Path('build')

PREAMBLE = r"""
\usepackage[T1]{fontenc}
% lmodern is absent in minimal TeX installations. Without a scalable font,
% microtype's expansion cannot run, so it is disabled where lmodern is missing.
\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{\microtypesetup{expansion=false}}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage[hidelinks]{hyperref}
\IfFileExists{caption.sty}{\usepackage{caption}}{}
\IfFileExists{ragged2e.sty}{\usepackage{ragged2e}}{}

% tables in this manuscript are wide and text-heavy; let them wrap and shrink
\IfFileExists{tabularx.sty}{\usepackage{tabularx}}{}
\IfFileExists{caption.sty}{\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}}{}
\setlength{\LTcapwidth}{\textwidth}
\IfFileExists{etoolbox.sty}{\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small\setlength{\tabcolsep}{4pt}}}{}

\setlength{\parskip}{0.4\baselineskip}
\setlength{\parindent}{0pt}
\raggedbottom
\widowpenalty=10000
\clubpenalty=10000

\IfFileExists{titlesec.sty}{\usepackage{titlesec}
\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\itshape}{\thesubsubsection}{0.6em}{}}{}
"""


def preprocess(text):
    """Markdown adjustments that must happen before pandoc.

    Unicode is deliberately NOT touched here. Emitting $...$ into markdown makes
    pandoc escape the dollars inside table cells, which then breaks math mode in
    LaTeX. Unicode is replaced in the generated .tex instead, where nothing else
    will re-escape it.
    """
    # pandoc's own subscript and superscript syntax, which it converts correctly
    text = re.sub(r'<sub>([^<]{1,12})</sub>', lambda m: '~' + m.group(1).replace(' ', r'\ ') + '~', text)
    text = re.sub(r'<sup>([^<]{1,12})</sup>', lambda m: '^' + m.group(1).replace(' ', r'\ ') + '^', text)

    lines = text.split('\n')
    title = lines[0].lstrip('# ').strip()
    subtitle = lines[1].lstrip('# ').strip() if lines[1].startswith('##') else ''
    body_start = next((i for i, l in enumerate(lines) if l.strip() == '---'), None)
    if body_start is None:                     # supplement: body starts at its first section
        body_start = next(i for i, l in enumerate(lines) if l.startswith('## S')) - 1
    header = '\n'.join(lines[2:body_start])
    authors = re.findall(r'\*\*([^*]+)\*\*\^', header)
    affil = [l for l in header.split('\n') if l.strip().startswith('^')]
    credit = ''
    m = re.search(r'\*\*Author contributions \(CRediT\)\.\*\*(.*?)(?=\n\n|\Z)',
                  header, re.S)
    if m:
        credit = ' '.join(m.group(1).split())
    text = '\n'.join(lines[body_start + 1:])

    def figure(m):
        num, path, cap = m.group(1), m.group(2), m.group(3).strip()
        return (f"\n```{{=latex}}\n\\begin{{figure}}[htbp]\n\\centering\n"
                f"\\includegraphics[width=\\textwidth]{{{path}}}\n"
                f"\\caption{{CAPTION{num}}}\n\\label{{fig:{num}}}\n"
                f"\\end{{figure}}\n```\n")
    caps = {}
    for m in re.finditer(r'!\[Figure (\d)\]\([^)]+\)\s*\n\s*\n\*\*Figure \d\.\*\* ([^\n]+(?:\n(?!\n)[^\n]+)*)', text):
        caps[m.group(1)] = ' '.join(m.group(2).split())
    text = re.sub(r'!\[Figure (\d)\]\(([^)]+)\)\s*\n\s*\n\*\*Figure \d\.\*\* ([^\n]+(?:\n(?!\n)[^\n]+)*)',
                  figure, text)
    return title, subtitle, authors, affil, credit, text, caps


UNI = {
    '\u2212': '$-$', '\u2013': '--', '\u00b7': r'$\cdot$', '\u00d7': r'$\times$',
    '\u00b1': r'$\pm$', '\u2248': r'$\approx$', '\u2260': r'$\neq$',
    '\u2265': r'$\geq$', '\u2264': r'$\leq$', '\u2208': r'$\in$',
    '\u221e': r'$\infty$', '\u2192': r'$\rightarrow$', '\u2032': r"$'$",
    '\u2026': r'\ldots{}', '\u03bb': r'$\lambda$', '\u03c1': r'$\rho$',
    '\u03ba': r'$\kappa$', '\u03b1': r'$\alpha$', '\u03b2': r'$\beta$',
    '\u03b3': r'$\gamma$', '\u03c3': r'$\sigma$', '\u03c4': r'$\tau$',
    '\u03b8': r'$\theta$', '\u03b7': r'$\eta$', '\u03bc': r'$\mu$',
    '\u03b4': r'$\delta$', '\u03c7': r'$\chi$', '\u0394': r'$\Delta$',
    '\U0001d4a9': r'$\mathcal{N}$', '\u00b2': r'\textsuperscript{2}',
    '\u2070': r'\textsuperscript{0}', '\u00b9': r'\textsuperscript{1}',
    '\u207b': r'\textsuperscript{$-$}', '\u2080': r'\textsubscript{0}',
    '\u00e1': r"\'a", '\u00ed': r"\'i", '\u00e9': r"\'e",
    '\u00fc': r'\"u', '\u00f6': r'\"o', '\u0304': '',
}


def de_unicode(tex):
    """Replace unicode in the generated LaTeX, after pandoc has finished escaping."""
    # pandoc escapes a literal dollar as \$; none of ours should survive as text
    for a, b in UNI.items():
        tex = tex.replace(a, b)
    tex = tex.replace(r'\$-\$', '$-$')
    left = sorted({c for c in tex if ord(c) > 127})
    if left:
        print(f"    unmapped non-ASCII, stripped: {left[:12]}")
        for c in left:
            tex = tex.replace(c, '')
    return tex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-pdf', action='store_true')
    ap.add_argument('--engine', default='pdflatex')
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--supplement', action='store_true',
                    help='number figures, tables and equations S1, S2, ...')
    ap.add_argument('--name', default='manuscript', help='output file stem')
    args = ap.parse_args()

    if not Path(args.src).exists():
        sys.exit(f"{args.src} not found. Run from the directory containing it.")
    BUILD.mkdir(exist_ok=True)
    (BUILD / 'figures').mkdir(exist_ok=True)
    figs = list(Path('figures').glob('*.pdf'))
    for f in figs:
        shutil.copy(f, BUILD / 'figures' / f.name)
    print(f"  {len(figs)} figures copied")

    title, subtitle, authors, affil, credit, body, caps = preprocess(
        Path(args.src).read_text(encoding='utf-8'))
    print(f"  title: {title[:60]}")
    print(f"  authors: {', '.join(authors) if authors else 'none parsed'}")

    pre = PREAMBLE + ((r"\renewcommand{\thefigure}{S\arabic{figure}}" "\n"
                       r"\renewcommand{\thetable}{S\arabic{table}}" "\n"
                       r"\renewcommand{\theequation}{S\arabic{equation}}" "\n")
                      if args.supplement else '')
    (BUILD / 'preamble.tex').write_text(pre, encoding='utf-8')
    (BUILD / 'body.md').write_text(body, encoding='utf-8')

    meta = [
        '---',
        f'title: "{title}"',
        f'subtitle: "{subtitle}"' if subtitle else '',
        'author:',
    ]
    for a in (authors or ['Author']):
        meta.append(f'  - "{a}"')
    meta += ['documentclass: article', 'fontsize: 11pt', 'linestretch: 1.05',
             'colorlinks: false', '---', '']
    (BUILD / 'meta.yaml').write_text('\n'.join(x for x in meta if x != ''),
                                     encoding='utf-8')

    cmd = ['pandoc', 'meta.yaml', 'body.md', '-o', f'{args.name}.tex',
           '--standalone', '--include-in-header', 'preamble.tex',
           '--from', 'markdown+pipe_tables+raw_tex',
           '--to', 'latex', '--top-level-division=section']
    r = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True)
    if r.returncode != 0:
        print("  pandoc failed:\n", r.stderr[:1200])
        return
    tex = (BUILD / f'{args.name}.tex').read_text(encoding='utf-8')

    # captions are extracted before pandoc runs, so they still hold markdown.
    # Convert each through pandoc rather than hand-rolling italics and subscripts.
    def md_to_tex(s):
        r = subprocess.run(['pandoc', '-f', 'markdown', '-t', 'latex'],
                           input=s, capture_output=True, text=True)
        out = r.stdout.strip() if r.returncode == 0 else s
        return de_unicode(' '.join(out.split()))
    for num, cap in caps.items():
        tex = tex.replace(f'CAPTION{num}', md_to_tex(cap))
    tex = de_unicode(tex)

    # pandoc's own template loads packages that are not in every TeX installation.
    # Guard every \usepackage it emits so a missing one degrades instead of aborting.
    def guard(m):
        opts, pkg = m.group(1) or '', m.group(2)
        if ',' in pkg:
            return m.group(0)
        probe = subprocess.run(['kpsewhich', pkg + '.sty'],
                               capture_output=True, text=True)
        if probe.stdout.strip():
            return m.group(0)
        print(f"    {pkg} absent, guarded")
        return (f"\\IfFileExists{{{pkg}.sty}}{{\\usepackage{opts}{{{pkg}}}}}{{}}")
    tex = re.sub(r'\\usepackage(\[[^\]]*\])?\{([a-zA-Z0-9\-]+)\}', guard, tex)
    (BUILD / f'{args.name}.tex').write_text(tex, encoding='utf-8')
    print(f"  manuscript.tex written, {len(tex.splitlines())} lines")

    if affil or credit:
        extra = de_unicode('\\\\\n'.join(a.replace('^', '\\textsuperscript{', 1)
                                       .replace('^', '}', 1) for a in affil))
        block = (f"\n\\begin{{center}}\\small\n{extra}\n"
                 + (f"\n\\vspace{{0.5em}}\\textbf{{Author contributions (CRediT).}} "
                    f"{de_unicode(credit)}\n" if credit else '')
                 + "\\end{center}\n")
        # anchor on the real \maketitle in the body, not the one mentioned in a
        # comment inside pandoc's \subtitle definition in the preamble
        anchor = '\\begin{document}\n\\maketitle'
        if anchor in tex:
            tex = tex.replace(anchor, anchor + '\n' + block, 1)
        else:
            i = tex.index('\\begin{document}')
            j = tex.index('\\maketitle', i)
            tex = tex[:j + len('\\maketitle')] + '\n' + block + tex[j + len('\\maketitle'):]
        (BUILD / f'{args.name}.tex').write_text(tex, encoding='utf-8')
        print("  affiliations and CRediT inserted after \\maketitle")

    if not args.no_pdf:
        print(f"\n  compiling with {args.engine}, three passes")
        for i in range(3):
            r = subprocess.run([args.engine, '-interaction=nonstopmode',
                                '-halt-on-error', f'{args.name}.tex'],
                               cwd=BUILD, capture_output=True, text=True)
            if r.returncode != 0:
                log = (BUILD / f'{args.name}.log')
                errs = []
                if log.exists():
                    for line in log.read_text(errors='ignore').split('\n'):
                        if line.startswith('!') or 'Undefined control' in line:
                            errs.append(line)
                print(f"  pass {i+1} failed:")
                for e in errs[:10]:
                    print("   ", e[:110])
                if not errs:
                    print("   ", r.stdout[-800:])
                return
        pdf = BUILD / f'{args.name}.pdf'
        if pdf.exists():
            print(f"  manuscript.pdf, {pdf.stat().st_size/1024:.0f} KB")

    tar = f'{args.name}-latex.tar.gz'
    with tarfile.open(tar, 'w:gz') as t:
        for f in (f'{args.name}.tex', 'preamble.tex', f'{args.name}.pdf'):
            p = BUILD / f
            if p.exists():
                t.add(p, arcname=f'{args.name}/{f}')
        for f in sorted((BUILD / 'figures').glob('*.pdf')):
            t.add(f, arcname=f'{args.name}/figures/{f.name}')
        readme = BUILD / 'HOWTO.txt'
        readme.write_text(
            "LaTeX submission package\n"
            "========================\n\n"
            "  manuscript.tex   converted from the markdown source\n"
            "  preamble.tex     included in the header, edit for a journal class\n"
            "  figures/         eight figures as PDF\n\n"
            "Compile:\n\n"
            "  pdflatex manuscript.tex   (three times, for cross-references)\n\n"
            "To target a journal class, replace \\documentclass in manuscript.tex\n"
            "and strip the geometry and titlesec lines from preamble.tex, which\n"
            "most journal classes set themselves.\n", encoding='utf-8')
        t.add(readme, arcname=f'{args.name}/HOWTO.txt')
    print(f"\n  wrote {tar}, {Path(tar).stat().st_size/1024:.0f} KB")


if __name__ == '__main__':
    main()
