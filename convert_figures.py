#!/usr/bin/env python3
"""
Convert GC-LCA Paper 1 figure PDFs to TIFF at 300 DPI.
Psych Review requires TIFF or JPG; 300 DPI for colour, 600 DPI for B&W.

Usage:
    python convert_figures.py [--input-dir C:\crewther] [--output-dir C:\crewther\tiff]

Requirements:
    pip install pdf2image Pillow
    Also requires poppler: https://github.com/osama-esh/poppler-windows/releases
    Add poppler bin/ to PATH, or pass poppler_path below.

Author: JP Cacioli
Date: March 2026
"""

import argparse
import sys
from pathlib import Path

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("Install requirements: pip install pdf2image Pillow")
    print("Also install poppler (see https://github.com/osama-esh/poppler-windows/releases)")
    sys.exit(1)


# Figure inventory — maps figure number to filename and DPI
# Colour figures at 300 DPI; B&W line art at 600 DPI
FIGURES = {
    1:  {"file": "fig1_model.pdf",           "dpi": 300, "desc": "Model architecture + rivalry traces"},
    2:  {"file": "fig2_parameter_space.pdf",  "dpi": 300, "desc": "CV heatmap, Levelt rho, grid summary"},
    3:  {"file": "fig3_levelt.pdf",           "dpi": 300, "desc": "Levelt Props I-IV with bootstrap CIs"},
    4:  {"file": "fig4_h4_headline.pdf",      "dpi": 300, "desc": "HEADLINE: Non-target duration + DPR"},
    5:  {"file": "fig5_dissociation.pdf",     "dpi": 300, "desc": "Dose-response, dorsal vs ventral"},
    6:  {"file": "fig6_mechanism.pdf",        "dpi": 300, "desc": "Rectifier, trajectory, ablations"},
    7:  {"file": "fig7_phase_portrait.pdf",   "dpi": 300, "desc": "Phase portraits (A) + bifurcation (B)"},
    8:  {"file": "fig8_sensitivity.pdf",      "dpi": 300, "desc": "Sensitivity / robustness analysis"},
    9:  {"file": "fig9_kappa_role.pdf",       "dpi": 300, "desc": "CV, controllability vs kappa"},
}

# Fig 7b gets merged into Fig 7 as panel B
MERGE_7B = "fig7b_bifurcation.pdf"


def convert_single(pdf_path: Path, output_path: Path, dpi: int = 300,
                   poppler_path: str = None):
    """Convert a single-page PDF to TIFF at specified DPI."""
    kwargs = {"dpi": dpi, "fmt": "tiff"}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    images = convert_from_path(str(pdf_path), **kwargs)
    if not images:
        print(f"  WARNING: No pages found in {pdf_path.name}")
        return False

    # Use first page (figures should be single-page)
    img = images[0]

    # Convert to RGB if needed (TIFF supports various modes)
    if img.mode == "RGBA":
        # White background for transparency
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Save as TIFF with LZW compression (lossless, smaller files)
    img.save(str(output_path), format="TIFF", compression="tiff_lzw",
             dpi=(dpi, dpi))
    return True


def merge_panels(panel_a_path: Path, panel_b_path: Path, output_path: Path,
                 dpi: int = 300, orientation: str = "vertical",
                 poppler_path: str = None):
    """
    Merge two single-page PDFs into one TIFF (for Fig 7A + 7B).
    orientation: 'vertical' stacks A on top of B; 'horizontal' places side by side.
    """
    kwargs = {"dpi": dpi, "fmt": "tiff"}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    imgs_a = convert_from_path(str(panel_a_path), **kwargs)
    imgs_b = convert_from_path(str(panel_b_path), **kwargs)

    if not imgs_a or not imgs_b:
        print(f"  WARNING: Could not load panels for merging")
        return False

    a = imgs_a[0].convert("RGB")
    b = imgs_b[0].convert("RGB")

    if orientation == "vertical":
        # Stack vertically: same width, heights add up
        max_w = max(a.width, b.width)
        # Resize to same width if needed
        if a.width != max_w:
            ratio = max_w / a.width
            a = a.resize((max_w, int(a.height * ratio)), Image.LANCZOS)
        if b.width != max_w:
            ratio = max_w / b.width
            b = b.resize((max_w, int(b.height * ratio)), Image.LANCZOS)

        gap = int(dpi * 0.15)  # ~0.15 inch gap between panels
        merged = Image.new("RGB", (max_w, a.height + gap + b.height),
                          (255, 255, 255))
        merged.paste(a, (0, 0))
        merged.paste(b, (0, a.height + gap))
    else:
        # Side by side
        max_h = max(a.height, b.height)
        if a.height != max_h:
            ratio = max_h / a.height
            a = a.resize((int(a.width * ratio), max_h), Image.LANCZOS)
        if b.height != max_h:
            ratio = max_h / b.height
            b = b.resize((int(b.width * ratio), max_h), Image.LANCZOS)

        gap = int(dpi * 0.15)
        merged = Image.new("RGB", (a.width + gap + b.width, max_h),
                          (255, 255, 255))
        merged.paste(a, (0, 0))
        merged.paste(b, (a.width + gap, 0))

    merged.save(str(output_path), format="TIFF", compression="tiff_lzw",
                dpi=(dpi, dpi))
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert Paper 1 figure PDFs to TIFF for Psych Review submission"
    )
    parser.add_argument("--input-dir", type=str, default=".",
                       help="Directory containing figure PDFs (default: current dir)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for TIFFs (default: input-dir/tiff)")
    parser.add_argument("--poppler-path", type=str, default=None,
                       help="Path to poppler bin/ directory (Windows only)")
    parser.add_argument("--merge-7", choices=["vertical", "horizontal", "skip"],
                       default="vertical",
                       help="How to merge fig7 + fig7b (default: vertical)")
    parser.add_argument("--also-jpg", action="store_true",
                       help="Also save JPG versions (95%% quality)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "tiff"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    success = 0
    failed = 0

    for fig_num, info in FIGURES.items():
        pdf_path = input_dir / info["file"]
        tiff_name = f"Figure{fig_num}.tiff"
        tiff_path = output_dir / tiff_name

        print(f"Figure {fig_num}: {info['desc']}")

        if fig_num == 7 and args.merge_7 != "skip":
            # Special case: merge fig7 + fig7b
            panel_b_path = input_dir / MERGE_7B
            if not pdf_path.exists():
                print(f"  SKIP: {pdf_path.name} not found")
                failed += 1
                continue
            if not panel_b_path.exists():
                print(f"  WARNING: {MERGE_7B} not found, converting fig7 alone")
                ok = convert_single(pdf_path, tiff_path, info["dpi"],
                                   args.poppler_path)
            else:
                print(f"  Merging {info['file']} (A) + {MERGE_7B} (B) [{args.merge_7}]")
                ok = merge_panels(pdf_path, panel_b_path, tiff_path,
                                 info["dpi"], args.merge_7, args.poppler_path)
        else:
            if not pdf_path.exists():
                print(f"  SKIP: {pdf_path.name} not found")
                failed += 1
                continue
            ok = convert_single(pdf_path, tiff_path, info["dpi"],
                               args.poppler_path)

        if ok:
            size_mb = tiff_path.stat().st_size / (1024 * 1024)
            print(f"  -> {tiff_name} ({size_mb:.1f} MB, {info['dpi']} DPI)")
            success += 1

            if args.also_jpg:
                jpg_path = output_dir / f"Figure{fig_num}.jpg"
                img = Image.open(tiff_path)
                img.save(str(jpg_path), format="JPEG", quality=95,
                        dpi=(info["dpi"], info["dpi"]))
                jpg_mb = jpg_path.stat().st_size / (1024 * 1024)
                print(f"  -> Figure{fig_num}.jpg ({jpg_mb:.1f} MB)")
        else:
            failed += 1

    print()
    print(f"Done: {success} converted, {failed} skipped/failed")
    print()
    print("Submission checklist:")
    print("  [ ] Fig 1A model schematic — redo in Inkscape before final submission")
    print("  [ ] Check all figures render correctly in Word after insertion")
    print("  [ ] Psych Review: TIFF or JPG; 300 DPI colour, 600 DPI B&W")


if __name__ == "__main__":
    main()
