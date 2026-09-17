"""Main OCR pipeline entry point."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from ocr_engine import OCREngine
from preprocess import load_image, preprocess


def run_ocr(
    image_path: str,
    *,
    languages: list[str] | None = None,
    gpu: bool = False,
    preprocess_options: dict | None = None,
    min_confidence: float = 0.0,
    output_format: str = "text",
) -> str:
    """Run the complete OCR pipeline on an image file.

    Args:
        image_path: Path to the input image.
        languages: List of languages for EasyOCR (default: ["en"]).
        gpu: Whether to use GPU acceleration.
        preprocess_options: Keyword arguments for the preprocess function.
        min_confidence: Minimum confidence threshold for text inclusion.
        output_format: Output format - "text", "json", or "lines".

    Returns:
        Extracted text in the requested format.
    """
    if preprocess_options is None:
        preprocess_options = {}

    img = load_image(image_path)
    processed = preprocess(img, **preprocess_options)

    engine = OCREngine(languages=languages, gpu=gpu)
    results = engine.extract(processed)

    if output_format == "json":
        data = [asdict(r) for r in results]
        return json.dumps(data, indent=2)

    if output_format == "lines":
        lines = [
            f"[{r.confidence:.2f}] {r.text}"
            for r in results
            if r.confidence >= min_confidence
        ]
        return "\n".join(lines)

    # default: text
    filtered = [r for r in results if r.confidence >= min_confidence]
    return "\n".join(r.text for r in filtered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OCR pipeline: preprocess image and extract text using EasyOCR"
    )
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        default=["en"],
        help="OCR languages (default: en)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for OCR inference (if available)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "lines"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (default: 0.0)",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip noise reduction",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Skip image upscaling",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Skip skew correction",
    )
    parser.add_argument(
        "--dilate",
        action="store_true",
        help="Apply dilation after thresholding",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Upscale factor (default: 2.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args(argv)

    options = {
        "denoise": not args.no_denoise,
        "resize": not args.no_resize,
        "scale": args.scale,
        "deskew": not args.no_deskew,
        "dilate": args.dilate,
    }

    output = run_ocr(
        args.image,
        languages=args.languages,
        gpu=args.gpu,
        preprocess_options=options,
        min_confidence=args.min_confidence,
        output_format=args.format,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
