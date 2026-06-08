from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from src.extraction.providers.gemini import GeminiExtractor, GeminiParams
from src.ocr.ocr import OCRPipeline
from src.postprocessing.autocorrect import AutoCorrector
from src.preprocessing.preprocess import Preprocessing


logging.getLogger("ppocr").setLevel(logging.ERROR)

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _safe_name(path: Path) -> str:
    try:
        display_path = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        display_path = path.resolve()
    return (
        display_path.as_posix()
        .replace(":", "")
        .replace("/", "__")
        .replace("\\", "__")
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _save_pack(pack: dict, output_dir: Path, file_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_name}_diagnostic.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(pack, file, ensure_ascii=False, indent=4, default=_json_default)
    return output_path


def iter_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_preprocessing_case(
    image_path: Path,
    preprocessor: Preprocessing,
    output_dir: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    case_name = _safe_name(image_path)
    image_output_dir = output_dir / "images"
    json_output_dir = output_dir / "diagnostics"

    pack: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "image_file": str(image_path),
        "mode": "preprocess",
        "stages": {},
    }

    row: dict[str, Any] = {
        "file": str(image_path),
        "status": "error",
        "decision_label": "",
        "decision_confidence": "",
        "qrcode_count": 0,
        "processed_image": "",
        "diagnostic_json": "",
        "elapsed_ms": 0,
        "error": "",
    }

    try:
        result = preprocessor.process(str(image_path))
        processed_path = image_output_dir / f"{case_name}_preproc.png"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(processed_path), result.image)

        qrcodes = result.metadata.get("qrcodes", [])
        row.update(
            {
                "status": "success",
                "decision_label": result.decision.label_name if result.decision else "",
                "decision_confidence": result.decision.confidence if result.decision else "",
                "qrcode_count": len(qrcodes),
                "processed_image": str(processed_path),
            }
        )
        pack["stages"]["preprocessing"] = {
            "status": "success",
            "decision_label": row["decision_label"],
            "decision_confidence": row["decision_confidence"],
            "metadata": result.metadata,
            "processed_image": str(processed_path),
        }
    except Exception as exc:
        row["error"] = str(exc)
        pack["stages"]["preprocessing"] = {
            "status": "error",
            "message": str(exc),
        }

    row["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
    diagnostic_path = _save_pack(pack, json_output_dir, case_name)
    row["diagnostic_json"] = str(diagnostic_path)
    return row


def write_summary_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "file",
        "status",
        "decision_label",
        "decision_confidence",
        "qrcode_count",
        "processed_image",
        "diagnostic_json",
        "elapsed_ms",
        "error",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return summary_path


def write_html_report(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    report_path = output_dir / "report.html"
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Preprocessing QA Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#222}",
        "table{border-collapse:collapse;width:100%;background:#fff}",
        "th,td{border:1px solid #ddd;padding:8px;vertical-align:top;font-size:13px}",
        "th{background:#eee;text-align:left;position:sticky;top:0}",
        "img{max-width:220px;max-height:260px;border:1px solid #ccc;background:#fff}",
        ".success{color:#087a2f;font-weight:bold}.error{color:#b00020;font-weight:bold}",
        "</style></head><body>",
        "<h1>Preprocessing QA Report</h1>",
        f"<p>Generated: {html.escape(datetime.now().isoformat())}</p>",
        "<table><thead><tr>",
        "<th>Input</th><th>Processed</th><th>Status</th><th>Label</th><th>QR</th><th>Time</th><th>Error</th>",
        "</tr></thead><tbody>",
    ]

    for row in rows:
        input_path = Path(row["file"])
        processed = row.get("processed_image") or ""
        status = row["status"]
        status_class = "success" if status == "success" else "error"

        input_src = os.path.relpath(input_path, output_dir).replace("\\", "/")
        processed_html = ""
        if processed:
            processed_src = os.path.relpath(processed, output_dir).replace("\\", "/")
            processed_html = f"<img src='{html.escape(processed_src)}'>"

        lines.extend(
            [
                "<tr>",
                f"<td><div>{html.escape(input_path.name)}</div><img src='{html.escape(input_src)}'></td>",
                f"<td>{processed_html}</td>",
                f"<td class='{status_class}'>{html.escape(status)}</td>",
                f"<td>{html.escape(str(row['decision_label']))}</td>",
                f"<td>{html.escape(str(row['qrcode_count']))}</td>",
                f"<td>{html.escape(str(row['elapsed_ms']))} ms</td>",
                f"<td>{html.escape(str(row['error']))}</td>",
                "</tr>",
            ]
        )

    lines.extend(["</tbody></table>", "</body></html>"])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_preprocessing_batch(input_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    files = iter_input_images(input_path)
    preprocessor = Preprocessing(provider=None)
    rows = [run_preprocessing_case(path, preprocessor, output_dir) for path in files]

    summary_path = write_summary_csv(rows, output_dir)
    report_path = write_html_report(rows, output_dir)

    success_count = sum(1 for row in rows if row["status"] == "success")
    error_count = len(rows) - success_count
    print(f"Processed {len(rows)} file(s): {success_count} success, {error_count} error")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return rows


def generate_diagnostic_pack(
    image_path: str,
    api_key: str | None = None,
    output_dir: str = "data/diagnostics",
    run_extraction: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    source_path = Path(image_path)
    file_name = source_path.stem

    diagnostic_pack: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "image_file": source_path.name,
        "mode": "full",
        "stages": {},
    }

    preprocessor = Preprocessing()
    try:
        pre_result = preprocessor.process(str(source_path))
        diagnostic_pack["stages"]["preprocessing"] = {
            "status": "success",
            "decision_label": pre_result.decision.label_name if pre_result.decision else None,
            "decision_confidence": pre_result.decision.confidence if pre_result.decision else None,
            "metadata": pre_result.metadata,
        }
        process_image = pre_result.image
    except Exception as exc:
        diagnostic_pack["stages"]["preprocessing"] = {"status": "error", "message": str(exc)}
        _save_pack(diagnostic_pack, output_path, file_name)
        return diagnostic_pack

    ocr_pipeline = OCRPipeline(engine="tesseract", threshold=0.8)
    try:
        ocr_result = ocr_pipeline.run(process_image)
        texts = [word.text for word in ocr_result.words]
        confidences = [word.confidence for word in ocr_result.words]

        diagnostic_pack["stages"]["ocr"] = {
            "status": "success",
            "overall_confidence": ocr_result.overall_confidence,
            "raw_text_length": len(ocr_result.raw_text),
            "word_count": len(texts),
            "words": texts,
            "confidences": confidences,
        }
    except Exception as exc:
        diagnostic_pack["stages"]["ocr"] = {"status": "error", "message": str(exc)}
        _save_pack(diagnostic_pack, output_path, file_name)
        return diagnostic_pack

    corrector = AutoCorrector(enabled=True)
    try:
        final_result = corrector.correct_list(texts, confidences)
        final_text = " ".join(final_result.corrected_texts)

        diagnostic_pack["stages"]["autocorrect"] = {
            "status": "success",
            "corrected_count": final_result.corrected_count,
            "final_text": final_text,
            "corrections_made": [
                {"original": item.original_text, "corrected": item.corrected_text, "reason": item.reason}
                for item in final_result.corrections
            ],
        }
    except Exception as exc:
        diagnostic_pack["stages"]["autocorrect"] = {"status": "error", "message": str(exc)}
        final_text = " ".join(texts)

    if run_extraction and api_key:
        extractor = GeminiExtractor(api_key=api_key, params=GeminiParams())
        try:
            extraction_result = extractor.extract(final_text)
            diagnostic_pack["stages"]["extraction"] = {
                "status": "success",
                "schema_used": extraction_result.schema_used,
                "confidence_overall": extraction_result.confidence_overall,
                "fill_rate": extraction_result.fill_rate,
                "extracted_data": extraction_result.extracted_dynamic_data,
            }
        except Exception as exc:
            diagnostic_pack["stages"]["extraction"] = {"status": "error", "message": str(exc)}

    diagnostic_path = _save_pack(diagnostic_pack, output_path, file_name)
    print(f"Diagnostic pack: {diagnostic_path}")
    return diagnostic_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR diagnostics or fast preprocessing QA.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/test_cases"),
        help="Image file or directory to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/test_results/preprocessing_manual"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "full"],
        default="preprocess",
        help="Use preprocess for fast manual tuning; full also runs OCR/autocorrect.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="In full mode, also run Gemini extraction when GEMINI_API_KEY is set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "preprocess":
        run_preprocessing_batch(args.input, args.output_dir)
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        for image_file in iter_input_images(args.input):
            generate_diagnostic_pack(
                str(image_file),
                api_key=api_key,
                output_dir=str(args.output_dir),
                run_extraction=args.extract,
            )
