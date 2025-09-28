#!/usr/bin/env python3
"""Convenience wrapper to run the full novelty assessment pipeline end to end.

Given a submission PDF (and optional pre-generated artifacts), this script
handles GROBID conversion, citation enrichment, retrieval, OCR, introduction
extraction, and the LLM-based novelty assessment stages. It mirrors the manual
CLI sequence described in the documentation while providing resumable execution
and basic health checks between stages.

External services (GROBID, MinerU/Nougat OCR, RankGPT) must be running and
configured before invoking this script. Environment variables such as
``OPENAI_API_KEY`` and ``SEMANTIC_SCHOLAR_API_KEY`` are loaded from ``.env`` if
present.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests
from dotenv import load_dotenv


# Ensure the ``src`` directory (this file's parent) is on sys.path for imports
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


from preprocess.extract_metadata import EnhancedGrobidParser
from preprocess import run_ocr
from retrieval import extract_introductions, get_cited_pdfs, match_papers_to_s2, retrieval
from novelty_assessment.pipeline import NoveltyAssessmentPipeline

load_dotenv()
LOGGER = logging.getLogger("full_pipeline_runner")


@dataclass
class Step:
    """Represents a single pipeline stage."""

    name: str
    action: Callable[[], None]
    should_skip: Callable[[], bool]


def configure_logging(verbose: bool) -> None:
    """Configure root logger with consistent formatting."""

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def ensure_submission_workspace(data_dir: Path, submission_id: str) -> Path:
    """Create submission directory if it does not exist."""

    submission_dir = data_dir / submission_id
    submission_dir.mkdir(parents=True, exist_ok=True)
    return submission_dir


def prepare_pdf(submission_dir: Path, submission_id: str, pdf_path: Optional[Path]) -> Path:
    """Ensure the submission PDF is available under the submission directory."""

    if pdf_path is None:
        # Attempt to locate an existing PDF in the submission directory.
        for candidate in (
            submission_dir / f"{submission_id}.pdf",
            submission_dir / "main.pdf",
            submission_dir / "paper.pdf",
        ):
            if candidate.exists():
                LOGGER.info("Using existing PDF: %s", candidate)
                return candidate
        raise FileNotFoundError(
            "No PDF supplied and none found in the submission directory."
        )

    if not pdf_path.exists():
        raise FileNotFoundError(f"Provided PDF does not exist: {pdf_path}")

    destination = submission_dir / f"{submission_id}{pdf_path.suffix.lower()}"

    if pdf_path.resolve() != destination.resolve():
        LOGGER.info("Copying PDF into submission workspace: %s -> %s", pdf_path, destination)
        shutil.copy2(pdf_path, destination)
    else:
        LOGGER.info("PDF already located in submission workspace: %s", destination)

    return destination


def run_grobid(pdf_path: Path, output_path: Path, grobid_url: str, force: bool) -> None:
    """Submit the PDF to a GROBID instance and persist the TEI XML output."""

    if output_path.exists() and not force:
        LOGGER.info("Skipping GROBID conversion (output exists) -> %s", output_path)
        return

    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    LOGGER.info("Sending PDF to GROBID at %s", endpoint)

    data = {"generateIDs": "1", "consolidateHeader": "0", "consolidateCitations": "0"}

    try:
        with pdf_path.open("rb") as handle:
            files = {"input": (pdf_path.name, handle, "application/pdf")}
            response = requests.post(endpoint, files=files, data=data, timeout=3600)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"GROBID request failed: {exc}") from exc

    output_path.write_text(response.text, encoding="utf-8")
    LOGGER.info("GROBID output saved to %s", output_path)


def metadata_needs_update(metadata_path: Path) -> bool:
    """Return True if metadata extraction should run."""

    if not metadata_path.exists():
        return True

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True

    return not data.get("title")


def needs_semantic_scholar_enrichment(metadata_path: Path) -> bool:
    """Check whether cited papers already contain Semantic Scholar metadata."""

    if not metadata_path.exists():
        return True

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True

    cited_papers = data.get("cited_papers", [])
    return not any(paper.get("ss_paper_obj") for paper in cited_papers)


def has_ranked_papers(results_dir: Path) -> bool:
    return (results_dir / "ranked_papers.json").exists()


def has_downloaded_pdfs(pdfs_dir: Path) -> bool:
    return pdfs_dir.exists() and any(pdfs_dir.glob("*.pdf"))


def has_ocr_outputs(ocr_dir: Path) -> bool:
    return ocr_dir.exists() and any(ocr_dir.glob("*.md"))


def has_introductions(intro_dir: Path, submission_id: str) -> bool:
    return (intro_dir / f"{submission_id}_intro.txt").exists()


def has_pipeline_summary(submission_dir: Path) -> bool:
    return (submission_dir / "summary.txt").exists()


def build_steps(
    *,
    submission_id: str,
    submission_dir: Path,
    pdf_path: Path,
    data_dir: Path,
    grobid_url: str,
    mineru_url: str,
    pipeline_model: str,
    temperature: float,
    force: bool,
    ocr_workers: int,
) -> list[Step]:
    """Create the ordered list of pipeline steps."""

    tei_path = submission_dir / f"{submission_id}_fulltext.tei.xml"
    metadata_path = submission_dir / f"{submission_id}.json"
    related_dir = submission_dir / "related_work_data"
    pdfs_dir = related_dir / "pdfs"
    ocr_dir = submission_dir / "ocr"
    intro_dir = submission_dir / "introductions"

    parser = EnhancedGrobidParser()
    pipeline = NoveltyAssessmentPipeline(model_name=pipeline_model, temperature=temperature)

    return [
        Step(
            name="GROBID conversion",
            action=lambda: run_grobid(pdf_path, tei_path, grobid_url, force),
            should_skip=lambda: tei_path.exists() and not force,
        ),
        Step(
            name="TEI metadata extraction",
            action=lambda: parser.process_for_pipeline(str(tei_path), str(data_dir), submission_id),
            should_skip=lambda: not force and not metadata_needs_update(metadata_path),
        ),
        Step(
            name="Semantic Scholar enrichment",
            action=lambda: match_papers_to_s2.process_for_pipeline(str(data_dir), submission_id),
            should_skip=lambda: not force and not needs_semantic_scholar_enrichment(metadata_path),
        ),
        Step(
            name="Paper retrieval & ranking",
            action=lambda: retrieval.process_for_pipeline(str(data_dir), submission_id),
            should_skip=lambda: not force and has_ranked_papers(related_dir),
        ),
        Step(
            name="Download related PDFs",
            action=lambda: get_cited_pdfs.process_for_pipeline(str(data_dir), submission_id),
            should_skip=lambda: not force and has_downloaded_pdfs(pdfs_dir),
        ),
        Step(
            name="OCR processing",
            action=lambda: run_ocr.process_for_pipeline(
                str(data_dir), submission_id, server_url=mineru_url, max_workers=ocr_workers
            ),
            should_skip=lambda: not force and has_ocr_outputs(ocr_dir),
        ),
        Step(
            name="Introduction extraction",
            action=lambda: extract_introductions.process_for_pipeline(str(data_dir), submission_id),
            should_skip=lambda: not force and has_introductions(intro_dir, submission_id),
        ),
        Step(
            name="Novelty assessment pipeline",
            action=lambda: pipeline.run_complete_pipeline(str(data_dir), submission_id),
            should_skip=lambda: not force and has_pipeline_summary(submission_dir),
        ),
    ]


def run_steps(steps: list[Step]) -> None:
    """Execute each step sequentially, ignoring skip checks to run full pipeline."""

    for step in steps:
        # Always run every step - no skipping
        LOGGER.info("Starting step: %s", step.name)
        try:
            step.action()
        except Exception as exc:  # pragma: no cover - runtime integration safeguard
            LOGGER.exception("Step failed: %s", step.name)
            raise RuntimeError(f"Failed at step '{step.name}': {exc}") from exc
        LOGGER.info("Completed step: %s", step.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full novelty assessment pipeline")
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    parser.add_argument("--submission-id", required=True, help="Identifier for the submission")
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Path to the submission PDF. If omitted, the script searches within the submission directory.",
    )
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="Base URL for the GROBID service (default: http://localhost:8070)",
    )
    parser.add_argument(
        "--mineru-url",
        default="http://localhost:8000",
        help="Base URL for the MinerU OCR service (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use for LLM stages (default: gpt-4o)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM calls (default: 0.0)",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=3,
        help="Maximum concurrent OCR workers (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if their outputs already exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    load_dotenv()

    data_dir = Path(args.data_dir).resolve()
    submission_dir = ensure_submission_workspace(data_dir, args.submission_id)

    try:
        pdf_path = prepare_pdf(submission_dir, args.submission_id, args.pdf)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        sys.exit(1)

    steps = build_steps(
        submission_id=args.submission_id,
        submission_dir=submission_dir,
        pdf_path=pdf_path,
        data_dir=data_dir,
        grobid_url=args.grobid_url,
        mineru_url=args.mineru_url,
        pipeline_model=args.model,
        temperature=args.temperature,
        force=args.force,
        ocr_workers=args.ocr_workers,
    )

    LOGGER.info("Starting full novelty assessment pipeline for %s", args.submission_id)
    run_steps(steps)
    LOGGER.info("Pipeline completed. Review outputs in %s", submission_dir)


if __name__ == "__main__":
    main()
