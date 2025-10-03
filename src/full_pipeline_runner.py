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

PRIMARY_PDF_MARKER = ".primary_pdf"


def _primary_pdf_marker(submission_dir: Path) -> Path:
    """Returns the Path to the primary PDF marker file."""
    return submission_dir / PRIMARY_PDF_MARKER


def _read_recorded_pdf(submission_dir: Path) -> Optional[Path]:
    """Reads the primary PDF marker file and returns the path if it exists."""
    marker = _primary_pdf_marker(submission_dir)
    if not marker.exists():
        return None

    try:
        recorded_name = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        LOGGER.warning("Unable to read primary PDF marker %s: %s", marker, exc)
        return None

    if not recorded_name:
        return None

    candidate = submission_dir / recorded_name
    return candidate if candidate.exists() else None


def _write_recorded_pdf(submission_dir: Path, pdf_path: Path) -> None:
    """Writes the name of the primary PDF to the marker file."""
    marker = _primary_pdf_marker(submission_dir)
    try:
        marker.write_text(pdf_path.name, encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Unable to write primary PDF marker %s: %s", marker, exc)


@dataclass
class Step:
    """Represents a single, executable stage in the processing pipeline.

    Attributes:
        name: The human-readable name of the pipeline step.
        action: A callable (function or lambda) that executes the step's logic.
        should_skip: A callable that returns `True` if the step can be skipped,
                     typically because its output already exists.
    """

    name: str
    action: Callable[[], None]
    should_skip: Callable[[], bool]


def configure_logging(verbose: bool) -> None:
    """Configures the root logger for the application.

    Sets the logging level and format for console output.

    Args:
        verbose: If `True`, sets the logging level to DEBUG; otherwise, INFO.
    """

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def ensure_submission_workspace(data_dir: Path, submission_id: str) -> Path:
    """Ensures that the necessary directory for a submission exists.

    Args:
        data_dir: The base directory for all data.
        submission_id: The unique identifier for the submission.

    Returns:
        The `Path` object for the submission's directory.
    """

    submission_dir = data_dir / submission_id
    submission_dir.mkdir(parents=True, exist_ok=True)
    return submission_dir


def prepare_pdf(submission_dir: Path, submission_id: str, pdf_path: Optional[Path]) -> Path:
    """Ensures the main submission PDF is present in the workspace.

    If a PDF path is provided, it's copied into the submission directory.
    If not, the function tries to find an existing PDF based on a marker file
    or common naming conventions.

    Args:
        submission_dir: The directory for the submission.
        submission_id: The unique identifier for the submission.
        pdf_path: An optional path to the source PDF file.

    Returns:
        The `Path` to the submission PDF within the workspace.

    Raises:
        FileNotFoundError: If no PDF is provided and none can be found in the
                           submission directory.
    """

    if pdf_path is None:
        recorded = _read_recorded_pdf(submission_dir)
        if recorded is not None:
            LOGGER.info("Using recorded primary PDF: %s", recorded)
            return recorded

        # Attempt to locate an existing PDF in the submission directory.
        for candidate in (
            submission_dir / f"{submission_id}.pdf",
            submission_dir / "main.pdf",
            submission_dir / "paper.pdf",
        ):
            if candidate.exists():
                LOGGER.info("Using existing PDF: %s", candidate)
                _write_recorded_pdf(submission_dir, candidate)
                return candidate
        raise FileNotFoundError(
            "No PDF supplied and none found in the submission directory."
        )

    if not pdf_path.exists():
        raise FileNotFoundError(f"Provided PDF does not exist: {pdf_path}")

    destination = submission_dir / pdf_path.name

    if pdf_path.resolve() != destination.resolve():
        if destination.exists():
            LOGGER.info("Overwriting existing PDF in workspace: %s", destination)
        else:
            LOGGER.info("Copying PDF into submission workspace: %s -> %s", pdf_path, destination)
        shutil.copy2(pdf_path, destination)
    else:
        LOGGER.info("PDF already located in submission workspace: %s", destination)

    _write_recorded_pdf(submission_dir, destination)

    return destination


def run_grobid(pdf_path: Path, output_path: Path, grobid_url: str, force: bool) -> None:
    """Processes a PDF with a GROBID service to generate TEI XML.

    Args:
        pdf_path: The path to the input PDF file.
        output_path: The path where the output TEI XML file will be saved.
        grobid_url: The base URL of the running GROBID service.
        force: If `True`, re-runs the conversion even if the output file exists.

    Raises:
        RuntimeError: If the request to the GROBID service fails.
    """

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
    """Checks if the metadata file needs to be (re)generated.

    Returns `True` if the file doesn't exist, is invalid JSON, or lacks a title.

    Args:
        metadata_path: The path to the submission's metadata JSON file.

    Returns:
        A boolean indicating whether the metadata extraction step should run.
    """

    if not metadata_path.exists():
        return True

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True

    return not data.get("title")


def needs_semantic_scholar_enrichment(metadata_path: Path) -> bool:
    """Checks if the cited papers in the metadata file need S2 enrichment.

    Returns `True` if the metadata file is missing or if none of the cited papers
    have the `ss_paper_obj` key, which indicates they haven't been enriched.

    Args:
        metadata_path: The path to the submission's metadata JSON file.

    Returns:
        A boolean indicating whether the enrichment step should run.
    """

    if not metadata_path.exists():
        return True

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True

    cited_papers = data.get("cited_papers", [])
    return not any(paper.get("ss_paper_obj") for paper in cited_papers)


def has_ranked_papers(results_dir: Path) -> bool:
    """Checks if the output file for ranked papers exists."""
    return (results_dir / "ranked_papers.json").exists()


def has_downloaded_pdfs(pdfs_dir: Path) -> bool:
    """Checks if any PDFs have been downloaded for related works."""
    return pdfs_dir.exists() and any(pdfs_dir.glob("*.pdf"))


def has_ocr_outputs(ocr_dir: Path) -> bool:
    """Checks if any markdown OCR output files exist."""
    return ocr_dir.exists() and any(ocr_dir.glob("*.md"))


def has_introductions(intro_dir: Path, submission_id: str) -> bool:
    """Checks if the introduction for the main submission has been extracted."""
    return (intro_dir / f"{submission_id}_intro.txt").exists()


def has_pipeline_summary(submission_dir: Path) -> bool:
    """Checks if the final summary file has been generated."""
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
    """Constructs the full list of pipeline steps in execution order.

    Each step is defined with its name, the action to perform, and a function
    to determine if it should be skipped.

    Args:
        submission_id: The unique identifier for the submission.
        submission_dir: The path to the submission's workspace directory.
        pdf_path: The path to the main submission PDF.
        data_dir: The base data directory.
        grobid_url: The URL for the GROBID service.
        mineru_url: The URL for the MinerU OCR service.
        pipeline_model: The identifier for the LLM to use in the assessment pipeline.
        temperature: The temperature setting for LLM calls.
        force: A boolean to force re-running all steps.
        ocr_workers: The number of workers for parallel OCR processing.

    Returns:
        A list of `Step` objects representing the complete pipeline.
    """

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
    """Executes a list of pipeline steps sequentially.

    This function iterates through the provided steps, checks if a step should
    be skipped, and runs its action if not.

    Args:
        steps: A list of `Step` objects to execute.

    Raises:
        RuntimeError: If any step fails during execution.
    """

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
    """Parses command-line arguments for the script.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
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
    """The main entry point for the script.

    Parses arguments, sets up the environment, builds the pipeline steps,
    and runs the full process.
    """
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
