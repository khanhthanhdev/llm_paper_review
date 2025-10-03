#!/usr/bin/env python3
"""
OCR Processing Script
Processes PDFs through MinerU FastAPI server for both main paper and related papers.
"""

import argparse
import json
import logging
import requests
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


PRIMARY_PDF_MARKER = ".primary_pdf"


class MinerUOCRClient:
    """A client for processing PDFs through a MinerU FastAPI server.

    This client handles the communication with a running MinerU instance,
    including connecting, sending PDFs for processing, and handling retries.

    Attributes:
        base_url: The base URL of the MinerU server.
        timeout: The request timeout in seconds.
        session: A `requests.Session` object for making HTTP requests.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 3600):
        """Initializes the MinerUOCRClient.

        Args:
            base_url: The base URL of the MinerU server.
            timeout: The default timeout for requests in seconds.

        Raises:
            requests.exceptions.RequestException: If the client cannot connect to the server.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        
        # Test connection
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Connected to MinerU server at {self.base_url}")
            else:
                logger.warning(f"⚠️ MinerU server responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MinerU server: {e}")
            raise
    
    def process_pdf(self, pdf_path: Path, page_range: str = "full", max_retries: int = 2) -> Optional[str]:
        """Sends a PDF to the MinerU server for OCR processing.

        This method handles file upload, request sending, and response parsing.
        It includes a retry mechanism to handle transient network or server errors.

        Args:
            pdf_path: The path to the PDF file to be processed.
            page_range: The range of pages to process. Note: MinerU typically
                        processes the full document regardless of this parameter.
            max_retries: The maximum number of times to retry the request upon failure.

        Returns:
            The extracted text content as a string if successful, otherwise None.
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return None
            
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        for attempt in range(max_retries + 1):
            try:
                with open(pdf_path, "rb") as f:
                    files = {"files": (pdf_path.name, f, "application/pdf")}
                    
                    # Make request to MinerU endpoint
                    response = self.session.post(
                        f"{self.base_url}/file_parse",
                        files=files,
                        timeout=3600 # 60 minute timeout for OCR processing
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if "results" in result:
                        # Extract text content from MinerU results
                        text_content = self._extract_text_from_results(result["results"])
                        if text_content:
                            logger.info(f"✅ Successfully processed {pdf_path.name}")
                            return text_content
                        else:
                            logger.error(f"❌ No text content extracted from {pdf_path.name}")
                            return None
                    else:
                        logger.error(f"❌ OCR processing failed for {pdf_path.name}: No results in response")
                        return None
                else:
                    logger.error(f"❌ OCR failed for {pdf_path.name}: {response.status_code} - {response.text}")
                    if attempt < max_retries:
                        logger.info(f"Retrying {pdf_path.name} (attempt {attempt + 2}/{max_retries + 1})")
                        time.sleep(2)  # Wait before retry
                        continue
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_path.name}: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying {pdf_path.name} after error (attempt {attempt + 2}/{max_retries + 1})")
                    time.sleep(2)
                    continue
                return None
        
        return None  # All retries exhausted
    
    def _extract_text_from_results(self, results: dict) -> Optional[str]:
        """Extracts text content from the structured MinerU JSON response.

        MinerU's response format can vary, so this method attempts to find the
        markdown content in common locations within the JSON structure.

        Args:
            results: The dictionary parsed from the MinerU server's JSON response.

        Returns:
            The extracted markdown content as a string, or None if not found.
        """
        try:
            # MinerU returns results as a dictionary with paper filenames as keys
            # Each paper has an "md_content" field with the extracted markdown
            for paper_id, paper_data in results.items():
                if isinstance(paper_data, dict) and "md_content" in paper_data:
                    content = paper_data["md_content"]
                    if content and len(content.strip()) > 50:
                        return content
            
            # Fallback: look for any large string content
            for key, value in results.items():
                if isinstance(value, str) and len(value.strip()) > 50:
                    return value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str) and len(sub_value.strip()) > 50:
                            return sub_value
            
            return None
                
        except Exception as e:
            logger.error(f"Error extracting text from results: {e}")
            return None


def find_main_paper_pdf(submission_dir: Path, submission_id: str) -> Optional[Path]:
    """Finds the main submission PDF within a directory.

    It uses a series of heuristics to locate the primary PDF:
    1. Checks for a `.primary_pdf` marker file containing the filename.
    2. Looks for common filenames like `{submission_id}.pdf` or `paper.pdf`.
    3. If only one PDF exists in the directory, assumes it is the main paper.

    Args:
        submission_dir: The directory where the submission files are located.
        submission_id: The unique identifier for the submission.

    Returns:
        A `Path` object pointing to the main PDF, or None if it cannot be found.
    """
    marker = submission_dir / PRIMARY_PDF_MARKER
    if marker.exists():
        try:
            recorded_name = marker.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning(f"Unable to read primary PDF marker {marker}: {exc}")
        else:
            if recorded_name:
                recorded_path = submission_dir / recorded_name
                if recorded_path.exists():
                    return recorded_path

    # Try common locations/names for main paper
    candidates = [
        submission_dir / f"{submission_id}.pdf",
        submission_dir / "main.pdf",
        submission_dir / "paper.pdf",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Look for any PDF in the submission directory
    pdf_files = list(submission_dir.glob("*.pdf"))
    if len(pdf_files) == 1:
        return pdf_files[0]
    elif len(pdf_files) > 1:
        logger.warning(f"Multiple PDFs found in {submission_dir}, using first one: {pdf_files[0]}")
        return pdf_files[0]
    
    return None


def process_single_pdf(ocr_client: MinerUOCRClient, pdf_path: Path, output_dir: Path, paper_id: str) -> Tuple[str, bool]:
    """Processes a single PDF file using the OCR client and saves the result.

    This function coordinates the OCR processing for one PDF, including checking
    for existing output and saving the new result to a file.

    Args:
        ocr_client: An initialized `MinerUOCRClient` instance.
        pdf_path: The path to the PDF file to process.
        output_dir: The directory where the output markdown file should be saved.
        paper_id: The unique identifier for the paper, used for naming the output file.

    Returns:
        A tuple containing the paper ID and a boolean indicating success.
    """
    ocr_file = output_dir / f"{paper_id}.md"
    
    if ocr_file.exists():
        logger.info(f"OCR already exists for {paper_id}: {ocr_file}")
        return (paper_id, True)
    
    # Process through OCR (full document)
    ocr_text = ocr_client.process_pdf(pdf_path)
    if ocr_text:
        with open(ocr_file, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        logger.info(f"✅ Saved OCR for {paper_id}: {ocr_file}")
        return (paper_id, True)
    else:
        logger.error(f"Failed to process {paper_id}")
        return (paper_id, False)


def process_for_pipeline(data_dir: str, submission_id: str, server_url: str = "http://localhost:8000", max_workers: int = 3) -> bool:
    """Orchestrates OCR processing for a full submission package.

    This function finds the main paper and all related papers, then processes
    them concurrently using a thread pool. It manages the entire OCR workflow
    for a single submission as part of the wider pipeline.

    Args:
        data_dir: The base directory for all submission data.
        submission_id: The unique identifier for the submission to process.
        server_url: The URL of the MinerU OCR server.
        max_workers: The number of parallel threads to use for processing PDFs.

    Returns:
        `True` if at least one PDF was processed successfully, otherwise `False`.
    """
    submission_dir = Path(data_dir) / submission_id
    ocr_dir = submission_dir / "ocr"
    ocr_dir.mkdir(exist_ok=True)
    
    logger.info(f"Processing OCR for submission {submission_id}")
    
    # Initialize OCR client
    try:
        ocr_client = MinerUOCRClient(server_url)
    except Exception as e:
        logger.error(f"Failed to initialize OCR client: {e}")
        return False
    
    stats = {
        "main_paper": {"found": False, "processed": False},
        "related_papers": {"total": 0, "found": 0, "processed": 0}
    }
    
    # Collect all PDFs to process
    pdfs_to_process = []
    
    # 1. Add main paper if exists
    main_pdf = find_main_paper_pdf(submission_dir, submission_id)
    if main_pdf:
        stats["main_paper"]["found"] = True
        pdfs_to_process.append((main_pdf, submission_id, "main"))
        logger.info(f"Found main paper PDF: {main_pdf}")
    else:
        logger.warning(f"No main paper PDF found for {submission_id}")
    
    # 2. Add related papers if exist
    related_work_dir = submission_dir / "related_work_data"
    pdfs_dir = related_work_dir / "pdfs"
    
    if pdfs_dir.exists():
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        stats["related_papers"]["total"] = len(pdf_files)
        stats["related_papers"]["found"] = len(pdf_files)
        
        for pdf_file in pdf_files:
            paper_id = pdf_file.stem
            pdfs_to_process.append((pdf_file, paper_id, "related"))
        
        logger.info(f"Found {len(pdf_files)} related paper PDFs")
    else:
        logger.warning(f"No related papers directory: {pdfs_dir}")
    
    # Process all PDFs concurrently
    logger.info(f"Processing {len(pdfs_to_process)} PDFs with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(process_single_pdf, ocr_client, pdf_path, ocr_dir, paper_id): (pdf_path, paper_id, pdf_type)
            for pdf_path, paper_id, pdf_type in pdfs_to_process
        }
        
        # Process completed tasks
        for future in as_completed(future_to_pdf):
            pdf_path, paper_id, pdf_type = future_to_pdf[future]
            try:
                result_id, success = future.result()
                if success:
                    if pdf_type == "main":
                        stats["main_paper"]["processed"] = True
                    else:
                        stats["related_papers"]["processed"] += 1
                    logger.info(f"✅ Completed: {result_id}")
                else:
                    logger.error(f"❌ Failed: {result_id}")
            except Exception as e:
                logger.error(f"❌ Exception processing {paper_id}: {e}")
    
    # Log final stats
    logger.info(f"OCR processing completed for {submission_id}:")
    logger.info(f"  Main paper: {'✅' if stats['main_paper']['processed'] else '❌'}")
    logger.info(f"  Related papers: {stats['related_papers']['processed']}/{stats['related_papers']['total']} processed")
    
    return stats["main_paper"]["processed"] or stats["related_papers"]["processed"] > 0


if __name__ == "__main__":
    """The main entry point for the script.

    Parses command-line arguments and initiates the OCR processing for a specified
    submission. This allows the script to be run directly from the command line.
    """
    parser = argparse.ArgumentParser(description="Process PDFs through MinerU FastAPI server - single submission mode only")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Base data directory"
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        required=True,
        help="Submission ID to process"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="MinerU server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of concurrent OCR requests (default: 3)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Processing submission {args.submission_id} with {args.max_workers} workers")
    success = process_for_pipeline(args.data_dir, args.submission_id, args.server_url, args.max_workers)
    
    if success:
        logger.info(f"✅ Successfully processed OCR for {args.submission_id}")
        print(f"✅ Successfully processed submission {args.submission_id}")
    else:
        logger.error(f"❌ Failed to process OCR for {args.submission_id}")
        print(f"❌ Failed to process submission {args.submission_id}")
