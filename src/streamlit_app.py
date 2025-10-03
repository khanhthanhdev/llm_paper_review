"""Streamlit dashboard for the LLM paper review pipeline."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv
from streamlit.components.v1 import html

from full_pipeline_runner import build_steps, ensure_submission_workspace, prepare_pdf


LOGGER = logging.getLogger("streamlit_app")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
load_dotenv()

DATA_ROOT = Path(os.getenv("PIPELINE_DATA_DIR", "data")).resolve()
DEFAULT_SUBMISSION_ID = os.getenv("DEFAULT_SUBMISSION_ID", "paper")
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")
MINERU_URL = os.getenv("MINERU_URL", "http://localhost:8000")
PIPELINE_MODEL = os.getenv("PIPELINE_MODEL", "gpt-4o")


def _safe_int(value: Optional[str], default: int) -> int:
    """Safely convert a string to an integer, falling back to a default."""
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(value: Optional[str], default: float) -> float:
    """Safely convert a string to a float, falling back to a default."""
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


PIPELINE_TEMPERATURE = _safe_float(os.getenv("PIPELINE_TEMPERATURE"), 0.0)
OCR_WORKERS = _safe_int(os.getenv("OCR_WORKERS"), 3)

DATA_ROOT.mkdir(parents=True, exist_ok=True)


def discover_runs(data_root: Path) -> list[dict[str, object]]:
    """Scans the data directory to find previous pipeline runs.

    Args:
        data_root: The root directory where submission data is stored.

    Returns:
        A list of dictionaries, each representing a run, sorted by modification
        time in descending order.
    """

    if not data_root.exists():
        return []

    runs: list[dict[str, object]] = []
    for candidate in data_root.iterdir():
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue

        summary_file = candidate / "summary.txt"
        timestamp_source = summary_file if summary_file.exists() else candidate
        timestamp = datetime.fromtimestamp(timestamp_source.stat().st_mtime)

        runs.append({
            "submission_id": candidate.name,
            "path": candidate,
            "timestamp": timestamp,
        })

    runs.sort(key=lambda item: item["timestamp"], reverse=True)
    return runs


def read_text_file(path: Path) -> Optional[str]:
    """Reads a file and returns its content as a string.

    Returns None if the file does not exist. Handles potential decoding errors
    by replacing problematic characters.

    Args:
        path: The `Path` object of the file to read.

    Returns:
        The file's content as a string, or None if the file doesn't exist.
    """

    if not path.exists():
        return None

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def render_copy_button(text: str, *, key: str, label: str = "Copy to clipboard") -> None:
    """Renders a button that copies the given text to the clipboard using JavaScript.

    Args:
        text: The text to be copied when the button is clicked.
        key: A unique key for the component (required by Streamlit's rendering logic,
             though not directly used by `st.html` before version 1.38).
        label: The text label to display on the button.
    """

    _ = key  # Preserve parameter for call sites; Streamlit html() lacks a key arg pre-1.38.
    escaped = json.dumps(text)
    html(
        f"""
        <div style=\"display:flex; justify-content:flex-start; margin:0.25rem 0;\">
            <button style=\"
                padding:0.3rem 0.85rem;
                border-radius:0.5rem;
                border:1px solid rgba(49,51,63,0.2);
                background-color:var(--secondary-background-color,#f0f2f6);
                color:inherit;
                cursor:pointer;
            \"
            onclick=\"navigator.clipboard.writeText({escaped});\">{label}</button>
        </div>
        """,
        height=52,
    )


# ``UploadedFile`` is only needed for type checkers. The runtime import lives in a
# private Streamlit module, so we guard it to avoid import errors during app
# execution if the internal path changes.
if TYPE_CHECKING:  # pragma: no cover - import only when type checking
    from streamlit.runtime.uploaded_file_manager import UploadedFile


class StreamlitLogHandler(logging.Handler):
    """A custom logging handler that writes log records to a Streamlit container.

    This allows the application to display a running log of the pipeline's
    progress directly in the user interface.

    Attributes:
        placeholder: The Streamlit container (e.g., `st.empty()`) to write logs into.
        max_lines: The maximum number of log lines to retain and display.
    """

    def __init__(self, placeholder: "st.delta_generator.DeltaGenerator", *, max_lines: int = 200) -> None:
        """Initializes the log handler.

        Args:
            placeholder: The Streamlit container to which log messages will be written.
            max_lines: The maximum number of lines to keep in the log display.
        """
        super().__init__(level=logging.INFO)
        self.placeholder = placeholder
        self.max_lines = max_lines
        self._lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Formats and displays a log record in the Streamlit UI."""
        message = self.format(record)
        self._lines.append(message)
        if len(self._lines) > self.max_lines:
            self._lines = self._lines[-self.max_lines :]
        payload = "\n".join(self._lines)
        self.placeholder.markdown(f"```text\n{payload}\n```")


def _create_run_log_path(submission_dir: Path, submission_id: str) -> Path:
    """Creates a timestamped log file path for a pipeline run."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = submission_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"streamlit_{submission_id}_{timestamp}.log"


@contextmanager
def _attach_run_loggers(log_path: Path, ui_placeholder: "st.delta_generator.DeltaGenerator") -> Iterator[None]:
    """A context manager to temporarily attach log handlers for a pipeline run.

    This sets up logging to both a file and the Streamlit UI and ensures the
    handlers are removed and cleaned up afterward.

    Args:
        log_path: The path to the file where logs should be saved.
        ui_placeholder: The Streamlit container for displaying logs in the UI.

    Yields:
        None.
    """

    root_logger = logging.getLogger()
    original_level = root_logger.level
    if original_level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.setLevel(logging.INFO)

    ui_handler = StreamlitLogHandler(ui_placeholder)
    ui_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(ui_handler)

    try:
        yield
    finally:
        root_logger.removeHandler(file_handler)
        root_logger.removeHandler(ui_handler)
        file_handler.close()
        ui_handler.close()
        root_logger.setLevel(original_level)


def save_uploaded_pdf(uploaded_file: "UploadedFile", submission_dir: Path, submission_id: str) -> Path:
    """Saves a PDF uploaded via Streamlit to the submission's workspace directory.

    Args:
        uploaded_file: The file object from `st.file_uploader`.
        submission_dir: The directory for the submission.
        submission_id: The unique identifier for the submission.

    Returns:
        The `Path` where the PDF was saved.
    """

    suffix = Path(uploaded_file.name or "").suffix.lower()
    if suffix != ".pdf":
        suffix = ".pdf"
    destination = submission_dir / f"{submission_id}{suffix}"
    destination.write_bytes(uploaded_file.getvalue())
    LOGGER.info("Uploaded PDF persisted to %s", destination)
    return destination


def run_pipeline_with_progress(
    *,
    submission_id: str,
    submission_dir: Path,
    pdf_path: Path,
    reuse_cached: bool,
) -> Iterator[tuple[str, str]]:
    """Runs the full pipeline and yields progress updates for the UI.

    This generator function executes each step of the pipeline and yields its
    status ('skipped', 'running', 'completed') and name, allowing the Streamlit
    UI to update in real time.

    Args:
        submission_id: The unique identifier for the submission.
        submission_dir: The path to the submission's workspace directory.
        pdf_path: The path to the main submission PDF.
        reuse_cached: If `True`, cached results will be used, and completed
                      steps will be skipped.

    Yields:
        A tuple containing the status and name of each pipeline step as it
        is processed.
    """

    force = not reuse_cached
    LOGGER.info("Starting pipeline build for submission '%s'", submission_id)
    steps = build_steps(
        submission_id=submission_id,
        submission_dir=submission_dir,
        pdf_path=pdf_path,
        data_dir=DATA_ROOT,
        grobid_url=GROBID_URL,
        mineru_url=MINERU_URL,
        pipeline_model=PIPELINE_MODEL,
        temperature=PIPELINE_TEMPERATURE,
        force=force,
        ocr_workers=OCR_WORKERS,
    )

    for step in steps:
        if reuse_cached and step.should_skip():
            LOGGER.info("Skipping step (cached): %s", step.name)
            yield "skipped", step.name
            continue

        LOGGER.info("Starting step: %s", step.name)
        yield "running", step.name
        step.action()
        LOGGER.info("Completed step: %s", step.name)
        yield "completed", step.name

    LOGGER.info("Pipeline finished for submission '%s'", submission_id)


def render_text_output(*, title: str, file_path: Path, submission_id: str) -> None:
    """Renders a text artifact in the Streamlit UI.

    This function displays the content of a text file and provides buttons for
    downloading the file and copying its content to the clipboard.

    Args:
        title: The title to display for this output section.
        file_path: The path to the text file to be rendered.
        submission_id: The ID of the current submission, used for unique keys
                       and filenames.
    """

    content = read_text_file(file_path)
    if content is None:
        st.info(f"{title} not available for submission '{submission_id}'.")
        return

    download_name = f"{submission_id}_{file_path.name}"
    st.download_button(
        label=f"Download {file_path.name}",
        data=content.encode("utf-8"),
        file_name=download_name,
        mime="text/plain",
        use_container_width=False,
    )
    render_copy_button(content, key=f"copy-{submission_id}-{file_path.name}")
    st.caption("Rendered view")
    st.markdown(content)

    with st.expander(f"Raw text · {file_path.name}"):
        st.text_area(
            label=f"Raw {title}",
            value=content,
            height=320,
            label_visibility="collapsed",
            disabled=True,
        )


def render_rankings_panel(submission_dir: Path, submission_id: str) -> None:
    """Renders the panel displaying the ranked list of related papers.

    This function reads the `ranked_papers.json` file, displays the results in
    a Streamlit dataframe, and provides buttons to download the data as JSON or CSV.

    Args:
        submission_dir: The directory for the submission being displayed.
        submission_id: The unique identifier for the submission.
    """

    ranked_path = submission_dir / "related_work_data" / "ranked_papers.json"
    if not ranked_path.exists():
        st.info(f"No ranked papers found for submission '{submission_id}'.")
        return

    try:
        payload = json.loads(ranked_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        st.error(f"Unable to parse ranked_papers.json: {exc}")
        return

    if isinstance(payload, dict) and "ranked_papers" in payload:
        papers = payload["ranked_papers"]
    else:
        papers = payload

    if not isinstance(papers, list):
        st.error("Unexpected data format in ranked_papers.json.")
        return

    table_rows: list[dict[str, object]] = []
    for idx, item in enumerate(papers, start=1):
        if not isinstance(item, dict):
            continue

        used_flag = item.get("novel")
        if isinstance(used_flag, bool):
            used = "Yes" if used_flag else "No"
        else:
            used = "Yes" if item.get("cited_paper") else "No"

        table_rows.append(
            {
                "Rank": idx,
                "Title": item.get("title", "—"),
                "Source": item.get("venue") or item.get("source") or "",
                "Used?": used,
            }
        )

    if not table_rows:
        st.info("Ranked paper list is empty.")
        return

    st.dataframe(table_rows, width='stretch')

    json_blob = json.dumps(papers, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download ranked_papers.json",
        data=json_blob.encode("utf-8"),
        file_name=f"{submission_id}_ranked_papers.json",
        mime="application/json",
        width='content',
    )
    render_copy_button(json_blob, key=f"copy-{submission_id}-ranked", label="Copy JSON to clipboard")

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["Rank", "Title", "Source", "Used?"])
    writer.writeheader()
    writer.writerows(table_rows)
    csv_payload = csv_buffer.getvalue()

    st.download_button(
        label="Download rankings CSV",
        data=csv_payload.encode("utf-8"),
        file_name=f"{submission_id}_ranked_papers.csv",
        mime="text/csv",
        width='content',
    )
    render_copy_button(csv_payload, key=f"copy-{submission_id}-ranked-csv", label="Copy CSV to clipboard")


def main() -> None:
    """The main function that defines and runs the Streamlit application.

    This function sets up the page configuration, manages session state, lays out
    all the UI components (sidebar, tabs, buttons), and handles the logic for
    running the pipeline and displaying results.
    """
    st.set_page_config(page_title="LLM Paper Review", layout="wide")

    if "submission_id_input" not in st.session_state:
        st.session_state["submission_id_input"] = DEFAULT_SUBMISSION_ID
    if "current_submission_id" not in st.session_state:
        st.session_state["current_submission_id"] = DEFAULT_SUBMISSION_ID
    if "reuse_cached" not in st.session_state:
        st.session_state["reuse_cached"] = True

    header_col, help_col = st.columns([6, 1])
    with header_col:
        st.title("LLM Paper Review")
        st.caption("Upload a submission PDF and review generated analyses in one place.")
    with help_col:
        st.markdown("[❔ Usage guide](docs/full_pipeline_guide.md)")

    with st.sidebar:
        st.header("Input Panel")
        st.text_input("Submission ID", key="submission_id_input")
        uploaded_pdf = st.file_uploader("Submission PDF", type=["pdf"])
        st.checkbox(
            "Reuse previous results when available",
            key="reuse_cached",
            help="When checked, cached artifacts are kept and only missing steps run.",
        )
        run_clicked = st.button("Run Analysis", type="primary", width='stretch')

        st.divider()

        st.subheader("Run History")
        runs = discover_runs(DATA_ROOT)
        if runs:
            options = [(run["submission_id"], f"{run['submission_id']} • {run['timestamp'].strftime('%Y-%m-%d %H:%M')}") for run in runs]
            current_id = st.session_state.get("current_submission_id")
            default_index = 0
            for idx, option in enumerate(options):
                if option[0] == current_id:
                    default_index = idx
                    break

            selection = st.selectbox(
                "Review existing run",
                options,
                index=default_index,
                format_func=lambda option: option[1],
                key="run_history_select",
            )
            st.session_state["current_submission_id"] = selection[0]
        else:
            st.caption("No completed runs discovered yet.")

    if run_clicked:
        submission_id = st.session_state["submission_id_input"].strip()
        if not submission_id:
            st.error("Please provide a submission ID before running the pipeline.")
        else:
            reuse_cached = st.session_state.get("reuse_cached", True)
            submission_dir = ensure_submission_workspace(DATA_ROOT, submission_id)
            pdf_input_path: Optional[Path] = None
            if uploaded_pdf is not None:
                pdf_input_path = save_uploaded_pdf(uploaded_file=uploaded_pdf, submission_dir=submission_dir, submission_id=submission_id)

            log_path = _create_run_log_path(submission_dir, submission_id)
            log_expander = st.expander("Run log", expanded=True)
            log_placeholder = log_expander.empty()

            run_succeeded = False
            with _attach_run_loggers(log_path, log_placeholder):
                LOGGER.info("Run log for submission '%s' stored at %s", submission_id, log_path)
                with st.status("Running analysis…", expanded=True) as status:
                    status.write(f"Workspace: {submission_dir}")
                    status.write(f"Log file: {log_path}")

                    try:
                        pdf_path = prepare_pdf(submission_dir, submission_id, pdf_input_path)
                    except FileNotFoundError as exc:
                        status.update(label="Pipeline failed", state="error")
                        st.error(str(exc))
                    else:
                        try:
                            for state, name in run_pipeline_with_progress(
                                submission_id=submission_id,
                                submission_dir=submission_dir,
                                pdf_path=pdf_path,
                                reuse_cached=reuse_cached,
                            ):
                                if state == "skipped":
                                    status.write(f"Skipping {name} (cached)")
                                elif state == "running":
                                    status.write(f"Running {name}…")
                                elif state == "completed":
                                    status.write(f"Completed {name}")
                        except Exception as exc:  # pragma: no cover - surfaced to UI
                            LOGGER.exception("Pipeline failed")
                            status.update(label="Pipeline failed", state="error")
                            st.error(f"Pipeline failed: {exc}")
                        else:
                            status.update(label="Analysis complete", state="complete", expanded=False)
                            st.success("Pipeline completed successfully.")
                            st.session_state["current_submission_id"] = submission_id
                            run_succeeded = True

            if run_succeeded:
                st.caption(f"Run log saved to `{log_path}`")
            else:
                st.caption(f"Run log saved to `{log_path}` for troubleshooting")

    selected_submission_id = st.session_state.get("current_submission_id")
    selected_submission_dir = DATA_ROOT / selected_submission_id if selected_submission_id else None

    if not selected_submission_dir or not selected_submission_dir.exists():
        st.info("No outputs available yet. Run the analysis to populate the dashboard.")
        return

    summary_tab, novelty_tab, landscape_tab, rankings_tab = st.tabs(
        ["Summary", "Novelty Delta", "Research Landscape", "Paper Rankings"]
    )

    with summary_tab:
        st.subheader("Summary")
        render_text_output(
            title="Summary",
            file_path=selected_submission_dir / "summary.txt",
            submission_id=selected_submission_id,
        )

    with novelty_tab:
        st.subheader("Novelty Delta")
        render_text_output(
            title="Novelty Delta",
            file_path=selected_submission_dir / "novelty_delta_analysis.txt",
            submission_id=selected_submission_id,
        )

    with landscape_tab:
        st.subheader("Research Landscape")
        render_text_output(
            title="Research Landscape",
            file_path=selected_submission_dir / "research_landscape.txt",
            submission_id=selected_submission_id,
        )

    with rankings_tab:
        st.subheader("Paper Rankings")
        render_rankings_panel(selected_submission_dir, selected_submission_id)


if __name__ == "__main__":
    main()
