# End-to-End Workflow (uv + Services)

This guide starts from a clean checkout and walks through installing dependencies with [uv](https://docs.astral.sh/uv/), launching the required services (GROBID and MinerU), and running the novelty assessment pipeline.

---

## 1. Prerequisites

- **Python**: 3.11 or newer.
- **uv**: Fast Python package and environment manager.
  ```bash
  # Install uv for your user if not already available
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # restart your shell so that `uv` is on PATH
  ```
- **Docker**: Required to run the GROBID container. Verify with `docker --version`.
- **GPU (optional)**: MinerU benefits from GPU acceleration but can run on CPU.

## 2. Clone and Inspect the Repository

```bash
git clone https://github.com/<your-org>/llm_paper_review.git
cd llm_paper_review
```

## 3. Create the Python Environment with uv

1. Create the virtual environment and install project dependencies declared in `pyproject.toml` and `requirements.txt`:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv sync --frozen
   uv add -r requirements.txt
   ```
   - `uv sync --frozen` recreates the environment exactly as captured in `uv.lock`.
2. (Optional) Confirm the expected entry points are present:
   ```bash
   uv pip show mineru
   uv pip show grobid-client-python
   ```

## 4. Configure Environment Variables

1. Copy the template if you do not already have a `.env` file:
   ```bash
   cp .env.example .env  # edit manually if the file already exists
   ```
2. Edit `.env` and provide values for:
   - `OPENAI_API_KEY` *(required)*
   - `SEMANTIC_SCHOLAR_API_KEY` *(optional but recommended)*
3. Load the variables in your current shell session when running commands:
   ```bash
   source .venv/bin/activate
   set -a
   source .env
   set +a
   ```

## 5. Prepare External Services

### 5.1 Download the GROBID Docker Image

```bash
docker pull lfoppiano/grobid:0.8.2
```

### 5.2 Launch GROBID and MinerU Together

Use the helper script created at `start_services_and_run.sh`:

```bash
./start_services_and_run.sh
```

What the script does:
- Starts (or reuses) the `lfoppiano/grobid:0.8.2` container on port `8070`.
- Launches `mineru-api --host 0.0.0.0 --port 8000` in the background.
- Waits until both services answer health checks (`http://localhost:8070/api/isalive` and `http://localhost:8000/docs`).
- Writes MinerU logs to `logs/mineru_api.log`.
- Keeps both services running until you press `Ctrl+C`. Use a second terminal for the pipeline.

> Tip: Run `tail -f logs/mineru_api.log` in another terminal if you need to monitor OCR traffic.

## 6. Run the Novelty Assessment Workflow

With the services up:

1. Decide which entry point to use:
   - Shell wrapper (`run_pipeline.sh`) – quick start with defaults.
   - Python CLI (`python src/full_pipeline_runner.py`) – exposes full set of arguments.

### 6.1 Using the Shell Wrapper

```bash
./run_pipeline.sh
```

The script checks that GROBID is live, verifies the target PDF (`seq2seq.pdf` by default), and calls the Python runner with settings defined inside the script.

### 6.2 Using the Python Runner Directly

```bash
python src/full_pipeline_runner.py \
  --submission-id SUB123 \
  --pdf /absolute/path/to/SUB123.pdf \
  --data-dir data \
  --grobid-url http://localhost:8070 \
  --mineru-url http://localhost:8000 \
  --model gpt-4o \
  --temperature 0.0 \
  --ocr-workers 4 \
  --verbose
```

Key artifacts are written under `data/SUB123/` (structured metadata, OCR output, novelty assessment summaries).

## 7. Shutting Down

- Return to the terminal running `start_services_and_run.sh` and press `Ctrl+C` to stop MinerU and the GROBID container cleanly.
- Deactivate the Python environment when finished:
  ```bash
  deactivate
  ```

## 8. Troubleshooting Checklist

- **uv errors**: Ensure you restarted the shell after installing uv and that you are on Python ≥3.11.
- **Docker permission issues**: Add your user to the `docker` group or run the script with `sudo` (not recommended for long-term use).
- **Missing `mineru-api` command**: Confirm `uv pip show mineru` lists the package; reinstall with `uv sync --reinstall` if needed.
- **Pipeline failures**: Use `--verbose` and inspect `data/<submission>/` for partial outputs. Logs from MinerU are in `logs/mineru_api.log`.
- **Services already running**: The helper script detects and reuses an active GROBID container; manually stop conflicting services before rerunning if ports are busy.

You now have an end-to-end workflow using uv for dependency management, Docker for GROBID, MinerU for OCR, and the repository’s pipeline scripts for novelty assessment.
