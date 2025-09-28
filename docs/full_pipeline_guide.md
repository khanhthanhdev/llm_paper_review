# End-to-End Novelty Assessment Pipeline Guide

This guide walks you from a raw PDF submission to the final reviewer-facing summary produced by the novelty assessment system. Follow the steps in order the first time; afterwards you can rerun individual stages as needed.

## 1. Prerequisites

- **Python**: 3.11 or newer.
- **Virtual environment** (recommended): `python -m venv .venv && source .venv/bin/activate`.
- **Dependencies**: `pip install . && pip install -r requirements.txt`.
- **Environment variables** (`.env` in the repo root):
  - `OPENAI_API_KEY` *(required)* – used by LangChain/OpenAI calls throughout the pipeline.
  - `SEMANTIC_SCHOLAR_API_KEY` *(optional but recommended)* – improves Semantic Scholar rate limits.
- **External services/tools**:
  - [GROBID](https://github.com/kermitt2/grobid) running locally to convert PDFs to TEI XML.
  - An OCR solution for PDFs (MinerU or Nougat). A helper script for MinerU is provided at `src/preprocess/run_ocr.py`.
  - Optional: `RankGPT` utilities available at `/home/afzal/novelty_assessment/RankGPT` (required by `src/retrieval/retrieval.py`). Update `sys.path.append` inside that script if your checkout lives elsewhere.

## 2. One-Command Runner (Recommended)

Once prerequisites are in place you can execute the entire workflow with a single command. The runner will reuse existing artifacts unless you pass `--force`.

```bash
python src/full_pipeline_runner.py \
  --submission-id SUB123 \
  --pdf /path/to/SUB123.pdf \
  --data-dir data \
  --grobid-url http://localhost:8070 \
  --mineru-url http://localhost:8000
```

- `--force`: reruns every stage even if outputs already exist (helpful when regenerating results).
- `--model`/`--temperature`: override the OpenAI model configuration for the LLM-heavy stages.
- `--ocr-workers`: control parallel OCR requests (default 3).

The script saves all intermediate files under `data/SUB123/` and prints the final artifact locations when the pipeline completes.

## 3. Prepare the Submission Workspace

Pick a submission identifier (e.g. `SUB123`). Create the directory layout under `data/` and collect the raw inputs:

```
data/
└── SUB123/
    ├── SUB123.pdf                        # main submission PDF (used for OCR)
    ├── SUB123_fulltext.tei.xml           # GROBID TEI output
    └── related_work_data/
        └── citations.json (optional seed material)
```

> 💡 Tip: Any supplementary metadata (e.g. author-provided citation lists) can live inside `data/SUB123/related_work_data/`. The retrieval step will populate this folder automatically when it runs.

## 4. Stage A – Metadata Extraction

1. **Run GROBID externally** to generate the TEI file saved as `data/SUB123/SUB123_fulltext.tei.xml`.
2. **Parse TEI into JSON**:

```bash
python src/preprocess/extract_metadata.py --data-dir data --submission-id SUB123
```

This creates `data/SUB123/SUB123.json` containing the structured metadata (title, abstract, references, citation contexts).

## 5. Stage B – Citation Enrichment & Retrieval

1. **Semantic Scholar enrichment**:

```bash
python src/retrieval/match_papers_to_s2.py --data-dir data --submission-id SUB123
```

2. **Query & rank related papers** (generates embeddings, runs RankGPT ranking, writes results under `related_work_data/`):

```bash
python src/retrieval/retrieval.py --data-dir data --submission-id SUB123
```

3. **Fetch related paper PDFs** (fills `data/SUB123/related_work_data/pdfs/`):

```bash
python src/retrieval/get_cited_pdfs.py --data-dir data --submission-id SUB123
```

## 6. Stage C – OCR & Introduction Extraction

1. **Run OCR** on the main paper and retrieved PDFs. A MinerU helper is provided:

```bash
python src/preprocess/run_ocr.py --data-dir data --submission-id SUB123 --server-url http://localhost:8000
```

This writes markdown files into `data/SUB123/ocr/`.

2. **Extract introductions** for both the submission and related work:

```bash
python src/retrieval/extract_introductions.py --data-dir data --submission-id SUB123
```

Outputs land in `data/SUB123/introductions/`, providing the text needed for later LLM stages.

## 7. Stage D – LLM-Driven Analysis Pipeline

With metadata, enriched citations, introductions, and PDFs in place, run the orchestrated pipeline to perform structured extraction, research landscape analysis, novelty assessment, and final guidance generation:

```bash
python src/novelty_assessment/pipeline.py --data-dir data --submission-id SUB123
```

The pipeline prints progress for each of the four steps and saves artifacts under `data/SUB123/`, including:

- `structured_representation.json`
- `research_landscape.txt`
- `novelty_delta_analysis.txt`
- `summary.txt`
- JSON exports inside `ours/` if configured by downstream modules

## 8. Reviewing Outputs

Key deliverables appear in `data/SUB123/`:

- `summary.txt` and (if generated) `ours/summary_SUB123.json`: reviewer-facing executive summary, guidance, and supporting evidence.
- `novelty_delta_analysis.txt`: detailed comparison between the submission and retrieved literature.
- `structured_representation.json`: machine-readable extraction of methods, problems, datasets, metrics, and novelty claims.

Refer to these files to inspect the final assessment or to feed them into downstream tools.

## 9. Resetting or Rerunning Stages

- You can rerun any stage independently; outputs are overwritten in place.
- If you modify earlier artifacts (e.g. update the TEI file), rerun downstream stages to propagate changes.
- To resume a partially completed LLM pipeline, use the `--start-step` flag (values 1–4) on `pipeline.py` once you confirm which artifacts already exist via `--check-status`.

```bash
python src/novelty_assessment/pipeline.py --data-dir data --submission-id SUB123 --check-status
```

## 10. Troubleshooting Checklist

- **Missing API keys**: ensure `.env` is loaded or export variables before running scripts.
- **RankGPT import errors**: adjust the `sys.path.append(...)` line near the top of `src/retrieval/retrieval.py` to match your local RankGPT checkout.
- **OCR gaps**: confirm PDFs were downloaded, MinerU/Nougat produced `.md`/`.mmd` files in `ocr/`, and rerun `extract_introductions.py`.
- **Rate limits**: set `SEMANTIC_SCHOLAR_API_KEY` or increase `--rate-limit` delays in `match_papers_to_s2.py`.

Once every stage completes successfully you will have a reproducible pipeline that ingests a research paper and returns a structured novelty assessment with supporting evidence.
