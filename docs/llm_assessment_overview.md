# LLM Assessment Stage Architecture

## Purpose
The LLM Assessment stage converts preprocessed submission artifacts into reviewer-ready analyses. It consumes structured metadata, retrieved related work, OCR-derived introductions, and citation contexts, then orchestrates multiple OpenAI-powered subtasks to assess novelty and generate a final summary for reviewers.

## Component Overview

### NoveltyAssessmentPipeline (`src/novelty_assessment/pipeline.py`)
- Entry point that wires together the downstream components.
- Exposes `run_complete_pipeline` and `run_from_step` to execute stages sequentially or resume midstream.
- Persists artifacts inside `data/<submission_id>/` and ensures consistent ordering of steps.

### StructuredRepresentationGenerator (`structured_extraction.py`)
- **Inputs:**
  - Submission JSON (`<submission_id>.json`) containing metadata, abstract, citation contexts.
  - OCR introductions (`introductions/*.txt`) for submission and top-ranked related papers.
  - Ranked related papers (`related_work_data/ranked_papers.json`).
- **Process:**
  - Builds prompts from abstract + introduction content.
  - Calls OpenAI ChatCompletions (LangChain `ChatOpenAI`) with JSON-schema enforced outputs.
  - Batch processes top-N related papers via `with_structured_output(...).batch` for efficiency.
- **Outputs:**
  - `structured_representation.json` storing parsed fields for the submission and selected related papers.

### ResearchLandscapeAnalyzer (`research_landscape.py`)
- **Inputs:**
  - `structured_representation.json` (submission + selected papers).
  - `ranked_papers.json` for title alignment.
- **Process:**
  - Formats submission and related paper representations into a consolidated prompt.
  - Calls OpenAI ChatCompletions to synthesize methodological, problem, evaluation, cluster, and evolution insights.
- **Outputs:**
  - `research_landscape.txt` textual analysis summarizing the competitive landscape.
  - Updates `metadata.json` with token usage and estimated cost.

### NoveltyAssessor (`novelty_assessment.py`)
- **Inputs:**
  - Submission entry from `structured_representation.json`.
  - `research_landscape.txt` analysis.
  - Ranked paper list with citation enrichment (`ranked_papers.json`).
  - Submission citation contexts (`<submission_id>.json`).
- **Process:**
  - Identifies related papers lacking citation matches via fuzzy matching.
  - Aggregates citation sentences for matched papers.
  - Feeds structured submission data, landscape, non-cited titles, and citation contexts into the `novelty_delta_analysis_prompt`.
  - Invokes ChatGPT to produce a sectioned novelty delta report.
- **Outputs:**
  - `novelty_delta_analysis.txt` with detailed comparisons.
  - `metadata.json` update capturing usage metrics.

### ReviewGuidanceGenerator (`generate_summary.py`)
- **Inputs:** `novelty_delta_analysis.txt`.
- **Process:** Summarizes the novelty assessment into five reviewer-oriented sentences via ChatGPT.
- **Outputs:**
  - `summary.txt` concise reviewer guidance.
  - `metadata.json` entry for summary generation cost.

## Data Flow & Dependencies
1. **Structured Representation:** Must run first; depends on successful OCR introduction extraction and ranked paper retrieval.
2. **Research Landscape:** Requires the structured representation outputs to be present.
3. **Novelty Delta Analysis:** Needs both structured data and the landscape report; reuses citation contexts produced in preprocessing.
4. **Reviewer Summary:** Finalizes once novelty analysis exists.

Each step persists artifacts under `data/<submission_id>/`, enabling resumability. `NoveltyAssessmentPipeline` checks for existing files when running from an arbitrary step.

## External Services & Configuration
- Relies on OpenAI ChatCompletions via LangChain (`ChatOpenAI`); `OPENAI_API_KEY` must be set.
- Uses `litellm.cost_calculator` for cost estimates stored in `metadata.json`.
- Fuzzy title matching employs `rapidfuzz` to correlate citation records with ranked papers.

## Output Layout
- `structured_representation.json`: Parsed fields for submission and related papers.
- `research_landscape.txt`: Narrative landscape summary.
- `novelty_delta_analysis.txt`: Detailed novelty assessment with citation evidence.
- `summary.txt`: Five-sentence reviewer briefing.
- `metadata.json`: Aggregated per-stage usage/cost metadata.

## Operational Considerations
- Downstream stages assume consistent file naming; missing introductions or ranked papers will cause upstream assertions to fail.
- Large LLM batches may incur significant token costs; monitor `metadata.json` for budgeting.
- Errors in any substage should trigger retries or manual inspection before resuming with `run_from_step` to avoid redundant API usage.
