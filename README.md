<p align="center">
  <img src="https://raw.githubusercontent.com/UKPLab/arxiv2025-assessing-paper-novelty/main/docs/logo.png" width="200" alt="Project Logo">
</p>

# Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback

[![Arxiv](https://img.shields.io/badge/Arxiv-2508.10795-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2508.10795)
[![License](https://img.shields.io/github/license/UKPLab/arxiv2025-assessing-paper-novelty)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This repository contains the code and data for the paper **Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback**.

Novelty assessment is a central yet understudied aspect of peer review, particularly in high-volume fields like NLP where reviewer capacity is strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence-based assessment. Our method produces detailed, literature-aware analyses that improve consistency and support more rigorous and transparent peer review.

This repository provides all the tools necessary to replicate our experiments and apply the novelty assessment pipeline to new research papers.

**Contact:** [Osama Mohammed Afzal](mailto:osama.afzal@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

## 🚀 Features

- **End-to-End Pipeline**: Automates the entire novelty assessment process, from PDF processing to generating a final summary for reviewers.
- **Modular Architecture**: Each stage of the pipeline can be run independently for greater flexibility and control.
- **Multi-Source Retrieval**: Gathers related work from Semantic Scholar, arXiv, and the ACL Anthology.
- **LLM-Powered Analysis**: Leverages large language models for deep, context-aware analysis of research papers.
- **Interactive Dashboard**: A Streamlit application to run the pipeline and visualize the results.

## ⚙️ How It Works: The Novelty Assessment Pipeline

The pipeline is structured into four main stages, designed to mimic the workflow of an expert reviewer:

1.  **Preprocessing and Retrieval**:
    *   **Metadata Extraction**: Parses a submission's PDF using **GROBID** to extract its title, abstract, bibliography, and citation contexts.
    *   **Citation Enrichment**: Enriches the bibliography by matching each cited paper with its metadata from **Semantic Scholar**.
    *   **Related Work Retrieval**: Generates search queries based on the submission's content and retrieves a broad set of related papers from Semantic Scholar.
    *   **Ranking & Filtering**: Ranks the retrieved papers for relevance using SPECTER2 embeddings and **RankGPT**.

2.  **Content Extraction**:
    *   **PDF Downloading**: Downloads the PDFs of the top-ranked related papers.
    *   **OCR Processing**: Uses an OCR service (like **Nougat** or **MinerU**) to convert the downloaded PDFs into machine-readable text.
    *   **Introduction Extraction**: Identifies and extracts the introduction section from each OCR-processed paper.

3.  **LLM-Powered Analysis**:
    *   **Structured Representation**: Uses an LLM to extract key information (methods, problems, datasets, novelty claims) from the submission and related papers.
    *   **Research Landscape**: Analyzes the extracted information to create a map of the research area, identifying methodological clusters and trends.
    *   **Novelty Assessment**: Performs a detailed, evidence-based comparison of the submission against the research landscape to assess its novelty.

4.  **Summary Generation**:
    *   **Reviewer Guidance**: Synthesizes the detailed novelty assessment into a concise, 5-sentence summary tailored for peer reviewers.

## 🔧 Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/UKPLab/arxiv2025-assessing-paper-novelty.git
cd arxiv2025-assessing-paper-novelty
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the project's Python dependencies using pip.

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

The pipeline requires API keys for OpenAI and Semantic Scholar.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Open the `.env` file and add your keys:
    ```
    OPENAI_API_KEY="sk-..."
    SEMANTIC_SCHOLAR_API_KEY="..."
    ```
    *   `OPENAI_API_KEY` is **required** for all LLM-based analysis steps.
    *   `SEMANTIC_SCHOLAR_API_KEY` is **optional but recommended** to get higher rate limits when fetching paper data.

### 5. Set Up External Services

This pipeline relies on external services for PDF processing. You must have them running locally.

-   **GROBID**: For parsing scholarly documents. Follow the instructions at the [GROBID repository](https://github.com/kermitt2/grobid) to run it as a service (typically at `http://localhost:8070`).
-   **OCR Service (MinerU)**: For extracting text from PDFs. We use **MinerU**, which you can set up by following the instructions at the [MinerU repository](https://github.com/opendatalab/MinerU). The service should be running at `http://localhost:8000`.

### 6. Set Up RankGPT

The retrieval stage uses **RankGPT** for re-ranking search results. A convenience script is provided to set it up automatically.

```bash
python setup_rankgpt.py
```

This will clone the RankGPT repository into a local cache directory.

## 📈 Usage

There are two main ways to use the pipeline: the automated runner for end-to-end execution and the interactive Streamlit dashboard for visualization.

### Option 1: Automated Full Pipeline Runner (Recommended)

The `full_pipeline_runner.py` script is the easiest way to process a single submission from start to finish.

**Command:**
```bash
python src/full_pipeline_runner.py --submission-id <your_submission_id> --pdf /path/to/your/paper.pdf
```

-   `--submission-id`: A unique name for your submission (e.g., `my_paper_2025`). This will be used to create a directory in `data/`.
-   `--pdf`: The path to the PDF file of the paper you want to analyze.

The script will run all pipeline stages sequentially and save all outputs to `data/<your_submission_id>/`.

### Option 2: Interactive Streamlit Dashboard

The Streamlit dashboard provides a user-friendly interface to run the pipeline and explore the results.

**To launch the dashboard:**
```bash
streamlit run src/streamlit_app.py
```

From the dashboard, you can:
-   Upload a new PDF and assign it a submission ID.
-   Run the full analysis with a single click.
-   View and download the generated outputs (Research Landscape, Novelty Delta, Summary).
-   Browse the list of ranked related papers.

### Option 3: Manual Execution (Advanced)

For more granular control, you can run each stage of the pipeline manually using the individual scripts located in `src/`. This is useful for debugging or custom workflows. Refer to the scripts in the following order:

1.  `src/preprocess/extract_metadata.py`
2.  `src/retrieval/match_papers_to_s2.py`
3.  `src/retrieval/retrieval.py`
4.  `src/retrieval/get_cited_pdfs.py`
5.  `src/preprocess/run_ocr.py`
6.  `src/retrieval/extract_introductions.py`
7.  `src/novelty_assessment/pipeline.py`

Each script accepts command-line arguments. Use the `--help` flag for more information on each one.

## 📂 Project Structure and Expected Outputs

After running the pipeline for a submission with the ID `{submission_id}`, you will find the following structure inside the `data/{submission_id}/` directory:

```
data/{submission_id}/
├── {submission_id}_fulltext.tei.xml # GROBID XML output
├── {submission_id}.json             # Extracted metadata and enriched citations
├── ocr/                             # OCR outputs for all papers
├── introductions/                   # Extracted introductions for all papers
├── related_work_data/
│   ├── ranked_papers.json           # Final list of ranked related papers
│   └── pdfs/                        # Downloaded PDFs of related papers
├── structured_representation.json   # LLM-extracted structured data for all papers
├── research_landscape.txt           # Generated research landscape analysis
├── novelty_delta_analysis.txt       # Detailed novelty assessment
└── summary.txt                      # **Final summary for reviewers**
```

**Key Output Files:**

1.  **`summary.txt`**: The main output. A concise, 5-sentence summary designed to give a peer reviewer a quick, evidence-based overview of the paper's novelty.
2.  **`novelty_delta_analysis.txt`**: A detailed technical analysis comparing the submission to related work across multiple dimensions.
3.  **`research_landscape.txt`**: An LLM-generated analysis of the research area, identifying key methods, problems, and clusters.
4.  **`ranked_papers.json`**: The list of related works identified and ranked by the retrieval system, which form the basis of the analysis.

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## 🙏 Citation

If you use this work, please cite our paper:
```
@misc{afzal2025notnovelenoughenriching,
      title={Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback},
      author={Osama Mohammed Afzal and Preslav Nakov and Tom Hope and Iryna Gurevych},
      year={2025},
      eprint={2508.10795},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.10795},
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.