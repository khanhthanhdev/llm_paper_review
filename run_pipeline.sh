#!/bin/bash

# Novelty Assessment Pipeline Runner
# Configured for the current environment

# Configuration
# Leave SUBMISSION_ID empty to automatically generate one from the PDF name + timestamp
SUBMISSION_ID=""
PDF_PATH="/workspace/llm_paper_review/seq2seq.pdf"
DATA_DIR="data"
GROBID_URL="http://localhost:8070"
MINERU_URL="http://localhost:8000"  # Update if MinerU runs on different port
MODEL="gpt-4o"  # Updated to use available OpenAI model
TEMPERATURE="0.0"
OCR_WORKERS="4"  # Increased for faster processing - adjust based on your system

# Check prerequisites
echo "Checking prerequisites..."

# Check if GROBID is running
if curl -s "$GROBID_URL/api/isalive" > /dev/null 2>&1; then
    echo "✓ GROBID is running at $GROBID_URL"
else
    echo "✗ GROBID is not running at $GROBID_URL"
    echo "  Please start GROBID service first"
    exit 1
fi

# Check if PDF exists
if [ -f "$PDF_PATH" ]; then
    echo "✓ PDF found at $PDF_PATH"
else
    echo "✗ PDF not found at $PDF_PATH"
    exit 1
fi

# Auto-generate a submission ID when not provided
if [ -z "$SUBMISSION_ID" ]; then
    pdf_filename=$(basename -- "$PDF_PATH")
    pdf_stem="${pdf_filename%.*}"
    pdf_stem=$(echo "$pdf_stem" | tr '[:upper:]' '[:lower:]')
    sanitized=$(echo "$pdf_stem" | tr -c 'a-z0-9' '-')
    sanitized=$(echo "$sanitized" | sed -E 's/-+/-/g' | sed -E 's/^-+|-+$//g')
    timestamp=$(date +"%Y%m%d-%H%M%S")
    SUBMISSION_ID="${sanitized:-submission}-${timestamp}"
fi

echo ""
echo "Starting Novelty Assessment Pipeline..."
echo "Submission ID: $SUBMISSION_ID"
echo "PDF: $PDF_PATH"
echo "Data Directory: $DATA_DIR"
echo "GROBID URL: $GROBID_URL"
echo "MinerU URL: $MINERU_URL"
echo ""



# Optional: Check MinerU service (comment out if running MinerU locally)
# if curl -s --connect-timeout 5 "$MINERU_URL/health" > /dev/null 2>&1; then
#     echo "✓ MinerU service is running at $MINERU_URL"
# else
#     echo "⚠ MinerU service is not running at $MINERU_URL"
#     echo "  The pipeline will attempt to use local MinerU installation"
# fi

echo ""
echo "Starting pipeline..."

# Run the pipeline
python src/full_pipeline_runner.py \
  --submission-id "$SUBMISSION_ID" \
  --pdf "$PDF_PATH" \
  --data-dir "$DATA_DIR" \
  --grobid-url "$GROBID_URL" \
  --mineru-url "$MINERU_URL" \
  --model "$MODEL" \
  --temperature "$TEMPERATURE" \
  --ocr-workers "$OCR_WORKERS" \
  --verbose

echo ""
echo "Pipeline completed!"
echo "Check results in: $DATA_DIR/$SUBMISSION_ID/"
