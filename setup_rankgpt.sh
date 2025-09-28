#!/bin/bash

# Setup script for RankGPT dependency
# This script automatically downloads RankGPT from GitHub

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKGPT_DIR="${SCRIPT_DIR}/external/RankGPT"

echo "Setting up RankGPT dependency..."

# Create external directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/external"

# Check if RankGPT already exists
if [ -d "${RANKGPT_DIR}" ] && [ -f "${RANKGPT_DIR}/rank_gpt.py" ]; then
    echo "✅ RankGPT already exists at: ${RANKGPT_DIR}"
else
    echo "📥 Downloading RankGPT from GitHub..."
    
    # Clone RankGPT repository
    git clone https://github.com/sunnweiwei/RankGPT.git "${RANKGPT_DIR}"
    
    if [ -f "${RANKGPT_DIR}/rank_gpt.py" ]; then
        echo "✅ RankGPT downloaded successfully to: ${RANKGPT_DIR}"
    else
        echo "❌ RankGPT download failed - rank_gpt.py not found"
        exit 1
    fi
fi

echo "🔧 RankGPT setup completed!"
echo "📍 RankGPT location: ${RANKGPT_DIR}"
echo ""
echo "To use RankGPT in your Python code, add this to your sys.path:"
echo "import sys"
echo "sys.path.insert(0, '${RANKGPT_DIR}')"