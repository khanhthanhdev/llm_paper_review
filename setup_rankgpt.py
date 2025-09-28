#!/usr/bin/env python3
"""
Setup script for RankGPT dependency.
This script automatically clones and sets up RankGPT from GitHub.
"""

import subprocess
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_rankgpt(target_dir: str = None):
    """
    Clone and setup RankGPT from GitHub.
    
    Args:
        target_dir: Target directory to clone RankGPT. If None, uses ~/.cache/llm_paper_review/RankGPT
    """
    if target_dir is None:
        target_dir = Path.home() / ".cache" / "llm_paper_review" / "RankGPT"
    else:
        target_dir = Path(target_dir)
    
    # Create parent directory if it doesn't exist
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if RankGPT already exists
    if target_dir.exists() and (target_dir / "rank_gpt.py").exists():
        logger.info(f"RankGPT already exists at: {target_dir}")
        return str(target_dir)
    
    try:
        # Clone RankGPT repository
        logger.info(f"Cloning RankGPT to: {target_dir}")
        subprocess.run([
            "git", "clone",
            "https://github.com/sunnweiwei/RankGPT.git",
            str(target_dir)
        ], check=True, capture_output=True, text=True)
        
        logger.info("Successfully cloned RankGPT")
        
        # Verify the installation
        if (target_dir / "rank_gpt.py").exists():
            logger.info("RankGPT setup completed successfully")
            return str(target_dir)
        else:
            logger.error("RankGPT cloned but rank_gpt.py not found")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone RankGPT: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error setting up RankGPT: {e}")
        return None

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup RankGPT dependency")
    parser.add_argument(
        "--target-dir", 
        type=str,
        help="Target directory to clone RankGPT (default: ~/.cache/llm_paper_review/RankGPT)"
    )
    
    args = parser.parse_args()
    
    result = setup_rankgpt(args.target_dir)
    if result:
        print(f"✅ RankGPT setup successful: {result}")
        print(f"You can now import RankGPT by adding this path to sys.path:")
        print(f"sys.path.insert(0, '{result}')")
        return 0
    else:
        print("❌ RankGPT setup failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())