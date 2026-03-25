"""
Download the VideoLLM-online training data from HuggingFace.

Dataset: chenjoya/videollm-online-chat-ego4d-134k
This contains the preprocessed streaming dialogue annotations for Ego4D.

Usage:
    python -m data.download --output_dir datasets/ego4d/v2/annotations
"""

import argparse, os
from huggingface_hub import snapshot_download

def main(output_dir: str, repo_id: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {repo_id} to {output_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
    )
    print(f"Done. Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download VideoLLM-online training data from HuggingFace")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/ego4d/v2/annotations",
        help="Directory to save downloaded annotations",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="chenjoya/videollm-online-chat-ego4d-134k",
        help="HuggingFace dataset repo ID",
    )
    args = parser.parse_args()
    main(args.output_dir, args.repo_id)
