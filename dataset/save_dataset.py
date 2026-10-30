import subprocess
from pathlib import Path
from config import DATASET_BASE_DIR, update_dataset_card


def save_dataset_via_git(pack_nb: int = 80):
    """
    Updates the dataset card and commits changes directly using
    the local Git repository and Git LFS.
    """
    if not DATASET_BASE_DIR.exists():
        print(f"Dataset directory not found: {DATASET_BASE_DIR}")
        return

    print("Fetching existing LFS objects...")
    subprocess.run(["git", "lfs", "pull"], cwd=DATASET_BASE_DIR, check=True)

    print(f"Updating README.md metadata card (default config: {pack_nb}p)...")
    update_dataset_card(default_config=f"{pack_nb}p")

    print(f"Staging files for {pack_nb}p subset...")
    subprocess.run(["git", "add", "."], cwd=DATASET_BASE_DIR, check=True)

    # 4. Commit and push
    commit_msg = f"feat: add/update {pack_nb}p subset with train/test splits"
    print(f"Creating commit: '{commit_msg}'...")
    subprocess.run(
        ["git", "commit", "-m", commit_msg], cwd=DATASET_BASE_DIR, check=True
    )

    print("Pushing updates to Hugging Face Hub...")
    subprocess.run(["git", "push", "origin", "main"], cwd=DATASET_BASE_DIR, check=True)

    print("Successfully updated and pushed dataset!")


if __name__ == "__main__":
    save_dataset_via_git(80)
