from datetime import datetime as dt
from huggingface_hub import HfApi
from config import DATASET_BASE_DIR, HF_AUTH_TOKEN, REPO_ID

def save_dataset_to_hf_hub(pack_nb: int = 80):
    if not HF_AUTH_TOKEN:
        print("No HF token found in configuration/environment. Stopping upload.")
        return

    api = HfApi(token=HF_AUTH_TOKEN)

    # Uploads the root mc-items-32x
    api.upload_folder(
        folder_path=str(DATASET_BASE_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"feat: upload {pack_nb}p subset at {dt.now().isoformat()}"
    )
    print(f"Successfully uploaded dataset directory to Hugging Face: {REPO_ID}")

if __name__ == "__main__":
    save_dataset_to_hf_hub(80)