import os
import zipfile
import gdown

# Replace this with the actual Google Drive file ID after uploading comp_stock_data.zip
DATASET_ID = "1G5RcSka58WehJx0LxyFT4Z_eLUCpbk-E"
ZIP_FILE = "comp_stock_data.zip"
OUTPUT_DIR = "."


def download_dataset():
    if os.path.exists("comp_stock_data"):
        print("Dataset already exists. Skipping download.")
        return

    url = f"https://drive.google.com/uc?id={DATASET_ID}"

    print("Downloading dataset (~5GB)...")
    gdown.download(url, ZIP_FILE, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)

    os.remove(ZIP_FILE)

    print("Dataset ready.")


if __name__ == "__main__":
    download_dataset()