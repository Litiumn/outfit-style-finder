import os
import json
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm

from dotenv import load_dotenv
import os

# ============================================================
# LOAD TOKEN
# ============================================================

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

# ============================================================
# LOAD FINAL MERGED DATA
# ============================================================

with open("final_merged.json") as f:
    merged_data = json.load(f)

# Collect needed item_ids
needed_item_ids = set()

for outfit in merged_data:
    for item in outfit["items"]:
        needed_item_ids.add(str(item["item_id"]))

print(f"Need {len(needed_item_ids)} unique item images")

# ============================================================
# LOAD HUGGINGFACE DATASET
# ============================================================

print("Loading dataset from HuggingFace...")
df = pd.read_parquet(
    "hf://datasets/mvasil/polyvore-outfits/data/nondisjoint/train.parquet",
    storage_options={"token": HF_TOKEN}
)

df["item_id"] = df["item_id"].astype(str)

# Filter only needed items
df = df[df["item_id"].isin(needed_item_ids)]

print(f"Found {len(df)} matching images")

# ============================================================
# SAVE IMAGES
# ============================================================

os.makedirs("images", exist_ok=True)

for _, row in tqdm(df.iterrows(), total=len(df)):
    item_id = row["item_id"]
    img_bytes = row["image"]["bytes"]

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(f"images/{item_id}.jpg", "JPEG")
    except Exception as e:
        print(f"Error saving {item_id}: {e}")

print("✅ All images downloaded!")