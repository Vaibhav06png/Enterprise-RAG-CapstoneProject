# =====================================================
# download_dataset.py
# -----------------------------------------------------
# Downloads the Bitext customer support dataset
# and saves it inside the data/ folder.
# Run this ONCE before building the RAG pipeline.
# =====================================================

import os
import pandas as pd

# Make sure the data folder exists
os.makedirs("data", exist_ok=True)

# Bitext customer support dataset (27K samples)
DATASET_URL = (
    "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

print("Downloading dataset from HuggingFace...")
df = pd.read_csv(DATASET_URL)

# Keep only the columns we actually need for RAG
# instruction = customer query, category/intent = metadata, response = ideal answer
keep_cols = [c for c in ["instruction", "category", "intent", "response"] if c in df.columns]
df = df[keep_cols]

# Save locally so the RAG pipeline can read it
output_path = "data/customer_support_data.csv"
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} rows to {output_path}")
print("Columns:", list(df.columns))
print(df.head(3))
