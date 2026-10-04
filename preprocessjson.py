import requests
import os
import json
import pandas as pd
import joblib


def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3:latest",
            "input": text_list
        },
        timeout=120
    )

    if r.status_code != 200:
        print("\n========== OLLAMA ERROR ==========")
        print(r.text)
        print("==================================\n")
        r.raise_for_status()

    data = r.json()

    if "embeddings" not in data:
        print("\nUnexpected Ollama response:")
        print(data)
        raise RuntimeError("No embeddings returned by Ollama.")

    return data["embeddings"]


# Number of chunks sent to Ollama at one time
BATCH_SIZE = 5


jsons = os.listdir("jsons")

my_dicts = []
chunk_id = 0


for json_file in jsons:

    if not json_file.endswith(".json"):
        continue

    print(f"\nCreating Embeddings for {json_file}")

    with open(
        f"jsons/{json_file}",
        "r",
        encoding="utf-8"
    ) as f:

        content = json.load(f)

    chunks = content["chunks"]

    print(f"Total chunks: {len(chunks)}")

    all_embeddings = []

    # --------------------------------------------
    # Process chunks in small batches
    # --------------------------------------------

    for start in range(0, len(chunks), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(chunks))

        batch = chunks[start:end]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        print(
            f"  Embedding chunks "
            f"{start + 1}-{end}"
        )

        embeddings = create_embedding(texts)

        all_embeddings.extend(embeddings)

    # --------------------------------------------
    # Verify
    # --------------------------------------------

    if len(all_embeddings) != len(chunks):

        raise RuntimeError(
            f"Embedding count mismatch!\n"
            f"Chunks: {len(chunks)}\n"
            f"Embeddings: {len(all_embeddings)}"
        )

    # --------------------------------------------
    # Add embeddings
    # --------------------------------------------

    for i, chunk in enumerate(chunks):

        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = all_embeddings[i]

        my_dicts.append(chunk)

        chunk_id += 1


# --------------------------------------------
# Create DataFrame
# --------------------------------------------

df = pd.DataFrame.from_records(my_dicts)

print("\n======================================")
print("Embedding generation completed!")
print("======================================")

print("Total chunks:", len(df))


# --------------------------------------------
# Save
# --------------------------------------------

joblib.dump(df, "embeddings.joblib")

print("\nSaved:")
print("embeddings.joblib")