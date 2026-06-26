"""Remove completions/features entries for specified models."""

from datasets import Dataset, load_from_disk
from tqdm import tqdm

SRC = "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci_qwen235_annotated-features"
DST = "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci_qwen235_annotated-features-filtered"

REMOVE = {"Trinity-Mini", "Phi-4-mini-instruct"}

print("Loading dataset...", flush=True)
dataset = load_from_disk(SRC)
print(dataset)

models = {c["model"] for row in dataset["completions"][:10] for c in row}
print(f"Models (sample): {models}")

def remove_models(row):
    filtered = [
        (c, f) for c, f in zip(row["completions"], row["features"])
        if c["model"] not in REMOVE
    ]
    if filtered:
        cs, fs = zip(*filtered)
    else:
        cs, fs = [], []
    row["completions"] = list(cs)
    row["features"] = list(fs)
    return row


dataset = dataset.map(remove_models, desc="Filtering models", num_proc=128)

print(dataset)

print("Saving to disk...", flush=True)
dataset.save_to_disk(DST)
print(f"Saved to {DST}")
