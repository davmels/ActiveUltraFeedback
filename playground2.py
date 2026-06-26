import os

from datasets import DatasetDict, concatenate_datasets, load_from_disk

NUM_PROC = os.cpu_count()

dataset = load_from_disk("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/ahey/MaxMin_Tr_3600-Filtered-Decontaminated/")
# load_from_disk returns a DatasetDict here (split: 'train_split'); flatten to one Dataset
if isinstance(dataset, DatasetDict):
    dataset = concatenate_datasets(list(dataset.values()))


def has_system_prompt(example: dict) -> bool:
    """True if the chosen conversation contains a system-role message."""
    return any(msg["role"] == "system" for msg in example["chosen"])


with_system = dataset.filter(
    has_system_prompt,
    num_proc=NUM_PROC,
    desc="Scanning chosen for system prompts",
)

print(f"{len(with_system)}/{len(dataset)} chosen conversations contain a system prompt")

def extract_roles(batch: dict) -> dict:
    """Per-row set of roles found in the chosen conversation."""
    return {
        "roles": [
            sorted({msg["role"] for msg in chosen}) for chosen in batch["chosen"]
        ]
    }


roles_ds = dataset.map(
    extract_roles,
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=dataset.column_names,
    desc="Collecting roles",
)

roles = set()
for r in roles_ds["roles"]:
    roles.update(r)
print("unique roles in chosen:", sorted(roles))
