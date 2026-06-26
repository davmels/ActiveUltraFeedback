from argparse import ArgumentParser
from datasets import Dataset, load_from_disk
import json
from tqdm import tqdm


def convert(source_path: str, output_path: str):
    source = load_from_disk(source_path)
    if "train" in source:
        source = source["train"]

    rows = []
    for example in tqdm(source, desc="Converting"):
        prompt = example["chosen"][:-1]
        annotations = json.loads(example["annotations"])

        completions = [
            {
                "response_text": ann["response"],
                "annotations": ann["detailed_annotations"],
                "model_name": ann["model"],
                "overall_score": ann["final_score"],
            }
            for ann in annotations
        ]

        rows.append({
            "prompt": prompt,
            "completions": completions,
        })

    dataset = Dataset.from_list(rows)
    print(dataset)
    print("Example row:")
    print(f"  prompt ({len(dataset[0]['prompt'])} turns): {dataset[0]['prompt'][:2]}")
    print(f"  completions: {len(dataset[0]['completions'])}")
    dataset.save_to_disk(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_path", required=True, help="Path to source HF dataset")
    parser.add_argument("--output_path", required=True, help="Path to save converted dataset")
    args = parser.parse_args()
    convert(args.source_path, args.output_path)
