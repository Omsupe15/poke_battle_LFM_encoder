import argparse
import json
import os
from collections import Counter
from datasets import load_dataset
from game_encoder.action_space import ACTION_SPACE


def verify_dataset(file_path: str):
    """
    Validates a generated JSONL dataset for formatting, class balance,
    token lengths, and Hugging Face compatibility.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return False

    print(f"\n[INFO] Validating dataset: {file_path}")
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    total_lines = 0
    corrupt_lines = 0
    label_counts = Counter()
    token_lengths = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                corrupt_lines += 1
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                corrupt_lines += 1
                continue

            if "text" not in data or "label" not in data or "label_id" not in data:
                corrupt_lines += 1
                continue

            label = data["label"]
            label_counts[label] += 1
            tokens = data["text"].split()
            token_lengths.append(len(tokens))

    print(f"  File Size           : {file_size_mb:.2f} MB")
    print(f"  Total Transitions   : {total_lines}")
    print(f"  Corrupt/Empty Lines : {corrupt_lines}")

    if total_lines == 0 or total_lines == corrupt_lines:
        print("[ERROR] Dataset contains no valid records!")
        return False

    avg_tokens = sum(token_lengths) / len(token_lengths)
    min_tokens = min(token_lengths)
    max_tokens = max(token_lengths)

    print(f"  Token Lengths (word): Min={min_tokens}, Mean={avg_tokens:.1f}, Max={max_tokens}")
    print("\n--- Action Label Distribution ---")
    move_total = sum(label_counts[f"MOVE_{i}"] for i in range(1, 5))
    switch_total = sum(label_counts[f"SWITCH_{i}"] for i in range(1, 6))

    for action in ACTION_SPACE:
        cnt = label_counts[action]
        pct = (cnt / total_lines) * 100
        bar = "#" * int(pct // 2)
        print(f"  {action:10s} : {cnt:6d} ({pct:5.1f}%) | {bar}")

    print(f"  Total Moves    : {move_total} ({(move_total / total_lines) * 100:.1f}%)")
    print(f"  Total Switches : {switch_total} ({(switch_total / total_lines) * 100:.1f}%)")

    # Verify Hugging Face ingestion
    print("\n--- Hugging Face Datasets Ingestion Test ---")
    try:
        hf_dataset = load_dataset("json", data_files=file_path, split="train")
        print(f"[SUCCESS] Successfully ingested into Hugging Face Dataset: {hf_dataset}")
        print("  Sample Record:")
        print(f"    text     : {hf_dataset[0]['text'][:100]}...")
        print(f"    label    : {hf_dataset[0]['label']}")
        print(f"    label_id : {hf_dataset[0]['label_id']}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load dataset with Hugging Face: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify poke-env JSONL dataset")
    parser.add_argument("--file", type=str, required=True, help="Path to JSONL dataset file")
    args = parser.parse_args()
    verify_dataset(args.file)


if __name__ == "__main__":
    main()
