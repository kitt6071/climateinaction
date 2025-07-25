import re
import os
from collections import defaultdict
from pathlib import Path
import argparse

PART_IDENTIFIERS = {
    "P2.1": "Entity Extraction",
    "P2.2": "Relationship Generation",
    "Classifying threat": "IUCN Classification",
    "Verifying": "Verification",
    "Normalizing species names": "Normalization",
    "relevance for": "Relevance Check (LLM)",
    "to_summary": "Summary Generation"
}

def analyze_log_file(log_path: Path) -> dict:
    """
    Analyzes a single log file and extracts error counts and LLM metrics.
    """
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    current_part = "Unknown"
    current_model = "Unknown"

    metric_regex = re.compile(
        r"LLM Metrics - Model: ([\w/.:-]+), "
        r"Prompt: (\d+), "
        r"Completion: (\d+), "
        r"Total: (\d+)"
        r"(?:, Cost: \$([\d.]+))?"
    )

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            for identifier, name in PART_IDENTIFIERS.items():
                if identifier in line:
                    current_part = name
                    break

            if "ERROR" in line:
                error_part = "Unknown"
                for identifier, name in PART_IDENTIFIERS.items():
                    if identifier in line:
                        error_part = name
                        break
                if current_model != "Unknown":
                    stats[current_model][error_part]["errors"] += 1

            metric_match = metric_regex.search(line)
            if metric_match:
                groups = metric_match.groups()
                model, prompt, completion, total = groups[0], groups[1], groups[2], groups[3]
                cost = float(groups[4]) if len(groups) > 4 and groups[4] is not None else 0.0
                
                current_model = model
                
                part_stats = stats[model][current_part]
                part_stats["total_calls"] += 1
                part_stats["prompt_tokens"] += int(prompt)
                part_stats["completion_tokens"] += int(completion)
                part_stats["total_tokens"] += int(total)
                part_stats["cost"] += cost

    return stats

def merge_stats(main_stats, new_stats):
    """Merges stats from a new file into the main stats dictionary."""
    for model, parts in new_stats.items():
        for part, metrics in parts.items():
            for metric, value in metrics.items():
                main_stats[model][part][metric] += value

def print_report(stats: dict):
    """
    Prints a formatted summary report of the analyzed log data.
    """
    print("--- Pipeline Performance Report ---\n")
    if not stats:
        print("No data found to generate a report.")
        return

    for model, parts in sorted(stats.items()):
        print("=" * 80)
        print(f"📊 Model: {model}")
        print("-" * 80)
        
        for part, metrics in sorted(parts.items()):
            calls = int(metrics.get("total_calls", 0))
            errors = int(metrics.get("errors", 0))

            print(f"  🔹 Part: {part}")
            print(f"    - Errors: {errors}")
            
            if calls > 0:
                avg_prompt = metrics["prompt_tokens"] / calls
                avg_completion = metrics["completion_tokens"] / calls
                avg_total = metrics["total_tokens"] / calls
                avg_cost = metrics["cost"] / calls
                
                print(f"    - LLM Calls: {calls}")
                print(f"    - Average Prompt Tokens: {avg_prompt:,.1f}")
                print(f"    - Average Completion Tokens: {avg_completion:,.1f}")
                print(f"    - Average Total Tokens: {avg_total:,.1f}")
                print(f"    - Average Cost per Call: ${avg_cost:.6f}")
            else:
                print("    - No LLM calls recorded for this part.")
            print()

        print("=" * 80)
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pipeline log files to get error counts and average LLM metrics.")
    parser.add_argument("log_directory", type=str, help="The path to the directory containing your log files.")
    
    args = parser.parse_args()

    log_dir = Path(args.log_directory)
    if not log_dir.is_dir():
        print(f"Error: Directory not found at '{log_dir}'")
        exit(1)

    aggregated_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        print(f"No .log files found in '{log_dir}'")
        exit(1)

    print(f"Found {len(log_files)} log file(s) to analyze...")

    for log_file in log_files:
        print(f"  -> Processing {log_file.name}...")
        file_stats = analyze_log_file(log_file)
        merge_stats(aggregated_stats, file_stats)

    print_report(aggregated_stats)