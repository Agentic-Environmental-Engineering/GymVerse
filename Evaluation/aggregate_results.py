#!/usr/bin/env python3
"""
Results Aggregation Script

This script aggregates evaluation results across models and categories:
1. Calculates average metrics for each model on each category
2. Calculates overall average across all categories for each model
3. Generates summary reports in JSON and CSV formats
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import csv


def find_result_files(results_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Find all result files organized by model and category.

    Returns:
        {model_name: {category: [result_files]}}
    """
    results = defaultdict(lambda: defaultdict(list))

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return {}

    # Check if results_dir is a model directory (contains category subdirs like game, code, etc.)
    # by looking for common category names
    common_categories = {'game', 'code', 'logic', 'tool-use', 'tool_use'}
    subdirs = [d.name for d in results_dir.iterdir() if d.is_dir()]
    is_model_dir = any(cat in subdirs for cat in common_categories)

    if is_model_dir:
        # User specified a model directory directly
        model_name = results_dir.name

        # Iterate through category directories
        for category_dir in results_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            # Find all report.json result files
            for report_file in category_dir.rglob("report.json"):
                results[model_name][category].append(report_file)
    else:
        # Standard case: results_dir contains model directories
        # Iterate through model directories
        for model_dir in results_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Iterate through category directories
            for category_dir in model_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                category = category_dir.name

                # Find all report.json result files
                for report_file in category_dir.rglob("report.json"):
                    results[model_name][category].append(report_file)

    return results


def parse_jsonl_results(report_file: Path) -> Dict[str, Any]:
    """
    Parse a single report.json file and extract metrics.

    Returns:
        Dictionary with metrics (success_rate, avg_reward, etc.)
    """
    metrics = {
        "success_rate": None,
        "avg_reward": None,
        "total_episodes": 0,
        "env_id": None
    }

    try:
        with open(report_file, 'r') as f:
            data = json.load(f)

            # Extract env_id
            metrics["env_id"] = data.get("env_id")

            # Extract metrics from difficulty dimension
            if "dimensions" in data and "difficulty" in data["dimensions"]:
                difficulty_data = data["dimensions"]["difficulty"]

                if difficulty_data.get("status") == "ok":
                    aggregate_metrics = difficulty_data.get("aggregate_metrics", {})

                    # Extract success_rate
                    metrics["success_rate"] = aggregate_metrics.get("avg_success_rate")

                    # Extract total episodes from config
                    if "config" in data:
                        config = data["config"]
                        num_difficulties = config.get("num_difficulties", 1)
                        runs_per_difficulty = config.get("runs_per_difficulty", 0)
                        metrics["total_episodes"] = num_difficulties * runs_per_difficulty

    except Exception as e:
        print(f"WARNING: Error parsing {report_file}: {e}")

    return metrics


def aggregate_category_results(result_files: List[Path]) -> Dict[str, Any]:
    """
    Aggregate results for a single category.

    Returns:
        Dictionary with aggregated metrics
    """
    all_metrics = []

    for result_file in result_files:
        metrics = parse_jsonl_results(result_file)
        if metrics["success_rate"] is not None:
            all_metrics.append(metrics)

    if not all_metrics:
        return {
            "num_envs": 0,
            "avg_success_rate": None,
            "avg_reward": None,
            "total_episodes": 0
        }

    # Calculate averages
    success_rates = [m["success_rate"] for m in all_metrics if m["success_rate"] is not None]
    rewards = [m["avg_reward"] for m in all_metrics if m["avg_reward"] is not None]

    return {
        "num_envs": len(all_metrics),
        "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else None,
        "avg_reward": sum(rewards) / len(rewards) if rewards else None,
        "total_episodes": sum(m["total_episodes"] for m in all_metrics),
        "env_details": all_metrics
    }


def generate_summary_report(results: Dict[str, Dict[str, List[Path]]]) -> Dict[str, Any]:
    """
    Generate comprehensive summary report.

    Returns:
        {
            model_name: {
                "categories": {
                    category: {metrics},
                    ...
                },
                "overall": {aggregated_metrics}
            }
        }
    """
    summary = {}

    for model_name, categories in results.items():
        model_summary = {
            "categories": {},
            "overall": {}
        }

        category_success_rates = []

        for category, result_files in categories.items():
            # Aggregate results for this category
            category_metrics = aggregate_category_results(result_files)
            model_summary["categories"][category] = category_metrics

            # Collect for overall calculation
            if category_metrics["avg_success_rate"] is not None:
                category_success_rates.append(category_metrics["avg_success_rate"])

        # Calculate overall metrics across all categories
        if category_success_rates:
            model_summary["overall"] = {
                "num_categories": len(category_success_rates),
                "avg_success_rate_across_categories": sum(category_success_rates) / len(category_success_rates)
            }
        else:
            model_summary["overall"] = {
                "num_categories": 0,
                "avg_success_rate_across_categories": None
            }

        summary[model_name] = model_summary

    return summary


def save_json_report(summary: Dict[str, Any], output_file: Path):
    """Save summary report as JSON."""
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ JSON report saved to: {output_file}")


def save_csv_report(summary: Dict[str, Any], output_file: Path):
    """Save summary report as CSV."""
    rows = []

    for model_name, model_data in summary.items():
        # Add category-level rows
        for category, metrics in model_data["categories"].items():
            rows.append({
                "model": model_name,
                "category": category,
                "num_envs": metrics.get("num_envs", 0),
                "avg_success_rate": metrics.get("avg_success_rate"),
                "avg_reward": metrics.get("avg_reward"),
                "total_episodes": metrics.get("total_episodes", 0)
            })

        # Add overall row
        overall = model_data["overall"]
        rows.append({
            "model": model_name,
            "category": "OVERALL",
            "num_envs": overall.get("num_categories", 0),
            "avg_success_rate": overall.get("avg_success_rate_across_categories"),
            "avg_reward": None,
            "total_episodes": None
        })

    # Write CSV
    if rows:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ CSV report saved to: {output_file}")


def print_summary_table(summary: Dict[str, Any]):
    """Print a formatted summary table to console."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    for model_name, model_data in summary.items():
        print(f"\n📊 Model: {model_name}")
        print("-" * 80)

        # Print category results
        for category, metrics in model_data["categories"].items():
            success_rate = metrics.get("avg_success_rate")
            num_envs = metrics.get("num_envs", 0)

            if success_rate is not None:
                print(f"  {category:15s}: {success_rate:6.2%} (n={num_envs} envs)")
            else:
                print(f"  {category:15s}: N/A (n={num_envs} envs)")

        # Print overall
        overall = model_data["overall"]
        overall_rate = overall.get("avg_success_rate_across_categories")
        if overall_rate is not None:
            print(f"  {'OVERALL':15s}: {overall_rate:6.2%} (across {overall['num_categories']} categories)")
        else:
            print(f"  {'OVERALL':15s}: N/A")

    print("\n" + "="*80)


def main():
    """Main function to aggregate and report results."""
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing evaluation results (default: ./results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save aggregated reports (default: ./results)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print("="*80)
    print("Results Aggregation")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Find all result files
    print("\n🔍 Scanning for result files...")
    results = find_result_files(results_dir)

    if not results:
        print("❌ No results found!")
        return

    print(f"✓ Found results for {len(results)} model(s)")
    for model_name, categories in results.items():
        print(f"  - {model_name}: {len(categories)} categories")

    # Generate summary report
    print("\n📊 Generating summary report...")
    summary = generate_summary_report(results)

    # Print summary table
    print_summary_table(summary)

    # Save reports
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / "summary_report.json"
    save_json_report(summary, json_file)

    csv_file = output_dir / "summary_report.csv"
    save_csv_report(summary, csv_file)

    print("\n✅ Aggregation complete!")
    print(f"📁 Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
