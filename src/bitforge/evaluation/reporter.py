"""Reporter: format aggregated results into human-readable tables and Markdown."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from bitforge.utils.metrics import mean_pm_std


def format_results_table(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of summary dicts into a DataFrame.

    Each summary is the output of run_multi_seed: {config, seeds, aggregated, runs}.
    """
    rows = []
    for s in summaries:
        cfg = s["config"]
        agg = s["aggregated"]
        row = {
            "model": cfg.get("model", {}).get("name", "?"),
            "dataset": cfg.get("data", {}).get("name", "?"),
            "tag": cfg.get("runtime", {}).get("tag", ""),
            "seeds": ",".join(str(x) for x in s["seeds"]),
            "n_seeds": len(s["seeds"]),
            "params": s["runs"][0].get("n_params", 0) if s["runs"] else 0,
            "bits": s["runs"][0].get("n_bits", 0) if s["runs"] else 0,
            "final_top1": mean_pm_std(agg, "final.top1"),
            "best_top1": mean_pm_std(agg, "best.top1"),
            "final_loss": mean_pm_std(agg, "final.loss"),
            "train_time_s": mean_ps(agg, "train_time_s"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def mean_ps(agg: Dict[str, Dict[str, float]], key: str) -> str:
    if key not in agg:
        return "N/A"
    a = agg[key]
    return f"{a['mean']:.1f} ± {a['std']:.1f}"


class Reporter:
    """Render Markdown reports from result summaries."""

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_markdown(self, summaries: List[Dict[str, Any]], filename: str = "report.md") -> str:
        df = format_results_table(summaries)
        lines = ["# BitForge Results Report", "", "## Aggregated Results", "", df.to_markdown(index=False), ""]
        # per-model details
        lines.append("## Per-Experiment Details")
        for s in summaries:
            cfg = s["config"]
            agg = s["aggregated"]
            model = cfg.get("model", {}).get("name", "?")
            ds = cfg.get("data", {}).get("name", "?")
            tag = cfg.get("runtime", {}).get("tag", "")
            lines.append(f"\n### {model} / {ds} / {tag}")
            lines.append(f"- Seeds: {s['seeds']}")
            lines.append(f"- Params: {s['runs'][0].get('n_params', 0):,}")
            lines.append(f"- Storage: {s['runs'][0].get('n_bits', 0):,} bits")
            lines.append(f"- Best top1: {mean_ps(agg, 'best.top1')}")
            lines.append(f"- Final top1: {mean_ps(agg, 'final.top1')}")
            lines.append(f"- Train time: {mean_ps(agg, 'train_time_s')} s")
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path
