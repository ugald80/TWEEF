"""Summarize the paired German aggregation benchmark without retraining models.

Scores across independently calibrated splits are descriptive. Ranking changes are
computed within each paired seed, never from the ranking of a grand mean.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tweef-german")
import numpy as np
import pandas as pd

MODELS = ["DT", "RF", "ET", "LR", "L-SVM", "AB", "GB", "XGB"]
OPERATORS = ["lwa", "gwa", "sl"]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def compact_scores(uniform, output):
    corrected = uniform[uniform.implementation == "corrected"]
    records, paired_frames = [], []
    for baseline in ("legacy_native", "legacy_quartiles"):
        before = uniform[uniform.implementation == baseline]
        paired = corrected.merge(before, on=["seed", "model", "operator"], suffixes=("_new", "_old"), validate="one_to_one")
        paired["delta"] = paired.compute_metric_new - paired.compute_metric_old
        paired["baseline"] = baseline
        paired_frames.append(paired[["seed", "model", "operator", "baseline", "compute_metric_old", "compute_metric_new", "delta"]])
        for (model, operator), group in paired.groupby(["model", "operator"]):
            records.append({"model": model, "operator": operator, "baseline": baseline,
                            "repetitions": len(group), "old_mean": group.compute_metric_old.mean(),
                            "old_std": group.compute_metric_old.std(), "new_mean": group.compute_metric_new.mean(),
                            "new_std": group.compute_metric_new.std(), "paired_delta_mean": group.delta.mean(),
                            "paired_delta_std": group.delta.std(), "paired_max_abs_delta": group.delta.abs().max()})
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(output / "paired_scores_audit.csv", index=False)
    compact = pd.DataFrame(records)
    compact.to_csv(output / "comparison_by_model.csv", index=False)
    by_operator = paired.groupby(["baseline", "operator"]).agg(
        paired_cases=("delta", "size"), mean_delta=("delta", "mean"), std_delta=("delta", "std"),
        mean_abs_delta=("delta", lambda series: series.abs().mean()), max_abs_delta=("delta", lambda series: series.abs().max()))
    by_operator.to_csv(output / "comparison_by_operator.csv")
    return compact, by_operator.reset_index().to_dict("records")


def rank_changes(uniform, output):
    ranks = uniform.copy()
    ranks["rank"] = ranks.groupby(["implementation", "seed", "operator"])["compute_metric"].rank(ascending=False, method="average")
    rows = []
    for baseline in ("legacy_native", "legacy_quartiles"):
        new = ranks[ranks.implementation == "corrected"]
        old = ranks[ranks.implementation == baseline]
        paired = new.merge(old, on=["seed", "model", "operator"], suffixes=("_new", "_old"), validate="one_to_one")
        for (seed, operator), group in paired.groupby(["seed", "operator"]):
            old_top = sorted(group.loc[group.rank_old == group.rank_old.min(), "model"].tolist())
            new_top = sorted(group.loc[group.rank_new == group.rank_new.min(), "model"].tolist())
            rho = (group.rank_new.corr(group.rank_old, method="spearman")
                   if group.rank_new.nunique() > 1 and group.rank_old.nunique() > 1 else np.nan)
            rows.append({"baseline": baseline, "seed": int(seed), "operator": operator,
                         "changed_model_ranks": int((group.rank_new != group.rank_old).sum()),
                         "mean_abs_rank_delta": float((group.rank_new - group.rank_old).abs().mean()),
                         "spearman_rho": None if not np.isfinite(rho) else float(rho),
                         "top_set_before": ",".join(old_top), "top_set_after": ",".join(new_top),
                         "top_set_changed": old_top != new_top})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "paired_rank_changes.csv", index=False)
    return [{"baseline": baseline, "operator": operator, "paired_splits": len(group),
             "splits_with_rank_changes": int((group.changed_model_ranks > 0).sum()),
             "splits_with_top_set_changes": int(group.top_set_changed.sum()),
             "mean_spearman_rho": float(group.spearman_rho.mean())}
            for (baseline, operator), group in frame.groupby(["baseline", "operator"])]


def sl_sensitivity(scores, output):
    subset = scores[(scores.operator == "sl") & (scores.scale == "normalized")]
    ranges = subset.groupby(["implementation", "seed", "model"])["compute_metric"].agg(["min", "max"])
    ranges["score_range"] = ranges["max"] - ranges["min"]
    ranges.reset_index().to_csv(output / "sl_weight_sensitivity.csv", index=False)
    return [{"implementation": implementation, "seed_model_cases": len(group),
             "cases_responding_to_weights": int((group.score_range > 1e-12).sum()),
             "mean_score_range": float(group.score_range.mean()), "maximum_score_range": float(group.score_range.max())}
            for implementation, group in ranges.groupby(level="implementation")]


def gwa_hesitation(output):
    rows = []
    with gzip.open(output / "details_corrected.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record["operator"] != "gwa" or record["scale"] != "normalized":
                continue
            if record["configuration"] != "uniform" and not record["configuration"].startswith("hesitation_"):
                continue
            fused = record["result"]["details"]["fused_ifv"]
            rows.append({"seed": record["seed"], "model": record["model"], "configuration": record["configuration"],
                         "input_hesitation": record["hesitation"], "score": record["compute_metric"],
                         "mu": fused["mu"], "nu": fused["nu"], "hesitation": fused["hesitation"],
                         "mass_error": abs(fused["mu"] + fused["nu"] + fused["hesitation"] - 1)})
    frame = pd.DataFrame(rows)
    reference = frame[frame.configuration == "uniform"][["seed", "model", "score"]].rename(columns={"score": "score_h0"})
    frame = frame.merge(reference, on=["seed", "model"], validate="many_to_one")
    frame["score_delta_h0"] = frame.score - frame.score_h0
    frame.to_csv(output / "gwa_hesitation_audit.csv", index=False)
    grouped = frame.groupby("configuration").agg(
        cases=("score", "size"), mean_score=("score", "mean"), std_score=("score", "std"),
        mean_hesitation=("hesitation", "mean"), min_hesitation=("hesitation", "min"), max_hesitation=("hesitation", "max"),
        mean_paired_score_delta=("score_delta_h0", "mean"), max_mass_error=("mass_error", "max"))
    grouped.to_csv(output / "gwa_hesitation_summary.csv")
    return grouped.reset_index().to_dict("records")


def metrics_and_timing(output):
    raw = pd.read_csv(output / "raw_metrics.csv")
    ec = raw.groupby(["partition", "model"])["Effective Complexity"].agg(["count", "mean", "std"])
    zeros = raw.assign(ec_zero=raw["Effective Complexity"].abs() <= 1e-15).groupby(["partition", "model"]).ec_zero.mean()
    ec["zero_rate"] = zeros
    ec.reset_index().to_csv(output / "effective_complexity_audit.csv", index=False)
    details, seconds = [], 0.
    for file in sorted(output.glob("metrics_seed_*.json")):
        payload = json.loads(file.read_text())
        for model, info in payload["models"].items():
            seconds += info["seconds"]
            for partition in ("calibration", "test"):
                detail = info[partition]
                details.append({"seed": payload["seed"], "model": model, "partition": partition,
                                "reference_entropy": detail["reference_entropy"],
                                "entropy_short_circuit": detail["ec_entropy_short_circuit"],
                                "attribution_source": detail["attribution_source"]})
    pd.DataFrame(details).to_csv(output / "effective_complexity_diagnostics.csv", index=False)
    return {"training_and_metric_seconds_sum": seconds,
            "time_definition": "Sum of recorded per-model training plus calibration/test metric times; not wall-clock duration",
            "ec_zero_rates": ec.reset_index().to_dict("records"),
            "ec_entropy_short_circuit_partition_cases": int(sum(row["entropy_short_circuit"] for row in details)),
            "metric_partition_cases": len(details)}


def plot_comparison(compact, output, repetitions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    frame = compact[compact.baseline == "legacy_native"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), sharey=True, sharex=True)
        positions = np.arange(len(MODELS))
        for ax, operator in zip(axes, OPERATORS):
            group = frame[frame.operator == operator].set_index("model").reindex(MODELS)
            for index, row in enumerate(group.itertuples()):
                ax.plot([row.old_mean, row.new_mean], [index, index], color="0.8", linewidth=1.1, zorder=1)
            ax.errorbar(group.old_mean, positions - .08, xerr=group.old_std, fmt="s", markersize=4.5,
                        color="#C66C20", elinewidth=.9, capsize=2, label="Antes")
            ax.errorbar(group.new_mean, positions + .08, xerr=group.new_std, fmt="o", markersize=4.5,
                        color="#2166AC", elinewidth=.9, capsize=2, label="Corregido")
            ax.set_title(operator.upper(), weight="bold")
            ax.set_xlim(-.08, 1.2)
            ax.axvline(0, color="0.65", linewidth=.6, linestyle=":")
            ax.axvline(1, color="0.65", linewidth=.6, linestyle=":")
            ax.set_xlabel("Trustworthiness [0, 1]")
            ax.set_yticks(positions, MODELS)
            ax.grid(axis="x", color="0.9", linewidth=.6)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Modelo")
        fig.suptitle("German Credit: implementación anterior y corregida", fontsize=14, y=.98)
        fig.legend(handles=[Line2D([], [], marker="s", color="#C66C20", linestyle="", label="Anterior: calibración nativa"),
                            Line2D([], [], marker="o", color="#2166AC", linestyle="", label="Corregida: cuartiles")],
                   loc="lower center", bbox_to_anchor=(.5, .075), ncol=2, frameon=False)
        fig.text(.5, .025, f"Pesos iguales. Puntos: media; barras: ±1 DE en {repetitions} particiones pareadas. Las métricas de entrada son idénticas.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .15, 1, .94))
        fig.savefig(output / "german_before_after.png", dpi=240, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", "--input", dest="results", type=Path, default=Path(__file__).resolve().parent / "results/german")
    args = parser.parse_args()
    output = args.results.resolve()
    required = ["scores.csv", "raw_metrics.csv", "manifest.json", "details_corrected.jsonl.gz"]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        parser.exit(2, "Benchmark must finish before summarizing; missing: " + ", ".join(missing) + "\n")
    scores = pd.read_csv(output / "scores.csv")
    manifest = json.loads((output / "manifest.json").read_text())
    uniform = scores[(scores.configuration == "uniform") & (scores.scale == "normalized")]
    compact, operator_comparison = compact_scores(uniform, output)
    interface = scores.groupby(["implementation", "operator", "scale"]).interface_gap.agg(["count", "mean", "max"]).reset_index()
    interface.to_csv(output / "interface_consistency_audit.csv", index=False)
    failures = {name: json.loads((output / f"failures_{name}.json").read_text())
                for name in scores.implementation.unique()}
    summary = {"protocol": {"dataset_rows": manifest["dataset_rows"], "dataset_sha256": manifest["dataset_sha256"],
                            "seeds": manifest["seeds"], "models": MODELS, "metrics": manifest["metric_names"],
                            "split": manifest["split"], "ec_mode": manifest["ec_mode"],
                            "baseline_commit": manifest["baseline_commit"], "environment": manifest["environment"]},
               "counts": {"trained_models": len(manifest["seeds"]) * len(MODELS), "score_rows": len(scores),
                          "numeric_api_scores": 2 * len(scores), "by_implementation": scores.implementation.value_counts().to_dict(),
                          "failures_by_implementation": {key: len(value) for key, value in failures.items()}},
               "comparison_by_model": compact.to_dict("records"), "comparison_by_operator": operator_comparison,
               "interface_consistency": interface.to_dict("records"), "sl_weight_sensitivity": sl_sensitivity(scores, output),
               "gwa_hesitation": gwa_hesitation(output), "rank_changes_within_paired_splits": rank_changes(uniform, output),
               "timing_and_effective_complexity": metrics_and_timing(output),
               "interpretation": ["Means and SD summarize scores over independently calibrated splits descriptively.",
                                  "Score and rank differences are calculated within matching seed/model/operator pairs.",
                                  "Legacy-native compares actual old default quantiles with corrected quartiles; legacy-quartiles holds calibration fixed.",
                                  "Legacy transform weights were explicitly assigned because its fit API cannot accept weights alongside calibration.",
                                  "Effective Complexity uses the unchanged implementation's absolute cross-entropy/disagreement definition.",
                                  "These checks validate implementation behavior; no aggregation operator is assumed to be universally best."]}
    write_json(output / "audit_summary.json", summary)
    plot_comparison(compact, output, len(manifest["seeds"]))
    print(compact[compact.baseline == "legacy_native"][["model", "operator", "old_mean", "new_mean", "new_std", "paired_delta_mean"]].round(6).to_string(index=False))
    print("SL weight sensitivity:", summary["sl_weight_sensitivity"])
    print("Counts:", summary["counts"])
    print("Saved", output / "audit_summary.json", "and", output / "german_before_after.png")


if __name__ == "__main__":
    main()
