"""Audit the isolated EC correction on paired German benchmark artifacts.

Both directories must contain completed benchmark outputs. This script does not
train models, modify either implementation, or overwrite the earlier results.
The comparison uses the corrected aggregation implementation in both runs;
only EC's fidelity loss and its induced EC calibration may differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tweef-ec-comparison")

import numpy as np
import pandas as pd


MODELS = ["DT", "RF", "ET", "LR", "L-SVM", "AB", "GB", "XGB"]
OPERATORS = ["lwa", "gwa", "sl"]
EC = "Effective Complexity"
UNCHANGED_METRICS = ["Accuracy", "equal Opportunity"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def require_equal(before, after, description):
    if before != after:
        raise ValueError(f"Unpaired experiment: {description} differs")


def audit_pairing(before, after):
    """Require fixed inputs exactly; audit entropy's dtype-rounding separately."""
    old_manifest = read_json(before / "manifest.json")
    new_manifest = read_json(after / "manifest.json")
    for key in ("seeds", "dataset_sha256", "dataset_rows", "split", "metric_names",
                "ec_loss_threshold", "ec_repeats", "environment"):
        require_equal(old_manifest[key], new_manifest[key], key)
    seeds = old_manifest["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("The compared manifests require unique nonempty seeds")
    checked_partitions = 0
    exact_fields = {"predictions", "attributions", "attribution_source"}
    entropy_absolute_tolerance = 2e-7
    entropy_differences = []
    entropy_unavailable_cases = 0
    for seed in seeds:
        old = read_json(before / f"metrics_seed_{seed}.json")
        new = read_json(after / f"metrics_seed_{seed}.json")
        for key in ("seed", "splits", "feature_names", "preprocessing"):
            require_equal(old[key], new[key], f"seed={seed}, {key}")
        require_equal(set(old["models"]), set(MODELS), f"seed={seed}, old model set")
        require_equal(set(new["models"]), set(MODELS), f"seed={seed}, new model set")
        for model in MODELS:
            require_equal(old["models"][model]["parameters"], new["models"][model]["parameters"],
                          f"seed={seed}, model={model}, parameters")
            for partition in ("calibration", "test"):
                for key in exact_fields:
                    require_equal(old["models"][model][partition][key],
                                  new["models"][model][partition][key],
                                  f"seed={seed}, model={model}, {partition}, {key}")
                old_entropy = old["models"][model][partition]["reference_entropy"]
                new_entropy = new["models"][model][partition]["reference_entropy"]
                if old_entropy is None or new_entropy is None:
                    require_equal(old_entropy, new_entropy,
                                  f"seed={seed}, model={model}, {partition}, unavailable entropy")
                    entropy_unavailable_cases += 1
                else:
                    difference = abs(old_entropy - new_entropy)
                    if not np.isfinite(difference) or difference > entropy_absolute_tolerance:
                        raise ValueError(
                            f"Unpaired experiment: seed={seed}, model={model}, {partition}, "
                            f"reference entropy difference {difference!r} exceeds {entropy_absolute_tolerance}"
                        )
                    entropy_differences.append(difference)
                for metric in UNCHANGED_METRICS:
                    require_equal(old[partition][model][metric], new[partition][model][metric],
                                  f"seed={seed}, model={model}, {partition}, {metric}")
                checked_partitions += 1
    return {"exact_pairing": True, "seeds": seeds, "models": MODELS,
            "trained_model_pairs": len(seeds) * len(MODELS),
            "checked_metric_partitions": checked_partitions,
            "checked_fields": ["dataset", "split indices", "feature encoding/scaling", "model parameters",
                               "predictions", "attributions", *UNCHANGED_METRICS],
            "reference_entropy_audit": {
                "absolute_tolerance": entropy_absolute_tolerance,
                "relative_tolerance": 0.0,
                "maximum_absolute_difference": max(entropy_differences, default=0.0),
                "finite_compared_cases": len(entropy_differences),
                "both_unavailable_cases": entropy_unavailable_cases,
                "reason": "Historical probability entropy used the model output dtype; the correction computes this diagnostic in float64.",
            },
            "probability_vector_limitation": "Historical artifacts do not store per-row probability vectors; exact probability-vector equality cannot be audited retrospectively.",
            "dataset_sha256": old_manifest["dataset_sha256"]}


def paired_metrics(before, after, output):
    old = pd.read_csv(before / "raw_metrics.csv")
    new = pd.read_csv(after / "raw_metrics.csv")
    keys = ["seed", "partition", "model"]
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError("Duplicate metric rows")
    pairs = old.merge(new, on=keys, suffixes=("_before", "_after"), validate="one_to_one", how="outer", indicator=True)
    if not pairs._merge.eq("both").all():
        raise ValueError("Metric rows are not matched between runs")
    pairs = pairs.drop(columns="_merge")
    for name in UNCHANGED_METRICS:
        if not np.array_equal(pairs[name + "_before"], pairs[name + "_after"]):
            raise ValueError(f"Saved CSV values differ for {name}")
    pairs["ec_delta"] = pairs[EC + "_after"] - pairs[EC + "_before"]
    pairs.to_csv(output / "ec_metric_pairs.csv", index=False)
    rows = []
    for (partition, model), group in pairs.groupby(["partition", "model"]):
        old_values, new_values = group[EC + "_before"], group[EC + "_after"]
        rows.append({"partition": partition, "model": model, "repetitions": len(group),
                     "old_mean": old_values.mean(), "old_std": old_values.std(),
                     "new_mean": new_values.mean(), "new_std": new_values.std(),
                     "old_zero_rate": old_values.eq(0).mean(), "new_zero_rate": new_values.eq(0).mean(),
                     "paired_delta_mean": group.ec_delta.mean(), "paired_delta_std": group.ec_delta.std(),
                     "changed_cases": int(group.ec_delta.abs().gt(1e-12).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "ec_comparison_by_model.csv", index=False)
    return summary


def uniform_scores(folder):
    scores = pd.read_csv(folder / "scores_corrected.csv")
    selected = scores[scores.configuration.eq("uniform") & scores.scale.eq("normalized")].copy()
    if selected.duplicated(["seed", "model", "operator"]).any():
        raise ValueError("Duplicate uniform score rows")
    return selected


def score_comparison(before, after, output):
    old, new = uniform_scores(before), uniform_scores(after)
    keys = ["seed", "model", "operator"]
    pairs = old.merge(new, on=keys, suffixes=("_before", "_after"), validate="one_to_one", how="outer", indicator=True)
    if not pairs._merge.eq("both").all():
        raise ValueError("Score rows are not matched between runs")
    pairs = pairs.drop(columns="_merge")
    pairs["score_delta"] = pairs.compute_metric_after - pairs.compute_metric_before
    pairs.to_csv(output / "ec_score_pairs.csv", index=False)
    rows = []
    for (model, operator), group in pairs.groupby(["model", "operator"]):
        rows.append({"model": model, "operator": operator, "repetitions": len(group),
                     "old_mean": group.compute_metric_before.mean(), "old_std": group.compute_metric_before.std(),
                     "new_mean": group.compute_metric_after.mean(), "new_std": group.compute_metric_after.std(),
                     "paired_delta_mean": group.score_delta.mean(), "paired_delta_std": group.score_delta.std()})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "ec_score_comparison.csv", index=False)
    dispersion = []
    for label, frame in (("before", old), ("after", new)):
        for operator, group in frame.groupby("operator"):
            model_means = group.groupby("model").compute_metric.mean()
            seed_model_sd = group.groupby("seed").compute_metric.std()
            model_seed_sd = group.groupby("model").compute_metric.std()
            dispersion.append({"version": label, "operator": operator,
                               "sd_across_eight_model_means": model_means.std(),
                               "mean_within_seed_sd_across_models": seed_model_sd.mean(),
                               "mean_within_model_sd_across_seeds": model_seed_sd.mean(),
                               "minimum_model_mean": model_means.min(), "maximum_model_mean": model_means.max()})
    dispersion = pd.DataFrame(dispersion)
    dispersion.to_csv(output / "ec_score_dispersion.csv", index=False)
    return summary, dispersion


def plot_comparison(metrics, scores, dispersion, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ec = metrics[metrics.partition.eq("test")].set_index("model").reindex(MODELS)
    sl = scores[scores.operator.eq("sl")].set_index("model").reindex(MODELS)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.5), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
        positions = np.arange(len(MODELS))
        for ax, frame, title, xlabel in [(axes[0], ec, "Effective Complexity", "Score de interpretabilidad"),
                                         (axes[1], sl, "Trustworthiness: SL", "Score normalizado")]:
            for index, row in enumerate(frame.itertuples()):
                ax.plot([row.old_mean, row.new_mean], [index, index], color="0.8", linewidth=1)
            ax.errorbar(frame.old_mean, positions - .08, xerr=frame.old_std, fmt="s", markersize=4,
                        color="#C66C20", capsize=2, linewidth=.8, label="Antes de corregir EC")
            ax.errorbar(frame.new_mean, positions + .08, xerr=frame.new_std, fmt="o", markersize=4,
                        color="#2166AC", capsize=2, linewidth=.8, label="EC corregida")
            ax.set_yticks(positions, MODELS)
            ax.invert_yaxis()
            ax.set_xlim(-.06, 1.06)
            ax.set_xlabel(xlabel)
            ax.set_title(title, weight="bold")
            ax.grid(axis="x", color="0.9", linewidth=.6)
        positions = np.arange(len(OPERATORS))
        for label, offset, color in [("before", -.17, "#C66C20"), ("after", .17, "#2166AC")]:
            group = dispersion[dispersion.version.eq(label)].set_index("operator").reindex(OPERATORS)
            axes[2].bar(positions + offset, group.sd_across_eight_model_means, width=.32, color=color)
        axes[2].set_xticks(positions, [name.upper() for name in OPERATORS])
        axes[2].set_title("Dispersión entre medias", weight="bold")
        axes[2].set_ylabel("DE de las ocho medias por modelo")
        axes[2].grid(axis="y", color="0.9", linewidth=.6)
        axes[2].set_axisbelow(True)
        fig.suptitle("German Credit: efecto aislado de corregir la pérdida de EC", fontsize=14)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .065), ncol=2, frameon=False)
        repetitions = int(metrics.repetitions.iloc[0])
        fig.text(.5, .022, f"Mismos modelos, particiones y predicciones. Pesos iguales. Puntos: media; barras: ±1 DE en {repetitions} semillas.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .15, 1, .94))
        fig.savefig(output / "ec_correction_comparison.png", dpi=240, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent / "results"
    parser.add_argument("--before", type=Path, default=root / "german")
    parser.add_argument("--after", type=Path, default=root / "german_ec_kl")
    args = parser.parse_args()
    before, after = args.before.resolve(), args.after.resolve()
    if before == after:
        parser.error("--before and --after must be different directories")
    try:
        audit = audit_pairing(before, after)
        metrics = paired_metrics(before, after, after)
        scores, dispersion = score_comparison(before, after, after)
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.exit(2, f"Comparison refused: {error}\n")
    manifests = {name: read_json(folder / "manifest.json") for name, folder in [("before", before), ("after", after)]}
    plot_comparison(metrics, scores, dispersion, after)
    summary = {"pairing_audit": audit,
               "before": str(before), "after": str(after),
               "source_manifests": {name: {"baseline_commit": manifest.get("baseline_commit"),
                                          "implementation_commit": manifest.get("implementation_commit"),
                                          "source_sha256": manifest["source_sha256"]}
                                    for name, manifest in manifests.items()},
               "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "test_ec_comparison": metrics[metrics.partition.eq("test")].to_dict("records"),
               "trustworthiness_comparison": scores.to_dict("records"),
               "score_dispersion": dispersion.to_dict("records"),
               "interpretation": [
                   "The trained models, predictions, Accuracy and Equal Opportunity are exactly identical between runs.",
                   "Only the EC loss and induced EC crests change; corrected aggregation is used in both versions.",
                   "Probabilistic fidelity is Bernoulli KL in nats; models without probabilities retain label disagreement as a fraction.",
                   "All aggregation weights are one third and SL alpha is 0.5; the GWA comparison has zero hesitation.",
                   "Dispersion of model means differs from within-split dispersion and uncertainty across seeds.",
                   "Changes are descriptive paired differences, not a preselected improvement or a significance test."]}
    write_json(after / "ec_correction_summary.json", summary)
    print("Exact pairing audit:", audit)
    print("Test EC before/after:")
    print(metrics[metrics.partition.eq("test")].round(6).to_string(index=False))
    print("Score dispersion:")
    print(dispersion.round(6).to_string(index=False))
    print("Saved", after / "ec_correction_summary.json")


if __name__ == "__main__":
    main()
