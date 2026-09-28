import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

key_experiment = ["category", "num_train_images", "seed"]

metrics = ["image_auroc", "pixel_aupr", "memory_bank_size_mb"]

performance_metrics = {"image_auroc": "Image AUROC", "pixel_aupr": "Pixel AUPR"}

resource_metrics = {"memory_bank_size_mb": ("Größe der Memory Bank", "Speicher [MiB]")}


def save_run_result(result: dict, csv_path: str | Path) -> None:
    """Speichert das Ergebnis eines einzelnen Versuchs."""

    required_columns = key_experiment + metrics
    missing_columns = [column for column in required_columns if column not in result]
    if missing_columns:
        raise ValueError(
            "Versuchsergebnis enthält nicht alle Pflichtfelder: " f"{missing_columns}"
        )

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    result_frame = pd.DataFrame([result], columns=required_columns)
    result_frame.to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
    )


def summarize_and_save_results(
    results: pd.DataFrame, csv_path: str | Path
) -> pd.DataFrame:
    """Berechnet Mittelwert und Standardabweichung über alle Seeds."""

    required_columns = key_experiment + metrics
    missing_columns = [
        column for column in required_columns if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(f"Fehlende Ergebnisspalten: {missing_columns}")

    duplicates = results.duplicated(
        subset=key_experiment,
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Doppelte Versuchsergebnisse gefunden:\n"
            + results.loc[
                duplicates,
                key_experiment,
            ].to_string(index=False)
        )

    numeric_results = results.copy()

    numeric_results[metrics] = numeric_results[metrics].apply(
        pd.to_numeric,
        errors="raise",
    )

    rows_with_nan = numeric_results[
        numeric_results[key_experiment + metrics].isna().any(axis=1)
    ]

    if not rows_with_nan.empty:
        raise ValueError(
            "Versuchsergebnisse enthalten fehlende Werte:\n"
            + rows_with_nan.to_string(index=False)
        )

    grouped_results = numeric_results.groupby(
        ["category", "num_train_images"],
        sort=True,
    )

    performance_summary = grouped_results[["image_auroc", "pixel_aupr"]].agg(
        ["mean", "std"]
    )

    performance_summary.columns = [
        f"{metric}_{calculation}" for metric, calculation in performance_summary.columns
    ]

    performance_summary = performance_summary.reset_index()

    num_runs = grouped_results.size().rename("num_runs").reset_index()

    memory_bank = (
        grouped_results["memory_bank_size_mb"]
        .first()
        .rename("memory_bank_size_mb")
        .reset_index()
    )

    summary = performance_summary.merge(
        num_runs,
        on=["category", "num_train_images"],
        validate="one_to_one",
    ).merge(
        memory_bank,
        on=["category", "num_train_images"],
        validate="one_to_one",
    )

    summary = summary[
        [
            "category",
            "num_train_images",
            "num_runs",
            "image_auroc_mean",
            "image_auroc_std",
            "pixel_aupr_mean",
            "pixel_aupr_std",
            "memory_bank_size_mb",
        ]
    ]

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)

    return summary


def plot_result_curves(
    summary: pd.DataFrame, output_dir: str | Path
) -> list[Path]:
    """Erzeugt Leistungs- und Ressourcenkurven je Kategorie."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created_plots = []

    for category, category_results in summary.groupby("category"):

        category_results = category_results.sort_values("num_train_images")
        safe_category = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(category))
        performance_path = output_dir / f"{safe_category}_performance.png"

        if _plot_performance(
            category_results,
            str(category),
            performance_path,
        ):
            created_plots.append(performance_path)

        resources_path = output_dir / f"{safe_category}_resources.png"
        if _plot_resources(category_results, str(category), resources_path):
            created_plots.append(resources_path)

    return created_plots


def _plot_performance(results: pd.DataFrame, category: str, output_path: Path) -> bool:
    """Zeichnet Mittelwerte und Standardabweichungen der Scores."""

    metrics = [
        metric_name
        for metric_name in performance_metrics
        if f"{metric_name}_mean" in results.columns
    ]
    if not metrics:
        return False

    figure, axis = plt.subplots(figsize=(8, 5))
    x_values = results["num_train_images"]

    for metric_name in metrics:
        mean_values = results[f"{metric_name}_mean"]
        std_column = f"{metric_name}_std"
        std_values = (
            results[std_column].fillna(0)
            if std_column in results.columns
            else pd.Series(0.0, index=results.index)
        )

        line = axis.plot(
            x_values,
            mean_values,
            marker="o",
            label=performance_metrics[metric_name],
        )[0]
        axis.fill_between(
            x_values,
            (mean_values - std_values).clip(0, 1),
            (mean_values + std_values).clip(0, 1),
            color=line.get_color(),
            alpha=0.15,
        )

    axis.set(
        title=f"Erkennungsleistung – {category}",
        xlabel="Anzahl der Trainingsbilder",
        ylabel="Metrikwert",
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return True


def _plot_resources(results: pd.DataFrame, category: str, output_path: Path) -> bool:
    """Zeichnet den Speicherbedarf der Memory Bank als einzelnes Diagramm."""

    metric_name = "memory_bank_size_mb"
    # Unterstützt aktuelle und ältere Zusammenfassungen.
    value_column = (
        metric_name if metric_name in results.columns else f"{metric_name}_mean"
    )
    if value_column not in results.columns or not results[value_column].notna().any():
        return False

    results = results.sort_values("num_train_images")
    title, y_label = resource_metrics[metric_name]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        results["num_train_images"],
        results[value_column],
        marker="o",
    )
    axis.set(
        title=f"{title} – {category}",
        xlabel="Anzahl der Trainingsbilder",
        ylabel=y_label,
    )
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return True
