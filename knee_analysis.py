import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from kneed import KneeLocator
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Feste Parameter der Auswertungsmethode
ONE_SHOT_GAIN_THRESHOLD = 0.01
CHANCE_LEVEL = 0.5
PERFORMANCE_RETENTION = 0.95
STABILITY_STD_MULTIPLIER = 1.0


def _calculate_target(image_auroc_max: float) -> float:
    """Berechnet das Ziel für die erhaltene AUROC-Trennleistung."""
    return CHANCE_LEVEL + PERFORMANCE_RETENTION * (image_auroc_max - CHANCE_LEVEL)


def _calculate_lower_bound(
    mean_value: float,
    std_value: float,
) -> float:
    """Berechnet die konservative Leistungsgrenze mean - 1 sigma."""
    return mean_value - STABILITY_STD_MULTIPLIER * std_value


def _calculate_auroc_retention(
    value: float,
    reference: float,
) -> float:
    """Berechnet die erhaltene, über Zufall liegende AUROC-Leistung."""
    return 100.0 * ((value - CHANCE_LEVEL) / (reference - CHANCE_LEVEL))


def _calculate_percent_of_reference(
    value: float,
    reference: float,
) -> float:
    """Berechnet den direkten Prozentanteil an einem Referenzwert."""
    return 100.0 * value / reference


def classify_categories(summary: pd.DataFrame) -> pd.DataFrame:
    """Teilt Kategorien in One-Shot- und Knickpunktkategorien ein.

    One-Shot gilt, wenn der maximale Zugewinn gegenüber N=1 höchstens
    0,01 beträgt und die konservative Leistungsgrenze bei N=1 das
    AUROC-Ziel erreicht.
    """
    classifications = []

    for category, category_results in summary.groupby("category"):
        sorted_results = category_results.sort_values("num_train_images")
        n1_row = sorted_results.loc[sorted_results["num_train_images"] == 1].iloc[0]

        image_auroc_n1 = float(n1_row["image_auroc_mean"])
        image_auroc_n1_std = float(n1_row["image_auroc_std"])
        image_auroc_max = float(sorted_results["image_auroc_mean"].max())
        image_auroc_gain = max(
            0.0,
            image_auroc_max - image_auroc_n1,
        )
        image_auroc_target = _calculate_target(image_auroc_max)
        image_auroc_n1_lower_bound = _calculate_lower_bound(
            image_auroc_n1,
            image_auroc_n1_std,
        )

        is_one_shot = (
            image_auroc_gain <= ONE_SHOT_GAIN_THRESHOLD
            and image_auroc_n1_lower_bound >= image_auroc_target - 1e-12
        )

        classifications.append(
            {
                "category": category,
                "classification": ("one_shot" if is_one_shot else "knee_point"),
                "image_auroc_n1": image_auroc_n1,
                "image_auroc_n1_std": image_auroc_n1_std,
                "image_auroc_n1_lower_bound": (image_auroc_n1_lower_bound),
                "image_auroc_max": image_auroc_max,
                "image_auroc_gain": image_auroc_gain,
                "image_auroc_target": image_auroc_target,
            }
        )

    return pd.DataFrame(classifications)


def _select_resource_point(
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_std_values: np.ndarray,
    kneedle_point: int | None,
    image_auroc_target: float,
) -> dict[str, object]:
    """Sucht ab Kneedle den ersten stabilen realen Messpunkt."""
    if kneedle_point is None:
        search_start_index = 1
        fallback_search = True
    else:
        search_start_index = int(
            np.flatnonzero(np.isclose(x_values, float(kneedle_point)))[0]
        )
        fallback_search = False

    lower_bounds = y_values - STABILITY_STD_MULTIPLIER * y_std_values
    valid_indices = np.flatnonzero(
        lower_bounds[search_start_index:] >= image_auroc_target - 1e-12
    )

    if len(valid_indices) == 0:
        return {
            "recommended_num_train_images": None,
            "selection_status": "no_valid_point",
            "selection_direction": None,
            "recommended_image_auroc": None,
            "recommended_image_auroc_std": None,
            "recommended_image_auroc_lower_bound": None,
        }

    selected_index = search_start_index + int(valid_indices[0])

    if fallback_search:
        selection_direction = "performance_fallback"
    elif selected_index == search_start_index:
        selection_direction = "at_kneedle"
    else:
        selection_direction = "forward_to_target"

    return {
        "recommended_num_train_images": int(x_values[selected_index]),
        "selection_status": "selected",
        "selection_direction": selection_direction,
        "recommended_image_auroc": float(y_values[selected_index]),
        "recommended_image_auroc_std": float(y_std_values[selected_index]),
        "recommended_image_auroc_lower_bound": float(lower_bounds[selected_index]),
    }


def knee_detection(
    summary: pd.DataFrame,
    classifications: pd.DataFrame,
) -> pd.DataFrame:
    """Bestimmt Kneedle und den stabilen Ressourcenpunkt.

    Ab dem Kneedle-Punkt wird der erste gemessene Punkt gewählt, für den

        Image-AUROC-Mittelwert - 1 sigma >= AUROC-Ziel

    gilt. Ohne verwendbaren Kneedle beginnt die Suche beim ersten
    Messpunkt nach N=1. Die Kurve wird nicht geglättet.
    """
    result_columns = [
        "category",
        "kneedle_point",
        "recommended_num_train_images",
        "selection_status",
        "selection_direction",
        "image_auroc_target",
        "recommended_image_auroc",
        "recommended_image_auroc_std",
        "recommended_image_auroc_lower_bound",
    ]
    knee_results = []

    knee_categories = set(
        classifications.loc[
            classifications["classification"] == "knee_point",
            "category",
        ]
    )

    for category, category_results in summary.groupby("category"):
        if category not in knee_categories:
            continue

        sorted_results = category_results.sort_values("num_train_images")
        x_values = sorted_results["num_train_images"].to_numpy(dtype=float)
        y_values = sorted_results["image_auroc_mean"].to_numpy(dtype=float)
        y_std_values = sorted_results["image_auroc_std"].to_numpy(dtype=float)

        image_auroc_target = _calculate_target(float(y_values.max()))

        kneedle = KneeLocator(
            x=x_values,
            y=y_values,
            S=1.0,
            curve="concave",
            direction="increasing",
            interp_method="interp1d",
            online=False,
        )

        if kneedle.knee is None:
            kneedle_point = None
        else:
            kneedle_index = int(
                np.flatnonzero(
                    np.isclose(
                        x_values,
                        float(kneedle.knee),
                    )
                )[0]
            )
            kneedle_point = None if kneedle_index == 0 else int(x_values[kneedle_index])

        selection = _select_resource_point(
            x_values=x_values,
            y_values=y_values,
            y_std_values=y_std_values,
            kneedle_point=kneedle_point,
            image_auroc_target=image_auroc_target,
        )

        knee_results.append(
            {
                "category": category,
                "kneedle_point": kneedle_point,
                "image_auroc_target": image_auroc_target,
                **selection,
            }
        )

    return pd.DataFrame(knee_results, columns=result_columns)


def create_recommendations(
    summary: pd.DataFrame,
    classifications: pd.DataFrame,
    knee_results: pd.DataFrame,
    csv_path: str | Path,
) -> pd.DataFrame:
    """Speichert Empfehlungen und Vergleiche mit dem All-Modell."""
    analysis_columns = [
        "kneedle_point",
        "recommended_num_train_images",
        "selection_status",
        "selection_direction",
        "image_auroc_target",
        "recommended_image_auroc",
        "recommended_image_auroc_std",
        "recommended_image_auroc_lower_bound",
    ]
    recommendations = []

    for _, classification_row in classifications.iterrows():
        category = classification_row["category"]
        classification = classification_row["classification"]
        category_results = summary.loc[summary["category"] == category].sort_values(
            "num_train_images"
        )

        all_row = category_results.iloc[-1]
        all_num_train_images = int(all_row["num_train_images"])

        if classification == "one_shot":
            n1_row = category_results.loc[
                category_results["num_train_images"] == 1
            ].iloc[0]
            analysis_values = {
                "kneedle_point": None,
                "recommended_num_train_images": 1,
                "selection_status": "one_shot",
                "selection_direction": None,
                "image_auroc_target": float(classification_row["image_auroc_target"]),
                "recommended_image_auroc": float(n1_row["image_auroc_mean"]),
                "recommended_image_auroc_std": float(n1_row["image_auroc_std"]),
                "recommended_image_auroc_lower_bound": (
                    _calculate_lower_bound(
                        float(n1_row["image_auroc_mean"]),
                        float(n1_row["image_auroc_std"]),
                    )
                ),
            }
        else:
            knee_row = knee_results.loc[knee_results["category"] == category].iloc[0]
            analysis_values = {column: knee_row[column] for column in analysis_columns}

        recommended_value = analysis_values["recommended_num_train_images"]
        recommended_num_images = (
            None if pd.isna(recommended_value) else int(recommended_value)
        )

        if recommended_num_images is None:
            recommended_row = None
        else:
            recommended_row = category_results.loc[
                category_results["num_train_images"] == recommended_num_images
            ].iloc[0]

        recommended_image_auroc = analysis_values["recommended_image_auroc"]
        recommended_image_auroc_std = analysis_values["recommended_image_auroc_std"]
        recommended_image_auroc_lower_bound = analysis_values[
            "recommended_image_auroc_lower_bound"
        ]
        image_auroc_max = float(classification_row["image_auroc_max"])

        if recommended_row is None:
            image_auroc_robust_retention = np.nan
            pixel_aupr_mean = np.nan
            pixel_aupr_std = np.nan
            pixel_aupr_retention = np.nan
            memory_bank_size_mb = np.nan
            memory_bank_reduction = np.nan
        else:
            image_auroc_robust_retention = _calculate_auroc_retention(
                float(recommended_image_auroc_lower_bound),
                image_auroc_max,
            )

            pixel_aupr_mean = float(recommended_row["pixel_aupr_mean"])
            pixel_aupr_std = float(recommended_row["pixel_aupr_std"])
            pixel_aupr_all = float(all_row["pixel_aupr_mean"])
            pixel_aupr_retention = _calculate_percent_of_reference(
                pixel_aupr_mean,
                pixel_aupr_all,
            )

            memory_bank_size_mb = float(recommended_row["memory_bank_size_mb"])
            memory_bank_all_mb = float(all_row["memory_bank_size_mb"])
            memory_bank_reduction = 100.0 - _calculate_percent_of_reference(
                memory_bank_size_mb,
                memory_bank_all_mb,
            )

        training_images_percent_of_all = (
            np.nan
            if recommended_num_images is None
            else 100.0 * recommended_num_images / all_num_train_images
        )

        recommendations.append(
            {
                "category": category,
                "classification": classification,
                **analysis_values,
                "all_num_train_images": all_num_train_images,
                "training_images_percent_of_all": (training_images_percent_of_all),
                "training_images_reduction_percent": (
                    100.0 - training_images_percent_of_all
                ),
                "image_auroc_n1": float(classification_row["image_auroc_n1"]),
                "image_auroc_n1_std": float(classification_row["image_auroc_n1_std"]),
                "image_auroc_n1_lower_bound": float(
                    classification_row["image_auroc_n1_lower_bound"]
                ),
                "image_auroc_max": float(classification_row["image_auroc_max"]),
                "image_auroc_gain": float(classification_row["image_auroc_gain"]),
                "chance_level": CHANCE_LEVEL,
                "performance_retention": PERFORMANCE_RETENTION,
                "stability_margin_std_multiplier": (STABILITY_STD_MULTIPLIER),
                "image_auroc_robust_retention_vs_max_percent": (
                    image_auroc_robust_retention
                ),
                "pixel_aupr_mean": pixel_aupr_mean,
                "pixel_aupr_std": pixel_aupr_std,
                "pixel_aupr_retention_vs_all_percent": (pixel_aupr_retention),
                "memory_bank_size_mb": memory_bank_size_mb,
                "memory_bank_reduction_percent": (memory_bank_reduction),
            }
        )

    results = pd.DataFrame(recommendations)

    for column in [
        "recommended_num_train_images",
        "all_num_train_images",
        "kneedle_point",
    ]:
        results[column] = pd.array(
            results[column],
            dtype="Int64",
        )

    output_columns = [
        "category",
        "classification",
        "kneedle_point",
        "recommended_num_train_images",
        "all_num_train_images",
        "training_images_reduction_percent",
        "chance_level",
        "performance_retention",
        "stability_margin_std_multiplier",
        "image_auroc_target",
        "image_auroc_max",
        "recommended_image_auroc",
        "recommended_image_auroc_std",
        "recommended_image_auroc_lower_bound",
        "image_auroc_robust_retention_vs_max_percent",
        "pixel_aupr_mean",
        "pixel_aupr_std",
        "pixel_aupr_retention_vs_all_percent",
        "memory_bank_size_mb",
        "memory_bank_reduction_percent",
    ]

    results = results[output_columns]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    return results


def plot_knee_analysis(
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Zeigt Messkurve, Seed-Streuung, Ziel und Empfehlung."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created_plots = []

    for category, category_results in summary.groupby("category"):
        recommendation = recommendations.loc[
            recommendations["category"] == category
        ].iloc[0]

        if recommendation["classification"] == "one_shot":
            continue

        sorted_results = category_results.sort_values("num_train_images")
        x_values = sorted_results["num_train_images"].to_numpy(dtype=float)
        y_values = sorted_results["image_auroc_mean"].to_numpy(dtype=float)
        y_std_values = sorted_results["image_auroc_std"].to_numpy(dtype=float)

        figure, axis = plt.subplots(figsize=(9, 5.5))
        axis.plot(
            x_values,
            y_values,
            color="tab:blue",
            marker="o",
            label="Image-AUROC-Mittelwert",
        )
        axis.fill_between(
            x_values,
            np.clip(y_values - y_std_values, 0, 1),
            np.clip(y_values + y_std_values, 0, 1),
            color="tab:blue",
            alpha=0.15,
            label="Seed-Standardabweichung (±1σ)",
        )

        image_auroc_target = float(recommendation["image_auroc_target"])
        axis.axhline(
            image_auroc_target,
            color="tab:green",
            linestyle="--",
            linewidth=1.6,
            label="95 % der maximalen Trennleistung",
        )

        kneedle_point = recommendation["kneedle_point"]
        recommended_point = recommendation["recommended_num_train_images"]
        same_point = (
            not pd.isna(kneedle_point)
            and not pd.isna(recommended_point)
            and np.isclose(kneedle_point, recommended_point)
        )

        if not pd.isna(kneedle_point) and not same_point:
            kneedle_index = int(
                np.flatnonzero(
                    np.isclose(
                        x_values,
                        float(kneedle_point),
                    )
                )[0]
            )
            axis.scatter(
                x_values[kneedle_index],
                y_values[kneedle_index],
                marker="^",
                color="tab:orange",
                s=90,
                edgecolor="black",
                linewidth=0.7,
                label="Kneedle",
                zorder=5,
            )

        if not pd.isna(recommended_point):
            recommended_index = int(
                np.flatnonzero(
                    np.isclose(
                        x_values,
                        float(recommended_point),
                    )
                )[0]
            )
            axis.scatter(
                x_values[recommended_index],
                y_values[recommended_index],
                marker="*",
                color="tab:red",
                s=190,
                edgecolor="black",
                linewidth=0.7,
                label=(
                    "Kneedle = Empfehlung"
                    if same_point
                    else "Empfohlener Ressourcenpunkt"
                ),
                zorder=6,
            )
            axis.axvline(
                float(recommended_point),
                color="tab:red",
                linestyle=":",
                alpha=0.65,
            )
            axis.text(
                0.98,
                0.05,
                (
                    f"AUROC-Ziel: {image_auroc_target:.4f}\n"
                    "Stabilitätsgrenze (μ − 1σ): "
                    f"{float(recommendation['recommended_image_auroc_lower_bound']):.4f}"
                ),
                transform=axis.transAxes,
                horizontalalignment="right",
                verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.85},
            )
        else:
            axis.text(
                0.98,
                0.05,
                "Kein gültiger Empfehlungspunkt",
                transform=axis.transAxes,
                horizontalalignment="right",
                verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.85},
            )

        lower_limit = max(
            0.0,
            float(np.min(y_values - y_std_values)) - 0.03,
        )
        axis.set_ylim(lower_limit, 1.01)
        axis.set(
            title=f"Kneedle- und Stabilitätsanalyse – {category}",
            xlabel="Anzahl der Trainingsbilder",
            ylabel="Image AUROC",
        )
        axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()
        figure.tight_layout()

        safe_category = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(category),
        )
        output_path = output_dir / f"{safe_category}_knee_analysis.png"
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)
        created_plots.append(output_path)

    return created_plots
