from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

RHO_VALUES = [0.90, 0.95, 0.99]
K_VALUES = [0.0, 0.5, 1.0]
CHANCE_LEVEL = 0.5


def select_point(
    data: pd.DataFrame,
    info: pd.Series,
    rho: float,
    k: float,
) -> tuple[int | float, float, float, float]:
    """Wählt für gegebene rho- und k-Werte den empfohlenen Ressourcenpunkt."""
    data = data.sort_values("num_train_images")
    a_max = data["image_auroc_mean"].max()
    target = CHANCE_LEVEL + rho * (a_max - CHANCE_LEVEL)

    if "one" in str(info["classification"]).lower():
        lower_bound = data["image_auroc_mean"] - k * data["image_auroc_std"].fillna(0)
        valid = data.loc[lower_bound >= target]
        selected = None if valid.empty else valid.iloc[0]
    else:
        start = info.get("search_start_point", np.nan)
        if pd.isna(start):
            start = info.get("kneedle_point", np.nan)

        search_data = (
            data.iloc[1:]
            if pd.isna(start)
            else data.loc[data["num_train_images"] >= start]
        )
        lower_bound = search_data["image_auroc_mean"] - k * search_data[
            "image_auroc_std"
        ].fillna(0)
        valid = search_data.loc[lower_bound >= target]
        selected = None if valid.empty else valid.iloc[0]

    if selected is None:
        return np.nan, np.nan, np.nan, np.nan

    n = int(selected["num_train_images"])
    mean = selected["image_auroc_mean"]
    std = selected["image_auroc_std"]
    effective_maximum = a_max - CHANCE_LEVEL

    saving = 100 * (1 - n / data["num_train_images"].max())
    mean_retention = 100 * (mean - CHANCE_LEVEL) / effective_maximum
    robust_retention = 100 * (mean - k * std - CHANCE_LEVEL) / effective_maximum
    return n, saving, mean_retention, robust_retention


def run_sensitivity_analysis(
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Wiederholt die Ressourcenpunktwahl für alle rho-/k-Kombinationen.

    Speichert die Übersicht je Parameterkombination in output_csv und die
    empfohlenen Punkte je Kategorie in sensitivity_recommendations.csv.
    """
    details = []

    for rho, k in product(RHO_VALUES, K_VALUES):
        for _, info in recommendations.iterrows():
            category = info["category"]
            data = summary.loc[summary["category"] == category]
            n, saving, mean_retention, robust_retention = select_point(
                data, info, rho, k
            )
            details.append(
                [category, rho, k, n, saving, mean_retention, robust_retention]
            )

    details = pd.DataFrame(
        details,
        columns=[
            "category",
            "rho",
            "k",
            "recommended_n",
            "saving_percent",
            "mean_retention_percent",
            "robust_retention_percent",
        ],
    )

    # Kategorie x Parameterkombination: direkt für Tabellen und Diagramme nutzbar.
    details["setting"] = details.apply(
        lambda row: f"rho_{row['rho']:.2f}_k_{row['k']:g}", axis=1
    )
    setting_columns = [
        f"rho_{rho:.2f}_k_{k:g}" for rho, k in product(RHO_VALUES, K_VALUES)
    ]
    recommendation_table = details.pivot(
        index="category", columns="setting", values="recommended_n"
    ).reindex(columns=setting_columns)
    recommendation_table = recommendation_table.reset_index()

    category_information = recommendations[
        ["category", "classification"]
    ].drop_duplicates("category")
    all_image_counts = (
        summary.groupby("category", as_index=False)["num_train_images"]
        .max()
        .rename(columns={"num_train_images": "all_num_train_images"})
    )
    recommendation_table = category_information.merge(
        all_image_counts, on="category"
    ).merge(recommendation_table, on="category")
    recommendation_table["all_num_train_images"] = recommendation_table[
        "all_num_train_images"
    ].astype(int)
    for column in setting_columns:
        recommendation_table[column] = recommendation_table[column].astype("Int64")

    baseline = details.loc[
        np.isclose(details["rho"], 0.95) & np.isclose(details["k"], 1.0)
    ].set_index("category")["recommended_n"]

    rows = []
    for (rho, k), setting in details.groupby(["rho", "k"], sort=True):
        changed = 0
        for row in setting.itertuples():
            base_n = baseline.loc[row.category]
            same = (pd.isna(row.recommended_n) and pd.isna(base_n)) or (
                pd.notna(row.recommended_n) and row.recommended_n == base_n
            )
            if not same:
                changed += 1

        rows.append(
            {
                "rho": rho,
                "k": k,
                "changed_vs_main": changed,
                "median_saving_percent": setting["saving_percent"].median(),
                "mean_saving_percent": setting["saving_percent"].mean(),
                "no_valid_points": setting["recommended_n"].isna().sum(),
            }
        )

    result = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, float_format="%.2f")
    recommendation_table.to_csv(
        output_csv.with_name("sensitivity_recommendations.csv"), index=False
    )
    return result
