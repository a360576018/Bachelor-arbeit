import datetime
from pathlib import Path

import pandas as pd

import data_sampler
import knee_analysis
import patchcore_runner
import result_process
import sensitivity_analysis

# Pfade
project_path = Path(__file__).resolve().parent

dataset_path = Path(r"C:/BA/mvtec_anomaly_detection")
subset_path = project_path / "temp_dataset"

run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
results_path = project_path / "results" / run_name
anomalib_path = results_path / "_anomalib"
raw_results_path = results_path / "results_raw.csv"
mean_results_path = results_path / "results_mean.csv"
manifest_path = results_path / "selection_manifests"
plots_path = results_path / "plots"
recommendations_path = results_path / "recommendations.csv"
knee_plots_path = results_path / "knee_plots"


# Versuchskonfiguration
category_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

num_train_images = [
    1,
    2,
    5,
    10,
    15,
    20,
    30,
    50,
    "25%",
    "50%",
    "75%",
    "all",
]


def main() -> None:
    print(
        "Hauptkonfiguration: "
        f"rho={knee_analysis.PERFORMANCE_RETENTION}, "
        f"k={knee_analysis.STABILITY_STD_MULTIPLIER}"
    )

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Datensatzverzeichnis nicht gefunden: {dataset_path}")

    results_path.mkdir(parents=True, exist_ok=False)
    subset_path.mkdir(parents=True, exist_ok=True)
    manifest_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    total_runs = len(category_list) * len(num_train_images) * len(seed_list)
    current_run = 0

    data_sampler.reset_temporary_dataset(
        subset_path=subset_path,
    )

    try:
        for category in category_list:
            for num_images in num_train_images:
                for seed in seed_list:
                    current_run += 1

                    print(
                        f"\nVersuch {current_run}/{total_runs}: "
                        f"Kategorie={category}, "
                        f"N={num_images}, "
                        f"Seed={seed}"
                    )

                    selected_images = data_sampler.select_training_images(
                        dataset_path=dataset_path,
                        subset_path=subset_path,
                        category=category,
                        num_images=num_images,
                        seed=seed,
                    )

                    if not selected_images:
                        print(
                            f"Übersprungen: {num_images} von {category} "
                            f"entspricht höchstens 50 Bildern."
                        )
                        continue

                    actual_num_train_images = len(selected_images)

                    manifest_file = data_sampler.save_selection_manifest(
                        selected_images=selected_images,
                        output_dir=manifest_path,
                        category=category,
                        num_train_images=actual_num_train_images,
                        seed=seed,
                    )

                    print(f"{actual_num_train_images} Trainingsbilder ausgewählt.")
                    print(f"Auswahl gespeichert: {manifest_file}")

                    experiment_name = (
                        f"{category}_n{actual_num_train_images}_seed{seed}"
                    )

                    run_result = patchcore_runner.run_patchcore(
                        train_dataset_path=subset_path,
                        test_dataset_path=dataset_path,
                        category=category,
                        output_dir=anomalib_path / experiment_name,
                        seed=seed,
                        num_train_images=actual_num_train_images,
                    )

                    result_process.save_run_result(
                        result=run_result,
                        csv_path=raw_results_path,
                    )

                    # Erst nach vollständig gespeichertem Ergebnis
                    # wird das temporäre Dataset entfernt.
                    data_sampler.reset_temporary_dataset(
                        subset_path=subset_path,
                    )

    finally:
        # Wird auch bei Fehlern oder manuellem Abbruch ausgeführt.
        data_sampler.reset_temporary_dataset(
            subset_path=subset_path,
        )

    if not raw_results_path.is_file():
        raise FileNotFoundError(f"Ergebnisdatei nicht gefunden: {raw_results_path}")

    results = pd.read_csv(raw_results_path)

    summary = result_process.summarize_and_save_results(
        results=results,
        csv_path=mean_results_path,
    )

    created_plots = result_process.plot_result_curves(
        summary=summary,
        output_dir=plots_path,
    )

    classifications = knee_analysis.classify_categories(summary)

    knee_results = knee_analysis.knee_detection(
        summary=summary,
        classifications=classifications,
    )

    recommendations = knee_analysis.create_recommendations(
        summary=summary,
        classifications=classifications,
        knee_results=knee_results,
        csv_path=recommendations_path,
    )

    created_knee_plots = knee_analysis.plot_knee_analysis(
        summary=summary,
        recommendations=recommendations,
        output_dir=knee_plots_path,
    )

    sensitivity_analysis.run_sensitivity_analysis(
        summary=summary,
        recommendations=recommendations,
        output_csv=results_path / "sensitivity_analysis.csv",
    )

    print("\nAlle Versuche wurden abgeschlossen.")
    print(f"Einzelergebnisse: {raw_results_path}")
    print(f"Zusammenfassung: {mean_results_path}")
    print(f"Erstellte Diagramme: {len(created_plots) + len(created_knee_plots)}")


if __name__ == "__main__":
    main()
