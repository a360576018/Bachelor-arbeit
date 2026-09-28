import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from anomalib.data import MVTecAD
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.metrics import AUPR, AUROC, Evaluator
from anomalib.models import Patchcore
from lightning import seed_everything

import data_sampler

settings = ("recommended", "all")
project_path = Path(__file__).resolve().parent


def build_evaluator() -> Evaluator:
    """Erstellt den Evaluator mit denselben Metriken wie im Experiment."""

    return Evaluator(
        test_metrics=[
            AUROC(
                fields=["pred_score", "gt_label"], prefix="image_", normalization=None
            ),
            AUPR(
                fields=["anomaly_map", "gt_mask"], prefix="pixel_", normalization=None
            ),
        ]
    )


def build_model() -> Patchcore:
    """Erstellt ein PatchCore-Modell mit denselben Einstellungen wie im Experiment."""

    return Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=1,
        pre_processor=Patchcore.configure_pre_processor(
            image_size=(256, 256),
            center_crop_size=(224, 224),
        ),
        evaluator=build_evaluator(),
        visualizer=False,
    )


def clear_cuda_cache() -> None:
    """Gibt nicht mehr referenzierte Python- und CUDA-Objekte frei."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_and_export(
    category: str,
    setting: str,
    num_images: int | str,
    dataset_path: Path,
    subset_path: Path,
    output_dir: Path,
    seed: int,
    run_sanity_test: bool,
) -> dict:
    """Trainiert ein Modell für eine (Kategorie, Setting)-Kombination und speichert es."""

    print(f"\n=== {category} / {setting} (n={num_images}, seed={seed}) ===")

    train_datamodule = None
    model = None
    engine = None
    memory_bank = None
    test_datamodule = None
    loaded_model = None
    test_engine = None
    test_results = None
    metrics = None

    data_sampler.reset_temporary_dataset(subset_path=subset_path)
    try:
        selected_images = data_sampler.select_training_images(
            dataset_path=dataset_path,
            subset_path=subset_path,
            category=category,
            num_images=num_images,
            seed=seed,
        )
        if not selected_images:
            raise ValueError(
                f"Keine Trainingsbilder ausgewählt für {category}/{setting} (n={num_images})."
            )
        actual_num_train_images = len(selected_images)

        seed_everything(seed, workers=True)

        train_datamodule = MVTecAD(
            root=subset_path,
            category=category,
            train_batch_size=16,
            eval_batch_size=1,
            num_workers=0,
            seed=seed,
            test_split_mode=TestSplitMode.NONE,
            val_split_mode=ValSplitMode.NONE,
        )

        model = build_model()

        model_name = (
            f"{category}_{setting}" f"_n{actual_num_train_images}_seed{seed}"
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = output_dir / f"{model_name}.ckpt"

        engine = Engine(
            accelerator="auto",
            devices=1,
            default_root_dir=output_dir / "_logs" / model_name,
            limit_val_batches=0,
            logger=False,
            enable_checkpointing=True,
        )

        engine.fit(
            model=model,
            datamodule=train_datamodule,
        )

        actual_loaded = len(train_datamodule.train_dataloader().dataset)
        if actual_loaded != actual_num_train_images:
            raise ValueError(
                f"{actual_num_train_images} Trainingsbilder erwartet, "
                f"aber {actual_loaded} geladen."
            )

        engine.trainer.save_checkpoint(str(ckpt_path))
        if not ckpt_path.is_file():
            raise RuntimeError(f"Checkpoint wurde nicht gespeichert: {ckpt_path}")

        memory_bank = model.model.memory_bank
        memory_bank_size_mb = (
            memory_bank.numel() * memory_bank.element_size() / (1024**2)
        )
        checkpoint_size_mb = ckpt_path.stat().st_size / (1024**2)

        memory_bank = None
        engine = None
        model = None
        train_datamodule = None
        clear_cuda_cache()

        sanity_metrics: dict = {}
        if run_sanity_test:
            test_datamodule = MVTecAD(
                root=dataset_path,
                category=category,
                train_batch_size=16,
                eval_batch_size=1,
                num_workers=0,
                seed=seed,
            )
            loaded_model = Patchcore.load_from_checkpoint(
                checkpoint_path=str(ckpt_path),
                map_location="cpu",
                weights_only=False,
                evaluator=build_evaluator(),
                visualizer=False,
            )

            test_engine = Engine(
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=True,
            )

            test_results = test_engine.test(
                model=loaded_model,
                datamodule=test_datamodule,
            )
            metrics = test_results[0]
            sanity_metrics = {
                "image_auroc": float(metrics["image_AUROC"]),
                "pixel_aupr": float(metrics["pixel_AUPR"]),
            }
            print(
                f"Sanity-Check (dieselbe Maschine, nur zur Kontrolle): {sanity_metrics}"
            )

        manifest = {
            "model_name": model_name,
            "category": category,
            "setting": setting,
            "num_train_images": actual_num_train_images,
            "seed": seed,
            "checkpoint_file": ckpt_path.name,
            "checkpoint_size_mb": checkpoint_size_mb,
            "memory_bank_size_mb": memory_bank_size_mb,
            "sanity_image_auroc": sanity_metrics.get("image_auroc"),
            "sanity_pixel_aupr": sanity_metrics.get("pixel_aupr"),
            "selected_images": [path.name for path in selected_images],
        }

        print(f"Gespeichert: {ckpt_path}")
        return manifest

    finally:
        data_sampler.reset_temporary_dataset(subset_path=subset_path)
        # Zuweisung von None funktioniert auch dann sicher, wenn vorher eine
        # Ausnahme aufgetreten ist, und löst alle starken Referenzen.
        train_datamodule = None
        model = None
        engine = None
        memory_bank = None
        test_datamodule = None
        loaded_model = None
        test_engine = None
        test_results = None
        metrics = None
        clear_cuda_cache()


def save_export_summary(manifests: list[dict], csv_path: Path) -> None:
    """Speichert den Modellindex nach jedem erfolgreich exportierten Modell."""
    summary_rows = []
    for manifest in manifests:
        row = manifest.copy()
        row["selected_images"] = json.dumps(
            row["selected_images"],
            ensure_ascii=False,
        )
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        csv_path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trainiert und exportiert PatchCore-Modelle am empfohlenen "
            "Ressourcenpunkt und mit vollem Trainingsset."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(r"C:/BA/mvtec_anomaly_detection"),
    )
    parser.add_argument(
        "--recommendations-csv",
        type=Path,
        default=project_path / "results" / "final" / "recommendations.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_path / "exported_models"
    )
    parser.add_argument(
        "--subset-path", type=Path, default=project_path / "temp_dataset_export"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=settings,
        default=list(settings),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--skip-sanity-test",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.dataset_path.is_dir():
        raise FileNotFoundError(
            f"Datensatzverzeichnis nicht gefunden: {args.dataset_path}"
        )

    if args.recommendations_csv is None or not args.recommendations_csv.is_file():
        raise FileNotFoundError(
            f"recommendations.csv nicht gefunden: {args.recommendations_csv} "
            "(Standard erwartet die Datei unter results/final/recommendations.csv, "
            "oder Pfad explizit per --recommendations-csv angeben)."
        )

    print(f"Verwende recommendations.csv: {args.recommendations_csv}")
    recommendations = pd.read_csv(args.recommendations_csv)

    required_columns = {"category", "recommended_num_train_images"}
    missing_columns = required_columns.difference(recommendations.columns)
    if missing_columns:
        raise ValueError(
            f"Fehlende Spalten in recommendations.csv: {sorted(missing_columns)}"
        )
    if recommendations["category"].duplicated().any():
        duplicates = recommendations.loc[
            recommendations["category"].duplicated(keep=False),
            "category",
        ].tolist()
        raise ValueError(f"Doppelte Kategorien gefunden: {duplicates}")

    categories = args.categories or recommendations["category"].tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    summary_path = args.output_dir / "export_summary.csv"

    for category in categories:
        row = recommendations.loc[recommendations["category"] == category]
        if row.empty:
            print(f"Übersprungen: {category} nicht in recommendations.csv gefunden.")
            continue
        row = row.iloc[0]

        setting_to_n = {
            "recommended": row.get("recommended_num_train_images"),
            "all": "all",
        }

        for setting in args.settings:
            num_images = setting_to_n[setting]
            if setting == "recommended" and pd.isna(num_images):
                print(
                    f"Übersprungen: {category}/recommended hat keinen "
                    "gültigen Empfehlungspunkt (no_valid_point)."
                )
                continue
            if setting == "recommended":
                num_images = int(num_images)

            manifest = train_and_export(
                category=category,
                setting=setting,
                num_images=num_images,
                dataset_path=args.dataset_path,
                subset_path=args.subset_path,
                output_dir=args.output_dir,
                seed=args.seed,
                run_sanity_test=not args.skip_sanity_test,
            )
            manifests.append(manifest)
            save_export_summary(
                manifests=manifests,
                csv_path=summary_path,
            )

    print(f"\nAlle Modelle exportiert ({len(manifests)}). Übersicht: {summary_path}")
    print(f"Verzeichnis für evaluate.py / Docker: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
