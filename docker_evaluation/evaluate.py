import argparse
import gc
import logging
import os
import threading
from pathlib import Path
from time import perf_counter

import pandas as pd
import psutil
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUPR, AUROC, Evaluator
from anomalib.models import Patchcore
from lightning import seed_everything
from lightning.pytorch.callbacks import Callback

MIB = 1024**2
RESULT_COLUMNS = [
    "device_name",
    "model_name",
    "category",
    "setting",
    "image_auroc",
    "pixel_aupr",
    "avg_inference_time_ms",
    "peak_ram_mb",
    "peak_vram_mb",
    "num_test_images",
]


def build_evaluator() -> Evaluator:
    """Erstellt den Evaluator mit Bild-AUROC und Pixel-AUPR."""
    return Evaluator(
        test_metrics=[
            AUROC(
                fields=["pred_score", "gt_label"],
                prefix="image_",
                normalization=None,
            ),
            AUPR(
                fields=["anomaly_map", "gt_mask"],
                prefix="pixel_",
                normalization=None,
            ),
        ]
    )


class InferenceTimer(Callback):
    """Misst die Modellzeit je Bild; die Datenladezeit ist nicht enthalten."""

    def __init__(self, use_cuda: bool) -> None:
        self.use_cuda = use_cuda
        self.times_ms: list[float] = []
        self.start_time = 0.0

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if self.use_cuda:
            torch.cuda.synchronize()
        self.start_time = perf_counter()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if self.use_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - self.start_time) * 1000
        image = getattr(batch, "image", None)
        if image is None and isinstance(batch, dict):
            image = batch.get("image")
        batch_size = int(image.shape[0]) if image is not None else 1
        self.times_ms.extend([elapsed_ms / batch_size] * batch_size)


class PeakRamMonitor:
    """Überwacht im Hintergrund den maximalen RAM-Verbrauch des Prozesses."""
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(0.01):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def start(self) -> None:
        self.peak_bytes = self.process.memory_info().rss
        self.thread.start()

    def stop(self) -> float:
        self.stop_event.set()
        self.thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        return self.peak_bytes / MIB


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_checkpoint(
    row: pd.Series,
    checkpoint_path: Path,
    dataset_path: Path,
    device_name: str,
    accelerator: str,
    warmup_batches: int,
) -> dict:
    """Bewertet einen Checkpoint und misst Inferenzzeit und Speicherbedarf."""
    category = str(row["category"])
    seed = int(row["seed"])
    use_cuda = accelerator != "cpu" and torch.cuda.is_available()

    seed_everything(seed, workers=True)
    clear_memory()

    model = Patchcore.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        map_location="cpu",
        weights_only=False,  # Nur für selbst erzeugte Checkpoints.
        pre_trained=False,  # Verhindert einen Backbone-Download.
        evaluator=build_evaluator(),
        visualizer=False,
    )
    datamodule = MVTecAD(
        root=dataset_path,
        category=category,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        seed=seed,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    num_test_images = len(datamodule.test_dataloader().dataset)
    if num_test_images == 0:
        raise RuntimeError(
            f"Keine Testbilder für {category}. Prüfe Pfad, ground_truth und pandas==2.2.3."
        )

    timer = InferenceTimer(use_cuda)
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        callbacks=[timer],
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir="/tmp/anomalib_evaluation",
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    ram_monitor = PeakRamMonitor()
    ram_monitor.start()
    try:
        test_results = engine.test(model=model, datamodule=datamodule)
        if use_cuda:
            torch.cuda.synchronize()
    finally:
        peak_ram_mb = ram_monitor.stop()

    if len(timer.times_ms) != num_test_images:
        raise RuntimeError(
            f"{num_test_images} Bilder, aber {len(timer.times_ms)} Zeiten gemessen."
        )

    skipped = min(warmup_batches, max(0, num_test_images - 1))
    measured_times = timer.times_ms[skipped:]
    metrics = test_results[0]
    result = {
        "device_name": device_name,
        "model_name": str(row["model_name"]),
        "category": category,
        "setting": str(row["setting"]),
        "image_auroc": float(metrics["image_AUROC"]),
        "pixel_aupr": float(metrics["pixel_AUPR"]),
        "avg_inference_time_ms": sum(measured_times) / len(measured_times),
        "peak_ram_mb": peak_ram_mb,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / MIB if use_cuda else None),
        "num_test_images": num_test_images,
    }

    del engine, datamodule, model, test_results
    clear_memory()
    return result


def main() -> None:
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning.fabric").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="PatchCore-Hardwarebenchmark")
    parser.add_argument("--dataset-path", type=Path, default=Path("/data"))
    parser.add_argument("--models-dir", type=Path, default=Path("/models"))
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/models/export_summary.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/output/hardware_evaluation.csv"),
    )
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=["recommended", "all"],
        default=["recommended", "all"],
    )
    parser.add_argument("--warmup-batches", type=int, default=5)
    args = parser.parse_args()

    if not args.dataset_path.is_dir():
        raise FileNotFoundError(f"Datensatz fehlt: {args.dataset_path}")
    if not args.models_dir.is_dir():
        raise FileNotFoundError(f"Modellverzeichnis fehlt: {args.models_dir}")
    if not args.summary_csv.is_file():
        raise FileNotFoundError(f"export_summary.csv fehlt: {args.summary_csv}")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU gewählt, aber CUDA ist nicht verfügbar.")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches darf nicht negativ sein.")

    summary = pd.read_csv(args.summary_csv)
    required = {
        "model_name",
        "category",
        "setting",
        "num_train_images",
        "seed",
        "checkpoint_file",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Fehlende CSV-Spalten: {sorted(missing)}")

    summary = summary[summary["setting"].isin(args.settings)]
    if args.categories:
        summary = summary[summary["category"].isin(args.categories)]
    if summary.empty:
        raise ValueError("Keine Modelle entsprechen den Filtern.")

    completed = set()
    if args.output_csv.is_file():
        previous = pd.read_csv(args.output_csv)
        if list(previous.columns) != RESULT_COLUMNS:
            raise ValueError(
                f"Altes CSV-Format. Datei zuerst umbenennen: {args.output_csv}"
            )
        completed = set(
            previous.loc[previous["device_name"] == args.device_name, "model_name"]
        )

    use_cuda = args.accelerator != "cpu" and torch.cuda.is_available()
    if use_cuda:  # CUDA vor der Messung einmal initialisieren.
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(summary)} Modelle | Gerät={args.device_name} | "
        f"Beschleuniger={'GPU' if use_cuda else 'CPU'}"
    )

    for index, (_, row) in enumerate(summary.iterrows(), start=1):
        model_name = str(row["model_name"])
        if model_name in completed:
            print(f"[{index}/{len(summary)}] übersprungen: {model_name}")
            continue

        checkpoint_name = str(row["checkpoint_file"])
        if Path(checkpoint_name).name != checkpoint_name:
            raise ValueError(f"Ungültiger Checkpointname: {checkpoint_name}")
        checkpoint_path = args.models_dir / checkpoint_name
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint fehlt: {checkpoint_path}")

        print(f"[{index}/{len(summary)}] {model_name}")
        result = evaluate_checkpoint(
            row,
            checkpoint_path,
            args.dataset_path,
            args.device_name,
            args.accelerator,
            args.warmup_batches,
        )
        pd.DataFrame([result], columns=RESULT_COLUMNS).to_csv(
            args.output_csv,
            mode="a",
            header=not args.output_csv.exists(),
            index=False,
        )

        vram = result["peak_vram_mb"]
        vram_text = f"{vram:.1f} MB" if vram is not None else "-"
        print(
            f"  AUROC={result['image_auroc']:.4f} | "
            f"Pixel-AUPR={result['pixel_aupr']:.4f} | "
            f"{result['avg_inference_time_ms']:.2f} ms/Bild | "
            f"RAM={result['peak_ram_mb']:.1f} MB | VRAM={vram_text}"
        )

    print(f"Fertig: {args.output_csv}")


if __name__ == "__main__":
    main()
