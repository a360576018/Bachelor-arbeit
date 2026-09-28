from pathlib import Path

from anomalib.callbacks import ModelCheckpoint
from anomalib.data import MVTecAD
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.metrics import AUPR, AUROC, Evaluator
from anomalib.models import Patchcore
from lightning import seed_everything


def run_patchcore(
    train_dataset_path: str | Path,
    test_dataset_path: str | Path,
    category: str,
    output_dir: str | Path,
    seed: int,
    num_train_images: int,
) -> dict:
    """Trainiert und bewertet PatchCore für einen einzelnen Versuch."""

    seed_everything(seed, workers=True)

    train_datamodule = MVTecAD(
        root=train_dataset_path,
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
        seed=seed,
        test_split_mode=TestSplitMode.NONE,
        val_split_mode=ValSplitMode.NONE,
    )

    test_datamodule = MVTecAD(
        root=test_dataset_path,
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
        seed=seed,
    )

    evaluator = Evaluator(
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

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=1,
        pre_processor=Patchcore.configure_pre_processor(
            image_size=(256, 256),
            center_crop_size=(224, 224),
        ),
        evaluator=evaluator,
        visualizer=False,
    )

    no_checkpoint = ModelCheckpoint(
        save_top_k=0,
        save_last=False,
    )

    engine = Engine(
        accelerator="auto",
        devices=1,
        default_root_dir=output_dir,
        limit_val_batches=0,
        callbacks=[no_checkpoint],
    )

    engine.fit(
        model=model,
        datamodule=train_datamodule,
    )

    actual_num_train_images = len(train_datamodule.train_dataloader().dataset)
    if actual_num_train_images != num_train_images:
        raise ValueError(
            f"{num_train_images} Trainingsbilder wurden erwartet, "
            f"aber {actual_num_train_images} wurden geladen."
        )

    memory_bank = model.model.memory_bank
    memory_bank_size_mb = memory_bank.numel() * memory_bank.element_size() / (1024**2)

    test_results = engine.test(
        model=model,
        datamodule=test_datamodule,
    )
    metrics = test_results[0]

    return {
        "category": category,
        "seed": seed,
        "num_train_images": actual_num_train_images,
        "image_auroc": float(metrics["image_AUROC"]),
        "pixel_aupr": float(metrics["pixel_AUPR"]),
        "memory_bank_size_mb": memory_bank_size_mb,
    }
