import json
import os
import random
import re
import shutil
from pathlib import Path


def select_training_images(
    dataset_path: Path,
    subset_path: Path,
    category: str,
    num_images: int | str,
    seed: int,
) -> list[Path]:
    """Wählt Trainingsbilder aus und erstellt die entsprechenden Hardlinks."""

    source_train_dir = dataset_path / category / "train" / "good"
    target_train_dir = subset_path / category / "train" / "good"

    if not source_train_dir.is_dir():
        raise FileNotFoundError(
            f"Trainingsverzeichnis nicht gefunden: {source_train_dir}"
        )

    image_paths = sorted(
        path
        for path in source_train_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    total_images = len(image_paths)

    if total_images == 0:
        raise ValueError(f"Keine Trainingsbilder gefunden: {source_train_dir}")

    shuffled_images = image_paths.copy()
    random.Random(seed).shuffle(shuffled_images)

    if num_images == "all":
        selected_images = image_paths

    elif num_images == "75%":
        selected_count = int(total_images * 0.75)

        if selected_count <= 50:
            return []

        selected_images = shuffled_images[:selected_count]

    elif num_images == "50%":
        selected_count = int(total_images * 0.50)

        if selected_count <= 50:
            return []

        selected_images = shuffled_images[:selected_count]

    elif num_images == "25%":
        selected_count = int(total_images * 0.25)

        if selected_count <= 50:
            return []

        selected_images = shuffled_images[:selected_count]

    elif isinstance(num_images, int):
        if not 1 <= num_images <= total_images:
            raise ValueError(
                f"{num_images} Bilder wurden angefordert, "
                f"aber nur {total_images} Trainingsbilder sind verfügbar."
            )

        selected_images = shuffled_images[:num_images]

    else:
        raise ValueError(f"Ungültige Angabe für num_images: {num_images}")

    target_train_dir.mkdir(parents=True, exist_ok=True)

    for source_path in selected_images:
        destination_path = target_train_dir / source_path.name
        os.link(source_path, destination_path)

    return selected_images


def reset_temporary_dataset(subset_path: Path) -> None:
    """Setzt das temporäre Trainingsdataset zurück."""

    subset_path = Path(subset_path)

    if subset_path.exists():
        shutil.rmtree(subset_path)

    subset_path.mkdir(parents=True, exist_ok=True)


def save_selection_manifest(
    selected_images: list[str | Path],
    output_dir: str | Path,
    category: str,
    num_train_images: int,
    seed: int,
) -> Path:
    """Speichert die ausgewählten Trainingsbilder als JSON-Datei."""

    if len(selected_images) != num_train_images:
        raise ValueError("num_train_images stimmt nicht mit selected_images überein.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_category = re.sub(r"[^A-Za-z0-9_.-]+", "_", category)
    output_path = (
        output_dir / f"{safe_category}_n{num_train_images}_seed{seed}.json"
    )

    manifest = {
        "category": category,
        "num_train_images": num_train_images,
        "seed": seed,
        "selected_images": [Path(image_path).name for image_path in selected_images],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    return output_path
