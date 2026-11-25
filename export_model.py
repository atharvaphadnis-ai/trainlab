"""export_model.py: Export a trained model as Keras, TensorFlow Lite, labels and config files."""

import json
import shutil

from utils import models_dir

EXPORT_FILES = ["model.keras", "model.tflite", "labels.txt", "config.json"]


def export_folder(project_id):
    folder = models_dir() / str(project_id) / "export"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def trained_model_folder(project_id):
    folder = models_dir() / str(project_id)
    if not (folder / "model.keras").exists() or not (folder / "config.json").exists():
        raise ValueError("This project has no trained model yet. Train one first.")
    return folder


def convert_to_tflite(model_path):
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    return tflite_bytes


def export_project_model(project_id):
    """Create all four export files. Returns their names and sizes."""
    source_folder = trained_model_folder(project_id)
    destination = export_folder(project_id)

    # Always rebuild the export from the current model.
    for old_file in destination.iterdir():
        old_file.unlink()

    shutil.copy2(source_folder / "model.keras", destination / "model.keras")

    config = json.loads((source_folder / "config.json").read_text(encoding="utf-8"))

    labels_text = "\n".join(config["class_names"]) + "\n"
    (destination / "labels.txt").write_text(labels_text, encoding="utf-8")

    # Copy the config so the export keeps the class order and image size.
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    tflite_path = destination / "model.tflite"
    tflite_path.write_bytes(convert_to_tflite(source_folder / "model.keras"))

    files = []
    for filename in EXPORT_FILES:
        size = (destination / filename).stat().st_size
        files.append({"filename": filename, "size_bytes": size})
    return files


def list_exported_files(project_id):
    """Show what is currently exported, without creating anything."""
    destination = export_folder(project_id)
    files = []
    for filename in EXPORT_FILES:
        path = destination / filename
        if path.exists():
            files.append({"filename": filename, "size_bytes": path.stat().st_size})
    return files


def get_export_file_path(project_id, filename):
    """Return the path of one export file, refusing anything unexpected."""
    if filename not in EXPORT_FILES:
        raise ValueError("Unknown export file.")
    path = export_folder(project_id) / filename
    if not path.exists():
        raise ValueError("This file has not been exported yet. Create the export first.")
    return path
