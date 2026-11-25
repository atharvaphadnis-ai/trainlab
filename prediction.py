"""Using a trained model: single image predictions and full testing."""

import io
import json

import numpy as np
from PIL import Image, ImageOps

import database
import dataset
from utils import load_settings, models_dir

# Trained models stay loaded in memory so repeated predictions are fast.
_loaded_models = {}


def clear_cached_model(project_id):
    _loaded_models.pop(project_id, None)


def load_model_for_project(project_id):
    """Load model.keras and config.json for a project (cached after first use)."""
    if project_id in _loaded_models:
        return _loaded_models[project_id]

    model_folder = models_dir() / str(project_id)
    model_file = model_folder / "model.keras"
    config_file = model_folder / "config.json"
    if not model_file.exists() or not config_file.exists():
        raise ValueError("This project has no trained model yet. Train one first.")

    import tensorflow as tf

    config = json.loads(config_file.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(model_file)
    entry = {"model": model, "config": config}
    _loaded_models[project_id] = entry
    return entry


def prepare_image_bytes(file_bytes, image_size):
    """Turn uploaded bytes into the (1, size, size, 3) array the model expects."""
    if len(file_bytes) == 0:
        raise ValueError("The file is empty.")
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    except Exception:
        raise ValueError("This file is not a readable image.")
    pixels = np.asarray(image, dtype=np.uint8)
    return np.expand_dims(pixels, axis=0)


def predict_image_bytes(project_id, file_bytes):
    """Predict one image and return the class plus the top three guesses."""
    entry = load_model_for_project(project_id)
    model = entry["model"]
    config = entry["config"]
    class_names = config["class_names"]
    image_size = int(config["image_size"])

    input_array = prepare_image_bytes(file_bytes, image_size)
    probabilities = model.predict(input_array, verbose=0)[0]

    # Class indexes sorted from most to least confident.
    ordered_indexes = np.argsort(probabilities)[::-1]
    top_predictions = []
    for index in ordered_indexes[:3]:
        top_predictions.append(
            {
                "class_name": class_names[int(index)],
                "confidence": round(float(probabilities[int(index)]) * 100, 1),
            }
        )

    settings = load_settings()
    threshold_percent = float(settings["confidence_threshold"]) * 100
    best_guess = top_predictions[0]
    unsure = best_guess["confidence"] < threshold_percent

    return {
        "predicted_class": best_guess["class_name"],
        "confidence": best_guess["confidence"],
        "unsure": unsure,
        "unsure_threshold": threshold_percent,
        "top_predictions": top_predictions,
        "message": "Confidence shows how certain the model feels, not whether it is correct.",
    }


def precision_recall_f1(confusion_matrix, class_index):
    """Precision, recall and F1 for one class, computed from the confusion matrix."""
    true_positive = confusion_matrix[class_index][class_index]
    predicted_as_this = sum(row[class_index] for row in confusion_matrix)
    actually_this = sum(confusion_matrix[class_index])

    precision = true_positive / predicted_as_this if predicted_as_this > 0 else 0.0
    recall = true_positive / actually_this if actually_this > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def load_all_images_with_labels(project_id, image_size):
    """Read every image of the project and label it with its class index."""
    images = []
    labels = []
    skipped_files = 0
    for label_index, one_class in enumerate(database.list_classes(project_id)):
        for image_row in database.list_images(one_class["id"]):
            try:
                pixels = dataset.load_image_array(dataset.image_path(image_row), image_size)
            except Exception:
                skipped_files += 1  # file deleted or damaged, skip it
                continue
            images.append(pixels)
            labels.append(label_index)
    return images, labels, skipped_files


def evaluate_model(project_id):
    """Run the model on every image in the project and compute real metrics."""
    entry = load_model_for_project(project_id)
    model = entry["model"]
    config = entry["config"]
    class_names = config["class_names"]
    num_classes = len(class_names)
    image_size = int(config["image_size"])

    images, labels, skipped_files = load_all_images_with_labels(project_id, image_size)
    total = len(labels)
    if total == 0:
        raise ValueError("This project has no images to evaluate.")

    all_images = np.stack(images)
    true_labels = np.array(labels, dtype=np.int32)

    predicted_labels = []
    batch_size = 32
    for start in range(0, total, batch_size):
        batch = all_images[start : start + batch_size]
        probabilities = model.predict(batch, verbose=0)
        predicted_labels.extend(np.argmax(probabilities, axis=1))
    predicted_labels = np.array(predicted_labels, dtype=np.int32)

    # confusion_matrix[true_class][predicted_class] = number of images
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[int(true_label)][int(predicted_label)] += 1

    per_class = []
    precisions = []
    recalls = []
    f1_scores = []
    for class_index in range(num_classes):
        precision, recall, f1 = precision_recall_f1(confusion_matrix, class_index)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class.append(
            {
                "class_name": class_names[class_index],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": sum(confusion_matrix[class_index]),
            }
        )

    correct = sum(confusion_matrix[i][i] for i in range(num_classes))
    return {
        "accuracy": round(correct / total, 4),
        "total_images": total,
        "macro_precision": round(sum(precisions) / num_classes, 4),
        "macro_recall": round(sum(recalls) / num_classes, 4),
        "macro_f1": round(sum(f1_scores) / num_classes, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "class_names": class_names,
            "matrix": confusion_matrix,
        },
        "skipped_files": skipped_files,
        "note": "Metrics are computed on the images currently in the project, "
        "which include the images used for training. Treat them as a sanity check.",
    }
