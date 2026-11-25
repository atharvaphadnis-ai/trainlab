""" dataset.py : Saving uploaded images to disk and loading the dataset for training.

p>
"""

import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import database
from utils import (
    ALLOWED_PIL_FORMATS,
    MAX_IMAGE_BYTES,
    bytes_sha256,
    file_extension,
    is_allowed_extension,
    projects_dir,
)

# Training needs a minimum amount of data to be meaningful.
MIN_CLASSES = 2
MIN_IMAGES_PER_CLASS = 5


def class_folder(project_id, class_id):
    folder = projects_dir() / str(project_id) / str(class_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def project_folder(project_id):
    folder = projects_dir() / str(project_id)
    return folder


def validate_image_bytes(file_bytes, original_name):
    """Check size, extension and that PIL can really decode the file."""
    if len(file_bytes) == 0:
        raise ValueError("The file is empty.")
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("The file is larger than 10 MB.")
    if not is_allowed_extension(original_name):
        raise ValueError("Allowed formats are JPG, JPEG, PNG and WEBP.")
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # checks the file is a complete, undamaged image
        pil_format = image.format
    except Exception:
        raise ValueError("This file is not a readable image.")

    if pil_format not in ALLOWED_PIL_FORMATS:
        raise ValueError("Allowed formats are JPG, JPEG, PNG and WEBP.")


def save_uploaded_image(project_id, class_id, file_bytes, original_name):
    """Store one uploaded image. Raises ValueError with a readable reason."""
    validate_image_bytes(file_bytes, original_name)

    sha256 = bytes_sha256(file_bytes)
    if database.project_has_image(project_id, sha256):
        raise ValueError(
            f"Duplicate: '{original_name}' already exists somewhere in this project."
        )

    extension = file_extension(original_name)
    filename = f"{uuid.uuid4().hex}.{extension}"
    save_path = class_folder(project_id, class_id) / filename
    save_path.write_bytes(file_bytes)

    image_id = database.add_image(class_id, filename, original_name, sha256)
    return {
        "id": image_id,
        "class_id": class_id,
        "filename": filename,
        "original_name": original_name,
    }


def image_path(image_row):
    return project_folder(image_row["project_id"]) / str(image_row["class_id"]) / image_row["filename"]


def delete_image_file(image_row):
    """Remove one image from disk and from the database."""
    path = image_path(image_row)
    if path.exists():
        path.unlink()
    database.delete_image(image_row["id"])


def delete_class_files(project_id, class_id):
    folder = projects_dir() / str(project_id) / str(class_id)
    if folder.exists():
        for file in folder.iterdir():
            file.unlink()
        folder.rmdir()


def delete_project_files(project_id):
    folder = project_folder(project_id)
    if folder.exists():
        for class_folder_path in folder.iterdir():
            for file in class_folder_path.iterdir():
                file.unlink()
            class_folder_path.rmdir()
        folder.rmdir()


def load_image_array(image_path, image_size):
    """Read one image file as a numpy array of shape (size, size, 3).

    uint8 keeps memory small; the model converts to float itself.
    """
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)  # respect phone photo rotation
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def load_dataset_for_training(project_id, image_size, validation_split):
    """Load every image of the project and split into train/validation.

    Returns a dict with numpy arrays and the class names in label order.
    Raises ValueError with a readable message when there is not enough data.
    """
    classes = database.list_classes(project_id)
    usable_classes = [c for c in classes if c["image_count"] > 0]

    if len(classes) < MIN_CLASSES:
        raise ValueError(
            f"Training needs at least {MIN_CLASSES} classes. This project has {len(classes)}."
        )

    too_small = [c["name"] for c in usable_classes if c["image_count"] < MIN_IMAGES_PER_CLASS]
    if too_small:
        raise ValueError(
            "Every class needs at least "
            f"{MIN_IMAGES_PER_CLASS} images. Too few images in: {', '.join(too_small)}."
        )

    class_names = []
    all_images = []
    all_labels = []
    for label_index, one_class in enumerate(usable_classes):
        class_names.append(one_class["name"])
        for image_row in database.list_images(one_class["id"]):
            path = image_path(image_row)
            try:
                pixels = load_image_array(path, image_size)
            except Exception:
                raise ValueError(f"Could not read image file '{image_row['original_name']}'.")
            all_images.append(pixels)
            all_labels.append(label_index)

    images = np.stack(all_images)  # shape: (total, size, size, 3)
    labels = np.array(all_labels, dtype=np.int32)

    train_indices, val_indices = split_train_validation(labels, validation_split)
    return {
        "class_names": class_names,
        "train_images": images[train_indices],
        "train_labels": labels[train_indices],
        "val_images": images[val_indices],
        "val_labels": labels[val_indices],
    }


def split_train_validation(labels, validation_split):
    """Split indices so every class appears in both train and validation."""
    train_indices = []
    val_indices = []
    rng = np.random.default_rng(42)  # fixed seed so retraining is repeatable
    for class_index in np.unique(labels):
        class_positions = np.where(labels == class_index)[0]
        rng.shuffle(class_positions)
        if validation_split > 0:
            val_count = max(round(len(class_positions) * validation_split), 1)
            val_indices.extend(class_positions[:val_count])
            train_indices.extend(class_positions[val_count:])
        else:
            train_indices.extend(class_positions)
    return np.array(train_indices), np.array(val_indices)
