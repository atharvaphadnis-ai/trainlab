"database.py ---- storing the datasets of the ai model"

import sqlite3

from utils import data_dir, now_string

# Plain English class names must be unique inside a project.
UNIQUE_NAME_ERROR = "That name is already used in this project."


def connect():
    connection = sqlite3.connect(data_dir() / "trainlab.db")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database():
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(project_id, name)
            );

            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL DEFAULT '',
                sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(class_id, sha256)
            );
            """
        )


def row_to_dict(row):
    return dict(row) if row is not None else None


# --------------------------- projects ---------------------------


def create_project(name, description):
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO projects (name, description, created_at) VALUES (?, ?, ?)",
            (name, description, now_string()),
        )
        return cursor.lastrowid


def list_projects():
    """All projects, newest first, with a total image count for each."""
    with connect() as db:
        rows = db.execute(
            """
            SELECT p.id, p.name, p.description, p.created_at,
                   COUNT(images.id) AS image_count
            FROM projects p
            LEFT JOIN classes ON classes.project_id = p.id
            LEFT JOIN images ON images.class_id = classes.id
            GROUP BY p.id
            ORDER BY p.id DESC
            """
        ).fetchall()
        return [row_to_dict(row) for row in rows]


def get_project(project_id):
    with connect() as db:
        row = db.execute(
            "SELECT id, name, description, created_at FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        return row_to_dict(row)


def delete_project(project_id):
    """Returns True if a project was actually deleted."""
    with connect() as db:
        cursor = db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0


# --------------------------- classes ---------------------------


def create_class(project_id, name):
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO classes (project_id, name, created_at) VALUES (?, ?, ?)",
            (project_id, name, now_string()),
        )
        return cursor.lastrowid


def list_classes(project_id):
    """Classes in creation order with how many images each one holds."""
    with connect() as db:
        rows = db.execute(
            """
            SELECT c.id, c.project_id, c.name, c.created_at,
                   COUNT(i.id) AS image_count
            FROM classes c
            LEFT JOIN images i ON i.class_id = c.id
            WHERE c.project_id = ?
            GROUP BY c.id
            ORDER BY c.id
            """,
            (project_id,),
        ).fetchall()
        return [row_to_dict(row) for row in rows]


def get_class(class_id):
    with connect() as db:
        row = db.execute(
            "SELECT id, project_id, name, created_at FROM classes WHERE id = ?",
            (class_id,),
        ).fetchone()
        return row_to_dict(row)


def class_name_exists(project_id, name, ignore_class_id=None):
    """True when another class in the same project already uses the name."""
    with connect() as db:
        row = db.execute(
            """
            SELECT id FROM classes
            WHERE project_id = ? AND LOWER(name) = LOWER(?)
            """,
            (project_id, name),
        ).fetchone()
        if row is None:
            return False
        return ignore_class_id is None or row["id"] != ignore_class_id


def rename_class(class_id, new_name):
    with connect() as db:
        db.execute("UPDATE classes SET name = ? WHERE id = ?", (new_name, class_id))


def delete_class(class_id):
    with connect() as db:
        db.execute("DELETE FROM classes WHERE id = ?", (class_id,))


# --------------------------- images ---------------------------


def add_image(class_id, filename, original_name, sha256):
    with connect() as db:
        cursor = db.execute(
            """
            INSERT INTO images (class_id, filename, original_name, sha256, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (class_id, filename, original_name, sha256, now_string()),
        )
        return cursor.lastrowid


def list_images(class_id):
    with connect() as db:
        rows = db.execute(
            """
            SELECT images.id, images.class_id, images.filename, images.original_name,
                   images.created_at, classes.project_id
            FROM images
            JOIN classes ON classes.id = images.class_id
            WHERE images.class_id = ? ORDER BY images.id
            """,
            (class_id,),
        ).fetchall()
        return [row_to_dict(row) for row in rows]


def get_image(image_id):
    with connect() as db:
        row = db.execute(
            """
            SELECT images.id, images.class_id, images.filename, images.original_name,
                   images.created_at, classes.project_id
            FROM images
            JOIN classes ON classes.id = images.class_id
            WHERE images.id = ?
            """,
            (image_id,),
        ).fetchone()
        return row_to_dict(row)


def delete_image(image_id):
    with connect() as db:
        db.execute("DELETE FROM images WHERE id = ?", (image_id,))


def project_has_image(project_id, sha256):
    """True when this exact file content is already stored anywhere in the project."""
    with connect() as db:
        row = db.execute(
            """
            SELECT images.id
            FROM images
            JOIN classes ON classes.id = images.class_id
            WHERE classes.project_id = ? AND images.sha256 = ?
            """,
            (project_id, sha256),
        ).fetchone()
        return row is not None


def count_project_images(project_id):
    with connect() as db:
        row = db.execute(
            """
            SELECT COUNT(images.id) AS total
            FROM images
            JOIN classes ON classes.id = images.class_id
            WHERE classes.project_id = ?
            """,
            (project_id,),
        ).fetchone()
        return row["total"]
