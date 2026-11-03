import getpass
import os
import sqlite3
from pathlib import Path
from typing import Any

try:
    import mysql.connector
except ImportError:  # pragma: no cover - optional backend dependency
    mysql = None


class PipelineStorage:
    """Store stego audio locally in SQLite or in an explicitly selected MySQL database."""

    def __init__(self) -> None:
        self.backend = os.environ.get("STORAGE_BACKEND", "mysql").strip().lower()
        self.sqlite_path: Path | None = None
        self.database: str | None = None
        self.host: str | None = None
        self.port: int | None = None
        self.user: str | None = None
        self.password: str | None = None

        if self.backend == "sqlite":
            default_path = Path(__file__).resolve().parent.parent / "output" / "stego_audio.sqlite3"
            self.sqlite_path = Path(
                os.environ.get("STORAGE_SQLITE_PATH", str(default_path))
            ).expanduser()
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self.database = str(self.sqlite_path)
            self._create_schema()
            return

        if self.backend != "mysql":
            raise ValueError("STORAGE_BACKEND must be either 'sqlite' or 'mysql'.")

        self.host = os.environ.get("MYSQL_HOST", "localhost")
        self.port = int(os.environ.get("MYSQL_PORT", "3306"))
        self.user = os.environ.get("MYSQL_USER", "root")
        self.database = os.environ.get("MYSQL_DATABASE", "secure_face_authentication")
        self.password = os.environ.get("MYSQL_PASSWORD")
        if self.password is None:
            self.password = getpass.getpass(
                f"MySQL password for {self.user}@{self.host}: "
            )

        self._create_schema()

    @property
    def description(self) -> str:
        if self.backend == "sqlite":
            return f"SQLite at {self.sqlite_path}"
        return f"MySQL {self.database} on {self.host}:{self.port}"

    def _connect(self) -> sqlite3.Connection | Any:
        if self.backend == "sqlite":
            if self.sqlite_path is None:
                raise RuntimeError("SQLite path is not configured.")
            return sqlite3.connect(str(self.sqlite_path))

        if (
            self.host is None
            or self.port is None
            or self.user is None
            or self.database is None
        ):
            raise RuntimeError("MySQL connection settings are not configured.")

        if mysql is None:
            raise RuntimeError(
                "mysql-connector-python is not installed; install requirements or use SQLITE backend."
            )

        return mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
        )

    def _create_schema(self) -> None:
        connection = self._connect()
        cursor = connection.cursor()
        try:
            if self.backend == "sqlite":
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS stego_audio_records (
                        user_id TEXT NOT NULL PRIMARY KEY,
                        name TEXT NOT NULL,
                        stego_audio BLOB NOT NULL
                    )
                    """
                )
            else:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS stego_audio_records (
                        user_id VARCHAR(191) NOT NULL PRIMARY KEY,
                        name VARCHAR(191) NOT NULL,
                        stego_audio LONGBLOB NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
            if self.backend == "sqlite":
                cursor.execute("PRAGMA table_info(stego_audio_records)")
                columns = {row[1] for row in cursor.fetchall()}
                if "name" not in columns and "username" in columns:
                    cursor.execute(
                        "ALTER TABLE stego_audio_records "
                        "RENAME COLUMN username TO name"
                    )
                elif "name" not in columns:
                    cursor.execute(
                        "ALTER TABLE stego_audio_records "
                        "ADD COLUMN name TEXT NOT NULL DEFAULT ''"
                    )
                    cursor.execute(
                        "UPDATE stego_audio_records SET name = user_id "
                        "WHERE name = ''"
                    )
            else:
                cursor.execute("SHOW COLUMNS FROM stego_audio_records")
                columns = {row[0] for row in cursor.fetchall()}
                if "name" not in columns and "username" in columns:
                    cursor.execute(
                        "ALTER TABLE stego_audio_records CHANGE COLUMN "
                        "username name VARCHAR(191) NOT NULL"
                    )
                elif "name" not in columns:
                    cursor.execute(
                        "ALTER TABLE stego_audio_records ADD COLUMN "
                        "name VARCHAR(191) NOT NULL DEFAULT ''"
                    )
                    cursor.execute(
                        "UPDATE stego_audio_records SET name = user_id WHERE name = ''"
                    )
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    def save_stego_audio(self, user_id: str, audio_path: Path, username: str | None = None) -> None:
        audio_data = Path(audio_path).read_bytes()
        username = user_id if username is None else username
        connection = self._connect()
        cursor = connection.cursor()
        try:
            if self.backend == "sqlite":
                cursor.execute(
                    """
                    INSERT INTO stego_audio_records (user_id, name, stego_audio)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        name = excluded.name,
                        stego_audio = excluded.stego_audio
                    """,
                    (user_id, username, audio_data),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO stego_audio_records (user_id, name, stego_audio)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        name = VALUES(name),
                        stego_audio = VALUES(stego_audio)
                    """,
                    (user_id, username, audio_data),
                )
            connection.commit()
            print(
                f"Stored {len(audio_data)} stego-audio bytes for user "
                f"{user_id!r} / name {username!r} in {self.description}."
            )
        finally:
            cursor.close()
            connection.close()

    def retrieve_stego_audio(self, user_id: str, destination: Path) -> Path:
        connection = self._connect()
        cursor = connection.cursor()
        try:
            placeholder = "?" if self.backend == "sqlite" else "%s"
            cursor.execute(
                f"SELECT stego_audio FROM stego_audio_records WHERE user_id = {placeholder}",
                (user_id,),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
            connection.close()

        if row is None:
            raise KeyError(f"No stego audio found for user {user_id!r}")

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(row[0])
        return destination

    def list_user_ids(self) -> list[str]:
        connection = self._connect()
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT user_id FROM stego_audio_records")
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
            connection.close()

    def get_stego_audio(self, user_id: str) -> bytes | None:
        connection = self._connect()
        cursor = connection.cursor()
        try:
            placeholder = "?" if self.backend == "sqlite" else "%s"
            cursor.execute(
                f"SELECT stego_audio FROM stego_audio_records WHERE user_id = {placeholder}",
                (user_id,),
            )
            row = cursor.fetchone()
            return None if row is None else bytes(row[0])
        finally:
            cursor.close()
            connection.close()
