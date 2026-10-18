import getpass
import os
from pathlib import Path

import mysql.connector
from mysql.connector.connection import MySQLConnection


class PipelineStorage:
    """Store and retrieve only stego-audio data in MySQL."""

    def __init__(self) -> None:
        self.host = os.environ.get("MYSQL_HOST", "localhost")
        self.port = int(os.environ.get("MYSQL_PORT", "3306"))
        self.user = os.environ.get("MYSQL_USER", "root")
        self.database = os.environ.get(
            "MYSQL_DATABASE",
            "secure_face_authentication",
        )
        self.password = os.environ.get("MYSQL_PASSWORD")
        if self.password is None:
            self.password = getpass.getpass(
                f"MySQL password for {self.user}@{self.host}: "
            )

        self._create_schema()

    def _connect(self) -> MySQLConnection:
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
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS stego_audio_records (
                    user_id VARCHAR(191) NOT NULL PRIMARY KEY,
                    stego_audio LONGBLOB NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    def save_stego_audio(self, user_id: str, audio_path: Path) -> None:
        audio_data = Path(audio_path).read_bytes()
        connection = self._connect()
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO stego_audio_records (user_id, stego_audio)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE stego_audio = VALUES(stego_audio)
                """,
                (user_id, audio_data),
            )
            connection.commit()
            print(
                f"Stored {len(audio_data)} stego-audio bytes for username "
                f"{user_id!r} in {self.database}.stego_audio_records."
            )
        finally:
            cursor.close()
            connection.close()

    def retrieve_stego_audio(self, user_id: str, destination: Path) -> Path:
        connection = self._connect()
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT stego_audio FROM stego_audio_records WHERE user_id = %s",
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
            cursor.execute(
                "SELECT stego_audio FROM stego_audio_records WHERE user_id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
            return None if row is None else bytes(row[0])
        finally:
            cursor.close()
            connection.close()