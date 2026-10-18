import getpass
import os

import mysql.connector

class AudioDatabase:

    def __init__(self):
        password = os.environ.get("MYSQL_PASSWORD")
        if password is None:
            password = getpass.getpass("MySQL password for root@localhost: ")

        self.connection = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "localhost"),
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=os.environ.get("MYSQL_USER", "root"),
            password=password,
            database=os.environ.get(
                "MYSQL_DATABASE",
                "secure_face_authentication",
            ),
        )

        print("Database connected successfully.")

    # ---------------------------------------------------------
    # STORE HIGH-FIDELITY STEGO AUDIO
    # ---------------------------------------------------------

    def store_stego_audio(self, username, audio_path):

        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Stego audio not found: {audio_path}"
            )

        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()

        cursor = self.connection.cursor()

        query = """
            INSERT INTO users
            (username, stego_audio, audio_format)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
            stego_audio = VALUES(stego_audio),
            audio_format = VALUES(audio_format)
        """

        cursor.execute(
            query,
            (
                username,
                audio_data,
                "wav"
            )
        )

        self.connection.commit()

        cursor.close()

        print(
            f"Stego audio stored successfully "
            f"for user: {username}"
        )

        print(
            f"Stored audio size: {len(audio_data)} bytes"
        )

    # ---------------------------------------------------------
    # RETRIEVE HIGH-FIDELITY STEGO AUDIO
    # ---------------------------------------------------------

    def retrieve_stego_audio(
        self,
        username,
        output_path
    ):

        cursor = self.connection.cursor()

        query = """
            SELECT stego_audio
            FROM users
            WHERE username = %s
        """

        cursor.execute(
            query,
            (username,)
        )

        result = cursor.fetchone()

        cursor.close()

        if result is None:
            raise ValueError(
                f"No stego audio found for user: {username}"
            )

        audio_data = result[0]

        os.makedirs(
            os.path.dirname(output_path)
            if os.path.dirname(output_path)
            else ".",
            exist_ok=True
        )

        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_data)

        print(
            f"Stego audio retrieved successfully."
        )

        print(
            f"Retrieved bytes: {len(audio_data)}"
        )

        print(
            f"Saved to: {output_path}"
        )

        return output_path

    # ---------------------------------------------------------
    # CHECK USER
    # ---------------------------------------------------------

    def user_exists(self, username):

        cursor = self.connection.cursor()

        query = """
            SELECT id
            FROM users
            WHERE username = %s
        """

        cursor.execute(
            query,
            (username,)
        )

        result = cursor.fetchone()

        cursor.close()

        return result is not None

    # ---------------------------------------------------------
    # CLOSE DATABASE
    # ---------------------------------------------------------

    def close(self):

        if self.connection.is_connected():
            self.connection.close()
            print("Database connection closed.")