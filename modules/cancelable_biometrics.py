import hashlib
import numpy as np


class CancelableBiometrics:

    def __init__(self, dimension=512):

        self.dimension = dimension

    def generate_projection_matrix(self, user_key):

        hash_value = hashlib.sha256(
            user_key.encode("utf-8")
        ).digest()

        seed = int.from_bytes(
            hash_value[:8],
            byteorder="big"
        )

        rng = np.random.default_rng(seed)

        matrix = rng.normal(
            0,
            1,
            (self.dimension, self.dimension),
        )
        projection, triangular = np.linalg.qr(matrix)
        signs = np.sign(np.diag(triangular))
        signs[signs == 0] = 1

        return (projection * signs).astype(np.float32)

    def transform(
        self,
        face_embedding,
        user_key
    ):

        face_embedding = np.asarray(
            face_embedding,
            dtype=np.float32
        ).flatten()

        if face_embedding.shape[0] != self.dimension:

            raise ValueError(
                f"Expected {self.dimension}-D "
                f"embedding but received "
                f"{face_embedding.shape[0]}-D"
            )

        projection_matrix = (
            self.generate_projection_matrix(
                user_key
            )
        )

        transformed = (
            projection_matrix @ face_embedding
        )

        # Normalize transformed template
        transformed = transformed / (
            np.linalg.norm(transformed)
            + 1e-8
        )

        return transformed.astype(
            np.float32
        )