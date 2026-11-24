from io import BytesIO
from minio import Minio

from .config import settings


class FrameStore:
    def __init__(self) -> None:
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False,
        )
        self.bucket = settings.minio_bucket

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def put_jpeg(self, object_name: str, data: bytes) -> str:
        self.client.put_object(
            self.bucket,
            object_name,
            BytesIO(data),
            length=len(data),
            content_type="image/jpeg",
        )
        return object_name

    def get_bytes(self, object_name: str) -> bytes:
        response = self.client.get_object(self.bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
