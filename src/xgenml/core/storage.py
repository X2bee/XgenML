import os, shutil, urllib.parse, boto3
from pathlib import Path

class StorageClient:
    """
    base_uri 예:
      - file:///base/dir
      - s3://bucket/prefix
    MinIO는 S3_ENDPOINT_URL 환경변수로 지정
    """
    def __init__(self, base_uri: str, s3_endpoint_url: str | None = None,
                 access_key: str | None = None, secret_key: str | None = None,
                 region: str = "ap-northeast-2"):
        self.base_uri = base_uri.rstrip("/")
        parsed = urllib.parse.urlparse(self.base_uri)
        self.scheme = parsed.scheme or "file"

        if self.scheme == "s3":
            self._s3 = boto3.client(
                "s3",
                endpoint_url=s3_endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
            )

    def _join_uri(self, *parts) -> str:
        tail = "/".join(str(p).strip("/") for p in parts)
        return f"{self.base_uri}/{tail}"

    def upload_file(self, local_path: str, rel_path: str) -> str:
        dst_uri = self._join_uri(rel_path)
        parsed = urllib.parse.urlparse(dst_uri)

        if parsed.scheme in ("file", ""):
            dst = Path(parsed.path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dst)
        elif parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            self._s3.upload_file(local_path, bucket, key)
        else:
            raise ValueError(f"Unsupported scheme: {parsed.scheme}")

        return dst_uri

    def download_file(self, src_uri: str, local_path: str) -> str:
        parsed = urllib.parse.urlparse(src_uri)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        if parsed.scheme in ("file", ""):
            shutil.copy2(Path(parsed.path), local_path)
        elif parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            self._s3.download_file(bucket, key, local_path)
        else:
            raise ValueError(f"Unsupported scheme: {parsed.scheme}")

        return local_path
