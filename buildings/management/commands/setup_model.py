"""Download and verify the pinned building model."""
import hashlib
import urllib.request
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

REVISION = "f7254545dfbbf10a40594d5a5fb62136902000ae"
SHA256 = "9a28e28403accc06c70a3104aedb31eca50401ed23c8ba4314b26357c25e9043"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


class Command(BaseCommand):
    help = "Download the pinned building model and verify its checksum."

    def handle(self, *args, **options):
        path = settings.MODEL_PATH
        url = f"https://huggingface.co/hotosm/dinov3s-buildings/resolve/{REVISION}/model.onnx"
        if not path.exists() or digest(path) != SHA256:
            self.stdout.write("Downloading 220 MB public building weights (no account required)…")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".download")
            try:
                urllib.request.urlretrieve(url, tmp)
                if digest(tmp) != SHA256:
                    raise CommandError("Checkpoint checksum mismatch; download rejected.")
                tmp.replace(path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        self.stdout.write(f"Model ready: {path}")
