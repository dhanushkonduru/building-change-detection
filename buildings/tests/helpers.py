import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from django.test import SimpleTestCase, override_settings

from buildings.services.detection import classify_footprints


def blank():
    return np.zeros((256, 320), np.float32)


def rect(a, x, y, w, h):
    a[y : y + h, x : x + w] = 0.95
    return a


def classify(a, b, common=None):
    rgb = np.full((*a.shape, 3), 50, np.uint8)
    rgb[b > 0.5] = 210
    return classify_footprints(
        a, b, np.ones_like(a, bool) if common is None else common, rgb, 0.09, "m²"
    )



class ImageryTestCase(SimpleTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.override = override_settings(
            IMAGERY_DIR=self.root,
            RESULTS_DIR=self.root / "out",
            MODEL_PATH=self.root / "weights.pt",
        )
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(self.tmp.cleanup)
        # Deterministic textured fixture lets alignment find a known transform.
        self.rgb = np.random.default_rng(4).integers(
            0, 255, (256, 320, 3), dtype=np.uint8
        )
        self.write("old.tif")
        self.write("new.tif")

    def write(self, name, transform=None, rgb=None):
        a = self.rgb if rgb is None else rgb
        with rasterio.open(
            self.root / name,
            "w",
            driver="GTiff",
            width=a.shape[1],
            height=a.shape[0],
            count=3,
            dtype="uint8",
            crs="EPSG:32644",
            transform=transform or from_origin(450000, 1800000, 0.3, 0.3),
        ) as s:
            s.write(a.transpose(2, 0, 1))

