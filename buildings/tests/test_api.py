import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from rasterio.transform import from_origin

from buildings.services.detection import align_images, read_image, resolve_image, InputError
from .helpers import ImageryTestCase, blank, rect


class ApiTests(ImageryTestCase):
    def post(self, payload):
        return self.client.post(
            "/api/detect-changes/",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_api_known_addition_counts_area_previews_and_xy(self):
        old = blank()
        new = rect(blank(), 175, 40, 25, 30)
        with patch(
            "buildings.services.detection.predict",
            side_effect=[(old, np.zeros_like(old)), (new, np.zeros_like(new))] * 2,
        ):
            response = self.post(
                {"old_image_path": "old.tif", "new_image_path": "new.tif"}
            )
        self.assertEqual(response.status_code, 200, response.content)
        data = response.json()
        self.assertEqual(data["new_building_count"], 1)
        self.assertNotIn("review_count", data)
        self.assertEqual(len(data["detections"]), 1)
        self.assertEqual(data["image_dimensions"], {"width": 320, "height": 256})
        d = data["detections"][0]
        self.assertAlmostEqual(d["area"]["value"], 67.5)
        self.assertEqual(d["area"]["unit"], "m²")
        self.assertNotIn("confidence", d)
        self.assertEqual(d["evidence"], "segmentation")
        self.assertGreater(d["building_score"], 0.8)
        self.assertIn("overlap_fraction", data["alignment"])
        self.assertTrue(data["warnings"])
        self.assertEqual(self.client.get(data["result_url"]).status_code, 200)
        for url in data["preview_urls"].values():
            self.assertEqual(self.client.get(url).status_code, 200)

    def test_same_image_api_is_zero(self):
        p = rect(blank(), 50, 50, 40, 40)
        with patch("buildings.services.detection.predict", return_value=(p, np.zeros_like(p))):
            r = self.post({"old_image_path": "old.tif", "new_image_path": "old.tif"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["new_building_count"], 0)

    def test_clear_errors(self):
        for payload, part in [
            ({}, "Provide both"),
            ([], "JSON object"),
            (
                {"old_image_path": "missing.tif", "new_image_path": "new.tif"},
                "not found",
            ),
            (
                {"old_image_path": "../secret.tif", "new_image_path": "new.tif"},
                "inside",
            ),
            ({"old_image_path": "old.png", "new_image_path": "new.tif"}, ".tif"),
        ]:
            r = self.post(payload)
            self.assertEqual(r.status_code, 400)
            self.assertIn(part, r.json()["error"])
        (self.root / "broken.tif").write_text("not an image")
        r = self.post({"old_image_path": "broken.tif", "new_image_path": "new.tif"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("TIFF", r.json()["error"])
        from PIL import Image

        Image.fromarray(self.rgb).save(self.root / "disguised.tif", format="PNG")
        self.assertIn(
            "real TIFF",
            self.post(
                {"old_image_path": "disguised.tif", "new_image_path": "new.tif"}
            ).json()["error"],
        )
        self.assertEqual(
            self.client.post(
                "/api/detect-changes/", data="{", content_type="application/json"
            ).status_code,
            400,
        )
        self.assertEqual(
            self.client.post(
                "/api/detect-changes/", data="x", content_type="text/plain"
            ).status_code,
            415,
        )
        self.assertEqual(self.client.get("/api/detect-changes/").status_code, 405)

    def test_malformed_path_and_large_body_are_json_errors(self):
        bad = self.post({"old_image_path": "bad\x00.tif", "new_image_path": "new.tif"})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("Invalid image path", bad.json()["error"])
        oversized = self.post({"padding": "x" * 3_000_000})
        self.assertEqual(oversized.status_code, 413)
        self.assertIn("too large", oversized.json()["error"])

    def test_symlink_escape_rejected(self):
        (self.root / "escape.tif").symlink_to(Path("/tmp/outside.tif"))
        with self.assertRaisesMessage(InputError, "inside"):
            resolve_image("escape.tif")

    def test_no_comparable_overlap(self):
        self.write("far.tif", from_origin(550000, 1900000, 0.3, 0.3))
        r = self.post({"old_image_path": "old.tif", "new_image_path": "far.tif"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("overlap", r.json()["error"])

    def test_registration_recovers_known_shift(self):
        moved = cv2.warpAffine(
            self.rgb, np.float32([[1, 0, 6], [0, 1, -4]]), (320, 256)
        )
        old, valid, om = read_image(self.root / "old.tif")
        new = moved
        _, common, report = align_images(old, valid, om, new, valid, om)
        self.assertGreater(report["inliers"], 12)
        self.assertAlmostEqual(report["matrix"][0][2], 6, delta=0.5)
        self.assertAlmostEqual(report["matrix"][1][2], -4, delta=0.5)

    def test_missing_weights_error_and_result_path_boundaries(self):
        with patch(
            "buildings.services.detection.predict",
            side_effect=__import__(
                "buildings.services.detection", fromlist=["ModelError"]
            ).ModelError("Building weights are missing. Run setup_model.py"),
        ):
            r = self.post({"old_image_path": "old.tif", "new_image_path": "new.tif"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("weights", r.json()["error"])
        self.assertEqual(self.client.get("/results/123/earlier.png").status_code, 404)

    def test_enhanced_prior_prevents_false_new_building(self):
        empty = blank()
        roof = rect(blank(), 175, 40, 25, 30)
        with patch("buildings.services.detection.predict", side_effect=[
            (empty, empty), (roof, empty), (roof, empty), (roof, empty)
        ]):
            response = self.post({"old_image_path": "old.tif", "new_image_path": "new.tif"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["new_building_count"], 0)


