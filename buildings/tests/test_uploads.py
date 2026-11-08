from unittest.mock import patch

from django.test import override_settings

from buildings.services.detection import read_image
from .helpers import ImageryTestCase, blank, rect


class UploadTests(ImageryTestCase):
    def upload(self, old_name="old.tif", old_content=None, include_new=True):
        from django.core.files.uploadedfile import SimpleUploadedFile
        data = {"old_image": SimpleUploadedFile(
            old_name, (self.root / "old.tif").read_bytes() if old_content is None else old_content,
            content_type="image/tiff",
        )}
        if include_new:
            data["new_image"] = SimpleUploadedFile("new.tif", (self.root / "new.tif").read_bytes())
        return self.client.post("/api/detect-changes/", data)

    def test_uploaded_pair_uses_temporary_files_and_cleans_up(self):
        paths = []
        def inspect(old, new):
            paths.extend([old, new])
            self.assertEqual(read_image(old)[0].shape, self.rgb.shape)
            self.assertTrue(new.is_file())
            return {"new_building_count": 0}
        with patch("buildings.views.detect", side_effect=inspect):
            response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(all(not p.exists() for p in paths))

    def test_missing_invalid_disguised_and_oversized_uploads(self):
        self.assertEqual(self.upload(include_new=False).status_code, 400)
        self.assertEqual(self.upload(old_name="old.png").status_code, 400)
        self.assertEqual(self.upload(old_content=b"invalid TIFF").status_code, 400)
        with override_settings(MAX_IMAGE_UPLOAD_BYTES=10):
            self.assertEqual(self.upload().status_code, 413)

    def test_real_upload_pipeline_returns_polygons_and_previews(self):
        roof = rect(blank(), 175, 40, 25, 30)
        empty = blank()
        with patch("buildings.services.detection.predict", side_effect=[
            (empty, empty), (roof, empty), (empty, empty), (roof, empty)
        ]):
            response = self.upload()
        self.assertEqual(response.status_code, 200, response.content)
        result = response.json()
        self.assertEqual(result["new_building_count"], 1)
        self.assertEqual(self.client.get(result["preview_urls"]["latest"]).status_code, 200)


