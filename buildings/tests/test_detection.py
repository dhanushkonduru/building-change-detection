import numpy as np
from django.test import SimpleTestCase

from buildings.services.detection import classify_footprints, recover_low_light_roofs
from .helpers import blank, rect, classify


class ChangeRuleTests(SimpleTestCase):
    def test_touching_roofs_split_but_small_roof_ridges_do_not(self):
        old = blank()
        new = rect(blank(), 80, 60, 100, 50)
        boundary = blank()
        boundary[60:110, 128:132] = 0.9
        rgb = np.full((*new.shape, 3), 200, np.uint8)
        detections = classify_footprints(
            old, new, np.ones_like(new, bool), rgb, boundary=boundary
        )
        self.assertEqual(len(detections), 2)
        self.assertTrue(all(d["classification"] == "new" for d in detections))
        small = rect(blank(), 80, 60, 40, 25)
        boundary = blank()
        boundary[60:85, 98:102] = 0.9
        detections = classify_footprints(
            old, small, np.ones_like(new, bool), rgb, boundary=boundary
        )
        self.assertEqual(len(detections), 1)

    def test_dark_texture_is_excluded_despite_high_model_probability(self):
        old = blank()
        new = rect(blank(), 80, 60, 40, 30)
        dark = np.full((*new.shape, 3), 30, np.uint8)
        detections = classify_footprints(old, new, np.ones_like(new, bool), dark)
        self.assertEqual(detections, [])

    def test_known_added_l_shaped_footprint_preserves_coordinates_and_corners(self):
        old = blank()
        new = blank()
        new[45:85, 170:190] = 0.95
        new[65:85, 190:220] = 0.95
        d = classify(old, new)
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]["classification"], "new")
        self.assertEqual(d[0]["id"], "N01")
        self.assertEqual(d[0]["polygon"][0], d[0]["polygon"][-1])
        coords = np.array(d[0]["polygon"])
        self.assertEqual(coords[:, 0].min(), 170)
        self.assertEqual(coords[:, 1].min(), 45)
        self.assertEqual(coords[:, 0].max(), 220)
        self.assertEqual(coords[:, 1].max(), 85)
        self.assertGreater(len(coords), 5)
        self.assertAlmostEqual(d[0]["area"]["value"], 126, delta=1)

    def test_same_existing_removed_and_shifted_buildings_return_zero_new(self):
        old = rect(blank(), 40, 40, 35, 40)
        rect(old, 200, 150, 40, 35)
        self.assertEqual(classify(old, old), [])
        self.assertEqual(classify(old, blank()), [])
        shifted = rect(blank(), 46, 43, 35, 40)
        self.assertEqual(classify(old, shifted), [])

    def test_extensions_are_excluded_and_neighbours_stay_separate(self):
        old = rect(blank(), 60, 40, 40, 40)
        ext = rect(blank(), 60, 40, 65, 60)
        self.assertEqual(classify(old, ext), [])
        new = rect(blank(), 150, 40, 25, 30)
        rect(new, 182, 40, 25, 30)
        self.assertEqual(len(classify(blank(), new)), 2)

    def test_boundary_and_prior_ambiguity_is_excluded(self):
        old = rect(blank(), 100, 50, 4, 30)
        new = rect(blank(), 110, 50, 35, 30)
        ds = classify(old, new)
        self.assertEqual(ds, [])
        new = rect(blank(), 0, 50, 25, 30)
        self.assertEqual(classify(blank(), new), [])

    def test_invalid_overlap_excluded(self):
        new = rect(blank(), 150, 50, 25, 30)
        valid = np.ones_like(new, bool)
        valid[:, 140:] = False
        self.assertEqual(classify(blank(), new, valid), [])


class ContrastRecoveryTests(SimpleTestCase):
    def recover(self, raw, enhanced, old=None, existing=None, rgb=None):
        return recover_low_light_roofs(
            [] if existing is None else existing,
            blank() if old is None else old, raw, enhanced, blank(),
            np.ones_like(raw, bool),
            np.full((*raw.shape, 3), 180, np.uint8) if rgb is None else rgb,
        )

    def test_recovers_supported_roof_and_rejects_enhancement_only(self):
        roof = rect(blank(), 80, 60, 30, 30)
        raw = roof * 0.65
        self.assertEqual(classify(blank(), raw), [])
        result = self.recover(raw, roof)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["evidence"], "contrast_recovery")
        self.assertEqual(self.recover(blank(), roof), [])

    def test_prior_roofs_dark_texture_and_empty_images_stay_excluded(self):
        roof = rect(blank(), 80, 60, 30, 30)
        self.assertEqual(self.recover(roof, roof, old=roof), [])
        self.assertEqual(self.recover(blank(), blank()), [])
        dark = np.full((*roof.shape, 3), 30, np.uint8)
        self.assertEqual(self.recover(roof, roof, rgb=dark), [])

    def test_recovery_does_not_duplicate_existing_roofs_or_merge_row_units(self):
        raw = rect(blank(), 80, 60, 30, 30)
        rect(raw, 120, 60, 30, 30)
        existing = classify(blank(), raw)
        self.assertEqual(len(existing), 2)
        merged = rect(blank(), 80, 60, 70, 30)
        result = self.recover(raw, merged, existing=existing)
        self.assertEqual(len(result), 2)
        self.assertEqual([d["id"] for d in result], ["N01", "N02"])
