"""Overhead building evidence -> full roof polygons -> tolerant temporal matching."""

from pathlib import Path
from functools import lru_cache
import hashlib, json, threading, tempfile
import cv2
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
from shapely.geometry import shape, Polygon
from PIL import Image
from django.conf import settings

VERSION = "atlas-1.0"
LOCK = threading.Lock()


class InputError(ValueError):
    pass


class ModelError(RuntimeError):
    pass


def resolve_image(value):
    if not isinstance(value, str) or not value.strip():
        raise InputError(
            "Provide both old_image_path and new_image_path as non-empty strings."
        )
    root = settings.IMAGERY_DIR.resolve()
    try:
        path = (root / value).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise InputError("Invalid image path.") from exc
    if not path.is_relative_to(root):
        raise InputError(
            "Image paths must stay inside the configured imagery directory."
        )
    if path.suffix.lower() != ".tif":
        raise InputError("Only .tif files are accepted.")
    if not path.is_file():
        raise InputError(f"Image file not found: {path.name}")
    return path


def read_image(path):
    try:
        with rasterio.open(path) as s:
            if s.driver != "GTiff":
                raise InputError(f"{path.name} is not a real TIFF image.")
            if s.count < 3 or s.dtypes[0] != "uint8":
                raise InputError("Use an RGB, 8-bit TIFF with at least three bands.")
            if not s.crs:
                raise InputError(
                    "TIFF imagery must have a coordinate reference system."
                )
            if s.width * s.height > 25_000_000:
                raise InputError("Images must contain at most 25 million pixels.")
            return (
                s.read([1, 2, 3]).transpose(1, 2, 0),
                s.dataset_mask() > 0,
                dict(transform=s.transform, crs=s.crs, width=s.width, height=s.height),
            )
    except rasterio.errors.RasterioError as e:
        raise InputError(f"Cannot read {path.name}: invalid or damaged TIFF.") from e


def align_images(old, old_valid, om, new, new_valid, nm):
    h, w = new.shape[:2]
    warped = np.zeros_like(new)
    for b in range(3):
        reproject(
            old[:, :, b],
            warped[:, :, b],
            src_transform=om["transform"],
            src_crs=om["crs"],
            dst_transform=nm["transform"],
            dst_crs=nm["crs"],
            resampling=Resampling.bilinear,
        )
    valid = np.zeros((h, w), np.uint8)
    reproject(
        old_valid.astype("uint8"),
        valid,
        src_transform=om["transform"],
        src_crs=om["crs"],
        dst_transform=nm["transform"],
        dst_crs=nm["crs"],
        resampling=Resampling.nearest,
    )
    common = (valid > 0) & new_valid
    if common.sum() < max(256, 0.01 * h * w):
        raise InputError("These images have no usable comparable overlap.")

    def gray(a):
        return cv2.createCLAHE(2, (8, 8)).apply(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY))

    sift = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = sift.detectAndCompute(gray(warped), common.astype("uint8") * 255)
    k2, d2 = sift.detectAndCompute(gray(new), common.astype("uint8") * 255)
    matrix = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    report = {
        "method": "georeprojection only",
        "inliers": 0,
        "median_residual_px": None,
        "tie_points": [],
    }
    if d1 is not None and d2 is not None and len(d1) > 2 and len(d2) > 2:
        pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
        if len(good) >= 12:
            src = np.float32([k1[m.queryIdx].pt for m in good])
            dst = np.float32([k2[m.trainIdx].pt for m in good])
            m, inliers = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=5000
            )
            if (
                m is not None
                and inliers.sum() >= 12
                and 0.95 < np.linalg.norm(m[0, :2]) < 1.05
                and np.linalg.norm(m[:, 2]) < 100
            ):
                matrix = m
                used = inliers.ravel().astype(bool)
                residual = np.linalg.norm(
                    src @ matrix[:, :2].T + matrix[:, 2] - dst, axis=1
                )
                report = {
                    "method": "georeprojection + SIFT/RANSAC similarity",
                    "inliers": int(used.sum()),
                    "median_residual_px": round(float(np.median(residual[used])), 3),
                    "tie_points": [
                        {
                            "earlier": a.round(2).tolist(),
                            "latest": b.round(2).tolist(),
                            "residual_px": round(float(e), 2),
                        }
                        for a, b, e in zip(src[used], dst[used], residual[used])
                    ],
                }
                warped = cv2.warpAffine(warped, m, (w, h), flags=cv2.INTER_LINEAR)
                valid = cv2.warpAffine(valid, m, (w, h), flags=cv2.INTER_NEAREST)
    # Strong temporal appearance changes can defeat SIFT. Local normalized
    # correlation on blurred texture supplies independent, spatially bounded ties.
    if report["inliers"] == 0 and min(h, w) >= 240:
        a = cv2.GaussianBlur(gray(warped), (9, 9), 0)
        b = cv2.GaussianBlur(gray(new), (9, 9), 0)
        ties = []
        radius = min(40, min(h, w) // 12)
        step = max(32, min(h, w) // 25)
        for half in (32, 48, 64):
            margin = half + radius + 2
            for y in range(margin, h - margin, step):
                for x in range(margin, w - margin, step):
                    if not common[y - half : y + half, x - half : x + half].all():
                        continue
                    template = a[y - half : y + half, x - half : x + half]
                    if template.std() < 8:
                        continue
                    search = b[
                        y - half - radius : y + half + radius,
                        x - half - radius : x + half + radius,
                    ]
                    corr = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                    _, score, _, loc = cv2.minMaxLoc(corr)
                    if (
                        score > 0.60
                        and 2 < loc[0] < 2 * radius - 2
                        and 2 < loc[1] < 2 * radius - 2
                    ):
                        ties.append(
                            ((x, y), (x + loc[0] - radius, y + loc[1] - radius), score)
                        )
        if len(ties) >= 12:
            src = np.float32([t[0] for t in ties])
            dst = np.float32([t[1] for t in ties])
            m, inliers = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=5, maxIters=15000
            )
            used = (
                inliers.ravel().astype(bool)
                if inliers is not None
                else np.zeros(len(ties), bool)
            )
            spread = np.ptp(src[used], axis=0) if used.any() else np.zeros(2)
            if (
                m is not None
                and used.sum() >= 20
                and used.mean() >= 0.2
                and 0.95 < np.linalg.norm(m[0, :2]) < 1.05
                and np.linalg.norm(m[:, 2]) < 80
                and spread[0] > 0.3 * w
                and spread[1] > 0.3 * h
            ):
                matrix = m
                residual = np.linalg.norm(
                    src @ matrix[:, :2].T + matrix[:, 2] - dst, axis=1
                )
                report = {
                    "method": "georeprojection + local NCC/RANSAC similarity",
                    "inliers": int(used.sum()),
                    "median_residual_px": round(float(np.median(residual[used])), 3),
                    "tie_points": [
                        {
                            "earlier": a.round(2).tolist(),
                            "latest": b.round(2).tolist(),
                            "residual_px": round(float(e), 2),
                        }
                        for a, b, e in zip(src[used], dst[used], residual[used])
                    ],
                }
                warped = cv2.warpAffine(warped, m, (w, h), flags=cv2.INTER_LINEAR)
                valid = cv2.warpAffine(valid, m, (w, h), flags=cv2.INTER_NEAREST)
    common = (valid > 0) & new_valid
    if common.sum() < max(256, 0.01 * h * w):
        raise InputError(
            "These images have no usable comparable overlap after registration."
        )
    report["note"] = (
        "Visual correction accepted."
        if report["inliers"]
        else "Visual matching was not reliable; retained the supplied GeoTIFF registration. Roof parallax remains."
    )
    report.update(
        matrix=matrix.tolist(),
        overlap_fraction=round(float(common.mean()), 4),
        georeferenced=True,
    )
    return warped, common, report


@lru_cache(maxsize=1)
def load_model():
    import onnxruntime as ort

    if not settings.MODEL_PATH.is_file():
        raise ModelError(
            "Building weights are missing. Run: python manage.py setup_model"
        )
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    try:
        return ort.InferenceSession(
            str(settings.MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise ModelError("Cannot load building weights. Run: python manage.py setup_model") from exc


def predict(rgb):
    session = load_model()
    mean = np.array(
        [0.4296737853453577, 0.4001659668453235, 0.34333372802741474], np.float32
    )
    std = np.array(
        [0.2056069389373208, 0.16738555558380538, 0.1598986422586595], np.float32
    )
    h, w = rgb.shape[:2]
    tile = 256
    stride = 192
    padded = np.pad(
        rgb, ((0, max(0, tile - h)), (0, max(0, tile - w)), (0, 0)), mode="reflect"
    )
    ph, pw = padded.shape[:2]
    probs = np.zeros((ph, pw), np.float32)
    weights = np.zeros_like(probs)
    boundaries = np.zeros_like(probs)
    ys = sorted(set(list(range(0, ph - tile + 1, stride)) + [ph - tile]))
    xs = sorted(set(list(range(0, pw - tile + 1, stride)) + [pw - tile]))
    axis = np.arange(tile) - (tile - 1) / 2
    g = np.exp(-(axis**2) / (2 * (tile * 0.125) ** 2))
    window = np.outer(g, g).astype("float32")
    for y in ys:
        for x in xs:
            a = (
                padded[y : y + tile, x : x + tile].astype("float32") / 255 - mean
            ) / std
            logits = session.run(None, {"image": a.transpose(2, 0, 1).copy()[None]})[0]
            # Independent heads: mask, boundary, signed distance. NOT a 3-class softmax.
            p = 1 / (1 + np.exp(-np.clip(logits[0, 0], -30, 30)))
            probs[y : y + tile, x : x + tile] += p * window
            weights[y : y + tile, x : x + tile] += window
            boundary = 1 / (1 + np.exp(-np.clip(logits[0, 1], -30, 30)))
            boundaries[y : y + tile, x : x + tile] += boundary * window
    return (probs / weights)[:h, :w], (boundaries / weights)[:h, :w]


def contrast_image(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def attached_roofs(rgb, probability, prior, common):
    """Split rows using cross-walls that span a model-supported, elongated roof."""
    gray = cv2.createCLAHE(2, (8, 8)).apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    mask = (probability > 0.48).astype("uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roofs = []
    for contour in contours:
        corners = cv2.boxPoints(cv2.minAreaRect(contour))
        sides = np.diff(corners[:3], axis=0)
        sizes = np.linalg.norm(sides, axis=1)
        width, length = sorted(sizes)
        if (
            width < 14
            or width > 128
            or length < 2.8 * width
            or cv2.contourArea(contour) < 0.55 * width * length
        ):
            continue
        across = sides[np.argmin(sizes)] / width
        along = sides[np.argmax(sizes)] / length
        center = corners.mean(axis=0)
        t = np.arange(-int(length / 2 + width * 7), int(length / 2 + width * 7))
        s = np.arange(-int(width * 0.70), int(width * 0.70) + 1)
        grid = center + t[:, None, None] * along + s[None, :, None] * across
        x, y = grid.transpose(2, 0, 1).astype("float32")
        strip = cv2.remap(gray, x, y, cv2.INTER_LINEAR)
        vertical = abs(cv2.Sobel(strip, cv2.CV_32F, 1, 0))
        horizontal = abs(cv2.Sobel(strip, cv2.CV_32F, 0, 1))
        middle = abs(s) < 0.36 * width
        score = (horizontal[:, middle] > 75).mean(axis=1)
        # Nonmaximum suppression collapses the two edges of each parapet.
        peaks = []
        for k in np.argsort(-score, kind="stable"):
            if score[k] < 0.70:
                break
            if all(abs(k - j) > 0.58 * width for j in peaks):
                peaks.append(k)
        peaks.sort()
        bays = []
        for a, b in zip(peaks, peaks[1:]):
            if not 0.59 * width < b - a < 1.48 * width:
                continue
            edge = cv2.blur(
                (vertical[a + 2 : b - 1] > 55).astype("float32"), (3, 1)
            ).mean(axis=0)
            left, right = [
                np.argmax(
                    np.where(
                        (s * sign > 0.32 * width) & (s * sign < 0.69 * width), edge, -1
                    )
                )
                for sign in (-1, 1)
            ]
            if min(edge[left], edge[right]) < 0.68:
                continue
            xx = np.rint(x[a:b, left:right]).astype(int)
            yy = np.rint(y[a:b, left:right]).astype(int)
            h, w = common.shape
            if xx.min() < 0 or xx.max() >= w or yy.min() < 0 or yy.max() >= h:
                continue
            if (
                not common[yy, xx].all()
                or prior[yy, xx].mean() > 0.04
                or probability[yy, xx].mean() < 0.05
            ):
                continue
            polygon = Polygon(
                [grid[a, left], grid[a, right], grid[b, right], grid[b, left]]
            )
            bays.append((a, b, polygon))
        # A single elongated roof can have ridges. Require at least four adjoining bays.
        for i, (a, b, poly) in enumerate(bays):
            lo = hi = i
            while lo and bays[lo - 1][1] == bays[lo][0]:
                lo -= 1
            while hi + 1 < len(bays) and bays[hi][1] == bays[hi + 1][0]:
                hi += 1
            if hi - lo >= 3 and not any(
                poly.intersection(p).area > 0.4 * min(poly.area, p.area) for p in roofs
            ):
                roofs.append(poly)
    return roofs


def classify_footprints(
    old_prob,
    new_prob,
    common,
    new_rgb,
    pixel_area=1,
    unit="px²",
    boundary=None,
    row_probability=None,
    row_prior=None,
):
    """Compare whole latest components, never just their changed strips."""
    old_mask = (old_prob >= 0.35).astype("uint8")
    prior = (
        cv2.dilate(old_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0
    )
    new_mask = ((new_prob >= 0.5) & common).astype("uint8")
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(new_mask, 8)
    if boundary is not None:
        # Predicted internal boundaries provide separate seeds for touching roofs.
        seeds = ((new_mask > 0) & (boundary < 0.06) & (new_prob > 0.6)).astype("uint8")
        seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        sn, sl, ss, _ = cv2.connectedComponentsWithStats(seeds, 8)
        markers = np.zeros_like(labels, dtype="int32")
        markers[new_mask == 0] = 1
        next_id = 2
        seeded_components = set()
        for j in sorted(range(1, sn), key=lambda j: -ss[j, cv2.CC_STAT_AREA]):
            source_label = labels[sl == j][0]
            # Small roofs often have separate cores on either side of a ridge.
            # Keep one seed there; reserve instance splitting for larger blobs.
            if (
                stats[source_label, cv2.CC_STAT_AREA] < 1500
                and source_label in seeded_components
            ):
                continue
            if ss[j, cv2.CC_STAT_AREA] >= max(
                35, 0.08 * stats[source_label, cv2.CC_STAT_AREA]
            ):
                markers[sl == j] = next_id
                next_id += 1
                seeded_components.add(source_label)
        # Components with no core still get one seed, so ambiguity is retained.
        for j in range(1, n):
            component = labels == j
            if not np.any(markers[component] > 1):
                points = np.argwhere(component)
                if len(points):
                    y, x = points[len(points) // 2]
                    markers[y, x] = next_id
                    next_id += 1
        relief = np.repeat(
            np.uint8(np.clip(boundary, 0, 1) * 255)[:, :, None], 3, axis=2
        )
        cv2.watershed(relief, markers)
        # Reassign the one-pixel watershed seam while keeping the original mask.
        from scipy.ndimage import distance_transform_edt

        _, indices = distance_transform_edt(markers <= 1, return_indices=True)
        filled = markers[tuple(indices)]
        labels = np.where(new_mask > 0, filled - 1, 0).astype("int32")
        n = int(labels.max()) + 1
        stats = np.zeros((n, 5), dtype="int32")
        centers = np.zeros((n, 2))
        for j in range(1, n):
            yy, xx = np.where(labels == j)
            stats[j, cv2.CC_STAT_AREA] = len(xx)
            if len(xx):
                centers[j] = [xx.mean(), yy.mean()]
    safe = (
        cv2.erode(
            common.astype("uint8"),
            np.ones((9, 9), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    detections = []
    bright = new_rgb.mean(axis=2) > 140
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 80:
            continue
        component = labels == i
        overlap = float(prior[component].mean())
        old_mean = float(old_prob[component].mean())
        new_mean = float(new_prob[component].mean())
        if (
            overlap > 0.04
            or old_mean > 0.04
            or new_mean < 0.80
            or not safe[component].all()
        ):
            continue
        near = cv2.dilate(component.astype("uint8"), np.ones((19, 19), np.uint8)) > 0
        if bright[near].mean() < 0.02:
            continue
        geoms = [
            shape(g)
            for g, v in shapes(component.astype("uint8"), mask=component)
            if v == 1
        ]
        for geom in geoms:
            poly = geom.simplify(1.2, preserve_topology=True)
            if not isinstance(poly, Polygon) or poly.area < 80:
                continue
            coords = [
                [round(float(x), 2), round(float(y), 2)]
                for x, y in poly.exterior.coords
            ]
            detections.append(
                {
                    "classification": "new",
                    "evidence": "segmentation",
                    "building_score": round(new_mean, 4),
                    "earlier_building_score": round(old_mean, 4),
                    "polygon": coords,
                    "area": {"value": round(poly.area * pixel_area, 2), "unit": unit},
                    "centroid": [round(poly.centroid.x, 2), round(poly.centroid.y, 2)],
                }
            )
    if row_probability is not None:
        buffered_prior = cv2.dilate(
            (row_prior >= 0.35).astype("uint8"),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        )
        for poly in attached_roofs(new_rgb, row_probability, buffered_prior, common):
            if any(
                poly.intersection(Polygon(d["polygon"])).area
                > 0.35 * min(poly.area, Polygon(d["polygon"]).area)
                for d in detections
            ):
                continue
            detections.append(
                {
                    "classification": "new",
                    "evidence": "attached_roof_heuristic",
                    "polygon": [
                        [round(x, 2), round(y, 2)] for x, y in poly.exterior.coords
                    ],
                    "centroid": [round(poly.centroid.x, 2), round(poly.centroid.y, 2)],
                    "area": {"value": round(poly.area * pixel_area, 2), "unit": unit},
                }
            )
    detections.sort(key=lambda d: (d["centroid"][1], d["centroid"][0]))
    for index, detection in enumerate(detections, start=1):
        detection["id"] = f"N{index:02d}"
    return detections


def recover_low_light_roofs(detections, old_prob, raw_prob, enhanced_prob,
                            enhanced_boundary, common, rgb, pixel_area=1, unit="px²"):
    """Add independently supported enhanced roofs without merging existing instances.

    Use raw RGB for the brightness guard: boosting brightness there admitted
    dark vegetation and water in the sample. Original model support is required
    so enhancement alone cannot create a building.
    """
    recovered = classify_footprints(
        old_prob, enhanced_prob, common, rgb, pixel_area, unit,
        boundary=enhanced_boundary,
    )
    footprints = [Polygon(d["polygon"]) for d in detections]
    for candidate in recovered:
        poly = Polygon(candidate["polygon"])
        if any(poly.intersection(other).area > 0.25 * min(poly.area, other.area)
               for other in footprints):
            continue
        mask = np.zeros(common.shape, np.uint8)
        cv2.fillPoly(mask, [np.rint(candidate["polygon"]).astype("int32")], 1)
        support = float(raw_prob[mask > 0].mean())
        if support < 0.5:
            continue
        candidate["evidence"] = "contrast_recovery"
        candidate["original_building_score"] = round(support, 4)
        detections.append(candidate)
        footprints.append(poly)
    detections.sort(key=lambda d: (d["centroid"][1], d["centroid"][0]))
    for index, detection in enumerate(detections, start=1):
        detection["id"] = f"N{index:02d}"
    return detections


def result_key(old_path, new_path):
    digest = hashlib.sha256(VERSION.encode() + Path(__file__).read_bytes())
    for p in [old_path, new_path, settings.MODEL_PATH]:
        if p.is_file():
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()[:20]


def detect(old_path, new_path):
    # Validate before looking at cached results.
    old, ov, om = read_image(old_path)
    new, nv, nm = read_image(new_path)
    key = result_key(old_path, new_path)
    dest = settings.RESULTS_DIR / key
    with LOCK:
        if (dest / "result.json").is_file():
            return json.loads((dest / "result.json").read_text())
        aligned, common, alignment = align_images(old, ov, om, new, nv, nm)
        op, _ = predict(aligned)
        np_, nb = predict(new)
        enhanced_old, _ = predict(contrast_image(aligned))
        enhanced_new, enhanced_boundary = predict(contrast_image(new))
        metric = nm["crs"].is_projected and str(nm["crs"].linear_units).lower() in (
            "metre",
            "meter",
        )
        pa = abs(nm["transform"].determinant) if metric else 1
        detections = classify_footprints(
            np.maximum(op, enhanced_old),
            np_,
            common,
            new,
            pa,
            "m²" if metric else "px²",
            boundary=nb,
            row_probability=np.maximum(np_, enhanced_new),
            row_prior=np.maximum(op, enhanced_old),
        )
        detections = recover_low_light_roofs(
            detections, np.maximum(op, enhanced_old), np_, enhanced_new,
            enhanced_boundary, common, new, pa, "m²" if metric else "px²",
        )
        dest.mkdir(parents=True, exist_ok=True)
        for name, a in [("earlier", aligned), ("latest", new)]:
            rgba = np.dstack([a, common.astype("uint8") * 255])
            Image.fromarray(rgba).save(dest / f"{name}.png")
        result = {
            "result_id": key,
            "new_building_count": len(detections),
            "image_dimensions": {"width": nm["width"], "height": nm["height"]},
            "coordinate_system": "[x, y] pixels on latest image, origin top-left; closed exterior rings",
            "preview_urls": {
                n: f"/results/{key}/{n}.png" for n in ["earlier", "latest"]
            },
            "alignment": alignment,
            "warnings": ["Candidate buildings require visual review; scores are not calibrated confidence."]
            + ([alignment["note"]] if not alignment["inliers"] else [])
            + (["Some attached roofs were inferred from repeated roof edges."]
               if any(d["evidence"] == "attached_roof_heuristic" for d in detections) else []),
            "result_url": f"/results/{key}/result.json",
            "detections": detections,
        }
        # Publish the cache manifest last and atomically. A separate CLI process
        # must never observe a half-written JSON response.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dest, prefix="result-", suffix=".tmp", delete=False
        ) as manifest:
            json.dump(result, manifest, indent=2)
            pending_manifest = Path(manifest.name)
        pending_manifest.replace(dest / "result.json")
        return result
