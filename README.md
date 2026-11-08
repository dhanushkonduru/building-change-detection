# Building change detection

A local Django API compares two aerial images and returns polygons for likely newly constructed buildings. The comparison page shows the aligned old image beside the new image, with detected buildings outlined in cyan.

## Run

From the project directory, using Python 3.13:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python manage.py setup_model
.venv/bin/python manage.py runserver 127.0.0.1:8000
```

Open [the comparison page](http://127.0.0.1:8000/) and click **Try sample images** to run the supplied pair. The model setup downloads verified weights once. Inference runs locally on CPU. On Windows use the virtual environment's Scripts directory for the Python executable.

## API

```sh
curl http://127.0.0.1:8000/api/detect-changes/ \
  -H 'Content-Type: application/json' \
  -d '{"old_image_path":"past_sample.tif","new_image_path":"current_sample (1).tif"}'
```

Inputs must be overlapping 8-bit RGB GeoTIFFs with a .tif extension. Paths are relative to `data/samples/` by default. Set IMAGERY_DIR to choose another input directory. The API returns the new-building count, aligned preview URLs, image dimensions and detections with IDs, closed polygon rings, centroids and areas. Coordinates are [x, y] pixels on the latest image, measured from the top left. Areas use square metres for projected metre coordinates and square pixels otherwise.

The detector aligns the images on the latest grid, segments buildings in both dates and excludes footprints with earlier building evidence or weak latest evidence. Shadows, touching roofs, and differences in viewing angle can cause misses or inaccurate boundaries. See the model and license details below.

Run the API and detection tests with `.venv/bin/python manage.py test`.

## Demonstration and verification

1. Start the server and open the comparison page.
2. Select old and new TIFFs using **Choose File**, then select **Compare images**, or choose **Try sample images**. The first run performs CPU inference; subsequent identical requests use cached results.
3. Review the aligned old image against the cyan polygons on the latest image. Hover over a polygon for its ID and area. The status reports comparable coverage and any alignment warnings.
4. Use **Download result JSON** to inspect polygons, detection evidence and alignment metadata.

Run `.venv/bin/python manage.py verify_sample` for a real-model integration check. It verifies polygon validity, bounds, counts and result URLs, not detection accuracy against ground truth.

Only `.tif` filenames are accepted (case insensitive), and actual TIFF content is validated. This implementation requires **8-bit RGB GeoTIFFs with a CRS**; grayscale, 16-bit, unreferenced TIFFs and nonoverlapping images return a descriptive 400 error. Files must be inside `IMAGERY_DIR`, including when absolute paths are supplied. The JSON path API reads server-local paths; the UI uploads files selected on the browser's computer.

API errors: 400 invalid input or unsupported imagery, 405 non-POST request, 413 oversized body, 415 incorrect content type, 503 missing/unreadable model, 500 unexpected processing error. This is a local demonstration server; results are retained under `var/results` and can be deleted to reclaim space.

The virtual environment, generated results and model weights are excluded from version control. Install dependencies and run model setup after cloning.

## Select files from a folder

The UI opens your computer's file picker for each date. Select one old and one new `.tif`, then click **Compare images**. Each file may be up to 50 MB. Uploaded files are processed in a temporary directory and removed after the request; generated previews and JSON results remain available. GeoTIFF format requirements still apply.

The existing JSON path API is unchanged. The same endpoint also accepts multipart uploads:

```sh
curl http://127.0.0.1:8000/api/detect-changes/ \
  -F 'old_image=@data/samples/past_sample.tif' \
  -F 'new_image=@data/samples/current_sample (1).tif'
```

## Project structure

```text
config/                 Django settings, root URLs, ASGI and WSGI entry points
buildings/              Django application
  management/commands/  Model setup and sample verification commands
  services/             Image alignment, model inference and change detection
  static/buildings/     Stylesheet and browser JavaScript
  templates/buildings/  Comparison page
  tests/                API, upload and detection regression tests
data/samples/           Supplied old and new GeoTIFFs
models/                 Downloaded ONNX weights (ignored)
  licenses/             Required third-party license notices
manage.py               Django command-line entry point
requirements.txt        Pinned Python dependencies
README.md               Setup, API, model details and limitations
```

Runtime results are generated in `var/results/`. Run `python manage.py collectstatic` when preparing static assets for deployment; output goes to `staticfiles/`. This app does not store database records, so no database migrations are required.

## Segmentation model provenance

Atlas uses **HOTOSM DINOv3-S buildings**: frozen DINOv3 ViT-S/16 backbone with a trained UperNet decoder for very-high-resolution RGB aerial and satellite building footprints. The model was trained on `hotosm/vhr-building-segmentation`; it has not been fine-tuned on the supplied pair.

- [Published model card and training-data attribution](https://huggingface.co/hotosm/dinov3s-buildings)
- [Pinned release](https://huggingface.co/hotosm/dinov3s-buildings/tree/f7254545dfbbf10a40594d5a5fb62136902000ae)
- Remote file: `model.onnx`; local file: `models/hot-buildings.onnx` (219,574,600 bytes).
- SHA-256: `9a28e28403accc06c70a3104aedb31eca50401ed23c8ba4314b26357c25e9043`.
- [Author's inference implementation](https://github.com/kshitijrajsharma/dinov3-hot/blob/dee030eec385aca489be152eea2a96ddf255a2ef/src/dinov3_hot/serve.py) establishes the normalization, three independent output heads and Gaussian blending. Atlas implements the same input/output convention without requiring its training stack.

## License and attribution

Credit: Humanitarian OpenStreetMap Team (HOTOSM), Kshitij Raj Sharma and the DINOv3 authors. The published building weights are **CC BY 4.0**, with the training-data terms listed as CC BY 4.0 / ODbL. The backbone is subject to the **separate DINOv3 license**; see [the official DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). These sources, rather than a blanket license for this app, govern use of the weights. The full license notices are retained in `models/licenses/`.

`python manage.py setup_model` downloads the pinned public weights once and checks every byte against the release checksum. No account, key, paid API, training or hosted inference is used. ONNX Runtime runs on CPU with four threads. Missing weights produce a clear 503 response. The runtime dependencies do not include PyTorch.

## Exact preprocessing and output

Rasterio reads RGB bands 1–3 in their original order, as unsigned 8-bit values; alpha / nodata define valid coverage. Earlier imagery is reprojected to the latest grid before segmentation. Both original RGB and CLAHE-enhanced RGB enter inference. No manually traced masks enter inference.

1. Divide RGB by 255.
2. Normalize with HOT dataset mean `[0.4296737853453577, 0.4001659668453235, 0.34333372802741474]` and standard deviation `[0.2056069389373208, 0.16738555558380538, 0.1598986422586595]`.
3. Run full-resolution 256 × 256 windows with stride 192, reflect-padding only when an entire dimension is shorter than one tile. Blend overlap with a Gaussian kernel, sigma 32 pixels. The pinned ONNX input is `image`, float32 NCHW `[1,3,256,256]`.
4. Output `logits` has three **independent** channels: building-mask logit, boundary logit, signed distance. Apply sigmoid to channels 0 and 1; no softmax. Channel 2 is unused. The runtime contract was inspected directly, rather than treating these as three semantic classes.
5. Latest mask threshold: 0.50. Earlier evidence threshold: 0.35. A 3 × 3 opening removes isolated noise. Predicted boundaries separate cores of touching roofs; seeded watershed partitions the original mask. Tiny cores and components are filtered by documented pixel-area rules in the source.

The detector uses mask probabilities internally to filter uncertain footprints.


## Temporal decision and attached roofs

The earlier evidence is the maximum of original and contrast-enhanced predictions, dilated by 10 pixels to tolerate roof displacement. Candidates must have at most 4% prior overlap, earlier mean score at most 0.04, latest mean score at least 0.80, at least 80 pixels of area and valid coverage with a four-pixel boundary margin. A nearby bright-pixel filter suppresses dark texture, but can also miss dark roofs. These pixel thresholds depend on imagery resolution.

A separate heuristic proposes repeated attached roofs from long model-supported components and repeated image edges. These detections carry `evidence: attached_roof_heuristic`; other detections carry `evidence: segmentation`. Neither is ground truth. The model scores are not calibrated confidence. The API exposes registration details and warnings so failed visual alignment is visible.

This demonstration detects candidate additions. It does not label demolition, extensions, or every modification to existing buildings. Accuracy on the supplied pair has not been measured against independently annotated ground truth. Visual inspection remains necessary, particularly for repetitive fields, shadows, dark roofs and adjacent buildings.

## Low-light recovery

A second, independent segmentation pass uses CLAHE-enhanced latest RGB and its own boundary predictions to recover roofs missed in the original pass. It uses the same conservative earlier-image evidence, area, overlap and original-RGB brightness checks. Recovered polygons require mean original-image model support of at least 0.50, and are rejected when their intersection with an existing detection exceeds 25% of the smaller area. Keeping these passes separate prevents a large enhanced component from merging already separated row units. Added candidates have `evidence: contrast_recovery` and `original_building_score`; their `building_score` comes from enhanced imagery.

These checks reduce enhancement-only false positives; they do not establish calibrated accuracy. The detector is still resolution-sensitive and may miss dark or adjoining roofs.

