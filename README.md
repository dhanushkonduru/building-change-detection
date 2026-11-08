# Building Change Detection

A Django application that compares old and new aerial GeoTIFF images to identify likely newly constructed buildings.

Select two `.tif` files or provide image paths through the REST API. The app displays both images side by side, highlights detected buildings with polygons, and returns building counts, areas, and downloadable JSON results.

Building segmentation uses [HOTOSM DINOv3-S](https://huggingface.co/hotosm/dinov3s-buildings) by HOTOSM and Kshitij Raj Sharma, built on DINOv3. Model license notices are in `models/licenses/`.
