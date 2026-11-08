import io, json, re, logging, tempfile
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse, FileResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from .services.detection import resolve_image, read_image, detect, InputError, ModelError
from PIL import Image

logger = logging.getLogger(__name__)


def index(request):
    return render(request, "buildings/index.html")


@csrf_exempt
@require_POST
def detect_changes(request):
    try:
        if request.content_type == "multipart/form-data":
            # Reject large browser uploads before Django parses the multipart body.
            limit = settings.MAX_IMAGE_UPLOAD_BYTES
            if int(request.META.get("CONTENT_LENGTH") or 0) > 2 * limit + 65536:
                return JsonResponse({"error": "Upload is too large. Maximum 50 MB per image."}, status=413)
            with tempfile.TemporaryDirectory(prefix="atlas-upload-") as directory:
                paths = []
                for field in ("old_image", "new_image"):
                    uploads = request.FILES.getlist(field)
                    if len(uploads) != 1:
                        raise InputError("Select exactly one old image and one new image.")
                    upload = uploads[0]
                    if Path(upload.name).suffix.lower() != ".tif":
                        raise InputError("Only .tif files are accepted.")
                    if upload.size > limit:
                        return JsonResponse({"error": "Upload is too large. Maximum 50 MB per image."}, status=413)
                    # Never use the client filename as a filesystem path.
                    path = Path(directory) / f"{field}.tif"
                    with path.open("wb") as output:
                        for chunk in upload.chunks():
                            output.write(chunk)
                    paths.append(path)
                return JsonResponse(detect(*paths))
        if request.content_type != "application/json":
            return JsonResponse(
                {"error": "Send application/json paths or multipart/form-data TIFF files."}, status=415,
            )
        body = request.read(8193)
        if len(body) > 8192:
            return JsonResponse({"error": "Request body is too large."}, status=413)
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise InputError("Request body must be a JSON object.")
        old = resolve_image(payload.get("old_image_path"))
        new = resolve_image(payload.get("new_image_path"))
        return JsonResponse(detect(old, new))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse(
            {"error": "Invalid JSON. Provide old_image_path and new_image_path."},
            status=400,
        )
    except InputError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except ModelError as e:
        return JsonResponse({"error": str(e)}, status=503)
    except Exception:
        logger.exception("Detection failed")
        return JsonResponse(
            {"error": "Analysis failed. See the local server log for details."},
            status=500,
        )


def result_file(request, key, name):
    if not re.fullmatch(r"[0-9a-f]{20}", key) or name not in {
        "earlier.png",
        "latest.png",
        "result.json",
    }:
        return HttpResponse(status=404)
    path = settings.RESULTS_DIR / key / name
    if not path.is_file():
        return HttpResponse(status=404)
    return FileResponse(
        path.open("rb"),
        content_type="application/json" if name.endswith("json") else "image/png",
    )


def sample(request, role):
    if role not in ["earlier", "latest"]:
        return HttpResponse(status=404)
    try:
        rgb, _, _ = read_image(
            resolve_image(
                "past_sample.tif" if role == "earlier" else "current_sample (1).tif"
            )
        )
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        buf.seek(0)
        return FileResponse(buf, content_type="image/png")
    except InputError as e:
        return JsonResponse({"error": str(e)}, status=400)
