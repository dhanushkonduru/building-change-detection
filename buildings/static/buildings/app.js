const form = document.querySelector("#compare-form");
const button = document.querySelector("#compare");
const statusLine = document.querySelector("#status");
const footprints = document.querySelector("#footprints");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const useSample = event.submitter?.id === "sample-pair";
  if (!useSample) {
    for (const input of form.querySelectorAll('input[type="file"]')) {
      const file = input.files[0];
      if (!file || !/\.tif$/i.test(file.name) || file.size > 50 * 1024 * 1024) {
        statusLine.textContent = "Select two .tif images, up to 50 MB each.";
        return;
      }
    }
  }
  button.disabled = true;
  document.querySelector("#sample-pair").disabled = true;
  statusLine.textContent = "Comparing images…";
  document.querySelector("#download").hidden = true;
  document.querySelector("#details").textContent = "";
  document.querySelector("section").style.opacity = "0.45";
  try {
    const response = await fetch("/api/detect-changes/", {
      method: "POST",
      ...(useSample ? {
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({old_image_path: "past_sample.tif", new_image_path: "current_sample (1).tif"}),
      } : {body: new FormData(form)}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    const { width, height } = result.image_dimensions;
    await Promise.all(Object.values(result.preview_urls).map(url => new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = resolve;
      image.onerror = () => reject(new Error("Could not load result images. Please compare again."));
      image.src = url;
    })));
    footprints.replaceChildren();
    document.querySelector("#old-image").src = result.preview_urls.earlier;
    const latest = document.querySelector("#new-image");
    latest.setAttribute("href", result.preview_urls.latest);
    latest.setAttribute("width", width);
    latest.setAttribute("height", height);
    document.querySelector("#latest").setAttribute("viewBox", `0 0 ${width} ${height}`);
    for (const detection of result.detections) {
      const polygon = document.createElementNS(latest.namespaceURI, "polygon");
      polygon.setAttribute("points", detection.polygon.map(point => point.join(",")).join(" "));
      const title = document.createElementNS(latest.namespaceURI, "title");
      title.textContent = `${detection.id} · ${detection.area.value} ${detection.area.unit}`;
      polygon.append(title);
      footprints.append(polygon);
    }
    statusLine.textContent = `Candidate new buildings: ${result.new_building_count}`;
    document.querySelector("#alignment-label").textContent = "Aligned to new image";
    document.querySelector("#details").textContent =
      `${result.alignment.method} · ${(result.alignment.overlap_fraction * 100).toFixed(1)}% comparable coverage. ${result.warnings.join(" ")}`;
    const download = document.querySelector("#download");
    download.href = result.result_url;
    download.hidden = false;
    document.querySelector("section").style.opacity = "1";
  } catch (error) {
    statusLine.textContent = error.message || "Could not compare images.";
    document.querySelector("#details").textContent = "Images below are from the previous preview, not this failed comparison.";
  } finally {
    button.disabled = false;
    document.querySelector("#sample-pair").disabled = false;
  }
});
