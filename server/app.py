#!/usr/bin/env python3
"""
app.py - simple Flask web frontend to upload images and run infer.py's model.

Behavior:
 - loads model once at startup (AttentionUNet loaded via infer.py helpers)
 - /         GET: show upload form
 - /predict  POST: accept image file, run inference, return page showing results (overlay + masks)
 - static results are saved under ./static/results/<uuid>/

Important:
 - Keep infer.py in the same directory so we can import helpers.
"""
import os
import uuid
from pathlib import Path
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename

# import the model + helpers from your infer.py
from infer import AttentionUNet, load_checkpoint, build_transforms, infer_single, colorize_mask

# configuration via environment variables (override when running container)
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pth")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "1024"))
DEVICE = os.environ.get("DEVICE", "cpu")  # "cpu" or "cuda"
MEAN = tuple(float(x) for x in os.environ.get("MEAN", "0.485,0.456,0.406").split(","))
STD  = tuple(float(x) for x in os.environ.get("STD", "0.229,0.224,0.225").split(","))
BASE = int(os.environ.get("BASE", "32"))
N_CLASSES = int(os.environ.get("N_CLASSES", "3"))

UPLOAD_FOLDER = Path("uploads")
RESULTS_FOLDER = Path("static") / "results"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me-for-production")
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB limit (adjust)

# load model on startup
print("Loading model:", MODEL_PATH)
device_str = DEVICE if ("cuda" in DEVICE and __import__("torch").cuda.is_available()) else "cpu"
device = __import__("torch").device(device_str)
model = AttentionUNet(in_ch=3, n_classes=N_CLASSES, base=BASE).to(device)
model = load_checkpoint(model, MODEL_PATH, device)
model.eval()
print("Model loaded and ready on", device)

# build transforms once
tf_vis, tf_model = build_transforms(IMG_SIZE, mean=MEAN, std=STD)


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part in request")
        return redirect(url_for("index"))
    f = request.files["image"]
    if f.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))
    if not allowed_file(f.filename):
        flash("Unsupported file type. Use jpg/png/tiff.")
        return redirect(url_for("index"))

    filename = secure_filename(f.filename)
    uid = uuid.uuid4().hex[:8]
    save_dir = UPLOAD_FOLDER / uid
    save_dir.mkdir(parents=True, exist_ok=True)
    input_path = save_dir / filename
    f.save(str(input_path))

    # create result dir
    out_dir = RESULTS_FOLDER / uid
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # run inference (infer_single from infer.py)
        # note: infer_single signature used here:
        ov, pred = infer_single(model, str(input_path), tf_vis, tf_model, device,
                                mean=MEAN, std=STD, use_fp16=False, alpha=0.4, legend=True,
                                labels=None)
    except Exception as e:
        flash(f"Inference failed: {e}")
        return redirect(url_for("index"))

    # save outputs
    stem = Path(filename).stem
    overlay_out = out_dir / f"{stem}_overlay.png"
    mask_color_out = out_dir / f"{stem}_mask_color.png"
    mask_raw_out = out_dir / f"{stem}_mask_raw.png"

    import cv2
    cv2.imwrite(str(overlay_out), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_color_out), cv2.cvtColor(colorize_mask(pred), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_raw_out), pred.astype("uint8"))

    # render result page showing images
    return render_template("result.html",
                           uid=uid,
                           input_image=url_for("uploaded_file", uid=uid, filename=filename),
                           overlay_image=url_for("result_file", uid=uid, filename=overlay_out.name),
                           mask_color_image=url_for("result_file", uid=uid, filename=mask_color_out.name),
                           mask_raw_image=url_for("result_file", uid=uid, filename=mask_raw_out.name))


@app.route("/uploads/<uid>/<filename>")
def uploaded_file(uid, filename):
    return send_from_directory(UPLOAD_FOLDER / uid, filename)


@app.route("/results/<uid>/<filename>")
def result_file(uid, filename):
    return send_from_directory(RESULTS_FOLDER / uid, filename)


if __name__ == "__main__":
    # development server (production: use gunicorn)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
