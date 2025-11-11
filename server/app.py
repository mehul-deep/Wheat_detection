#!/usr/bin/env python3
"""
app.py - simple Flask web frontend to upload images and run infer.py's model.

Behavior:
 - loads model once at startup (AttentionUNet loaded via infer.py helpers)
 - /         GET: show upload form (requires authentication)
 - /predict  POST: accept image file, run inference, return page showing results (overlay + masks)
 - static results are saved under ./static/results/<uuid>/

Important:
 - Keep infer.py in the same directory so we can import helpers.
 - Firebase Authentication required for all routes except /login
"""
import math
import os
import shutil
import tempfile
import uuid
from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional
from functools import wraps

import numpy as np
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore

try:
    from ultralytics import YOLO
except ImportError:  # optional dependency
    YOLO = None

try:
    from google.cloud import storage as gcs_storage
except ImportError:  # optional dependency for GCS deployments
    gcs_storage = None

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
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "best.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.45"))
YOLO_MAX_DET = int(os.environ.get("YOLO_MAX_DET", "100"))
YOLO_IMG_SIZE = int(os.environ.get("YOLO_IMG_SIZE", str(IMG_SIZE)))
YOLO_DISEASE_CLASS = int(os.environ.get("YOLO_DISEASE_CLASS", "2"))
YOLO_WHEAT_CLASSES = tuple(int(x.strip()) for x in os.environ.get("YOLO_WHEAT_CLASSES", "1,2").split(",") if x.strip())

STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local").strip().lower()
LOCAL_UPLOAD_DIR = Path(os.environ.get("LOCAL_UPLOAD_DIR", "uploads"))
LOCAL_RESULTS_DIR = Path(os.environ.get("LOCAL_RESULTS_DIR", "static/results"))
WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/tmp/wheat_work"))
GCS_UPLOAD_BUCKET = os.environ.get("GCS_UPLOAD_BUCKET")
GCS_RESULTS_BUCKET = os.environ.get("GCS_RESULTS_BUCKET", GCS_UPLOAD_BUCKET)
GCS_UPLOAD_PREFIX = os.environ.get("GCS_UPLOAD_PREFIX", "uploads/")
GCS_RESULTS_PREFIX = os.environ.get("GCS_RESULTS_PREFIX", "results/")
GCS_MAKE_PUBLIC = os.environ.get("GCS_MAKE_PUBLIC", "true").strip().lower() in {"1", "true", "yes"}
GCS_SIGNED_URL_TTL = int(os.environ.get("GCS_SIGNED_URL_TTL", "3600"))

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WORK_ROOT.mkdir(parents=True, exist_ok=True)


class StorageBackend:
    """Abstract interface for persistence backends."""

    def save_upload(self, uid: str, filename: str, local_path: Path) -> str:
        raise NotImplementedError

    def save_result(self, uid: str, filename: str, local_path: Path) -> str:
        raise NotImplementedError


class LocalStorage(StorageBackend):
    """Persist files under the local filesystem for dev / docker-compose usage."""

    def __init__(self, upload_root: Path, result_root: Path):
        self.upload_root = upload_root
        self.result_root = result_root
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _move(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    def save_upload(self, uid: str, filename: str, local_path: Path) -> str:
        dest = self.upload_root / uid / filename
        self._move(local_path, dest)
        return url_for("uploaded_file", uid=uid, filename=filename)

    def save_result(self, uid: str, filename: str, local_path: Path) -> str:
        dest = self.result_root / uid / filename
        self._move(local_path, dest)
        return url_for("result_file", uid=uid, filename=filename)


class GCSStorage(StorageBackend):
    """Persist files inside Google Cloud Storage buckets."""

    def __init__(
        self,
        upload_bucket: str,
        results_bucket: str,
        upload_prefix: str = "uploads/",
        results_prefix: str = "results/",
        make_public: bool = True,
        signed_url_ttl: int = 3600,
    ):
        if gcs_storage is None:
            raise RuntimeError("google-cloud-storage is not installed; cannot use GCS backend.")
        if not upload_bucket or not results_bucket:
            raise ValueError("GCS upload and results buckets must be provided.")
        self.client = gcs_storage.Client()
        self.upload_bucket = self.client.bucket(upload_bucket)
        self.results_bucket = self.client.bucket(results_bucket)
        self.upload_prefix = self._normalize_prefix(upload_prefix)
        self.results_prefix = self._normalize_prefix(results_prefix)
        self.make_public = make_public
        self.signed_url_ttl = signed_url_ttl
        # Only get signing credentials if we need signed URLs (not making public)
        self._signing_credentials = None if make_public else self._get_signing_credentials()

    @staticmethod
    def _normalize_prefix(prefix: Optional[str]) -> str:
        if not prefix:
            return ""
        prefix = prefix.strip("/")
        return f"{prefix}/" if prefix else ""

    def _get_signing_credentials(self):
        """Return service-account credentials that support signing signed URLs."""
        from google.auth import default as google_auth_default
        from google.oauth2 import service_account

        # If running on Cloud Run default compute credentials (no private key),
        # look for GOOGLE_APPLICATION_CREDENTIALS pointing to a service account key.
        creds, _ = google_auth_default()
        if hasattr(creds, "sign_bytes") and callable(getattr(creds, "sign_bytes", None)):
            return creds

        key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if key_path and Path(key_path).exists():
            return service_account.Credentials.from_service_account_file(key_path)

        raise RuntimeError(
            "Cannot generate signed URLs without a service account key. "
            "Set GOOGLE_APPLICATION_CREDENTIALS to a JSON key file or disable signed URLs "
            "by setting GCS_MAKE_PUBLIC=true."
        )

    def _upload_and_url(self, bucket, prefix: str, uid: str, filename: str, local_path: Path) -> str:
        blob_path = f"{prefix}{uid}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_path))
        if self.make_public:
            blob.make_public()
            return blob.public_url
        return blob.generate_signed_url(
            expiration=timedelta(seconds=self.signed_url_ttl),
            credentials=self._signing_credentials,
        )

    def save_upload(self, uid: str, filename: str, local_path: Path) -> str:
        return self._upload_and_url(self.upload_bucket, self.upload_prefix, uid, filename, local_path)

    def save_result(self, uid: str, filename: str, local_path: Path) -> str:
        return self._upload_and_url(self.results_bucket, self.results_prefix, uid, filename, local_path)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me-for-production")
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB limit (adjust)

# Initialize Firebase Admin SDK
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "wheat-detection-cb988")
print(f"[FIREBASE] Initializing Firebase Admin SDK for project: {FIREBASE_PROJECT_ID}")
try:
    # Try to use default credentials (works on Cloud Run)
    firebase_admin.initialize_app(options={'projectId': FIREBASE_PROJECT_ID})
    print("[FIREBASE] Firebase Admin initialized with default credentials ✓")
except Exception as e:
    print(f"[FIREBASE] Initialization with defaults failed: {e}")
    # Fallback: try loading from service account file
    firebase_creds_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
    if firebase_creds_path and Path(firebase_creds_path).exists():
        cred = credentials.Certificate(firebase_creds_path)
        firebase_admin.initialize_app(cred)
        print(f"[FIREBASE] Firebase Admin initialized with credentials from {firebase_creds_path} ✓")
    else:
        print("[FIREBASE] ⚠️  WARNING: Firebase Admin not initialized - authentication will not work")

# Get Firestore client
print(f"[FIRESTORE] Initializing Firestore client for project: {FIREBASE_PROJECT_ID}...")
try:
    # Try using firebase_admin's firestore client first
    from google.cloud import firestore as gcp_firestore
    from google.auth import default

    # Get default credentials
    credentials, project = default()
    print(f"[FIRESTORE] Using credentials for project: {project}")

    # Initialize Firestore with explicit project
    db = gcp_firestore.Client(project=FIREBASE_PROJECT_ID, credentials=credentials)
    print("[FIRESTORE] Firestore client initialized ✓")
except Exception as e:
    print(f"[FIRESTORE] ⚠️  Firestore initialization failed: {e}")
    print("[FIRESTORE] Will run without Firestore (history won't be saved)")
    import traceback
    traceback.print_exc()
    db = None

# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

if STORAGE_BACKEND == "gcs":
    storage_backend = GCSStorage(
        upload_bucket=GCS_UPLOAD_BUCKET,
        results_bucket=GCS_RESULTS_BUCKET,
        upload_prefix=GCS_UPLOAD_PREFIX,
        results_prefix=GCS_RESULTS_PREFIX,
        make_public=GCS_MAKE_PUBLIC,
        signed_url_ttl=GCS_SIGNED_URL_TTL,
    )
    print(f"Using GCS storage (uploads={GCS_UPLOAD_BUCKET}, results={GCS_RESULTS_BUCKET}).")
else:
    storage_backend = LocalStorage(LOCAL_UPLOAD_DIR, LOCAL_RESULTS_DIR)
    print(f"Using local storage under {LOCAL_UPLOAD_DIR.resolve()} and {LOCAL_RESULTS_DIR.resolve()}.")

# load model on startup
print("Loading model:", MODEL_PATH)
device_str = DEVICE if ("cuda" in DEVICE and __import__("torch").cuda.is_available()) else "cpu"
device = __import__("torch").device(device_str)
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", device_str)
model = AttentionUNet(in_ch=3, n_classes=N_CLASSES, base=BASE).to(device)
model = load_checkpoint(model, MODEL_PATH, device)
model.eval()
print("Model loaded and ready on", device)

# build transforms once
tf_vis, tf_model = build_transforms(IMG_SIZE, mean=MEAN, std=STD)

yolo_model = None
yolo_class_names = {}
if YOLO is None:
    print("ultralytics not installed; YOLO detections disabled.")
else:
    yolo_path = Path(YOLO_MODEL_PATH)
    if not yolo_path.exists():
        print(f"YOLO model not found at {yolo_path.resolve()}; detections disabled.")
    else:
        try:
            yolo_model = YOLO(str(yolo_path))
            yolo_class_names = getattr(yolo_model, "names", {})
            print(f"Loaded YOLO model from {yolo_path}.")
        except Exception as exc:
            print(f"Failed to load YOLO model: {exc}")
            yolo_model = None


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT


@app.route("/login", methods=["GET"])
def login():
    """Show login/signup page"""
    return render_template("login.html")


@app.route("/test", methods=["GET", "POST"])
def test():
    """Test endpoint to verify server is responding"""
    print("[TEST] Test endpoint called")
    return jsonify({'status': 'ok', 'message': 'Server is working!'}), 200


@app.route("/auth/verify", methods=["POST"])
def verify_token():
    """Verify Firebase ID token and create session"""
    try:
        print(f"[AUTH] Received verify request")
        print(f"[AUTH] Request data: {request.json}")

        id_token = request.json.get('idToken')
        display_name = request.json.get('displayName')

        if not id_token:
            print("[AUTH] ERROR: No token provided")
            return jsonify({'error': 'No token provided'}), 400

        print(f"[AUTH] Verifying token (length: {len(id_token)})")

        # Verify the token with Firebase Admin SDK
        decoded_token = firebase_auth.verify_id_token(id_token)
        user_id = decoded_token['uid']
        user_email = decoded_token.get('email')

        print(f"[AUTH] Token verified successfully for user: {user_id}")

        # Create or update user document in Firestore
        if db and display_name:
            try:
                print(f"[AUTH] Creating/updating user document in Firestore for {user_id}")
                user_ref = db.collection('users').document(user_id)
                user_doc = user_ref.get()

                if not user_doc.exists:
                    # New user - create document
                    user_ref.set({
                        'email': user_email,
                        'displayName': display_name,
                        'uploadCount': 0,
                        'createdAt': datetime.utcnow().isoformat(),
                        'lastLogin': datetime.utcnow().isoformat()
                    })
                    print(f"[AUTH] Created new user document for {user_id}")
                else:
                    # Existing user - update last login
                    user_ref.update({'lastLogin': datetime.utcnow().isoformat()})
                    print(f"[AUTH] Updated last login for {user_id}")
            except Exception as firestore_error:
                print(f"[AUTH] Warning: Firestore update failed: {firestore_error}")
                # Don't fail the auth if Firestore fails

        # Store user_id in session
        session['user_id'] = user_id
        session['email'] = user_email
        session['name'] = display_name or decoded_token.get('name', user_email.split('@')[0] if user_email else 'User')

        print(f"[AUTH] Session created for user: {user_id}")

        return jsonify({'status': 'success', 'user_id': user_id}), 200

    except Exception as e:
        print(f"[AUTH] ERROR: Token verification failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Invalid token', 'details': str(e)}), 401


@app.route("/auth/logout", methods=["POST"])
def logout():
    """Clear session"""
    session.clear()
    return jsonify({'status': 'success'}), 200


@app.route("/", methods=["GET"])
@require_auth
def index():
    return render_template("index_auth.html",
                           config_img_size=IMG_SIZE,
                           yolo_enabled=bool(yolo_model),
                           yolo_model_path=YOLO_MODEL_PATH if yolo_model else None)


@app.route("/predict", methods=["GET", "POST"])
@require_auth
def predict():
    print(f"[PREDICT] Request method: {request.method}")
    print(f"[PREDICT] Session user_id: {session.get('user_id')}")
    print(f"[PREDICT] Request headers: {dict(request.headers)}")

    # If GET request, redirect to index
    if request.method == "GET":
        print("[PREDICT] GET request received - redirecting to index")
        flash("Please upload an image to get predictions")
        return redirect(url_for("index"))

    user_id = session.get('user_id')
    if not user_id:
        print("[PREDICT] No user_id in session!")
        return jsonify({'error': 'Unauthorized'}), 401

    if "image" not in request.files:
        print("[PREDICT] No image in request.files")
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
    work_dir = Path(tempfile.mkdtemp(prefix=f"{uid}_", dir=str(WORK_ROOT)))
    input_path = work_dir / filename
    f.save(str(input_path))

    try:
        try:
            # run inference (infer_single from infer.py)
            ov, pred = infer_single(model, str(input_path), tf_vis, tf_model, device,
                                    mean=MEAN, std=STD, use_fp16=False, alpha=0.4, legend=True,
                                    labels=None)
        except Exception as e:
            flash(f"Inference failed: {e}")
            return redirect(url_for("index"))

        stem = Path(filename).stem
        overlay_out = work_dir / f"{stem}_overlay.png"
        mask_color_out = work_dir / f"{stem}_mask_color.png"
        yolo_out = work_dir / f"{stem}_yolo_det.png"

        import cv2
        cv2.imwrite(str(overlay_out), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_color_out), cv2.cvtColor(colorize_mask(pred), cv2.COLOR_RGB2BGR))

        yolo_written = False
        yolo_detections = []
        yolo_error = None
        if yolo_model:
            try:
                results = yolo_model.predict(source=str(input_path),
                                             conf=YOLO_CONF,
                                             iou=YOLO_IOU,
                                             imgsz=YOLO_IMG_SIZE,
                                             max_det=YOLO_MAX_DET,
                                             device=YOLO_DEVICE,
                                             verbose=False)
                if results:
                    res0 = results[0]
                    annotated = res0.plot()  # BGR array
                    cv2.imwrite(str(yolo_out), annotated)
                    yolo_written = True
                    boxes = getattr(res0, "boxes", None)
                    orig_shape = getattr(res0, "orig_shape", None)
                    if isinstance(orig_shape, (tuple, list)) and len(orig_shape) >= 2:
                        orig_h, orig_w = int(orig_shape[0]), int(orig_shape[1])
                    else:
                        orig_img = getattr(res0, "orig_img", None)
                        if orig_img is not None:
                            orig_h, orig_w = orig_img.shape[:2]
                        else:
                            tmp = cv2.imread(str(input_path))
                            orig_h, orig_w = tmp.shape[:2] if tmp is not None else (None, None)

                    mask_h, mask_w = pred.shape
                    scale = None
                    pad_top = pad_left = None
                    if orig_h and orig_w:
                        scale = min(mask_h / orig_h, mask_w / orig_w)
                        scaled_h = orig_h * scale
                        scaled_w = orig_w * scale
                        pad_y = max(mask_h - scaled_h, 0.0)
                        pad_x = max(mask_w - scaled_w, 0.0)
                        pad_top = pad_y / 2.0
                        pad_left = pad_x / 2.0

                    wheat_class_set = set(YOLO_WHEAT_CLASSES)
                    if boxes is not None and boxes.xyxy is not None:
                        xyxy = boxes.xyxy.cpu().tolist()
                        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
                        classes = boxes.cls.cpu().tolist() if boxes.cls is not None else []
                        for idx, coords in enumerate(xyxy):
                            label_idx = int(classes[idx]) if idx < len(classes) else None
                            confidence = float(confs[idx]) if idx < len(confs) else None
                            label = yolo_class_names.get(label_idx, str(label_idx)) if label_idx is not None else "unknown"

                            detection = {
                                "label": label,
                                "confidence": confidence,
                                "bbox": [float(v) for v in coords],
                                "wheat_pixels": None,
                                "disease_pixels": None,
                                "patch_pixels": None,
                                "disease_fraction_bbox": None,
                                "disease_fraction_wheat": None,
                                "wheat_fraction_bbox": None,
                            }

                            if scale and pad_left is not None and pad_top is not None:
                                x1, y1, x2, y2 = coords
                                x1s = int(math.floor(x1 * scale + pad_left))
                                y1s = int(math.floor(y1 * scale + pad_top))
                                x2s = int(math.ceil(x2 * scale + pad_left))
                                y2s = int(math.ceil(y2 * scale + pad_top))
                                x1s = max(0, min(mask_w - 1, x1s))
                                x2s = max(0, min(mask_w, x2s))
                                y1s = max(0, min(mask_h - 1, y1s))
                                y2s = max(0, min(mask_h, y2s))
                                if x2s > x1s and y2s > y1s:
                                    mask_patch = pred[y1s:y2s, x1s:x2s]
                                    if mask_patch.size > 0:
                                        patch_pixels = int(mask_patch.size)
                                        wheat_pixels = int(np.isin(mask_patch, list(wheat_class_set)).sum()) if wheat_class_set else 0
                                        disease_pixels = int((mask_patch == YOLO_DISEASE_CLASS).sum())
                                        detection["patch_pixels"] = patch_pixels
                                        detection["wheat_pixels"] = wheat_pixels
                                        detection["disease_pixels"] = disease_pixels
                                        detection["disease_fraction_bbox"] = (disease_pixels / patch_pixels) if patch_pixels else None
                                        detection["wheat_fraction_bbox"] = (wheat_pixels / patch_pixels) if patch_pixels else None
                                        detection["disease_fraction_wheat"] = (disease_pixels / wheat_pixels) if wheat_pixels else None

                            yolo_detections.append(detection)
            except Exception as exc:
                yolo_error = str(exc)

        input_image_url = storage_backend.save_upload(uid, filename, input_path)
        overlay_image_url = storage_backend.save_result(uid, overlay_out.name, overlay_out)
        mask_color_image_url = storage_backend.save_result(uid, mask_color_out.name, mask_color_out)
        yolo_image_url = None
        if yolo_written and yolo_out.exists():
            yolo_image_url = storage_backend.save_result(uid, yolo_out.name, yolo_out)

        # Save to Firestore
        print(f"[FIRESTORE] About to save upload. db={db is not None}, user_id={user_id}, uid={uid}")
        if db:
            try:
                print(f"[FIRESTORE] Inside db block, attempting to save...")
                upload_data = {
                    'userId': user_id,
                    'uid': uid,
                    'filename': filename,
                    'timestamp': datetime.utcnow().isoformat(),
                    'detectionCount': len(yolo_detections) if yolo_detections else 0,
                    'yolo_detections': yolo_detections,
                    'inputImageUrl': input_image_url,
                    'overlayImageUrl': overlay_image_url,
                    'maskImageUrl': mask_color_image_url,
                    'yoloImageUrl': yolo_image_url,
                    'hasError': bool(yolo_error)
                }
                print(f"[FIRESTORE] Calling db.collection('uploads').document({uid}).set()")
                db.collection('uploads').document(uid).set(upload_data)
                print(f"[FIRESTORE] ✓ Saved upload {uid} to Firestore for user {user_id}")

                # Update user stats
                from google.cloud.firestore import Increment
                user_ref = db.collection('users').document(user_id)
                user_doc = user_ref.get()
                if user_doc.exists:
                    user_ref.update({
                        'uploadCount': Increment(1),
                        'lastUpload': datetime.utcnow().isoformat()
                    })
                else:
                    user_ref.set({
                        'email': session.get('email', ''),
                        'displayName': session.get('name', ''),
                        'uploadCount': 1,
                        'createdAt': datetime.utcnow().isoformat(),
                        'lastUpload': datetime.utcnow().isoformat()
                    })
                print(f"[FIRESTORE] ✓ User stats updated for {user_id}")
            except Exception as e:
                print(f"[FIRESTORE] ❌ Failed to save to Firestore: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[FIRESTORE] ⚠️  db is None, skipping Firestore save")

        # Render result page directly (session storage doesn't work - too large)
        return render_template("result.html",
                               uid=uid,
                               input_image=input_image_url,
                               overlay_image=overlay_image_url,
                               mask_color_image=mask_color_image_url,
                               config_img_size=IMG_SIZE,
                               yolo_image=yolo_image_url,
                               yolo_detections=yolo_detections,
                               yolo_error=yolo_error)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.route("/result/<uid>")
@require_auth
def view_result(uid):
    """View a previous result by UID"""
    user_id = session.get('user_id')

    # Try session first (for fresh results or when Firestore is unavailable)
    session_key = f'result_{uid}'
    if session_key in session:
        print(f"[RESULT] Loading result from session for uid: {uid}")
        data = session[session_key]
        return render_template("result.html",
                               uid=uid,
                               input_image=data.get('input_image'),
                               overlay_image=data.get('overlay_image'),
                               mask_color_image=data.get('mask_color_image'),
                               config_img_size=IMG_SIZE,
                               yolo_image=data.get('yolo_image'),
                               yolo_detections=data.get('yolo_detections', []),
                               yolo_error=data.get('yolo_error'))

    # Fall back to Firestore for historical results
    if not db:
        flash("Result not found")
        return redirect(url_for("index"))

    try:
        print(f"[RESULT] Loading result from Firestore for uid: {uid}")
        doc = db.collection('uploads').document(uid).get()
        if not doc.exists:
            flash("Result not found")
            return redirect(url_for("index"))

        data = doc.to_dict()
        # Verify this upload belongs to the current user
        if data.get('userId') != user_id:
            flash("Unauthorized access")
            return redirect(url_for("index"))

        return render_template("result.html",
                               uid=uid,
                               input_image=data.get('inputImageUrl'),
                               overlay_image=data.get('overlayImageUrl'),
                               mask_color_image=data.get('maskImageUrl'),
                               config_img_size=IMG_SIZE,
                               yolo_image=data.get('yoloImageUrl'),
                               yolo_detections=data.get('yolo_detections', []),
                               yolo_error=None)
    except Exception as e:
        print(f"Error loading result: {e}")
        flash("Result not found")
        return redirect(url_for("index"))


@app.route("/uploads/<uid>/<filename>")
def uploaded_file(uid, filename):
    return send_from_directory(LOCAL_UPLOAD_DIR / uid, filename)


@app.route("/results/<uid>/<filename>")
def result_file(uid, filename):
    return send_from_directory(LOCAL_RESULTS_DIR / uid, filename)


if __name__ == "__main__":
    # development server (production: use gunicorn)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
