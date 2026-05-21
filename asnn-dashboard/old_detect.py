#!/usr/bin/env python3
"""
ASNN Detection Script — Ultralytics-compatible preprocessing & post-processing.

Key fix: ASNN tensor order is NOT assumed. Each tensor's stride is derived
from its actual element count (size / LISTSIZE → grid cells → sqrt → grid_h/w
→ stride = IMGSZ / grid_h).

Preprocessing  : letterbox resize to IMGSZ×IMGSZ (preserves aspect ratio, grey pad)
Post-processing: auto-stride DFL decode → xyxy → scale back to original image coords
NMS            : per-class IoU NMS identical to Ultralytics non_max_suppression()

Each stdout line:
  {
    "frame": N, "fps": F, "inference_ms": T,
    "detections": [{"class_id":0,"class_name":"Car","score":0.92,
                    "box":[x1,y1,x2,y2]},...],   ← absolute pixels, original image
    "jpeg": "<base64 JPEG of the original frame>"
  }
"""

import numpy as np
import os, sys, json, base64, time, threading, queue, argparse, math
import cv2 as cv

try:
    from asnn.api import asnn
    from asnn.types import output_format
    ASNN_AVAILABLE = True
except ImportError:
    ASNN_AVAILABLE = False
    print(json.dumps({"type": "log", "level": "warn",
          "message": "asnn not found — running simulation mode"}), flush=True)

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--library")
parser.add_argument("--model")
parser.add_argument("--type",         default="usb")
parser.add_argument("--device",       default="0")
parser.add_argument("--level",        default="0")
parser.add_argument("--obj-thresh",   type=float, default=0.25)
parser.add_argument("--nms-thresh",   type=float, default=0.45)
parser.add_argument("--platform",     default="ONNX")
parser.add_argument("--jpeg-quality", type=int,   default=75)
parser.add_argument("--classes",      nargs="+",  default=None)
parser.add_argument("--num-cls",      type=int,   default=None)
parser.add_argument("--listsize",     type=int,   default=None)
parser.add_argument("--imgsz",        type=int,   default=640)
args = parser.parse_args()

OBJ_THRESH   = args.obj_thresh
NMS_THRESH   = args.nms_thresh
JPEG_QUALITY = max(10, min(95, args.jpeg_quality))
IMGSZ        = args.imgsz

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# DFL reg_max: YOLOv8/v9/v10 standard = 16 → listsize = nc + 64
REG_MAX = 16

# ── Load class names ───────────────────────────────────────────────────────────
CLASSES  = None
NUM_CLS  = 1
LISTSIZE = NUM_CLS + REG_MAX * 4

if args.classes:
    CLASSES  = tuple(args.classes)
    NUM_CLS  = len(CLASSES)
    LISTSIZE = NUM_CLS + REG_MAX * 4
elif args.model:
    model_dir = os.path.dirname(os.path.abspath(args.model))
    for yn in ['data.yaml', 'dataset.yaml']:
        yp = os.path.join(model_dir, yn)
        if os.path.exists(yp):
            try:
                import yaml
                with open(yp) as f:
                    yd = yaml.safe_load(f)
                names = yd.get('names', [])
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names.keys())]
                if names:
                    CLASSES  = tuple(n.strip() for n in names)
                    NUM_CLS  = len(CLASSES)
                    LISTSIZE = NUM_CLS + REG_MAX * 4
                    print(json.dumps({"type": "log", "level": "info",
                          "message": f"Loaded {NUM_CLS} classes from {yn}"}), flush=True)
                break
            except Exception as e:
                print(json.dumps({"type": "log", "level": "warn",
                      "message": f"YAML error: {e}"}), flush=True)

if CLASSES is None:
    CLASSES  = ("Object",)
    NUM_CLS  = 1
    LISTSIZE = NUM_CLS + REG_MAX * 4

if args.num_cls:
    NUM_CLS  = args.num_cls
    LISTSIZE = NUM_CLS + REG_MAX * 4
if args.listsize:
    LISTSIZE = args.listsize

# DFL weight vector
DFL_PROJ = np.arange(REG_MAX, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LETTERBOX  (Ultralytics-identical)
# ══════════════════════════════════════════════════════════════════════════════
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """
    Resize + pad to new_shape while preserving aspect ratio.
    Returns: (padded_img, scale_ratio, (pad_w_half, pad_h_half))
    """
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)

    new_unpad = (int(round(w * r)), int(round(h * r)))   # (new_w, new_h)
    dw = new_shape[1] - new_unpad[0]   # total width  padding
    dh = new_shape[0] - new_unpad[1]   # total height padding
    dw /= 2;  dh /= 2

    if (w, h) != new_unpad:
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)

    top    = int(round(dh - 0.1));  bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1));  right  = int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right,
                            cv.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(img, imgsz=640):
    """
    Letterbox → BGR→RGB → /255 → CHW float32.
    Returns (blob_CHW, ratio, pad_wh)
    """
    padded, ratio, pad = letterbox(img, new_shape=imgsz)
    rgb  = cv.cvtColor(padded, cv.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)          # HWC → CHW
    return blob, ratio, pad


# ══════════════════════════════════════════════════════════════════════════════
# 2.  AUTO-DETECT TENSOR STRIDE FROM SIZE
# ══════════════════════════════════════════════════════════════════════════════
def tensor_stride(raw_flat, listsize, imgsz):
    """
    Given a flat float32 array from one ASNN output head, determine its stride.

    Assumption: raw_flat.size == listsize * gh * gw  where gh == gw == imgsz//stride.
    Solve: grid_cells = size / listsize  →  grid_side = sqrt(grid_cells)
           stride = imgsz / grid_side

    Raises ValueError if the size doesn't factor cleanly.
    """
    n = raw_flat.size
    if n % listsize != 0:
        raise ValueError(
            f"Tensor size {n} is not divisible by listsize {listsize}. "
            f"Check --listsize / --num-cls."
        )
    grid_cells = n // listsize
    grid_side  = int(math.isqrt(grid_cells))
    if grid_side * grid_side != grid_cells:
        raise ValueError(
            f"grid_cells={grid_cells} is not a perfect square (tensor size={n}, "
            f"listsize={listsize}). Non-square feature maps are not supported."
        )
    if imgsz % grid_side != 0:
        raise ValueError(
            f"imgsz={imgsz} is not divisible by grid_side={grid_side}."
        )
    stride = imgsz // grid_side
    return stride, grid_side


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DFL + DIST2BBOX
# ══════════════════════════════════════════════════════════════════════════════
def dfl_decode(reg):
    """
    reg : (N, 4*REG_MAX)
    Returns (N, 4) — [l, t, r, b] distances in stride units
    """
    n = reg.shape[0]
    reg = reg.reshape(n, 4, REG_MAX)            # (N, 4, 16)
    # Softmax over last axis
    reg = reg - reg.max(axis=-1, keepdims=True)
    reg = np.exp(reg)
    reg /= reg.sum(axis=-1, keepdims=True)
    # Weighted sum → distance
    dist = (reg * DFL_PROJ).sum(axis=-1)        # (N, 4)
    return dist


def dist2xyxy(dist, anchors):
    """
    dist    : (N, 4)  [l, t, r, b] in stride units
    anchors : (N, 2)  [cx, cy]     in stride units
    Returns : (N, 4)  [x1, y1, x2, y2] in stride units
    """
    x1y1 = anchors - dist[:, :2]
    x2y2 = anchors + dist[:, 2:]
    return np.concatenate([x1y1, x2y2], axis=-1)


def make_anchors(grid_side):
    """
    Anchor centre points at half-pixel offset (0.5, 1.5, …) in stride units.
    Returns (grid_side*grid_side, 2)
    """
    g = np.arange(grid_side, dtype=np.float32) + 0.5
    gy, gx = np.meshgrid(g, g, indexing='ij')
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SINGLE-HEAD DECODE
# ══════════════════════════════════════════════════════════════════════════════
def decode_head(raw_flat, listsize, imgsz, ratio, pad):
    """
    Decode one ASNN output tensor completely.

    raw_flat : 1-D float32 numpy array (any size, stride auto-detected)
    Returns  : (boxes_px, scores, class_ids) — absolute pixels, original image
               or (None, None, None) if no detections pass threshold.
    """
    # --- figure out stride & grid size ---
    stride, gs = tensor_stride(raw_flat, listsize, imgsz)

    # --- reshape: ASNN stores as [SPAN, LISTSIZE, GH, GW] ---
    head = raw_flat.reshape(1, listsize, gs, gs)    # (1, ls, gh, gw)
    head = head.transpose(0, 2, 3, 1)               # (1, gh, gw, ls)
    head = head.reshape(gs * gs, listsize)           # (A, ls)

    # --- split cls / reg ---
    cls_raw  = head[:, :NUM_CLS]                            # (A, nc)
    reg_raw  = head[:, NUM_CLS : NUM_CLS + REG_MAX * 4]    # (A, 64)

    # --- sigmoid class scores ---
    cls_scores = 1.0 / (1.0 + np.exp(-cls_raw))            # (A, nc)

    # --- confidence filter (max class score across all classes) ---
    max_scores = cls_scores.max(axis=-1)                    # (A,)
    keep = max_scores >= OBJ_THRESH
    if not keep.any():
        return None, None, None

    cls_scores = cls_scores[keep]
    reg_raw    = reg_raw[keep]

    # --- DFL → dist → xyxy in stride units ---
    dist    = dfl_decode(reg_raw)                           # (A', 4)
    anchors = make_anchors(gs)[keep]                        # (A', 2)
    boxes_s = dist2xyxy(dist, anchors)                      # (A', 4) stride units

    # --- scale to letterboxed pixel coords ---
    boxes_px = boxes_s * stride                             # (A', 4) letterbox px

    # --- remove padding, undo scale ratio → original image pixels ---
    pad_w, pad_h = pad
    boxes_px[:, [0, 2]] -= pad_w
    boxes_px[:, [1, 3]] -= pad_h
    boxes_px /= ratio
    boxes_px  = np.maximum(boxes_px, 0.0)

    best_cls   = cls_scores.argmax(axis=-1)
    best_score = cls_scores.max(axis=-1)

    return boxes_px, best_score, best_cls


# ══════════════════════════════════════════════════════════════════════════════
# 5.  NMS
# ══════════════════════════════════════════════════════════════════════════════
def nms_boxes(boxes, scores):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0];  keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[np.where(iou <= NMS_THRESH)[0] + 1]
    return keep


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FULL POST-PROCESS  (all heads → merge → NMS → format)
# ══════════════════════════════════════════════════════════════════════════════
def yolo_postprocess(data, ratio, pad, imgsz=640):
    """
    data  : list/tuple of N flat float32 arrays from ASNN (any order — auto-sorted)
    ratio : letterbox scale ratio
    pad   : (pad_w_half, pad_h_half)

    Returns list of detection dicts sorted by score descending.
    """
    all_boxes, all_scores, all_cls = [], [], []

    for i, raw in enumerate(data):
        try:
            b, s, c = decode_head(raw, LISTSIZE, imgsz, ratio, pad)
        except ValueError as e:
            print(json.dumps({"type": "log", "level": "warn",
                  "message": f"Head {i} decode error: {e}"}), flush=True)
            continue
        if b is None:
            continue
        all_boxes.append(b);  all_scores.append(s);  all_cls.append(c)

    if not all_boxes:
        return []

    boxes   = np.concatenate(all_boxes,  axis=0)
    scores  = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_cls,    axis=0)

    # Per-class NMS
    final = []
    for c in np.unique(classes):
        idx  = np.where(classes == c)[0]
        keep = nms_boxes(boxes[idx], scores[idx])
        for ki in keep:
            b = boxes[idx[ki]]
            final.append({
                "class_id":   int(c),
                "class_name": CLASSES[int(c)] if int(c) < len(CLASSES) else "Unknown",
                "score":      round(float(scores[idx[ki]]), 4),
                "box":        [round(float(b[0]), 1), round(float(b[1]), 1),
                               round(float(b[2]), 1), round(float(b[3]), 1)]
            })

    final.sort(key=lambda d: d["score"], reverse=True)
    return final


# ══════════════════════════════════════════════════════════════════════════════
# 7.  JPEG ENCODE
# ══════════════════════════════════════════════════════════════════════════════
def encode_jpeg(img):
    h, w = img.shape[:2]
    if w > 640:
        img = cv.resize(img, (640, int(h * 640 / w)))
    ok, buf = cv.imencode('.jpg', img, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode('utf-8') if ok else None


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SIMULATION (no ASNN)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_dets(h=480, w=640):
    import random
    count = random.randint(0, 3) if random.random() > 0.35 else 0
    dets  = []
    for _ in range(count):
        cl = random.randint(0, NUM_CLS - 1)
        x1 = random.uniform(10, w * 0.5)
        y1 = random.uniform(10, h * 0.5)
        x2 = min(x1 + random.uniform(30, w * 0.35), w - 1)
        y2 = min(y1 + random.uniform(30, h * 0.35), h - 1)
        dets.append({
            "class_id":   cl,
            "class_name": CLASSES[cl],
            "score":      round(random.uniform(0.50, 0.97), 3),
            "box":        [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        })
    return dets


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CAPTURE THREAD
# ══════════════════════════════════════════════════════════════════════════════
frame_q = queue.Queue(maxsize=2)

def capture_frames(cap_type, cap_device):
    if cap_type == "usb":
        cap = cv.VideoCapture(int(cap_device), cv.CAP_V4L2)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    elif cap_type == "rtsp":
        url = f"{cap_device}?tcp&buffer_size=0&fflags=nobuffer&flags=low_delay"
        cap = cv.VideoCapture(url, cv.CAP_FFMPEG)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    elif cap_type == "video":
        cap = cv.VideoCapture(cap_device)
    elif cap_type == "mipi":
        pipeline = (f"v4l2src device=/dev/video{cap_device} io-mode=dmabuf !"
                    "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 !"
                    "videoconvert ! appsink")
        cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    else:
        print(json.dumps({"type": "log", "level": "err",
              "message": f"Unsupported source type: {cap_type}"}), flush=True)
        sys.exit(1)

    if not cap.isOpened():
        print(json.dumps({"type": "log", "level": "err",
              "message": f"Cannot open: {cap_type}:{cap_device}"}), flush=True)
        sys.exit(1)

    skip = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if cap_type == "video":
                break
            continue
        if cap_type in ("usb", "rtsp", "mipi"):
            skip += 1
            if skip % 2 != 0:
                continue
        try:
            frame_q.put_nowait(frame)
        except queue.Full:
            pass
    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# 10.  INFERENCE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
# Populated on first inference call to log tensor shapes once
_shapes_logged = False

def run_inference(nn, img):
    global _shapes_logged

    t0 = time.time()

    blob, ratio, pad = preprocess(img, imgsz=IMGSZ)

    data = nn.nn_inference(
        [blob],
        platform=args.platform,
        reorder='2 1 0',
        output_tensor=3,
        output_format=output_format.OUT_FORMAT_FLOAT32
    )

    # Log tensor sizes once so the operator can verify ordering
    if not _shapes_logged:
        shapes = [f"data[{i}]: size={data[i].size} ({data[i].size // LISTSIZE} cells)" for i in range(len(data))]
        print(json.dumps({"type": "log", "level": "info",
              "message": "ASNN tensor sizes: " + ", ".join(shapes)}), flush=True)
        _shapes_logged = True

    # Pass ALL tensors — stride is auto-detected per tensor from its size
    dets = yolo_postprocess(list(data), ratio, pad, imgsz=IMGSZ)

    inf_ms = round((time.time() - t0) * 1000, 2)
    return dets, inf_ms


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cap_type = args.type
    cap_dev  = args.device
    level    = int(args.level) if args.level in ['1', '2'] else 0

    if ASNN_AVAILABLE:
        for label, path_ in [("model", args.model), ("library", args.library)]:
            if not path_ or not os.path.exists(path_):
                print(json.dumps({"type": "log", "level": "err",
                      "message": f"{label} not found: {path_}"}), flush=True)
                sys.exit(1)

    print(json.dumps({"type": "log", "level": "info",
          "message": (f"type={cap_type} device={cap_dev} imgsz={IMGSZ} "
                      f"obj_thresh={OBJ_THRESH} nms_thresh={NMS_THRESH} "
                      f"nc={NUM_CLS} listsize={LISTSIZE} jpeg_q={JPEG_QUALITY}")}),
          flush=True)

    nn = None
    if ASNN_AVAILABLE:
        nn = asnn('Electron')
        nn.nn_init(library=args.library, model=args.model, level=level)
        print(json.dumps({"type": "log", "level": "info",
              "message": "Neural network initialised"}), flush=True)

    # ── Single image mode ──────────────────────────────────────────────────────
    if cap_type == "image":
        img = cv.imread(cap_dev)
        if img is None:
            print(json.dumps({"type": "log", "level": "err",
                  "message": f"Cannot read image: {cap_dev}"}), flush=True)
            sys.exit(1)
        if nn:
            dets, inf_ms = run_inference(nn, img)
        else:
            dets, inf_ms = simulate_dets(img.shape[0], img.shape[1]), 0.0
        print(json.dumps({
            "frame": 1, "fps": 1.0, "inference_ms": inf_ms,
            "detections": dets, "jpeg": encode_jpeg(img)
        }), flush=True)
        return

    # ── Video / Webcam / RTSP ──────────────────────────────────────────────────
    threading.Thread(target=capture_frames, args=(cap_type, cap_dev), daemon=True).start()
    print(json.dumps({"type": "log", "level": "info",
          "message": "Capture thread started"}), flush=True)

    frame_num = 0
    fps_count = 0
    fps_val   = 0.0
    fps_start = time.time()

    while True:
        try:
            orig = frame_q.get(timeout=3)
        except queue.Empty:
            print(json.dumps({"type": "log", "level": "warn",
                  "message": "Frame timeout — waiting for camera..."}), flush=True)
            continue

        frame_num += 1

        if nn:
            dets, inf_ms = run_inference(nn, orig)
        else:
            dets   = simulate_dets(orig.shape[0], orig.shape[1])
            inf_ms = round(float(np.random.uniform(8, 25)), 2)

        fps_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_val   = round(fps_count / elapsed, 2)
            fps_count = 0
            fps_start = time.time()

        print(json.dumps({
            "frame":        frame_num,
            "fps":          fps_val,
            "inference_ms": inf_ms,
            "detections":   dets,
            "jpeg":         encode_jpeg(orig)
        }), flush=True)


if __name__ == '__main__':
    main()

