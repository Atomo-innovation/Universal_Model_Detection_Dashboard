#!/usr/bin/env python3
"""
ASNN Detection Dashboard Script
Preprocessing & post-processing taken directly from mitesh4.py (proven working).

Pipeline:
  BGR→RGB → letterbox(scale=target/max(h,w), pad=114, centre) →
  subtract mean / divide var → CHW float32 → ASNN inference →
  process() [DFL→normalised xyxy] → filter_boxes → per-class NMS →
  unletterbox_boxes (×640 → remove pad → ÷scale → clip) → absolute pixels

Output per stdout line (JSON):
  { frame, fps, inference_ms,
    detections: [{class_id, class_name, score, box:[x1,y1,x2,y2]}],
    jpeg, orig_w, orig_h, jpeg_w, jpeg_h }
  box coords = absolute pixels in the ORIGINAL frame.
"""

import numpy as np
import os, sys, json, base64, time, threading, queue, argparse
import cv2 as cv

try:
    from asnn.api import asnn
    from asnn.types import output_format
    ASNN_AVAILABLE = True
except ImportError:
    ASNN_AVAILABLE = False
    print(json.dumps({"type":"log","level":"warn",
          "message":"asnn not found — running simulation mode"}), flush=True)

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--library")
parser.add_argument("--model")
parser.add_argument("--type",         default="usb")
parser.add_argument("--device",       default="0")
parser.add_argument("--level",        default="0")
parser.add_argument("--obj-thresh",   type=float, default=0.35)
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
TARGET       = args.imgsz      # letterbox target size

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ── Constants (from mitesh4.py) ────────────────────────────────────────────────
GRID0    = TARGET // 32   # 20  for 640
GRID1    = TARGET // 16   # 40
GRID2    = TARGET // 8    # 80

mean = [0, 0, 0]
var  = [255]

constant_matrix = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]).T

# ── Load class names ───────────────────────────────────────────────────────────
CLASSES  = None
NUM_CLS  = 1
LISTSIZE = NUM_CLS + 64   # nc + 4*16

if args.classes:
    CLASSES  = tuple(args.classes)
    NUM_CLS  = len(CLASSES)
    LISTSIZE = NUM_CLS + 64
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
                    LISTSIZE = NUM_CLS + 64
                    print(json.dumps({"type":"log","level":"info",
                          "message":f"Loaded {NUM_CLS} classes from {yn}: {', '.join(CLASSES[:10])}{'...' if NUM_CLS>10 else ''}"}), flush=True)
                break
            except Exception as e:
                print(json.dumps({"type":"log","level":"warn",
                      "message":f"YAML error: {e}"}), flush=True)

if CLASSES is None:
    CLASSES  = ("Object",)
    NUM_CLS  = 1
    LISTSIZE = NUM_CLS + 64

if args.num_cls:
    NUM_CLS  = args.num_cls
    LISTSIZE = NUM_CLS + 64
if args.listsize:
    LISTSIZE = args.listsize

SPAN = 1   # batch size = 1

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LETTERBOX  — from mitesh4.py
# ══════════════════════════════════════════════════════════════════════════════
def letterbox(frame, target=640):
    """
    Scale so the longest side = target, pad remainder with 114 (centre).
    Returns: (canvas, scale, x0, y0)
      scale : resize ratio (applied to original dims)
      x0,y0 : left/top padding in pixels
    """
    h, w    = frame.shape[:2]
    scale   = target / max(h, w)
    nh, nw  = int(h * scale), int(w * scale)
    resized = cv.resize(frame, (nw, nh))
    canvas  = np.full((target, target, 3), 114, dtype=np.uint8)
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas, scale, x0, y0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  UNLETTERBOX  — from mitesh4.py
# ══════════════════════════════════════════════════════════════════════════════
def unletterbox_boxes(boxes, scale, x0, y0, orig_w, orig_h, target=640):
    """
    boxes : (N,4) normalised [0,1] xyxy from process()
    Returns (N,4) absolute pixels clipped to orig_w×orig_h
    """
    boxes = boxes.copy()
    # 1. scale back to letterboxed pixel coords
    boxes[:, 0] *= target;  boxes[:, 2] *= target
    boxes[:, 1] *= target;  boxes[:, 3] *= target
    # 2. remove padding offset
    boxes[:, 0] -= x0;  boxes[:, 2] -= x0
    boxes[:, 1] -= y0;  boxes[:, 3] -= y0
    # 3. undo scale → original image pixels
    boxes /= scale
    # 4. clip
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREPROCESS  — from mitesh4.py
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(orig_bgr, target=640):
    """
    BGR→RGB → letterbox → subtract mean → /var → CHW float32
    Returns (blob, scale, x0, y0)
    """
    rgb = cv.cvtColor(orig_bgr, cv.COLOR_BGR2RGB)
    lb, scale, x0, y0 = letterbox(rgb, target=target)

    img = lb.astype(np.float32)
    img[:, :, 0] -= mean[0]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[2]
    img /= var[0]
    img = img.transpose(2, 0, 1)   # HWC → CHW

    return img, scale, x0, y0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DFL / GRID DECODE  — from mitesh4.py (process + helpers)
# ══════════════════════════════════════════════════════════════════════════════
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=0):
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)

def process(inp):
    """
    inp : (grid_h, grid_w, 1, LISTSIZE)
    Returns (result, box_class_probs)
      result          : (grid_h, grid_w, 1, 4)  normalised [0,1] xyxy
      box_class_probs : (grid_h, grid_w, 1, NUM_CLS)  sigmoid class scores
    """
    grid_h, grid_w = inp.shape[0], inp.shape[1]

    box_class_probs = sigmoid(inp[..., :NUM_CLS])

    box_0 = softmax(inp[..., NUM_CLS:      NUM_CLS+16], -1)
    box_1 = softmax(inp[..., NUM_CLS+16:   NUM_CLS+32], -1)
    box_2 = softmax(inp[..., NUM_CLS+32:   NUM_CLS+48], -1)
    box_3 = softmax(inp[..., NUM_CLS+48:   NUM_CLS+64], -1)

    result = np.zeros((grid_h, grid_w, 1, 4))
    result[..., 0] = np.dot(box_0, constant_matrix)[..., 0]
    result[..., 1] = np.dot(box_1, constant_matrix)[..., 0]
    result[..., 2] = np.dot(box_2, constant_matrix)[..., 0]
    result[..., 3] = np.dot(box_3, constant_matrix)[..., 0]

    col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
    col = col.reshape(grid_h, grid_w, 1, 1)
    row = row.reshape(grid_h, grid_w, 1, 1)
    grid = np.concatenate((col, row), axis=-1)

    result[..., 0:2] = (0.5 - result[..., 0:2] + grid) / (grid_w, grid_h)
    result[..., 2:4] = (0.5 + result[..., 2:4] + grid) / (grid_w, grid_h)

    return result, box_class_probs


def filter_boxes(boxes, box_class_probs):
    box_classes      = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs,    axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)
    return boxes[pos], box_classes[pos], box_class_scores[pos]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  NMS  — from mitesh4.py
# ══════════════════════════════════════════════════════════════════════════════
def nms_boxes(boxes, scores):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0];  keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w1  = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h1  = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w1 * h1
        ovr   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= NMS_THRESH)[0] + 1]
    return np.array(keep)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FULL POST-PROCESS  — from mitesh4.py (yolov3_post_process)
# ══════════════════════════════════════════════════════════════════════════════
def yolov3_post_process(input_data):
    """
    input_data : list of 3 arrays already transposed to (gh, gw, 1, LISTSIZE)
                 order: [GRID0=20, GRID1=40, GRID2=80]
    Returns (boxes_norm, scores, classes) — boxes are normalised [0,1]
            or (None, None, None) if no detections.
    """
    boxes, classes, scores = [], [], []
    for i in range(3):
        result, confidence = process(input_data[i])
        b, c, s = filter_boxes(result, confidence)
        boxes.append(b);  classes.append(c);  scores.append(s)

    boxes   = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores  = np.concatenate(scores)

    if len(boxes) == 0:
        return None, None, None

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, cc, s = boxes[inds], classes[inds], scores[inds]
        keep = nms_boxes(b, s)
        if keep.size:
            nboxes.append(b[keep]);  nclasses.append(cc[keep]);  nscores.append(s[keep])

    if not nclasses:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nscores), np.concatenate(nclasses)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FORMAT DETECTIONS  (convert to dashboard JSON format)
# ══════════════════════════════════════════════════════════════════════════════
def format_detections(boxes_px, scores, classes):
    """boxes_px already in absolute original-image pixels."""
    dets = []
    for box, score, cl in zip(boxes_px, scores, classes):
        x1, y1, x2, y2 = box
        dets.append({
            "class_id":   int(cl),
            "class_name": CLASSES[int(cl)] if int(cl) < len(CLASSES) else "Unknown",
            "score":      round(float(score), 4),
            "box":        [round(float(x1), 1), round(float(y1), 1),
                           round(float(x2), 1), round(float(y2), 1)]
        })
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


# ══════════════════════════════════════════════════════════════════════════════
# 8.  JPEG ENCODE  (resized to ≤640, returns orig + jpeg dims for JS scaling)
# ══════════════════════════════════════════════════════════════════════════════
def encode_jpeg(img):
    """
    Returns (b64_str, orig_w, orig_h, jpeg_w, jpeg_h)
    Box coords are in orig_w/orig_h space.
    """
    orig_h, orig_w = img.shape[:2]
    if orig_w > 640:
        jpeg_w = 640
        jpeg_h = int(orig_h * 640 / orig_w)
        out = cv.resize(img, (jpeg_w, jpeg_h))
    else:
        jpeg_w, jpeg_h = orig_w, orig_h
        out = img
    ok, buf = cv.imencode('.jpg', out, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None, orig_w, orig_h, jpeg_w, jpeg_h
    return base64.b64encode(buf).decode('utf-8'), orig_w, orig_h, jpeg_w, jpeg_h


# ══════════════════════════════════════════════════════════════════════════════
# 9.  INFERENCE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
_shapes_logged = False

def run_inference(nn, orig_bgr):
    """
    Full pipeline matching mitesh4.py exactly.
    Returns (dets, inf_ms)  where dets = list of detection dicts.
    """
    global _shapes_logged
    t0 = time.time()

    orig_h, orig_w = orig_bgr.shape[:2]

    # --- Preprocess (BGR→RGB → letterbox → normalise → CHW) ---
    blob, scale, x0, y0 = preprocess(orig_bgr, target=TARGET)

    # --- ASNN forward pass ---
    data = nn.nn_inference(
        [blob],
        platform=args.platform,
        reorder='2 1 0',
        output_tensor=3,
        output_format=output_format.OUT_FORMAT_FLOAT32
    )

    # Log tensor shapes once
    if not _shapes_logged:
        info = [f"data[{i}]: {data[i].size} elements" for i in range(len(data))]
        print(json.dumps({"type":"log","level":"info",
              "message":"Tensor sizes: " + ", ".join(info)}), flush=True)
        _shapes_logged = True

    # --- Reshape exactly as in mitesh4.py ---
    # data[2] → GRID0=20×20, data[1] → GRID1=40×40, data[0] → GRID2=80×80
    input0_data = data[2].reshape(SPAN, LISTSIZE, GRID0, GRID0)
    input1_data = data[1].reshape(SPAN, LISTSIZE, GRID1, GRID1)
    input2_data = data[0].reshape(SPAN, LISTSIZE, GRID2, GRID2)

    input_data = [
        np.transpose(input0_data, (2, 3, 0, 1)),   # (20,20,1,LISTSIZE)
        np.transpose(input1_data, (2, 3, 0, 1)),   # (40,40,1,LISTSIZE)
        np.transpose(input2_data, (2, 3, 0, 1)),   # (80,80,1,LISTSIZE)
    ]

    # --- Post-process (normalised xyxy) ---
    boxes_norm, scores, classes = yolov3_post_process(input_data)

    dets = []
    if boxes_norm is not None:
        # --- Unletterbox → absolute original-image pixels ---
        boxes_px = unletterbox_boxes(
            boxes_norm, scale, x0, y0, orig_w, orig_h, target=TARGET)
        dets = format_detections(boxes_px, scores, classes)

    inf_ms = round((time.time() - t0) * 1000, 2)
    return dets, inf_ms


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SIMULATION (no ASNN)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_dets(h=480, w=640):
    import random
    count = random.randint(0, 3) if random.random() > 0.35 else 0
    dets  = []
    for _ in range(count):
        cl = random.randint(0, NUM_CLS - 1)
        x1 = random.uniform(10, w * 0.5);  y1 = random.uniform(10, h * 0.5)
        x2 = min(x1 + random.uniform(30, w * 0.35), w - 1)
        y2 = min(y1 + random.uniform(30, h * 0.35), h - 1)
        dets.append({
            "class_id":   cl,
            "class_name": CLASSES[cl],
            "score":      round(random.uniform(0.5, 0.97), 3),
            "box":        [round(x1,1), round(y1,1), round(x2,1), round(y2,1)]
        })
    return dets


# ══════════════════════════════════════════════════════════════════════════════
# 11.  CAPTURE THREAD
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
        print(json.dumps({"type":"log","level":"err",
              "message":f"Unsupported: {cap_type}"}), flush=True)
        sys.exit(1)

    if not cap.isOpened():
        print(json.dumps({"type":"log","level":"err",
              "message":f"Cannot open: {cap_type}:{cap_device}"}), flush=True)
        sys.exit(1)

    skip = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if cap_type == "video": break
            continue
        if cap_type in ("usb", "rtsp", "mipi"):
            skip += 1
            if skip % 2 != 0: continue
        try: frame_q.put_nowait(frame)
        except queue.Full: pass
    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# 12.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cap_type = args.type
    cap_dev  = args.device
    level    = int(args.level) if args.level in ['1','2'] else 0

    if ASNN_AVAILABLE:
        for label, path_ in [("model", args.model), ("library", args.library)]:
            if not path_ or not os.path.exists(path_):
                print(json.dumps({"type":"log","level":"err",
                      "message":f"{label} not found: {path_}"}), flush=True)
                sys.exit(1)

    print(json.dumps({"type":"log","level":"info",
          "message":(f"type={cap_type} device={cap_dev} target={TARGET} "
                     f"obj={OBJ_THRESH} nms={NMS_THRESH} "
                     f"nc={NUM_CLS} listsize={LISTSIZE} jpeg_q={JPEG_QUALITY}")}), flush=True)

    nn = None
    if ASNN_AVAILABLE:
        nn = asnn('Electron')
        nn.nn_init(library=args.library, model=args.model, level=level)
        print(json.dumps({"type":"log","level":"info",
              "message":"Neural network initialised"}), flush=True)

    # ── Image mode ─────────────────────────────────────────────────────────────
    if cap_type == "image":
        img = cv.imread(cap_dev)
        if img is None:
            print(json.dumps({"type":"log","level":"err",
                  "message":f"Cannot read: {cap_dev}"}), flush=True)
            sys.exit(1)
        if nn:
            dets, inf_ms = run_inference(nn, img)
        else:
            dets, inf_ms = simulate_dets(img.shape[0], img.shape[1]), 0.0
        jpeg_b64, orig_w, orig_h, jpeg_w, jpeg_h = encode_jpeg(img)
        print(json.dumps({"frame":1,"fps":1.0,"inference_ms":inf_ms,
                          "detections":dets,"jpeg":jpeg_b64,
                          "orig_w":orig_w,"orig_h":orig_h,
                          "jpeg_w":jpeg_w,"jpeg_h":jpeg_h}), flush=True)
        return

    # ── Stream / video mode ────────────────────────────────────────────────────
    threading.Thread(target=capture_frames, args=(cap_type, cap_dev), daemon=True).start()
    print(json.dumps({"type":"log","level":"info",
          "message":"Capture thread started"}), flush=True)

    frame_num = 0
    fps_count = 0
    fps_val   = 0.0
    fps_start = time.time()

    while True:
        try:
            orig = frame_q.get(timeout=3)
        except queue.Empty:
            print(json.dumps({"type":"log","level":"warn",
                  "message":"Frame timeout — waiting..."}), flush=True)
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

        jpeg_b64, orig_w, orig_h, jpeg_w, jpeg_h = encode_jpeg(orig)
        print(json.dumps({
            "frame":        frame_num,
            "fps":          fps_val,
            "inference_ms": inf_ms,
            "detections":   dets,
            "jpeg":         jpeg_b64,
            "orig_w":       orig_w,
            "orig_h":       orig_h,
            "jpeg_w":       jpeg_w,
            "jpeg_h":       jpeg_h
        }), flush=True)


if __name__ == '__main__':
    main()
