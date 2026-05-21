#!/usr/bin/env python3
"""
ASNN Detection Script — outputs JSON lines with embedded JPEG frames.
Patched: encode_jpeg now returns orig_w/orig_h/jpeg_w/jpeg_h so the
dashboard can correctly scale absolute-pixel bounding boxes onto the canvas.
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
    print(json.dumps({"type":"log","level":"warn",
          "message":"asnn not found — running simulation mode"}), flush=True)

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
REG_MAX      = 16

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ── Class names ────────────────────────────────────────────────────────────────
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
                    print(json.dumps({"type":"log","level":"info",
                          "message":f"Loaded {NUM_CLS} classes from {yn}"}), flush=True)
                break
            except Exception as e:
                print(json.dumps({"type":"log","level":"warn",
                      "message":f"YAML error: {e}"}), flush=True)

if CLASSES is None:
    CLASSES  = ("Object",)
    NUM_CLS  = 1
    LISTSIZE = NUM_CLS + REG_MAX * 4

if args.num_cls:
    NUM_CLS  = args.num_cls
    LISTSIZE = NUM_CLS + REG_MAX * 4
if args.listsize:
    LISTSIZE = args.listsize

DFL_PROJ = np.arange(REG_MAX, dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# LETTERBOX
# ══════════════════════════════════════════════════════════════════════════════
def letterbox(img, new_shape=640, color=(114,114,114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0]/h, new_shape[1]/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2;  dh /= 2
    if (w,h) != new_unpad:
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top    = int(round(dh-0.1)); bottom = int(round(dh+0.1))
    left   = int(round(dw-0.1)); right  = int(round(dw+0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right,
                            cv.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess(img, imgsz=640):
    padded, ratio, pad = letterbox(img, new_shape=imgsz)
    rgb  = cv.cvtColor(padded, cv.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2,0,1)
    return blob, ratio, pad

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-STRIDE + DFL DECODE
# ══════════════════════════════════════════════════════════════════════════════
def tensor_stride(raw_flat, listsize, imgsz):
    n = raw_flat.size
    if n % listsize != 0:
        raise ValueError(f"size {n} not divisible by listsize {listsize}")
    grid_cells = n // listsize
    gs = int(math.isqrt(grid_cells))
    if gs * gs != grid_cells:
        raise ValueError(f"grid_cells={grid_cells} not a perfect square")
    if imgsz % gs != 0:
        raise ValueError(f"imgsz={imgsz} not divisible by gs={gs}")
    return imgsz // gs, gs

def dfl_decode(reg):
    n = reg.shape[0]
    reg = reg.reshape(n, 4, REG_MAX)
    reg = reg - reg.max(axis=-1, keepdims=True)
    reg = np.exp(reg)
    reg /= reg.sum(axis=-1, keepdims=True)
    return (reg * DFL_PROJ).sum(axis=-1)

def make_anchors(gs):
    g = np.arange(gs, dtype=np.float32) + 0.5
    gy, gx = np.meshgrid(g, g, indexing='ij')
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)

def decode_head(raw_flat, listsize, imgsz, ratio, pad):
    stride, gs = tensor_stride(raw_flat, listsize, imgsz)
    head = raw_flat.reshape(1, listsize, gs, gs).transpose(0,2,3,1).reshape(gs*gs, listsize)

    cls_raw = head[:, :NUM_CLS]
    reg_raw = head[:, NUM_CLS : NUM_CLS + REG_MAX*4]

    cls_scores = 1.0 / (1.0 + np.exp(-cls_raw))
    max_scores = cls_scores.max(axis=-1)
    keep = max_scores >= OBJ_THRESH
    if not keep.any():
        return None, None, None

    cls_scores = cls_scores[keep]
    reg_raw    = reg_raw[keep]

    dist    = dfl_decode(reg_raw)
    anchors = make_anchors(gs)[keep]
    x1y1    = anchors - dist[:, :2]
    x2y2    = anchors + dist[:, 2:]
    boxes_s = np.concatenate([x1y1, x2y2], axis=-1)

    boxes_px = boxes_s * stride
    pad_w, pad_h = pad
    boxes_px[:, [0,2]] -= pad_w
    boxes_px[:, [1,3]] -= pad_h
    boxes_px /= ratio
    boxes_px  = np.maximum(boxes_px, 0.0)

    return boxes_px, cls_scores.max(axis=-1), cls_scores.argmax(axis=-1)

# ══════════════════════════════════════════════════════════════════════════════
# NMS
# ══════════════════════════════════════════════════════════════════════════════
def nms_boxes(boxes, scores):
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas  = np.maximum(0.0,x2-x1)*np.maximum(0.0,y2-y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0.0,xx2-xx1)*np.maximum(0.0,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-7)
        order=order[np.where(iou<=NMS_THRESH)[0]+1]
    return keep

# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESS
# ══════════════════════════════════════════════════════════════════════════════
def yolo_postprocess(data, ratio, pad, imgsz=640):
    all_boxes, all_scores, all_cls = [], [], []
    for i, raw in enumerate(data):
        try:
            b, s, c = decode_head(raw, LISTSIZE, imgsz, ratio, pad)
        except ValueError as e:
            print(json.dumps({"type":"log","level":"warn",
                  "message":f"Head {i}: {e}"}), flush=True)
            continue
        if b is None: continue
        all_boxes.append(b); all_scores.append(s); all_cls.append(c)

    if not all_boxes:
        return []

    boxes   = np.concatenate(all_boxes)
    scores  = np.concatenate(all_scores)
    classes = np.concatenate(all_cls)

    final = []
    for c in np.unique(classes):
        idx  = np.where(classes==c)[0]
        keep = nms_boxes(boxes[idx], scores[idx])
        for ki in keep:
            b = boxes[idx[ki]]
            final.append({
                "class_id":   int(c),
                "class_name": CLASSES[int(c)] if int(c)<len(CLASSES) else "Unknown",
                "score":      round(float(scores[idx[ki]]),4),
                "box":        [round(float(b[0]),1), round(float(b[1]),1),
                               round(float(b[2]),1), round(float(b[3]),1)]
            })

    final.sort(key=lambda d: d["score"], reverse=True)
    return final

# ══════════════════════════════════════════════════════════════════════════════
# JPEG ENCODE  ← KEY CHANGE: returns orig + jpeg dimensions
# ══════════════════════════════════════════════════════════════════════════════
def encode_jpeg(img):
    """
    Returns (b64_str, orig_w, orig_h, jpeg_w, jpeg_h).
    Box coords are in orig_w/orig_h space.
    The JPEG is resized to ≤640px wide — that's what the canvas renders at.
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
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def simulate_dets(h=480, w=640):
    import random
    count = random.randint(0,3) if random.random()>0.35 else 0
    dets  = []
    for _ in range(count):
        cl=random.randint(0,NUM_CLS-1)
        x1=random.uniform(10,w*0.5); y1=random.uniform(10,h*0.5)
        x2=min(x1+random.uniform(30,w*0.35),w-1)
        y2=min(y1+random.uniform(30,h*0.35),h-1)
        dets.append({"class_id":cl,"class_name":CLASSES[cl],
                     "score":round(random.uniform(0.5,0.97),3),
                     "box":[round(x1,1),round(y1,1),round(x2,1),round(y2,1)]})
    return dets

# ══════════════════════════════════════════════════════════════════════════════
# CAPTURE THREAD
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
        pipeline=(f"v4l2src device=/dev/video{cap_device} io-mode=dmabuf !"
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
            if cap_type=="video": break
            continue
        if cap_type in ("usb","rtsp","mipi"):
            skip += 1
            if skip % 2 != 0: continue
        try: frame_q.put_nowait(frame)
        except queue.Full: pass
    cap.release()

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
_shapes_logged = False

def run_inference(nn, img):
    global _shapes_logged
    t0 = time.time()

    blob, ratio, pad = preprocess(img, imgsz=IMGSZ)
    data = nn.nn_inference([blob], platform=args.platform, reorder='2 1 0',
                           output_tensor=3,
                           output_format=output_format.OUT_FORMAT_FLOAT32)

    if not _shapes_logged:
        shapes=[f"data[{i}]:size={data[i].size}({data[i].size//LISTSIZE}cells)" for i in range(len(data))]
        print(json.dumps({"type":"log","level":"info",
              "message":"Tensor sizes: "+", ".join(shapes)}), flush=True)
        _shapes_logged = True

    dets   = yolo_postprocess(list(data), ratio, pad, imgsz=IMGSZ)
    inf_ms = round((time.time()-t0)*1000, 2)
    return dets, inf_ms

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cap_type = args.type
    cap_dev  = args.device
    level    = int(args.level) if args.level in ['1','2'] else 0

    if ASNN_AVAILABLE:
        for label, path_ in [("model",args.model),("library",args.library)]:
            if not path_ or not os.path.exists(path_):
                print(json.dumps({"type":"log","level":"err",
                      "message":f"{label} not found: {path_}"}), flush=True)
                sys.exit(1)

    print(json.dumps({"type":"log","level":"info",
          "message":f"type={cap_type} device={cap_dev} imgsz={IMGSZ} "
                    f"obj={OBJ_THRESH} nms={NMS_THRESH} nc={NUM_CLS} "
                    f"listsize={LISTSIZE} jpeg_q={JPEG_QUALITY}"}), flush=True)

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
        dets, inf_ms = run_inference(nn, img) if nn else (simulate_dets(*img.shape[:2][::-1]), 0.0)
        jpeg_b64, orig_w, orig_h, jpeg_w, jpeg_h = encode_jpeg(img)
        print(json.dumps({"frame":1,"fps":1.0,"inference_ms":inf_ms,"detections":dets,
                          "jpeg":jpeg_b64,
                          "orig_w":orig_w,"orig_h":orig_h,
                          "jpeg_w":jpeg_w,"jpeg_h":jpeg_h}), flush=True)
        return

    # ── Stream / video mode ────────────────────────────────────────────────────
    threading.Thread(target=capture_frames, args=(cap_type,cap_dev), daemon=True).start()
    print(json.dumps({"type":"log","level":"info","message":"Capture started"}), flush=True)

    frame_num=0; fps_count=0; fps_val=0.0; fps_start=time.time()

    while True:
        try:
            orig = frame_q.get(timeout=3)
        except queue.Empty:
            print(json.dumps({"type":"log","level":"warn",
                  "message":"Frame timeout..."}), flush=True)
            continue

        frame_num += 1
        if nn:
            dets, inf_ms = run_inference(nn, orig)
        else:
            dets   = simulate_dets(orig.shape[0], orig.shape[1])
            inf_ms = round(float(np.random.uniform(8,25)), 2)

        fps_count += 1
        elapsed = time.time()-fps_start
        if elapsed >= 1.0:
            fps_val=round(fps_count/elapsed,2); fps_count=0; fps_start=time.time()

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
