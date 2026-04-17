#!/usr/bin/env python3
"""
ASNN Detection Script — outputs JSON lines with embedded JPEG frames
so the browser can display the actual video feed with detection overlays.

Each stdout line:
  {
    "frame": N, "fps": F, "inference_ms": T,
    "detections": [{"class_id":0,"class_name":"Car","score":0.92,"box":[x1,y1,x2,y2]},...],
    "jpeg": "<base64 JPEG of the original frame>"
  }
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

# ── Args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--library")
parser.add_argument("--model")
parser.add_argument("--type",         default="usb")
parser.add_argument("--device",       default="0")
parser.add_argument("--level",        default="0")
parser.add_argument("--obj-thresh",   type=float, default=0.4)
parser.add_argument("--nms-thresh",   type=float, default=0.5)
parser.add_argument("--platform",     default="ONNX")
parser.add_argument("--jpeg-quality", type=int,   default=75)
parser.add_argument("--classes",      nargs="+",  default=None)
parser.add_argument("--num-cls",      type=int,   default=None)
parser.add_argument("--listsize",     type=int,   default=None)
args = parser.parse_args()

OBJ_THRESH   = args.obj_thresh
NMS_THRESH   = args.nms_thresh
JPEG_QUALITY = max(10, min(95, args.jpeg_quality))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ── Constants ─────────────────────────────────────────────────────
GRID0, GRID1, GRID2, SPAN = 20, 40, 80, 1
constant_matrix = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]).T
mean = [0, 0, 0]
var  = [255]

# ── Load class names from data.yaml ───────────────────────────────
CLASSES  = None
NUM_CLS  = 1
LISTSIZE = 65

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
                    CLASSES  = tuple(names)
                    NUM_CLS  = len(CLASSES)
                    LISTSIZE = NUM_CLS + 64
                    print(json.dumps({"type":"log","level":"info",
                          "message":f"Loaded {NUM_CLS} classes from {yn}: {', '.join(CLASSES)}"}), flush=True)
                break
            except Exception as e:
                print(json.dumps({"type":"log","level":"warn",
                      "message":f"YAML error: {e}"}), flush=True)

if CLASSES is None:
    CLASSES = ("Object",)
    NUM_CLS = 1
    LISTSIZE = 65

if args.num_cls:  NUM_CLS  = args.num_cls
if args.listsize: LISTSIZE = args.listsize

# ── Post-processing ───────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=0):
    x = np.exp(x - x.max(axis=axis, keepdims=True))
    return x / x.sum(axis=axis, keepdims=True)

def process(inp):
    gh, gw = inp.shape[0], inp.shape[1]
    probs = sigmoid(inp[..., :NUM_CLS])
    b0 = softmax(inp[..., NUM_CLS:NUM_CLS+16], -1)
    b1 = softmax(inp[..., NUM_CLS+16:NUM_CLS+32], -1)
    b2 = softmax(inp[..., NUM_CLS+32:NUM_CLS+48], -1)
    b3 = softmax(inp[..., NUM_CLS+48:NUM_CLS+64], -1)
    res = np.zeros((gh, gw, 1, 4))
    res[...,0] = np.dot(b0, constant_matrix)[...,0]
    res[...,1] = np.dot(b1, constant_matrix)[...,0]
    res[...,2] = np.dot(b2, constant_matrix)[...,0]
    res[...,3] = np.dot(b3, constant_matrix)[...,0]
    col = np.tile(np.arange(0, gw), gw).reshape(-1, gw).reshape(gh, gw, 1, 1)
    row = np.tile(np.arange(0, gh).reshape(-1,1), gh).reshape(gh, gw, 1, 1)
    grid = np.concatenate((col, row), axis=-1)
    res[...,0:2] = (0.5 - res[...,0:2] + grid) / (gw, gh)
    res[...,2:4] = (0.5 + res[...,2:4] + grid) / (gw, gh)
    return res, probs

def filter_boxes(boxes, probs):
    cls  = np.argmax(probs, axis=-1)
    scr  = np.max(probs, axis=-1)
    pos  = np.where(scr >= OBJ_THRESH)
    return boxes[pos], cls[pos], scr[pos]

def nms_boxes(boxes, scores):
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1+1e-5)*np.maximum(0,yy2-yy1+1e-5)
        ovr=inter/(areas[i]+areas[order[1:]]-inter)
        order=order[np.where(ovr<=NMS_THRESH)[0]+1]
    return np.array(keep)

def yolov3_post(input_data):
    bl,cl,sl=[],[],[]
    for i in range(3):
        res,conf=process(input_data[i]); b,c,s=filter_boxes(res,conf)
        bl.append(b); cl.append(c); sl.append(s)
    boxes=np.concatenate(bl); classes=np.concatenate(cl); scores=np.concatenate(sl)
    if len(boxes)==0: return None,None,None
    nb,nc,ns=[],[],[]
    for c in set(classes):
        idx=np.where(classes==c); b,cc,s=boxes[idx],classes[idx],scores[idx]
        keep=nms_boxes(b,s); nb.append(b[keep]); nc.append(cc[keep]); ns.append(s[keep])
    if not nc: return None,None,None
    return np.concatenate(nb),np.concatenate(ns),np.concatenate(nc)

def format_dets(boxes, scores, classes):
    dets = []
    if boxes is None: return dets
    for box,score,cl in zip(boxes,scores,classes):
        x1,y1,x2,y2=[float(np.clip(v,0,1)) for v in box]
        dets.append({"class_id":int(cl),
                     "class_name":CLASSES[int(cl)] if int(cl)<len(CLASSES) else "Unknown",
                     "score":round(float(score),4),
                     "box":[round(x1,4),round(y1,4),round(x2,4),round(y2,4)]})
    return dets

# ── Frame encode ──────────────────────────────────────────────────
def encode_jpeg(img):
    """Resize to ≤640px wide and encode as base64 JPEG."""
    h, w = img.shape[:2]
    if w > 640:
        img = cv.resize(img, (640, int(h * 640 / w)))
    ok, buf = cv.imencode('.jpg', img, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok: return None
    return base64.b64encode(buf).decode('utf-8')

# ── Preprocess for inference ──────────────────────────────────────
def preprocess(img):
    r = cv.resize(img,(640,640)).astype(np.float32)
    r[:,:,0]-=mean[0]; r[:,:,1]-=mean[1]; r[:,:,2]-=mean[2]
    r/=var[0]
    return r.transpose(2,0,1)

# ── Simulate (no ASNN) ────────────────────────────────────────────
def simulate_dets():
    import random
    count = random.randint(0,2) if random.random()>0.4 else 0
    dets  = []
    for _ in range(count):
        cl=random.randint(0,NUM_CLS-1); x1=random.uniform(.05,.55); y1=random.uniform(.05,.55)
        dets.append({"class_id":cl,"class_name":CLASSES[cl],
                     "score":round(random.uniform(.5,.97),3),
                     "box":[round(x1,4),round(y1,4),
                            round(min(x1+random.uniform(.1,.3),.97),4),
                            round(min(y1+random.uniform(.1,.3),.97),4)]})
    return dets

# ── Capture thread ────────────────────────────────────────────────
frame_q = queue.Queue(maxsize=2)

def capture_frames(cap_type, cap_device):
    if cap_type == "usb":
        cap = cv.VideoCapture(int(cap_device), cv.CAP_V4L2)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
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
        print(json.dumps({"type":"log","level":"err","message":f"Unsupported: {cap_type}"}),flush=True)
        sys.exit(1)

    if not cap.isOpened():
        print(json.dumps({"type":"log","level":"err",
              "message":f"Cannot open: {cap_type}:{cap_device}"}),flush=True)
        sys.exit(1)

    skip=0
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            if cap_type=="video": break
            continue
        if cap_type in ("usb","rtsp","mipi"):
            skip+=1
            if skip%2!=0: continue   # every other frame for live sources
        try: frame_q.put_nowait(frame)
        except queue.Full: pass
    cap.release()

# ── Main ──────────────────────────────────────────────────────────
def main():
    cap_type = args.type
    cap_dev  = args.device
    level    = int(args.level) if args.level in ['1','2'] else 0

    if ASNN_AVAILABLE:
        for label,path in [("model",args.model),("library",args.library)]:
            if not path or not os.path.exists(path):
                print(json.dumps({"type":"log","level":"err",
                      "message":f"{label} not found: {path}"}),flush=True); sys.exit(1)

    print(json.dumps({"type":"log","level":"info",
          "message":f"type={cap_type} device={cap_dev} classes={list(CLASSES)} jpeg_q={JPEG_QUALITY}"}),flush=True)

    nn = None
    if ASNN_AVAILABLE:
        nn = asnn('Electron')
        nn.nn_init(library=args.library, model=args.model, level=level)
        print(json.dumps({"type":"log","level":"info","message":"Neural network ready"}),flush=True)

    # ── Single image mode ─────────────────────────────────────────
    if cap_type == "image":
        img = cv.imread(cap_dev)
        if img is None:
            print(json.dumps({"type":"log","level":"err","message":f"Cannot read: {cap_dev}"}),flush=True)
            sys.exit(1)
        t0 = time.time()
        if nn:
            data=nn.nn_inference([preprocess(img)],platform=args.platform,
                                 reorder='2 1 0',output_tensor=3,
                                 output_format=output_format.OUT_FORMAT_FLOAT32)
            input_data=[
                np.transpose(data[2].reshape(SPAN,LISTSIZE,GRID0,GRID0),(2,3,0,1)),
                np.transpose(data[1].reshape(SPAN,LISTSIZE,GRID1,GRID1),(2,3,0,1)),
                np.transpose(data[0].reshape(SPAN,LISTSIZE,GRID2,GRID2),(2,3,0,1)),
            ]
            boxes,scores,classes=yolov3_post(input_data)
            dets=format_dets(boxes,scores,classes)
        else:
            dets=simulate_dets()
        inf_ms=round((time.time()-t0)*1000,2)
        jpeg=encode_jpeg(img)
        print(json.dumps({"frame":1,"fps":1.0,"inference_ms":inf_ms,"detections":dets,"jpeg":jpeg}),flush=True)
        return

    # ── Video / Webcam / RTSP mode ────────────────────────────────
    threading.Thread(target=capture_frames,args=(cap_type,cap_dev),daemon=True).start()
    print(json.dumps({"type":"log","level":"info","message":"Capture started"}),flush=True)

    frame_num=0; fps_count=0; fps_val=0.0; fps_start=time.time()

    while True:
        try: orig=frame_q.get(timeout=3)
        except queue.Empty:
            print(json.dumps({"type":"log","level":"warn","message":"Frame timeout"}),flush=True)
            continue

        t0=time.time(); frame_num+=1

        if nn:
            data=nn.nn_inference([preprocess(orig)],platform=args.platform,
                                 reorder='2 1 0',output_tensor=3,
                                 output_format=output_format.OUT_FORMAT_FLOAT32)
            input_data=[
                np.transpose(data[2].reshape(SPAN,LISTSIZE,GRID0,GRID0),(2,3,0,1)),
                np.transpose(data[1].reshape(SPAN,LISTSIZE,GRID1,GRID1),(2,3,0,1)),
                np.transpose(data[0].reshape(SPAN,LISTSIZE,GRID2,GRID2),(2,3,0,1)),
            ]
            boxes,scores,classes=yolov3_post(input_data)
            dets=format_dets(boxes,scores,classes)
        else:
            dets=simulate_dets()

        inf_ms=round((time.time()-t0)*1000,2)
        jpeg=encode_jpeg(orig)   # encode ORIGINAL frame (not 640x640 preprocessed)

        fps_count+=1
        elapsed=time.time()-fps_start
        if elapsed>=1.0:
            fps_val=round(fps_count/elapsed,2); fps_count=0; fps_start=time.time()

        print(json.dumps({
            "frame":        frame_num,
            "fps":          fps_val,
            "inference_ms": inf_ms,
            "detections":   dets,
            "jpeg":         jpeg        # <-- key addition: actual frame
        }), flush=True)

if __name__=='__main__':
    main()
