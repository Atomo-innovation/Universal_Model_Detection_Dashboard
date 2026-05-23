#!/usr/bin/env python3
"""
YOLO26s person detection on Khadas Electron (asnn).

Supports:
  --type rtsp   --device <url>          RTSP stream
  --type usb    --device <index>        USB webcam via V4L2
  --type mipi   --device <index>        MIPI cam via GStreamer
  --type video  --device <path>         Video file
  --type image  --device <path>         Single image

Add --json-stream to emit dashboard-compatible JSON lines on stdout:
  {"frame":N, "fps":F, "inference_ms":T, "detections":[...], "jpeg":"<b64>"}
Log/status lines go to stderr in that mode, keeping stdout clean.

No person_live.json writes.
"""
from __future__ import annotations

import argparse, base64, json, logging, os, sys, threading, time
from dataclasses import dataclass, field
from typing import Any

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2 as cv
import numpy as np
from asnn.api import asnn
from asnn.types import output_format

log = logging.getLogger("person_detect")

# ── Model geometry (YOLO26s, 84 ch / scale) ──────────────────────
GRID_SIZES     = (20, 40, 80)
STRIDES        = (32, 16, 8)
LISTSIZE       = 84
NUM_CLS        = 80
SCALE_CONF_MUL = (1.0, 0.92, 0.78)
INV_255        = 1.0 / 255.0
LETTERBOX_PAD  = 114

DEFAULT_CONF = 0.34
DEFAULT_NMS  = 0.56
DEFAULT_IMGSZ = (640, 640)

DECODE_MARGIN_DARK = 0.06
DECODE_FLOOR       = 0.26
MIN_BOX_AREA = 0.00008
MIN_BOX_W    = 0.004
MIN_BOX_H    = 0.006
MIN_ASPECT   = 0.35
MAX_ASPECT   = 5.5
VERT_NMS_SEP = 0.055

cv.setNumThreads(2)


# ── Runtime state ─────────────────────────────────────────────────
@dataclass
class RuntimeState:
    img_w: int = 640
    img_h: int = 640
    prebuf: np.ndarray | None = None
    grid_caches: tuple = ()
    gamma_luts: dict = field(default_factory=dict)

@dataclass
class PreprocessConfig:
    enabled: bool = False
    clahe_clip: float = 2.4
    clahe_grid: int = 8
    gamma: float = 1.42
    luma_scale: float = 1.06
    dark_threshold: int = 100
    brightness_beta: int = 16
    max_boost: int = 48

@dataclass
class LetterboxMeta:
    ratio: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int

RT = RuntimeState()
PP = PreprocessConfig()
_CLAHE = None


class GridCache:
    __slots__ = ("ax","ay","inv_w","inv_h","grid_h","scale_conf_mul","y_center_norm","spatial_thresh")
    def __init__(self, grid_h, grid_w, stride, img_w, img_h, scale_conf_mul):
        col = np.arange(grid_w, dtype=np.float32)
        row = np.arange(grid_h, dtype=np.float32)
        self.ax = (col + 0.5).reshape(1, grid_w)
        self.ay = (row + 0.5).reshape(grid_h, 1)
        self.inv_w = stride / float(img_w)
        self.inv_h = stride / float(img_h)
        self.grid_h = grid_h
        self.scale_conf_mul = float(scale_conf_mul)
        y = ((row + 0.5) * stride / float(img_h)).reshape(grid_h, 1)
        self.y_center_norm = y.astype(np.float32)
        t = np.ones((grid_h, 1), dtype=np.float32)
        t = np.where(y >= 0.46, t * 0.82, t)
        if grid_h >= 40:
            t = np.where(y <= 0.34, t * 0.88, t)
            t = np.where((y >= 0.18) & (y <= 0.42), t * 0.90, t)
        self.spatial_thresh = t


def init_runtime(img_w, img_h):
    RT.img_w, RT.img_h = int(img_w), int(img_h)
    RT.prebuf = np.empty((3, RT.img_h, RT.img_w), dtype=np.float32)
    RT.grid_caches = tuple(
        GridCache(g, g, s, RT.img_w, RT.img_h, m)
        for g, s, m in zip(GRID_SIZES, STRIDES, SCALE_CONF_MUL)
    )


# ── Utilities ─────────────────────────────────────────────────────
class FpsMeter:
    def __init__(self, interval=0.5):
        self.interval = interval; self.count = 0; self.fps = 0.0; self.t0 = time.perf_counter()
    def tick(self):
        self.count += 1
        elapsed = time.perf_counter() - self.t0
        if elapsed >= self.interval:
            self.fps = self.count / elapsed; self.count = 0; self.t0 = time.perf_counter()
        return self.fps

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
def measure_brightness(rgb): return float(np.mean(cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)))

def to_rgb(frame):
    if frame is None: raise ValueError("empty frame")
    if frame.ndim == 2: return cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    if frame.shape[2] == 3: return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if frame.shape[2] == 4: return cv.cvtColor(frame, cv.COLOR_BGRA2RGB)
    raise ValueError("unsupported channels")

def decode_confidence(display_conf, mean_l_in):
    if mean_l_in >= 100: return display_conf
    t = max(0.0, min(1.0, (100.0 - mean_l_in) / 50.0))
    return max(DECODE_FLOOR, display_conf - DECODE_MARGIN_DARK * t)


# ── Preprocess ────────────────────────────────────────────────────
def _clahe_inst(clip):
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv.createCLAHE(clipLimit=clip, tileGridSize=(PP.clahe_grid, PP.clahe_grid))
    return _CLAHE

def _gamma_lut(gamma):
    key = round(gamma, 3)
    if key not in RT.gamma_luts:
        RT.gamma_luts[key] = (np.linspace(0,1,256)**(1.0/gamma)*255).astype(np.uint8)
    return RT.gamma_luts[key]

def _cap_luma_rgb(rgb, target):
    mean_l = measure_brightness(rgb)
    if mean_l <= target + 2: return rgb
    lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB); l, a, b = cv.split(lab)
    l = np.clip(l.astype(np.float32) * (target/mean_l), 0, 255).astype(np.uint8)
    return cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2RGB)

def enhance_rgb(rgb):
    mean_in = measure_brightness(rgb)
    if mean_in >= 82:   mode,clip,luma,gamma,boost_k = "contrast",2.0,1.02,1.0,0.0
    elif mean_in >= 48: mode,clip,luma,gamma,boost_k = "mild",2.2,min(PP.luma_scale,1.06),min(PP.gamma,1.28),0.5
    else:               mode,clip,luma,gamma,boost_k = "full",PP.clahe_clip,PP.luma_scale,PP.gamma,1.0
    lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB); l, a, b = cv.split(lab)
    l = _clahe_inst(clip).apply(l)
    l = np.clip(l.astype(np.float32)*luma, 0, 255).astype(np.uint8)
    if boost_k > 0 and mean_in < PP.dark_threshold:
        boost = int(boost_k * min(PP.max_boost, (PP.dark_threshold-mean_in)*0.4 + PP.brightness_beta*0.5))
        if boost > 0: l = cv.add(l, boost)
    out = cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2RGB)
    if mode == "full" and mean_in < 72: gamma = max(gamma, 1.42 + (72.0-mean_in)*0.003)
    if gamma > 1.0: out = cv.LUT(out, _gamma_lut(gamma))
    cap = 96.0 if mode=="contrast" else 94.0 if mode=="mild" else 90.0
    return _cap_luma_rgb(out, cap), mean_in, measure_brightness(out)

def letterbox(rgb):
    h, w = rgb.shape[:2]; tw, th = RT.img_w, RT.img_h
    r = min(tw/w, th/h)
    nw, nh = max(1, int(round(w*r))), max(1, int(round(h*r)))
    resized = cv.resize(rgb, (nw, nh), interpolation=cv.INTER_LINEAR)
    px, py = (tw-nw)//2, (th-nh)//2
    out = np.full((th, tw, 3), LETTERBOX_PAD, dtype=np.uint8)
    out[py:py+nh, px:px+nw] = resized
    return out, LetterboxMeta(r, px, py, w, h)

def prepare_frame(bgr):
    rgb = to_rgb(bgr)
    mean_in = measure_brightness(rgb)
    if PP.enabled: rgb, mean_in, mean_out = enhance_rgb(rgb)
    else: mean_out = mean_in
    lettered, meta = letterbox(rgb)
    np.copyto(RT.prebuf, lettered.astype(np.float32).transpose(2,0,1) * INV_255)
    return RT.prebuf, mean_in, mean_out, meta


# ── Postprocess ───────────────────────────────────────────────────
def decode_scale(raw, cache, conf_decode):
    probs = sigmoid(raw[0])
    thresh = conf_decode * cache.scale_conf_mul * cache.spatial_thresh
    mask = probs >= thresh
    if not np.any(mask): return np.empty((0,4), np.float32), np.empty(0, np.float32)
    l,t,r,b = raw[NUM_CLS:NUM_CLS+4]
    x1 = (cache.ax - l) * cache.inv_w; y1 = (cache.ay - t) * cache.inv_h
    x2 = (cache.ax + r) * cache.inv_w; y2 = (cache.ay + b) * cache.inv_h
    boxes = np.stack((x1[mask], y1[mask], x2[mask], y2[mask]), axis=-1).astype(np.float32)
    return boxes, probs[mask].astype(np.float32)

def refine_boxes(boxes, scores):
    if boxes.size == 0: return boxes, scores
    w = boxes[:,2]-boxes[:,0]; h = boxes[:,3]-boxes[:,1]; ar = h/(w+1e-6)
    ok = (w>MIN_BOX_W)&(h>MIN_BOX_H)&(w*h>MIN_BOX_AREA)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])&(ar>MIN_ASPECT)&(ar<MAX_ASPECT)
    return boxes[ok], scores[ok]

def nms_desk_aware(boxes, scores, iou_thresh):
    if boxes.size == 0: return np.array([], dtype=np.int64)
    x1,y1,x2,y2 = boxes.T; cy = (y1+y2)*0.5
    areas = np.maximum(0.0,x2-x1)*np.maximum(0.0,y2-y1)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = int(order[0]); keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        xx1=np.maximum(x1[i],x1[rest]); yy1=np.maximum(y1[i],y1[rest])
        xx2=np.minimum(x2[i],x2[rest]); yy2=np.minimum(y2[i],y2[rest])
        inter=np.maximum(0.0,xx2-xx1)*np.maximum(0.0,yy2-yy1)
        ovr=inter/(areas[i]+areas[rest]-inter+1e-6)
        order=rest[~((ovr>iou_thresh)&(np.abs(cy[i]-cy[rest])<VERT_NMS_SEP))]
    return np.array(keep, dtype=np.int64)

def suppress_nested(boxes, scores, min_score):
    n = len(scores)
    if n < 2: return boxes, scores
    cy = (boxes[:,1]+boxes[:,3])*0.5
    order = scores.argsort()[::-1]; keep = np.ones(n, dtype=bool)
    for ii,i in enumerate(order):
        if not keep[i] or scores[i]<min_score: continue
        for j in order[ii+1:]:
            if not keep[j] or scores[j]>=min_score: continue
            if abs(cy[i]-cy[j])>0.045: continue
            xx1,yy1=max(boxes[i,0],boxes[j,0]),max(boxes[i,1],boxes[j,1])
            xx2,yy2=min(boxes[i,2],boxes[j,2]),min(boxes[i,3],boxes[j,3])
            if xx2<=xx1 or yy2<=yy1: continue
            inter=(xx2-xx1)*(yy2-yy1)
            aj=(boxes[j,2]-boxes[j,0])*(boxes[j,3]-boxes[j,1])
            ai=(boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
            if inter/(ai+aj-inter+1e-6)>0.38: keep[j]=False
    return boxes[keep], scores[keep]

def map_boxes_to_frame(boxes, meta):
    tw,th=float(RT.img_w),float(RT.img_h)
    ow,oh=float(meta.orig_w),float(meta.orig_h)
    r,px,py=meta.ratio,float(meta.pad_x),float(meta.pad_y)
    out=boxes.copy()
    out[:,0]=(boxes[:,0]*tw-px)/r/ow; out[:,2]=(boxes[:,2]*tw-px)/r/ow
    out[:,1]=(boxes[:,1]*th-py)/r/oh; out[:,3]=(boxes[:,3]*th-py)/r/oh
    np.clip(out, 0.0, 1.0, out=out)
    return out

def postprocess(outputs, conf_decode, conf_display, nms_thresh, meta):
    bl, sl = [], []
    for raw, cache in zip(outputs, RT.grid_caches):
        b, s = decode_scale(raw, cache, conf_decode)
        if b.size: bl.append(b); sl.append(s)
    if not bl: return None, None
    boxes=np.concatenate(bl); scores=np.concatenate(sl)
    boxes,scores=refine_boxes(boxes,scores)
    if boxes.size==0: return None, None
    keep=nms_desk_aware(boxes,scores,nms_thresh)
    boxes,scores=boxes[keep],scores[keep]
    boxes,scores=suppress_nested(boxes,scores,conf_display)
    boxes=map_boxes_to_frame(boxes,meta)
    ok=scores>=conf_display; boxes,scores=boxes[ok],scores[ok]
    if boxes.size==0: return None, None
    return boxes, scores

def parse_npu_output(arr, grid_h, grid_w):
    arr=np.asarray(arr,dtype=np.float32); expected=LISTSIZE*grid_h*grid_w
    if arr.ndim==4: arr=arr[0]
    if arr.ndim==3:
        if arr.shape[0]==LISTSIZE: return arr
        if arr.shape[-1]==LISTSIZE: return arr.transpose(2,0,1)
    if arr.size==expected: return arr.reshape(LISTSIZE,grid_h,grid_w)
    raise ValueError("unexpected asnn shape {} for {}x{}".format(arr.shape,grid_h,grid_w))

def run_inference(net, tensor):
    data=net.nn_inference([tensor],platform="ONNX",reorder="2 1 0",output_tensor=3,
                          output_format=output_format.OUT_FORMAT_FLOAT32)
    return [
        parse_npu_output(data[2],GRID_SIZES[0],GRID_SIZES[0]),
        parse_npu_output(data[1],GRID_SIZES[1],GRID_SIZES[1]),
        parse_npu_output(data[0],GRID_SIZES[2],GRID_SIZES[2]),
    ]


# ── JPEG encode ───────────────────────────────────────────────────
def encode_jpeg(bgr, quality=75):
    h,w=bgr.shape[:2]
    if w>640: bgr=cv.resize(bgr,(640,int(h*640/w)))
    ok,buf=cv.imencode('.jpg',bgr,[cv.IMWRITE_JPEG_QUALITY,quality])
    return base64.b64encode(buf).decode('utf-8') if ok else None

def format_dets(boxes, scores):
    if boxes is None or scores is None: return []
    return [{
        "class_id":0,"class_name":"person",
        "score":round(float(s),4),
        "box":[round(float(np.clip(v,0,1)),4) for v in b]
    } for b,s in zip(boxes,scores)]


# ── Capture sources ───────────────────────────────────────────────

class LatestFrameReader:
    """Non-blocking RTSP reader — always returns the most recent frame."""
    def __init__(self, url, transport="tcp"):
        self.url=url; self.transport=transport
        self._lock=threading.Lock(); self._frame=None; self._ok=False
        self._stamp=0; self._stop=threading.Event(); self._thread=None; self._cap=None

    def _open(self):
        proto="tcp" if self.transport=="tcp" else "udp"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]=(
            "rtsp_transport;{}|fflags;nobuffer|flags;low_delay|max_delay;0".format(proto))
        cap=cv.VideoCapture(self.url,cv.CAP_FFMPEG); cap.set(cv.CAP_PROP_BUFFERSIZE,1)
        return cap

    def start(self):
        self._cap=self._open()
        if not self._cap.isOpened(): return False
        self._thread=threading.Thread(target=self._loop,daemon=True,name="rtsp-reader")
        self._thread.start(); return True

    def _loop(self):
        while not self._stop.is_set():
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.5); self._cap=self._open(); continue
            ret,frame=self._cap.read()
            if not ret:
                time.sleep(0.2)
                if self._cap: self._cap.release()
                self._cap=self._open(); continue
            with self._lock: self._frame=frame; self._ok=True; self._stamp+=1

    def get_frame(self):
        with self._lock:
            if not self._ok or self._frame is None: return None
            return self._frame.copy()

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=2.0)
        if self._cap: self._cap.release()


def open_local_cap(cap_type, device):
    """Open a local capture (USB / MIPI / video file). Returns cv.VideoCapture or None."""
    if cap_type == "usb":
        cap=cv.VideoCapture(int(device), cv.CAP_V4L2)
        cap.set(cv.CAP_PROP_FRAME_WIDTH,640); cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
    elif cap_type == "mipi":
        pipeline=(f"v4l2src device=/dev/video{device} io-mode=dmabuf !"
                  "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 !"
                  "videoconvert ! appsink")
        cap=cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    elif cap_type == "video":
        cap=cv.VideoCapture(device)
    else:
        return None
    return cap if cap.isOpened() else None


# ── CLI ───────────────────────────────────────────────────────────
def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--library",      required=True)
    p.add_argument("--model",        required=True)
    # unified interface matching detect.py
    p.add_argument("--type",         default="rtsp",
                   choices=["rtsp","usb","mipi","video","image"],
                   help="Input source type")
    p.add_argument("--device",       default="",
                   help="URL for rtsp, index for usb/mipi, path for video/image")
    # kept for backwards compat
    p.add_argument("--rtsp",         default=None, help="RTSP URL (alias for --type rtsp --device <url>)")
    p.add_argument("--transport",    default="tcp", choices=["tcp","udp"])
    p.add_argument("--level",        default="0")
    p.add_argument("--conf",         type=float, default=DEFAULT_CONF)
    p.add_argument("--nms",          type=float, default=DEFAULT_NMS)
    p.add_argument("--imgsz",        type=int, nargs="+", default=list(DEFAULT_IMGSZ), metavar="N")
    p.add_argument("--headless",     action="store_true")
    p.add_argument("--low-light",    action="store_true")
    p.add_argument("--log-level",    default="WARNING", choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--json-stream",  action="store_true",
                   help="Emit dashboard JSON lines on stdout; all logging goes to stderr")
    p.add_argument("--jpeg-quality", type=int, default=75)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────
def main():
    args=parse_args()

    log_stream=sys.stderr if args.json_stream else sys.stdout
    logging.basicConfig(level=getattr(logging,args.log_level),
                        format="%(levelname)s %(message)s", stream=log_stream)

    def jlog(level, message):
        if args.json_stream:
            print(json.dumps({"type":"log","level":level,"message":message}),flush=True)
        else:
            print("[{}] {}".format(level.upper(), message), file=log_stream, flush=True)

    if not os.path.isfile(args.model):
        jlog("err","model not found: {}".format(args.model)); sys.exit(1)
    if not os.path.isfile(args.library):
        jlog("err","library not found: {}".format(args.library)); sys.exit(1)

    # Resolve --rtsp alias → --type rtsp --device <url>
    cap_type = args.type
    device   = args.device
    if args.rtsp:
        cap_type = "rtsp"
        device   = args.rtsp

    if not device:
        jlog("err","--device is required"); sys.exit(1)

    conf = max(0.05, min(0.95, args.conf))
    nms  = max(0.3,  min(0.9,  args.nms))
    imgsz = args.imgsz
    img_w,img_h = (imgsz[0],imgsz[0]) if len(imgsz)==1 else (imgsz[0],imgsz[1])
    init_runtime(img_w,img_h)

    PP.enabled = args.low_light
    global _CLAHE; _CLAHE=None

    jpeg_q   = max(10, min(95, args.jpeg_quality))
    headless = args.headless or args.json_stream
    level    = int(args.level) if args.level in ("1","2") else 0

    jlog("info","init conf={:.2f} nms={:.2f} size={}x{} type={} device={} low_light={} json_stream={}".format(
        conf,nms,img_w,img_h,cap_type,device,PP.enabled,args.json_stream))

    net=asnn("Electron")
    net.nn_init(library=args.library,model=args.model,level=level)
    jlog("info","Neural network ready")

    fps_m   = FpsMeter()
    frame_n = 0

    # ── Single image ─────────────────────────────────────────────
    if cap_type == "image":
        img=cv.imread(device)
        if img is None: jlog("err","cannot read: {}".format(device)); sys.exit(1)
        t0=time.perf_counter()
        tensor,mean_in,mean_out,meta=prepare_frame(img)
        outputs=run_inference(net,tensor)
        boxes,scores=postprocess(outputs,decode_confidence(conf,mean_in),conf,nms,meta)
        inf_ms=round((time.perf_counter()-t0)*1000,2)
        dets=format_dets(boxes,scores)
        jpeg=encode_jpeg(img,jpeg_q)
        if args.json_stream:
            print(json.dumps({"frame":1,"fps":1.0,"inference_ms":inf_ms,"detections":dets,"jpeg":jpeg}),flush=True)
        else:
            print("{} person(s) | {:.1f}ms".format(len(dets),inf_ms))
        return

    # ── RTSP ─────────────────────────────────────────────────────
    if cap_type == "rtsp":
        reader=LatestFrameReader(device, args.transport)
        if not reader.start():
            jlog("err","cannot open RTSP: {}".format(device)); sys.exit(1)
        jlog("info","RTSP capture started")

        if not headless:
            cv.namedWindow("Person RTSP", cv.WINDOW_NORMAL)

        try:
            while True:
                frame=reader.get_frame()
                if frame is None: time.sleep(0.005); continue

                tensor,mean_in,mean_out,meta=prepare_frame(frame)
                t0=time.perf_counter()
                outputs=run_inference(net,tensor)
                boxes,scores=postprocess(outputs,decode_confidence(conf,mean_in),conf,nms,meta)
                inf_ms=round((time.perf_counter()-t0)*1000,2)
                fps_val=fps_m.tick(); frame_n+=1

                if args.json_stream:
                    print(json.dumps({
                        "frame":frame_n,"fps":round(fps_val,2),"inference_ms":inf_ms,
                        "detections":format_dets(boxes,scores),
                        "jpeg":encode_jpeg(frame,jpeg_q)
                    }),flush=True)
                else:
                    n=0 if boxes is None else len(boxes)
                    sys.stderr.write("\r[RTSP] persons={} fps={:.1f} {:.0f}ms     ".format(n,fps_val,inf_ms))
                    sys.stderr.flush()

                if not headless:
                    show=frame.copy()
                    if boxes is not None:
                        h,w=show.shape[:2]
                        for box,sc in zip(boxes,scores):
                            x1,y1,x2,y2=int(box[0]*w),int(box[1]*h),int(box[2]*w),int(box[3]*h)
                            cv.rectangle(show,(x1,y1),(x2,y2),(0,255,0),2)
                            cv.putText(show,"person {:.2f}".format(sc),(x1,max(0,y1-4)),
                                       cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv.LINE_AA)
                    cv.imshow("Person RTSP",show)
                    if cv.waitKey(1)&0xFF==ord('q'): break
        except KeyboardInterrupt:
            pass
        finally:
            reader.stop()
            if not headless: cv.destroyAllWindows()
        return

    # ── USB / MIPI / Video file ───────────────────────────────────
    cap=open_local_cap(cap_type, device)
    if cap is None:
        jlog("err","cannot open {}: {}".format(cap_type,device)); sys.exit(1)
    jlog("info","{} capture started".format(cap_type.upper()))

    is_live = cap_type in ("usb","mipi")
    skip    = 0

    if not headless:
        cv.namedWindow("Person {}".format(cap_type.upper()), cv.WINDOW_NORMAL)

    try:
        while True:
            ret,frame=cap.read()
            if not ret:
                if not is_live: break   # video file ended
                time.sleep(0.01); continue

            # throttle live sources to every other frame
            if is_live:
                skip+=1
                if skip%2!=0: continue

            tensor,mean_in,mean_out,meta=prepare_frame(frame)
            t0=time.perf_counter()
            outputs=run_inference(net,tensor)
            boxes,scores=postprocess(outputs,decode_confidence(conf,mean_in),conf,nms,meta)
            inf_ms=round((time.perf_counter()-t0)*1000,2)
            fps_val=fps_m.tick(); frame_n+=1

            if args.json_stream:
                print(json.dumps({
                    "frame":frame_n,"fps":round(fps_val,2),"inference_ms":inf_ms,
                    "detections":format_dets(boxes,scores),
                    "jpeg":encode_jpeg(frame,jpeg_q)
                }),flush=True)
            else:
                n=0 if boxes is None else len(boxes)
                sys.stderr.write("\r[{}] persons={} fps={:.1f} {:.0f}ms     ".format(
                    cap_type.upper(),n,fps_val,inf_ms))
                sys.stderr.flush()

            if not headless:
                show=frame.copy()
                if boxes is not None:
                    h,w=show.shape[:2]
                    for box,sc in zip(boxes,scores):
                        x1,y1,x2,y2=int(box[0]*w),int(box[1]*h),int(box[2]*w),int(box[3]*h)
                        cv.rectangle(show,(x1,y1),(x2,y2),(0,255,0),2)
                        cv.putText(show,"person {:.2f}".format(sc),(x1,max(0,y1-4)),
                                   cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv.LINE_AA)
                cv.imshow("Person {}".format(cap_type.upper()),show)
                if cv.waitKey(1)&0xFF==ord('q'): break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not headless: cv.destroyAllWindows()


if __name__=="__main__":
    init_runtime(*DEFAULT_IMGSZ)
    main()
