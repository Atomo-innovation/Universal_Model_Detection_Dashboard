"""
p2pnet_dnn.py  —  P2PNet crowd counting via OpenCV DNN (TimVX NPU / CPU)
=========================================================================
v5  —  realtime RTSP without grab/retrieve split
-------------------------------------------------
ROOT CAUSE of "Picture does not contain data":
  grab() advances the decoder's internal pointer.  If retrieve() is not
  called immediately in the same thread, FFmpeg may reuse/overwrite that
  slot.  The grab+retrieve split is unreliable with CAP_FFMPEG.

SOLUTION:
  • Use cap.read() (safe, atomic grab+decode).
  • The reader thread calls cap.read() in a tight loop and stores frames
    in a threading.Event + shared variable (not a Queue).  Every new
    frame simply overwrites the previous one.  The inference thread wakes
    on the Event, clears it, and processes whatever is there right now.
  • To keep the RTSP decoder from buffering internally we set FFmpeg
    options that force minimal buffering: nobuffer + low_delay.
  • CAP_PROP_BUFFERSIZE=1 tells OpenCV's ring buffer to hold only 1 frame.
  • If inference is slower than the camera FPS (common), older frames are
    overwritten before inference ever sees them → always newest frame.

CPU IDLE:
  • Reader thread: cap.read() blocks inside C until a frame arrives from
    the network — zero Python busy-wait.
  • Inference thread: Event.wait(timeout) blocks — zero spin.
  • No timers, no polling anywhere.
"""

import os, sys, argparse, time, json, base64, signal, threading
import cv2
import numpy as np

# ── Shutdown ───────────────────────────────────────────────────────────────────
_stop_event = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: _stop_event.set())
signal.signal(signal.SIGINT,  lambda *_: _stop_event.set())

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_W, INPUT_H = 640, 480
MEAN      = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD       = (0.229, 0.224, 0.225)
N_ANCHORS = 19200
VIDEO_FPS = 20.0


# ── Anchor points ─────────────────────────────────────────────────────────────
def build_anchor_points(h=INPUT_H, w=INPUT_W, stride=8, row=2, line=2):
    row_step, line_step = stride / row, stride / line
    sx = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    sy = (np.arange(1, row  + 1) - 0.5) * row_step  - stride / 2
    sx, sy = np.meshgrid(sx, sy)
    local_pts = np.vstack((sx.ravel(), sy.ravel())).T
    feat_h = (h + stride - 1) // stride
    feat_w = (w + stride - 1) // stride
    gx = (np.arange(feat_w) + 0.5) * stride
    gy = (np.arange(feat_h) + 0.5) * stride
    gx, gy = np.meshgrid(gx, gy)
    grid = np.vstack((gx.ravel(), gy.ravel())).T
    K, A = len(grid), len(local_pts)
    pts = (local_pts[None] + grid[:, None, :]).reshape(K * A, 2)
    return pts.astype(np.float32)

ANCHOR_PTS = build_anchor_points()


# ── Pre / post ────────────────────────────────────────────────────────────────
def make_blob(bgr):
    img  = cv2.resize(bgr, (INPUT_W, INPUT_H))
    blob = cv2.dnn.blobFromImage(
        img, 1.0 / 255.0, (INPUT_W, INPUT_H), MEAN, swapRB=True, crop=False)
    blob[:, 0] /= STD[0]; blob[:, 1] /= STD[1]; blob[:, 2] /= STD[2]
    return blob

def softmax2(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def postprocess(outs, thr):
    scores = softmax2(outs[1].reshape(N_ANCHORS, 2))[:, 1]
    coords = outs[0].reshape(N_ANCHORS, 2) * 100.0 + ANCHOR_PTS
    mask   = scores > thr
    return coords[mask].tolist(), int(mask.sum())

def draw(frame, pts, count, ms):
    out = frame.copy()
    for p in pts:
        x, y = int(p[0]), int(p[1])
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
    cv2.putText(out, f'Count:{count}  {ms:.0f}ms',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return out

def emit(obj):
    print(json.dumps(obj), flush=True)

def to_b64(frame, q=75):
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


# ── FrameReader ───────────────────────────────────────────────────────────────
class FrameReader:
    """
    Background thread calls cap.read() continuously.
    Each new frame is stored in self._frame (overwrites previous).
    self._event is set to wake the inference thread.
    Because frames are overwritten, inference always gets the LATEST
    frame — never a stale buffered one.
    """

    def __init__(self, source, is_rtsp=False, reconnect_delay=5):
        self._src      = source
        self._is_rtsp  = is_rtsp
        self._delay    = reconnect_delay

        self._frame    = None
        self._event    = threading.Event()
        self._lock     = threading.Lock()
        self._alive    = True
        self._src_fps  = VIDEO_FPS

        self._cap = self._open()
        if self._cap.isOpened():
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self._src_fps = fps

        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    # ── public ───────────────────────────────────────────────────────────────
    @property
    def src_fps(self):
        return self._src_fps

    def is_opened(self):
        return self._cap and self._cap.isOpened()

    def is_alive(self):
        return self._alive and not _stop_event.is_set()

    def get_frame(self, timeout=0.15):
        """
        Block until a new frame arrives (or timeout).
        Returns (frame, True) or (None, False).
        Because _frame is always overwritten, we get the newest one.
        """
        if not self._event.wait(timeout):
            return None, False
        self._event.clear()
        with self._lock:
            return self._frame, self._frame is not None

    def release(self):
        self._alive = False
        if self._cap:
            self._cap.release()

    # ── internal ─────────────────────────────────────────────────────────────
    def _open(self):
        if self._is_rtsp:
            # Tell FFmpeg: no internal buffering, low latency, TCP transport
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp'
                '|fflags;nobuffer'
                '|flags;low_delay'
                '|tune;zerolatency'
                '|max_delay;0'
            )
            cap = cv2.VideoCapture(self._src, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self._src)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        while not _stop_event.is_set() and self._alive:
            ret, frame = self._cap.read()

            if not ret:
                if self._is_rtsp and self._delay > 0:
                    self._cap.release()
                    # Sleep in 50 ms steps so SIGTERM is noticed fast
                    for _ in range(self._delay * 20):
                        if _stop_event.is_set() or not self._alive:
                            break
                        time.sleep(0.05)
                    if not _stop_event.is_set() and self._alive:
                        self._cap = self._open()
                    continue
                break   # EOF / unrecoverable

            # Overwrite with latest frame; set event to wake inference thread
            with self._lock:
                self._frame = frame
            self._event.set()   # non-blocking; inference will clear it

        self._alive = False


# ── Inference loop ────────────────────────────────────────────────────────────
def video_loop(net, out_names, reader, args, is_rtsp=False):
    writer    = None
    fps_hist  = []
    frame_idx = 0
    min_gap   = 1.0 / max(args.max_fps, 1)
    last_emit = 0.0

    if args.save_video:
        p = 'rtsp_out.mp4' if is_rtsp else 'video_out.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(p, fourcc, reader.src_fps, (INPUT_W, INPUT_H))

    try:
        while not _stop_event.is_set() and reader.is_alive():
            frame, ok = reader.get_frame(timeout=0.15)
            if not ok:
                continue

            # ── inference ─────────────────────────────────────────────────
            net.setInput(make_blob(frame))
            t0   = time.time()
            outs = net.forward(out_names)
            ms   = (time.time() - t0) * 1000
            fps  = 1000.0 / max(ms, 1e-3)
            fps_hist.append(fps)

            pts, count = postprocess(outs, args.threshold)
            result = draw(cv2.resize(frame, (INPUT_W, INPUT_H)), pts, count, ms)

            if writer:
                writer.write(result)

            # ── throttle output ────────────────────────────────────────────
            now = time.time()
            if now - last_emit < min_gap:
                frame_idx += 1
                continue
            last_emit = now

            if args.json_stream:
                b64 = to_b64(result)
                if b64:
                    emit({'type': 'frame', 'jpeg': b64, 'count': count,
                          'ms': round(ms, 1), 'fps': round(fps, 1),
                          'frame': frame_idx})
            else:
                if args.show:
                    cv2.imshow('P2PNet', result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        _stop_event.set(); break
                print(f'\rFrame {frame_idx:6d}  count={count:4d}  '
                      f'{ms:.1f}ms  {fps:.1f}fps', end='', flush=True)

            frame_idx += 1

    finally:
        reader.release()
        if writer: writer.release()
        if args.show and not args.json_stream:
            cv2.destroyAllWindows()

    mean_fps = float(np.mean(fps_hist)) if fps_hist else 0.0
    if args.json_stream:
        emit({'type': 'done', 'frames': frame_idx, 'mean_fps': round(mean_fps, 2)})
    else:
        print(f'\nDone — {frame_idx} frames, mean {mean_fps:.2f} fps')


# ── Single image ──────────────────────────────────────────────────────────────
def run_image(net, out_names, frame, args):
    net.setInput(make_blob(frame))
    t0   = time.time()
    outs = net.forward(out_names)
    ms   = (time.time() - t0) * 1000
    pts, count = postprocess(outs, args.threshold)
    result = draw(cv2.resize(frame, (INPUT_W, INPUT_H)), pts, count, ms)

    if args.json_stream:
        emit({'type': 'frame', 'jpeg': to_b64(result), 'count': count,
              'ms': round(ms, 1), 'fps': 0, 'frame': 0})
        emit({'type': 'done', 'frames': 1, 'mean_fps': 0})
    else:
        cv2.imwrite(args.output, result)
        print(f'count={count}  {ms:.1f}ms  →  {args.output}')
        if args.show:
            cv2.imshow('P2PNet', result); cv2.waitKey(0); cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--image');     g.add_argument('--image-b64')
    g.add_argument('--video');     g.add_argument('--rtsp')
    p.add_argument('--threshold',      type=float, default=0.5)
    p.add_argument('--cpu',            action='store_true')
    p.add_argument('--output',         default='output.jpg')
    p.add_argument('--show',           action='store_true')
    p.add_argument('--save-video',     action='store_true')
    p.add_argument('--rtsp-reconnect', type=int,   default=5)
    p.add_argument('--max-fps',        type=float, default=20.0)
    p.add_argument('--json-stream',    action='store_true')
    return p.parse_args()

def main():
    a = parse_args()

    def die(msg):
        emit({'type': 'error', 'msg': msg}) if a.json_stream \
            else print(msg, file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(a.model): die(f'Model not found: {a.model}')

    net = cv2.dnn.readNet(a.model)
    if a.cpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_TIMVX)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

    names = ['regression', 'classification']

    if a.image:
        if not os.path.exists(a.image): die(f'Not found: {a.image}')
        f = cv2.imread(a.image)
        if f is None: die(f'Cannot read: {a.image}')
        run_image(net, names, f, a)

    elif a.image_b64:
        try:
            s = a.image_b64
            if ',' in s: s = s.split(',', 1)[1]
            f = cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8),
                             cv2.IMREAD_COLOR)
            if f is None: raise ValueError
        except Exception: die('Bad base64 image')
        run_image(net, names, f, a)

    elif a.video is not None:
        src = int(a.video) if str(a.video).isdigit() else a.video
        r   = FrameReader(src)
        if not r.is_opened(): die(f'Cannot open: {a.video}')
        video_loop(net, names, r, a)

    elif a.rtsp:
        r = FrameReader(a.rtsp, is_rtsp=True, reconnect_delay=a.rtsp_reconnect)
        if not r.is_opened(): die(f'Cannot open RTSP: {a.rtsp}')
        video_loop(net, names, r, a, is_rtsp=True)

    else:
        die('Provide --image / --image-b64 / --video / --rtsp')

if __name__ == '__main__':
    main()
