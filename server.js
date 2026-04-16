/**
 * P2PNet Dashboard Server  v4
 * ---------------------------
 * Fixes vs v3
 *   1. spawnLock is now a Promise-based mutex, not a setTimeout.
 *      startInference() awaits killCurrent() to fully complete before
 *      spawning.  Double-spawn is impossible even with rapid clicks.
 *
 *   2. CPU idle: Socket.io ping interval raised to 25 s (default 5 s
 *      hammers the event loop).  No polling timers anywhere.
 *
 *   3. Image path removed — images pass as base64 in the POST body
 *      (memoryStorage), so Multer never touches the disk for images.
 *      Videos still land on disk (OpenCV needs seekable access).
 *
 *   4. killCurrent() returns a Promise that resolves only after the
 *      child process has actually exited (or SIGKILL fires), so the
 *      next spawn never races with the dying process.
 */

const express    = require('express');
const http       = require('http');
const { Server } = require('socket.io');
const multer     = require('multer');
const path       = require('path');
const fs         = require('fs');
const { spawn }  = require('child_process');
const args       = require('minimist')(process.argv.slice(2));

// ── Config ────────────────────────────────────────────────────────────────────
const PORT     = args.port       || 3000;
const MODEL    = args.model      || './p2pnet_sim.onnx';
const PYTHON   = args.python     || 'python3';
const CPU_FLAG = args.cpu        ? ['--cpu'] : [];
const MAX_FPS  = String(args['max-fps'] || 20);
const SCRIPT   = path.join(__dirname, 'p2pnet_dnn.py');

const UPLOAD_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// ── Express / Socket.io ───────────────────────────────────────────────────────
const app    = express();
const server = http.createServer(app);
const io     = new Server(server, {
  maxHttpBufferSize: 1e8,
  pingInterval:      25000,   // ← default 5000 burns CPU; 25 s is fine
  pingTimeout:       20000,
});

app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Multer — memory for images, disk for video ────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits:  { fileSize: 500 * 1024 * 1024 },
  fileFilter: (_req, file, cb) =>
    cb(null, /image\/(jpeg|png|bmp|webp)|video\/(mp4|avi|mkv|mov|webm)/.test(file.mimetype))
});

// ── Process state ─────────────────────────────────────────────────────────────
let pyProc   = null;
let pySource = null;

// Promise-based mutex: awaited by startInference() before spawning.
// Guarantees the previous process is dead before the new one starts.
let killPromise = Promise.resolve();

const KILL_GRACE_MS = 2500;

/**
 * Kill the current process group and return a Promise that resolves
 * only after the process has actually exited (or SIGKILL fires).
 */
function killCurrent() {
  if (!pyProc) return Promise.resolve();

  const proc    = pyProc;
  pyProc        = null;
  pySource      = null;

  return new Promise(resolve => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };

    // Escalate to SIGKILL after grace period
    const timer = setTimeout(() => {
      try { process.kill(-proc.pid, 'SIGKILL'); } catch (_) {}
      finish();
    }, KILL_GRACE_MS);

    proc.once('close', () => { clearTimeout(timer); finish(); });

    try { process.kill(-proc.pid, 'SIGTERM'); } catch (_) { finish(); }
  });
}

/**
 * Start inference.  Awaits the previous kill before spawning — no races.
 */
async function startInference(sourceArgs, label) {
  // Chain onto the existing kill promise so concurrent calls queue up
  killPromise = killPromise.then(() => killCurrent());
  await killPromise;

  if (_serverShuttingDown) return;

  const pyArgs = [
    SCRIPT,
    '--model',     MODEL,
    '--threshold', '0.5',
    '--max-fps',   MAX_FPS,
    '--json-stream',
    ...CPU_FLAG,
    ...sourceArgs,
  ];

  // Log without leaking b64 data
  const safeArgs = pyArgs.map(a => (a.length > 200 ? '<b64-image>' : a));
  console.log(`[inference] spawn: ${PYTHON} ${safeArgs.join(' ')}`);

  pyProc   = spawn(PYTHON, pyArgs, {
    stdio:    ['ignore', 'pipe', 'pipe'],
    detached: true,    // own process group → group kill works
  });
  pySource = label;

  io.emit('status', { running: true, source: label });

  // ── stdout → JSON lines ───────────────────────────────────────────────────
  let buf = '';
  pyProc.stdout.on('data', chunk => {
    buf += chunk.toString();
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        switch (msg.type) {
          case 'frame': io.emit('frame', msg); break;
          case 'done':
            io.emit('done', msg);
            io.emit('status', { running: false, source: null });
            pyProc = null; pySource = null;
            break;
          case 'error':
            io.emit('inferenceError', msg);
            io.emit('status', { running: false, source: null });
            pyProc = null; pySource = null;
            break;
          case 'log': io.emit('log', msg); break;
        }
      } catch (_) {
        if (line) io.emit('log', { msg: line });
      }
    }
  });

  pyProc.stderr.on('data', d => {
    const txt = d.toString().trim();
    if (txt) { console.error('[py]', txt); io.emit('log', { msg: txt }); }
  });

  pyProc.on('close', code => {
    console.log(`[inference] exit code=${code}`);
    if (pyProc) {        // exited on its own (not via killCurrent)
      pyProc = null; pySource = null;
      io.emit('status', { running: false, source: null });
    }
  });
}

// ── REST ──────────────────────────────────────────────────────────────────────
app.post('/api/upload', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file' });

  const isImage = req.file.mimetype.startsWith('image/');

  if (isImage) {
    const b64 = req.file.buffer.toString('base64');
    await startInference(['--image-b64', b64], req.file.originalname);
    return res.json({ ok: true, file: req.file.originalname, type: 'image' });
  }

  // Video → write to disk
  const ext   = path.extname(req.file.originalname);
  const fpath = path.join(UPLOAD_DIR, `upload_${Date.now()}${ext}`);
  try {
    await fs.promises.writeFile(fpath, req.file.buffer);
  } catch (e) {
    return res.status(500).json({ error: 'Cannot save video' });
  }
  await startInference(['--video', fpath], req.file.originalname);
  res.json({ ok: true, file: req.file.originalname, type: 'video' });
});

app.post('/api/rtsp', async (req, res) => {
  const { url, reconnect } = req.body;
  if (!url) return res.status(400).json({ error: 'No URL' });
  await startInference(
    ['--rtsp', url, '--rtsp-reconnect', String(reconnect || 5)], url);
  res.json({ ok: true, url });
});

app.post('/api/webcam', async (req, res) => {
  const idx = req.body.index ?? 0;
  await startInference(['--video', String(idx)], `Webcam ${idx}`);
  res.json({ ok: true, index: idx });
});

app.post('/api/stop', async (req, res) => {
  await killCurrent();
  io.emit('status', { running: false, source: null });
  res.json({ ok: true });
});

app.get('/api/status', (_req, res) =>
  res.json({ running: !!pyProc, source: pySource, model: MODEL }));

// ── Socket.io ─────────────────────────────────────────────────────────────────
io.on('connection', socket => {
  console.log(`[ws] +${socket.id}`);
  socket.emit('status', { running: !!pyProc, source: pySource });
});

// ── Shutdown ──────────────────────────────────────────────────────────────────
let _serverShuttingDown = false;
async function gracefulShutdown(sig) {
  console.log(`\n[server] ${sig} — shutting down`);
  _serverShuttingDown = true;
  await killCurrent();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 3000);
}
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT',  () => gracefulShutdown('SIGINT'));

// ── Start ─────────────────────────────────────────────────────────────────────
server.listen(PORT, '0.0.0.0', () => {
  console.log(`\n╔══════════════════════════════════════════════╗`);
  console.log(`║   P2PNet Dashboard  v4                       ║`);
  console.log(`║   http://0.0.0.0:${PORT}                        ║`);
  console.log(`╚══════════════════════════════════════════════╝\n`);
  console.log(`Model   : ${MODEL}`);
  console.log(`Script  : ${SCRIPT}`);
  console.log(`Python  : ${PYTHON}`);
  console.log(`Max FPS : ${MAX_FPS}\n`);
});
