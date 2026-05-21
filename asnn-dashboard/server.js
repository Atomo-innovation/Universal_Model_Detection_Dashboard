/**
 * ASNN Detection Dashboard Server
 * Runs on aarch64 device, accessible from any browser on the network
 * 
 * Features:
 *  - Serves dashboard UI over HTTP
 *  - Scans /models directory for available models (reads data.yaml)
 *  - Accepts file uploads (video/image)
 *  - Spawns Python inference processes
 *  - Streams output frames + logs via WebSocket
 *  - Streams RTSP/webcam output back to browser
 */

const express    = require('express');
const http       = require('http');
const WebSocket  = require('ws');
const multer     = require('multer');
const path       = require('path');
const fs         = require('fs');
const { spawn, execSync }  = require('child_process');
const { v4: uuidv4 } = require('uuid');
const chokidar   = require('chokidar');

// ── Try to parse YAML (data.yaml for model metadata) ────────────
let YAML;
try { YAML = require('yaml'); } catch(e) { YAML = null; }

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocket.Server({ server });

const PORT         = process.env.PORT || 8080;
const MODELS_DIR   = process.env.MODELS_DIR  || path.join(__dirname, 'models');
const UPLOADS_DIR  = process.env.UPLOADS_DIR || path.join(__dirname, 'uploads');
const PUBLIC_DIR   = path.join(__dirname, 'public');
const DETECT_SCRIPT = process.env.DETECT_SCRIPT || path.join(__dirname, 'detect.py');

// Ensure dirs exist
[MODELS_DIR, UPLOADS_DIR, PUBLIC_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));

// ── Active inference sessions ────────────────────────────────────
const sessions = new Map();  // sessionId -> { proc, ws, type }

// ── Multer upload config ─────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename:    (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${Date.now()}-${uuidv4().slice(0,8)}${ext}`);
  }
});
const upload = multer({ storage, limits: { fileSize: 2 * 1024 * 1024 * 1024 } }); // 2GB

// ── Express middleware ───────────────────────────────────────────
app.use(express.json());
app.use(express.static(PUBLIC_DIR));        // root: /logo.png, /index.html etc.
app.use('/public', express.static(PUBLIC_DIR));  // keep /public/ for compat
app.use('/uploads', express.static(UPLOADS_DIR));

// ── API: List available models ───────────────────────────────────
app.get('/api/models', (req, res) => {
  const models = scanModels();
  res.json({ models });
});

// ── API: Upload file ─────────────────────────────────────────────
app.post('/api/upload', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
  res.json({
    filename: req.file.filename,
    originalname: req.file.originalname,
    path: req.file.path,
    size: req.file.size
  });
});

// ── API: Delete uploaded file ────────────────────────────────────
app.delete('/api/upload/:filename', (req, res) => {
  const fp = path.join(UPLOADS_DIR, path.basename(req.params.filename));
  try { fs.unlinkSync(fp); res.json({ ok: true }); }
  catch(e) { res.status(404).json({ error: 'File not found' }); }
});

// ── API: System info ─────────────────────────────────────────────
app.get('/api/system', (req, res) => {
  let arch = 'unknown', hostname = 'unknown', ip = [];
  try { arch     = execSync('uname -m').toString().trim(); } catch(e){}
  try { hostname = execSync('hostname').toString().trim();  } catch(e){}
  try {
    const raw = execSync("hostname -I 2>/dev/null || ip addr show | grep 'inet ' | awk '{print $2}' | cut -d/ -f1").toString().trim();
    ip = raw.split(/\s+/).filter(Boolean);
  } catch(e){}
  res.json({ arch, hostname, ip, port: PORT, uptime: process.uptime() });
});

// ── API: Start inference session ─────────────────────────────────
app.post('/api/inference/start', (req, res) => {
  const { modelName, inputType, inputValue, objThresh, nmsThresh, platform, logLevel, sessionId: existingId } = req.body;

  // Find model
  const models = scanModels();
  const model  = models.find(m => m.name === modelName);
  if (!model) return res.status(404).json({ error: `Model '${modelName}' not found` });

  // Stop existing session if reusing
  const sid = existingId || uuidv4();
  if (sessions.has(sid)) stopSession(sid);

  // Build Python args
  const args = buildPythonArgs({ model, inputType, inputValue, objThresh, nmsThresh, platform, logLevel });

  // Store session metadata (process spawned when WS connects)
  sessions.set(sid, { model, args, inputType, inputValue, status: 'pending', proc: null, ws: null });

  res.json({ sessionId: sid, command: `python3 detect.py ${args.join(' ')}` });
});

// ── API: Stop inference session ──────────────────────────────────
app.post('/api/inference/stop/:sid', (req, res) => {
  stopSession(req.params.sid);
  res.json({ ok: true });
});

// ── API: List active sessions ────────────────────────────────────
app.get('/api/inference/sessions', (req, res) => {
  const list = [];
  sessions.forEach((v, k) => list.push({ id: k, status: v.status, model: v.model?.name, inputType: v.inputType }));
  res.json({ sessions: list });
});

// ── Serve main dashboard HTML ────────────────────────────────────
app.get('/', (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

// Catch-all — only redirect HTML navigation, not assets
app.get('*', (req, res) => {
  const ext = require('path').extname(req.path);
  if (ext && ext !== '.html') return res.status(404).send('Not found');
  res.sendFile(require('path').join(PUBLIC_DIR, 'index.html'));
});

// ── WebSocket: real-time inference stream ────────────────────────
wss.on('connection', (ws, req) => {
  console.log(`[WS] Client connected from ${req.socket.remoteAddress}`);

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch(e) { return; }

    switch(msg.type) {
      case 'attach':   handleAttach(ws, msg);   break;
      case 'start':    handleStart(ws, msg);     break;
      case 'stop':     handleStop(ws, msg);      break;
      case 'ping':     ws.send(JSON.stringify({ type: 'pong' })); break;
    }
  });

  ws.on('close', () => {
    console.log('[WS] Client disconnected');
    // Don't stop inference on disconnect — allow re-attach
  });

  ws.on('error', (e) => console.error('[WS] Error:', e.message));
});

// ── WS: attach to existing session ──────────────────────────────
function handleAttach(ws, msg) {
  const { sessionId } = msg;
  const session = sessions.get(sessionId);
  if (!session) { ws.send(JSON.stringify({ type: 'error', message: 'Session not found' })); return; }
  session.ws = ws;
  wsend(ws, { type: 'attached', sessionId, status: session.status });
}

// ── WS: start inference ──────────────────────────────────────────
function handleStart(ws, msg) {
  const { sessionId } = msg;
  let session = sessions.get(sessionId);
  if (!session) { wsend(ws, { type: 'error', message: 'Call /api/inference/start first' }); return; }

  session.ws = ws;
  session.status = 'running';
  sessions.set(sessionId, session);

  wsend(ws, { type: 'status', status: 'starting', message: 'Spawning inference process...' });

  // Spawn Python detect.py
  spawnInference(sessionId, session);
}

// ── WS: stop inference ───────────────────────────────────────────
function handleStop(ws, msg) {
  stopSession(msg.sessionId);
  wsend(ws, { type: 'status', status: 'stopped' });
}

// ── Spawn Python inference process ──────────────────────────────
function spawnInference(sid, session) {
  const { args, model, inputType, ws } = session;

  // Check if detect.py exists
  if (!fs.existsSync(DETECT_SCRIPT)) {
    // Use mock script for development/demo
    wsend(ws, { type: 'log', level: 'warn', message: `detect.py not found at ${DETECT_SCRIPT} — using simulation mode` });
    startSimulation(sid, session);
    return;
  }

  let proc;
  try {
    proc = spawn('python3', [DETECT_SCRIPT, ...args], {
      cwd: path.dirname(DETECT_SCRIPT),
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });
  } catch(e) {
    wsend(ws, { type: 'error', message: `Failed to spawn: ${e.message}` });
    return;
  }

  session.proc = proc;
  sessions.set(sid, session);

  wsend(ws, { type: 'status', status: 'running', pid: proc.pid });

  // Stream stdout (JSON lines from detect.py)
  let buf = '';
  proc.stdout.on('data', (chunk) => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();
    lines.forEach(line => {
      if (!line.trim()) return;
      // Try JSON (structured output), else plain log
      try {
        const data = JSON.parse(line);
        wsend(ws, { type: 'inference', ...data });
      } catch(e) {
        wsend(ws, { type: 'log', level: 'info', message: line });
      }
    });
  });

  proc.stderr.on('data', (chunk) => {
    wsend(ws, { type: 'log', level: 'stderr', message: chunk.toString() });
  });

  proc.on('close', (code) => {
    session.status = 'stopped';
    session.proc   = null;
    wsend(ws, { type: 'status', status: 'stopped', exitCode: code });
    console.log(`[${sid}] Process exited with code ${code}`);
  });

  proc.on('error', (e) => {
    wsend(ws, { type: 'error', message: e.message });
    session.status = 'error';
  });
}

// ── Simulation mode (when no real detect.py) ────────────────────
function startSimulation(sid, session) {
  const { model, inputType, ws } = session;
  let frame = 0;
  const classes = model.classes || ['Object'];

  wsend(ws, { type: 'status', status: 'running', simulated: true });

  const interval = setInterval(() => {
    const s = sessions.get(sid);
    if (!s || s.status !== 'running') { clearInterval(interval); return; }

    frame++;
    const dets = [];
    const count = Math.random() > 0.4 ? Math.floor(Math.random() * 3) + 1 : 0;
    for (let i = 0; i < count; i++) {
      const cls = Math.floor(Math.random() * classes.length);
      const score = 0.4 + Math.random() * 0.55;
      const x1 = Math.random() * 0.6, y1 = Math.random() * 0.6;
      dets.push({
        class_id: cls,
        class_name: classes[cls],
        score: parseFloat(score.toFixed(3)),
        box: [
          parseFloat(x1.toFixed(4)), parseFloat(y1.toFixed(4)),
          parseFloat(Math.min(x1 + 0.1 + Math.random() * 0.25, 1).toFixed(4)),
          parseFloat(Math.min(y1 + 0.1 + Math.random() * 0.25, 1).toFixed(4))
        ]
      });
    }

    wsend(ws, {
      type: 'inference',
      frame,
      fps: parseFloat((15 + Math.random() * 10).toFixed(1)),
      inference_ms: parseFloat((8 + Math.random() * 12).toFixed(1)),
      detections: dets,
      simulated: true
    });

    if (frame % 30 === 0) {
      wsend(ws, { type: 'log', level: 'info', message: `[SIM] Frame ${frame} | ${dets.length} detections` });
    }
  }, 66); // ~15fps simulation

  session.simInterval = interval;
  sessions.set(sid, session);
}

// ── Stop a session ───────────────────────────────────────────────
function stopSession(sid) {
  const session = sessions.get(sid);
  if (!session) return;

  if (session.proc) {
    try { session.proc.kill('SIGTERM'); } catch(e) {}
    session.proc = null;
  }
  if (session.simInterval) {
    clearInterval(session.simInterval);
    session.simInterval = null;
  }
  session.status = 'stopped';
  sessions.set(sid, session);
  console.log(`[${sid}] Session stopped`);
}

// ── Scan /models directory ───────────────────────────────────────
function scanModels() {
  const models = [];
  if (!fs.existsSync(MODELS_DIR)) return models;

  const dirs = fs.readdirSync(MODELS_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name);

  for (const name of dirs) {
    const dir = path.join(MODELS_DIR, name);
    const files = fs.readdirSync(dir);

    // Find .nb file
    const nbFile  = files.find(f => f.endsWith('.nb'));
    // Find .so library
    const soFile  = files.find(f => f.endsWith('.so'));
    // Find data.yaml
    const yamlFile = files.find(f => f === 'data.yaml' || f === 'dataset.yaml' || f.endsWith('.yaml'));

    if (!nbFile || !soFile) continue; // skip incomplete model dirs

    const model = {
      name,
      dir,
      nb:  nbFile,
      lib: soFile,
      nb_path:  path.join(dir, nbFile),
      lib_path: path.join(dir, soFile),
      classes:  [name.charAt(0).toUpperCase() + name.slice(1)],
      num_cls:  1,
      listsize: 65,
      yaml:     yamlFile || null
    };

    // Parse data.yaml if present
    if (yamlFile && YAML) {
      try {
        const raw = fs.readFileSync(path.join(dir, yamlFile), 'utf8');
        const parsed = YAML.parse(raw);
        if (parsed.names) {
          const names = Array.isArray(parsed.names) ? parsed.names : Object.values(parsed.names);
          model.classes = names;
          model.num_cls = names.length;
          model.listsize = model.num_cls + 64; // NUM_CLS + 64 (4 * 16 for DFL)
        }
        if (parsed.nc) model.num_cls = parsed.nc;
      } catch(e) {
        console.warn(`[models] Failed to parse ${yamlFile} in ${name}: ${e.message}`);
      }
    }

    models.push(model);
  }

  return models;
}

// ── Build Python CLI args ────────────────────────────────────────
function buildPythonArgs({ model, inputType, inputValue, objThresh, nmsThresh, platform, logLevel }) {
  const args = [
    '--model',   model.nb_path,
    '--library', model.lib_path,
    '--level',   String(logLevel || 0),
  ];

  if (inputType === 'rtsp') {
    args.push('--type', 'rtsp', '--device', inputValue);
  } else if (inputType === 'webcam') {
    const [capType, devNum] = (inputValue || 'usb:0').split(':');
    args.push('--type', capType || 'usb', '--device', devNum || '0');
  } else if (inputType === 'video') {
    args.push('--type', 'video', '--device', inputValue);
  } else if (inputType === 'image') {
    args.push('--type', 'image', '--device', inputValue);
  }

  if (objThresh) args.push('--obj-thresh', String(objThresh));
  if (nmsThresh) args.push('--nms-thresh', String(nmsThresh));
  if (platform)  args.push('--platform',   platform);

  return args;
}

// ── Watch models dir for changes ─────────────────────────────────
chokidar.watch(MODELS_DIR, { depth: 1, ignoreInitial: true })
  .on('addDir', (p) => {
    console.log(`[models] New model directory detected: ${p}`);
    broadcastModels();
  })
  .on('add', (p) => {
    if (p.endsWith('.nb') || p.endsWith('.so') || p.endsWith('.yaml')) {
      console.log(`[models] New model file: ${p}`);
      broadcastModels();
    }
  });

function broadcastModels() {
  const models = scanModels();
  const msg = JSON.stringify({ type: 'models_updated', models });
  wss.clients.forEach(c => { if (c.readyState === WebSocket.OPEN) c.send(msg); });
}

// ── Helper: safe WS send ─────────────────────────────────────────
function wsend(ws, data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify(data)); } catch(e) {}
  }
}

// ── Start server ─────────────────────────────────────────────────
server.listen(PORT, '0.0.0.0', () => {
  console.log('\n╔══════════════════════════════════════════════════╗');
  console.log('║         ASNN DETECTION DASHBOARD SERVER          ║');
  console.log('╠══════════════════════════════════════════════════╣');
  console.log(`║  HTTP   : http://0.0.0.0:${PORT}                   ║`);
  console.log(`║  Models : ${MODELS_DIR}`);
  console.log(`║  Uploads: ${UPLOADS_DIR}`);
  console.log('╠══════════════════════════════════════════════════╣');

  // Print all network IPs
  try {
    const ips = execSync("hostname -I 2>/dev/null || ip addr show | grep 'inet ' | awk '{print $2}' | cut -d/ -f1").toString().trim().split(/\s+/);
    ips.forEach(ip => {
      if (ip) console.log(`║  Access : http://${ip}:${PORT}`);
    });
  } catch(e) {}

  console.log('╚══════════════════════════════════════════════════╝\n');
  console.log(`Models found: ${scanModels().length}`);
  console.log('Press Ctrl+C to stop\n');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  sessions.forEach((_, sid) => stopSession(sid));
  server.close(() => process.exit(0));
});
