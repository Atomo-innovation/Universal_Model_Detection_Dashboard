# ASNN Detection Dashboard

A web-based detection dashboard that runs as a **Node.js server** on your aarch64 device.
Access it from any browser on your network — laptop, phone, tablet — via the device's IP address.

---

## Architecture

```
aarch64 Device
├── Node.js server (server.js)        ← HTTP + WebSocket server
│   ├── Serves dashboard UI           ← public/index.html
│   ├── Scans ./models/ directory     ← auto-detects .nb + .so + data.yaml
│   ├── Handles file uploads          ← video / image to ./uploads/
│   ├── Spawns Python detect.py       ← actual ASNN inference
│   └── Streams JSON via WebSocket    ← real-time to browser
│
└── Python detect.py                  ← ASNN inference
    ├── Reads model (.nb) + lib (.so)
    ├── Reads class names from data.yaml
    ├── Runs inference (ASNN / simulated)
    └── Outputs JSON lines to stdout  ← server reads and forwards

Remote Devices (any browser)
├── http://192.168.1.x:8080           ← access dashboard
└── WebSocket ws://192.168.1.x:8080  ← real-time inference stream
```

---

## Directory Structure

```
asnn-dashboard/
├── server.js           ← Node.js server
├── detect.py           ← Python inference script
├── package.json        ← Node dependencies
├── start.sh            ← Setup & start script
├── public/
│   └── index.html      ← Dashboard UI (served by Node)
├── models/             ← Place model directories here
│   ├── car/
│   │   ├── car.nb
│   │   ├── libnn_car.so
│   │   └── data.yaml
│   ├── person/
│   │   ├── person.nb
│   │   ├── libnn_person.so
│   │   └── data.yaml
│   └── vehicle/
│       ├── vehicle.nb
│       ├── libnn_vehicle.so
│       └── data.yaml   ← { names: [Car, Truck, Bus, Motorcycle] }
└── uploads/            ← Auto-created, stores uploaded video/image files
```

### data.yaml format
```yaml
nc: 4
names:
  - Car
  - Truck
  - Bus
  - Motorcycle
```
The server reads `data.yaml` to auto-populate class names, NUM_CLS, and LISTSIZE.

---

## Setup & Run

### 1. Install Node.js (if not installed)
```bash
# On aarch64 Debian/Ubuntu:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 2. Install dependencies
```bash
cd asnn-dashboard
npm install
```

### 3. Add your models
```bash
mkdir -p models/car
cp /path/to/car.nb         models/car/
cp /path/to/libnn_car.so   models/car/
cp /path/to/data.yaml      models/car/
```

### 4. Start the server
```bash
# Default (port 8080)
./start.sh

# Custom port
PORT=9090 ./start.sh

# Custom models directory
MODELS_DIR=/mnt/sdcard/models ./start.sh
```

### 5. Access from any device on the network
```
http://<device-ip>:8080
```
Find your device IP: `hostname -I`

---

## Environment Variables

| Variable        | Default              | Description                              |
|-----------------|----------------------|------------------------------------------|
| `PORT`          | `8080`               | HTTP/WS server port                      |
| `MODELS_DIR`    | `./models`           | Path to models directory                 |
| `UPLOADS_DIR`   | `./uploads`          | Path for uploaded video/image files      |
| `DETECT_SCRIPT` | `./detect.py`        | Path to Python inference script          |

---

## API Reference

| Method | Endpoint                      | Description                    |
|--------|-------------------------------|--------------------------------|
| GET    | `/api/models`                 | List all detected models       |
| GET    | `/api/system`                 | System info (arch, IPs, etc)   |
| POST   | `/api/upload`                 | Upload video or image file     |
| DELETE | `/api/upload/:filename`       | Delete uploaded file           |
| POST   | `/api/inference/start`        | Create inference session       |
| POST   | `/api/inference/stop/:sid`    | Stop inference session         |
| GET    | `/api/inference/sessions`     | List active sessions           |

### WebSocket Messages

**Client → Server:**
```json
{ "type": "start", "sessionId": "..." }
{ "type": "stop",  "sessionId": "..." }
{ "type": "ping" }
```

**Server → Client:**
```json
{ "type": "inference", "frame": 42, "fps": 14.2, "inference_ms": 18.5,
  "detections": [{ "class_id": 0, "class_name": "Car", "score": 0.92,
                   "box": [0.12, 0.34, 0.56, 0.78] }] }

{ "type": "log", "level": "info", "message": "..." }
{ "type": "status", "status": "running" }
{ "type": "models_updated", "models": [...] }
```

---

## Adding detect.py Output Support

Your `detect.py` must print JSON lines to stdout:
```python
import json

# Per frame:
result = {
    "frame": frame_num,
    "fps": 14.5,
    "inference_ms": 18.2,
    "detections": [
        { "class_id": 0, "class_name": "Car", "score": 0.92,
          "box": [0.1, 0.2, 0.4, 0.6] }  # normalized [x1,y1,x2,y2]
    ]
}
print(json.dumps(result), flush=True)  # flush=True is critical!
```

If `detect.py` is not found, the server runs in **simulation mode** automatically.

---

## Running as a System Service (optional)

```bash
sudo nano /etc/systemd/system/asnn-dashboard.service
```

```ini
[Unit]
Description=ASNN Detection Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/asnn-dashboard
ExecStart=/usr/bin/node server.js
Restart=on-failure
Environment=PORT=8080
Environment=MODELS_DIR=/home/pi/models

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable asnn-dashboard
sudo systemctl start  asnn-dashboard
sudo systemctl status asnn-dashboard
```

---

## Firewall (if needed)

```bash
sudo ufw allow 8080/tcp
# or for iptables:
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```
