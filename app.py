import cv2
import json
import numpy as np
from ultralytics import solutions
from imutils.video import VideoStream
import imutils
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import threading
import time
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------------------- RTSP URL ----------------------
RTSP_URL = "rtsp://admin:Deltaits1@192.168.1.111:554/Streaming/Channels/101"

# ---------------------- GLOBALS ----------------------
region_points = []
region_ready = False
config_file = "region_config.json"
count_file = "count_data.json"

# Counter data
counter_data = {
    "in_count": 0,
    "out_count": 0,
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

counter_lock = threading.Lock()
video_stream = None
counter_obj = None
processing = False

# --------------- LOAD/SAVE FUNCTIONS ----------------
def load_region():
    global region_points, region_ready
    try:
        with open(config_file, "r") as f:
            data = json.load(f)
            region_points = [tuple(p) for p in data.get("region", [])]
            if len(region_points) == 4:
                region_ready = True
                print("[INFO] Region loaded from config")
    except FileNotFoundError:
        print("[INFO] No region config found")

def save_region():
    data = {"region": region_points}
    with open(config_file, "w") as f:
        json.dump(data, f, indent=4)
    print("[INFO] Region saved")

def load_count_data():
    global counter_data
    try:
        with open(count_file, "r") as f:
            counter_data = json.load(f)
            print(f"[INFO] Count data loaded: IN={counter_data['in_count']}, OUT={counter_data['out_count']}")
    except FileNotFoundError:
        print("[INFO] No count data found, starting fresh")
        save_count_data()

def save_count_data():
    with counter_lock:
        counter_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(count_file, "w") as f:
            json.dump(counter_data, f, indent=4)

def update_counts(in_count, out_count):
    with counter_lock:
        counter_data["in_count"] = in_count
        counter_data["out_count"] = out_count
        save_count_data()
        socketio.emit('count_update', counter_data)

# --------------- VIDEO PROCESSING ----------------
def generate_frames():
    global video_stream, counter_obj, region_ready, processing
    
    print("[INFO] Starting video stream...")
    video_stream = VideoStream(src=RTSP_URL).start()
    time.sleep(2.0)
    
    frame_count = 0
    
    while processing:
        frame = video_stream.read()
        if frame is None:
            continue
            
        frame_count += 1
        if frame_count % 3 != 0:
            continue
            
        frame = imutils.resize(frame, width=1020)
        
        # Draw region
        if len(region_points) > 0:
            for p in region_points:
                cv2.circle(frame, p, 5, (0, 255, 255), -1)
            if len(region_points) > 1:
                cv2.polylines(frame, [np.array(region_points)], False, (0, 255, 0), 2)
        
        # Apply counter if region ready
        if region_ready and counter_obj is not None:
            try:
                results = counter_obj(frame)
                output_frame = results.plot_im
                
                # Extract counts from counter object
                if hasattr(counter_obj, 'in_count') and hasattr(counter_obj, 'out_count'):
                    update_counts(counter_obj.in_count, counter_obj.out_count)
                
            except Exception as e:
                print(f"[ERROR] Counter processing: {e}")
                output_frame = frame
        else:
            output_frame = frame
            
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if video_stream:
        video_stream.stop()

# --------------- FLASK ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    with counter_lock:
        return jsonify(counter_data)

@app.route('/adjust_count', methods=['POST'])
def adjust_count():
    data = request.json
    count_type = data.get('type')  # 'in' or 'out'
    action = data.get('action')    # 'increment' or 'decrement'
    
    with counter_lock:
        if count_type == 'in':
            if action == 'increment':
                counter_data['in_count'] += 1
            elif action == 'decrement':
                counter_data['in_count'] = max(0, counter_data['in_count'] - 1)
        elif count_type == 'out':
            if action == 'increment':
                counter_data['out_count'] += 1
            elif action == 'decrement':
                counter_data['out_count'] = max(0, counter_data['out_count'] - 1)
        
        save_count_data()
        socketio.emit('count_update', counter_data)
    
    return jsonify({"status": "success", "data": counter_data})

@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    with counter_lock:
        counter_data['in_count'] = 0
        counter_data['out_count'] = 0
        save_count_data()
        socketio.emit('count_update', counter_data)
    
    return jsonify({"status": "success", "data": counter_data})

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global processing, counter_obj, region_ready
    
    if not processing:
        load_region()
        
        if region_ready:
            processing = True
            counter_obj = solutions.ObjectCounter(
                show=False,
                conf=0.5,
                region=region_points,
                model="best.pt"
            )
            return jsonify({"status": "success", "message": "Processing started"})
        else:
            return jsonify({"status": "error", "message": "Region not configured"})
    
    return jsonify({"status": "info", "message": "Already processing"})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global processing
    processing = False
    return jsonify({"status": "success", "message": "Processing stopped"})

# --------------- HTML TEMPLATE ----------------
@app.route('/template')
def get_template():
    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RTSP Object Counter Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .video-section {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .controls-section {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .count-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: transform 0.3s;
        }
        
        .count-card:hover {
            transform: translateY(-5px);
        }
        
        .count-card.in-card {
            border-left: 5px solid #4CAF50;
        }
        
        .count-card.out-card {
            border-left: 5px solid #f44336;
        }
        
        .count-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        
        .count-label {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        
        .count-value {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
            color: #667eea;
        }
        
        .count-controls {
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-increment {
            background: #4CAF50;
            color: white;
        }
        
        .btn-decrement {
            background: #ff9800;
            color: white;
        }
        
        .btn-reset {
            background: #f44336;
            color: white;
            width: 100%;
            margin-top: 10px;
        }
        
        .info-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        
        .info-row:last-child {
            border-bottom: none;
        }
        
        .info-label {
            font-weight: bold;
            color: #666;
        }
        
        .info-value {
            color: #333;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active {
            background: #4CAF50;
            animation: pulse 2s infinite;
        }
        
        .status-inactive {
            background: #ccc;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        @media (max-width: 1024px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 RTSP Object Counter Dashboard</h1>
            <p>Real-time monitoring and counting system</p>
        </div>
        
        <div class="dashboard">
            <div class="video-section">
                <h2 style="margin-bottom: 15px;">Live Feed</h2>
                <div class="video-container">
                    <img src="/video_feed" alt="Video Feed">
                </div>
            </div>
            
            <div class="controls-section">
                <div class="count-card in-card">
                    <div class="count-header">
                        <span class="count-label">👇 IN Count</span>
                    </div>
                    <div class="count-value" id="in-count">0</div>
                    <div class="count-controls">
                        <button class="btn btn-increment" onclick="adjustCount('in', 'increment')">+ Add</button>
                        <button class="btn btn-decrement" onclick="adjustCount('in', 'decrement')">- Remove</button>
                    </div>
                </div>
                
                <div class="count-card out-card">
                    <div class="count-header">
                        <span class="count-label">👆 OUT Count</span>
                    </div>
                    <div class="count-value" id="out-count">0</div>
                    <div class="count-controls">
                        <button class="btn btn-increment" onclick="adjustCount('out', 'increment')">+ Add</button>
                        <button class="btn btn-decrement" onclick="adjustCount('out', 'decrement')">- Remove</button>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3 style="margin-bottom: 15px;">System Info</h3>
                    <div class="info-row">
                        <span class="info-label">Status</span>
                        <span class="info-value">
                            <span class="status-indicator status-active"></span>
                            Active
                        </span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Total</span>
                        <span class="info-value" id="total-count">0</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Last Updated</span>
                        <span class="info-value" id="last-updated">-</span>
                    </div>
                    <button class="btn btn-reset" onclick="resetCounts()">🔄 Reset All Counts</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // Socket.IO event listeners
        socket.on('connect', function() {
            console.log('Connected to server');
            loadCounts();
        });
        
        socket.on('count_update', function(data) {
            updateDisplay(data);
        });
        
        // Load initial counts
        function loadCounts() {
            fetch('/get_counts')
                .then(response => response.json())
                .then(data => updateDisplay(data));
        }
        
        // Update display
        function updateDisplay(data) {
            document.getElementById('in-count').textContent = data.in_count;
            document.getElementById('out-count').textContent = data.out_count;
            document.getElementById('total-count').textContent = data.in_count + data.out_count;
            document.getElementById('last-updated').textContent = data.last_updated;
        }
        
        // Adjust count
        function adjustCount(type, action) {
            fetch('/adjust_count', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ type: type, action: action })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    updateDisplay(data.data);
                }
            });
        }
        
        // Reset counts
        function resetCounts() {
            if (confirm('Are you sure you want to reset all counts?')) {
                fetch('/reset_counts', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        updateDisplay(data.data);
                    }
                });
            }
        }
        
        // Auto-refresh counts every 2 seconds
        setInterval(loadCounts, 2000);
    </script>
</body>
</html>
    '''
    return html

# --------------- MAIN ----------------
if __name__ == '__main__':
    load_count_data()
    
    # Start processing in background
    processing = True
    load_region()
    if region_ready:
        counter_obj = solutions.ObjectCounter(
            show=False,
            conf=0.5,
            region=region_points,
            model="best.pt"
        )
    
    print("[INFO] Starting Flask app on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
