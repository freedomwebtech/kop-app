import cv2
import importlib.util
import json
import os
import time
from ultralytics import YOLO
from imutils.video import VideoStream   # ✅ ADDED

# -----------------------------
# LOAD CLASS
# -----------------------------
LOCAL_FILE = r"C:\Users\freed\Downloads\ultrmycode\testultra2.py"
spec = importlib.util.spec_from_file_location("local_app", LOCAL_FILE)
local_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_app)
ObjectCounter = local_app.ObjectCounter

# -----------------------------
# CONFIG FILE PATH
# -----------------------------
CONFIG_FILE = "line_coordinates.json"

# -----------------------------
# LOAD/SAVE FUNCTIONS
# -----------------------------
def load_coordinates():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return data.get('region_points', [(20, 400), (1080, 400)])
        except:
            return [(20, 400), (1080, 400)]
    return [(20, 400), (1080, 400)]

def save_coordinates(coords):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({'region_points': coords}, f)
    print(f"Line coordinates saved: {coords}")

# -----------------------------
# COUNTING LINE
# -----------------------------
region_points = load_coordinates()
print(f"Loaded line coordinates: {region_points}")

# -----------------------------
# INIT COUNTER
# -----------------------------
counter = ObjectCounter(
    model=r"C:\Users\freed\Downloads\ultrmycode\best_float32.tflite",
    region=region_points,
    classes=[0],
    conf=0.70,
    show=False
)

# -----------------------------
# MOUSE
# -----------------------------
points = []
temp = None

def draw(event, x, y, flags, param):
    global points, temp
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print("Clicked:", (x, y))
        if len(points) == 2:
            counter.region = points.copy()
            counter.counted_ids.clear()
            save_coordinates(points.copy())
            points = []

cv2.namedWindow("Counter")
cv2.setMouseCallback("Counter", draw)

# -----------------------------
# RTSP / VIDEO SOURCE
# -----------------------------
USE_RTSP = True   # ✅ change to False to use file

RTSP_URL = "rtsp://username:password@ip:port/stream"   # 🛑 PUT YOUR RTSP HERE
VIDEO_FILE = r"C:\Users\freed\Videos\new kol\topview\mynew\My Movie3.mp4"

if USE_RTSP:
    cap = VideoStream(src=RTSP_URL).start()   # ✅ imutils RTSP stream
    time.sleep(2.0)   # allow camera to warm up
    print("RTSP stream started")
else:
    cap = cv2.VideoCapture(VIDEO_FILE)
    print("Video file loaded")

# -----------------------------
# LOOP
# -----------------------------
frame_count = 0

print("\n=== CONTROLS ===")
print("Left Click: Set counting line (2 clicks)")
print("Press 'r': Reset to default line")
print("Press 'q': Quit")
print("================\n")

while True:

    # ---- READ FRAME ----
    if USE_RTSP:
        frame = cap.read()
        if frame is None:
            print("RTSP Frame not received, reconnecting...")
            cap.stop()
            time.sleep(2)
            cap = VideoStream(src=RTSP_URL).start()
            continue
    else:
        ret, frame = cap.read()
        if not ret:
            break

    frame_count += 1

    # Skip frames for performance
    if frame_count % 2 != 0:
        continue

    # Resize
    frame = cv2.resize(frame, (1020, 600))

    # Detect + count
    result = counter(frame)

    temp = result.plot_im.copy()

    # Overlay instructions
    cv2.putText(temp, "Click 2 points to set line | Press 'r' to reset | 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show
    cv2.imshow("Counter", temp)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Reset
    elif key == ord("r"):
        default_coords = [(20, 400), (1080, 400)]
        counter.region = default_coords
        counter.counted_ids.clear()
        save_coordinates(default_coords)
        points = []
        print("Line reset to default coordinates")

# -----------------------------
# CLEANUP
# -----------------------------
if USE_RTSP:
    cap.stop()
else:
    cap.release()

cv2.destroyAllWindows()
