import cv2
import json
import os
import time
from ultralytics import solutions
from imutils.video import VideoStream

# ----------------------------------------------------
# CONFIG FILE
# ----------------------------------------------------
CONFIG_FILE = "line_coordinates.json"

# ----------------------------------------------------
# LOAD / SAVE LINE
# ----------------------------------------------------
def load_line():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)["points"]
        except:
            pass
    return [(20, 400), (1080, 400)]  # default line

def save_line(line):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"points": line}, f)
    print("✅ Saved line:", line)

# ----------------------------------------------------
# LOAD PREVIOUS LINE
# ----------------------------------------------------
region_points = load_line()
print("Loaded Line:", region_points)

# ----------------------------------------------------
# OBJECT COUNTER
# ----------------------------------------------------
counter = solutions.ObjectCounter(
    show=False,
    region=region_points,
    model="best_float32.tflite",
    conf=0.80,
)

# ----------------------------------------------------
# RTSP STREAM
# ----------------------------------------------------
RTSP_URL = "rtsp://admin:admin%401234@41.139.156.38:554/Streaming/Channels/10111"

cap = VideoStream(src=RTSP_URL).start()
time.sleep(2.0)
print("✅ RTSP stream started")

# ----------------------------------------------------
# MOUSE HANDLER
# ----------------------------------------------------
points = []
temp_frame = None

def mouse_draw(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print("Clicked:", (x, y))

        if len(points) == 2:
            counter.region = points.copy()
            counter.counted_ids.clear()
            save_line(points.copy())
            points = []

cv2.namedWindow("Object Counter")
cv2.setMouseCallback("Object Counter", mouse_draw)

print("""
CONTROLS:
========
Left Click twice  → draw new line
R                 → reset line
Q                 → quit
""")

# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
while True:

    frame = cap.read()

    if frame is None:
        print("Reconnecting RTSP...")
        cap.stop()
        time.sleep(2)
        cap = VideoStream(src=RTSP_URL).start()
        continue

    frame = cv2.resize(frame, (1020, 600))

    results = counter(frame)

    output = results.plot_im.copy()

    # Instructions
    cv2.putText(output, "Click 2 points to draw line | Press R to reset | Q quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    cv2.imshow("Object Counter", output)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Reset Line
    elif key == ord("r"):
        print("🟡 Reset mode: Click 2 new points")
        points = []
        counter.region = []
        counter.counted_ids.clear()

# ----------------------------------------------------
# CLEANUP
# ----------------------------------------------------
cap.stop()
cv2.destroyAllWindows()
