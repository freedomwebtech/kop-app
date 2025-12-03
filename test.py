import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model (pre-trained on COCO)
model = YOLO('yolo12n.pt')
names = model.names  # Class name mapping

# Video source
cap = cv2.VideoCapture("vid.mp4")  # Change to 0 for webcam



frame_count = 0

hist={}
counted=set()
in_count = 0
out_count = 0
line_p1 = (706, 216)
line_p2 = (983, 311)


# Mouse callback for pixel debug (optional)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

#def counter(x,y,x1,y1,x2,y2):
#    return(x-x1)*(y2-y1)*(y-y1)*(x2-x1)
def counter(px, py, x1, y1, x2, y2):
    # Returns positive or negative depending on the side of the line
    return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue  # Skip frames for faster processing

    # Resize frame
    frame = cv2.resize(frame, (1020,600))

    # Run YOLOv8 tracking (only on car=2, motorcycle=3)
    results = model.track(frame, persist=True, classes=[0])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            name = names[class_id]
            if track_id in hist:
                prev_cx,prev_cy=hist[track_id]
                prev_side=counter(prev_cx,prev_cy,*line_p1,*line_p2)
                curr_side=counter(cx,cy,*line_p1,*line_p2)
                if prev_side*curr_side < 0 and track_id not in counted:
                    if curr_side<0:
                        direction="IN"
                        in_count+=1
                    else:
                        direction="OUT"
                        out_count+=1
                    counted.add(track_id)    

            # Draw box and label
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
            hist[track_id]=(cx,cy)
        
            
               

          
        

          

    cvzone.putTextRect(frame, f'IN: {in_count}', (50, 30), 2, 2, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (50, 80), 2, 2, colorR=(0, 0, 255))
    cv2.line(frame, line_p1, line_p2, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("RGB", frame)

    # Exit on ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

