
import cv2
from ultralytics import YOLO

# ============================================================
# SETTINGS — Change these to your preference
# ============================================================
USE_WEBCAM  = False         
VIDEO_FILE  = "cars.mp4"    
CONFIDENCE  = 0.5           
MODEL_NAME  = "yolov8n.pt"  

# Colors for bounding boxes (BGR format)
BOX_COLOR   = (0, 255, 0)    # Green
TEXT_COLOR  = (0, 0, 0)      # Black
ID_COLOR    = (0, 200, 255)  # Yellow-ish

# ============================================================
# LOAD THE MODEL
# (Downloads automatically on first run — ~6 MB for nano)
# ============================================================
print(f"Loading YOLO model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)

# ============================================================
# OPEN VIDEO SOURCE
# ============================================================
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)   # 0 = default webcam
    print("Opening webcam... Press 'Q' to quit.")
else:
    cap = cv2.VideoCapture(VIDEO_FILE)
    print(f"Opening video file: {VIDEO_FILE}")

if not cap.isOpened():
    print("ERROR: Could not open video source.")
    exit()

# ============================================================
# MAIN LOOP — Process each frame
# ============================================================
while True:
    ret, frame = cap.read()   # Read one frame from the video

    if not ret:
        print("Video ended or cannot read frame.")
        break

    # --- Run YOLOv8 Tracking ---
    # `persist=True` keeps track of objects across frames
    # `tracker="bytetrack.yaml"` uses ByteTrack algorithm
    results = model.track(frame, persist=True, conf=CONFIDENCE,
                          tracker="bytetrack.yaml", verbose=False)

    # --- Draw Results on Frame ---
    for result in results:
        boxes = result.boxes   # All detected bounding boxes

        if boxes is None:
            continue

        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Get class name (e.g. "person", "car")
            cls_id     = int(box.cls[0])
            class_name = model.names[cls_id]

            # Get confidence score
            confidence = float(box.conf[0])

            # Get tracking ID (None if tracking fails for this box)
            track_id = int(box.id[0]) if box.id is not None else -1

            # --- Draw bounding box ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # --- Draw label background ---
            label = f"{class_name} {confidence:.2f}"
            if track_id != -1:
                label += f"  ID:{track_id}"

            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 8), (x1 + label_w + 4, y1), BOX_COLOR, -1)

            # --- Draw label text ---
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

    # --- Show frame count / FPS (optional) ---
    cv2.putText(frame, "Press Q to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- Display the frame ---
    cv2.imshow("Object Detection & Tracking - YOLOv8", frame)

    # Press 'Q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit by user.")
        break

# ============================================================
# CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("Done.")
