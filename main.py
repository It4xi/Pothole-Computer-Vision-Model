#20FPS CODE

import cv2
import numpy as np
import time
import tensorflow as tf

# Print available GPUs, if any
print(tf.config.list_physical_devices('GPU'))

# ===== CONFIG =====
MODEL_PATH = "Yolov8-fintuned-on-potholes_int8.tflite"
LABELS = ["pothole"]
CONF_THRESH = 0.35
IOU_THRESH = 0.45
IMG_SIZE = 320
# ==================

def preprocess_frame(frame):
    """Resize and normalize frame to feed into the TFLite model"""
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """Simple NMS using OpenCV's built-in function"""
    if not boxes:
        return [], []
    
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=CONF_THRESH,
        nms_threshold=iou_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = [boxes[i] for i in indices]
        scores = [scores[i] for i in indices]
    else:
        boxes, scores = [], []
        
    return boxes, scores

def run_detection(video_source=0):
    print("[INFO] Loading model...")
    # --- OPTIMIZATION: Set num_threads directly ---
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
        
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    print(f"[INFO] Input shape: {input_details[0]['shape']}")
    print(f"[INFO] Output shape: {output_details[0]['shape']}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera or video source!")
        return

    prev_time = time.time()
    fps_smooth = 0

    print("[INFO] Starting detection (press 'q' to quit)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        input_data = preprocess_frame(frame)

        # ---- Inference ----
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)
        pred = np.squeeze(output_data) # [batch, 8400, 5] -> [8400, 5]

        # ---- Output normalization ----
        # Transpose if the model output is [5, 8400] instead of [8400, 5]
        if pred.shape[0] == 5:
            pred = pred.T
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        # --- OPTIMIZATION 2: Vectorized Post-processing ---
        # Get all confidences (column 4)
        confs = pred[:, 4]
        
        # Create a boolean "mask" of all detections > CONF_THRESH
        mask = confs > CONF_THRESH
        
        # Apply the mask to our predictions [8400, 5] -> [N, 5]
        filtered_pred = pred[mask]
        
        boxes, scores = [], []
        
        # This 'if' block only runs if we have any good detections
        if filtered_pred.shape[0] > 0:
            
            # Extract all good detections at once (NumPy is fast)
            x_center, y_center, bw, bh, scores_list = (
                filtered_pred[:, 0], filtered_pred[:, 1], 
                filtered_pred[:, 2], filtered_pred[:, 3], 
                filtered_pred[:, 4]
            )
            
            # Calculate all box coordinates at once (vectorized)
            x1 = ((x_center - bw / 2) * w).astype(int)
            y1 = ((y_center - bh / 2) * h).astype(int)
            x2 = ((x_center + bw / 2) * w).astype(int)
            y2 = ((y_center + bh / 2) * h).astype(int)
            
            # Format for NMS [x, y, w, h]
            boxes_list = np.column_stack([x1, y1, x2 - x1, y2 - y1]).tolist()
            scores_list = scores_list.tolist()

            boxes, scores = non_max_suppression(boxes_list, scores_list, IOU_THRESH)
        # ----------------------------------------------------

        # ---- Draw detections ----
        for box, score in zip(boxes, scores):
            x, y, bw, bh = box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Pothole {score:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ---- Compute FPS ----
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        fps_smooth = fps_smooth * 0.9 + fps * 0.1  # exponential smoothing
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps_smooth:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Pothole Detection (Laptop Optimized)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Exiting...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection(video_source=0)