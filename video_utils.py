# video_utils.py
import cv2
from facenet_utils import mtcnn, extract_embedding
from PIL import Image
import torch

def update_frame(cap, last_boxes, current_model, current_face, root, video_label, frame_count):
    if not cap.isOpened(): return
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame_count % 15 == 0:
        boxes, _ = mtcnn.detect(rgb)
        last_boxes = boxes
    else:
        boxes = last_boxes

    if boxes is not None and current_model:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = Image.fromarray(rgb[y1:y2, x1:x2])
            emb = extract_embedding(face)
            with torch.no_grad():
                out = current_model(torch.tensor(emb).unsqueeze(0)).item()
                conf = torch.sigmoid(torch.tensor(out)).item()
                label = f"{current_face} ({conf:.2f})" if conf > 0.97 else "Unknown"
                color = (0, 255, 0) if conf > 0.97 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.configure(image=imgtk)
    video_label.image = imgtk
    frame_count += 1
    root.after(30, update_frame, cap, last_boxes, current_model, current_face, root, video_label, frame_count)
