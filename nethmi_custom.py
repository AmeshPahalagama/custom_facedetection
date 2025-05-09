# Update of the above - Final Story

import tkinter as tk
from tkinter import filedialog
import cv2
import os
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from PIL import Image, ImageTk

# Device and model loading
trained_model_path = r"C:\Users\ameshp\Downloads\face_classifier.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

class FaceClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

clf = FaceClassifier().to(device)
clf.load_state_dict(torch.load(trained_model_path, map_location=device))
clf.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

def is_nethmi(face_img):
    face_img = face_img.convert("RGB")
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(face_tensor)
        prediction = clf(embedding)
        return prediction.item() > 0.63, prediction.item()

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nethmi Face Detection")
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.last_boxes = None

        tk.Label(root, text="Face Verification", font=("Arial", 20)).pack()

        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Buttons below
        control_frame = tk.Frame(root)
        control_frame.pack()

        tk.Button(control_frame, text="Start Webcam", command=self.start_webcam,
                  bg="green", fg="white", width=20).grid(row=0, column=0, padx=5)
        tk.Button(control_frame, text="Open Video File", command=self.open_video_file,
                  bg="blue", fg="white", width=20).grid(row=0, column=1, padx=5)
        tk.Button(control_frame, text="Open Video Folder", command=self.open_video_folder,
                  bg="purple", fg="white", width=20).grid(row=0, column=2, padx=5)

        self.result_box = tk.Text(root, height=10, width=90)
        self.result_box.pack(pady=10)

        # Overlay buttons
        self.stop_button = tk.Button(root, text="⏹ Stop", command=self.stop_video,
                                     bg="orange", fg="black")
        self.exit_button = tk.Button(root, text="❌ Exit", command=root.quit,
                                     bg="red", fg="white")

        self.stop_button.place(x=10, y=10)
        self.exit_button.place(x=90, y=10)

    def start_webcam(self):
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.frame_count = 0
        self.update_frame()

    def open_video_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.stop_video()
            self.cap = cv2.VideoCapture(path)
            self.running = True
            self.frame_count = 0
            self.update_frame()

    def open_video_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        self.result_box.delete(1.0, tk.END)
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        total = len(video_files)
        processed = 0

        for filename in video_files:
            processed += 1
            full_path = os.path.join(folder_path, filename)
            self.result_box.insert(tk.END, f"🔍 Processing {filename} ({processed}/{total})...\n")
            self.root.update()

            if self.contains_nethmi(full_path):
                self.result_box.insert(tk.END, f"✅ Nethmi detected: {filename}\n")
            else:
                self.result_box.insert(tk.END, f"❌ Not detected: {filename}\n")

    def contains_nethmi(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 15 != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face = Image.fromarray(rgb[y1:y2, x1:x2])
                    is_nethmi_flag, _ = is_nethmi(face)
                    if is_nethmi_flag:
                        cap.release()
                        return True
        cap.release()
        return False

    def update_frame(self):
        if not self.running or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        self.frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.frame_count % 15 == 0:
            boxes, _ = mtcnn.detect(rgb)
            self.last_boxes = boxes
        else:
            boxes = self.last_boxes

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = Image.fromarray(rgb[y1:y2, x1:x2])
                is_nethmi_flag, prob = is_nethmi(face)
                color = (0, 255, 0) if is_nethmi_flag else (0, 0, 255)
                label = f"{'Nethmi' if is_nethmi_flag else 'Unknown'}: {prob:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

        self.root.after(30, self.update_frame)

    def stop_video(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.running = False
        self.video_label.configure(image='')
        self.frame_count = 0
        self.last_boxes = None

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
