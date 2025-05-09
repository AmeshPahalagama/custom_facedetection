# main.py
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import cv2
from facenet_utils import mtcnn, extract_embedding
from video_utils import update_frame
from face_management import add_face, select_face

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection System")
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.last_boxes = None
        self.current_face = None
        self.current_model = None

        tk.Label(root, text="Face Detection", font=("Arial", 20)).pack()
        self.selected_label = tk.Label(root, text="Selected Face: None", font=("Arial", 12))
        self.selected_label.pack()

        self.video_label = tk.Label(root)
        self.video_label.pack()

        btns = tk.Frame(root)
        btns.pack(pady=5)
        tk.Button(btns, text="Start Webcam", command=self.start_webcam).grid(row=0, column=0, padx=5)
        tk.Button(btns, text="Video File", command=self.open_video_file).grid(row=0, column=1, padx=5)
        tk.Button(btns, text="Video Folder", command=self.open_video_folder).grid(row=0, column=2, padx=5)
        tk.Button(btns, text="Add Face", command=self.add_face).grid(row=0, column=3, padx=5)
        tk.Button(btns, text="Select Face", command=self.select_face).grid(row=0, column=4, padx=5)

        self.result_box = tk.Text(root, height=10, width=100)
        self.result_box.pack()

        self.stop_button = tk.Button(root, text="⏹ Stop", command=self.stop_video, bg="orange")
        self.exit_button = tk.Button(root, text="❌ Exit", command=root.quit, bg="red", fg="white")
        self.stop_button.place(x=10, y=10)
        self.exit_button.place(x=90, y=10)

    def start_webcam(self):
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        update_frame(self.cap, self.last_boxes, self.current_model, self.current_face, self.root, self.video_label, self.frame_count)

    def open_video_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.stop_video()
            self.cap = cv2.VideoCapture(path)
            self.running = True
            update_frame(self.cap, self.last_boxes, self.current_model, self.current_face, self.root, self.video_label, self.frame_count)

    def open_video_folder(self):
        folder = filedialog.askdirectory()
        if not folder or not self.current_model: return
        files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        self.result_box.delete(1.0, tk.END)
        for i, file in enumerate(files, 1):
            self.result_box.insert(tk.END, f"🔍 Processing {i}/{len(files)}: {file}\n")
            self.root.update()
            if self.contains_face(os.path.join(folder, file)):
                self.result_box.insert(tk.END, f"✅ {self.current_face} detected in {file}\n")
            else:
                self.result_box.insert(tk.END, f"❌ {self.current_face} not found in {file}\n")
            self.root.update()

    def contains_face(self, path):
        pass  # Retain the function as before

    def stop_video(self):
        if self.cap and self.cap.isOpened(): self.cap.release()
        self.cap = None
        self.running = False
        self.video_label.configure(image='')
        self.frame_count = 0
        self.last_boxes = None

    def add_face(self):
        cap = cv2.VideoCapture(0)
        name = simpledialog.askstring("Add Face", "Enter face name:")
        if name:
            add_face(cap, name, self.root, self.result_box)

    def select_face(self):
        select_face(self.root, self.selected_label, self.result_box)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
