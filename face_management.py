# face_management.py
import os
from tkinter import simpledialog, messagebox
from PIL import Image
from facenet_utils import save_model, load_model, load_thumbnails

def add_face(cap, name, root, result_box):
    if len(load_thumbnails()) >= 5:
        messagebox.showerror("Limit Reached", "Maximum of 5 faces allowed.")
        return
    embeddings = []
    saved = 0
    result_box.insert(tk.END, f"\nCapturing images for {name}...\n")
    while saved < 200:
        ret, frame = cap.read()
        if not ret: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)
        if boxes is not None:
            x1, y1, x2, y2 = map(int, boxes[0])
            face = Image.fromarray(rgb[y1:y2, x1:x2])
            emb = extract_embedding(face)
            embeddings.append(emb)
            saved += 1
            result_box.insert(tk.END, f"Saved {saved}/200\n")
            root.update()
    cap.release()
    Image.fromarray(rgb[y1:y2, x1:x2]).save(os.path.join(THUMB_DIR, f"{name}.jpg"))
    save_model(name, embeddings)
    result_box.insert(tk.END, f"Done! Model saved for {name}.\n")

def select_face(root, selected_label, result_box):
    top = tk.Toplevel(root)
    top.title("Select or Delete Face")
    thumbs = load_thumbnails()
    for name in thumbs:
        img_path = os.path.join(THUMB_DIR, f"{name}.jpg")
        img = Image.open(img_path).resize((100, 100))
        imgtk = ImageTk.PhotoImage(img)

        btn = tk.Button(top, image=imgtk, bg="lightgray",
                        command=lambda n=name, t=top: handle_face_action(n, t, selected_label, result_box))
        btn.image = imgtk
        btn.pack(side=tk.LEFT, padx=5, pady=5)

def handle_face_action(name, window, selected_label, result_box):
    action = messagebox.askquestion(
        "Choose Action",
        f"What would you like to do with '{name}'?",
        icon='question',
        type=messagebox.YESNOCANCEL,
        default=messagebox.CANCEL,
        detail="Yes = Select | No = Delete | Cancel = Do nothing"
    )
    if action == 'yes':
        current_model = load_model(name)
        selected_label.config(text=f"Selected Face: {name}")
        result_box.insert(tk.END, f"\n✅ Switched to {name} for detection.\n")
        window.destroy()
    elif action == 'no':
        try:
            os.remove(os.path.join(THUMB_DIR, f"{name}.jpg"))
            os.remove(os.path.join(MODEL_DIR, f"{name}.pth"))
            result_box.insert(tk.END, f"\n🗑 Deleted face: {name}\n")
            window.destroy()
            select_face(window)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete {name}: {e}")
