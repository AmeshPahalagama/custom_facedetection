---
## Use Case
This app can be used to detect the availability of a certain face in;
          - **A video folder of your local computer** - This iterate through all the videos available in the input folder and returns the names of the videos where the 'required' face can be detected.
          - **Webcam** - Live detecting a certain input face option is available
          - **A video file on local computer** - Here, we can find the certain input face availability in a video file

---

## 🧬 Project Origin

**Nethmi Custom** – This application was initially developed to detect a specific individual named **Nethmi** within video folders on a local PC. It started as a focused tool using a well-trained binary classifier model built on **9,000 images** of Nethmi. Over time, the project evolved to support **custom face uploads**, allowing users to detect **any individual** by dynamically training a model using webcam input. The original detection logic and model for Nethmi are still included in this project as a reference.

---
## Tabs
<img width="592" alt="image" src="https://github.com/user-attachments/assets/c6f83bf9-93e2-4e6a-9b82-c66828e183d3" />

  1. Add Face - Captures images using webcam and train a CV model to detect the face. Can upload upto 5 faces
  2. Select Face -  From the given 5 faces one face can be selected. Delete option also available
  3. Video Folder - Can navigate through the files in dekstop and select a folder to check the availability of the selected faces in the videos in the folder
  4. Video File - Open a dialog box to navigate through the dekstop to find a video file to detect the selected face
  5. Start Webcam - Open the webcam for detection

---
## Enhance Accuracy

          - Train a seperate model if you have a huge data set and work with "Nethmi_Custom" code
          - Increase the number of images that will be captured at **Add Face** (Initially set to 200 here)
          - Adjust the threshold accuracy using "Trial and Error"
---

## Features

- 🎥 Real-time face detection from webcam or video files
- 📁 Batch scan an entire video folder for target face presence
- 🧑‍💻 Train face recognition models on up to 5 individuals
- 🖼 Select and delete faces using thumbnail previews
- 🧠 Face embedding using **FaceNet**, detection with **MTCNN**
- 🖥 Simple and intuitive GUI with Tkinter

---
## 🧰 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- facenet-pytorch
- torchvision
- PIL (Pillow)
- Tkinter (comes with Python standard library)
  
## 🎬 Demo

Watch the demo here: [https://youtu.be/DhNJqkpKpy4](https://youtu.be/DhNJqkpKpy4)

Install dependencies:

```bash
pip install torch torchvision opencv-python facenet-pytorch pillow


