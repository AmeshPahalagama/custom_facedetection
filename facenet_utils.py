# facenet_utils.py
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
import os

# Directories
FACE_DIR = "saved_faces"
THUMB_DIR = os.path.join(FACE_DIR, "thumbnails")
MODEL_DIR = os.path.join(FACE_DIR, "models")
os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# FaceNet setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)
transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])

# Helper functions

def extract_embedding(img):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return facenet(tensor).cpu().numpy()[0]

def save_model(name, embeddings):
    clf = nn.Linear(512, 1)
    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.ones(len(X), dtype=torch.float32).unsqueeze(1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)
    clf.train()
    for _ in range(200):
        optimizer.zero_grad()
        out = clf(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    torch.save(clf.state_dict(), os.path.join(MODEL_DIR, f"{name}.pth"))

def load_model(name):
    clf = nn.Linear(512, 1)
    path = os.path.join(MODEL_DIR, f"{name}.pth")
    if os.path.exists(path):
        clf.load_state_dict(torch.load(path))
        clf.eval()
        return clf
    return None

def load_thumbnails():
    return [f.replace(".jpg", "") for f in os.listdir(THUMB_DIR)]
