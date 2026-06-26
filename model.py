import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.pkl"
)

# Load OpenCV's built-in, highly compatible Haar Cascade Face Detector
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

if face_cascade.empty():
    print("WARNING: Could not load OpenCV Haar Cascade XML configuration.")

# ---- Utility: extract face crop -> small grayscale vector (embedding) ----
def extract_embedding_from_frame(img):
    try:
        if img is None:
            return None
            
        # Convert to grayscale for Haar detection and embedding consistency
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (scaleFactor and minNeighbors tuned for fast webcams)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
            
        # Grab the largest detected face area to avoid background noise
        longest_face = max(faces, key=lambda b: b[2] * b[3])
        x, y, w, h = longest_face
        
        # Crop out the face box segment
        face_crop = gray[y:y+h, x:x+w]
        if face_crop.size == 0:
            return None
            
        # Downsample cleanly to a dense 32x32 spatial tracking fingerprint matrix
        face_resized = cv2.resize(face_crop, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values mapping them tightly between [0.0, 1.0]
        emb = face_resized.flatten().astype(np.float32) / 255.0
        return emb
    except Exception as e:
        print(f"Face extraction handling exception: {e}")
        return None

def extract_embedding_for_image(stream_or_bytes):
    try:
        data = stream_or_bytes.read()
        if not data:
            return None
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return extract_embedding_from_frame(img)
    except Exception as e:
        print(f"Inference processing stream crash prevented: {e}")
        return None

# ---- Load model helpers ----
def load_model_if_exists():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            return None

def predict_with_model(clf, emb):
    try:
        if clf is None or emb is None:
            return None, 0.0
            
        if len(clf.classes_) == 1:
            return clf.classes_[0], 1.0
            
        proba = clf.predict_proba([emb])[0]
        idx = np.argmax(proba)
        label = clf.classes_[idx]
        conf = float(proba[idx])
        return label, conf
    except Exception:
        return None, 0.0

# ---- Training function used in background ----
def train_model_background(dataset_dir, progress_callback=None):
    X = []
    y = []
    
    if not os.path.exists(dataset_dir):
        if progress_callback:
            progress_callback(0, "Dataset directory does not exist")
        return

    student_dirs = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    total_students = max(1, len(student_dirs))
    processed = 0

    for sid in student_dirs:
        folder = os.path.join(dataset_dir, sid)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        for fn in files:
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                continue
                
            emb = extract_embedding_from_frame(img)
            if emb is None:
                continue
                
            X.append(emb)
            y.append(str(sid)) # Keep as string to match filesystem layout
            
        processed += 1
        if progress_callback:
            pct = int((processed / total_students) * 80)
            progress_callback(pct, f"Processed {processed}/{total_students} students")

    if len(X) == 0:
        if progress_callback:
            progress_callback(0, "No valid faces extracted from registered datasets")
        return

    X = np.stack(X)
    y = np.array(y)

    if progress_callback:
        progress_callback(85, "Training RandomForest Core...")
        
    clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    if progress_callback:
        progress_callback(100, "Training complete")