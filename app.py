import os
import io
import csv
import threading
import sqlite3
import datetime
import json
from flask import Flask, render_template, request, jsonify, send_file, abort, make_response
from model import train_model_background, extract_embedding_for_image, MODEL_PATH

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

# Mutex lock for global multi-threaded database transaction security
db_lock = threading.Lock()

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            student_id TEXT,
            enrollment_no TEXT,
            branch TEXT,
            year TEXT,
            section TEXT,
            email TEXT,
            phone TEXT,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            name TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()
init_db()

# ---------- Train status helpers ----------
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"running": False, "progress": 0, "message": "Status read error"}

# ensure initial train status file exists
if not os.path.exists(TRAIN_STATUS_FILE):
    write_train_status({"running": False, "progress": 0, "message": "No training yet."})

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
    conn.close()
    
    if df.empty:
        days = [(datetime.date.today() - datetime.timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
        return jsonify({"dates": days, "counts": [0]*30})
        
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_30 = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1)]
    counts = [int(df[df['date'] == d].shape[0]) for d in last_30]
    dates = [d.strftime("%d-%b") for d in last_30]
    return jsonify({"dates": dates, "counts": counts})

@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")
        
    data = request.form
    full_name = data.get("name", "").strip()
    student_id = data.get("roll", "").strip()
    enrollment_no = data.get("reg_no", "").strip()
    branch = data.get("class", "").strip()
    section = data.get("sec", "").strip()
    year = data.get("year", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()

    if not full_name:
        return jsonify({"error": "Full name required"}), 400

    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        c = conn.cursor()
        now = datetime.datetime.now(datetime.UTC).isoformat()

        c.execute("""
            INSERT INTO students 
            (full_name, student_id, enrollment_no, branch, year, section, email, phone, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (full_name, student_id, enrollment_no, branch, year, section, email, phone, now))
        
        sid = c.lastrowid
        conn.commit()
        conn.close()
    
    os.makedirs(os.path.join(DATASET_DIR, str(sid)), exist_ok=True)
    return jsonify({"student_id": sid})

@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error":"student_id required"}), 400
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for f in files:
        try:
            fname = f"{datetime.datetime.now(datetime.UTC).timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
    return jsonify({"saved": saved})

@app.route("/train_model", methods=["GET"])
def train_model_route():
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status":"already_running"}), 202
    
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    t = threading.Thread(target=training_wrapper)
    t.daemon = True
    t.start()
    return jsonify({"status":"started"}), 202

@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")

# -------- Recognize face endpoint (POST image) --------
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"recognized": False, "error":"no image"}), 400
    img_file = request.files["image"]
    try:
        emb = extract_embedding_for_image(img_file.stream)
        if emb is None:
            return jsonify({"recognized": False, "error":"no face detected"}), 200
            
        from model import load_model_if_exists, predict_with_model
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error":"model not trained"}), 200
            
        pred_label, conf = predict_with_model(clf, emb)
        
        if conf < 0.5:
            return jsonify({"recognized": False, "confidence": float(conf)}), 200
            
        with db_lock:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)
            c = conn.cursor()
            
            # Fetch student name
            c.execute("SELECT full_name FROM students WHERE id=?", (int(pred_label),))
            row = c.fetchone()
            name = row[0] if row else "Unknown"
            
            # --- FIXED: ANTI-DUPLICATE SAFETY CHECK ---
            # Extract today's date in local YYYY-MM-DD format
            today_date = datetime.date.today().isoformat() # e.g., "2026-06-16"
            
            # Look for an entry for this specific student id registered on the same day
            c.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND timestamp LIKE ?
            """, (int(pred_label), f"{today_date}%"))
            
            existing_record = c.fetchone()
            
            if existing_record:
                # Student is already marked present today! Skip insert.
                conn.close()
                return jsonify({
                    "recognized": True, 
                    "student_id": int(pred_label), 
                    "name": name, 
                    "confidence": float(conf),
                    "message": "Already marked present today"
                }), 200
            
            # If no record exists for today, log it fresh!
            ts = datetime.datetime.now(datetime.UTC).isoformat()
            c.execute("INSERT INTO attendance (student_id, name, timestamp) VALUES (?, ?, ?)", (int(pred_label), name, ts))
            conn.commit()
            conn.close()
        
        return jsonify({"recognized": True, "student_id": int(pred_label), "name": name, "confidence": float(conf)}), 200
    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500
    
@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    period = request.args.get("period", "all")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    q = "SELECT id, student_id, name, timestamp FROM attendance"
    params = ()
    if period == "daily":
        today = datetime.date.today().isoformat()
        q += " WHERE date(timestamp) = ?"
        params = (today,)
    elif period == "weekly":
        start = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        q += " WHERE date(timestamp) >= ?"
        params = (start,)
    elif period == "monthly":
        start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        q += " WHERE date(timestamp) >= ?"
        params = (start,)
    q += " ORDER BY timestamp DESC LIMIT 5000"
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    return render_template("attendance_record.html", records=rows, period=period)

@app.route("/download_csv", methods=["GET"])
def download_csv():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, student_id, name, timestamp FROM attendance ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    mem = io.StringIO()
    writer = csv.writer(mem)
    writer.writerow(["Record ID", "Student ID", "Name", "Timestamp"])
    writer.writerows(rows)
    
    output = io.BytesIO()
    output.write(mem.getvalue().encode('utf-8'))
    output.seek(0)
    mem.close()

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=attendance.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route("/students", methods=["GET"])
def students_list():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, full_name, student_id, branch, section, enrollment_no, created_at FROM students ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    
    data = [{
        "id": r[0],
        "name": r[1],
        "roll": r[2],
        "class": r[3],
        "section": r[4],
        "reg_no": r[5],
        "created_at": r[6]
    } for r in rows]
    return jsonify({"students": data})

@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        c = conn.cursor()
        c.execute("DELETE FROM students WHERE id=?", (sid,))
        c.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
        conn.commit()
        conn.close()
    
    folder = os.path.join(DATASET_DIR, str(sid))
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"deleted": True})

def training_wrapper():
    try:
        train_model_background(
            DATASET_DIR,
            lambda p, m: write_train_status({
                "running": True,
                "progress": p,
                "message": m
            })
        )
        write_train_status({
            "running": False,
            "progress": 100,
            "message": "Training complete"
        })
    except Exception as e:
        write_train_status({
            "running": False,
            "progress": 0,
            "message": f"Training failed: {str(e)}"
        })


# -------- TEMPORARY ROUTE: Clear Attendance Logs --------
@app.route("/clear_attendance_records", methods=["GET"])
def clear_attendance_records():
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        c = conn.cursor()
        # Deletes all rows from the attendance table
        c.execute("DELETE FROM attendance")
        # Resets the auto-increment counter back to 1
        c.execute("DELETE FROM sqlite_sequence WHERE name='attendance'")
        conn.commit()
        conn.close()
    return "All attendance records have been cleared successfully! You can now remove this route from app.py."

@app.route("/manage_students", methods=["GET"])
def manage_students_page():
    # Renders the UI panel containing the list of registered students
    return render_template("manage_students.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)