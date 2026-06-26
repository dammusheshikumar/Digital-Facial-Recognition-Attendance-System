// camera_add_student.js
const saveInfoBtn = document.getElementById("saveInfoBtn");
const startCaptureBtn = document.getElementById("startCaptureBtn");
const addStudentBtn = document.getElementById("addStudentBtn");
const video = document.getElementById("video");
const captureStatus = document.getElementById("captureStatus");
const progressBar = document.getElementById("progressBar");

let student_id = null;
let captured = 0;
const maxImages = 50;
let images = [];
let stream = null;

document.getElementById("studentForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const res = await fetch("/add_student", { method: "POST", body: formData });
  if (!res.ok) {
    alert("Failed to save student info");
    return;
  }
  const j = await res.json();
  student_id = j.student_id;
  alert("Student info saved successfully. Click 'Start Face Capture' to open the camera.");
  startCaptureBtn.disabled = false;
  saveInfoBtn.disabled = true;
});

startCaptureBtn.addEventListener("click", async () => {
  startCaptureBtn.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    await video.play();
    captureImagesLoop();
  } catch (err) {
    alert("Camera access error: " + err.message);
    startCaptureBtn.disabled = false;
  }
});

async function captureImagesLoop() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");

  while (captured < maxImages) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.9));
    images.push(blob);
    captured++;
    captureStatus.innerText = `Captured ${captured} / ${maxImages}`;
    progressBar.style.width = `${(captured / maxImages) * 100}%`;
    progressBar.innerText = `${Math.round((captured / maxImages) * 100)}%`;
    
    // 200ms delay to give the student time to slightly adjust their head angles
    await new Promise(r => setTimeout(r, 200));
  }

  // Upload all captured images simultaneously to the backend
  const form = new FormData();
  form.append("student_id", student_id);
  images.forEach((b, i) => form.append("images[]", b, `img_${i}.jpg`));
  
  captureStatus.innerText = "Uploading dataset frames to server...";
  const resp = await fetch("/upload_face", { method: "POST", body: form });
  
  if (resp.ok) {
    captureStatus.innerText = "Dataset uploaded successfully! Complete registration.";
    alert("Captured face datasets uploaded cleanly.");
    addStudentBtn.disabled = false;
  } else {
    alert("Dataset upload failed. Please try again.");
    startCaptureBtn.disabled = false;
    captured = 0;
    images = [];
  }

  // Stop camera tracks cleanly to release hardware locks
  if (stream) stream.getTracks().forEach(t => t.stop());
}

addStudentBtn.addEventListener("click", () => {
  window.location.href = "/";
});