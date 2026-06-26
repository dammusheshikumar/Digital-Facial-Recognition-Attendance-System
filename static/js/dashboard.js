// dashboard.js
document.addEventListener("DOMContentLoaded", () => {
  const trainBtn = document.getElementById("trainBtn");
  const trainProgress = document.getElementById("trainProgress");
  const trainMsg = document.getElementById("trainMsg");

  async function pollStatus() {
    try {
      const res = await fetch("/train_status");
      const data = await res.json();
      trainProgress.style.width = data.progress + "%";
      trainProgress.innerText = data.progress + "%";
      trainMsg.innerText = data.message || "";
      return data;
    } catch (e) {
      console.error("Polling error:", e);
      return null;
    }
  }

  trainBtn.addEventListener("click", async () => {
    trainBtn.disabled = true;
    const start = await fetch("/train_model");
    if (!start.ok && start.status !== 202) {
      alert("Failed to initiate training sequence");
      trainBtn.disabled = false;
      return;
    }
    trainMsg.innerText = "Asynchronous background training started...";
    
    // Poll the status file every 1500ms
    const t = setInterval(async () => {
      const s = await pollStatus();
      if (s && s.running === false) {
        clearInterval(t);
        trainBtn.disabled = false;
        if (s.progress === 100) {
            alert("RandomForest Classifier core training completed!");
        } else {
            alert("Training stopped: " + s.message);
        }
      }
    }, 1500);
  });

  // Chart rendering using Chart.js
  let chart = null;
  async function updateChart() {
    try {
        const res = await fetch("/attendance_stats");
        const data = await res.json();
        const ctx = document.getElementById("attendanceChart").getContext("2d");
        
        if (!chart) {
          chart = new Chart(ctx, {
            type: "bar",
            data: {
              labels: data.dates,
              datasets: [{ 
                label: "Daily Logs Count", 
                data: data.counts, 
                backgroundColor: "rgba(59, 130, 246, 0.75)",
                borderColor: "rgb(59, 130, 246)",
                borderWidth: 1
              }]
            },
            options: { 
              responsive: true, 
              maintainAspectRatio: false,
              scales: {
                y: { beginAtZero: true, ticks: { stepSize: 1 } }
              }
            }
          });
        } else {
          chart.data.labels = data.dates;
          chart.data.datasets[0].data = data.counts;
          chart.update();
        }
    } catch (err) {
        console.error("Chart updating tracking error:", err);
    }
  }
  
  updateChart();
  // Automatically refresh analytical tracking data every 10 seconds
  setInterval(updateChart, 10000);
});