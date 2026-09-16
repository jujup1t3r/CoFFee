const BACKEND_HOST = "localhost:8000";
const API_URL = `http://${BACKEND_HOST}/api/predict`;
const WS_URL = `ws://${BACKEND_HOST}/ws`;

// DOM Elements: Realtime Dashboard
const statTotal = document.getElementById("stat-total");
const statGood = document.getElementById("stat-good");
const statDefects = document.getElementById("stat-defects");
const wsIndicator = document.getElementById("ws-indicator");
const wsText = document.getElementById("ws-text");
const latestEvent = document.getElementById("latest-event");

// DOM Elements: Manual Classifier
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const promptText = document.getElementById("prompt-text");
const imagePreview = document.getElementById("image-preview");
const predictBtn = document.getElementById("predict-btn");
const resetBtn = document.getElementById("reset-btn");
const loader = document.getElementById("loader");
const resultsCard = document.getElementById("results");

let selectedFile = null;

// ==================== WEBSOCKET CONNECTION ====================
function initWebSocket() {
  const socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    wsIndicator.className = "status-dot connected";
    wsText.textContent = "Live Stream Active";
    console.log("[WebSocket] Connected to Inspection Backend");
  };

  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);

    if (payload.type === "init" || payload.type === "bean_detected") {
      const stats = payload.stats;
      statTotal.textContent = stats.total;
      statGood.textContent = stats.good;
      statDefects.textContent = stats.defects;

      if (payload.data) {
        const { track_id, defect_type, is_defect } = payload.data;
        latestEvent.innerHTML = `Bean <strong>#${track_id}</strong>: <span style="color: ${
          is_defect ? "var(--danger)" : "var(--success)"
        };">${defect_type}</span> detected`;
      }
    }
  };

  socket.onclose = () => {
    wsIndicator.className = "status-dot disconnected";
    wsText.textContent = "Conveyor Offline";
    console.warn("[WebSocket] Disconnected. Reconnecting in 3 seconds...");
    setTimeout(initWebSocket, 3000);
  };

  socket.onerror = (err) => {
    console.error("[WebSocket] Connection error:", err);
  };
}

// Start WebSocket listener
initWebSocket();

// ==================== MANUAL PREDICTION FLOW ====================
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) {
    handleFileSelect(e.target.files[0]);
  }
});

function handleFileSelect(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please select a valid image file.");
    return;
  }
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    imagePreview.classList.remove("preview-hidden");
    promptText.classList.add("preview-hidden");
    predictBtn.disabled = false;
    resetBtn.classList.remove("preview-hidden");
    resultsCard.classList.add("preview-hidden");
  };
  reader.readAsDataURL(file);
}

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  imagePreview.src = "";
  imagePreview.classList.add("preview-hidden");
  promptText.classList.remove("preview-hidden");
  predictBtn.disabled = true;
  resetBtn.classList.add("preview-hidden");
  resultsCard.classList.add("preview-hidden");
  loader.classList.add("preview-hidden");
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append("file", selectedFile);

  predictBtn.disabled = true;
  loader.classList.remove("preview-hidden");
  resultsCard.classList.add("preview-hidden");

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();

    displayResults(data);
  } catch (err) {
    alert(`Inference failed: ${err.message}. Ensure backend is running.`);
  } finally {
    loader.classList.add("preview-hidden");
    predictBtn.disabled = false;
  }
});

function displayResults(data) {
  const { label, confidence, is_defective } = data.prediction;

  const badge = document.getElementById("status-badge");
  badge.textContent = is_defective ? "Defective Bean" : "Healthy / Good";
  badge.className = `badge ${is_defective ? "fail" : "pass"}`;

  document.getElementById("predicted-label").textContent = label;
  document.getElementById("predicted-conf").textContent = `${confidence}% Confidence`;

  const top3Container = document.getElementById("top3-list");
  top3Container.innerHTML = "";

  data.top3.forEach((item) => {
    const row = document.createElement("div");
    row.className = "top3-item";
    row.innerHTML = `
      <span>${item.class}</span>
      <strong>${item.confidence}%</strong>
    `;
    top3Container.appendChild(row);
  });

  resultsCard.classList.remove("preview-hidden");
}

// ==================== RESET STATS BUTTON ====================
const resetStatsBtn = document.getElementById("reset-stats-btn");

if (resetStatsBtn) {
  resetStatsBtn.addEventListener("click", async () => {
    try {
      const res = await fetch(`http://${BACKEND_HOST}/api/reset`, {
        method: "POST"
      });

      if (res.ok) {
        // เคลียร์ค่าตัวเลขสถิติบนหน้าจอทันที
        statTotal.textContent = "0";
        statGood.textContent = "0";
        statDefects.textContent = "0";
        latestEvent.innerHTML = "<em>Waiting for detections...</em>";
        console.log("[System] Stats reset successfully");
      }
    } catch (err) {
      console.error("Failed to reset stats:", err);
    }
  });
}