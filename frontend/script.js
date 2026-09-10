const API_URL = "http://localhost:8000/api/predict";

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const promptText = document.getElementById("prompt-text");
const imagePreview = document.getElementById("image-preview");
const predictBtn = document.getElementById("predict-btn");
const resetBtn = document.getElementById("reset-btn");
const loader = document.getElementById("loader");
const resultsCard = document.getElementById("results");

let selectedFile = null;

// Trigger input on zone click
dropZone.addEventListener("click", () => fileInput.click());

// Drag & drop handlers
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

// Reset UI
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

// Run Prediction API call
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
    alert(`Inference failed: ${err.message}. Ensure the backend is running at ${API_URL}`);
  } finally {
    loader.classList.add("preview-hidden");
    predictBtn.disabled = false;
  }
});

function displayResults(data) {
  const { label, confidence, is_defective } = data.prediction;

  // Status Badge
  const badge = document.getElementById("status-badge");
  badge.textContent = is_defective ? "Defective Bean" : "Healthy / Good";
  badge.className = `badge ${is_defective ? "fail" : "pass"}`;

  // Metrics
  document.getElementById("predicted-label").textContent = label;
  document.getElementById("predicted-conf").textContent = `${confidence}% Confidence`;

  // Top 3 Items
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