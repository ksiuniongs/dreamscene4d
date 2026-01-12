import { Viewer } from "https://unpkg.com/gaussian-splats-3d@0.4.3/dist/gaussian-splats-3d.module.js";

const container = document.getElementById("viewer");
const fileInput = document.getElementById("fileInput");
const frameSlider = document.getElementById("frameSlider");
const frameLabel = document.getElementById("frameLabel");
const playPause = document.getElementById("playPause");
const resetCam = document.getElementById("resetCam");

let viewer = null;
let files = [];
let currentFrame = 0;
let playing = true;
let timer = null;
let currentScene = null;

function initViewer() {
  if (viewer) {
    viewer.dispose();
  }
  viewer = new Viewer({
    container,
    cameraUp: [0, -1, 0],
    initialCameraPosition: [0, 0, 2],
    initialCameraLookAt: [0, 0, 0],
  });
  viewer.start();
}

function updateLabel() {
  if (!files.length) {
    frameLabel.textContent = "No files loaded";
    return;
  }
  frameLabel.textContent = `Frame ${currentFrame + 1}/${files.length}: ${files[currentFrame].name}`;
}

async function loadFrame(index) {
  if (!files.length) return;
  currentFrame = Math.max(0, Math.min(index, files.length - 1));
  const file = files[currentFrame];
  const url = URL.createObjectURL(file);
  if (currentScene !== null) {
    viewer.removeSplatScene(currentScene);
  }
  currentScene = await viewer.addSplatScene(url, {
    position: [0, 0, 0],
    scale: [1, 1, 1],
  });
  updateLabel();
  frameSlider.value = String(currentFrame);
  URL.revokeObjectURL(url);
}

function stopPlaying() {
  playing = false;
  playPause.textContent = "Play";
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

function startPlaying() {
  if (!files.length) return;
  playing = true;
  playPause.textContent = "Pause";
  timer = setInterval(() => {
    const next = (currentFrame + 1) % files.length;
    loadFrame(next);
  }, 120);
}

playPause.addEventListener("click", () => {
  if (playing) {
    stopPlaying();
  } else {
    startPlaying();
  }
});

resetCam.addEventListener("click", () => {
  viewer?.resetCamera();
});

frameSlider.addEventListener("input", () => {
  stopPlaying();
  loadFrame(parseInt(frameSlider.value, 10));
});

fileInput.addEventListener("change", () => {
  const list = Array.from(fileInput.files || []).filter((f) =>
    f.name.toLowerCase().endsWith(".splat")
  );
  if (!list.length) return;
  files = list.sort((a, b) => a.name.localeCompare(b.name));
  frameSlider.min = "0";
  frameSlider.max = String(files.length - 1);
  initViewer();
  loadFrame(0);
  if (playing) startPlaying();
});

initViewer();
