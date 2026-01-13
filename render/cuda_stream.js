import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.178.0/examples/jsm/controls/OrbitControls.js";

const canvas = document.getElementById("stream");
const statusEl = document.getElementById("status");
const timeEl = document.getElementById("time");
const fpsEl = document.getElementById("fps");
const resEl = document.getElementById("res");
const playBtn = document.getElementById("play");
const resetBtn = document.getElementById("reset");
const orbitEl = document.getElementById("orbit");
const orbitSpeedEl = document.getElementById("orbitSpeed");
const latencyEl = document.getElementById("latency");
const frameEl = document.getElementById("frame");

const ctx = canvas.getContext("2d");

const camera = new THREE.PerspectiveCamera(49.1, 1.0, 0.01, 100.0);
camera.position.set(0, 0, 2.5);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.autoRotate = false;
controls.autoRotateSpeed = 1.2;
controls.target.set(0, 0, 0);

let ws = null;
let ready = false;
let awaiting = false;
let lastSend = 0;
let lastTick = performance.now();
let play = true;
let T = 1;
let currentTime = 0;
let imageType = "image/jpeg";

function setResolution() {
  const [w, h] = resEl.value.split("x").map(Number);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function connect() {
  ws = new WebSocket("ws://localhost:8765");
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    statusEl.textContent = "Connected";
    ready = true;
  };

  ws.onclose = () => {
    statusEl.textContent = "Disconnected";
    ready = false;
    awaiting = false;
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "hello") {
        T = msg.T || T;
        timeEl.max = Math.max(T - 1, 1);
        camera.fov = msg.fov ?? camera.fov;
        camera.updateProjectionMatrix();
        imageType = msg.format === "png" ? "image/png" : "image/jpeg";
        frameEl.textContent = `0 / ${T}`;
      } else if (msg.type === "error") {
        statusEl.textContent = `Error: ${msg.message}`;
      }
      return;
    }

    const recvTime = performance.now();
    awaiting = false;
    const blob = new Blob([event.data], { type: imageType });
    createImageBitmap(blob).then((bitmap) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      const rtt = Math.max(0, recvTime - lastSend);
      latencyEl.textContent = `${rtt.toFixed(0)} ms`;
    });
  };
}

function sendCamera(now) {
  if (!ready || awaiting) return;
  awaiting = true;
  lastSend = now;

  const payload = {
    type: "camera",
    pos: [camera.position.x, camera.position.y, camera.position.z],
    target: [controls.target.x, controls.target.y, controls.target.z],
    fov: camera.fov,
    width: Math.round(canvas.width),
    height: Math.round(canvas.height),
    time: currentTime,
  };

  ws.send(JSON.stringify(payload));
}

function tick(now) {
  const delta = (now - lastTick) / 1000;
  lastTick = now;

  controls.autoRotate = orbitEl.checked;
  controls.autoRotateSpeed = Number(orbitSpeedEl.value);
  controls.update();

  if (play) {
    const fps = Number(fpsEl.value);
    currentTime = (currentTime + delta * fps) % T;
    timeEl.value = Math.round(currentTime);
  } else {
    currentTime = Number(timeEl.value);
  }

  frameEl.textContent = `${Math.round(currentTime)} / ${T}`;

  const fpsTarget = Number(fpsEl.value);
  if (now - lastSend > 1000 / fpsTarget) {
    sendCamera(now);
  }

  requestAnimationFrame(tick);
}

playBtn.addEventListener("click", () => {
  play = !play;
  playBtn.textContent = play ? "Pause" : "Play";
});

resetBtn.addEventListener("click", () => {
  camera.position.set(0, 0, 2.5);
  controls.target.set(0, 0, 0);
  controls.update();
});

orbitEl.addEventListener("change", () => {
  controls.autoRotate = orbitEl.checked;
});

orbitSpeedEl.addEventListener("input", () => {
  controls.autoRotateSpeed = Number(orbitSpeedEl.value);
});

resEl.addEventListener("change", () => {
  setResolution();
});

timeEl.addEventListener("input", () => {
  play = false;
  playBtn.textContent = "Play";
});

setResolution();
connect();
requestAnimationFrame(tick);
