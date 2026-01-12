import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js";

const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0f1115, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.01,
  1000
);
camera.position.set(0, 0, 2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const light = new THREE.DirectionalLight(0xffffff, 1.0);
light.position.set(1, 1, 1);
scene.add(light);

let currentObject = null;

function fitToView(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  object.position.sub(center);

  const maxDim = Math.max(size.x, size.y, size.z);
  const scale = 1.2 / Math.max(maxDim, 1e-6);
  object.scale.setScalar(scale);

  controls.target.set(0, 0, 0);
  camera.position.set(0, 0, 2);
  controls.update();
}

function loadPLY(arrayBuffer) {
  const loader = new PLYLoader();
  const geometry = loader.parse(arrayBuffer);
  geometry.computeVertexNormals();

  const material = new THREE.PointsMaterial({
    color: 0xd0d0d0,
    size: 0.01,
    sizeAttenuation: true,
  });

  const points = new THREE.Points(geometry, material);
  if (currentObject) {
    scene.remove(currentObject);
  }
  currentObject = points;
  scene.add(points);
  fitToView(points);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const fileInput = document.getElementById("fileInput");
const dirInput = document.getElementById("dirInput");
const frameSlider = document.getElementById("frameSlider");
const frameLabel = document.getElementById("frameLabel");
const playBtn = document.getElementById("playBtn");

let files = [];
const cache = new Map();
let playing = false;
let playTimer = null;

function loadFiles(list) {
  const onlyPly = list.filter((f) => f.name.toLowerCase().endsWith(".ply"));
  if (!onlyPly.length) return;
  files = onlyPly.sort((a, b) => a.name.localeCompare(b.name));
  cache.clear();
  frameSlider.min = "0";
  frameSlider.max = String(files.length - 1);
  frameSlider.value = "0";
  updateLabel(0);
  loadFrame(0);
}

function updateLabel(index) {
  if (!files.length) {
    frameLabel.textContent = "No files loaded";
    return;
  }
  frameLabel.textContent = `Frame ${index + 1}/${files.length}: ${files[index].name}`;
}

async function loadFrame(index) {
  if (!files.length) return;
  const file = files[index];
  if (!file) return;
  if (cache.has(file.name)) {
    loadPLY(cache.get(file.name));
    return;
  }
  const buffer = await file.arrayBuffer();
  cache.set(file.name, buffer);
  loadPLY(buffer);
}

function stopPlaying() {
  playing = false;
  playBtn.textContent = "Play";
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
}

function startPlaying() {
  if (!files.length) return;
  playing = true;
  playBtn.textContent = "Pause";
  playTimer = setInterval(() => {
    const next = (parseInt(frameSlider.value, 10) + 1) % files.length;
    frameSlider.value = String(next);
    updateLabel(next);
    loadFrame(next);
  }, 120);
}

playBtn.addEventListener("click", () => {
  if (!files.length) return;
  if (playing) {
    stopPlaying();
  } else {
    startPlaying();
  }
});

frameSlider.addEventListener("input", () => {
  const index = parseInt(frameSlider.value, 10);
  updateLabel(index);
  loadFrame(index);
  stopPlaying();
});

fileInput.addEventListener("change", (event) => {
  const list = Array.from(event.target.files || []);
  if (!list.length) return;
  loadFiles(list);
});

dirInput.addEventListener("change", (event) => {
  const list = Array.from(event.target.files || []);
  if (!list.length) return;
  loadFiles(list);
});
