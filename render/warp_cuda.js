import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.178.0/examples/jsm/controls/OrbitControls.js";

const container = document.getElementById("app");
const statusEl = document.getElementById("status");
const frameEl = document.getElementById("frame");
const perfEl = document.getElementById("perf");
const toggleMpiEl = document.getElementById("toggleMpi");
const toggleCudaEl = document.getElementById("toggleCuda");
const controlModeEl = document.getElementById("controlMode");
const cudaXEl = document.getElementById("cudaX");
const cudaYEl = document.getElementById("cudaY");
const cudaZEl = document.getElementById("cudaZ");
const cudaZoomEl = document.getElementById("cudaZoom");
const cudaOffsetXEl = document.getElementById("cudaOffsetX");
const cudaOffsetYEl = document.getElementById("cudaOffsetY");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1016);

const mpiCamera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  200
);
mpiCamera.position.set(0, 0, 3);

const cudaCamera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  200
);
cudaCamera.position.set(0, 0, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.autoClear = false;
container.appendChild(renderer.domElement);

const mpiControls = new OrbitControls(mpiCamera, renderer.domElement);
mpiControls.enableDamping = true;
mpiControls.enablePan = true;

const cudaControls = new OrbitControls(cudaCamera, renderer.domElement);
cudaControls.enableDamping = true;
cudaControls.enablePan = true;
cudaControls.enabled = false;

const bgScene = new THREE.Scene();
const bgCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const bgCanvas = document.createElement("canvas");
const bgCtx = bgCanvas.getContext("2d");
const bgTexture = new THREE.CanvasTexture(bgCanvas);
bgTexture.minFilter = THREE.LinearFilter;
bgTexture.magFilter = THREE.LinearFilter;
bgTexture.generateMipmaps = false;
const bgMaterial = new THREE.MeshBasicMaterial({
  map: bgTexture,
  transparent: true,
  depthTest: false,
  depthWrite: false,
});
const bgPlane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMaterial);
bgScene.add(bgPlane);

// Static atlas background (32-layer MPI atlas, 4x8 layout).
const atlasTexture = new THREE.TextureLoader().load("/output/static/atlas.png");
atlasTexture.minFilter = THREE.LinearFilter;
atlasTexture.magFilter = THREE.LinearFilter;
atlasTexture.generateMipmaps = false;
atlasTexture.needsUpdate = true;

const mpiVertexShader = `
  varying highp vec2 vUv;
  uniform mat3 vUvTransform;
  void main() {
    vUv = (vUvTransform * vec3(uv.x, uv.y, 1.0)).xy;
    vec4 modelViewPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * modelViewPosition;
  }
`;

const mpiFragmentShader = `
  uniform sampler2D tColor;
  varying highp vec2 vUv;
  void main() {
    vec4 rgba = texture2D(tColor, vUv);
    gl_FragColor = rgba;
  }
`;

const depths = [
  100,
  23.846155,
  13.537119,
  9.45122,
  7.259953,
  5.893536,
  4.96,
  4.281768,
  3.7667074,
  3.362256,
  3.0362391,
  2.7678573,
  2.5430682,
  2.3520486,
  2.1877205,
  2.0448549,
  1.9195048,
  1.808635,
  1.7098732,
  1.6213388,
  1.5415217,
  1.4691944,
  1.40335,
  1.3431542,
  1.2879103,
  1.2370312,
  1.1900192,
  1.1464497,
  1.105958,
  1.0682288,
  1.032989,
  1,
];

const radius = 3.0;
const height = 5.0;
const baseGeometry = new THREE.CylinderGeometry(
  radius,
  radius,
  radius * height,
  360,
  1,
  true
);

for (let i = 0; i < 32; i++) {
  const c = i % 4;
  const r = Math.floor(i / 4);
  const uvTransform = new THREE.Matrix3();
  uvTransform.set(1 / 4, 0, c / 4, 0, 1 / 8, r / 8, 0, 0, 1);

  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    uniforms: {
      tColor: { value: atlasTexture },
      vUvTransform: { value: uvTransform },
    },
    vertexShader: mpiVertexShader,
    fragmentShader: mpiFragmentShader,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneMinusSrcAlphaFactor,
    transparent: true,
    depthWrite: false,
  });

  const geom = baseGeometry.clone().scale(depths[i], depths[i], depths[i]);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.renderOrder = i;
  scene.add(mesh);
}

let ws = null;
let ready = false;
let awaiting = false;
let lastSend = 0;
let lastTick = performance.now();
let T = 1;
let currentTime = 0;
let imageType = "image/jpeg";
let fpsSamples = [];
let lastFpsTime = performance.now();

function updateVisibility() {
  scene.visible = toggleMpiEl.checked;
  bgPlane.visible = toggleCudaEl.checked;
}

let activeControl = "mpi";
function setActiveControl(mode) {
  activeControl = mode;
  mpiControls.enabled = mode === "mpi";
  cudaControls.enabled = mode === "cuda";
  controlModeEl.textContent =
    mode === "cuda" ? "Control: CUDA (release Shift for MPI)" : "Control: MPI (hold Shift for CUDA)";
}

function setStreamSize(width, height) {
  const w = Math.max(1, Math.floor(width));
  const h = Math.max(1, Math.floor(height));
  if (bgCanvas.width === w && bgCanvas.height === h) return;
  bgCanvas.width = w;
  bgCanvas.height = h;
  bgTexture.needsUpdate = true;
}

function applyCudaSliders() {
  const x = Number(cudaXEl.value);
  const y = Number(cudaYEl.value);
  const z = Number(cudaZEl.value);
  cudaCamera.position.set(x, y, z);
  const fov = Number(cudaZoomEl.value);
  cudaCamera.fov = fov;
  cudaCamera.updateProjectionMatrix();
  bgPlane.position.set(Number(cudaOffsetXEl.value), Number(cudaOffsetYEl.value), 0);
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
        imageType = msg.format === "png" ? "image/png" : "image/jpeg";
        setStreamSize(msg.width ?? 512, msg.height ?? 512);
      } else if (msg.type === "error") {
        statusEl.textContent = `Error: ${msg.message}`;
      }
      return;
    }

    const blob = new Blob([event.data], { type: imageType });
    createImageBitmap(blob).then((bitmap) => {
      setStreamSize(bitmap.width, bitmap.height);
      bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
      bgCtx.drawImage(bitmap, 0, 0, bgCanvas.width, bgCanvas.height);
      bgTexture.needsUpdate = true;
      awaiting = false;
    });
  };
}

function sendCamera(now) {
  if (!ready || awaiting) return;
  awaiting = true;
  lastSend = now;

  const payload = {
    type: "camera",
    pos: [cudaCamera.position.x, cudaCamera.position.y, cudaCamera.position.z],
    target: [cudaControls.target.x, cudaControls.target.y, cudaControls.target.z],
    fov: cudaCamera.fov,
    width: bgCanvas.width || 512,
    height: bgCanvas.height || 512,
    time: currentTime,
  };

  ws.send(JSON.stringify(payload));
}

function tick(now) {
  const delta = (now - lastTick) / 1000;
  lastTick = now;

  mpiControls.update();
  cudaControls.update();

  const fps = 12;
  currentTime = (currentTime + delta * fps) % T;
  frameEl.textContent = `Frame ${Math.round(currentTime)} / ${T}`;
  const frameDelta = now - lastFpsTime;
  if (frameDelta > 0) {
    const instFps = 1000 / frameDelta;
    fpsSamples.push(instFps);
    if (fpsSamples.length > 30) fpsSamples.shift();
  }
  lastFpsTime = now;
  const avgFps =
    fpsSamples.length > 0
      ? fpsSamples.reduce((a, b) => a + b, 0) / fpsSamples.length
      : 0;
  let memText = "n/a";
  if (performance && performance.memory) {
    const used = performance.memory.usedJSHeapSize / (1024 * 1024);
    const total = performance.memory.totalJSHeapSize / (1024 * 1024);
    memText = `${used.toFixed(0)} / ${total.toFixed(0)} MB`;
  }
  perfEl.textContent = `FPS ${avgFps.toFixed(1)} | Mem ${memText}`;

  if (now - lastSend > 1000 / fps) {
    sendCamera(now);
  }

  renderer.clear();
  renderer.render(scene, mpiCamera);
  if (bgPlane.visible && bgCanvas.width > 0 && bgCanvas.height > 0) {
    renderer.render(bgScene, bgCamera);
  }

  requestAnimationFrame(tick);
}

connect();
updateVisibility();
toggleMpiEl.addEventListener("change", updateVisibility);
toggleCudaEl.addEventListener("change", updateVisibility);
setActiveControl("mpi");
applyCudaSliders();
[
  cudaXEl,
  cudaYEl,
  cudaZEl,
  cudaZoomEl,
  cudaOffsetXEl,
  cudaOffsetYEl,
].forEach((el) => {
  el.addEventListener("input", applyCudaSliders);
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Shift") {
    setActiveControl("cuda");
  }
});
window.addEventListener("keyup", (event) => {
  if (event.key === "Shift") {
    setActiveControl("mpi");
  }
});
setStreamSize(renderer.domElement.width, renderer.domElement.height);
requestAnimationFrame(tick);

window.addEventListener("resize", () => {
  mpiCamera.aspect = window.innerWidth / window.innerHeight;
  mpiCamera.updateProjectionMatrix();
  cudaCamera.aspect = window.innerWidth / window.innerHeight;
  cudaCamera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  setStreamSize(renderer.domElement.width, renderer.domElement.height);
});
