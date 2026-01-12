import * as THREE from "three";
import { SplatMesh } from "@sparkjsdev/spark";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.178.0/examples/jsm/controls/OrbitControls.js";

const container = document.getElementById("app");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1016);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  200
);
camera.position.set(0, 0, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Atlas background (single static atlas.png).
const atlasTexture = new THREE.TextureLoader().load("/output/static/atlas.png");
atlasTexture.wrapS = THREE.RepeatWrapping;
atlasTexture.repeat.x = -1;
atlasTexture.needsUpdate = true;

const radius = 3.0;
const height = 5.0;
const cylinderGeometry = new THREE.CylinderGeometry(radius, radius, radius * height, 360, 1, true);
const cylinderMaterial = new THREE.MeshBasicMaterial({
  map: atlasTexture,
  side: THREE.BackSide,
  transparent: true,
});
const cylinder = new THREE.Mesh(cylinderGeometry, cylinderMaterial);
scene.add(cylinder);

// Splat foreground (sequence, load PLY directly).
const totalFrames = 10;
const splatFps = 12;
const splats = [];
for (let i = 0; i < totalFrames; i++) {
  const name = String(i).padStart(3, "0");
  const mesh = new SplatMesh({ url: `/output/4d_ply/frame_${name}.ply` });
  mesh.visible = i === 0;
  mesh.position.set(0, -0.2, -1.2);
  scene.add(mesh);
  splats.push(mesh);
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener("resize", onResize);

renderer.setAnimationLoop((time) => {
  const t = time * 0.001;
  const frame = Math.floor(t * splatFps) % totalFrames;
  for (let i = 0; i < splats.length; i++) {
    splats[i].visible = i === frame;
  }
  const active = splats[frame];
  if (active) {
    active.position.x = Math.sin(t) * 0.3;
    active.rotation.y = t * 0.3;
  }
  controls.update();
  renderer.render(scene, camera);
});
