import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.178.0/examples/jsm/controls/OrbitControls.js";

const container = document.getElementById("app");
const hud = document.getElementById("hud");
const statusEl = document.getElementById("status");
const sliderTx = document.getElementById("tx");
const sliderTy = document.getElementById("ty");
const sliderTz = document.getElementById("tz");
const sliderTs = document.getElementById("ts");
const sliderTi = document.getElementById("ti");
const sliderTo = document.getElementById("to");

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
controls.enablePan = true;

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.05;
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane();
const dragOffset = new THREE.Vector3();
const dragIntersection = new THREE.Vector3();
let isDragging = false;
let dragTarget = null;

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
  uvTransform.set(
    1 / 4, 0, c / 4,
    0, 1 / 8, r / 8,
    0, 0, 1
  );

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

async function loadBinary(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  return res.arrayBuffer();
}

async function main() {
  const meta = await fetch("./data/meta.json").then((r) => r.json());
  const pos0Buffer = await loadBinary("./data/pos0.bin");
  const deltaBuffer = await loadBinary("./data/delta.bin");
  const rgbaBuffer = await loadBinary("./data/rgba.bin");
  const scaleBuffer = await loadBinary("./data/scale.bin");
  const fRestPath = meta.frest ? `./data/${meta.frest}` : null;
  const fRestBuffer = fRestPath ? await loadBinary(fRestPath) : null;

  const pos0 = new Float32Array(pos0Buffer);
  const positions = new Float32Array(pos0.length);
  positions.set(pos0);
  const rgba = new Float32Array(rgbaBuffer);
  const scales = new Float32Array(scaleBuffer);
  const fRest = fRestBuffer ? new Float32Array(fRestBuffer) : null;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(rgba, 4));

  const deltaTexture = new THREE.DataTexture(
    new Float32Array(deltaBuffer),
    meta.width,
    meta.height,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  deltaTexture.magFilter = THREE.NearestFilter;
  deltaTexture.minFilter = THREE.NearestFilter;
  deltaTexture.generateMipmaps = false;
  deltaTexture.needsUpdate = true;

  const rgbaTexture = new THREE.DataTexture(
    new Float32Array(rgbaBuffer),
    meta.width,
    meta.height,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  rgbaTexture.magFilter = THREE.NearestFilter;
  rgbaTexture.minFilter = THREE.NearestFilter;
  rgbaTexture.generateMipmaps = false;
  rgbaTexture.needsUpdate = true;

  const scaleTexture = new THREE.DataTexture(
    new Float32Array(scaleBuffer),
    meta.width,
    meta.height,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  scaleTexture.magFilter = THREE.NearestFilter;
  scaleTexture.minFilter = THREE.NearestFilter;
  scaleTexture.generateMipmaps = false;
  scaleTexture.needsUpdate = true;

  let frestTexture = null;
  if (fRest) {
    // 27 floats per point -> 7 RGBA texels (28 slots, last unused)
    const texels = meta.width * meta.height * 7 * 4;
    const padded = new Float32Array(texels);
    for (let f = 0; f < meta.height; f++) {
      for (let i = 0; i < meta.width; i++) {
        const base = (f * meta.width + i) * 27;
        const outBase = (f * meta.width + i) * 28;
        for (let j = 0; j < 27; j++) {
          padded[outBase + j] = fRest[base + j];
        }
      }
    }
    frestTexture = new THREE.DataTexture(
      padded,
      meta.width,
      meta.height * 7,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    frestTexture.magFilter = THREE.NearestFilter;
    frestTexture.minFilter = THREE.NearestFilter;
    frestTexture.generateMipmaps = false;
    frestTexture.needsUpdate = true;
  }

  const material = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      uDeltaTex: { value: deltaTexture },
      uRgbaTex: { value: rgbaTexture },
      uScaleTex: { value: scaleTexture },
      uFrestTex: { value: frestTexture },
      uHasFrest: { value: frestTexture ? 1 : 0 },
      uFrame0: { value: 0 },
      uFrame1: { value: 0 },
      uAlpha: { value: 0 },
      uPointSize: { value: 2.0 },
      uViewProj: { value: new THREE.Matrix4() },
      uIntensity: { value: 1.0 },
      uOpacity: { value: 1.0 },
    },
    vertexShader: `
      precision highp float;
      uniform sampler2D uDeltaTex;
      uniform sampler2D uRgbaTex;
      uniform sampler2D uScaleTex;
      uniform sampler2D uFrestTex;
      uniform int uHasFrest;
      uniform int uFrame0;
      uniform int uFrame1;
      uniform float uAlpha;
      uniform float uPointSize;
      uniform mat4 uViewProj;
      in vec3 position;
      out vec4 vColor;
      const float SH_C1 = 0.488603;
      void main() {
        int idx = gl_VertexID;
        vec4 d0 = texelFetch(uDeltaTex, ivec2(idx, uFrame0), 0);
        vec4 d1 = texelFetch(uDeltaTex, ivec2(idx, uFrame1), 0);
        vec4 c0 = texelFetch(uRgbaTex, ivec2(idx, uFrame0), 0);
        vec4 c1 = texelFetch(uRgbaTex, ivec2(idx, uFrame1), 0);
        vec4 s0 = texelFetch(uScaleTex, ivec2(idx, uFrame0), 0);
        vec4 s1 = texelFetch(uScaleTex, ivec2(idx, uFrame1), 0);
        vec3 pos = position + mix(d0.xyz, d1.xyz, uAlpha);
        vec3 sc = mix(s0.xyz, s1.xyz, uAlpha);
        gl_Position = uViewProj * vec4(pos, 1.0);
        gl_PointSize = uPointSize * (sc.x + sc.y + sc.z) / 3.0;
        vec4 base = mix(c0, c1, uAlpha);
        vec3 rgb = base.rgb;
        if (uHasFrest == 1) {
          int row0 = uFrame0 * 7;
          int row1 = uFrame1 * 7;
          vec4 f00 = texelFetch(uFrestTex, ivec2(idx, row0 + 0), 0);
          vec4 f01 = texelFetch(uFrestTex, ivec2(idx, row0 + 1), 0);
          vec4 f02 = texelFetch(uFrestTex, ivec2(idx, row0 + 2), 0);
          vec4 f03 = texelFetch(uFrestTex, ivec2(idx, row0 + 3), 0);
          vec4 f04 = texelFetch(uFrestTex, ivec2(idx, row0 + 4), 0);
          vec4 f05 = texelFetch(uFrestTex, ivec2(idx, row0 + 5), 0);
          vec4 f06 = texelFetch(uFrestTex, ivec2(idx, row0 + 6), 0);
          vec4 g00 = texelFetch(uFrestTex, ivec2(idx, row1 + 0), 0);
          vec4 g01 = texelFetch(uFrestTex, ivec2(idx, row1 + 1), 0);
          vec4 g02 = texelFetch(uFrestTex, ivec2(idx, row1 + 2), 0);
          vec4 g03 = texelFetch(uFrestTex, ivec2(idx, row1 + 3), 0);
          vec4 g04 = texelFetch(uFrestTex, ivec2(idx, row1 + 4), 0);
          vec4 g05 = texelFetch(uFrestTex, ivec2(idx, row1 + 5), 0);
          vec4 g06 = texelFetch(uFrestTex, ivec2(idx, row1 + 6), 0);
          vec4 h0 = mix(f00, g00, uAlpha);
          vec4 h1 = mix(f01, g01, uAlpha);
          vec4 h2 = mix(f02, g02, uAlpha);
          vec4 h3 = mix(f03, g03, uAlpha);
          vec4 h4 = mix(f04, g04, uAlpha);
          vec4 h5 = mix(f05, g05, uAlpha);
          vec4 h6 = mix(f06, g06, uAlpha);
          float r1 = h0.x;
          float r2 = h0.y;
          float r3 = h0.z;
          float r4 = h0.w;
          float r5 = h1.x;
          float r6 = h1.y;
          float r7 = h1.z;
          float r8 = h1.w;
          float g1 = h2.x;
          float g2 = h2.y;
          float g3 = h2.z;
          float g4 = h2.w;
          float g5 = h3.x;
          float g6 = h3.y;
          float g7 = h3.z;
          float g8 = h3.w;
          float b1 = h4.x;
          float b2 = h4.y;
          float b3 = h4.z;
          float b4 = h4.w;
          float b5 = h5.x;
          float b6 = h5.y;
          float b7 = h5.z;
          float b8 = h5.w;
          vec3 dir = normalize(-pos);
          float x = dir.x;
          float y = dir.y;
          float z = dir.z;
          float sh1 = SH_C1 * y;
          float sh2 = SH_C1 * z;
          float sh3 = SH_C1 * x;
          float sh4 = SH_C1 * (x * y);
          float sh5 = SH_C1 * (y * z);
          float sh6 = SH_C1 * (3.0 * z * z - 1.0);
          float sh7 = SH_C1 * (x * z);
          float sh8 = SH_C1 * (x * x - y * y);
          rgb.r += r1 * sh1 + r2 * sh2 + r3 * sh3 + r4 * sh4 + r5 * sh5 + r6 * sh6 + r7 * sh7 + r8 * sh8;
          rgb.g += g1 * sh1 + g2 * sh2 + g3 * sh3 + g4 * sh4 + g5 * sh5 + g6 * sh6 + g7 * sh7 + g8 * sh8;
          rgb.b += b1 * sh1 + b2 * sh2 + b3 * sh3 + b4 * sh4 + b5 * sh5 + b6 * sh6 + b7 * sh7 + b8 * sh8;
        }
        vColor = vec4(rgb, base.a);
      }
    `,
    fragmentShader: `
      precision highp float;
      uniform float uIntensity;
      uniform float uOpacity;
      in vec4 vColor;
      out vec4 outColor;
      void main() {
        vec2 uv = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(uv, uv);
        if (r2 > 1.0) discard;
        float a = exp(-r2 * 4.0);
        vec3 srgb = pow(clamp(vColor.rgb * uIntensity, 0.0, 1.0), vec3(1.0 / 2.2));
        outColor = vec4(srgb, vColor.a * uOpacity * a);
      }
    `,
    transparent: true,
    depthWrite: false,
  });

  geometry.computeBoundingSphere();
  const points = new THREE.Points(geometry, material);
  points.renderOrder = 999;
  points.frustumCulled = false;
  material.depthTest = false;
  material.depthWrite = false;
  scene.add(points);

  function readSlider(el) {
    return Number.parseFloat(el.value);
  }

  function applyUi() {
    points.position.set(
      readSlider(sliderTx),
      readSlider(sliderTy),
      readSlider(sliderTz)
    );
    const scale = readSlider(sliderTs);
    points.scale.set(scale, scale, scale);
    material.uniforms.uIntensity.value = readSlider(sliderTi);
    material.uniforms.uOpacity.value = readSlider(sliderTo);
  }

  [sliderTx, sliderTy, sliderTz, sliderTs, sliderTi, sliderTo].forEach((el) => {
    el.addEventListener("input", applyUi);
  });
  applyUi();

  function updatePointer(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function onPointerDown(event) {
    if (!event.shiftKey) return;
    if (event.button !== 0) return;
    updatePointer(event);
    raycaster.setFromCamera(pointer, camera);
    raycaster.params.Points.threshold = 0.2;
    const hit = raycaster.intersectObject(points, true)[0];
    if (!hit) return;
    dragTarget = points;
    isDragging = true;
    controls.enabled = false;
    dragPlane.setFromNormalAndCoplanarPoint(
      camera.getWorldDirection(dragPlane.normal),
      dragTarget.position
    );
    raycaster.ray.intersectPlane(dragPlane, dragIntersection);
    dragOffset.copy(dragIntersection).sub(dragTarget.position);
  }

  function onPointerMove(event) {
    if (!isDragging || !dragTarget) return;
    updatePointer(event);
    raycaster.setFromCamera(pointer, camera);
    if (raycaster.ray.intersectPlane(dragPlane, dragIntersection)) {
      dragTarget.position.copy(dragIntersection.sub(dragOffset));
    }
  }

  function onPointerUp() {
    isDragging = false;
    dragTarget = null;
    controls.enabled = true;
  }

  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);

  const fps = meta.fps ?? 12;
  const totalFrames = meta.frames ?? 1;

  renderer.setAnimationLoop((time) => {
    const t = time * 0.001;
    const f = (t * fps) % totalFrames;
    const f0 = Math.floor(f);
    const f1 = (f0 + 1) % totalFrames;
    const alpha = f - f0;

    material.uniforms.uFrame0.value = f0;
    material.uniforms.uFrame1.value = f1;
    material.uniforms.uAlpha.value = alpha;
    material.uniforms.uViewProj.value.copy(camera.projectionMatrix).multiply(camera.matrixWorldInverse);

    controls.update();
    renderer.render(scene, camera);
    statusEl.textContent = `Frame ${f0 + 1}/${totalFrames} (alpha ${alpha.toFixed(2)})`;
  });
}

main().catch((err) => {
  statusEl.textContent = `Failed: ${err.message}`;
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
