import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

/**
 * APPLICATION STATE
 */
const state = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    pointCloud: null,
    selectedPoints: [],
    measurementMarkers: [],
    measurementLine: null,
    isMeasurementAllowed: false,
    currentDistance: 0,

    // Interaction
    longPressTimer: null,
    longPressOrigin: null,
    longPressDelay: 600, // ms
};

/**
 * DOM ELEMENTS
 */
const elements = {
    viewer: document.getElementById('viewer'),
    status: document.getElementById('status'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    imageInput: document.getElementById('imageInput'),
    distanceResult: document.getElementById('distanceResult'),

    // Sample Modal
    samplesModal: document.getElementById('samplesModal'),
    openSamplesBtn: document.getElementById('openSamplesBtn'),
    closeSamples: document.getElementById('closeSamples'),
    sampleImages: document.querySelectorAll('.sample-img'),

    // Scale Modal
    scaleModal: document.getElementById('scaleModal'),
    scaleMessage: document.getElementById('scaleMessage'),
    closeScaleModal: document.getElementById('closeScaleModal'),

    // Manual Calibration
    correctMeasureNum: document.getElementById('correctMeasureNum'),
    correctMeasureBtn: document.getElementById('correctMeasureBtn'),
};

/**
 * INITIALIZATION
 */
function init() {
    initScene();
    initEventListeners();
}

function initScene() {
    const { viewer } = elements;

    state.scene = new THREE.Scene();
    state.camera = new THREE.PerspectiveCamera(
        75,
        viewer.clientWidth / viewer.clientHeight,
        0.01,
        1500
    );

    state.renderer = new THREE.WebGLRenderer({ antialias: true });
    state.renderer.setSize(viewer.clientWidth, viewer.clientHeight);
    state.renderer.setPixelRatio(window.devicePixelRatio);
    viewer.appendChild(state.renderer.domElement);

    // Disable right-click menu on the canvas
    state.renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());

    state.controls = new OrbitControls(state.camera, state.renderer.domElement);
    state.camera.position.z = 5;

    setupAxes();
    animate();
}

function setupAxes() {
    const axisLength = 1000;
    const createAxis = (color, start, end) => {
        const material = new THREE.LineBasicMaterial({ color });
        const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
        const line = new THREE.Line(geometry, material);
        state.scene.add(line);
    };

    createAxis(0xff0000, new THREE.Vector3(-axisLength, 0, 0), new THREE.Vector3(axisLength, 0, 0)); // X
    createAxis(0x00ff00, new THREE.Vector3(0, -axisLength, 0), new THREE.Vector3(0, axisLength, 0)); // Y
    createAxis(0x0000ff, new THREE.Vector3(0, 0, -axisLength), new THREE.Vector3(0, 0, axisLength)); // Z
}

/**
 * EVENT LISTENERS
 */
function initEventListeners() {
    // Modal controls
    elements.openSamplesBtn.onclick = () => elements.samplesModal.style.display = 'block';
    elements.closeSamples.onclick = () => elements.samplesModal.style.display = 'none';
    elements.closeScaleModal.onclick = () => elements.scaleModal.style.display = 'none';

    window.onclick = (event) => {
        if (event.target === elements.samplesModal) elements.samplesModal.style.display = 'none';
        if (event.target === elements.scaleModal) elements.scaleModal.style.display = 'none';
    };

    // Sample selection
    elements.sampleImages.forEach(img => {
        img.addEventListener('click', async () => {
            const filename = img.dataset.filename;
            elements.samplesModal.style.display = 'none';
            await handleProcessStart(() => loadSample(filename));
        });
    });

    // Main actions
    elements.analyzeBtn.addEventListener('click', () => {
        const file = elements.imageInput.files[0];
        if (file) handleProcessStart(() => uploadImage(file));
    });

    elements.correctMeasureBtn.addEventListener('click', updateLocalScale);

    // Measurement interaction (Mouse & Touch)
    const canvas = state.renderer.domElement;
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointercancel', onPointerUp);

    // Resize handling
    window.addEventListener('resize', onWindowResize);
}

/**
 * PROCESS HANDLERS
 */
async function handleProcessStart(processFn) {
    elements.analyzeBtn.disabled = true;
    elements.correctMeasureBtn.disabled = true;
    elements.correctMeasureNum.disabled = true;
    state.isMeasurementAllowed = false;
    elements.status.innerText = 'Processing...';

    try {
        await processFn();
        elements.status.innerText = 'Done!';
    } catch (error) {
        console.error(error);
        elements.status.innerText = 'Error processing image.';
    } finally {
        elements.analyzeBtn.disabled = false;
    }
}

async function loadSample(filename) {
    const response = await fetch(`/analyze/${filename}`, { method: 'POST' });
    await processAnalyzeResponse(response);
}

async function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/analyze', {
        method: 'POST',
        body: formData
    });
    await processAnalyzeResponse(response);
}

async function processAnalyzeResponse(response) {
    if (!response.ok) throw new Error('Network response was not ok');

    const scale = response.headers.get('X-Scaling-Factor');
    const method = response.headers.get('X-Scaling-Method');

    if (scale && method) {
        showScaleModal(scale, method);
    }

    const buffer = await response.arrayBuffer();
    handlePCD(buffer);
}

/**
 * POINT CLOUD LOGIC
 */
function handlePCD(buffer) {
    const pointSize = 15; // Structured: f32 x 3 (12 bytes) + u8 x 3 (3 bytes) = 15 bytes
    const count = buffer.byteLength / pointSize;
    const dataView = new DataView(buffer);

    const points = new Float32Array(count * 3);
    const colors = new Uint8Array(count * 3);

    for (let i = 0; i < count; i++) {
        const offset = i * pointSize;
        points[i * 3 + 0] = dataView.getFloat32(offset + 0, true);
        points[i * 3 + 1] = dataView.getFloat32(offset + 4, true);
        points[i * 3 + 2] = dataView.getFloat32(offset + 8, true);

        colors[i * 3 + 0] = dataView.getUint8(offset + 12);
        colors[i * 3 + 1] = dataView.getUint8(offset + 13);
        colors[i * 3 + 2] = dataView.getUint8(offset + 14);
    }

    clearMeasurement();
    renderPointCloud(points, colors);
    state.isMeasurementAllowed = true;
}

function renderPointCloud(points, colors) {
    if (state.pointCloud) {
        state.scene.remove(state.pointCloud);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(points, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));

    const material = new THREE.PointsMaterial({
        size: 1,
        vertexColors: true
    });

    state.pointCloud = new THREE.Points(geometry, material);
    state.scene.add(state.pointCloud);

    centerAndZoomCamera();
}

function centerAndZoomCamera() {
    const { camera, controls, pointCloud } = state;
    const geometry = pointCloud.geometry;

    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;
    const center = new THREE.Vector3();
    bbox.getCenter(center);

    // Center the points
    geometry.translate(-center.x, -center.y, -center.z);

    // Reset bounding box after translation
    geometry.computeBoundingBox();
    const size = new THREE.Vector3();
    geometry.boundingBox.getSize(size);

    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 1.5;

    camera.position.set(0, 0, cameraZ);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    controls.update();
}

/**
 * MEASUREMENT LOGIC
 */
function onPointerDown(event) {
    if (!state.isMeasurementAllowed) return;

    // Touch or Left-click: start long press timer
    if (event.pointerType === 'touch' || (event.pointerType === 'mouse' && event.button === 0)) {
        clearLongPress();
        state.longPressOrigin = { x: event.clientX, y: event.clientY };
        state.longPressTimer = setTimeout(() => {
            selectPointAt(event.clientX, event.clientY);
            state.longPressTimer = null;
        }, state.longPressDelay);
    }
}

function onPointerUp() {
    clearLongPress();
}

function onPointerMove(event) {
    if (!state.longPressOrigin) return;

    // If moved significantly, cancel long press
    const dist = Math.hypot(
        event.clientX - state.longPressOrigin.x,
        event.clientY - state.longPressOrigin.y
    );

    if (dist > 10) {
        clearLongPress();
    }
}

function clearLongPress() {
    if (state.longPressTimer) {
        clearTimeout(state.longPressTimer);
        state.longPressTimer = null;
    }
    state.longPressOrigin = null;
}

function selectPointAt(clientX, clientY) {
    if (!state.pointCloud) return;

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const rect = state.renderer.domElement.getBoundingClientRect();

    mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, state.camera);
    const intersects = raycaster.intersectObject(state.pointCloud);

    if (intersects.length > 0) {
        const point = intersects[0].point.clone();

        elements.correctMeasureBtn.disabled = true;
        elements.correctMeasureNum.disabled = true;

        if (state.selectedPoints.length === 2) {
            clearMeasurement();
        }

        state.selectedPoints.push(point);
        addMarker(point);

        if (state.selectedPoints.length === 2) {
            measureDistance();
            elements.correctMeasureBtn.disabled = false;
            elements.correctMeasureNum.disabled = false;
        }

        // Vibrate for feedback if available
        if (window.navigator && window.navigator.vibrate) {
            window.navigator.vibrate(50);
        }
    }
}

function measureDistance() {
    const [p1, p2] = state.selectedPoints;
    const distance = p1.distanceTo(p2);
    state.currentDistance = distance;

    elements.distanceResult.innerText = `${distance.toFixed(1)} Cm`;
    drawMeasurementLine(p1, p2);
}

function drawMeasurementLine(p1, p2) {
    const geometry = new LineGeometry();
    geometry.setPositions([p1.x, p1.y, p1.z, p2.x, p2.y, p2.z]);

    const material = new LineMaterial({
        color: 0xff0000,
        linewidth: 3,
        resolution: new THREE.Vector2(window.innerWidth, window.innerHeight)
    });

    state.measurementLine = new Line2(geometry, material);
    state.measurementLine.computeLineDistances();
    state.scene.add(state.measurementLine);
}

function addMarker(position) {
    const geometry = new THREE.SphereGeometry(1.0, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.copy(position);
    state.scene.add(sphere);
    state.measurementMarkers.push(sphere);
}

function clearMeasurement() {
    state.measurementMarkers.forEach(m => state.scene.remove(m));
    state.measurementMarkers = [];

    if (state.measurementLine) {
        state.scene.remove(state.measurementLine);
        state.measurementLine = null;
    }

    state.selectedPoints = [];
    elements.distanceResult.innerText = '';
}

function updateLocalScale() {
    const newMeasure = parseFloat(elements.correctMeasureNum.value);
    if (isNaN(newMeasure) || state.currentDistance === 0) return;

    const ratio = newMeasure / state.currentDistance;
    rescalePointCloud(ratio);

    clearMeasurement();
    elements.correctMeasureNum.value = '';
    elements.correctMeasureBtn.disabled = true;
}

function rescalePointCloud(ratio) {
    const positions = state.pointCloud.geometry.attributes.position.array;
    for (let i = 0; i < positions.length; i++) {
        positions[i] *= ratio;
    }
    state.pointCloud.geometry.attributes.position.needsUpdate = true;
    state.pointCloud.geometry.computeBoundingSphere();
    state.pointCloud.geometry.computeBoundingBox();
}

/**
 * UTILS
 */
function showScaleModal(scale, method) {
    const methods = {
        'scene priors': 'detected objects with known size priors.',
        'ground plane detection': 'ground plane estimation with assumed camera height.',
        'bottom image as ground': 'scene heuristics assuming the lower image region is ground.'
    };

    const methodText = methods[method] || 'fallback estimation.';
    elements.scaleMessage.innerHTML = `
        Scale factor: <b>${parseFloat(scale).toFixed(4)}</b><br><br>
        This scene was scaled using <b>${methodText}</b><br><br>
        Measurements are approximate and may vary depending on scene quality and accuracy.
    `;
    elements.scaleModal.style.display = 'block';
}

function animate() {
    requestAnimationFrame(animate);
    state.controls.update();
    state.renderer.render(state.scene, state.camera);
}

function onWindowResize() {
    const { camera, renderer, measurementLine } = state;
    const { viewer } = elements;
    camera.aspect = viewer.clientWidth / viewer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);

    if (measurementLine) {
        measurementLine.material.resolution.set(window.innerWidth, window.innerHeight);
    }
}

// Start the app
init();
