import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

let scene, camera, renderer, controls;
let pointCloud;
let selectedPoints = [];
let measurementMarkers = [];
let measurementLine = null;
let measureAllowed = false;

initScene();

function initScene() {

    const container = document.getElementById("viewer");

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.01,
        1500
    );

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    renderer.domElement.addEventListener("contextmenu", (event) => {
        event.preventDefault();
    });

    controls = new OrbitControls(camera, renderer.domElement);

    camera.position.z = 5;

    const axisLength = 1000;  // large enough to span scene

    function createAxis(color, start, end) {
        const material = new THREE.LineBasicMaterial({ color: color });
        const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
        const line = new THREE.Line(geometry, material);
        scene.add(line);
    }

    // X axis (red)
    createAxis(
        0xff0000,
        new THREE.Vector3(-axisLength, 0, 0),
        new THREE.Vector3(axisLength, 0, 0)
    );

    // Y axis (green)
    createAxis(
        0x00ff00,
        new THREE.Vector3(0, -axisLength, 0),
        new THREE.Vector3(0, axisLength, 0)
    );

    // Z axis (blue)
    createAxis(
        0x0000ff,
        new THREE.Vector3(0, 0, -axisLength),
        new THREE.Vector3(0, 0, axisLength)
    );

    renderer.domElement.addEventListener("pointerdown", (event) => {
        if (event.button !== 2) return;
        if (!measureAllowed) return;

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);

        const intersects = raycaster.intersectObject(pointCloud);

        if (intersects.length > 0) {
            const point = intersects[0].point.clone();

            // If already measured, reset everything
            if (selectedPoints.length === 2) {
                clearMeasurement();
            }

            selectedPoints.push(point);
            addMarker(point);

            if (selectedPoints.length === 2) {
                measureDistance();
            }
        }
    });

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

document.getElementById("analyzeBtn").addEventListener("click", uploadImage);

async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];
    if (!file) return;

    document.getElementById("analyzeBtn").disabled = true;
    measureAllowed = false;

    document.getElementById("status").innerText = "Processing...";

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/analyze", {
        method: "POST",
        body: formData
    });

    const buffer = await response.arrayBuffer();

    const pointSize = 15; // 4+4+4+1+1+1
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
    document.getElementById("analyzeBtn").disabled = false;
    document.getElementById("status").innerText = "Done!";

    renderPointCloud(points, colors);
}

function renderPointCloud(points, colors) {
    if (pointCloud) {
        scene.remove(pointCloud);
    }

    selectedPoints = [];
    document.getElementById("distanceResult").innerText = "";

    const geometry = new THREE.BufferGeometry();

    geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(points, 3)
    );

    geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(colors, 3, true)
    );

    const material = new THREE.PointsMaterial({
        size: 1,
        vertexColors: true
    });

    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);

    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;
    const center = new THREE.Vector3();
    bbox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);

    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5;

    camera.position.set(0, 0, cameraZ);
    camera.lookAt(0, 0, 0);

    controls.target.set(0, 0, 0);
    controls.update();

    measureAllowed = true;
}

function addMarker(position) {
    const geometry = new THREE.SphereGeometry(1.0, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.copy(position);
    scene.add(sphere);

    measurementMarkers.push(sphere);
}

function clearMeasurement() {

    measurementMarkers.forEach(marker => {
        scene.remove(marker);
    });

    measurementMarkers = [];

    if (measurementLine) {
        scene.remove(measurementLine);
        measurementLine = null;
    }

    selectedPoints = [];

    document.getElementById("distanceResult").innerText = "";
}

function measureDistance() {
    const p1 = selectedPoints[0];
    const p2 = selectedPoints[1];

    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = p1.z - p2.z;

    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    document.getElementById("distanceResult").innerText =
        "Distance: " + dist.toFixed(3) + " Cm";

    // Create line
    const geometry = new LineGeometry();
    geometry.setPositions([
        p1.x, p1.y, p1.z,
        p2.x, p2.y, p2.z
    ]);

    const material = new LineMaterial({
        color: 0xff0000,
        linewidth: 3, // in pixels
    });

    material.resolution.set(window.innerWidth, window.innerHeight);

    measurementLine = new Line2(geometry, material);
    measurementLine.computeLineDistances();

    scene.add(measurementLine);

    window.addEventListener("resize", () => {
        if (measurementLine) {
            measurementLine.material.resolution.set(
                window.innerWidth,
                window.innerHeight
            );
        }
    });
}
