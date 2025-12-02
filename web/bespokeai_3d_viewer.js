import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Load Three.js and GLTFLoader from CDN
const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
const GLTF_LOADER_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";
const ORBIT_CONTROLS_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

let THREE = null;
let GLTFLoader = null;
let OrbitControls = null;

async function loadThreeJS() {
    if (THREE) return;

    try {
        THREE = await import(THREE_CDN);
        const gltfModule = await import(GLTF_LOADER_CDN);
        GLTFLoader = gltfModule.GLTFLoader;
        const orbitModule = await import(ORBIT_CONTROLS_CDN);
        OrbitControls = orbitModule.OrbitControls;
    } catch (e) {
        console.error("[BespokeAI] Failed to load Three.js:", e);
    }
}

// Create 3D viewer widget
function create3DViewer(node, inputName, inputData) {
    const container = document.createElement("div");
    container.style.cssText = `
        width: 100%;
        height: 300px;
        background: #1a1a2e;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
    `;

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width: 100%; height: 100%; display: block;";
    container.appendChild(canvas);

    // Loading indicator
    const loadingDiv = document.createElement("div");
    loadingDiv.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #888;
        font-size: 14px;
        display: none;
    `;
    loadingDiv.textContent = "Loading 3D model...";
    container.appendChild(loadingDiv);

    // Info text
    const infoDiv = document.createElement("div");
    infoDiv.style.cssText = `
        position: absolute;
        bottom: 8px;
        left: 8px;
        color: #666;
        font-size: 11px;
        pointer-events: none;
    `;
    infoDiv.textContent = "Drag to rotate, scroll to zoom";
    container.appendChild(infoDiv);

    let scene, camera, renderer, controls, model;
    let animationId = null;
    let isInitialized = false;

    async function initScene() {
        await loadThreeJS();
        if (!THREE) {
            loadingDiv.style.display = "block";
            loadingDiv.textContent = "Failed to load 3D viewer";
            return false;
        }

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);

        const width = container.clientWidth || 300;
        const height = container.clientHeight || 300;

        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(2, 2, 2);

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.outputColorSpace = THREE.SRGBColorSpace;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7.5);
        scene.add(directionalLight);

        const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
        backLight.position.set(-5, -5, -5);
        scene.add(backLight);

        // Controls
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = false;
        controls.minDistance = 0.5;
        controls.maxDistance = 20;

        isInitialized = true;
        animate();
        return true;
    }

    function animate() {
        if (!isInitialized) return;
        animationId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    function clearModel() {
        if (model) {
            scene.remove(model);
            model.traverse((child) => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => m.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            });
            model = null;
        }
    }

    async function loadModel(url) {
        if (!isInitialized) {
            const success = await initScene();
            if (!success) return;
        }

        loadingDiv.style.display = "block";
        loadingDiv.textContent = "Loading 3D model...";
        clearModel();

        const loader = new GLTFLoader();

        try {
            const gltf = await new Promise((resolve, reject) => {
                loader.load(url, resolve, undefined, reject);
            });

            model = gltf.scene;

            // Center and scale the model
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim;
            model.scale.setScalar(scale);

            model.position.sub(center.multiplyScalar(scale));

            scene.add(model);

            // Reset camera
            camera.position.set(2, 2, 2);
            controls.target.set(0, 0, 0);
            controls.update();

            loadingDiv.style.display = "none";
        } catch (error) {
            console.error("[BespokeAI] Error loading model:", error);
            loadingDiv.textContent = "Error loading model";
        }
    }

    function resize() {
        if (!isInitialized || !renderer) return;
        const width = container.clientWidth;
        const height = container.clientHeight;
        if (width > 0 && height > 0) {
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }
    }

    // Widget object
    const widget = {
        type: "bespokeai_3d_viewer",
        name: inputName,
        value: "",
        draw: function(ctx, node, width, y) {
            // Widget draws the container
        },
        computeSize: function() {
            return [300, 320];
        },
        element: container,
        loadModel: loadModel,
        resize: resize
    };

    // Handle resize
    const resizeObserver = new ResizeObserver(() => resize());
    resizeObserver.observe(container);

    // Cleanup on node removal
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (animationId) cancelAnimationFrame(animationId);
        resizeObserver.disconnect();
        clearModel();
        if (renderer) renderer.dispose();
        if (originalOnRemoved) originalOnRemoved.call(this);
    };

    return widget;
}

// Register extension
app.registerExtension({
    name: "BespokeAI.3DViewer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "BespokeAI3DPreview") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                if (message && message["3d"] && message["3d"].length > 0) {
                    const fileInfo = message["3d"][0];
                    const { filename, subfolder, type } = fileInfo;

                    // Build URL to fetch the GLB file
                    let url = `/view?filename=${encodeURIComponent(filename)}`;
                    if (subfolder) url += `&subfolder=${encodeURIComponent(subfolder)}`;
                    if (type) url += `&type=${encodeURIComponent(type)}`;

                    // Find or create the 3D viewer widget
                    let viewerWidget = this.widgets?.find(w => w.type === "bespokeai_3d_viewer");

                    if (!viewerWidget) {
                        viewerWidget = create3DViewer(this, "3d_preview", {});
                        if (!this.widgets) this.widgets = [];
                        this.widgets.push(viewerWidget);

                        // Add DOM element to node
                        if (!this.domElement) {
                            this.domElement = viewerWidget.element;
                        }
                    }

                    // Load the model
                    viewerWidget.loadModel(url);

                    // Resize node to fit viewer
                    this.setSize([Math.max(this.size[0], 320), Math.max(this.size[1], 380)]);
                }
            };
        }

        // Also handle generation nodes
        if (nodeData.name === "BespokeAI3DGeneration" || nodeData.name === "BespokeAI3DGenerationFromURL") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                if (message && message["3d"] && message["3d"].length > 0) {
                    const fileInfo = message["3d"][0];
                    const { filename, subfolder, type } = fileInfo;

                    let url = `/view?filename=${encodeURIComponent(filename)}`;
                    if (subfolder) url += `&subfolder=${encodeURIComponent(subfolder)}`;
                    if (type) url += `&type=${encodeURIComponent(type)}`;

                    let viewerWidget = this.widgets?.find(w => w.type === "bespokeai_3d_viewer");

                    if (!viewerWidget) {
                        viewerWidget = create3DViewer(this, "3d_preview", {});
                        if (!this.widgets) this.widgets = [];
                        this.widgets.push(viewerWidget);
                    }

                    viewerWidget.loadModel(url);
                    this.setSize([Math.max(this.size[0], 320), Math.max(this.size[1], 480)]);
                }
            };
        }
    },

    async nodeCreated(node) {
        if (node.comfyClass === "BespokeAI3DPreview") {
            // Pre-initialize the viewer
            await loadThreeJS();
        }
    }
});

console.log("[BespokeAI] 3D Viewer extension loaded");
