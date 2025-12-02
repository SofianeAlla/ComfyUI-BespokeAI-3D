import { app } from "../../scripts/app.js";

// Load Three.js and dependencies from CDN
const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
const GLTF_LOADER_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";
const ORBIT_CONTROLS_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

let THREE = null;
let GLTFLoader = null;
let OrbitControls = null;
let threeLoaded = false;
let threeLoading = false;

async function loadThreeJS() {
    if (threeLoaded) return true;
    if (threeLoading) {
        // Wait for existing load
        while (threeLoading) {
            await new Promise(r => setTimeout(r, 100));
        }
        return threeLoaded;
    }

    threeLoading = true;
    try {
        THREE = await import(THREE_CDN);
        const gltfModule = await import(GLTF_LOADER_CDN);
        GLTFLoader = gltfModule.GLTFLoader;
        const orbitModule = await import(ORBIT_CONTROLS_CDN);
        OrbitControls = orbitModule.OrbitControls;
        threeLoaded = true;
        console.log("[BespokeAI] Three.js loaded successfully");
    } catch (e) {
        console.error("[BespokeAI] Failed to load Three.js:", e);
        threeLoaded = false;
    }
    threeLoading = false;
    return threeLoaded;
}

// 3D Viewer class
class BespokeAI3DViewerWidget {
    constructor(node) {
        this.node = node;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.model = null;
        this.animationId = null;
        this.isInitialized = false;
        this.currentUrl = null;

        // Create container
        this.container = document.createElement("div");
        this.container.style.cssText = `
            width: 100%;
            height: 280px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            border: 1px solid #333;
        `;

        // Canvas
        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = "width: 100%; height: 100%; display: block;";
        this.container.appendChild(this.canvas);

        // Loading overlay
        this.loadingDiv = document.createElement("div");
        this.loadingDiv.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(26, 26, 46, 0.9);
            color: #0f0;
            font-size: 14px;
            font-family: monospace;
        `;
        this.loadingDiv.textContent = "Initializing 3D Viewer...";
        this.container.appendChild(this.loadingDiv);

        // Controls hint
        this.hintDiv = document.createElement("div");
        this.hintDiv.style.cssText = `
            position: absolute;
            bottom: 8px;
            left: 8px;
            color: rgba(255,255,255,0.5);
            font-size: 10px;
            pointer-events: none;
        `;
        this.hintDiv.textContent = "🖱 Drag: rotate | Scroll: zoom | Right-drag: pan";
        this.container.appendChild(this.hintDiv);
    }

    async init() {
        const loaded = await loadThreeJS();
        if (!loaded) {
            this.loadingDiv.textContent = "Failed to load 3D engine";
            this.loadingDiv.style.color = "#f00";
            return false;
        }

        const width = this.container.clientWidth || 300;
        const height = this.container.clientHeight || 280;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.camera.position.set(2, 1.5, 2);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 1);
        keyLight.position.set(5, 10, 7.5);
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 0, -5);
        this.scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0xffffff, 0.2);
        rimLight.position.set(0, -5, 0);
        this.scene.add(rimLight);

        // Ground grid
        const gridHelper = new THREE.GridHelper(4, 20, 0x333333, 0x222222);
        gridHelper.position.y = -1;
        this.scene.add(gridHelper);

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = true;
        this.controls.minDistance = 0.5;
        this.controls.maxDistance = 20;
        this.controls.target.set(0, 0, 0);

        this.isInitialized = true;
        this.loadingDiv.style.display = "none";
        this.animate();

        return true;
    }

    animate() {
        if (!this.isInitialized) return;
        this.animationId = requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    clearModel() {
        if (this.model) {
            this.scene.remove(this.model);
            this.model.traverse((child) => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => {
                            if (m.map) m.map.dispose();
                            m.dispose();
                        });
                    } else {
                        if (child.material.map) child.material.map.dispose();
                        child.material.dispose();
                    }
                }
            });
            this.model = null;
        }
    }

    async loadModel(url) {
        if (this.currentUrl === url) return; // Already loaded
        this.currentUrl = url;

        if (!this.isInitialized) {
            const success = await this.init();
            if (!success) return;
        }

        this.loadingDiv.style.display = "flex";
        this.loadingDiv.textContent = "Loading 3D model...";
        this.loadingDiv.style.color = "#0f0";
        this.clearModel();

        const loader = new GLTFLoader();

        try {
            const gltf = await new Promise((resolve, reject) => {
                loader.load(
                    url,
                    resolve,
                    (progress) => {
                        if (progress.total > 0) {
                            const pct = Math.round((progress.loaded / progress.total) * 100);
                            this.loadingDiv.textContent = `Loading: ${pct}%`;
                        }
                    },
                    reject
                );
            });

            this.model = gltf.scene;

            // Center and scale
            const box = new THREE.Box3().setFromObject(this.model);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim;

            this.model.scale.setScalar(scale);
            this.model.position.x = -center.x * scale;
            this.model.position.y = -center.y * scale;
            this.model.position.z = -center.z * scale;

            this.scene.add(this.model);

            // Reset camera view
            this.camera.position.set(2, 1.5, 2);
            this.controls.target.set(0, 0, 0);
            this.controls.update();

            this.loadingDiv.style.display = "none";
            console.log("[BespokeAI] 3D model loaded successfully");

        } catch (error) {
            console.error("[BespokeAI] Error loading model:", error);
            this.loadingDiv.textContent = "Error loading model";
            this.loadingDiv.style.color = "#f00";
        }
    }

    resize() {
        if (!this.isInitialized || !this.renderer) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        if (width > 0 && height > 0) {
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
        }
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.clearModel();
        if (this.renderer) {
            this.renderer.dispose();
        }
        this.isInitialized = false;
    }
}

// Store viewers by node ID
const viewers = new Map();

function getOrCreateViewer(node) {
    if (!viewers.has(node.id)) {
        const viewer = new BespokeAI3DViewerWidget(node);
        viewers.set(node.id, viewer);
    }
    return viewers.get(node.id);
}

// Register extension
app.registerExtension({
    name: "BespokeAI.3DViewer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Handle all BespokeAI 3D nodes
        if (["BespokeAI3DGeneration", "BespokeAI3DGenerationFromURL", "BespokeAI3DPreview"].includes(nodeData.name)) {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                // Create the 3D viewer widget
                const viewer = getOrCreateViewer(this);

                // Add as DOM widget
                const widget = this.addDOMWidget("3d_viewer", "customwidget", viewer.container, {
                    serialize: false,
                    hideOnZoom: false,
                });

                widget.computeSize = () => [this.size[0], 290];
                widget.viewer = viewer;

                // Initialize Three.js in background
                viewer.init();

                // Store reference
                this._bespokeViewer = viewer;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                if (message && message["3d"] && message["3d"].length > 0) {
                    const fileInfo = message["3d"][0];
                    const { filename, subfolder, type } = fileInfo;

                    // Build URL
                    let url = `/view?filename=${encodeURIComponent(filename)}`;
                    if (subfolder) url += `&subfolder=${encodeURIComponent(subfolder)}`;
                    if (type) url += `&type=${encodeURIComponent(type)}`;

                    // Load model in viewer
                    if (this._bespokeViewer) {
                        this._bespokeViewer.loadModel(url);
                    }
                }
            };

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (this._bespokeViewer) {
                    this._bespokeViewer.destroy();
                    viewers.delete(this.id);
                }
                if (onRemoved) onRemoved.apply(this, arguments);
            };
        }
    }
});

// Preload Three.js
loadThreeJS();

console.log("[BespokeAI] 3D Viewer extension registered");
