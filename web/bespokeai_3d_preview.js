import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Three.js imports via CDN (loaded dynamically)
let THREE = null;
let OrbitControls = null;
let GLTFLoader = null;

// Load Three.js and required modules dynamically
async function loadThreeJS() {
    if (THREE) return; // Already loaded

    const importMap = {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    };

    // Create import map if not exists
    let existingMap = document.querySelector('script[type="importmap"]');
    if (!existingMap) {
        const script = document.createElement('script');
        script.type = 'importmap';
        script.textContent = JSON.stringify(importMap);
        document.head.appendChild(script);
    }

    try {
        // Dynamic imports
        THREE = await import("https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js");
        const OrbitControlsModule = await import("https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js");
        const GLTFLoaderModule = await import("https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js");

        OrbitControls = OrbitControlsModule.OrbitControls;
        GLTFLoader = GLTFLoaderModule.GLTFLoader;

        console.log("[BespokeAI] Three.js loaded successfully");
    } catch (error) {
        console.error("[BespokeAI] Failed to load Three.js:", error);
        throw error;
    }
}

// Create 3D viewer widget
function create3DViewer(node, container) {
    const viewerContainer = document.createElement("div");
    viewerContainer.style.cssText = `
        width: 100%;
        height: 300px;
        background: #1a1a1a;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
    `;

    const canvas = document.createElement("canvas");
    canvas.style.cssText = `
        width: 100%;
        height: 100%;
        display: block;
    `;
    viewerContainer.appendChild(canvas);

    // Loading indicator
    const loadingDiv = document.createElement("div");
    loadingDiv.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #888;
        font-family: sans-serif;
        font-size: 14px;
        text-align: center;
    `;
    loadingDiv.innerHTML = "Loading 3D model...";
    viewerContainer.appendChild(loadingDiv);

    // Error display
    const errorDiv = document.createElement("div");
    errorDiv.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #ff6b6b;
        font-family: sans-serif;
        font-size: 12px;
        text-align: center;
        padding: 10px;
        display: none;
    `;
    viewerContainer.appendChild(errorDiv);

    container.appendChild(viewerContainer);

    return {
        container: viewerContainer,
        canvas: canvas,
        loadingDiv: loadingDiv,
        errorDiv: errorDiv,
        scene: null,
        camera: null,
        renderer: null,
        controls: null,
        model: null,
        animationId: null
    };
}

// Initialize Three.js scene
async function initScene(viewer) {
    await loadThreeJS();

    const width = viewer.container.clientWidth;
    const height = viewer.container.clientHeight;

    // Scene
    viewer.scene = new THREE.Scene();
    viewer.scene.background = new THREE.Color(0x1a1a1a);

    // Camera
    viewer.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    viewer.camera.position.set(2, 2, 2);

    // Renderer
    viewer.renderer = new THREE.WebGLRenderer({
        canvas: viewer.canvas,
        antialias: true,
        alpha: true
    });
    viewer.renderer.setSize(width, height);
    viewer.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    viewer.renderer.outputColorSpace = THREE.SRGBColorSpace;
    viewer.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    viewer.renderer.toneMappingExposure = 1;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    viewer.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7.5);
    viewer.scene.add(directionalLight);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight2.position.set(-5, -5, -5);
    viewer.scene.add(directionalLight2);

    // Grid helper
    const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x333333);
    viewer.scene.add(gridHelper);

    // Controls
    viewer.controls = new OrbitControls(viewer.camera, viewer.renderer.domElement);
    viewer.controls.enableDamping = true;
    viewer.controls.dampingFactor = 0.05;
    viewer.controls.screenSpacePanning = true;
    viewer.controls.minDistance = 0.5;
    viewer.controls.maxDistance = 50;

    // Animation loop
    function animate() {
        viewer.animationId = requestAnimationFrame(animate);
        viewer.controls.update();
        viewer.renderer.render(viewer.scene, viewer.camera);
    }
    animate();

    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
        const newWidth = viewer.container.clientWidth;
        const newHeight = viewer.container.clientHeight;
        viewer.camera.aspect = newWidth / newHeight;
        viewer.camera.updateProjectionMatrix();
        viewer.renderer.setSize(newWidth, newHeight);
    });
    resizeObserver.observe(viewer.container);
}

// Load 3D model
async function loadModel(viewer, modelPath) {
    if (!viewer.scene) {
        await initScene(viewer);
    }

    viewer.loadingDiv.style.display = "block";
    viewer.errorDiv.style.display = "none";

    // Remove existing model
    if (viewer.model) {
        viewer.scene.remove(viewer.model);
        viewer.model.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        viewer.model = null;
    }

    try {
        const loader = new GLTFLoader();

        const gltf = await new Promise((resolve, reject) => {
            loader.load(
                modelPath,
                resolve,
                (xhr) => {
                    if (xhr.lengthComputable) {
                        const percent = (xhr.loaded / xhr.total * 100).toFixed(0);
                        viewer.loadingDiv.innerHTML = `Loading... ${percent}%`;
                    }
                },
                reject
            );
        });

        viewer.model = gltf.scene;

        // Center and scale model
        const box = new THREE.Box3().setFromObject(viewer.model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2 / maxDim;
        viewer.model.scale.multiplyScalar(scale);

        // Re-center after scaling
        box.setFromObject(viewer.model);
        box.getCenter(center);
        viewer.model.position.sub(center);
        viewer.model.position.y += size.y * scale / 2;

        viewer.scene.add(viewer.model);

        // Reset camera
        viewer.camera.position.set(2, 2, 2);
        viewer.controls.target.set(0, size.y * scale / 2, 0);
        viewer.controls.update();

        viewer.loadingDiv.style.display = "none";
        console.log("[BespokeAI] Model loaded successfully:", modelPath);

    } catch (error) {
        console.error("[BespokeAI] Error loading model:", error);
        viewer.loadingDiv.style.display = "none";
        viewer.errorDiv.style.display = "block";
        viewer.errorDiv.innerHTML = `Failed to load model<br><small>${error.message}</small>`;
    }
}

// Cleanup viewer
function cleanupViewer(viewer) {
    if (viewer.animationId) {
        cancelAnimationFrame(viewer.animationId);
    }
    if (viewer.renderer) {
        viewer.renderer.dispose();
    }
    if (viewer.controls) {
        viewer.controls.dispose();
    }
    if (viewer.model) {
        viewer.model.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
    }
}

// Register the extension
app.registerExtension({
    name: "BespokeAI.3DPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "BespokeAI3DPreview") {
            return;
        }

        // Store original onExecuted
        const origOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = async function(message) {
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }

            // Get mesh data from the output
            const meshData = message?.mesh;
            if (!meshData || meshData.length === 0) {
                console.log("[BespokeAI] No mesh data received");
                return;
            }

            const mesh = meshData[0];
            if (!mesh || !mesh.filename) {
                console.log("[BespokeAI] Invalid mesh data:", mesh);
                return;
            }

            // Construct the URL for the model
            // ComfyUI serves files from input directory via /view endpoint
            const subfolder = mesh.subfolder || "";
            const filename = mesh.filename;
            const type = mesh.type || "input";

            let modelUrl;
            if (type === "input") {
                modelUrl = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder)}&type=${type}`);
            } else {
                modelUrl = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=${type}`);
            }

            console.log("[BespokeAI] Loading model from:", modelUrl);

            // Initialize viewer if not exists
            if (!this.viewer) {
                // Find or create widget container
                let container = this.widgets?.find(w => w.name === "preview_container")?.element;

                if (!container) {
                    // Create a custom widget for the 3D viewer
                    const widget = {
                        name: "preview_container",
                        type: "custom",
                        draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                            // Widget drawing is handled by DOM
                        },
                        computeSize: function() {
                            return [this.node.size[0], 320];
                        }
                    };

                    // Add container element
                    container = document.createElement("div");
                    container.style.cssText = "width: 100%; padding: 5px;";

                    widget.element = container;
                    widget.node = this;

                    if (!this.widgets) this.widgets = [];
                    this.widgets.push(widget);
                }

                this.viewer = create3DViewer(this, container);

                // Adjust node size
                const minHeight = 400;
                if (this.size[1] < minHeight) {
                    this.size[1] = minHeight;
                    this.setDirtyCanvas(true, true);
                }
            }

            // Load the model
            await loadModel(this.viewer, modelUrl);
        };

        // Cleanup on node removal
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            if (this.viewer) {
                cleanupViewer(this.viewer);
                this.viewer = null;
            }
            if (origOnRemoved) {
                origOnRemoved.apply(this, arguments);
            }
        };
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "BespokeAI3DPreview") {
            return;
        }

        // Set minimum size for the node
        node.size[0] = Math.max(node.size[0], 300);
        node.size[1] = Math.max(node.size[1], 400);

        // Add a placeholder widget
        const placeholderWidget = node.addCustomWidget({
            name: "3d_viewer_placeholder",
            type: "custom",
            draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                ctx.save();
                ctx.fillStyle = "#1a1a1a";
                ctx.fillRect(10, y, widgetWidth - 20, 280);
                ctx.fillStyle = "#666";
                ctx.font = "14px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("3D Preview", widgetWidth / 2, y + 140);
                ctx.fillStyle = "#444";
                ctx.font = "12px sans-serif";
                ctx.fillText("Run workflow to preview model", widgetWidth / 2, y + 165);
                ctx.restore();
            },
            computeSize: function() {
                return [node.size[0], 300];
            }
        });
    }
});

console.log("[BespokeAI] 3D Preview extension loaded");
