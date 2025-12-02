import { app } from "../../scripts/app.js";

// This extension hooks into BespokeAI 3D nodes to display results
// The actual 3D viewer is ComfyUI's built-in Preview3D component

app.registerExtension({
    name: "BespokeAI.3DViewer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Handle all BespokeAI 3D nodes
        if (["BespokeAI3DGeneration", "BespokeAI3DGenerationFromURL", "BespokeAI3DPreview"].includes(nodeData.name)) {

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                // Log execution results for debugging
                console.log("[BespokeAI] Node executed:", nodeData.name, message);

                // The "3d" key in the ui message triggers ComfyUI's built-in 3D viewer
                if (message && message["3d"] && message["3d"].length > 0) {
                    console.log("[BespokeAI] 3D file ready:", message["3d"][0]);
                }
            };
        }
    }
});

console.log("[BespokeAI] 3D extension loaded - using ComfyUI built-in 3D viewer");
