import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "DiT360Plus.Viewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Equirect360Viewer") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

            // Add a preview widget
            const container = document.createElement("div");
            container.style.width = "100%";
            container.style.height = "300px";
            container.style.overflow = "hidden";
            container.style.position = "relative";
            container.style.backgroundColor = "#1a1a2e";
            container.style.borderRadius = "4px";
            container.style.cursor = "grab";

            const img = document.createElement("img");
            img.style.width = "100%";
            img.style.height = "100%";
            img.style.objectFit = "contain";
            img.style.display = "none";
            container.appendChild(img);

            const label = document.createElement("div");
            label.style.position = "absolute";
            label.style.bottom = "4px";
            label.style.right = "8px";
            label.style.color = "#888";
            label.style.fontSize = "11px";
            label.style.fontFamily = "monospace";
            label.textContent = "360° Panorama Preview";
            container.appendChild(label);

            this.addDOMWidget("preview", "customtext", container, {
                getValue() { return ""; },
                setValue() {},
            });

            this._viewerContainer = container;
            this._viewerImage = img;
        };

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);

            if (output && output.images && output.images.length > 0) {
                const imageData = output.images[0];
                const src = `/view?filename=${encodeURIComponent(imageData.filename)}&type=${imageData.type}&subfolder=${imageData.subfolder || ""}`;
                if (this._viewerImage) {
                    this._viewerImage.src = src;
                    this._viewerImage.style.display = "block";
                }
            }
        };
    },
});
