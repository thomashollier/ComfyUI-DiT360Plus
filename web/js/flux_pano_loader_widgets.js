import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "DiT360Plus.FluxPanoLoaderWidgets",
    async nodeCreated(node) {
        if (node.comfyClass !== "FluxPanoramaLoader") return;

        const baseWidget = node.widgets?.find(w => w.name === "base_pipeline");
        const loraWidget = node.widgets?.find(w => w.name === "lora_id");
        if (!baseWidget || !loraWidget) return;

        // Draw a semi-transparent overlay with hint text when the widget is disabled
        const origDrawFg = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (origDrawFg) origDrawFg.call(this, ctx);
            if (!loraWidget.disabled || loraWidget.last_y == null) return;

            const h = LiteGraph.NODE_WIDGET_HEIGHT || 20;
            ctx.save();
            ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
            ctx.fillRect(15, loraWidget.last_y, node.size[0] - 30, h);
            ctx.fillStyle = "#aaa";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("Not used with Kontext", node.size[0] / 2, loraWidget.last_y + h / 2);
            ctx.restore();
        };

        function updateLoraState() {
            const isKontext = baseWidget.value?.includes("Kontext");
            loraWidget.disabled = isKontext;
            app.graph.setDirtyCanvas(true, true);
        }

        // Initial state
        updateLoraState();

        // React to combo changes
        const origCallback = baseWidget.callback;
        baseWidget.callback = function (...args) {
            if (origCallback) origCallback.apply(this, args);
            updateLoraState();
        };
    },
});
