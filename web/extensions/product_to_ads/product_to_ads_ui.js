/**
 * Product to Ads UI Extension for ComfyUI
 * Provides API key verification and re-run functionality
 */

import { app } from "../../scripts/app.js";

const NODE_TYPES = ["ProductToAds_Manual", "ProductToAds_Auto", "BrandIdentityAnalyzer"];

app.registerExtension({
    name: "ProductToAds.UI",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!NODE_TYPES.includes(nodeData.name)) {
            return;
        }
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            const apiKeyWidget = this.widgets?.find(w => w.name === "gemini_api_key");
            const statusWidget = this.widgets?.find(w => w.name === "api_key_status");
            const rerunNonceWidget = this.widgets?.find(w => w.name === "rerun_nonce");
            
            if (apiKeyWidget && statusWidget) {
                const verifyBtn = this.addWidget("button", "Verify API Key", null, () => {
                    const apiKey = apiKeyWidget.value;
                    if (!apiKey) {
                        statusWidget.value = "Missing API Key";
                        return;
                    }
                    
                    statusWidget.value = "Verifying...";
                    
                    fetch("/product_to_ads/verify_api", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ api_key: apiKey })
                    })
                        .then(r => r.json())
                        .then(data => {
                            statusWidget.value = data.message;
                        })
                        .catch(e => {
                            statusWidget.value = "Network Error";
                            console.error(e);
                        });
                });
                verifyBtn.serialize = false;
            }
            
            if (rerunNonceWidget) {
                const rerunBtn = this.addWidget("button", "Re-run from here", null, () => {
                    rerunNonceWidget.value = (rerunNonceWidget.value || 0) + 1;
                    app.queuePrompt(0, 1);
                });
                rerunBtn.serialize = false;
            }
            
            return result;
        };
    }
});
