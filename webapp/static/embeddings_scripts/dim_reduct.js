(function() {
    'use strict';

    if (typeof ThreatEmbeddingsAnalyzer !== 'undefined') {

        ThreatEmbeddingsAnalyzer.prototype.performDimensionalityReduction = async function(method = 'umap') {
            if (!this.embeddings) {
                await this.loadEmbeddingsData();
            }
            
            console.log(`Starting dimensionality reduction: ${method.toUpperCase()}`);
            this.showLoadingIndicator('Reducing dimensions...');
            
            try {
                const embeddings = this.embeddings.map(item => item.embedding);
                this.currentMethod = method;
                
                switch (method.toLowerCase()) {
                    case 'tsne':
                        this.reducedEmbeddings = await this.performTSNE(embeddings);
                        break;
                    case 'umap':
                        this.reducedEmbeddings = await this.performUMAP(embeddings);
                        break;
                    case 'pca':
                        this.reducedEmbeddings = await this.performPCA(embeddings);
                        break;
                    default:
                        this.reducedEmbeddings = await this.performUMAP(embeddings);
                }
                
                if (this.reducedEmbeddings) {
                    console.log('Dimensionality reduction complete.');
                    this.createInteractiveVisualization();
                }
                
            } catch (error) {
                console.error('Dimensionality reduction failed:', error);
                this.showError(`Reduction failed: ${error.message}`);
            } finally {
                this.hideLoadingIndicator();
            }
        };

        ThreatEmbeddingsAnalyzer.prototype._fetchReducedDimensions = async function(method, embeddings, params = {}) {
            console.log(`Requesting ${method.toUpperCase()} reduction from backend...`);
            
            const maxRetries = 3;
            let lastError = null;
            
            for (let attempt = 1; attempt <= maxRetries; attempt++) {
                try {
                    console.log(`Attempt ${attempt}/${maxRetries} for ${method.toUpperCase()} reduction...`);
                    
                    const response = await fetch('/api/dimensionality_reduction', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ embeddings, method, ...params }),
                        timeout: 30000 // 30 second timeout
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    const result = await response.json();

                    if (!result.success) {
                        throw new Error(result.error || `Backend ${method.toUpperCase()} call failed`);
                    }

                    console.log(`${method.toUpperCase()} reduction successful.`);
                    return result.reduced_embeddings.map((coords, idx) => ({
                        x: coords[0],
                        y: coords[1],
                        originalIndex: idx
                    }));
                    
                } catch (error) {
                    lastError = error;
                    console.warn(`${method.toUpperCase()} attempt ${attempt} failed:`, error.message);
                    
                    if (attempt < maxRetries) {
                        console.log(`Retrying in ${attempt * 2} seconds...`);
                        await new Promise(resolve => setTimeout(resolve, attempt * 2000));
                    }
                }
            }
            
            console.error(`${method.toUpperCase()} reduction failed after ${maxRetries} attempts:`, lastError);
            this.showError(`${method.toUpperCase()} failed: Server may be sleeping. Please try again in a moment.`);
            return null;
        };

        ThreatEmbeddingsAnalyzer.prototype.performPCA = function(embeddings) {
            return this._fetchReducedDimensions('pca', embeddings);
        };

        ThreatEmbeddingsAnalyzer.prototype.performTSNE = function(embeddings) {
            return this._fetchReducedDimensions('tsne', embeddings, { perplexity: 30 });
        };

        ThreatEmbeddingsAnalyzer.prototype.performUMAP = function(embeddings) {
            return this._fetchReducedDimensions('umap', embeddings, { n_neighbors: 15, min_dist: 0.1 });
        };

        console.log('Dimensionality reduction module loaded.');
    }
})();
