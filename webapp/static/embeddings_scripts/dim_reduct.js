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
            try {
                const response = await fetch('/api/dimensionality_reduction', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ embeddings, method, ...params })
                });

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
                console.error(`${method.toUpperCase()} reduction error:`, error);
                this.showError(`${method.toUpperCase()} failed: ${error.message}`);
                return null;
            }
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
