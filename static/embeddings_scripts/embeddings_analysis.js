class ThreatEmbeddingsAnalyzer {
    constructor() {
        this.embeddings = [];
        this.reducedEmbeddings = [];
        this.clusters = null;
        this.clusterCentroids = null;
        this.clusterLabels = new Map();
        this.config = {
            clustering: {
                method: 'kmeans',
                numClusters: 8,
                minClusterSize: 3
            },
            visualization: {
                width: 800,
                height: 600,
                pointSize: 4,
                pointOpacity: 0.7
            },
            dimensionalityReduction: {
                method: 'umap',
                nNeighbors: 15,
                minDist: 0.1,
                perplexity: 30
            }
        };

        this.clusteringAlgorithmSelect = null;
        this.clusterCountSlider = null;

        console.log('ThreatEmbeddingsAnalyzer initialized');
    }

    async initialize() {
        console.log('Initializing ThreatEmbeddingsAnalyzer...');

        if (!this.checkDependencies()) {
            console.error('Dependencies not available');
            return false;
        }

        this.clusteringAlgorithmSelect = document.getElementById('clusteringMethod');
        this.clusterCountSlider = document.getElementById('numClustersSlider');

        this.initializeControls();
        this.initializeEventListeners();

        console.log('ThreatEmbeddingsAnalyzer initialized successfully');
        return true;
    }

    checkDependencies() {
        const requiredLibraries = {
            'Plotly': window.Plotly
        };

        const missingLibraries = [];
        for (const [name, lib] of Object.entries(requiredLibraries)) {
            if (!lib) {
                missingLibraries.push(name);
            }
        }

        if (missingLibraries.length > 0) {
            console.warn('Missing required libraries:', missingLibraries);
            this.showError(`Missing required libraries: ${missingLibraries.join(', ')}. Please ensure Plotly is loaded.`);
            return false;
        }

        return true;
    }

    initializeControls() {
        const controlsContainer = document.querySelector('.controls-panel');
        if (controlsContainer) {
            const stabilityButton = document.createElement('button');
            stabilityButton.id = 'stabilityAnalysisBtn';
            stabilityButton.className = 'btn btn-secondary';
            stabilityButton.textContent = 'Run Stability Analysis';
            stabilityButton.title = 'Perform clustering stability analysis to find optimal parameters';

            const clusteringControls = document.querySelector('.clustering-controls');
            if (clusteringControls) {
                clusteringControls.appendChild(stabilityButton);
            } else {
                controlsContainer.appendChild(stabilityButton);
            }
        }

        const clusterButton = document.getElementById('clusterBtn');
        if (clusterButton) {
            const originalClickHandler = clusterButton.onclick;
            clusterButton.onclick = async () => {
                if (originalClickHandler) {
                    await originalClickHandler();
                }
                setTimeout(() => {
                    this.analyzeClusterQuality();
                }, 500);
            };
        }
    }

    initializeEventListeners() {
        console.log('Setting up event listeners...');
        
        const dimensionalityMethod = document.getElementById('dimensionalityMethod');
        if (dimensionalityMethod) {
            dimensionalityMethod.addEventListener('change', (e) => {
                console.log('Dimensionality method changed:', e.target.value);
                this.performDimensionalityReduction(e.target.value);
            });
        } else {
            console.warn('dimensionalityMethod element not found');
        }
        
        const loadEmbeddings = document.getElementById('loadEmbeddings');
        if (loadEmbeddings) {
            loadEmbeddings.addEventListener('click', async () => {
                const method = document.getElementById('dimensionalityMethod')?.value || 'umap';
                await this.loadEmbeddingsData();
                if (this.embeddings && this.embeddings.length > 0) {
                    this.performDimensionalityReduction(method);
                }
            });
        } else {
            console.warn('loadEmbeddings button not found');
        }
        
        const clusteringMethod = document.getElementById('clusteringMethod');
        if (clusteringMethod) {
            clusteringMethod.addEventListener('change', (e) => {
                this.config.clustering.method = e.target.value;
                this.performClustering();
            });
        } else {
            console.warn('clusteringMethod element not found');
        }
        
        const numClustersSlider = document.getElementById('numClustersSlider');
        const numClustersValue = document.getElementById('numClustersValue');
        if (numClustersSlider) {
            numClustersSlider.addEventListener('input', (e) => {
                const value = parseInt(e.target.value);
                this.config.clustering.numClusters = value;
                if (numClustersValue) {
                    numClustersValue.textContent = value;
                }
                this.performClustering();
            });
        } else {
            console.warn('numClustersSlider element not found');
        }
        
        const performClustering = document.getElementById('performClustering');
        if (performClustering) {
            performClustering.addEventListener('click', () => {
                this.performClustering();
            });
        } else {
            console.warn('performClustering button not found');
        }
        
        const exportVisualization = document.getElementById('exportVisualization');
        if (exportVisualization) {
            exportVisualization.addEventListener('click', () => {
                this.exportVisualization();
            });
        } else {
            console.warn('exportVisualization button not found');
        }
        
        const generateClusterLabels = document.getElementById('generateClusterLabels');
        if (generateClusterLabels) {
            generateClusterLabels.addEventListener('click', () => {
                this.generateClusterLabels();
            });
        } else {
            console.warn('generateClusterLabels button not found');
        }
        
        console.log('Event listeners setup completed.');
        
        const stabilityBtn = document.getElementById('stabilityAnalysisBtn');
        if (stabilityBtn) {
            stabilityBtn.addEventListener('click', async () => {
                const results = await this.performStabilityAnalysis();
                if (results && results.stabilityReport) {
                    this.displayStabilityResults(results);
                }
            });
        }
    }

    async loadEmbeddingsData() {
        try {
            console.log('Loading threat embeddings data...');
            this.showLoadingIndicator('Loading threat embeddings data...');
            
            const response = await fetch('/api/threat_embeddings');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success && data.embeddings && data.embeddings.length > 0) {
                this.embeddings = data.embeddings.map(item => {
                    let embedding = item.embedding;
                    
                    if (typeof embedding === 'string') {
                        try {
                            embedding = JSON.parse(embedding);
                        } catch (e) {
                            console.warn(`Failed to parse embedding string for item: ${item.id}`);
                            embedding = null;
                        }
                    }
                    
                    if (!Array.isArray(embedding) || embedding.length === 0) {
                        console.warn(`Invalid embedding for item: ${item.id}`, embedding);
                        embedding = null;
                    }
                    
                    if (embedding && embedding.some(val => isNaN(val) || !isFinite(val))) {
                        console.warn(`Embedding contains invalid values for item: ${item.id}`);
                        embedding = null;
                    }
                    
                    return {
                        id: item.id,
                        text: item.text,
                        species: item.species,
                        impact: item.impact,
                        category: item.category,
                        predicate: item.predicate,
                        doi: item.doi,
                        embedding: embedding
                    };
                }).filter(item => item.embedding !== null);
                
                console.log(`Loaded ${this.embeddings.length} valid threat embeddings.`);
                
                if (this.embeddings.length < 10) {
                    console.warn(`Warning: Only ${this.embeddings.length} valid embeddings found.`);
                }
                
            } else {
                console.warn('No embeddings found in backend, generating synthetic data.');
                await this.generateSyntheticEmbeddings();
            }
            
            this.updateEmbeddingsInfo();
            this.hideLoadingIndicator();
            
        } catch (error) {
            console.error('Error loading embeddings data:', error);
            this.showError('Failed to load embeddings from backend. Using synthetic data for demonstration.');
            await this.generateSyntheticEmbeddings();
            this.hideLoadingIndicator();
        }
    }

    async generateSyntheticEmbeddings() {
        console.log('Generating synthetic embeddings...');
        
        const threatTemplates = [
            "experienced habitat degradation due to",
            "faced population decline from", 
            "suffered breeding disruption caused by",
            "encountered feeding interference from",
            "underwent territory loss through"
        ];
        
        const species = [
            "Greater Spotted Eagle", "Arctic Tern", "Snow Leopard", "Mountain Gorilla",
            "Monarch Butterfly", "Polar Bear", "Sea Turtle", "Coral Trout"
        ];
        
        const impacts = [
            "deforestation", "climate change", "pollution", "hunting",
            "habitat fragmentation", "invasive species", "agricultural expansion", 
            "urban development"
        ];
        
        const categories = ["Habitat", "Climate", "Pollution", "Human Activity"];
        
        const syntheticData = [];
        for (let i = 0; i < 50; i++) {
            const selectedSpecies = species[Math.floor(Math.random() * species.length)];
            const predicate = threatTemplates[Math.floor(Math.random() * threatTemplates.length)];
            const impact = impacts[Math.floor(Math.random() * impacts.length)];
            const category = categories[Math.floor(Math.random() * categories.length)];
            
            const fullThreatSentence = `${selectedSpecies} ${predicate} ${impact}`;
            
            syntheticData.push({
                id: `demo_${i}`,
                text: fullThreatSentence,
                predicate: predicate,
                species: selectedSpecies,
                impact: impact,
                category: category,
                embedding: this.generateRandomEmbedding(64)
            });
        }
        
        this.embeddings = syntheticData;
        console.log(`Generated ${syntheticData.length} synthetic embeddings.`);
        
        return syntheticData;
    }

    generateRandomEmbedding(dimensions = 64) {
        const embedding = [];
        for (let i = 0; i < dimensions; i++) {
            embedding.push((Math.random() - 0.5) * 2);
        }
        return embedding;
    }

    euclideanDistance(vec1, vec2) {
        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            sum += Math.pow(vec1[i] - vec2[i], 2);
        }
        return Math.sqrt(sum);
    }

    analyzeClusterQuality() {
        if (!this.clusters) return;
        
        console.log('Performing cluster quality analysis...');
        
        const silhouetteScore = this.calculateSilhouetteScore();
        const daviesBouldinIndex = this.calculateDaviesBouldinIndex();
        const calinskiHarabaszIndex = this.calculateCalinskiHarabaszIndex();
        
        const clusterStats = this.calculateClusterStatistics();
        
        const interpretabilityScore = this.assessClusterInterpretability(clusterStats);
        
        const qualityResults = {
            silhouetteScore,
            daviesBouldinIndex,
            calinskiHarabaszIndex,
            interpretabilityScore,
            clusterStats,
            overallQuality: this.calculateOverallQuality(silhouetteScore, daviesBouldinIndex, interpretabilityScore)
        };
        
        console.log('Cluster Quality Analysis Results:', qualityResults);
        
        this.updateComprehensiveClusterQualityDisplay(qualityResults);
        
        return qualityResults;
    }

    updateComprehensiveClusterQualityDisplay(qualityResults) {
        const container = document.getElementById('clusteringSummary');
        if (!container) return;
        
        let html = `
            <div class="clustering-summary">
                <h4>Clustering Quality</h4>
                <div class="quality-metrics">
                    <p><strong>Silhouette Score:</strong> ${qualityResults.silhouetteScore.toFixed(3)} ${this.getQualityInterpretation('silhouette', qualityResults.silhouetteScore)}</p>
                    <p><strong>Davies-Bouldin Index:</strong> ${qualityResults.daviesBouldinIndex.toFixed(3)} ${this.getQualityInterpretation('davies_bouldin', qualityResults.daviesBouldinIndex)}</p>
                    <p><strong>Calinski-Harabasz Index:</strong> ${qualityResults.calinskiHarabaszIndex.toFixed(2)} ${this.getQualityInterpretation('calinski_harabasz', qualityResults.calinskiHarabaszIndex)}</p>
                    <p><strong>Interpretability Score:</strong> ${qualityResults.interpretabilityScore.toFixed(3)} ${this.getQualityInterpretation('interpretability', qualityResults.interpretabilityScore)}</p>
                    <p><strong>Overall Quality:</strong> ${qualityResults.overallQuality.toFixed(3)} ${this.getQualityInterpretation('overall', qualityResults.overallQuality)}</p>
                </div>
                <h5>Cluster Distribution:</h5>
                <ul>`;
        
        Object.entries(qualityResults.clusterStats.clusterCounts).forEach(([cluster, count]) => {
            const percentage = this.embeddings.length > 0 ? 
                (count / this.embeddings.length * 100).toFixed(1) : '0';
            html += `<li>Cluster ${cluster}: ${count} threats (${percentage}%)</li>`;
        });

        html += `
                </ul>
                <div class="quality-recommendations">
                    <h5>Recommendations:</h5>
                    ${this.generateQualityRecommendations(qualityResults)}
                </div>
            </div>`;

        container.innerHTML = html;
        this.populatePatternDiscovery(qualityResults.clusterStats);
    }

    getQualityInterpretation(metricType, value) {
        switch (metricType) {
            case 'silhouette':
                if (value > 0.7) return '(Excellent)';
                if (value > 0.5) return '(Good)';
                if (value > 0.25) return '(Fair)';
                return '(Poor)';
            case 'davies_bouldin':
                if (value < 0.5) return '(Excellent)';
                if (value < 1.0) return '(Good)';
                if (value < 2.0) return '(Fair)';
                return '(Poor)';
            case 'calinski_harabasz':
                if (value > 100) return '(Excellent)';
                if (value > 50) return '(Good)';
                if (value > 20) return '(Fair)';
                return '(Poor)';
            case 'interpretability':
                if (value > 0.8) return '(Highly Interpretable)';
                if (value > 0.6) return '(Interpretable)';
                if (value > 0.4) return '(Moderately Interpretable)';
                return '(Low Interpretability)';
            case 'overall':
                if (value > 0.8) return '(Excellent Clustering)';
                if (value > 0.6) return '(Good Clustering)';
                if (value > 0.4) return '(Fair Clustering)';
                return '(Poor Clustering)';
            default:
                return '';
        }
    }

    generateQualityRecommendations(qualityResults) {
        let recommendations = '<ul>';
        
        if (qualityResults.silhouetteScore < 0.3) {
            recommendations += '<li>Consider adjusting the number of clusters.</li>';
        }
        if (qualityResults.daviesBouldinIndex > 1.5) {
            recommendations += '<li>Clusters show significant overlap; try a different clustering algorithm.</li>';
        }
        if (qualityResults.interpretabilityScore < 0.5) {
            recommendations += '<li>Clusters lack semantic coherence.</li>';
        }
        if (qualityResults.overallQuality < 0.5) {
            recommendations += '<li>Overall clustering quality is low. Run stability analysis for optimal parameters.</li>';
        }
        if (qualityResults.overallQuality > 0.7) {
            recommendations += '<li>Clustering quality is good. Proceed with analysis.</li>';
        }
        
        recommendations += '</ul>';
        return recommendations;
    }

    calculateClusterStatistics() {
        if (!this.clusters) return {};
        
        const clusterCounts = {};
        const clusterThreats = {};
        const clusterSpecies = {};
        const clusterPredicates = {};
        const clusterData = {};
        
        this.clusters.forEach((cluster, idx) => {
            if (!clusterCounts[cluster]) {
                clusterCounts[cluster] = 0;
                clusterThreats[cluster] = [];
                clusterSpecies[cluster] = [];
                clusterPredicates[cluster] = [];
                clusterData[cluster] = [];
            }
            clusterCounts[cluster]++;
            
            const embedding = this.embeddings[idx];
            clusterThreats[cluster].push(embedding.text || embedding.impact || '');
            clusterSpecies[cluster].push(embedding.species || '');
            clusterPredicates[cluster].push(embedding.predicate || '');
            clusterData[cluster].push(embedding);
        });
        
        return {
            clusterCounts,
            clusterThreats,
            clusterSpecies,
            clusterPredicates,
            clusterData,
            numClusters: Object.keys(clusterCounts).length,
            avgClusterSize: Object.values(clusterCounts).reduce((a, b) => a + b, 0) / Object.keys(clusterCounts).length
        };
    }

    handlePointClick(data, plotData) {
        if (!data.points || data.points.length === 0) return;
        
        const pointIndex = data.points[0].pointIndex;
        if (pointIndex >= plotData.length) {
            console.warn('Point index out of bounds:', pointIndex);
            return;
        }
        
        const originalIndex = plotData[pointIndex].originalIndex;
        const threatData = this.embeddings[originalIndex];
        
        if (!threatData) {
            console.warn('No threat data found for index:', originalIndex);
            return;
        }
        
        this.displayThreatDetails(threatData, plotData[pointIndex].cluster);
    }

    displayThreatDetails(threat, clusterInfo = null) {
        const detailsPanel = document.getElementById('threat-details');
        if (!detailsPanel) return;
        
        const fullThreatText = threat.text || threat.predicate || '';
        
        detailsPanel.innerHTML = `
            <h3>Threat Details</h3>
            <div class="threat-info">
                <p><strong>Threat:</strong> "${fullThreatText}"</p>
                <p><strong>Species:</strong> ${threat.species || 'N/A'}</p>
                <p><strong>Impact:</strong> ${threat.impact || 'N/A'}</p>
                <p><strong>Category:</strong> ${threat.category || 'Unknown'}</p>
                ${clusterInfo ? `<p><strong>Cluster:</strong> ${clusterInfo}</p>` : ''}
                ${threat.doi ? `<p><strong>DOI:</strong> <a href="${threat.doi}" target="_blank">${threat.doi}</a></p>` : ''}
                <p><strong>Embedding Dimensions:</strong> ${threat.embedding ? threat.embedding.length : 'N/A'}</p>
            </div>
        `;
    }

    exportVisualization() {
        if (!this.interactiveChart) {
            this.showError('No visualization to export');
            return;
        }
        
        Plotly.downloadImage(this.interactiveChart, {
            format: 'png',
            width: 1200,
            height: 800,
            filename: 'threat_embeddings_visualization'
        });
    }

    showLoadingIndicator(message = 'Loading...') {
        const indicator = document.getElementById('loadingIndicator');
        if (indicator) {
            indicator.textContent = message;
            indicator.style.display = 'block';
        }
    }

    hideLoadingIndicator() {
        const loadingContainer = document.getElementById('loadingIndicator');
        if (loadingContainer) {
            loadingContainer.style.display = 'none';
        }
    }

    showMessage(message, type = 'info') {
        const container = document.getElementById('messageContainer') || document.getElementById('errorContainer');
        if (container) {
            const alertClass = type === 'error' ? 'alert-danger' : 
                              type === 'success' ? 'alert-success' : 'alert-info';
            container.innerHTML = `<div class="alert ${alertClass}">${message}</div>`;
            container.style.display = 'block';
            
            setTimeout(() => {
                container.style.display = 'none';
            }, 3000);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }

    showError(message) {
        const container = document.getElementById('errorContainer');
        if (container) {
            container.innerHTML = `<div class="alert alert-danger">${message}</div>`;
            container.style.display = 'block';
            
            setTimeout(() => {
                container.style.display = 'none';
            }, 5000);
        }
    }
}

class SeededRandom {
    constructor(seed) {
        this.seed = seed;
    }
    
    random() {
        this.seed = (this.seed * 9301 + 49297) % 233280;
        return this.seed / 233280;
    }
    
    uniform(min, max) {
        return min + (max - min) * this.random();
    }
    
    normal(mean = 0, std = 1) {
        if (this.spare !== undefined) {
            const value = this.spare;
            this.spare = undefined;
            return value * std + mean;
        }
        
        const u = this.random();
        const v = this.random();
        const mag = std * Math.sqrt(-2 * Math.log(u));
        this.spare = mag * Math.cos(2 * Math.PI * v);
        return mag * Math.sin(2 * Math.PI * v) + mean;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ThreatEmbeddingsAnalyzer, SeededRandom };
}

(function() {
    'use strict';
    
    let threatEmbeddingsAnalyzer = null;
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeEmbeddingsAnalysis);
    } else {
        setTimeout(initializeEmbeddingsAnalysis, 100);
    }
    
    async function initializeEmbeddingsAnalysis() {
        console.log('Initializing Embeddings Analysis module...');
        
        try {
            threatEmbeddingsAnalyzer = new ThreatEmbeddingsAnalyzer();
            
            const success = await threatEmbeddingsAnalyzer.initialize();
            
            if (success) {
                console.log('Embeddings Analysis module initialized successfully');
                
                window.threatEmbeddingsAnalyzer = threatEmbeddingsAnalyzer;
                
                window.ThreatEmbeddingsAnalyzer = ThreatEmbeddingsAnalyzer;
            } else {
                console.error('Failed to initialize Embeddings Analysis module');
            }
        } catch (error) {
            console.error('Error initializing Embeddings Analysis module:', error);
        }
    }
})(); 