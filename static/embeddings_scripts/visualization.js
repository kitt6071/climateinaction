(function() {
    'use strict';

    if (typeof ThreatEmbeddingsAnalyzer !== 'undefined') {
        ThreatEmbeddingsAnalyzer.prototype.createInteractiveVisualization = function() {
            console.log('Creating visualization.');

            if (!this.reducedEmbeddings) {
                console.warn('No reduced embeddings available.');
                return;
            }

            if (!window.Plotly) {
                console.error('Plotly is not available.');
                this.showError('Plotly library is required for visualization.');
                return;
            }

            const container = document.getElementById('embeddingsVisualization');
            if (!container) {
                console.warn('Visualization container not found.');
                return;
            }

            try {
                container.innerHTML = '';
                container.style.width = '100%';
                container.style.height = '500px';
                container.style.display = 'block';

                const plotData = this.reducedEmbeddings.map((point, idx) => {
                    const originalData = this.embeddings[point.originalIndex || point.index || idx];
                    const threatText = originalData.text ||
                        originalData.threat ||
                        `${originalData.subject || originalData.species || 'Unknown'} ${originalData.predicate || 'experienced'} ${originalData.object || originalData.impact || 'Unknown'}`;

                    return {
                        x: point.x,
                        y: point.y,
                        text: threatText.substring(0, 100) + (threatText.length > 100 ? '...' : ''),
                        species: originalData.species || originalData.subject || 'Unknown',
                        impact: originalData.impact || originalData.object || 'Unknown',
                        cluster: this.clusters ? this.clusters[point.originalIndex || point.index || idx] : 0,
                        originalIndex: point.originalIndex || point.index || idx
                    };
                });

                console.log(`Plot data prepared: ${plotData.length} points.`);

                const trace = {
                    x: plotData.map(d => d.x),
                    y: plotData.map(d => d.y),
                    mode: 'markers',
                    type: 'scattergl',
                    text: plotData.map(d => `Species: ${d.species}<br>Impact: ${d.impact}<br>${d.text}`),
                    hovertemplate: '%{text}<extra></extra>',
                    marker: {
                        size: Math.max(4, Math.min(8, 200 / Math.sqrt(plotData.length))),
                        opacity: Math.max(0.6, Math.min(1.0, 1000 / plotData.length)),
                        color: this.clusters ? plotData.map(d => d.cluster) : '#1f77b4',
                        colorscale: this.clusters ? 'Viridis' : undefined,
                        showscale: !!this.clusters,
                        colorbar: this.clusters ? {
                            title: 'Cluster',
                            titleside: 'right'
                        } : undefined
                    }
                };

                const layout = {
                    title: `Threat Embeddings Visualization (${plotData.length} points)`,
                    xaxis: { title: 'Dimension 1' },
                    yaxis: { title: 'Dimension 2' },
                    autosize: true,
                    margin: { l: 50, r: 50, t: 50, b: 50 }
                };

                const config = {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false
                };

                console.log('Creating Plotly plot.');

                Plotly.newPlot(container, [trace], layout, config)
                    .then(() => {
                        console.log('Plotly visualization created.');
                        this.interactiveChart = container;

                        container.on('plotly_click', (data) => {
                            if (data.points && data.points.length > 0) {
                                const pointIndex = data.points[0].pointIndex;
                                const originalIndex = plotData[pointIndex].originalIndex;
                                const threatData = this.embeddings[originalIndex];
                                this.displayThreatDetails(threatData);
                            }
                        });
                    })
                    .catch(error => {
                        console.error('Error creating Plotly plot:', error);
                        this.showError(`Failed to create visualization: ${error.message}`);
                    });

            } catch (error) {
                console.error('Error in createInteractiveVisualization:', error);
                this.showError(`Failed to create visualization: ${error.message}`);
            }
        };

        ThreatEmbeddingsAnalyzer.prototype.updateVisualizationWithClusters = function() {
            if (!this.interactiveChart || !this.clusters) return;

            const colors = this.clusters.map(cluster => cluster);

            Plotly.restyle(this.interactiveChart, {
                'marker.color': [colors],
                'marker.showscale': true
            });
        };

        ThreatEmbeddingsAnalyzer.prototype.displayClusterLabels = function() {
            const container = document.getElementById('clusterLabels');
            if (!container) return;

            const html = `
                <div class="cluster-labels">
                    <h4>Cluster Labels</h4>
                    ${this.clusterLabels.map((labelInfo, clusterId) => `
                        <div class="cluster-label-item">
                            <h5>Cluster ${clusterId}: ${labelInfo.label}</h5>
                            ${labelInfo.keywords && labelInfo.keywords.length ? `<p><strong>Key Terms:</strong> ${labelInfo.keywords.join(', ')}</p>` : ''}
                            ${labelInfo.threatKeywords && labelInfo.threatKeywords.length ? `<p><strong>Threat Keywords:</strong> ${labelInfo.threatKeywords.join(', ')}</p>` : ''}
                            ${labelInfo.predicateKeywords && labelInfo.predicateKeywords.length ? `<p><strong>Relationship Keywords:</strong> ${labelInfo.predicateKeywords.join(', ')}</p>` : ''}
                            ${labelInfo.speciesKeywords && labelInfo.speciesKeywords.length ? `<p><strong>Species Keywords:</strong> ${labelInfo.speciesKeywords.join(', ')}</p>` : ''}
                            
                            ${labelInfo.topPredicates && labelInfo.topPredicates.length ? `
                                <p><strong>Common Relationships:</strong></p>
                                <ul>${labelInfo.topPredicates.map(p => `<li><strong>"${p}"</strong></li>`).join('')}</ul>
                            ` : ''}

                            ${labelInfo.topSpecies && labelInfo.topSpecies.length ? `
                                <p><strong>Top Species:</strong></p>
                                <ul>${labelInfo.topSpecies.map(s => `<li><em>${s}</em></li>`).join('')}</ul>
                            ` : ''}

                            <p><strong>Confidence:</strong> ${(labelInfo.confidence * 100).toFixed(1)}%</p>
                            
                            ${labelInfo.sampleThreats && labelInfo.sampleThreats.length ? `
                                <p><strong>Sample Threats:</strong></p>
                                <ul>${labelInfo.sampleThreats.map(t => `<li>${t}</li>`).join('')}</ul>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>`;
            container.innerHTML = html.trim();
        };

        ThreatEmbeddingsAnalyzer.prototype.updateEmbeddingsInfo = function() {
            const container = document.getElementById('embeddingsInfo');
            if (!container || !this.embeddings || this.embeddings.length === 0) return;

            const uniqueSpecies = new Set(this.embeddings.map(e => e.species || e.subject || '')).size;
            const uniqueImpacts = new Set(this.embeddings.map(e => e.impact || e.object || '')).size;
            const uniqueCategories = new Set(this.embeddings.map(e => e.category || '')).size;
            
            let clusterInfo = '';
            if (this.clusters) {
                const numClusters = new Set(this.clusters.filter(c => c >= 0)).size;
                const noisePoints = this.clusters.filter(c => c === -1).length;
                let noiseText = noisePoints > 0 ? ` (${noisePoints} noise points)` : '';
                clusterInfo = `<p><strong>Clusters Found:</strong> ${numClusters}${noiseText}</p>`;
            }

            let reductionInfo = '';
            if (this.reducedEmbeddings) {
                const methodName = (this.currentMethod || 'unknown').toUpperCase();
                reductionInfo = `<p><strong>Dimensionality Reduction:</strong> ${methodName} to 2D</p>`;
            }

            const html = `
                <div class="embeddings-info">
                    <h4>Embeddings Dataset Information</h4>
                    <p><strong>Total Threats:</strong> ${this.embeddings.length.toLocaleString()}</p>
                    <p><strong>Unique Species:</strong> ${uniqueSpecies.toLocaleString()}</p>
                    <p><strong>Unique Impact Types:</strong> ${uniqueImpacts.toLocaleString()}</p>
                    ${uniqueCategories > 0 ? `<p><strong>Unique Categories:</strong> ${uniqueCategories.toLocaleString()}</p>` : ''}
                    <p><strong>Embedding Dimensions:</strong> ${this.embeddings[0].embedding ? this.embeddings[0].embedding.length : 'N/A'}</p>
                    ${reductionInfo}
                    ${clusterInfo}
                </div>`;

            container.innerHTML = html.trim();
        };

        ThreatEmbeddingsAnalyzer.prototype.updateClusterQualityDisplay = function(silhouetteScore, clusterStats) {
            const container = document.getElementById('clusteringSummary');
            if (!container) return;

            let qualityText = '';
            if (!isNaN(silhouetteScore)) {
                if (silhouetteScore > 0.7) qualityText = 'Excellent clustering';
                else if (silhouetteScore > 0.5) qualityText = 'Good clustering';
                else if (silhouetteScore > 0.25) qualityText = 'Reasonable clustering';
                else qualityText = 'Poor clustering';
            }

            const clusterCountsList = Object.entries(clusterStats.clusterCounts).map(([cluster, count]) => {
                const percentage = this.embeddings && this.embeddings.length > 0 ?
                    (count / this.embeddings.length * 100).toFixed(1) : '0';
                return `<li>Cluster ${cluster}: ${count} threats (${percentage}%)</li>`;
            }).join('');

            const html = `
                <div class="clustering-summary">
                    <h4>Clustering Results & Quality</h4>
                    <p><strong>Number of Clusters:</strong> ${clusterStats.numClusters}</p>
                    <p><strong>Average Cluster Size:</strong> ${clusterStats.avgClusterSize.toFixed(1)}</p>
                    <p><strong>Silhouette Score:</strong> ${isNaN(silhouetteScore) ? 'N/A' : silhouetteScore.toFixed(3)}</p>
                    ${qualityText ? `<p><strong>Quality Assessment:</strong> ${qualityText}</p>` : ''}
                    <h5>Cluster Sizes:</h5>
                    <ul>${clusterCountsList}</ul>
                </div>`;
            
            container.innerHTML = html.trim();
            this.populatePatternDiscovery(clusterStats);
        };

        ThreatEmbeddingsAnalyzer.prototype.displayStabilityResults = function(stabilityResults) {
            let resultsContainer = document.getElementById('stabilityResults');
            if (!resultsContainer) {
                resultsContainer = document.createElement('div');
                resultsContainer.id = 'stabilityResults';
                resultsContainer.className = 'stability-results-modal';
                document.body.appendChild(resultsContainer);
            }

            const chartHtml = stabilityResults.numClusters.length > 0 
                ? `<h4>Stability Metrics by K</h4><div id="stabilityChart" style="width: 100%; height: 300px;"></div>`
                : '';

            const html = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>Clustering Stability Analysis Results</h3>
                        <button class="close-btn" onclick="this.closest('.stability-results-modal').style.display='none'">&#x2715;</button>
                    </div>
                    <div class="modal-body">
                        <h4>Recommended Settings</h4>
                        <p><strong>Optimal Number of Clusters:</strong> ${stabilityResults.bestK || 'Not determined'}</p>
                        
                        <h4>Stability Analysis Report</h4>
                        <pre>${stabilityResults.stabilityReport}</pre>
                        
                        ${chartHtml}

                        <div class="stability-actions">
                            <button class="btn btn-primary" onclick="this.setOptimalClusters(${stabilityResults.bestK})">Apply Optimal Settings</button>
                        </div>
                    </div>
                </div>`;

            resultsContainer.innerHTML = html.trim();
            resultsContainer.style.display = 'block';

            if (stabilityResults.numClusters.length > 0) {
                this.createStabilityChart(stabilityResults);
            }
        };

        ThreatEmbeddingsAnalyzer.prototype.createStabilityChart = function(stabilityResults) {
            const chartContainer = document.getElementById('stabilityChart');
            if (!chartContainer || !stabilityResults.numClusters || stabilityResults.numClusters.length === 0) return;
            
            const maxStability = Math.max(...stabilityResults.proportionalStability.filter(s => isFinite(s)));
            const maxSilhouette = Math.max(...stabilityResults.silhouetteScores.filter(s => isFinite(s)));

            const bars = stabilityResults.numClusters.map((k, i) => {
                const stability = stabilityResults.proportionalStability[i];
                const silhouette = stabilityResults.silhouetteScores[i];

                const stabilityHeight = (stability / maxStability) * 100;
                const silhouetteHeight = (silhouette / maxSilhouette) * 100;

                return `
                    <div class="chart-bar">
                        <div class="bar stability-bar" style="height: ${stabilityHeight}%" title="Stability: ${stability.toFixed(3)}"></div>
                        <div class="bar silhouette-bar" style="height: ${silhouetteHeight}%" title="Silhouette: ${silhouette.toFixed(3)}"></div>
                        <div class="bar-label">K=${k}</div>
                    </div>`;
            }).join('');

            const html = `
                <div class="stability-chart">${bars}</div>
                <div class="chart-legend">
                    <span class="legend-item"><span class="legend-color stability-color"></span> Stability Score</span>
                    <span class="legend-item"><span class="legend-color silhouette-color"></span> Silhouette Score</span>
                </div>`;

            chartContainer.innerHTML = html.trim();
        };
    }

    console.log('Visualization module loaded.');
})();

