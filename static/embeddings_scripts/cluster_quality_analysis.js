(function() {
    'use strict';

    if (typeof ThreatEmbeddingsAnalyzer !== 'undefined') {

        ThreatEmbeddingsAnalyzer.prototype.analyzeClusterQuality = function() {
            if (!this.clusters) return;

            console.log('Analyzing cluster quality...');

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

            console.log('Cluster Quality:', qualityResults);
            this.updateComprehensiveClusterQualityDisplay(qualityResults);
            return qualityResults;
        };

        ThreatEmbeddingsAnalyzer.prototype.calculateSilhouetteScore = function() {
            if (!this.clusters || !this.embeddings) return 0;

            const embeddings = this.embeddings.map(item => item.embedding);
            const n = embeddings.length;
            const MAX_SAMPLES_FOR_SILHOUETTE = 15000;
            const shouldSample = n > MAX_SAMPLES_FOR_SILHOUETTE;

            let sampleEmbeddings, sampleClusters;

            if (shouldSample) {
                console.log(`Sampling for silhouette score: ${MAX_SAMPLES_FOR_SILHOUETTE}/${n} points`);
                const indices = Array.from({ length: n }, (_, i) => i);
                for (let i = indices.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
                const sampleIndices = indices.slice(0, MAX_SAMPLES_FOR_SILHOUETTE);
                sampleEmbeddings = sampleIndices.map(i => embeddings[i]);
                sampleClusters = sampleIndices.map(i => this.clusters[i]);
            } else {
                sampleEmbeddings = embeddings;
                sampleClusters = this.clusters;
            }

            const sampleSize = sampleEmbeddings.length;
            let totalScore = 0;
            let validPoints = 0;
            const uniqueClusters = [...new Set(sampleClusters)].filter(c => c >= 0);

            if (uniqueClusters.length < 2) {
                console.warn('Need at least 2 clusters for silhouette score.');
                return 0;
            }

            for (let i = 0; i < sampleSize; i++) {
                const clusterI = sampleClusters[i];
                if (clusterI < 0) continue;

                const sameClusterPoints = [];
                for (let j = 0; j < sampleSize; j++) {
                    if (i !== j && sampleClusters[j] === clusterI) {
                        sameClusterPoints.push(j);
                    }
                }

                let a = 0;
                if (sameClusterPoints.length > 0) {
                    a = sameClusterPoints.reduce((sum, pointIdx) => sum + this.euclideanDistance(sampleEmbeddings[i], sampleEmbeddings[pointIdx]), 0) / sameClusterPoints.length;
                }

                let minB = Infinity;
                for (const otherCluster of uniqueClusters) {
                    if (otherCluster === clusterI) continue;

                    const otherClusterPoints = [];
                    for (let j = 0; j < sampleSize; j++) {
                        if (sampleClusters[j] === otherCluster) {
                            otherClusterPoints.push(j);
                        }
                    }

                    if (otherClusterPoints.length > 0) {
                        const avgDistToOtherCluster = otherClusterPoints.reduce((sum, pointIdx) => sum + this.euclideanDistance(sampleEmbeddings[i], sampleEmbeddings[pointIdx]), 0) / otherClusterPoints.length;
                        minB = Math.min(minB, avgDistToOtherCluster);
                    }
                }

                const b = minB === Infinity ? 0 : minB;
                const maxValue = Math.max(a, b);
                if (maxValue > 0) {
                    totalScore += (b - a) / maxValue;
                    validPoints++;
                }
            }

            const finalScore = validPoints > 0 ? totalScore / validPoints : 0;
            if (shouldSample) {
                console.log(`Sampled silhouette score: ${finalScore.toFixed(4)}`);
            }
            return finalScore;
        };

        ThreatEmbeddingsAnalyzer.prototype.calculateDaviesBouldinIndex = function() {
            if (!this.clusters || !this.embeddings) return Infinity;

            const embeddings = this.embeddings.map(item => item.embedding);
            const uniqueClusters = [...new Set(this.clusters)].filter(c => c >= 0);
            if (uniqueClusters.length < 2) return Infinity;

            const clusterPoints = {};
            uniqueClusters.forEach(cluster => {
                clusterPoints[cluster] = [];
            });
            this.clusters.forEach((cluster, idx) => {
                if (cluster >= 0) {
                    clusterPoints[cluster].push(embeddings[idx]);
                }
            });

            const centroids = {};
            uniqueClusters.forEach(cluster => {
                const points = clusterPoints[cluster];
                centroids[cluster] = new Array(embeddings[0].length).fill(0);
                if (points.length > 0) {
                    for (let dim = 0; dim < centroids[cluster].length; dim++) {
                        centroids[cluster][dim] = points.reduce((sum, point) => sum + point[dim], 0) / points.length;
                    }
                }
            });

            const intraClusterDistances = {};
            uniqueClusters.forEach(cluster => {
                const points = clusterPoints[cluster];
                if (points.length === 0) {
                    intraClusterDistances[cluster] = 0;
                    return;
                }
                intraClusterDistances[cluster] = points.reduce((sum, point) => sum + this.euclideanDistance(point, centroids[cluster]), 0) / points.length;
            });

            let totalDB = 0;
            uniqueClusters.forEach(clusterI => {
                let maxRatio = 0;
                uniqueClusters.forEach(clusterJ => {
                    if (clusterI !== clusterJ) {
                        const interClusterDistance = this.euclideanDistance(centroids[clusterI], centroids[clusterJ]);
                        if (interClusterDistance > 0) {
                            const ratio = (intraClusterDistances[clusterI] + intraClusterDistances[clusterJ]) / interClusterDistance;
                            maxRatio = Math.max(maxRatio, ratio);
                        }
                    }
                });
                totalDB += maxRatio;
            });

            return totalDB / uniqueClusters.length;
        };

        ThreatEmbeddingsAnalyzer.prototype.calculateCalinskiHarabaszIndex = function() {
            if (!this.clusters || !this.embeddings) return 0;

            const embeddings = this.embeddings.map(item => item.embedding);
            const n = embeddings.length;
            const uniqueClusters = [...new Set(this.clusters)].filter(c => c >= 0);
            const k = uniqueClusters.length;

            if (k < 2 || n === k) return 0;

            const overallCentroid = new Array(embeddings[0].length).fill(0);
            embeddings.forEach(embedding => {
                for (let dim = 0; dim < embedding.length; dim++) {
                    overallCentroid[dim] += embedding[dim];
                }
            });
            overallCentroid.forEach((val, idx) => {
                overallCentroid[idx] /= n;
            });

            const centroids = {};
            const clusterSizes = {};
            uniqueClusters.forEach(cluster => {
                centroids[cluster] = new Array(embeddings[0].length).fill(0);
                clusterSizes[cluster] = 0;
            });

            this.clusters.forEach((cluster, idx) => {
                if (cluster >= 0) {
                    for (let dim = 0; dim < embeddings[idx].length; dim++) {
                        centroids[cluster][dim] += embeddings[idx][dim];
                    }
                    clusterSizes[cluster]++;
                }
            });

            uniqueClusters.forEach(cluster => {
                if (clusterSizes[cluster] > 0) {
                    for (let dim = 0; dim < centroids[cluster].length; dim++) {
                        centroids[cluster][dim] /= clusterSizes[cluster];
                    }
                }
            });

            let bcss = 0;
            uniqueClusters.forEach(cluster => {
                const distance = this.euclideanDistance(centroids[cluster], overallCentroid);
                bcss += clusterSizes[cluster] * distance * distance;
            });

            let wcss = 0;
            this.clusters.forEach((cluster, idx) => {
                if (cluster >= 0) {
                    const distance = this.euclideanDistance(embeddings[idx], centroids[cluster]);
                    wcss += distance * distance;
                }
            });

            if (wcss === 0 || n - k === 0) return 0;

            const chIndex = (bcss / (k - 1)) / (wcss / (n - k));
            return isFinite(chIndex) ? chIndex : 0;
        };

        ThreatEmbeddingsAnalyzer.prototype.calculateOverallQuality = function(silhouetteScore, daviesBouldinIndex, interpretabilityScore) {
            const normalizedSilhouette = Math.max(0, Math.min(1, (silhouetteScore + 1) / 2));
            const normalizedDB = Math.max(0, Math.min(1, 1 / (1 + daviesBouldinIndex)));
            const normalizedInterpretability = Math.max(0, Math.min(1, interpretabilityScore));

            const weights = { silhouette: 0.4, daviesBouldin: 0.3, interpretability: 0.3 };

            return weights.silhouette * normalizedSilhouette +
                weights.daviesBouldin * normalizedDB +
                weights.interpretability * normalizedInterpretability;
        };

        ThreatEmbeddingsAnalyzer.prototype.performStabilityAnalysis = async function() {
            if (!this.embeddings || this.embeddings.length === 0) {
                console.warn('No embeddings for stability analysis.');
                return null;
            }

            console.log('Performing stability analysis...');
            this.showLoadingIndicator('Analyzing stability...');

            const stabilityResults = {
                numClusters: [],
                amiScores: [],
                silhouetteScores: [],
                bestK: null
            };

            const maxK = Math.min(15, Math.floor(this.embeddings.length / 20));
            const minK = 2;
            const numRuns = 5;
            const originalK = this.config.clustering.numClusters;

            try {
                const embeddings = this.embeddings.map(item => item.embedding);

                for (let k = minK; k <= maxK; k++) {
                    console.log(`Testing K=${k}...`);
                    this.config.clustering.numClusters = k;

                    const runResults = [];
                    const silhouetteScores = [];

                    for (let run = 0; run < numRuns; run++) {
                        const clusters = await this.performKMeans(embeddings);
                        if (clusters && clusters.length > 0) {
                            runResults.push(clusters);
                            const tempClusters = this.clusters;
                            this.clusters = clusters;
                            silhouetteScores.push(this.calculateSilhouetteScore());
                            this.clusters = tempClusters;
                        }
                    }

                    if (runResults.length >= 2) {
                        let totalAMI = 0;
                        for (let i = 0; i < runResults.length; i++) {
                            for (let j = i + 1; j < runResults.length; j++) {
                                totalAMI += this.calculateAMI(runResults[i], runResults[j]);
                            }
                        }

                        const comparisons = (runResults.length * (runResults.length - 1)) / 2;
                        const avgAMI = comparisons > 0 ? totalAMI / comparisons : 0;
                        const avgSilhouette = silhouetteScores.length > 0 ?
                            silhouetteScores.reduce((a, b) => a + b, 0) / silhouetteScores.length : 0;

                        stabilityResults.numClusters.push(k);
                        stabilityResults.amiScores.push(avgAMI);
                        stabilityResults.silhouetteScores.push(avgSilhouette);
                        console.log(`K=${k}: AMI=${avgAMI.toFixed(3)}, Silhouette=${avgSilhouette.toFixed(3)}`);
                    }
                }

                const bestKIndex = this.findOptimalK(stabilityResults);
                if (bestKIndex >= 0) {
                    stabilityResults.bestK = stabilityResults.numClusters[bestKIndex];
                }

                this.displayStabilityResults(stabilityResults);
                return stabilityResults;

            } catch (error) {
                console.error('Stability analysis error:', error);
                this.showError('Stability analysis failed: ' + error.message);
                return null;
            } finally {
                this.config.clustering.numClusters = originalK;
                this.hideLoadingIndicator();
            }
        };

        ThreatEmbeddingsAnalyzer.prototype.calculateAMI = function(clusters1, clusters2) {
            if (clusters1.length !== clusters2.length) return 0;

            const n = clusters1.length;
            const contingencyTable = {};
            const counts1 = {};
            const counts2 = {};

            for (let i = 0; i < n; i++) {
                const c1 = clusters1[i];
                const c2 = clusters2[i];
                if (!contingencyTable[c1]) contingencyTable[c1] = {};
                if (!contingencyTable[c1][c2]) contingencyTable[c1][c2] = 0;
                contingencyTable[c1][c2]++;
                counts1[c1] = (counts1[c1] || 0) + 1;
                counts2[c2] = (counts2[c2] || 0) + 1;
            }

            let mi = 0;
            const clusters1Set = Object.keys(counts1);
            const clusters2Set = Object.keys(counts2);

            for (const c1 of clusters1Set) {
                for (const c2 of clusters2Set) {
                    const nij = contingencyTable[c1]?.[c2] || 0;
                    if (nij > 0) {
                        const ni = counts1[c1];
                        const nj = counts2[c2];
                        mi += (nij / n) * Math.log((nij * n) / (ni * nj));
                    }
                }
            }
            return mi;
        };

        ThreatEmbeddingsAnalyzer.prototype.findOptimalK = function(stabilityResults) {
            if (stabilityResults.amiScores.length === 0) return -1;
            
            const scores = stabilityResults.amiScores.map((ami, i) => {
                return 0.6 * ami + 0.4 * stabilityResults.silhouetteScores[i];
            });

            let bestIndex = -1;
            let maxScore = -Infinity;

            scores.forEach((score, i) => {
                if (score > maxScore) {
                    maxScore = score;
                    bestIndex = i;
                }
            });

            return bestIndex;
        };

        ThreatEmbeddingsAnalyzer.prototype.displayStabilityResults = function(results) {
            console.log('Displaying stability results.');

            const modalHtml = `
                <div class="modal fade" id="stabilityResultsModal" tabindex="-1">
                    <div class="modal-dialog modal-lg">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">Clustering Stability</h5>
                                <button type="button" class="close" data-dismiss="modal"><span>&times;</span></button>
                            </div>
                            <div class="modal-body">
                                <div id="stabilityChart" style="height: 400px;"></div>
                                ${results.bestK ? `
                                    <div class="alert alert-success mt-3">
                                        Recommended: <strong>${results.bestK} clusters</strong>.
                                        <button id="applyOptimalClusters" class="btn btn-sm btn-success ml-2">Apply</button>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;

            document.body.insertAdjacentHTML('beforeend', modalHtml);
            $('#stabilityResultsModal').modal('show');
            this.createStabilityChart(results);

            const applyButton = document.getElementById('applyOptimalClusters');
            if (applyButton && results.bestK) {
                applyButton.addEventListener('click', () => {
                    this.setOptimalClusters(results.bestK);
                    $('#stabilityResultsModal').modal('hide');
                });
            }

            $('#stabilityResultsModal').on('hidden.bs.modal', function() {
                this.remove();
            });
        };

        ThreatEmbeddingsAnalyzer.prototype.createStabilityChart = function(results) {
            const traces = [{
                x: results.numClusters,
                y: results.amiScores,
                name: 'Stability (AMI)',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#007bff' }
            }, {
                x: results.numClusters,
                y: results.silhouetteScores,
                name: 'Quality (Silhouette)',
                type: 'scatter',
                mode: 'lines+markers',
                yaxis: 'y2',
                line: { color: '#28a745' }
            }];

            const layout = {
                title: 'Stability vs Quality',
                xaxis: { title: 'Number of Clusters' },
                yaxis: { title: 'AMI Score', side: 'left' },
                yaxis2: { title: 'Silhouette Score', side: 'right', overlaying: 'y' },
                legend: { x: 0.7, y: 1 }
            };

            Plotly.newPlot('stabilityChart', traces, layout);
        };
    }

    console.log('Cluster quality analysis module loaded.');
})();
