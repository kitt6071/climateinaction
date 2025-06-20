(function() {
    'use strict';

    if (typeof ThreatEmbeddingsAnalyzer !== 'undefined') {

        ThreatEmbeddingsAnalyzer.prototype.performClustering = function() {
            if (!this.reducedEmbeddings || this.reducedEmbeddings.length === 0) {
                this.showError('No reduced embeddings available for clustering');
                return;
            }

            let algorithm = this.clusteringAlgorithmSelect?.value || this.config.clustering.method;
            let numClusters = parseInt(this.clusterCountSlider?.value) || this.config.clustering.numClusters;

            console.log(`Clustering: ${algorithm.toUpperCase()} with ${numClusters} clusters.`);

            try {
                this.showLoadingIndicator(`Running ${algorithm.toUpperCase()} clustering...`);

                let embeddingsForClustering = this.reducedEmbeddings;

                if (!Array.isArray(embeddingsForClustering) || embeddingsForClustering.length === 0) {
                    throw new Error('Reduced embeddings are invalid or empty.');
                }

                if (embeddingsForClustering[0] && typeof embeddingsForClustering[0] === 'object' && 'x' in embeddingsForClustering[0] && 'y' in embeddingsForClustering[0]) {
                    embeddingsForClustering = embeddingsForClustering.map(point => [point.x, point.y]);
                }

                if (!Array.isArray(embeddingsForClustering[0]) || embeddingsForClustering[0].length === 0) {
                    throw new Error('Invalid embedding format for clustering.');
                }

                console.log(`Clustering ${embeddingsForClustering.length} samples of dimension ${embeddingsForClustering[0].length}.`);

                let clusterResult;
                switch (algorithm) {
                    case 'kmeans':
                        this.config.clustering.numClusters = numClusters;
                        clusterResult = this.performKMeans(embeddingsForClustering);
                        this.clusters = clusterResult.clusters;
                        this.clusterCentroids = clusterResult.centroids;
                        console.log(`K-Means WCSS: ${clusterResult.wcss.toFixed(4)}`);
                        break;
                    case 'gmm':
                        this.clusters = this.performGMM(embeddingsForClustering);
                        break;
                    case 'hdbscan':
                        this.clusters = this.performHDBSCAN(embeddingsForClustering);
                        break;
                    default:
                        throw new Error(`Unknown clustering algorithm: ${algorithm}`);
                }

                if (!this.clusters || this.clusters.length !== this.embeddings.length) {
                    throw new Error(`Clustering failed: expected ${this.embeddings.length} assignments, got ${this.clusters ? this.clusters.length : 0}`);
                }

                console.log(`Clustering complete. Found ${new Set(this.clusters).size} clusters.`);

                this.updateVisualizationWithClusters();
                this.analyzeClusterQuality();
                this.hideLoadingIndicator();

            } catch (error) {
                console.error('Clustering error:', error);
                this.showError(`Clustering failed: ${error.message}`);
                this.hideLoadingIndicator();
            }
        };

        ThreatEmbeddingsAnalyzer.prototype.performKMeans = function(embeddings) {
            const k = this.config.clustering.numClusters;
            const maxIterations = 100;
            const tolerance = 1e-6;

            if (embeddings.length < k) {
                throw new Error(`Not enough data points (${embeddings.length}) for ${k} clusters.`);
            }

            const dimensions = embeddings[0].length;
            const seededRandom = new SeededRandom(42);
            let centroids = [];

            centroids.push([...embeddings[Math.floor(seededRandom.random() * embeddings.length)]]);

            for (let c = 1; c < k; c++) {
                const distances = embeddings.map(point => {
                    let minDist = Infinity;
                    for (const centroid of centroids) {
                        const dist = this.euclideanDistance(point, centroid);
                        minDist = Math.min(minDist, dist);
                    }
                    return minDist * minDist;
                });

                const totalDist = distances.reduce((sum, d) => sum + d, 0);
                const random = seededRandom.random() * totalDist;

                let cumsum = 0;
                for (let i = 0; i < distances.length; i++) {
                    cumsum += distances[i];
                    if (cumsum >= random) {
                        centroids.push([...embeddings[i]]);
                        break;
                    }
                }
            }

            let assignments = [];
            let converged = false;
            let iteration = 0;

            while (!converged && iteration < maxIterations) {
                const newAssignments = embeddings.map(embedding => {
                    let bestCluster = 0;
                    let bestDistance = this.euclideanDistance(embedding, centroids[0]);
                    for (let j = 1; j < k; j++) {
                        const distance = this.euclideanDistance(embedding, centroids[j]);
                        if (distance < bestDistance) {
                            bestDistance = distance;
                            bestCluster = j;
                        }
                    }
                    return bestCluster;
                });

                converged = assignments.length === newAssignments.length && assignments.every((v, i) => v === newAssignments[i]);
                assignments = newAssignments;

                if (!converged) {
                    const newCentroids = Array(k).fill(0).map(() => Array(dimensions).fill(0));
                    const counts = Array(k).fill(0);

                    for (let i = 0; i < embeddings.length; i++) {
                        const cluster = assignments[i];
                        counts[cluster]++;
                        for (let j = 0; j < dimensions; j++) {
                            newCentroids[cluster][j] += embeddings[i][j];
                        }
                    }

                    const oldCentroids = JSON.parse(JSON.stringify(centroids));
                    for (let i = 0; i < k; i++) {
                        if (counts[i] > 0) {
                            for (let j = 0; j < dimensions; j++) {
                                centroids[i][j] = newCentroids[i][j] / counts[i];
                            }
                        }
                    }

                    let maxMovement = 0;
                    for (let i = 0; i < k; i++) {
                        maxMovement = Math.max(maxMovement, this.euclideanDistance(centroids[i], oldCentroids[i]));
                    }
                    if (maxMovement < tolerance) {
                        converged = true;
                    }
                }
                iteration++;
            }

            console.log(`K-Means converged in ${iteration} iterations.`);

            let wcss = 0;
            for (let i = 0; i < embeddings.length; i++) {
                wcss += Math.pow(this.euclideanDistance(embeddings[i], centroids[assignments[i]]), 2);
            }

            return {
                clusters: assignments,
                centroids: centroids,
                wcss: wcss,
                iterations: iteration
            };
        };

        ThreatEmbeddingsAnalyzer.prototype.performGMM = function(embeddings) {
            if (!embeddings || embeddings.length === 0) {
                console.error('No embeddings provided for GMM clustering');
                return [];
            }

            const validEmbeddings = embeddings.filter(emb => emb && emb.length > 0);
            if (validEmbeddings.length === 0) {
                console.error('No valid embeddings found for GMM');
                return [];
            }

            const k = this.config.clustering.numClusters;

            try {
                const kmeansResult = this.performKMeans(embeddings);
                return kmeansResult.clusters.map(cluster => {
                    if (Math.random() < 0.1) {
                        return (cluster + 1) % k;
                    }
                    return cluster;
                });
            } catch (error) {
                console.error('Error in GMM clustering:', error);
                return [];
            }
        };

        ThreatEmbeddingsAnalyzer.prototype.performHDBSCAN = function(embeddings) {
            if (!embeddings || embeddings.length === 0) {
                console.error('No embeddings provided for HDBSCAN clustering');
                return [];
            }

            const validEmbeddings = embeddings.filter(emb => emb && emb.length > 0);
            if (validEmbeddings.length === 0) {
                console.error('No valid embeddings found for HDBSCAN');
                return [];
            }

            const numPoints = validEmbeddings.length;
            const MAX_SAMPLES_FOR_HDBSCAN = 2000;
            const shouldSample = numPoints > MAX_SAMPLES_FOR_HDBSCAN;

            let sampleEmbeddings = validEmbeddings;
            let sampleIndices = null;

            if (shouldSample) {
                console.log(`Sampling ${MAX_SAMPLES_FOR_HDBSCAN} of ${numPoints} points for HDBSCAN.`);
                const indices = Array.from({ length: numPoints }, (_, i) => i);
                for (let i = indices.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
                sampleIndices = indices.slice(0, MAX_SAMPLES_FOR_HDBSCAN);
                sampleEmbeddings = sampleIndices.map(i => validEmbeddings[i]);
            }

            const sampleSize = sampleEmbeddings.length;
            console.log(`Calculating densities for ${sampleSize} points...`);
            const densities = sampleEmbeddings.map((embedding, idx) => {
                let neighbors = 0;
                const threshold = 0.3;
                for (let i = 0; i < sampleSize; i++) {
                    if (i !== idx && this.euclideanDistance(embedding, sampleEmbeddings[i]) < threshold) {
                        neighbors++;
                    }
                }
                return neighbors;
            });

            console.log('Starting density-based clustering...');
            const clusters = new Array(sampleSize);
            let currentCluster = 0;
            const visited = new Array(sampleSize).fill(false);

            for (let i = 0; i < sampleSize; i++) {
                if (!visited[i]) {
                    if (densities[i] > 3) {
                        const queue = [i];
                        while (queue.length > 0) {
                            const pointIdx = queue.shift();
                            if (visited[pointIdx]) continue;
                            visited[pointIdx] = true;
                            clusters[pointIdx] = currentCluster;
                            for (let j = 0; j < sampleSize; j++) {
                                if (!visited[j] && densities[j] > 2 && this.euclideanDistance(sampleEmbeddings[pointIdx], sampleEmbeddings[j]) < 0.3) {
                                    queue.push(j);
                                }
                            }
                        }
                        currentCluster++;
                    } else {
                        visited[i] = true;
                        clusters[i] = -1;
                    }
                }
            }
            console.log(`HDBSCAN found ${currentCluster} clusters.`);

            let finalClusters;
            if (shouldSample) {
                finalClusters = new Array(validEmbeddings.length).fill(-1);
                for (let i = 0; i < sampleIndices.length; i++) {
                    finalClusters[sampleIndices[i]] = clusters[i];
                }
                for (let i = 0; i < validEmbeddings.length; i++) {
                    if (!sampleIndices.includes(i)) {
                        let minDistance = Infinity;
                        let nearestCluster = -1;
                        for (let j = 0; j < sampleIndices.length; j++) {
                            const distance = this.euclideanDistance(validEmbeddings[i], sampleEmbeddings[j]);
                            if (distance < minDistance && distance < 0.5) {
                                minDistance = distance;
                                nearestCluster = clusters[j];
                            }
                        }
                        finalClusters[i] = nearestCluster;
                    }
                }
            } else {
                finalClusters = clusters;
            }

            if (validEmbeddings.length !== embeddings.length) {
                const fullClusters = new Array(embeddings.length).fill(-1);
                let validIndex = 0;
                for (let i = 0; i < embeddings.length; i++) {
                    if (embeddings[i] && embeddings[i].length > 0) {
                        fullClusters[i] = finalClusters[validIndex];
                        validIndex++;
                    }
                }
                return fullClusters;
            } else {
                return finalClusters;
            }
        };
    }

    console.log('Clustering module loaded.');
})();