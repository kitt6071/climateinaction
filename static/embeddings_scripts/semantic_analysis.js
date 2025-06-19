(function() {
    'use strict';

    if (typeof ThreatEmbeddingsAnalyzer === 'undefined') {
        return;
    }

    const _extract = (items, stopWords, minLength = 3) => {
        const text = items.filter(i => i && i.trim()).join(' ').toLowerCase();
        const words = text.split(/\s+/);
        return words.filter(word =>
            word.length >= minLength &&
            !stopWords.has(word) &&
            /^[a-z]+$/.test(word)
        );
    };

    const threatStopWords = new Set(['and', 'or', 'of', 'to', 'in', 'from', 'by', 'for', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cannot', 'due', 'caused', 'through', 'with', 'within', 'without', 'between', 'among', 'during', 'before', 'after', 'above', 'below', 'such', 'some', 'many', 'much', 'more', 'most', 'other', 'another', 'same', 'different']);
    const speciesStopWords = new Set(['species', 'genus', 'family', 'order', 'class', 'subspecies', 'var', 'subsp']);
    const predicateStopWords = new Set(['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cannot', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']);
    
    ThreatEmbeddingsAnalyzer.prototype.generateClusterLabels = async function() {
        if (!this.clusters) {
            this.showError('No clusters available. Please perform clustering first.');
            return;
        }
        
        this.showLoadingIndicator('Generating labels...');
        this.clusterLabels = [];
        
        try {
            const clusterStats = this.calculateClusterStatistics();
            const labelPromises = [];

            for (const clusterId in clusterStats.clusterThreats) {
                if (clusterStats.clusterThreats.hasOwnProperty(clusterId)) {
                    const threats = clusterStats.clusterThreats[clusterId];
                    const species = clusterStats.clusterSpecies[clusterId];
                    const predicates = clusterStats.clusterPredicates[clusterId];
                    
                    const promise = this.generateClusterLabel(threats, species, predicates)
                        .then(label => {
                            this.clusterLabels[parseInt(clusterId)] = label;
                        });
                    labelPromises.push(promise);
                }
            }

            await Promise.all(labelPromises);
            
            this.displayClusterLabels();
            this.hideLoadingIndicator();
            
        } catch (error) {
            console.error('Error generating cluster labels:', error);
            this.hideLoadingIndicator();
            this.showError('Failed to generate cluster labels: ' + error.message);
        }
    };

    ThreatEmbeddingsAnalyzer.prototype.generateClusterLabel = async function(threats, species, predicates) {
        const threatSample = threats.slice(0, 10);
        const speciesSample = species.slice(0, 10);
        const predicateSample = predicates.slice(0, 10);
        
        const threatKeywords = this.extractKeywords(threatSample);
        const speciesKeywords = this.extractSpeciesKeywords(speciesSample);
        const predicateKeywords = this.extractPredicateKeywords(predicateSample);
        
        const keywordFreq = {};
        threatKeywords.forEach(keyword => { keywordFreq[keyword] = (keywordFreq[keyword] || 0) + 3; });
        predicateKeywords.forEach(keyword => { keywordFreq[keyword] = (keywordFreq[keyword] || 0) + 2; });
        speciesKeywords.forEach(keyword => { keywordFreq[keyword] = (keywordFreq[keyword] || 0) + 1; });
        
        const topKeywords = Object.entries(keywordFreq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 6)
            .map(([keyword, _]) => keyword);
        
        let label = 'Unknown Threats';
        
        if (predicateKeywords.some(p => ['affected', 'threatened'].includes(p))) {
            if (topKeywords.some(k => ['habitat', 'loss'].includes(k))) { label = 'Species Affected by Habitat Loss'; }
            else if (topKeywords.some(k => ['climate', 'warming'].includes(k))) { label = 'Species Threatened by Climate Change'; }
            else { label = 'Species Under Threat'; }
        }
        else if (predicateKeywords.some(p => ['caused', 'due'].includes(p))) {
            if (topKeywords.some(k => ['pollution', 'contamination'].includes(k))) { label = 'Threats Caused by Pollution'; }
            else if (topKeywords.some(k => ['development', 'urban'].includes(k))) { label = 'Threats from Development'; }
            else { label = 'Human-Caused Threats'; }
        }
        else if (predicateKeywords.some(p => ['exposed', 'subjected'].includes(p))) { label = 'Species Exposed to Environmental Stressors'; }
        else if (predicateKeywords.some(p => ['impacted', 'influenced'].includes(p))) { label = 'Species Impacted by External Factors'; }
        else if (topKeywords.some(k => ['habitat', 'loss', 'degradation'].includes(k))) {
            if (topKeywords.some(k => ['forest', 'tree'].includes(k))) { label = 'Forest Habitat Loss'; }
            else if (topKeywords.some(k => ['wetland', 'marsh'].includes(k))) { label = 'Wetland Degradation'; }
            else if (topKeywords.some(k => ['marine', 'ocean'].includes(k))) { label = 'Marine Habitat Loss'; }
            else { label = 'Habitat Loss and Degradation'; }
        } 
        else if (topKeywords.some(k => ['pollution', 'contamination', 'chemical'].includes(k))) { label = 'Pollution and Contamination'; }
        else if (topKeywords.some(k => ['climate', 'temperature', 'warming'].includes(k))) { label = 'Climate Change Impacts'; }
        else if (topKeywords.some(k => ['invasive', 'introduced', 'alien'].includes(k))) { label = 'Invasive Species'; }
        else if (topKeywords.some(k => ['disease', 'pathogen', 'virus'].includes(k))) { label = 'Disease and Pathogens'; }
        else if (topKeywords.some(k => ['hunting', 'harvesting', 'fishing'].includes(k))) {
            if (topKeywords.some(k => ['fishing', 'marine'].includes(k))) { label = 'Overfishing'; }
            else { label = 'Overexploitation'; }
        } 
        else if (topKeywords.some(k => ['development', 'urban', 'construction'].includes(k))) { label = 'Urban Development'; }
        else if (topKeywords.some(k => ['bird', 'avian', 'passerine', 'raptor'].includes(k))) { label = `Bird Species Threats`; }
        else if (topKeywords.some(k => ['mammal', 'primate', 'ungulate'].includes(k))) { label = `Mammal Species Threats`; }
        else if (topKeywords.some(k => ['reptile', 'snake', 'lizard', 'turtle'].includes(k))) { label = `Reptile Species Threats`; }
        else if (topKeywords.some(k => ['amphibian', 'frog', 'salamander'].includes(k))) { label = `Amphibian Species Threats`; }
        else if (topKeywords.some(k => ['fish', 'marine', 'freshwater'].includes(k))) { label = `Aquatic Species Threats`; }
        else {
            const mainKeywords = topKeywords.slice(0, 2).filter(k => k.length > 4);
            if (mainKeywords.length > 0) {
                label = `${mainKeywords.join(' & ')} Related Threats`;
            }
        }
        
        const speciesCounts = {};
        speciesSample.forEach(sp => { if (sp && sp.trim()) { speciesCounts[sp] = (speciesCounts[sp] || 0) + 1; } });
        
        const predicateCounts = {};
        predicateSample.forEach(pred => { if (pred && pred.trim()) { predicateCounts[pred] = (predicateCounts[pred] || 0) + 1; } });
        
        const topSpecies = Object.entries(speciesCounts).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([species, _]) => species);
        const topPredicates = Object.entries(predicateCounts).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([predicate, _]) => predicate);
        
        return {
            label: label,
            keywords: topKeywords,
            threatKeywords: threatKeywords.slice(0, 3),
            speciesKeywords: speciesKeywords.slice(0, 3),
            predicateKeywords: predicateKeywords.slice(0, 3),
            topSpecies: topSpecies,
            topPredicates: topPredicates,
            confidence: 0.85,
            sampleThreats: threatSample.slice(0, 3)
        };
    };

    ThreatEmbeddingsAnalyzer.prototype.populatePatternDiscovery = function(clusterStats) {
        this.identifyOutliers(clusterStats);
        this.analyzeThreatGradients();
        this.analyzeClusterRelationships(clusterStats);
    };

    ThreatEmbeddingsAnalyzer.prototype.identifyOutliers = function(clusterStats) {
        const outlierContainer = document.getElementById('outlierThreats');
        if (!outlierContainer) return;
        
        const totalThreats = this.embeddings.length;
        const outlierThreshold = Math.max(2, totalThreats * 0.05);
        
        const outlierClusters = Object.entries(clusterStats.clusterCounts)
            .filter(([cluster, count]) => count <= outlierThreshold)
            .map(([cluster, count]) => ({ cluster: parseInt(cluster), count }));
        
        let html = '';
        if (outlierClusters.length > 0) {
            html += '<h5>Outlier Groups</h5>';
            outlierClusters.forEach(({cluster, count}) => {
                const threats = (clusterStats.clusterThreats[cluster] || []).slice(0, 3);
                let threatText = threats.join(', ');
                if (clusterStats.clusterThreats[cluster].length > 3) {
                    threatText += ` and ${clusterStats.clusterThreats[cluster].length - 3} more.`;
                }
                html += `<p><strong>Cluster ${cluster}</strong> (${count} threats): ${threatText}</p>`;
            });
        } else {
            html += '<p>No significant outlier groups detected.</p>';
        }
        
        outlierContainer.innerHTML = html;
    };

    ThreatEmbeddingsAnalyzer.prototype.analyzeThreatGradients = function() {
        const gradientContainer = document.getElementById('threatGradients');
        if (!gradientContainer) return;
        
        const gradients = [
            { name: 'Severity', description: 'From mild environmental changes to severe habitat destruction.', examples: ['Temperature fluctuation', 'Habitat fragmentation', 'Complete habitat loss'] },
            { name: 'Human Impact', description: 'From indirect human effects to direct exploitation.', examples: ['Climate change', 'Urban development', 'Direct hunting/harvesting'] },
            { name: 'Temporal Scale', description: 'From acute, immediate threats to chronic, long-term pressures.', examples: ['Chemical spill', 'Pollution buildup', 'Climate change adaptation'] }
        ];
        
        let html = '<h5>Semantic Gradients</h5>';
        gradients.forEach(g => {
            html += `<p><strong>${g.name}:</strong> ${g.description} <br><em>e.g., ${g.examples.join(' → ')}</em></p>`;
        });
        
        gradientContainer.innerHTML = html;
    };

    ThreatEmbeddingsAnalyzer.prototype.analyzeClusterRelationships = function(clusterStats) {
        const relationshipContainer = document.getElementById('clusterRelationships');
        if (!relationshipContainer) return;
        
        let html = '<h5>Cluster Relationships</h5>';
        
        if (this.reducedEmbeddings && this.clusters) {
            const centroids = {};
            const clusterCounts = {};
            
            this.reducedEmbeddings.forEach((point, idx) => {
                const cluster = this.clusters[point.originalIndex];
                if (!centroids[cluster]) {
                    centroids[cluster] = { x: 0, y: 0 };
                    clusterCounts[cluster] = 0;
                }
                centroids[cluster].x += point.x;
                centroids[cluster].y += point.y;
                clusterCounts[cluster]++;
            });
            
            Object.keys(centroids).forEach(cluster => {
                centroids[cluster].x /= clusterCounts[cluster];
                centroids[cluster].y /= clusterCounts[cluster];
            });
            
            const clusterPairs = [];
            const clusterIds = Object.keys(centroids);
            
            for (let i = 0; i < clusterIds.length; i++) {
                for (let j = i + 1; j < clusterIds.length; j++) {
                    const id1 = clusterIds[i];
                    const id2 = clusterIds[j];
                    const dist = Math.hypot(centroids[id1].x - centroids[id2].x, centroids[id1].y - centroids[id2].y);
                    clusterPairs.push({ cluster1: id1, cluster2: id2, distance: dist });
                }
            }
            
            clusterPairs.sort((a, b) => a.distance - b.distance);
            
            html += '<h6>Closest (Most Similar) Clusters:</h6>';
            clusterPairs.slice(0, 3).forEach(pair => {
                const threat1 = (clusterStats.clusterThreats[pair.cluster1] || ['N/A'])[0];
                const threat2 = (clusterStats.clusterThreats[pair.cluster2] || ['N/A'])[0];
                html += `<p><strong>Cluster ${pair.cluster1} & ${pair.cluster2}:</strong><br><em>${threat1} | ${threat2}</em></p>`;
            });
        } else {
            html += '<p>Analysis requires clustering and dimensionality reduction.</p>';
        }
        
        relationshipContainer.innerHTML = html;
    };

    ThreatEmbeddingsAnalyzer.prototype.assessClusterInterpretability = function(clusterStats) {
        let interpretabilityScore = 0;
        let totalClusters = 0;
        
        for (const [clusterId, threats] of Object.entries(clusterStats.clusterThreats)) {
            if (threats.length === 0) continue;
            
            const keywords = this.extractKeywords(threats);
            const keywordFreq = {};
            keywords.forEach(keyword => { keywordFreq[keyword] = (keywordFreq[keyword] || 0) + 1; });
            
            const totalKeywords = keywords.length;
            const maxKeywordFreq = Math.max(...Object.values(keywordFreq), 0);
            const coherence = totalKeywords > 0 ? maxKeywordFreq / totalKeywords : 0;
            
            interpretabilityScore += coherence;
            totalClusters++;
        }
        
        return totalClusters > 0 ? interpretabilityScore / totalClusters : 0;
    };

    ThreatEmbeddingsAnalyzer.prototype.extractKeywords = function(threats) {
        return _extract(threats, threatStopWords, 3);
    };

    ThreatEmbeddingsAnalyzer.prototype.extractSpeciesKeywords = function(species) {
        return _extract(species, speciesStopWords, 3);
    };

    ThreatEmbeddingsAnalyzer.prototype.extractPredicateKeywords = function(predicates) {
        return _extract(predicates, predicateStopWords, 2);
    };

    console.log('Semantic analysis module loaded and methods added to ThreatEmbeddingsAnalyzer');
})();
