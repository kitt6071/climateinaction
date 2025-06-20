(function() {
    'use strict';

    if (typeof ThreatEmbeddingsAnalyzer !== 'undefined') {
        ThreatEmbeddingsAnalyzer.prototype.setOptimalClusters = function(optimalK) {
            if (!optimalK) return;
            
            this.config.clustering.numClusters = optimalK;
            
            const clusterSlider = document.getElementById('numClusters');
            if (clusterSlider) {
                clusterSlider.value = optimalK;
                
                const event = new Event('input');
                clusterSlider.dispatchEvent(event);
            }
            
            this.showMessage(`Applied optimal K=${optimalK}`, 'success');
            
            const modal = document.getElementById('stabilityResults');
            if (modal) {
                modal.style.display = 'none';
            }
        };

        ThreatEmbeddingsAnalyzer.prototype.handlePointClick = function(data, plotData) {
            if (!data.points || data.points.length === 0) return;
            
            const pointIndex = data.points[0].pointIndex;
            if (pointIndex >= plotData.length) {
                console.warn('Point index out of bounds:', pointIndex, 'max:', plotData.length);
                return;
            }
            
            const originalIndex = plotData[pointIndex].originalIndex;
            const threatData = this.embeddings[originalIndex];
            
            if (!threatData) {
                console.warn('No threat data found for index:', originalIndex);
                return;
            }
            
            this.displayThreatDetails(threatData, plotData[pointIndex].cluster);
        };
    }

    console.log('Click handler loaded.');
})();