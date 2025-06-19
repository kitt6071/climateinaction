function calculateSystemicRiskMetrics() {
    console.log('Calculating systemic risk metrics...');
    
    const knowledgeGraph = window.knowledgeGraph || window.AppState?.knowledgeGraph;
    if (!knowledgeGraph || !knowledgeGraph.nodes || !knowledgeGraph.edges) {
        console.warn('Knowledge graph not available for systemic risk metrics');
        return;
    }
    
    const modularity = calculateNetworkModularity();
    const nestedness = calculateNetworkNestedness();
    const ecosystemVulnerability = calculateEcosystemVulnerability();
    const criticalSpeciesCount = countCriticalSpecies();
    
    const modularityElement = document.getElementById('modularityValue');
    if (modularityElement) modularityElement.textContent = modularity.toFixed(3);
    
    const nestednessElement = document.getElementById('nestednessValue');
    if (nestednessElement) nestednessElement.textContent = nestedness.toFixed(3);
    
    const vulnerabilityElement = document.getElementById('ecosystemVulnerability');
    if (vulnerabilityElement) vulnerabilityElement.textContent = ecosystemVulnerability.toFixed(2);
    
    const criticalCountElement = document.getElementById('criticalSpeciesCount');
    if (criticalCountElement) criticalCountElement.textContent = criticalSpeciesCount;
    
    console.log('Systemic risk metrics calculated:', {
        modularity,
        nestedness,
        ecosystemVulnerability,
        criticalSpeciesCount
    });
    if (typeof createSystemicRiskCharts === 'function') {
        createSystemicRiskCharts();
    }
    populateCriticalNodesAnalysis();
    populateVulnerabilityCorridors();
}

function calculateNetworkModularity() {
    const totalEdges = window.knowledgeGraph.edges.size;
    if (totalEdges === 0) return 0;
    
    const communities = new Map();
    window.knowledgeGraph.edges.forEach(edge => {
        if (edge.type === 'EXPERIENCES_IMPACT') {
            const impactNode = window.knowledgeGraph.nodes.get(edge.target);
            if (!communities.has(impactNode.name)) {
                communities.set(impactNode.name, new Set());
            }
            communities.get(impactNode.name).add(edge.source);
        }
    });
    
    let internalEdges = 0;
    communities.forEach(speciesSet => {
        if (speciesSet.size > 1) {
            let withinCommunityEdges = 0;
            window.knowledgeGraph.edges.forEach(edge => {
                if (speciesSet.has(edge.source) && speciesSet.has(edge.target)) {
                    withinCommunityEdges++;
                }
            });
            internalEdges += withinCommunityEdges;
        }
    });
    
    const modularity = internalEdges / totalEdges;
    return Math.min(modularity, 1.0);
}

function calculateNetworkNestedness() {
    const speciesThreats = new Map();
    window.knowledgeGraph.edges.forEach(edge => {
        if (edge.type === 'EXPERIENCES_IMPACT') {
            if (!speciesThreats.has(edge.source)) {
                speciesThreats.set(edge.source, new Set());
            }
            speciesThreats.get(edge.source).add(edge.target);
        }
    });
    
    if (speciesThreats.size < 2) return 0;
    
    let totalPairs = 0;
    let nestedPairs = 0;
    const speciesArray = Array.from(speciesThreats.values());
    
    for (let i = 0; i < speciesArray.length; i++) {
        for (let j = i + 1; j < speciesArray.length; j++) {
            const threats1 = speciesArray[i];
            const threats2 = speciesArray[j];
            
            totalPairs++;
            
            const intersectionSize = [...threats1].filter(t => threats2.has(t)).length;
            const smallerSize = Math.min(threats1.size, threats2.size);
            
            if (intersectionSize === smallerSize) {
                nestedPairs++;
            }
        }
    }
    
    return totalPairs > 0 ? nestedPairs / totalPairs : 0;
}

function calculateEcosystemVulnerability() {
    const totalSpecies = window.knowledgeGraph.speciesNodes.size;
    const totalThreats = window.knowledgeGraph.threatNodes.size;
    const totalEdges = window.knowledgeGraph.edges.size;
    
    if (totalSpecies === 0) return 0;
    
    const avgThreatsPerSpecies = totalThreats / totalSpecies;
    const maxPossibleEdges = totalSpecies * totalThreats;
    const networkDensity = maxPossibleEdges > 0 ? totalEdges / maxPossibleEdges : 0;
    
    const speciesVulnerabilities = [];
    window.knowledgeGraph.speciesNodes.forEach(speciesNode => {
        let threatCount = 0;
        window.knowledgeGraph.edges.forEach(edge => {
            if (edge.type === 'EXPERIENCES_IMPACT' && edge.source === speciesNode.id) {
                threatCount++;
            }
        });
        speciesVulnerabilities.push(threatCount);
    });
    
    const meanVulnerability = speciesVulnerabilities.reduce((a, b) => a + b, 0) / speciesVulnerabilities.length;
    
    let vulnerabilityCoefficient = 0;
    if (meanVulnerability > 0) {
        const variance = speciesVulnerabilities.reduce((acc, val) => acc + Math.pow(val - meanVulnerability, 2), 0) / speciesVulnerabilities.length;
        vulnerabilityCoefficient = Math.sqrt(variance) / meanVulnerability;
    }
    
    const vulnerabilityScore = Math.min(10, 
        (avgThreatsPerSpecies * 0.3) + 
        (networkDensity * 5) + 
        (vulnerabilityCoefficient * 2)
    );
    
    return isNaN(vulnerabilityScore) ? 0 : vulnerabilityScore;
}


function countCriticalSpecies() {
    const speciesConnections = new Map();
    window.knowledgeGraph.edges.forEach(edge => {
        if (edge.type === 'EXPERIENCES_IMPACT') {
            const count = speciesConnections.get(edge.source) || 0;
            speciesConnections.set(edge.source, count + 1);
        }
    });
    
    if (speciesConnections.size === 0) return 0;
    
    const connections = Array.from(speciesConnections.values()).sort((a, b) => b - a);
    const threshold = Math.max(1, Math.ceil(connections.length * 0.1));
    const criticalThreshold = connections[threshold - 1] || 1;
    
    let criticalCount = 0;
    speciesConnections.forEach(count => {
        if (count >= criticalThreshold) {
            criticalCount++;
        }
    });
    
    return criticalCount;
}

function populateCriticalNodesAnalysis() {
    const knowledgeGraph = window.knowledgeGraph || window.AppState?.knowledgeGraph;
    if (!knowledgeGraph || !knowledgeGraph.nodes) {
        console.warn('Knowledge graph not available for critical nodes analysis');
        return;
    }
    
    const nodeScores = [];
    knowledgeGraph.nodes.forEach((node, nodeId) => {
        let criticalityScore = 0;
        let threatCount = 0;
        let connectionCount = node.connections ? node.connections.size : 0;
        
        if (node.type === 'Species') {
            knowledgeGraph.edges.forEach(edge => {
                if (edge.type === 'EXPERIENCES_IMPACT' && edge.source === nodeId) {
                    threatCount++;
                }
            });
            criticalityScore = (threatCount * 0.6) + (connectionCount * 0.4);
        } else if (node.type === 'Threat') {
            knowledgeGraph.edges.forEach(edge => {
                if (edge.type === 'EXPERIENCES_IMPACT' && edge.target === nodeId) {
                    threatCount++;
                }
            });
            criticalityScore = (threatCount * 0.7) + (connectionCount * 0.3);
        }
        
        nodeScores.push({
            id: nodeId,
            name: node.name,
            type: node.type,
            criticalityScore: criticalityScore,
            threatCount: threatCount,
            connectionCount: connectionCount
        });
    });
    
    const criticalNodes = nodeScores
        .sort((a, b) => b.criticalityScore - a.criticalityScore)
        .slice(0, 15);
    
    const container = document.getElementById('criticalNodesTable');
    if (container) {
        let tableHTML = `
            <table class="critical-nodes-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Node</th>
                        <th>Type</th>
                        <th>Criticality Score</th>
                        <th>Threats/Affected</th>
                        <th>Connections</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        criticalNodes.forEach((node, index) => {
            tableHTML += `
                <tr>
                    <td>${index + 1}</td>
                    <td class="node-name">${node.name}</td>
                    <td class="node-type">${node.type}</td>
                    <td class="criticality-score">${node.criticalityScore.toFixed(2)}</td>
                    <td class="threat-count">${node.threatCount}</td>
                    <td class="connection-count">${node.connectionCount}</td>
                </tr>
            `;
        });
        
        tableHTML += '</tbody></table>';
        container.innerHTML = tableHTML;
    }
    
    console.log(`Critical nodes analysis completed: ${criticalNodes.length} critical nodes identified`);
}

function populateVulnerabilityCorridors() {
    const knowledgeGraph = window.knowledgeGraph || window.AppState?.knowledgeGraph;
    if (!knowledgeGraph || !knowledgeGraph.nodes) {
        console.warn('Knowledge graph not available for vulnerability corridors');
        return;
    }
    
    const vulnerableSpecies = [];
    knowledgeGraph.speciesNodes.forEach((speciesNode, nodeId) => {
        let threatCount = 0;
        knowledgeGraph.edges.forEach(edge => {
            if (edge.type === 'EXPERIENCES_IMPACT' && edge.source === nodeId) {
                threatCount++;
            }
        });
        
        if (threatCount > 3) {
            vulnerableSpecies.push({
                id: nodeId,
                name: speciesNode.name,
                threatCount: threatCount,
                connections: speciesNode.connections
            });
        }
    });
    
    const corridors = [];
    for (let i = 0; i < vulnerableSpecies.length; i++) {
        for (let j = i + 1; j < vulnerableSpecies.length; j++) {
            const species1 = vulnerableSpecies[i];
            const species2 = vulnerableSpecies[j];
            
            const sharedThreats = [];
            knowledgeGraph.edges.forEach(edge1 => {
                if (edge1.type === 'EXPERIENCES_IMPACT' && edge1.source === species1.id) {
                    knowledgeGraph.edges.forEach(edge2 => {
                        if (edge2.type === 'EXPERIENCES_IMPACT' && 
                            edge2.source === species2.id && 
                            edge1.target === edge2.target) {
                            const threatNode = knowledgeGraph.nodes.get(edge1.target);
                            if (threatNode && !sharedThreats.includes(threatNode.name)) {
                                sharedThreats.push(threatNode.name);
                            }
                        }
                    });
                }
            });
            
            if (sharedThreats.length > 0) {
                const riskLevel = sharedThreats.length > 2 ? 'High' : 
                                 sharedThreats.length > 1 ? 'Medium' : 'Low';
                
                corridors.push({
                    species1: species1.name,
                    species2: species2.name,
                    sharedThreats: sharedThreats,
                    riskLevel: riskLevel,
                    corridorStrength: sharedThreats.length
                });
            }
        }
    }
    
    const topCorridors = corridors
        .sort((a, b) => b.corridorStrength - a.corridorStrength)
        .slice(0, 10);
    
    const container = document.getElementById('vulnerabilityCorridors');
    if (container) {
        if (topCorridors.length === 0) {
            container.innerHTML = '<p>No significant vulnerability corridors detected.</p>';
            return;
        }
        
        let corridorHTML = '<div class="corridor-list">';
        
        topCorridors.forEach((corridor, index) => {
            const riskClass = corridor.riskLevel.toLowerCase();
            corridorHTML += `
                <div class="corridor-item risk-${riskClass}">
                    <div class="corridor-header">
                        <h5>Corridor ${index + 1}</h5>
                        <span class="risk-badge ${riskClass}">${corridor.riskLevel} Risk</span>
                    </div>
                    <div class="corridor-details">
                        <div class="species-pair">
                            <strong>${corridor.species1}</strong> ↔ <strong>${corridor.species2}</strong>
                        </div>
                        <div class="shared-threats">
                            <strong>Shared Threats (${corridor.sharedThreats.length}):</strong>
                            ${corridor.sharedThreats.slice(0, 3).join(', ')}
                            ${corridor.sharedThreats.length > 3 ? '...' : ''}
                        </div>
                    </div>
                </div>
            `;
        });
        
        corridorHTML += '</div>';
        container.innerHTML = corridorHTML;
    }
    
    console.log(`Vulnerability corridors analysis completed: ${topCorridors.length} corridors identified`);
}

window.addEventListener('knowledgeGraphReady', function(event) {
    console.log('Knowledge graph ready, calculating systemic risk metrics...');
    setTimeout(() => {
        calculateSystemicRiskMetrics();
    }, 200);
});
