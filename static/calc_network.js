function calculateSystemicRiskMetrics() {
    console.log('Calculating systemic risk metrics...');
    
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
}

function calculateNetworkModularity() {
    const totalEdges = knowledgeGraph.edges.size;
    if (totalEdges === 0) return 0;
    
    const communities = new Map();
    knowledgeGraph.edges.forEach(edge => {
        if (edge.type === 'EXPERIENCES_IMPACT') {
            const impactNode = knowledgeGraph.nodes.get(edge.target);
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
            knowledgeGraph.edges.forEach(edge => {
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
    knowledgeGraph.edges.forEach(edge => {
        if (edge.type === 'HAS_THREAT') {
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
    const totalSpecies = knowledgeGraph.speciesNodes.size;
    const totalThreats = knowledgeGraph.threatNodes.size;
    const totalEdges = knowledgeGraph.edges.size;
    
    if (totalSpecies === 0) return 0;
    
    const avgThreatsPerSpecies = totalThreats / totalSpecies;
    const maxPossibleEdges = totalSpecies * totalThreats;
    const networkDensity = maxPossibleEdges > 0 ? totalEdges / maxPossibleEdges : 0;
    
    const speciesVulnerabilities = [];
    knowledgeGraph.speciesNodes.forEach(speciesNode => {
        let threatCount = 0;
        knowledgeGraph.edges.forEach(edge => {
            if (edge.type === 'HAS_THREAT' && edge.source === speciesNode.id) {
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
    knowledgeGraph.edges.forEach(edge => {
        if (edge.type === 'HAS_THREAT') {
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
