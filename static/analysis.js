(function() {
    'use strict';
    
    if (typeof window.AppState === 'undefined' || typeof window.AppUtils === 'undefined') {
        console.error('Global state not ready for analysis.js');
        return;
    }

    async function performNetworkAnalysis() {
        try {
            console.log('Performing network analysis...');
            let kg = window.AppState.knowledgeGraph;

            if (!kg || !kg.nodes || kg.nodes.size === 0) {
                console.log('Knowledge graph not initialized, creating from triplet data...');
                kg = createBasicKnowledgeGraph();
                if (!kg || !kg.nodes || kg.nodes.size === 0) {
                    throw new Error('No data available for network analysis.');
                }
                console.log(`Basic knowledge graph created: ${kg.nodes.size} nodes`);
            } else {
                console.log(`Using existing knowledge graph: ${kg.nodes.size} nodes`);
            }

            if (kg.nodes.size > 15000) {
                throw new Error(`Network too large for analysis (${kg.nodes.size} nodes). Max: 15000.`);
            }

            const edgeCount = kg.edges ? (kg.edges.size || kg.edges.length || 0) : 0;
            console.log(`Analyzing network with ${kg.nodes.size} nodes and ${edgeCount} edges`);

            const analysisResults = {
                centralityMetrics: calculateCentralityMetrics(kg),
                communityStructure: detectCommunities(kg),
                vulnerabilityScores: calculateVulnerabilityScores(kg),
                connectivityAnalysis: analyzeConnectivity(kg),
                timestamp: new Date().toISOString()
            };

            window.AppState.analysis.networkAnalysis = analysisResults;
            displayNetworkAnalysisResults(analysisResults);
            console.log('Network analysis completed.');
            return analysisResults;

        } catch (error) {
            console.error('Error performing network analysis:', error);
            displayAnalysisError('Network Analysis', error.message);
            throw error;
        }
    }

    function createBasicKnowledgeGraph() {
        const triplets = window.AppState.allTripletsData;
        if (!triplets || triplets.length === 0) {
            console.log('No triplets data available');
            return null;
        }

        const kg = {
            nodes: new Map(),
            edges: new Map(),
            speciesNodes: new Map(),
            threatNodes: new Map(),
            impactNodes: new Map()
        };

        console.log(`Creating graph from ${triplets.length} triplets...`);
        const tripletsToProcess = triplets.slice(0, 1000);
        console.log(`Processing ${tripletsToProcess.length} triplets.`);
        
        tripletsToProcess.forEach((triplet, index) => {
            if (!triplet.subject || !triplet.predicate || !triplet.object) return;

            const speciesId = `species_${triplet.subject}`;
            const threatId = `threat_${triplet.predicate}`;
            const impactId = `impact_${triplet.object}`;

            if (!kg.nodes.has(speciesId)) {
                kg.nodes.set(speciesId, {
                    id: speciesId,
                    label: triplet.subject,
                    type: 'species',
                    connections: new Set(),
                    threatCount: 0
                });
                kg.speciesNodes.set(speciesId, kg.nodes.get(speciesId));
            }

            if (!kg.nodes.has(threatId)) {
                kg.nodes.set(threatId, {
                    id: threatId,
                    label: triplet.predicate,
                    type: 'threat',
                    connections: new Set(),
                    speciesCount: 0
                });
                kg.threatNodes.set(threatId, kg.nodes.get(threatId));
            }

            if (!kg.nodes.has(impactId)) {
                kg.nodes.set(impactId, {
                    id: impactId,
                    label: triplet.object,
                    type: 'impact',
                    connections: new Set(),
                    threatCount: 0
                });
                kg.impactNodes.set(impactId, kg.nodes.get(impactId));
            }

            const edgeId1 = `${speciesId}_${threatId}`;
            const edgeId2 = `${threatId}_${impactId}`;

            if (!kg.edges.has(edgeId1)) {
                kg.edges.set(edgeId1, {
                    id: edgeId1,
                    source: speciesId,
                    target: threatId,
                    type: 'species_threat'
                });
                kg.nodes.get(speciesId).connections.add(threatId);
                kg.nodes.get(threatId).connections.add(speciesId);
                kg.nodes.get(speciesId).threatCount++;
                kg.nodes.get(threatId).speciesCount++;
            }

            if (!kg.edges.has(edgeId2)) {
                kg.edges.set(edgeId2, {
                    id: edgeId2,
                    source: threatId,
                    target: impactId,
                    type: 'threat_impact'
                });
                kg.nodes.get(threatId).connections.add(impactId);
                kg.nodes.get(impactId).connections.add(threatId);
                kg.nodes.get(impactId).threatCount++;
            }
        });

        console.log(`Created graph: ${kg.nodes.size} nodes, ${kg.edges.size} edges`);
        return kg;
    }

    function calculateCentralityMetrics(kg) {
        const metrics = {
            degreeCentrality: new Map(),
            betweennessCentrality: new Map(),
            closenessCentrality: new Map(),
            eigenvectorCentrality: new Map()
        };
        
        kg.nodes.forEach((node, nodeId) => {
            const degree = node.connections ? node.connections.size : 0;
            metrics.degreeCentrality.set(nodeId, degree);
        });
        
        kg.nodes.forEach((node, nodeId) => {
            const degree = node.connections ? node.connections.size : 0;
            metrics.betweennessCentrality.set(nodeId, degree * 0.1);
        });
        
        kg.nodes.forEach((node, nodeId) => {
            const degree = node.connections ? node.connections.size : 0;
            const maxDegree = Math.max(...Array.from(kg.nodes.values()).map(n => n.connections ? n.connections.size : 0));
            const normalizedCloseness = maxDegree > 0 ? degree / maxDegree : 0;
            metrics.closenessCentrality.set(nodeId, normalizedCloseness);
        });
        
        console.log('Centrality metrics calculated.');
        return metrics;
    }

    function detectCommunities(kg) {
        const communities = [];
        const visited = new Set();
        
        kg.nodes.forEach((node, nodeId) => {
            if (!visited.has(nodeId)) {
                const community = exploreConnectedComponent(kg, nodeId, visited);
                if (community.length > 1) {
                    communities.push({
                        id: `community_${communities.length}`,
                        nodes: community,
                        size: community.length,
                        density: calculateCommunityDensity(kg, community)
                    });
                }
            }
        });
        
        return communities.sort((a, b) => b.size - a.size);
    }

    function exploreConnectedComponent(kg, startNode, visited) {
        const component = [];
        const stack = [startNode];
        
        while (stack.length > 0) {
            const currentNode = stack.pop();
            
            if (!visited.has(currentNode)) {
                visited.add(currentNode);
                component.push(currentNode);
                
                const node = kg.nodes.get(currentNode);
                if (node && node.connections) {
                    node.connections.forEach(neighbor => {
                        if (!visited.has(neighbor)) {
                            stack.push(neighbor);
                        }
                    });
                }
            }
        }
        
        return component;
    }

    function calculateCommunityDensity(kg, community) {
        if (community.length < 2) return 0;
        
        let internalEdges = 0;
        const maxPossibleEdges = community.length * (community.length - 1) / 2;
        
        for (let i = 0; i < community.length; i++) {
            for (let j = i + 1; j < community.length; j++) {
                const node1 = kg.nodes.get(community[i]);
                const node2 = kg.nodes.get(community[j]);
                
                if (node1?.connections?.has(community[j]) || node2?.connections?.has(community[i])) {
                    internalEdges++;
                }
            }
        }
        
        return maxPossibleEdges > 0 ? internalEdges / maxPossibleEdges : 0;
    }

    function calculateVulnerabilityScores(kg) {
        const vulnerabilityScores = new Map();
        
        kg.nodes.forEach((node, nodeId) => {
            if (node.type === 'species') {
                let score = 0;
                
                const maxThreats = Math.max(...Array.from(kg.nodes.values())
                    .filter(n => n.type === 'species')
                    .map(n => n.threatCount || 0));
                score += ((node.threatCount || 0) / (maxThreats || 1)) * 0.4;
                
                const maxConnections = Math.max(...Array.from(kg.nodes.values())
                    .map(n => n.connections ? n.connections.size : 0));
                const connectivity = node.connections ? node.connections.size : 0;
                score += (connectivity / (maxConnections || 1)) * 0.3;
                
                const uniqueImpacts = new Set();
                if (node.connections) {
                    node.connections.forEach(connectedNodeId => {
                        const connectedNode = kg.nodes.get(connectedNodeId);
                        if (connectedNode?.type === 'impact') {
                            uniqueImpacts.add(connectedNodeId);
                        }
                    });
                }
                score += Math.min(uniqueImpacts.size / 10, 1) * 0.3;
                
                vulnerabilityScores.set(nodeId, Math.min(score, 1));
            }
        });
        
        return vulnerabilityScores;
    }

    function analyzeConnectivity(kg) {
        const connectivity = {
            totalNodes: kg.nodes.size,
            totalEdges: kg.edges ? (kg.edges.size || kg.edges.length || 0) : 0,
            avgDegree: 0,
            density: 0,
            components: 0,
            largestComponent: 0,
            isolatedNodes: 0
        };
        
        if (connectivity.totalNodes > 0) {
            let totalDegree = 0;
            kg.nodes.forEach(node => {
                totalDegree += node.connections ? node.connections.size : 0;
            });
            connectivity.avgDegree = totalDegree / connectivity.totalNodes;
            
            const maxPossibleEdges = connectivity.totalNodes * (connectivity.totalNodes - 1) / 2;
            connectivity.density = maxPossibleEdges > 0 ? connectivity.totalEdges / maxPossibleEdges : 0;
            
            const visited = new Set();
            const componentSizes = [];
            
            kg.nodes.forEach((node, nodeId) => {
                if (!visited.has(nodeId)) {
                    const component = exploreConnectedComponent(kg, nodeId, visited);
                    componentSizes.push(component.length);
                }
            });
            
            connectivity.components = componentSizes.length;
            connectivity.largestComponent = Math.max(...componentSizes, 0);
            connectivity.isolatedNodes = componentSizes.filter(size => size === 1).length;
        }
        
        return connectivity;
    }

    async function performCriticalNodesAnalysis() {
        try {
            console.log('Performing critical nodes analysis...');
            const kg = window.AppState.knowledgeGraph;
            if (!kg || kg.nodes.size === 0) {
                throw new Error('Knowledge graph not initialized');
            }
            
            const criticalNodes = identifyCriticalNodes(kg);
            
            window.AppState.analysis.criticalNodes = criticalNodes;
            
            displayCriticalNodesResults(criticalNodes);
            
            console.log('Critical nodes analysis completed');
            return criticalNodes;
            
        } catch (error) {
            console.error('Error performing critical nodes analysis:', error);
            displayAnalysisError('Critical Nodes Analysis', error.message);
            throw error;
        }
    }

    function identifyCriticalNodes(kg) {
        const criticalNodes = {
            keySpecies: [],
            hubNodes: [],
            bridgeNodes: [],
            vulnerableNodes: []
        };
        
        const nodeMetrics = new Map();
        
        kg.nodes.forEach((node, nodeId) => {
            const metrics = {
                id: nodeId,
                label: node.label,
                type: node.type,
                degree: node.connections ? node.connections.size : 0,
                betweenness: 0,
                clustering: 0,
                vulnerability: 0
            };
            
            if (node.connections && node.connections.size > 1) {
                const neighbors = Array.from(node.connections);
                let triangles = 0;
                const possibleTriangles = neighbors.length * (neighbors.length - 1) / 2;
                
                for (let i = 0; i < neighbors.length; i++) {
                    for (let j = i + 1; j < neighbors.length; j++) {
                        const neighbor1 = kg.nodes.get(neighbors[i]);
                        if (neighbor1?.connections?.has(neighbors[j])) {
                            triangles++;
                        }
                    }
                }
                
                metrics.clustering = possibleTriangles > 0 ? triangles / possibleTriangles : 0;
            }
            
            nodeMetrics.set(nodeId, metrics);
        });
        
        const sortedByDegree = Array.from(nodeMetrics.values()).sort((a, b) => b.degree - a.degree);
        const sortedByClustering = Array.from(nodeMetrics.values()).sort((a, b) => a.clustering - b.clustering);
        
        const maxDegree = sortedByDegree.length > 0 ? sortedByDegree[0].degree : 1;
        
        criticalNodes.keySpecies = sortedByDegree
            .filter(node => node.type === 'species')
            .slice(0, 10)
            .map(node => ({
                ...node,
                criticalityScore: node.degree / maxDegree
            }));
        
        criticalNodes.hubNodes = sortedByDegree
            .slice(0, 15)
            .map(node => ({
                ...node,
                hubScore: node.degree / maxDegree
            }));
        
        criticalNodes.bridgeNodes = sortedByClustering
            .filter(node => node.degree > 2 && node.clustering < 0.3)
            .slice(0, 10)
            .map(node => ({
                ...node,
                bridgeScore: (1 - node.clustering) * (node.degree / maxDegree)
            }));
        
        return criticalNodes;
    }

    async function identifyVulnerabilityCorridors() {
        try {
            console.log('Identifying vulnerability corridors...');
            const kg = window.AppState.knowledgeGraph;
            if (!kg || kg.nodes.size === 0) {
                throw new Error('Knowledge graph not initialized');
            }
            
            const corridors = findVulnerabilityCorridors(kg);
            
            window.AppState.analysis.vulnerabilityCorridors = corridors;
            
            displayVulnerabilityCorridorsResults(corridors);
            
            console.log('Vulnerability corridors identified');
            return corridors;
            
        } catch (error) {
            console.error('Error identifying vulnerability corridors:', error);
            displayAnalysisError('Vulnerability Corridors', error.message);
            throw error;
        }
    }

    function findVulnerabilityCorridors(kg) {
        const corridors = [];
        
        const threatenedSpecies = Array.from(kg.nodes.values())
            .filter(node => node.type === 'species' && (node.threatCount || 0) > 3)
            .sort((a, b) => (b.threatCount || 0) - (a.threatCount || 0))
            .slice(0, 20);
        
        for (let i = 0; i < threatenedSpecies.length; i++) {
            for (let j = i + 1; j < threatenedSpecies.length; j++) {
                const paths = findShortestPaths(kg, threatenedSpecies[i].id, threatenedSpecies[j].id);
                
                if (paths.length > 0 && paths[0].length <= 5) {
                    const corridor = {
                        id: `corridor_${corridors.length}`,
                        species1: threatenedSpecies[i].label,
                        species2: threatenedSpecies[j].label,
                        path: paths[0],
                        length: paths[0].length,
                        sharedThreats: findSharedThreats(kg, threatenedSpecies[i].id, threatenedSpecies[j].id),
                        riskLevel: calculateCorridorRisk(kg, paths[0])
                    };
                    
                    corridors.push(corridor);
                }
            }
        }
        
        return corridors.sort((a, b) => b.riskLevel - a.riskLevel).slice(0, 10);
    }

    function findSharedThreats(kg, species1Id, species2Id) {
        const species1 = kg.nodes.get(species1Id);
        const species2 = kg.nodes.get(species2Id);
        
        if (!species1?.connections || !species2?.connections) return [];
        
        const sharedThreats = [];
        
        species1.connections.forEach(connection => {
            const connectedNode = kg.nodes.get(connection);
            if (connectedNode?.type === 'threat' && species2.connections.has(connection)) {
                sharedThreats.push(connectedNode.label);
            }
        });
        
        return sharedThreats;
    }

    function calculateCorridorRisk(kg, path) {
        let riskScore = 0;
        
        path.forEach(nodeId => {
            const node = kg.nodes.get(nodeId);
            if (node) {
                if (node.type === 'species') {
                    riskScore += (node.threatCount || 0) * 0.3;
                } else if (node.type === 'threat') {
                    riskScore += (node.speciesCount || 0) * 0.2;
                }
                
                riskScore += (node.connections ? node.connections.size : 0) * 0.1;
            }
        });
        
        return riskScore / path.length;
    }

    async function performIndirectImpactsAnalysis() {
        try {
            console.log('Performing indirect impacts analysis...');
            const kg = window.AppState.knowledgeGraph;
            if (!kg || kg.nodes.size === 0) {
                throw new Error('Knowledge graph not initialized');
            }
            
            const indirectImpacts = analyzeIndirectImpacts(kg);
            
            window.AppState.analysis.indirectImpacts = indirectImpacts;
            
            displayIndirectImpactsResults(indirectImpacts);
            
            console.log('Indirect impacts analysis completed');
            return indirectImpacts;
            
        } catch (error) {
            console.error('Error performing indirect impacts analysis:', error);
            displayAnalysisError('Indirect Impacts Analysis', error.message);
            throw error;
        }
    }

    function analyzeIndirectImpacts(kg) {
        const indirectImpacts = {
            cascadeChains: [],
            secondOrderEffects: [],
            systemicRisks: []
        };
        
        kg.nodes.forEach((node, nodeId) => {
            if (node.type === 'threat') {
                const cascades = findCascadeChains(kg, nodeId);
                indirectImpacts.cascadeChains.push(...cascades);
            }
        });
        
        indirectImpacts.cascadeChains.sort((a, b) => {
            const scoreA = a.length * a.affectedSpeciesCount;
            const scoreB = b.length * b.affectedSpeciesCount;
            return scoreB - scoreA;
        });
        
        indirectImpacts.cascadeChains = indirectImpacts.cascadeChains.slice(0, 20);
        
        return indirectImpacts;
    }

    function findCascadeChains(kg, threatNodeId) {
        const chains = [];
        const visited = new Set();
        
        function exploreCascade(currentPath, currentNode, depth) {
            if (depth > 5 || visited.has(currentNode)) return;
            
            visited.add(currentNode);
            const node = kg.nodes.get(currentNode);
            
            if (node?.connections) {
                node.connections.forEach(connectedNodeId => {
                    const connectedNode = kg.nodes.get(connectedNodeId);
                    if (!currentPath.includes(connectedNodeId)) {
                        const newPath = [...currentPath, connectedNodeId];
                        
                        if (connectedNode?.type === 'impact' && newPath.length > 2) {
                            const speciesInChain = newPath.filter(nodeId => {
                                const n = kg.nodes.get(nodeId);
                                return n?.type === 'species';
                            });
                            
                            chains.push({
                                id: `cascade_${chains.length}`,
                                path: newPath,
                                length: newPath.length,
                                startThreat: kg.nodes.get(threatNodeId)?.label,
                                endImpact: connectedNode.label,
                                affectedSpeciesCount: speciesInChain.length,
                                riskScore: calculateCascadeRisk(kg, newPath)
                            });
                        } else {
                            exploreCascade(newPath, connectedNodeId, depth + 1);
                        }
                    }
                });
            }
            
            visited.delete(currentNode);
        }
        
        exploreCascade([threatNodeId], threatNodeId, 0);
        return chains;
    }

    function calculateCascadeRisk(kg, path) {
        let riskScore = 0;
        
        path.forEach(nodeId => {
            const node = kg.nodes.get(nodeId);
            if (node) {
                const connectivity = node.connections ? node.connections.size : 0;
                riskScore += connectivity;
                
                if (node.type === 'species') {
                    riskScore += (node.threatCount || 0) * 2;
                }
            }
        });
        
        return riskScore / path.length;
    }

    function displayNetworkAnalysisResults(results) {
        const container = window.AppUtils.getElement('networkAnalysisResults');
        if (!container) return;
        const conn = results.connectivityAnalysis;
        container.innerHTML = `
            <div class="analysis-results">
                <h4>Network Analysis Results</h4>
                <div class="analysis-section">
                    <h5>Network Connectivity</h5>
                    <p>Nodes: ${conn.totalNodes}, Edges: ${conn.totalEdges}</p>
                    <p>Avg Degree: ${conn.avgDegree.toFixed(2)}, Density: ${conn.density.toFixed(4)}</p>
                    <p>Components: ${conn.components}</p>
                </div>
                <div class="analysis-section">
                    <h5>Community Structure</h5>
                    <p>Communities Found: ${results.communityStructure.length}</p>
                    ${results.communityStructure.length > 0 ? `<ul>
                        ${results.communityStructure.slice(0, 5).map((c, i) => `<li>Community ${i + 1}: ${c.size} nodes (density: ${c.density.toFixed(3)})</li>`).join('')}
                    </ul>` : ''}
                </div>
            </div>`;

        console.log('Creating network visualization...');
        if (typeof window.createEcologicalNetworkVisualization === 'function') {
            try {
                window.createEcologicalNetworkVisualization();
            } catch (error) {
                console.error('Error creating network visualization:', error);
            }
        } else {
            console.warn('createEcologicalNetworkVisualization function not available');
        }
    }

    function displayCriticalNodesResults(results) {
        const container = window.AppUtils.getElement('criticalNodesResults');
        if (!container) return;
        
        container.innerHTML = `
            <div class="analysis-results">
                <h4>Critical Nodes Analysis</h4>
                <div class="analysis-section">
                    <h5>Key Species</h5>
                    ${results.keySpecies.length > 0 ? `<ul>
                        ${results.keySpecies.slice(0, 5).map(s => `<li><b>${s.label}</b> - Degree: ${s.degree}, Criticality: ${s.criticalityScore.toFixed(3)}</li>`).join('')}
                    </ul>` : '<p>None found.</p>'}
                </div>
                <div class="analysis-section">
                    <h5>Network Hubs</h5>
                    ${results.hubNodes.length > 0 ? `<ul>
                        ${results.hubNodes.slice(0, 5).map(h => `<li><b>${h.label}</b> (${h.type}) - Degree: ${h.degree}, Hub Score: ${h.hubScore.toFixed(3)}</li>`).join('')}
                    </ul>` : '<p>None found.</p>'}
                </div>
            </div>`;
    }

    function displayVulnerabilityCorridorsResults(results) {
        const container = window.AppUtils.getElement('vulnerabilityCorridorsResults');
        if (!container) return;
        
        container.innerHTML = `
            <div class="analysis-results">
                <h4>Vulnerability Corridors</h4>
                ${results.length > 0 ? `<div class="corridors-list">
                    ${results.slice(0, 5).map(corridor => `
                        <div class="corridor-item">
                            <h6>${corridor.species1} ↔ ${corridor.species2}</h6>
                            <p><strong>Path Length:</strong> ${corridor.length}, <strong>Risk Level:</strong> ${corridor.riskLevel.toFixed(3)}</p>
                            ${corridor.sharedThreats.length > 0 ? `<p><strong>Shared Threats:</strong> ${corridor.sharedThreats.slice(0, 3).join(', ')}</p>` : ''}
                        </div>
                    `).join('')}
                </div>` : '<p>No significant vulnerability corridors found.</p>'}
            </div>`;
    }

    function displayIndirectImpactsResults(results) {
        const container = window.AppUtils.getElement('indirectImpactsResults');
        if (!container) return;
        
        container.innerHTML = `
            <div class="analysis-results">
                <h4>Indirect Impacts Analysis</h4>
                <div class="analysis-section">
                    <h5>Cascade Chains</h5>
                    ${results.cascadeChains.length > 0 ? `<div class="cascade-list">
                        ${results.cascadeChains.slice(0, 5).map(c => `
                            <div class="cascade-item">
                                <h6>${c.startThreat} → ${c.endImpact}</h6>
                                <p>Length: ${c.length}, Affected Species: ${c.affectedSpeciesCount}, Risk: ${c.riskScore.toFixed(3)}</p>
                            </div>
                        `).join('')}
                    </div>` : '<p>No significant cascade chains identified.</p>'}
                </div>
            </div>`;
    }

    function displayAnalysisError(analysisType, errorMessage) {
        const container = window.AppUtils.getElement('analysisErrorContainer') || 
                         window.AppUtils.getElement('networkAnalysisResults');
        
        if (container) {
            container.innerHTML = `
                <div class="analysis-error">
                    <h4>${analysisType} Error</h4>
                    <p>${errorMessage}</p>
                </div>
            `;
        }
    }

    window.performNetworkAnalysis = performNetworkAnalysis;
    window.performCriticalNodesAnalysis = performCriticalNodesAnalysis;
    window.identifyVulnerabilityCorridors = identifyVulnerabilityCorridors;
    window.performIndirectImpactsAnalysis = performIndirectImpactsAnalysis;

    if (window.AppFunctions) {
        Object.assign(window.AppFunctions, {
            performNetworkAnalysis,
            performCriticalNodesAnalysis,
            identifyVulnerabilityCorridors,
            performIndirectImpactsAnalysis,
            calculateCentralityMetrics,
            detectCommunities,
            calculateVulnerabilityScores,
            findVulnerabilityCorridors,
            analyzeIndirectImpacts
        });
    }
    
    console.log('Analysis module loaded.');

})();

