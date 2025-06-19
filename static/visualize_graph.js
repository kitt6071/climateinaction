function displayKnowledgeGraphResults(results, queryType) {
    const container = document.getElementById('queryResults');
    if (!container) return;
    
    container.innerHTML = `<h4>Knowledge Graph Query Results: ${queryType}</h4>`;
    
    if (!results || results.length === 0) {
        container.innerHTML += '<p>No results found for this query.</p>';
        return;
    }
    
    createQueryVisualization(results, queryType);
    
    let html = '<div class="kg-query-results">';
    
    switch (queryType) {
        case 'interacting_species_same_threat_category':
            html += '<h5>Species Pairs Affected by Same Threat Categories:</h5>';
            results.forEach(result => {
                html += `
                    <div class="kg-result-item">
                        <strong>${result.species1} ↔ ${result.species2}</strong>
                        <br><span class="threat-category">Shared Threat Category: ${result.shared_threat_category}</span>
                        <br><span class="interaction-type">Interaction Type: ${result.interaction_type}</span>
                        <br><span class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                `;
            });
            break;
            
        case 'species_similar_threats_with_interactions':
            html += '<h5>Species with Similar Threats to Focal Species:</h5>';
            results.forEach(result => {
                html += `
                    <div class="kg-result-item">
                        <strong>${result.focal_species} → ${result.other_species}</strong>
                        <br><span class="similarity">Max Similarity: ${(result.max_similarity * 100).toFixed(1)}%</span>
                        <br><span class="shared-threats">Shared Threats: ${result.shared_threats.length}</span>
                        <div class="threat-details">
                `;
                result.shared_threats.slice(0, 3).forEach(threat => {
                    html += `<div class="threat-pair">• ${threat.focal_threat} ↔ ${threat.similar_threat} (${(threat.similarity * 100).toFixed(1)}%)</div>`;
                });
                html += '</div></div>';
            });
            break;
            
        case 'threat_impact_chains':
            html += '<h5>Multi-hop Threat-Impact Chains:</h5>';
            results.forEach(result => {
                html += `
                    <div class="kg-result-item">
                        <strong>Chain: ${result.chain_start} → ${result.chain_end}</strong>
                        <br><span class="chain-length">Length: ${result.chain_length} steps</span>
                        <br><span class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</span>
                        <div class="chain-path">
                            Path: ${result.path.map(step => `${step.node_name} (${step.node_type})`).join(' → ')}
                        </div>
                    </div>
                `;
            });
            break;
            
        case 'vulnerability_pathways':
            html += '<h5>Vulnerability Pathway Analysis:</h5>';
            results.forEach(result => {
                html += `
                    <div class="kg-result-item">
                        <strong>${result.species}</strong>
                        <br><span class="vulnerability-score">Vulnerability Score: ${result.vulnerability_score}</span>
                        <br><span class="connected-count">Connected to ${result.connected_species_count} other species</span>
                        <br><span class="connected-species">Connected Species: ${result.connected_species.join(', ')}</span>
                        <br><span class="pathway-type">Pathway Type: ${result.pathway_type}</span>
                    </div>
                `;
            });
            break;
    }
    
    html += '</div>';
    container.innerHTML += html;
}

function displayNetworkSummary() {
    const speciesConnections = new Map();
    
    window.knowledgeGraph.edges.forEach(edge => {
        if (edge.type === 'HAS_THREAT') {
            const speciesNode = window.knowledgeGraph.nodes.get(edge.source);
            if (speciesNode) {
                const count = speciesConnections.get(speciesNode.name) || 0;
                speciesConnections.set(speciesNode.name, count + 1);
            }
        }
    });
    
    const topSpecies = Array.from(speciesConnections.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);
    
    const containers = ['queryResults', 'queryResultsTable', 'criticalNodesTable', 'vulnerabilityCorridors', 'indirectImpactResults'];
    let container = null;
    
    for (const containerId of containers) {
        container = document.getElementById(containerId); 
        if (container) break;
    }
    
    if (!container) return;
    
    let html = '<h4>Network Analysis Results</h4>';
    html += '<div class="network-summary">';
    html += '<h5>Most Connected Species (by threat count):</h5>';
    html += '<ul>';
    topSpecies.forEach(([species, count]) => {
        html += `<li><strong>${species}</strong>: ${count} documented threats</li>`;
    });
    html += '</ul>';
    
    html += `<p><strong>Network Summary:</strong></p>`;
    html += `<ul>`;
    html += `<li>Total Species: ${window.knowledgeGraph.speciesNodes.size}</li>`;
    html += `<li>Total Threats: ${window.knowledgeGraph.threatNodes.size}</li>`;
    html += `<li>Total Impacts: ${window.knowledgeGraph.impactNodes.size}</li>`;
    html += `<li>Total Connections: ${window.knowledgeGraph.edges.size}</li>`;
    html += `</ul>`;
    html += '</div>';
    
    container.innerHTML = html;
    container.style.display = 'block';
    if (container.parentElement) {
        container.parentElement.style.display = 'block';
    }
}

function createQueryVisualization(results, queryType) {
    const container = document.getElementById('networkCanvas');
    if (!container) return;
    
    container.innerHTML = '';
    
    const width = container.offsetWidth || 800;
    const height = 400;
    
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('border', '1px solid #ddd');
    
    const g = svg.append('g');
    
    const zoom = d3.zoom()
        .scaleExtent([0.5, 3])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    
    svg.call(zoom);
    
    switch (queryType) {
        case 'interacting_species_same_threat_category':
            createSpeciesPairsVisualization(g, results, width, height);
            break;
        case 'species_similar_threats_with_interactions':
            createSimilarThreatsVisualization(g, results, width, height);
            break;
        case 'threat_impact_chains':
            createThreatChainsVisualization(g, results, width, height);
            break;
        case 'vulnerability_pathways':
            createVulnerabilityVisualization(g, results, width, height);
            break;
        default:
            createEcologicalNetworkVisualization();
            return;
    }
    
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .style('font-weight', 'bold')
        .text(`${queryType.replace(/_/g, ' ').toUpperCase()} Visualization`);
}

function createSpeciesPairsVisualization(g, results, width, height) {
    const nodes = [];
    const links = [];
    const speciesMap = new Map();
    
    results.forEach((result, index) => {
        if (!speciesMap.has(result.species1)) {
            speciesMap.set(result.species1, {
                id: result.species1,
                name: result.species1,
                type: 'species',
                connections: 0
            });
        }
        if (!speciesMap.has(result.species2)) {
            speciesMap.set(result.species2, {
                id: result.species2,
                name: result.species2,
                type: 'species',
                connections: 0
            });
        }
        
        links.push({
            source: result.species1,
            target: result.species2,
            threat_category: result.shared_threat_category,
            confidence: result.confidence
        });
        
        speciesMap.get(result.species1).connections++;
        speciesMap.get(result.species2).connections++;
    });
    
    nodes.push(...Array.from(speciesMap.values()));
    
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(80))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, height / 2));
    
    const link = g.append('g')
        .selectAll('line')
        .data(links)
        .enter().append('line')
        .attr('stroke', '#e74c3c')
        .attr('stroke-width', d => 2 + d.confidence * 3)
        .attr('stroke-opacity', 0.7);
    
    const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('r', d => 10 + Math.sqrt(d.connections) * 2)
        .attr('fill', '#3498db')
        .attr('stroke', '#2c3e50')
        .attr('stroke-width', 2)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    const label = g.append('g')
        .selectAll('text')
        .data(nodes)
        .enter().append('text')
        .text(d => d.name.length > 15 ? d.name.substring(0, 12) + '...' : d.name)
        .attr('font-size', '10px')
        .attr('dx', 15)
        .attr('dy', 4)
        .style('pointer-events', 'none');
    
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
    
    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }
    
    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }
    
    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }
    
    node.append('title')
        .text(d => `${d.name}\nConnections: ${d.connections}`);
    
    link.append('title')
        .text(d => `Shared threat: ${d.threat_category}\nConfidence: ${(d.confidence * 100).toFixed(1)}%`);
}

function createVulnerabilityVisualization(g, results, width, height) {
    const nodes = results.map(result => ({
        id: result.species || result.species1 || 'Unknown',
        name: result.species || result.species1 || 'Unknown',
        vulnerability_score: result.vulnerability_score || result.confidence || 0.5,
        connected_count: result.connected_species_count || result.species_count || 1,
        connected_species: result.connected_species || []
    }));
    
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) / 3;
    
    const maxVulnerability = Math.max(...nodes.map(n => n.vulnerability_score || 0));
    const safeMaxVulnerability = maxVulnerability > 0 ? maxVulnerability : 1;
    
    nodes.forEach((node, index) => {
        const angle = (index / nodes.length) * 2 * Math.PI;
        const normalizedScore = (node.vulnerability_score || 0) / safeMaxVulnerability;
        const radius = normalizedScore * maxRadius * 0.8 + maxRadius * 0.2;
        
        node.x = centerX + Math.cos(angle) * radius;
        node.y = centerY + Math.sin(angle) * radius;
        
        // Ensure coordinates are valid numbers
        if (isNaN(node.x) || isNaN(node.y)) {
            node.x = centerX + Math.cos(angle) * maxRadius * 0.5;
            node.y = centerY + Math.sin(angle) * maxRadius * 0.5;
        }
    });
    
    const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('cx', d => isNaN(d.x) ? centerX : d.x)
        .attr('cy', d => isNaN(d.y) ? centerY : d.y)
        .attr('r', d => {
            const radius = 5 + Math.sqrt(d.connected_count || 1) * 2;
            return isNaN(radius) ? 7 : radius;
        })
        .attr('fill', d => {
            const intensity = (d.vulnerability_score || 0) / safeMaxVulnerability;
            const safeIntensity = isNaN(intensity) ? 0.5 : Math.min(Math.max(intensity, 0), 1);
            return d3.interpolateReds(0.3 + safeIntensity * 0.7);
        })
        .attr('stroke', '#000')
        .attr('stroke-width', 1);
    
    const label = g.append('g')
        .selectAll('text')
        .data(nodes)
        .enter().append('text')
        .attr('x', d => isNaN(d.x) ? centerX : d.x)
        .attr('y', d => isNaN(d.y) ? centerY - 15 : d.y - 15)
        .attr('text-anchor', 'middle')
        .text(d => {
            const name = d.name || 'Unknown';
            return name.length > 12 ? name.substring(0, 9) + '...' : name;
        })
        .style('font-size', '9px')
        .style('pointer-events', 'none');
    
    const scoreLabel = g.append('g')
        .selectAll('text')
        .data(nodes)
        .enter().append('text')
        .attr('x', d => isNaN(d.x) ? centerX : d.x)
        .attr('y', d => isNaN(d.y) ? centerY + 4 : d.y + 4)
        .attr('text-anchor', 'middle')
        .text(d => {
            const score = d.vulnerability_score || 0;
            return isNaN(score) ? '0' : score.toFixed(1);
        })
        .style('font-size', '8px')
        .style('fill', '#fff')
        .style('font-weight', 'bold')
        .style('pointer-events', 'none');
    
    const levels = [0.25, 0.5, 0.75, 1.0];
    levels.forEach(level => {
        g.append('circle')
            .attr('cx', centerX)
            .attr('cy', centerY)
            .attr('r', maxRadius * level)
            .attr('fill', 'none')
            .attr('stroke', '#ddd')
            .attr('stroke-dasharray', '2,2')
            .attr('stroke-opacity', 0.5);
    });
    
    node.append('title')
        .text(d => `${d.name}\nVulnerability Score: ${d.vulnerability_score}\nConnected Species: ${d.connected_count}\nConnections: ${d.connected_species.join(', ')}`);
}

function createSystemicRiskCharts() {
    createVulnerabilityDistributionChart();
    createThreatPropagationChart();
}


function createVulnerabilityDistributionChart() {
    const ctx = document.getElementById('vulnerabilityDistributionChart');
    if (!ctx || typeof Chart === 'undefined') {
        console.log('Chart context or Chart.js not available');
        return;
    }
    
    const knowledgeGraph = window.knowledgeGraph || window.AppState?.knowledgeGraph;
    if (!knowledgeGraph || !knowledgeGraph.speciesNodes || !knowledgeGraph.edges) {
        console.log('Knowledge graph not available for vulnerability chart');
        return;
    }
    
    const vulnerabilityBins = [0, 0, 0, 0, 0];
    const binLabels = ['0-2', '3-5', '6-10', '11-20', '20+'];
    
    knowledgeGraph.speciesNodes.forEach(speciesNode => {
        let threatCount = 0;
        knowledgeGraph.edges.forEach(edge => {
            if (edge.type === 'EXPERIENCES_IMPACT' && edge.source === speciesNode.id) {
                threatCount++;
            }
        });
        
        if (threatCount <= 2) vulnerabilityBins[0]++;
        else if (threatCount <= 5) vulnerabilityBins[1]++;
        else if (threatCount <= 10) vulnerabilityBins[2]++;
        else if (threatCount <= 20) vulnerabilityBins[3]++;
        else vulnerabilityBins[4]++;
    });
    
    if (window.vulnerabilityDistributionChartInstance) {
        window.vulnerabilityDistributionChartInstance.destroy();
    }
    
    window.vulnerabilityDistributionChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: 'Number of Species',
                data: vulnerabilityBins,
                backgroundColor: [
                    '#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad'
                ],
                borderColor: '#2c3e50',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Species'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Number of Threats'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Species Vulnerability Distribution'
                }
            }
        }
    });
}

function createThreatPropagationChart() {
    const ctx = document.getElementById('threatPropagationChart');
    if (!ctx || typeof Chart === 'undefined') {
        console.log('Chart context or Chart.js not available');
        return;
    }
    
    const knowledgeGraph = window.knowledgeGraph || window.AppState?.knowledgeGraph;
    if (!knowledgeGraph || !knowledgeGraph.nodes || !knowledgeGraph.edges) {
        console.log('Knowledge graph not available for threat propagation chart');
        return;
    }
    
    const impactCategories = new Map();
    
    knowledgeGraph.edges.forEach(edge => {
        if (edge.type === 'EXPERIENCES_IMPACT') {
            const impactNode = knowledgeGraph.nodes.get(edge.target);
            if (impactNode && impactNode.name) {
                const count = impactCategories.get(impactNode.name) || 0;
                impactCategories.set(impactNode.name, count + 1);
            }
        }
    });
    
    if (impactCategories.size === 0) {
        console.log('No impact categories found for chart');
        return;
    }
    
    const topImpacts = Array.from(impactCategories.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8);
    
    const labels = topImpacts.map(([impact, _]) => impact.length > 20 ? impact.substring(0, 17) + '...' : impact);
    const data = topImpacts.map(([_, count]) => count);
    
    if (window.threatPropagationChartInstance) {
        window.threatPropagationChartInstance.destroy();
    }
    
    window.threatPropagationChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#e74c3c', '#f39c12', '#f1c40f', '#2ecc71',
                    '#3498db', '#9b59b6', '#1abc9c', '#e67e22'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        boxWidth: 12,
                        padding: 10
                    }
                },
                title: {
                    display: true,
                    text: 'Impact Category Distribution'
                }
            }
        }
    });
}

window.networkVisState = {
    simulation: null,
    svg: null,
    zoom: null,
    isPhysicsOn: true
};

function createEcologicalNetworkVisualization() {
    d3.select('#networkCanvas').selectAll('*').remove();
    
    const knowledgeGraph = window.AppState?.knowledgeGraph;
    if (!knowledgeGraph || !knowledgeGraph.nodes || !knowledgeGraph.edges) {
        d3.select('#networkCanvas').append('div')
            .style('text-align', 'center')
            .style('padding', '50px')
            .html('<h3>Knowledge Graph Not Available</h3><p>Please load data first.</p>');
        return;
    }
    
    const nodeSizeMode = document.getElementById('nodeSize')?.value || 'degree';
    const edgeWeightMode = document.getElementById('edgeWeight')?.value || 'uniform';
    
    const container = document.getElementById('networkCanvas');
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;
    
    const svg = d3.select('#networkCanvas')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    
    svg.call(zoom);
    
    // Store state
    window.networkVisState.svg = svg;
    window.networkVisState.zoom = zoom;
    
    const g = svg.append('g');
    
    const nodes = [];
    const links = [];
    
    knowledgeGraph.nodes.forEach((nodeData, nodeId) => {
        nodes.push({
            id: nodeId,
            name: nodeData.name || nodeId,
            type: nodeData.type || 'unknown',
            properties: nodeData.properties || {}
        });
    });
    
    knowledgeGraph.edges.forEach((edgeData, edgeId) => {
        links.push({
            source: edgeData.source,
            target: edgeData.target,
            type: edgeData.type || 'unknown',
            properties: edgeData.properties || {}
        });
    });
    
    const nodeDegrees = new Map();
    links.forEach(link => {
        nodeDegrees.set(link.source, (nodeDegrees.get(link.source) || 0) + 1);
        nodeDegrees.set(link.target, (nodeDegrees.get(link.target) || 0) + 1);
    });
    
    if (nodes.length > 200) {
        const nodesByDegree = nodes.map(node => ({
            ...node,
            degree: nodeDegrees.get(node.id) || 0
        })).sort((a, b) => b.degree - a.degree);
        
        const topNodes = nodesByDegree.slice(0, 200);
        const topNodeIds = new Set(topNodes.map(n => n.id));
        
        nodes.length = 0;
        nodes.push(...topNodes);
        
        const filteredLinks = links.filter(l => 
            topNodeIds.has(l.source) && topNodeIds.has(l.target)
        );
        links.length = 0;
        links.push(...filteredLinks);
    }
    
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(60))
        .force('charge', d3.forceManyBody().strength(-150))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(20));
    
    window.networkVisState.simulation = simulation;
    
    const link = g.append('g')
        .selectAll('line')
        .data(links)
        .enter().append('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', 1.5);
    
    const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('r', d => {
            const degree = nodeDegrees.get(d.id) || 1;
            return nodeSizeMode === 'degree' ? 
                Math.max(5, Math.min(20, 5 + Math.sqrt(degree) * 2)) : 8;
        })
        .attr('fill', d => {
            const colors = {
                'Species': '#2E8B57',
                'Threat': '#CD853F',
                'Impact': '#9370DB'
            };
            return colors[d.type] || '#708090';
        })
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    const importantNodes = nodes
        .filter(d => (nodeDegrees.get(d.id) || 0) > 5)
        .slice(0, 20);
    
    const label = g.append('g')
        .selectAll('text')
        .data(importantNodes)
        .enter().append('text')
        .text(d => d.name.length > 15 ? d.name.substring(0, 15) + '...' : d.name)
        .attr('font-size', '10px')
        .attr('dx', 12)
        .attr('dy', 4);
    
    node.append('title')
        .text(d => `${d.name}\nType: ${d.type}\nConnections: ${nodeDegrees.get(d.id) || 0}`);
    
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
    
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
    
        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
    
    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }
    
    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }
    
    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }

    d3.select('#centerNetwork').on('click', centerNetwork);
    d3.select('#togglePhysics').on('click', togglePhysics);
    d3.select('#exportNetwork').on('click', exportNetwork);
    d3.select('#nodeSize').on('change', createEcologicalNetworkVisualization);
    d3.select('#edgeWeight').on('change', createEcologicalNetworkVisualization);
}

function centerNetwork() {
    const { svg, zoom } = window.networkVisState;
    if (svg && zoom) {
        svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
    }
}

function togglePhysics() {
    const { simulation } = window.networkVisState;
    if (simulation) {
        if (window.networkVisState.isPhysicsOn) {
            simulation.stop();
            d3.select('#togglePhysics').text('Resume Physics');
        } else {
            simulation.alpha(0.3).restart();
            d3.select('#togglePhysics').text('Pause Physics');
        }
        window.networkVisState.isPhysicsOn = !window.networkVisState.isPhysicsOn;
    }
}

function exportNetwork() {
    const { svg } = window.networkVisState;
    if (svg) {
        const svgData = svg.node().outerHTML;
        const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'ecological_network.svg';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

window.addEventListener('knowledgeGraphReady', function(event) {
    console.log('Knowledge graph ready, creating systemic risk charts and network visualization');
    setTimeout(() => {
        createSystemicRiskCharts();
        createEcologicalNetworkVisualization();
    }, 100);
});

window.displayKnowledgeGraphResults = displayKnowledgeGraphResults;
window.displayNetworkSummary = displayNetworkSummary;
window.createQueryVisualization = createQueryVisualization;
window.createEcologicalNetworkVisualization = createEcologicalNetworkVisualization;
window.createSystemicRiskCharts = createSystemicRiskCharts;
window.createVulnerabilityDistributionChart = createVulnerabilityDistributionChart;
window.createThreatPropagationChart = createThreatPropagationChart;