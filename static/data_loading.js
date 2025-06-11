(function() {
    'use strict';
    
    if (typeof window.AppState === 'undefined' || typeof window.AppUtils === 'undefined') {
        console.error('Data loading prerequisites not met.');
        return;
    }

    async function fetchTripletData() {
        try {
            console.log('Fetching triplets...');
            const response = await fetch('/api/triplets');
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            window.AppState.allTripletsData = data;
            window.AppState.allSpecies = [...new Set(data.map(triplet => triplet.subject))].sort();
            
            if (typeof Fuse !== 'undefined') {
                window.AppState.fuse = new Fuse(data, window.AppConfig.fuseOptions);
                window.AppState.speciesFuse = new Fuse(window.AppState.allSpecies, {
                    threshold: 0.3,
                    distance: 100
                });
            } else {
                console.warn('Fuse.js not available, search will be limited.');
            }
            
            initializeKnowledgeGraph(data);
            
            if (typeof window.displayTriplets === 'function') {
                window.displayTriplets(data);
            }
            
            updateDataStatistics(data);
            
            console.log(`Loaded ${data.length} triplets.`);
            
            return data;
            
        } catch (error) {
            console.error('Failed to fetch triplet data:', error);
            
            const tripletsContainer = window.AppUtils.getElement('tripletsContainer');
            if (tripletsContainer) {
                tripletsContainer.textContent = `Error loading data: ${error.message}`;
            }
            
            throw error;
        }
    }

    function initializeKnowledgeGraph(data = null) {
        const tripletData = data || window.AppState.allTripletsData;
        
        if (!tripletData || tripletData.length === 0) {
            console.warn('No data available for knowledge graph initialization.');
            return;
        }
        
        const kg = {
            nodes: new Map(),
            edges: [],
            species: new Set(),
            threats: new Set(),
            impacts: new Set(),
            semanticClusters: new Map()
        };
        window.AppState.knowledgeGraph = kg;
        
        tripletData.forEach((triplet, index) => {
            const { subject: species, predicate: threat, object: impact } = triplet;
            
            kg.species.add(species);
            kg.threats.add(threat);
            kg.impacts.add(impact);
            
            const speciesNodeId = `species_${species}`;
            const threatNodeId = `threat_${threat}`;
            const impactNodeId = `impact_${impact}`;
            
            if (!kg.nodes.has(speciesNodeId)) {
                kg.nodes.set(speciesNodeId, { id: speciesNodeId, label: species, type: 'species', threatCount: 0, connections: new Set() });
            }
            
            if (!kg.nodes.has(threatNodeId)) {
                kg.nodes.set(threatNodeId, { id: threatNodeId, label: threat, type: 'threat', speciesCount: 0, connections: new Set() });
            }
            
            if (!kg.nodes.has(impactNodeId)) {
                kg.nodes.set(impactNodeId, { id: impactNodeId, label: impact, type: 'impact', frequency: 0, connections: new Set() });
            }
            
            kg.edges.push(
                { id: `edge_${index}_st`, source: speciesNodeId, target: threatNodeId, type: 'affected_by', weight: 1, doi: triplet.doi },
                { id: `edge_${index}_ti`, source: threatNodeId, target: impactNodeId, type: 'causes', weight: 1, doi: triplet.doi }
            );
            
            const speciesNode = kg.nodes.get(speciesNodeId);
            speciesNode.threatCount++;
            speciesNode.connections.add(threatNodeId);
            
            const threatNode = kg.nodes.get(threatNodeId);
            threatNode.speciesCount++;
            threatNode.connections.add(speciesNodeId).add(impactNodeId);
            
            const impactNode = kg.nodes.get(impactNodeId);
            impactNode.frequency++;
            impactNode.connections.add(threatNodeId);
        });
        
        generateSemanticClusters();
        
        console.log(`Knowledge graph initialized: ${kg.nodes.size} nodes, ${kg.edges.length} edges.`);
        window.dispatchEvent(new CustomEvent('knowledgeGraphReady', { detail: { knowledgeGraph: kg } }));
    }

    function generateSemanticClusters() {
        const kg = window.AppState.knowledgeGraph;
        
        const threatClusters = clusterByCommonTerms(Array.from(kg.threats), 'threat');
        const impactClusters = clusterByCommonTerms(Array.from(kg.impacts), 'impact');
        
        kg.semanticClusters.set('threats', threatClusters);
        kg.semanticClusters.set('impacts', impactClusters);
    }

    function clusterByCommonTerms(items, type) {
        const clusters = [];
        const processed = new Set();
        
        items.forEach(item => {
            if (processed.has(item)) return;
            
            const cluster = {
                id: `${type}_cluster_${clusters.length}`,
                category: extractMainCategory(item),
                items: [item],
                type: type
            };
            
            items.forEach(otherItem => {
                if (item !== otherItem && !processed.has(otherItem) && areItemsSimilar(item, otherItem)) {
                    cluster.items.push(otherItem);
                    processed.add(otherItem);
                }
            });
            
            processed.add(item);
            clusters.push(cluster);
        });
        
        return clusters.sort((a, b) => b.items.length - a.items.length);
    }

    function extractMainCategory(description) {
        const commonCategories = {
            threat: ['habitat', 'pollution', 'climate', 'invasive', 'hunting', 'disease', 'human'],
            impact: ['population', 'mortality', 'reproduction', 'behavior', 'physiology', 'genetic']
        };
        
        const lowerDesc = description.toLowerCase();
        
        for (const keywords of Object.values(commonCategories)) {
            for (const keyword of keywords) {
                if (lowerDesc.includes(keyword)) {
                    return keyword.charAt(0).toUpperCase() + keyword.slice(1);
                }
            }
        }
        
        const words = description.split(' ').filter(word => word.length > 3);
        return words.length > 0 ? words[0] : 'Other';
    }

    function areItemsSimilar(item1, item2) {
        const getWords = (item) => new Set(item.toLowerCase().split(' ').filter(word => word.length > 3));
        const words1 = getWords(item1);
        const words2 = getWords(item2);
        
        const intersection = new Set([...words1].filter(x => words2.has(x)));
        const union = new Set([...words1, ...words2]);
        
        return union.size > 0 ? intersection.size / union.size > 0.3 : false;
    }

    async function loadSystemicRiskData() {
        try {
            console.log('Loading systemic risk data...');
            const kg = window.AppState.knowledgeGraph;
            
            if (!kg || kg.nodes.size === 0) {
                throw new Error('Knowledge graph must be initialized before loading risk data.');
            }
            
            const riskData = calculateSystemicRisks(kg);
            window.AppState.systemicRisk = riskData;
            
            if (typeof window.updateSystemicRiskDisplay === 'function') {
                window.updateSystemicRiskDisplay(riskData);
            }
            
            console.log('Systemic risk data loaded.');
            return riskData;
            
        } catch (error) {
            console.error('Failed to load systemic risk data:', error);
            throw error;
        }
    }

    function calculateSystemicRisks(kg) {
        const speciesNodes = Array.from(kg.nodes.values()).filter(node => node.type === 'species');
        const criticalSpecies = speciesNodes
            .sort((a, b) => b.threatCount - a.threatCount)
            .slice(0, 10)
            .map(node => ({
                species: node.label,
                threatCount: node.threatCount,
                connectivityScore: node.connections.size,
                riskLevel: node.threatCount > 5 ? 'High' : node.threatCount > 2 ? 'Medium' : 'Low'
            }));
        
        const networkMetrics = {
            totalNodes: kg.nodes.size,
            totalEdges: kg.edges.length,
            avgConnectivity: kg.nodes.size > 0 ? kg.edges.length / kg.nodes.size : 0,
            clusteringCoefficient: calculateClusteringCoefficient(kg),
            networkDensity: kg.nodes.size > 1 ? (2 * kg.edges.length) / (kg.nodes.size * (kg.nodes.size - 1)) : 0
        };

        return { criticalSpecies, vulnerabilityCorridors: [], cascadeRisks: [], networkMetrics };
    }

    function calculateClusteringCoefficient(kg) {
        let totalTriangles = 0;
        let totalTriplets = 0;
        
        kg.nodes.forEach(node => {
            const neighbors = Array.from(node.connections);
            if (neighbors.length < 2) return;
            
            totalTriplets += neighbors.length * (neighbors.length - 1) / 2;
            
            for (let i = 0; i < neighbors.length; i++) {
                for (let j = i + 1; j < neighbors.length; j++) {
                    if (kg.nodes.get(neighbors[i])?.connections.has(neighbors[j])) {
                        totalTriangles++;
                    }
                }
            }
        });
        
        return totalTriplets > 0 ? totalTriangles / totalTriplets : 0;
    }

    function updateDataStatistics(data) {
        const stats = {
            totalTriplets: data.length,
            uniqueSpecies: window.AppState.allSpecies.length,
            uniqueThreats: new Set(data.map(t => t.predicate)).size,
            uniqueImpacts: new Set(data.map(t => t.object)).size,
            uniqueDOIs: new Set(data.map(t => t.doi)).size
        };
        
        Object.entries(stats).forEach(([key, value]) => {
            const element = window.AppUtils.getElement(`${key}Count`);
            if (element) {
                element.textContent = value.toLocaleString();
            }
        });
    }

    function initializeKnowledgeTransfer() {
        const kg = window.AppState.knowledgeGraph;
        if (kg && kg.nodes.size > 0) {
            console.log('Knowledge transfer module ready.');
            window.dispatchEvent(new CustomEvent('knowledgeTransferReady', { detail: { knowledgeGraph: kg } }));
        }
    }

    const dataFunctions = {
        fetchTripletData,
        initializeKnowledgeGraph,
        loadSystemicRiskData,
        initializeKnowledgeTransfer,
        generateSemanticClusters,
        updateDataStatistics
    };

    Object.assign(window, dataFunctions);

    if (window.AppFunctions) {
        Object.assign(window.AppFunctions, dataFunctions);
    }
    
    console.log('Data loading module initialized.');

})();
    
    