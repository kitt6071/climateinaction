    function executeGraphQuery(queryType, parameters = {}) {
        const queryFunctions = {
            'interacting_species_same_threat_category': findInteractingSpeciesSameThreatCategory,
            'species_similar_threats_with_interactions': findSpeciesSimilarThreatsWithInteractions,
            'threat_impact_chains': findThreatImpactChains,
            'vulnerability_pathways': findVulnerabilityPathways
        };
        const func = queryFunctions[queryType];
        return func ? func(parameters) : undefined;
    }


    function executeKnowledgeGraphQuery(queryType) {
        const queryTypeMap = {
            'vulnerable_pairs': 'interacting_species_same_threat_category',
            'threat_clusters': 'vulnerability_pathways',
            'cascade_risks': 'threat_impact_chains',
            'keystone_threats': 'species_similar_threats_with_interactions'
        };
        const mappedQueryType = queryTypeMap[queryType] || queryType;

        if (knowledgeGraph.nodes.size === 0) {
            alert('Knowledge Graph not initialized. Please ensure data is loaded.');
            return;
        }

        try {
            let parameters = {};
            if (mappedQueryType === 'species_similar_threats_with_interactions') {
                parameters.focal_species = focalSpeciesSelect ? focalSpeciesSelect.value : '';
            } else if (mappedQueryType === 'threat_impact_chains') {
                parameters.max_depth = 3;
            }

            const results = executeGraphQuery(mappedQueryType, parameters);

            if (results === undefined) {
                alert('Unknown query type: ' + queryType);
                return;
            }

            displayKnowledgeGraphResults(results, mappedQueryType);
        } catch (error) {
            alert('Error executing query: ' + error.message);
        }
    }


    function findInteractingSpeciesSameThreatCategory() {
        const results = [];
        const threatCategoryMap = new Map();

        knowledgeGraph.edges.forEach(edge => {
            if (edge.type === 'EXPERIENCES_IMPACT') {
                const speciesNode = knowledgeGraph.nodes.get(edge.source);
                const impactNode = knowledgeGraph.nodes.get(edge.target);

                if (!threatCategoryMap.has(impactNode.name)) {
                    threatCategoryMap.set(impactNode.name, new Set());
                }
                threatCategoryMap.get(impactNode.name).add(speciesNode.name);
            }
        });

        threatCategoryMap.forEach((speciesSet, threatCategory) => {
            const speciesArray = Array.from(speciesSet);
            if (speciesArray.length > 1) {
                for (let i = 0; i < speciesArray.length; i++) {
                    for (let j = i + 1; j < speciesArray.length; j++) {
                        results.push({
                            species1: speciesArray[i],
                            species2: speciesArray[j],
                            shared_threat_category: threatCategory,
                            interaction_type: 'potential',
                            confidence: 0.7
                        });
                    }
                }
            }
        });

        return results.slice(0, 20);
    }

    function findSpeciesSimilarThreatsWithInteractions({ focal_species }) {
        if (!focal_species) return [];

        const focalSpeciesNode = Array.from(knowledgeGraph.speciesNodes.values())
            .find(node => node.name.toLowerCase() === focal_species.toLowerCase());

        if (!focalSpeciesNode) return [];

        const getThreats = (speciesId) => {
            return Array.from(knowledgeGraph.edges.values())
                .filter(edge => edge.type === 'HAS_THREAT' && edge.source === speciesId)
                .map(edge => knowledgeGraph.nodes.get(edge.target));
        };

        const focalThreats = getThreats(focalSpeciesNode.id);
        const similarityThreshold = 0.6;
        const results = [];

        knowledgeGraph.speciesNodes.forEach(speciesNode => {
            if (speciesNode.id === focalSpeciesNode.id) return;

            const speciesThreats = getThreats(speciesNode.id);
            let maxSimilarity = 0;
            const sharedThreats = [];

            focalThreats.forEach(focalThreat => {
                speciesThreats.forEach(speciesThreat => {
                    const similarity = calculateTextSimilarity(
                        focalThreat.properties.description,
                        speciesThreat.properties.description
                    );
                    maxSimilarity = Math.max(maxSimilarity, similarity);
                    if (similarity > similarityThreshold) {
                        sharedThreats.push({
                            focal_threat: focalThreat.properties.description,
                            similar_threat: speciesThreat.properties.description,
                            similarity: similarity
                        });
                    }
                });
            });

            if (maxSimilarity > similarityThreshold) {
                results.push({
                    focal_species,
                    other_species: speciesNode.name,
                    max_similarity: maxSimilarity,
                    shared_threats: sharedThreats,
                    interaction_potential: 'unknown'
                });
            }
        });

        return results.sort((a, b) => b.max_similarity - a.max_similarity).slice(0, 15);
    }

    function findThreatImpactChains({ max_depth = 3 } = {}) {
        const results = [];

        knowledgeGraph.speciesNodes.forEach(startSpecies => {
            function findChains(currentNode, currentPath, depth) {
                if (depth >= max_depth) return;

                knowledgeGraph.edges.forEach(edge => {
                    if (edge.source === currentNode.id) {
                        const targetNode = knowledgeGraph.nodes.get(edge.target);

                        if (currentPath.some(step => step.node.id === targetNode.id)) return;

                        const newPath = [...currentPath, {
                            node: targetNode,
                            edge_type: edge.type,
                            edge_id: edge.id
                        }];

                        if (targetNode.type === 'Species' && depth > 1 && targetNode.id !== startSpecies.id) {
                            results.push({
                                chain_start: startSpecies.name,
                                chain_end: targetNode.name,
                                path: newPath.map(step => ({
                                    node_type: step.node.type,
                                    node_name: step.node.name,
                                    relationship: step.edge_type
                                })),
                                chain_length: newPath.length,
                                confidence: 1.0 / newPath.length
                            });
                        }

                        findChains(targetNode, newPath, depth + 1);
                    }
                });
            }

            findChains(startSpecies, [], 0);
        });

        return results
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 20);
    }

    function findVulnerabilityPathways() {
        const results = [];
        const impactSpeciesMap = new Map();

        knowledgeGraph.edges.forEach(edge => {
            if (edge.type === 'EXPERIENCES_IMPACT') {
                const speciesNode = knowledgeGraph.nodes.get(edge.source);
                const impactNode = knowledgeGraph.nodes.get(edge.target);

                if (!impactSpeciesMap.has(impactNode.name)) {
                    impactSpeciesMap.set(impactNode.name, new Set());
                }
                impactSpeciesMap.get(impactNode.name).add(speciesNode.name);
            }
        });

        knowledgeGraph.speciesNodes.forEach(species => {
            let vulnerabilityScore = 0;
            const sharedImpactSpecies = new Set();

            impactSpeciesMap.forEach(speciesSet => {
                if (speciesSet.has(species.name)) {
                    vulnerabilityScore += speciesSet.size;
                    speciesSet.forEach(otherSpecies => {
                        if (otherSpecies !== species.name) {
                            sharedImpactSpecies.add(otherSpecies);
                        }
                    });
                }
            });

            if (sharedImpactSpecies.size > 0) {
                results.push({
                    species: species.name,
                    vulnerability_score: vulnerabilityScore,
                    connected_species_count: sharedImpactSpecies.size,
                    connected_species: Array.from(sharedImpactSpecies).slice(0, 5),
                    pathway_type: 'shared_vulnerability'
                });
            }
        });

        return results
            .sort((a, b) => b.vulnerability_score - a.vulnerability_score)
            .slice(0, 15);
    }
    