import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class SpeciesAnalyzer:
    def __init__(self, triplets_data):
        self.triplets_data = triplets_data
        from .knowledge_graph import EnhancedKnowledgeGraph
        from .ecological_processor import EcologicalKnowledgeProcessor
        
        self.enhanced_kg = EnhancedKnowledgeGraph()
        self.kg_data = self.enhanced_kg.build_enriched_graph(triplets_data, load_ecological=False)
        self.graph = self.kg_data['graph']
        self.threat_clusters = self.kg_data['threat_clusters']
        iucn_processor = EcologicalKnowledgeProcessor()
        self.iucn_threats = iucn_processor._initialize_iucn_classification()
        
    def analyze_species(self, species_name):
        kg_results = self.analyze_knowledge_graph(species_name)
        
        ecological_context = self.analyze_ecological_context(species_name)
        
        semantic_analysis = self.analyze_semantic_threats(species_name)
        
        return {
            'species': species_name,
            'knowledgeGraph': kg_results,
            'semanticAnalysis': semantic_analysis,
            'ecologicalContext': ecological_context,
            'metadata': {
                'totalTriplets': len(self.triplets_data),
                'speciesCount': self.kg_data['species_count'],
                'threatCount': self.kg_data['threat_count'],
                'threatClusters': len(self.threat_clusters)
            }
        }
    
    def analyze_knowledge_graph(self, species_name):
        try:
            if not self.graph.has_node(species_name):
                return {
                    'nodeCount': 0,
                    'edgeCount': 0,
                    'connectedThreats': 0,
                    'threatCategories': {},
                    'interactionPartners': [],
                    'centrality': {
                        'betweenness': 0.0,
                        'closeness': 0.0,
                        'degree': 0.0
                    },
                    'semanticClusters': []
                }

            try:
                import networkx as nx
                degree_centrality = nx.degree_centrality(self.graph).get(species_name, 0.0)
                
                if self.graph.number_of_nodes() > 1000:
                    neighbors = list(self.graph.neighbors(species_name))
                    subgraph_nodes = [species_name] + neighbors[:50]
                    subgraph = self.graph.subgraph(subgraph_nodes)
                    
                    if subgraph.number_of_nodes() > 1:
                        betweenness_centrality = nx.betweenness_centrality(subgraph).get(species_name, 0.0)
                        closeness_centrality = nx.closeness_centrality(subgraph).get(species_name, 0.0)
                    else:
                        betweenness_centrality = 0.0
                        closeness_centrality = 0.0
                else:
                    betweenness_centrality = nx.betweenness_centrality(self.graph).get(species_name, 0.0)
                    closeness_centrality = nx.closeness_centrality(self.graph).get(species_name, 0.0)
                    
            except Exception as e:
                logger.warning(f"Error calculating centrality measures: {e}")
                degree_centrality = 0.0
                betweenness_centrality = 0.0
                closeness_centrality = 0.0

            centrality_measures = {
                'betweenness': betweenness_centrality,
                'closeness': closeness_centrality,
                'degree': degree_centrality
            }

            connected_threats = []
            threat_categories = {}
            interaction_partners = []

            for neighbor in self.graph.neighbors(species_name):
                node_data = self.graph.nodes[neighbor]
                if node_data.get('node_type') == 'threat':
                    connected_threats.append(neighbor)
                    category = self.categorize_threat_node(neighbor)
                    threat_categories[category] = threat_categories.get(category, 0) + 1
                elif node_data.get('node_type') == 'species':
                    interaction_partners.append(neighbor)

            semantic_clusters = self.get_threat_clusters_for_species(species_name)

            return {
                'nodeCount': len(self.graph.nodes()),
                'edgeCount': len(self.graph.edges()),
                'connectedThreats': len(connected_threats),
                'threatCategories': threat_categories,
                'interactionPartners': interaction_partners[:10],
                'centrality': centrality_measures,
                'semanticClusters': semantic_clusters[:5]
            }
        except Exception as e:
            logger.error(f"Error analyzing knowledge graph for {species_name}: {e}")
            return {
                'nodeCount': 0,
                'edgeCount': 0,
                'connectedThreats': 0,
                'threatCategories': {},
                'interactionPartners': [],
                'centrality': {
                    'betweenness': 0.0,
                    'closeness': 0.0,
                    'degree': 0.0
                },
                'semanticClusters': []
            }
    
    def categorize_threat_node(self, threat_node):
        threat_text = threat_node.lower()
        
        iucn_match = re.search(r'\[iucn:\s*([\d\.]+)\]', threat_text)
        if iucn_match:
            code = iucn_match.group(1)
            main_category_code = code.split('.')[0]
            if main_category_code in self.iucn_threats:
                return self.iucn_threats[main_category_code]['category']

        if 'habitat' in threat_text or 'deforestation' in threat_text or 'land use' in threat_text:
            return 'Habitat Loss & Degradation'
        elif 'climate' in threat_text or 'temperature' in threat_text or 'warming' in threat_text:
            return 'Climate Change'
        elif 'pollution' in threat_text or 'contamination' in threat_text or 'chemical' in threat_text:
            return 'Pollution'
        elif 'hunting' in threat_text or 'harvesting' in threat_text or 'exploitation' in threat_text:
            return 'Overexploitation'
        elif 'invasive' in threat_text or 'alien' in threat_text or 'introduced' in threat_text:
            return 'Invasive Species'
        elif 'disease' in threat_text or 'pathogen' in threat_text or 'virus' in threat_text:
            return 'Disease & Pathogens'
        elif 'development' in threat_text or 'urban' in threat_text or 'infrastructure' in threat_text:
            return 'Infrastructure Development'
        else:
            return 'Other Threats'
    
    def get_threat_clusters_for_species(self, species_name):
        try:
            species_semantic = self.enhanced_kg.get_species_semantic_analysis(species_name)
            
            if species_semantic and species_semantic.get('cluster_info'):
                return [
                    {
                        'label': cluster['label'],
                        'size': cluster.get('size', 0),
                        'keywords': cluster.get('keywords', [])
                    }
                    for cluster in species_semantic['cluster_info']
                ]
            else:
                return []
        except Exception as e:
            logger.warning(f"Error getting threat clusters for {species_name}: {e}")
            return []
    
    def analyze_semantic_threats(self, species_name):
        species_threats = []
        
        for triplet in self.triplets_data:
            if triplet.get('subject', '').lower() == species_name.lower():
                threat = triplet.get('object', '')
                if threat:
                    species_threats.append(threat)
        
        if not species_threats:
            return {
                'total_threats': 0,
                'unique_threats': 0,
                'clusters': [],
                'dominant_cluster': None,
                'cluster_distribution': {}
            }
        
        threat_clusters = defaultdict(list)
        cluster_info = {}
        
        logger.info(f"Performing species-specific semantic clustering for {species_name}")
        
        unique_threats = list(set(species_threats))
        
        if len(unique_threats) >= 2:
            species_semantic = self.enhanced_kg.get_species_semantic_analysis(species_name)
            
            if species_semantic and species_semantic.get('cluster_info'):
                threat_clusters = defaultdict(list)
                
                # Assign threats to clusters based on species-specific analysis
                for i, threat in enumerate(unique_threats):
                    if i < len(species_semantic.get('clusters', [])):
                        cluster_idx = species_semantic['clusters'][i]
                        if cluster_idx != -1 and cluster_idx < len(species_semantic.get('cluster_info', [])):
                            cluster_label = species_semantic['cluster_info'][cluster_idx]['label']
                            threat_clusters[cluster_label].append(threat)
                            
                            if cluster_label not in cluster_info:
                                cluster_info[cluster_label] = species_semantic['cluster_info'][cluster_idx]
        
        if not threat_clusters:
            threat_clusters['All Threats'] = unique_threats
            cluster_info['All Threats'] = {
                'label': 'All Threats',
                'keywords': [],
                'size': len(unique_threats)
            }
        
        cluster_distribution = {
            cluster: len(threats) for cluster, threats in threat_clusters.items()
        }
        
        dominant_cluster = max(cluster_distribution, key=cluster_distribution.get) if cluster_distribution else None
        
        return {
            'total_threats': len(species_threats),
            'unique_threats': len(set(species_threats)),
            'clusters': [
                {
                    'label': cluster_label,
                    'threats': threats,
                    'count': len(threats),
                    'keywords': cluster_info.get(cluster_label, {}).get('keywords', []),
                    'percentage': (len(threats) / len(species_threats)) * 100 if len(species_threats) > 0 else 0
                }
                for cluster_label, threats in threat_clusters.items()
            ],
            'dominant_cluster': dominant_cluster,
            'cluster_distribution': cluster_distribution
        }
    
    def analyze_ecological_context(self, species_name):
        self.enhanced_kg._ensure_ecological_integrator()
        
        interactions = self.enhanced_kg.ecological_integrator.fetch_species_interactions(species_name)
        
        if not interactions:
            return {
                'interactions': [],
                'network_size': 0,
                'interaction_types': {},
                'vulnerability_score': 0,
                'shared_threats': []
            }
        
        network_analysis = self.enhanced_kg.ecological_integrator.analyze_interaction_network(interactions)
        
        interaction_partners = []
        for interaction in interactions[:10]:
            partner_name = interaction.get('target_name', '') or interaction.get('source_name', '')
            if partner_name and partner_name != species_name:
                interaction_partners.append(partner_name)
        
        shared_threats = self.enhanced_kg.ecological_integrator.identify_shared_threats(
            species_name, interaction_partners, self.triplets_data
        )
        
        if self.graph.has_node(species_name):
            self.graph.nodes[species_name]['interactions'] = interactions
            self.graph.nodes[species_name]['network_analysis'] = network_analysis
            self.graph.nodes[species_name]['ecological_processed'] = True
        
        formatted_interactions = []
        for interaction in interactions[:15]:
            partner = interaction.get('target_name', '') or interaction.get('source_name', '')
            if partner and partner != species_name:
                formatted_interactions.append({
                    'partner': partner,
                    'type': interaction.get('standardized_type', 'other'),
                    'direction': interaction.get('direction', 'outgoing')
                })
        
        return {
            'interactions': formatted_interactions,
            'network_size': network_analysis.get('network_size', 0),
            'interaction_types': network_analysis.get('interaction_types', {}),
            'vulnerability_score': network_analysis.get('vulnerability_score', 0),
            'type_diversity': network_analysis.get('type_diversity', 0),
            'shared_threats': shared_threats[:10]
        }

    def analyze_mechanisms(self, species_name):
        mechanisms = defaultdict(int)
        total_threats = 0
        
        for triplet in self.triplets_data:
            if triplet.get('subject', '').lower() == species_name.lower():
                predicate = triplet.get('predicate', '').lower()
                threat_obj = triplet.get('object', '').lower()
                
                total_threats += 1
                
                if any(term in predicate for term in ['threatens', 'endangers', 'affects']):
                    if any(term in threat_obj for term in ['habitat', 'deforestation', 'land use']):
                        mechanisms['habitat_loss'] += 1
                    elif any(term in threat_obj for term in ['climate', 'temperature', 'warming']):
                        mechanisms['climate_change'] += 1
                    elif any(term in threat_obj for term in ['pollution', 'contamination', 'chemical']):
                        mechanisms['pollution'] += 1
                    elif any(term in threat_obj for term in ['hunting', 'exploitation', 'harvesting']):
                        mechanisms['overexploitation'] += 1
                    elif any(term in threat_obj for term in ['disease', 'pathogen', 'virus']):
                        mechanisms['disease'] += 1
                    else:
                        mechanisms['other'] += 1
                else:
                    mechanisms['indirect'] += 1
        
        return {
            'total_analyzed': total_threats,
            'mechanisms': dict(mechanisms),
            'dominant_mechanism': max(mechanisms.keys(), key=mechanisms.get) if mechanisms else None
        }

    def analyze_directness(self, species_name):
        direct_count = 0
        indirect_count = 0
        total_count = 0
        
        for triplet in self.triplets_data:
            if triplet.get('subject', '').lower() == species_name.lower():
                predicate = triplet.get('predicate', '').lower()
                total_count += 1
                
                if any(term in predicate for term in ['directly', 'kills', 'destroys', 'removes']):
                    direct_count += 1
                elif any(term in predicate for term in ['indirectly', 'influences', 'affects', 'contributes']):
                    indirect_count += 1
                else:
                    threat_obj = triplet.get('object', '').lower()
                    if any(term in threat_obj for term in ['habitat', 'climate', 'pollution']):
                        indirect_count += 1
                    else:
                        direct_count += 1
        
        return {
            'total_threats': total_count,
            'direct_threats': direct_count,
            'indirect_threats': indirect_count,
            'directness_ratio': direct_count / total_count if total_count > 0 else 0
        }

    def analyze_sources(self, species_name):
        sources = defaultdict(int)
        dois = set()
        
        for triplet in self.triplets_data:
            if triplet.get('subject', '').lower() == species_name.lower():
                doi = triplet.get('doi')
                if doi:
                    dois.add(doi)
                    if '/' in doi:
                        parts = doi.split('/')
                        if len(parts) >= 2:
                            source_key = parts[0] + '/' + parts[1]
                            sources[source_key] += 1
                    else:
                        sources['unknown'] += 1
                else:
                    sources['no_source'] += 1
        
        return {
            'total_sources': len(dois),
            'unique_dois': list(dois)[:10],
            'source_distribution': dict(sources),
            'most_cited_source': max(sources.keys(), key=sources.get) if sources else None
        }

    def analyze_insights(self, species_name):
        species_threats = []
        for triplet in self.triplets_data:
            if triplet.get('subject', '').lower() == species_name.lower():
                species_threats.append(triplet.get('object', ''))
        
        if not species_threats:
            return {
                'conservation_priority': 'unknown',
                'key_insights': [],
                'recommendations': [],
                'risk_level': 'unknown'
            }
        
        unique_threats = len(set(species_threats))
        total_threats = len(species_threats)
        
        if total_threats > 20:
            risk_level = 'high'
        elif total_threats > 10:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        insights = []
        if unique_threats > 15:
            insights.append("High threat diversity indicates multiple stressors affecting this species")
        if total_threats > unique_threats * 2:
            insights.append("Multiple sources confirm similar threats, suggesting robust evidence")
        
        recommendations = []
        if risk_level == 'high':
            recommendations.append("Immediate conservation action required")
            recommendations.append("Multi-faceted approach needed to address diverse threats")
        elif risk_level == 'medium':
            recommendations.append("Monitor threat development and implement targeted interventions")
        else:
            recommendations.append("Maintain current conservation status with regular monitoring")
        
        return {
            'conservation_priority': risk_level,
            'threat_diversity_score': unique_threats / total_threats if total_threats > 0 else 0,
            'key_insights': insights,
            'recommendations': recommendations,
            'risk_level': risk_level,
            'total_threats_analyzed': total_threats
        } 