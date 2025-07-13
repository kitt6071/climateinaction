from flask import Blueprint, request, jsonify
from collections import defaultdict
import logging
from urllib.parse import unquote
import config
from config import logger

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/species_analysis', methods=['POST'])
def analyze_species():
    try:
        data = request.get_json()
        species_name = data.get('species_name')
        
        if not species_name:
            return jsonify({'error': 'Species name required'}), 400
        
        analysis_result = config.analyzer.analyze_species(species_name)
        
        species_threats = []
        for triplet in config.triplets_data:
            if triplet.get('subject', '').lower() == species_name.lower():
                threat = triplet.get('object', '')
                if threat:
                    species_threats.append(threat)
        
        semantic_clusters = []
        if analysis_result.get('semanticAnalysis') and analysis_result['semanticAnalysis'].get('clusters'):
            for cluster in analysis_result['semanticAnalysis']['clusters']:
                semantic_clusters.append({
                    'category': cluster.get('label', 'Unknown'),
                    'threats': cluster.get('threats', []),
                    'keywords': cluster.get('keywords', []),
                    'size': cluster.get('count', 0)
                })
        
        impact_analysis = []
        if analysis_result.get('knowledgeGraph') and analysis_result['knowledgeGraph'].get('threatCategories'):
            for category, count in analysis_result['knowledgeGraph']['threatCategories'].items():
                impact_analysis.append({
                    'category': category,
                    'count': count,
                    'percentage': (count / len(species_threats) * 100) if species_threats else 0
                })
        
        comprehensive_profile = {
            'total_threats': len(species_threats),
            'threat_categories': analysis_result.get('knowledgeGraph', {}).get('threatCategories', {}),
            'ecological_interactions': len(analysis_result.get('ecologicalContext', {}).get('interactions', [])),
            'centrality_scores': analysis_result.get('knowledgeGraph', {}).get('centrality', {}),
            'semantic_clusters_count': len(semantic_clusters)
        }
        
        impact_keywords = {
            'Population Decline': ['population', 'decline', 'decrease', 'reduction'],
            'Habitat Degradation': ['habitat', 'degradation', 'loss', 'fragmentation', 'destruction'],
            'Behavioral Changes': ['behavior', 'behaviour', 'movement', 'foraging', 'avoidance', 'disturbance'],
            'Reproductive Impact': ['reproduction', 'breeding', 'nesting', 'fecundity', 'hatching', 'offspring'],
            'Mortality': ['mortality', 'death', 'kill', 'survival', 'die']
        }
        
        category_impact_counts = defaultdict(lambda: defaultdict(int))
        
        species_triplets = [t for t in config.triplets_data if t.get('subject', '').lower() == species_name.lower()]
        
        for triplet in species_triplets:
            threat_text = triplet.get('object', '')
            impact_text = triplet.get('predicate', '').lower()
            category = config.analyzer.categorize_threat_node(threat_text)
            
            category_impact_counts[category]['total'] += 1
            
            for impact_type, keywords in impact_keywords.items():
                if any(keyword in impact_text for keyword in keywords):
                    category_impact_counts[category][impact_type] += 1
        
        threat_impact_probabilities = defaultdict(dict)
        for category, counts in category_impact_counts.items():
            total_in_category = counts['total']
            if total_in_category > 0:
                for impact_type in impact_keywords.keys():
                    probability = counts.get(impact_type, 0) / total_in_category
                    threat_impact_probabilities[category][impact_type] = round(probability, 2)

        comprehensive_profile['threat_impact_probabilities'] = dict(threat_impact_probabilities)
        
        threat_categories = analysis_result.get('knowledgeGraph', {}).get('threatCategories', {})
        
        response_data = {
            'species_name': species_name,
            'total_threats': len(species_threats),
            'semantic_clusters': semantic_clusters,
            'impact_analysis': impact_analysis,
            'comprehensive_profile': comprehensive_profile,
            'threat_categories': threat_categories,
            'impact_categories_count': len(impact_analysis),
            'analysis_result': analysis_result
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in species analysis: {e}")
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/species/<species_name>/deepdive')
def get_species_deepdive(species_name):
    try:
        result = config.analyzer.analyze_species(species_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/semantic/threat-landscape')
def get_semantic_threat_landscape():
    try:
        landscape = config.analyzer.enhanced_kg.get_semantic_threat_landscape()
        return jsonify(landscape)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/ecological/interactions/<species_name>')
def get_species_interactions(species_name):
    try:
        integrator = config.analyzer.enhanced_kg.ecological_integrator
        interactions = integrator.fetch_species_interactions(species_name)
        network_analysis = integrator.analyze_interaction_network(interactions)
        
        return jsonify({
            'species': species_name,
            'interactions': interactions[:20],
            'network_analysis': network_analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/semantic/clusters/<species_name>')
def get_species_threat_clusters(species_name):
    try:
        semantic_analysis = config.analyzer.analyze_semantic_threats(species_name)
        return jsonify(semantic_analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/knowledge-graph/stats')
def get_knowledge_graph_stats():
    try:
        graph = config.analyzer.enhanced_kg.graph
        
        stats = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'species_nodes': len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'species']),
            'threat_nodes': len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'threat']),
            'threat_clusters': len(config.analyzer.threat_clusters),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        }
        
        cluster_summary = [
            {
                'label': cluster['label'],
                'size': cluster['size'],
                'keywords': cluster['keywords'][:3]
            }
            for cluster in config.analyzer.threat_clusters
        ]
        
        stats['cluster_summary'] = cluster_summary
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/shared-threats/<species_name>')
def get_shared_threats(species_name):
    try:
        if not config.analyzer.enhanced_kg.graph.has_node(species_name):
            return jsonify({'shared_threats': [], 'message': 'Species not found in knowledge graph'})
        
        node_data = config.analyzer.enhanced_kg.graph.nodes[species_name]
        interactions = node_data.get('interactions', [])
        
        partners = []
        for interaction in interactions[:15]:
            partner_name = interaction.get('target_name', '') or interaction.get('source_name', '')
            if partner_name and partner_name != species_name:
                partners.append(partner_name)
        
        shared_threats = config.analyzer.enhanced_kg.ecological_integrator.identify_shared_threats(
            species_name, partners, config.analyzer.triplets_data
        )
        
        return jsonify({
            'species': species_name,
            'interaction_partners': len(partners),
            'shared_threats': shared_threats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/centrality/<species_name>')
def get_species_centrality(species_name):
    try:
        graph = config.analyzer.enhanced_kg.graph
        
        if not graph.has_node(species_name):
            return jsonify({'error': 'Species not found in knowledge graph'}), 404
        
        centrality_measures = {}
        
        try:
            import networkx as nx
            centrality_measures['degree'] = nx.degree_centrality(graph).get(species_name, 0)
            centrality_measures['betweenness'] = nx.betweenness_centrality(graph).get(species_name, 0)
            centrality_measures['closeness'] = nx.closeness_centrality(graph).get(species_name, 0)
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(graph, max_iter=1000).get(species_name, 0)
        except:
            centrality_measures = {
                'degree': graph.degree(species_name) / (graph.number_of_nodes() - 1) if graph.number_of_nodes() > 1 else 0,
                'betweenness': 0,
                'closeness': 0,
                'eigenvector': 0
            }
        
        neighbors = list(graph.neighbors(species_name))
        
        return jsonify({
            'species': species_name,
            'centrality_measures': centrality_measures,
            'direct_connections': len(neighbors),
            'connected_threats': [n for n in neighbors if graph.nodes[n].get('node_type') == 'threat'],
            'connected_species': [n for n in neighbors if graph.nodes[n].get('node_type') == 'species']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 