from flask import Blueprint, request, jsonify
import logging
import config
from config import logger
from utils import load_data_if_needed
from models import SystemicRiskAnalyzer

network_bp = Blueprint('network', __name__)

# Create systemic analyzer instance
systemic_analyzer = SystemicRiskAnalyzer()

@network_bp.route('/network_analysis', methods=['POST'])
def network_analysis():
    try:
        logger.info("Starting network analysis...")
        
        if not load_data_if_needed():
            logger.error("Failed to load data for network analysis")
            return jsonify({'success': False, 'error': 'Failed to load triplet data'}), 500
        
        data = request.get_json()
        analysis_type = data.get('analysis_type', 'shared_threats')
        species_list = data.get('species_list', [])
        
        logger.info(f"Analysis type: {analysis_type}, Species count: {len(species_list) if species_list else 'all'}")
        
        if not config.triplets_data:
            logger.error("No triplet data available")
            return jsonify({'success': False, 'error': 'No triplet data loaded'}), 500
        
        species_threats_data = {}
        
        if not species_list:
            species_list = list(set([triplet.get('subject', '') for triplet in config.triplets_data]))
            logger.info(f"Using all species: {len(species_list)} species found")
        
        for triplet in config.triplets_data:
            species = triplet.get('subject', '')
            if species in species_list:
                threat_obj = triplet.get('object', '')
                if '[IUCN:' in threat_obj:
                    threat_name = threat_obj.split('[IUCN:')[0].strip()
                else:
                    threat_name = threat_obj.strip()
                
                if species not in species_threats_data:
                    species_threats_data[species] = []
                
                if threat_name and threat_name not in species_threats_data[species]:
                    species_threats_data[species].append(threat_name)
        
        species_threats_data = {k: v for k, v in species_threats_data.items() if v}
        logger.info(f"✅ Processed {len(species_threats_data)} species with threats")
        
        if analysis_type == 'shared_threats':
            network = systemic_analyzer.build_ecological_network(species_threats_data)
        else:
            network = systemic_analyzer.build_ecological_network(species_threats_data)
        
        logger.info("✅ Network analysis completed successfully")
        return jsonify({
            'success': True,
            'network': network,
            'analysis_type': analysis_type,
            'species_count': len(species_threats_data),
            'species_included': list(species_threats_data.keys())
        })
        
    except Exception as e:
        logger.error(f"❌ Network analysis failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Network analysis failed: {str(e)}'}), 500

@network_bp.route('/indirect_impacts', methods=['POST'])
def indirect_impacts():
    try:
        data = request.get_json()
        focal_species = data.get('focal_species')
        
        if not focal_species:
            return jsonify({'success': False, 'error': 'Focal species required'}), 400
        
        species_threats_data = {}
        for triplet in config.triplets_data:
            species = triplet.get('subject', '')
            threat = triplet.get('object', '')
            if species and threat:
                if species not in species_threats_data:
                    species_threats_data[species] = []
                if threat not in species_threats_data[species]:
                    species_threats_data[species].append(threat)
        
        impacts = systemic_analyzer.find_indirect_impacts(focal_species, species_threats_data)
        
        return jsonify({
            'success': True,
            'focal_species': focal_species,
            'indirect_impacts': impacts
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@network_bp.route('/knowledge_graph_query', methods=['POST'])
def knowledge_graph_query():
    try:
        data = request.get_json()
        query_type = data.get('query_type')
        custom_query = data.get('custom_query', '')
        
        if query_type == 'shared_threats':
            species_threats = {}
            for triplet in config.triplets_data:
                species = triplet.get('subject', '')
                threat = triplet.get('object', '')
                if species and threat:
                    if species not in species_threats:
                        species_threats[species] = set()
                    species_threats[species].add(threat)
            
            results = []
            species_list = list(species_threats.keys())
            for i, species1 in enumerate(species_list):
                for species2 in species_list[i+1:]:
                    shared = species_threats[species1] & species_threats[species2]
                    for threat in list(shared)[:3]:
                        results.append({
                            'species': species1, 
                            'threat': threat, 
                            'connection': species2
                        })
            results = results[:10]
            
        elif query_type == 'semantic_similarity':
            threats = list(set([t.get('object', '') for t in config.triplets_data if t.get('object')]))
            results = []
            for i, threat1 in enumerate(threats[:5]):
                for threat2 in threats[i+1:6]:
                    similarity = len(set(threat1.lower().split()) & set(threat2.lower().split())) / max(len(threat1.split()), len(threat2.split()))
                    if similarity > 0.3:
                        results.append({
                            'threat1': threat1, 
                            'threat2': threat2, 
                            'similarity': round(similarity, 2)
                        })
            results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:5]
            
        elif query_type == 'cascade_paths':
            graph = config.analyzer.enhanced_kg.graph
            results = []
            species_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'species']
            for species in species_nodes[:5]:
                threats = [n for n in graph.neighbors(species) if graph.nodes[n].get('node_type') == 'threat']
                for threat in threats[:2]:
                    risk_score = min(graph.degree(species) / 10, 1.0)
                    results.append({
                        'path': f'{threat} → {species}',
                        'risk': round(risk_score, 2)
                    })
            results = sorted(results, key=lambda x: x['risk'], reverse=True)[:5]
            
        else:
            results = [{'message': 'Custom query executed', 'query': custom_query}]
        
        return jsonify({
            'success': True,
            'query_type': query_type,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@network_bp.route('/systemic_metrics', methods=['GET'])
def systemic_metrics():
    try:
        species_threats_data = {}
        for triplet in config.triplets_data:
            species = triplet.get('subject', '')
            threat = triplet.get('object', '')
            if species and threat:
                if species not in species_threats_data:
                    species_threats_data[species] = []
                if threat not in species_threats_data[species]:
                    species_threats_data[species].append(threat)
        
        metrics = systemic_analyzer.calculate_systemic_metrics(species_threats_data)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500 