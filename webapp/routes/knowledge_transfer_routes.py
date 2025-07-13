from flask import Blueprint, request, jsonify
from collections import defaultdict
import logging
import config
from config import logger
from utils import load_data_if_needed

knowledge_transfer_bp = Blueprint('knowledge_transfer', __name__)

@knowledge_transfer_bp.route('/knowledge_transfer', methods=['POST'])
def knowledge_transfer_analysis():
    """
    Knowledge transfer analysis endpoint - analyzes similar species to predict potential threats
    """
    try:
        data = request.get_json()
        target_species = data.get('target_species')
        similarity_threshold = data.get('similarity_threshold', 0.7)
        min_evidence_count = data.get('min_evidence_count', 3)
        
        if not target_species:
            return jsonify({'error': 'Target species name required'}), 400
        
        if not load_data_if_needed():
            return jsonify({'error': 'Failed to load triplet data'}), 500
        
        logger.info(f"Knowledge transfer analysis for '{target_species}' with threshold {similarity_threshold}")
        
        # Find target species data
        target_threats = []
        target_triplets = []
        for triplet in config.triplets_data:
            if triplet.get('subject', '').lower() == target_species.lower():
                target_threats.append(triplet.get('object', ''))
                target_triplets.append(triplet)
        
        logger.info(f"Found {len(target_triplets)} triplets for target species '{target_species}'")
        
        if len(target_triplets) == 0:
            # Try partial matching
            logger.info(f"No exact match for '{target_species}', trying partial matching...")
            for triplet in config.triplets_data:
                species_name = triplet.get('subject', '').lower()
                if target_species.lower() in species_name or species_name in target_species.lower():
                    target_threats.append(triplet.get('object', ''))
                    target_triplets.append(triplet)
            
            logger.info(f"Partial matching found {len(target_triplets)} triplets for '{target_species}'")
        
        if len(target_triplets) == 0:
            # Provide suggestions
            all_species = list(set([triplet.get('subject', '') for triplet in config.triplets_data if triplet.get('subject')]))
            suggestions = []
            
            for species in all_species:
                if (target_species.lower() in species.lower() or 
                    species.lower() in target_species.lower() or
                    any(word in species.lower() for word in target_species.lower().split())):
                    suggestions.append(species)
            
            return jsonify({
                'error': f'No data found for species "{target_species}"',
                'suggestions': suggestions[:10],
                'total_species_count': len(all_species),
                'message': 'Try one of the suggested species names or use the Explorer tab to browse available species.'
            }), 404

        target_threat_set = set(target_threats)
        
        # Find similar species
        similar_species = {}
        
        for triplet in config.triplets_data:
            species = triplet.get('subject', '')
            threat = triplet.get('object', '')
            
            if species.lower() != target_species.lower() and threat in target_threat_set:
                if species not in similar_species:
                    similar_species[species] = {
                        'shared_threats': set(),
                        'unique_threats': set(),
                        'all_threats': set(),
                        'triplets': []
                    }
                similar_species[species]['shared_threats'].add(threat)
                similar_species[species]['all_threats'].add(threat)
                similar_species[species]['triplets'].append(triplet)
        
        logger.info(f"Found {len(similar_species)} species with shared threats")
        
        # Collect all threats for similar species
        for triplet in config.triplets_data:
            species = triplet.get('subject', '')
            threat = triplet.get('object', '')
            
            if species in similar_species:
                similar_species[species]['all_threats'].add(threat)
                if threat not in target_threat_set:
                    similar_species[species]['unique_threats'].add(threat)
                if triplet not in similar_species[species]['triplets']:
                    similar_species[species]['triplets'].append(triplet)
        
        knowledge_transfer_candidates = []
        
        for species, data_dict in similar_species.items():
            if len(data_dict['shared_threats']) < 2:
                continue
                
            jaccard_similarity = len(data_dict['shared_threats']) / len(data_dict['all_threats'].union(target_threat_set))
            coverage_similarity = len(data_dict['shared_threats']) / len(target_threat_set) if target_threat_set else 0
            combined_similarity = (jaccard_similarity + coverage_similarity) / 2
            
            logger.debug(f"Species {species}: jaccard={jaccard_similarity:.3f}, coverage={coverage_similarity:.3f}, combined={combined_similarity:.3f}")
            
            if combined_similarity >= similarity_threshold:
                transferable_threats = []
                
                for unique_threat in data_dict['unique_threats']:
                    evidence_triplets = [t for t in data_dict['triplets'] if t.get('object') == unique_threat]
                    
                    if len(evidence_triplets) >= min_evidence_count:
                        threat_analysis = analyze_threat_transferability(
                            unique_threat, evidence_triplets, target_species, target_triplets
                        )
                        
                        # Clean evidence triplets for JSON serialization
                        clean_evidence_triplets = []
                        for triplet in evidence_triplets[:5]:
                            clean_triplet = {}
                            for key, value in triplet.items():
                                if hasattr(value, 'tolist'):
                                    clean_triplet[key] = value.tolist()
                                elif hasattr(value, 'item'):
                                    clean_triplet[key] = value.item()
                                else:
                                    clean_triplet[key] = value
                            clean_evidence_triplets.append(clean_triplet)
                        
                        transferable_threats.append({
                            'threat': unique_threat,
                            'evidence_count': len(evidence_triplets),
                            'evidence_triplets': clean_evidence_triplets,
                            'transferability_score': float(threat_analysis['transferability_score']),
                            'transfer_reasoning': threat_analysis['reasoning'],
                            'suggested_research': threat_analysis['research_suggestions']
                        })
                
                transferable_threats.sort(key=lambda x: x['transferability_score'], reverse=True)
                
                # Clean taxonomy info for JSON serialization
                taxonomy_info = {}
                if data_dict['triplets']:
                    raw_taxonomy = data_dict['triplets'][0].get('taxonomy', {})
                    for key, value in raw_taxonomy.items():
                        if hasattr(value, 'tolist'):
                            taxonomy_info[key] = value.tolist()
                        elif hasattr(value, 'item'):
                            taxonomy_info[key] = value.item()
                        else:
                            taxonomy_info[key] = value
                
                knowledge_transfer_candidates.append({
                    'similar_species': species,
                    'jaccard_similarity': float(jaccard_similarity),
                    'coverage_similarity': float(coverage_similarity),
                    'combined_similarity': float(combined_similarity),
                    'shared_threats': list(data_dict['shared_threats']),
                    'shared_threat_count': len(data_dict['shared_threats']),
                    'total_threats': len(data_dict['all_threats']),
                    'transferable_threats': transferable_threats[:10],
                    'taxonomy_info': taxonomy_info
                })
        
        logger.info(f"Found {len(knowledge_transfer_candidates)} candidates above similarity threshold {similarity_threshold}")
        
        knowledge_transfer_candidates.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        # Generate analysis summaries
        gap_analysis = analyze_knowledge_gaps_for_transfer(target_species, target_triplets, knowledge_transfer_candidates)
        research_recommendations = generate_research_recommendations(target_species, knowledge_transfer_candidates, gap_analysis)
        
        return jsonify({
            'target_species': target_species,
            'current_threat_count': len(target_threats),
            'current_threats': list(target_threat_set),
            'similar_species_count': len(knowledge_transfer_candidates),
            'knowledge_transfer_candidates': knowledge_transfer_candidates[:10],
            'knowledge_gaps': gap_analysis,
            'research_recommendations': research_recommendations,
            'analysis_parameters': {
                'similarity_threshold': float(similarity_threshold),
                'min_evidence_count': int(min_evidence_count)
            },
            'debug_info': {
                'target_triplets_found': len(target_triplets),
                'species_with_shared_threats': len(similar_species),
                'candidates_above_threshold': len(knowledge_transfer_candidates)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in knowledge transfer analysis: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def analyze_threat_transferability(threat, evidence_triplets, target_species, target_triplets):
    """Analyze how transferable a threat is from one species to another"""
    transferability_score = 0.0
    reasoning_factors = []
    
    # Factor 1: Threat type generalizability
    threat_lower = threat.lower()
    if any(keyword in threat_lower for keyword in ['climate change', 'habitat loss', 'pollution', 'pesticide', 'chemical', 'contamination']):
        transferability_score += 0.3
        reasoning_factors.append("Broad environmental threat likely affects multiple species")
    
    # Factor 2: Evidence strength
    evidence_strength = min(len(evidence_triplets) / 10.0, 0.3)
    transferability_score += evidence_strength
    reasoning_factors.append(f"Strong evidence base with {len(evidence_triplets)} documented cases")
    
    # Factor 3: Similar impact mechanisms
    predicates = [t.get('predicate', '') for t in evidence_triplets]
    common_impact_types = ['mortality', 'population decline', 'habitat degradation', 'behavioral change']
    
    predicate_scores = []
    for predicate in predicates:
        for impact_type in common_impact_types:
            if impact_type.replace(' ', '') in predicate.lower().replace(' ', ''):
                predicate_scores.append(0.1)
                break
    
    if predicate_scores:
        transferability_score += min(sum(predicate_scores), 0.2)
        reasoning_factors.append("Similar impact mechanisms documented")
    
    # Factor 4: Habitat similarity (basic)
    target_habitat_keywords = []
    for triplet in target_triplets:
        predicate = triplet.get('predicate', '').lower()
        if any(keyword in predicate for keyword in ['habitat', 'ecosystem', 'environment']):
            target_habitat_keywords.extend(predicate.split())
    
    evidence_habitat_keywords = []
    for triplet in evidence_triplets:
        predicate = triplet.get('predicate', '').lower()
        if any(keyword in predicate for keyword in ['habitat', 'ecosystem', 'environment']):
            evidence_habitat_keywords.extend(predicate.split())
    
    if target_habitat_keywords and evidence_habitat_keywords:
        habitat_overlap = len(set(target_habitat_keywords) & set(evidence_habitat_keywords))
        if habitat_overlap > 0:
            transferability_score += 0.1
            reasoning_factors.append("Shared habitat characteristics identified")
    
    # Factor 5: Recency of evidence
    recent_studies = 0
    for triplet in evidence_triplets:
        doi = triplet.get('doi', '')
        if doi and any(year in doi for year in ['2020', '2021', '2022', '2023', '2024']):
            recent_studies += 1
    
    if recent_studies > 0:
        transferability_score += 0.1
        reasoning_factors.append(f"{recent_studies} recent studies provide current evidence")
    
    # Generate research suggestions
    research_suggestions = []
    
    if transferability_score > 0.6:
        research_suggestions.append(f"High priority: Investigate {threat} in {target_species}")
        research_suggestions.append(f"Search for evidence of similar impacts: {', '.join(set([t.get('predicate', '')[:50] for t in evidence_triplets[:3]]))}")
    elif transferability_score > 0.4:
        research_suggestions.append(f"Moderate priority: Consider {threat} as potential threat")
        research_suggestions.append("Conduct preliminary habitat/exposure assessment")
    else:
        research_suggestions.append(f"Low priority: Limited evidence for transferability")
    
    return {
        'transferability_score': transferability_score,
        'reasoning': '; '.join(reasoning_factors),
        'research_suggestions': research_suggestions
    }

def analyze_knowledge_gaps_for_transfer(target_species, target_triplets, transfer_candidates):
    """Analyze knowledge gaps that could be filled through transfer learning"""
    gaps = {
        'threat_categories_missing': [],
        'impact_mechanisms_understudied': [],
        'geographic_coverage_gaps': [],
        'temporal_coverage_gaps': [],
        'methodological_gaps': []
    }
    
    # Identify missing threat categories
    target_threat_categories = set()
    for triplet in target_triplets:
        threat = triplet.get('object', '').lower()
        if 'climate' in threat:
            target_threat_categories.add('Climate Change')
        elif 'habitat' in threat:
            target_threat_categories.add('Habitat Loss')
        elif 'pollution' in threat:
            target_threat_categories.add('Pollution')
        elif 'invasive' in threat:
            target_threat_categories.add('Invasive Species')
        elif 'disease' in threat:
            target_threat_categories.add('Disease')
    
    all_categories = {'Climate Change', 'Habitat Loss', 'Pollution', 'Invasive Species', 'Disease', 'Overexploitation'}
    missing_categories = all_categories - target_threat_categories
    
    # Check what categories could be filled by transfer candidates
    for candidate in transfer_candidates:
        for threat_info in candidate['transferable_threats']:
            threat = threat_info['threat'].lower()
            for category in missing_categories:
                if category.lower().replace(' ', '') in threat.replace(' ', ''):
                    if category not in gaps['threat_categories_missing']:
                        gaps['threat_categories_missing'].append(category)
    
    # Identify understudied impact mechanisms
    target_mechanisms = set()
    for triplet in target_triplets:
        predicate = triplet.get('predicate', '').lower()
        if 'mortality' in predicate:
            target_mechanisms.add('Direct Mortality')
        elif 'population' in predicate:
            target_mechanisms.add('Population Effects')
        elif 'behavior' in predicate:
            target_mechanisms.add('Behavioral Changes')
        elif 'reproduction' in predicate:
            target_mechanisms.add('Reproductive Impacts')
    
    all_mechanisms = {'Direct Mortality', 'Population Effects', 'Behavioral Changes', 'Reproductive Impacts', 'Physiological Stress', 'Habitat Modification'}
    gaps['impact_mechanisms_understudied'] = list(all_mechanisms - target_mechanisms)
    
    return gaps

def generate_research_recommendations(target_species, transfer_candidates, gap_analysis):
    """Generate actionable research recommendations based on transfer analysis"""
    recommendations = []
    
    # High-priority threats from transfer analysis
    high_priority_threats = []
    for candidate in transfer_candidates[:3]:
        for threat_info in candidate['transferable_threats'][:2]:
            if threat_info['transferability_score'] > 0.6:
                high_priority_threats.append({
                    'threat': threat_info['threat'],
                    'similar_species': candidate['similar_species'],
                    'score': threat_info['transferability_score'],
                    'evidence_count': threat_info['evidence_count']
                })
    
    if high_priority_threats:
        recommendations.append({
            'type': 'immediate_research',
            'priority': 'High',
            'title': f'Investigate High-Priority Threats for {target_species}',
            'description': f'Based on analysis of similar species, investigate these threats with strong evidence bases',
            'specific_actions': [f"Study {threat['threat']} (evidence: {threat['evidence_count']} studies from {threat['similar_species']})" 
                               for threat in sorted(high_priority_threats, key=lambda x: x['score'], reverse=True)[:3]]
        })
    
    # Literature search recommendations
    search_terms = []
    for candidate in transfer_candidates[:2]:
        for threat_info in candidate['transferable_threats'][:1]:
            search_terms.append(f'"{target_species}" AND "{threat_info["threat"]}"')
    
    if search_terms:
        recommendations.append({
            'type': 'literature_search',
            'priority': 'Medium',
            'title': 'Targeted Literature Search',
            'description': 'Search for existing evidence of predicted threats',
            'specific_actions': [f"Search: {term}" for term in search_terms[:3]]
        })
    
    # Field study recommendations
    if gap_analysis['threat_categories_missing']:
        recommendations.append({
            'type': 'field_study',
            'priority': 'Medium',
            'title': 'Field Assessment of Understudied Threat Categories',
            'description': f'Conduct field studies to assess {target_species} exposure to these threat categories',
            'specific_actions': [f"Assess exposure to {category}" for category in gap_analysis['threat_categories_missing'][:3]]
        })
    
    return recommendations 