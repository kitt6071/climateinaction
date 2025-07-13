import requests
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class EcologicalContextIntegrator:
    def __init__(self):
        self.globi_api_base = "https://api.globalbioticinteractions.org"
        self.interaction_cache = {}
        self.interaction_types = {
            'predatorOf': 'predation',
            'preyOf': 'predation',
            'eats': 'predation',
            'eatenBy': 'predation',
            'competitorOf': 'competition',
            'competesWith': 'competition',
            'mutualistOf': 'mutualism',
            'symbiotWith': 'symbiosis',
            'parasiteOf': 'parasitism',
            'hostOf': 'parasitism',
            'pollinatorOf': 'pollination',
            'pollinatedBy': 'pollination'
        }
    
    def fetch_species_interactions(self, species_name, max_interactions=50):
        if species_name in self.interaction_cache:
            return self.interaction_cache[species_name]
        
        interactions = []
        
        try:
            url = f"{self.globi_api_base}/interaction"
            params = {
                'sourceTaxon': species_name,
                'limit': max_interactions,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('data', []):
                    if len(item) >= 6:
                        interaction = {
                            'source_taxon': item[0],
                            'interaction_type': item[1],
                            'target_taxon': item[2],
                            'source_name': item[3],
                            'target_name': item[5],
                            'standardized_type': self._standardize_interaction_type(item[1])
                        }
                        interactions.append(interaction)
            
            params['targetTaxon'] = species_name
            params.pop('sourceTaxon', None)
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('data', []):
                    if len(item) >= 6:
                        interaction = {
                            'source_taxon': item[0],
                            'interaction_type': item[1],
                            'target_taxon': item[2],
                            'source_name': item[3],
                            'target_name': item[5],
                            'standardized_type': self._standardize_interaction_type(item[1]),
                            'direction': 'incoming'
                        }
                        interactions.append(interaction)
            
        except Exception as e:
            logger.error(f"Failed to fetch interactions for {species_name}: {e}")
        
        self.interaction_cache[species_name] = interactions
        return interactions
    
    def _standardize_interaction_type(self, interaction_type):
        interaction_lower = interaction_type.lower()
        
        for standard_type, category in self.interaction_types.items():
            if standard_type.lower() in interaction_lower:
                return category
        
        if any(word in interaction_lower for word in ['eat', 'prey', 'predator', 'hunt']):
            return 'predation'
        elif any(word in interaction_lower for word in ['compete', 'competition']):
            return 'competition'
        elif any(word in interaction_lower for word in ['mutualist', 'benefit']):
            return 'mutualism'
        elif any(word in interaction_lower for word in ['parasite', 'host']):
            return 'parasitism'
        elif any(word in interaction_lower for word in ['pollinate', 'pollinator']):
            return 'pollination'
        else:
            return 'other'
    
    def analyze_interaction_network(self, species_interactions):
        if not species_interactions:
            return {
                'network_size': 0,
                'interaction_types': {},
                'key_partners': [],
                'vulnerability_score': 0
            }
        
        interaction_types = Counter()
        partners = set()
        
        for interaction in species_interactions:
            interaction_types[interaction['standardized_type']] += 1
            
            if interaction.get('direction') == 'incoming':
                partners.add(interaction['source_name'])
            else:
                partners.add(interaction['target_name'])
        
        type_diversity = len(interaction_types)
        partner_count = len(partners)
        
        vulnerability_score = min(1.0, (type_diversity * 0.2 + min(partner_count, 10) * 0.08))
        
        return {
            'network_size': len(species_interactions),
            'interaction_types': dict(interaction_types),
            'key_partners': list(partners)[:10],
            'vulnerability_score': vulnerability_score,
            'type_diversity': type_diversity
        }
    
    def identify_shared_threats(self, focal_species, related_species, all_triplets):
        if not related_species:
            return []
        
        focal_threats = set()
        for triplet in all_triplets:
            if triplet.get('subject', '').lower() == focal_species.lower():
                focal_threats.add(triplet.get('object', ''))
        
        shared_threats = []
        
        for partner in related_species:
            partner_threats = set()
            for triplet in all_triplets:
                if triplet.get('subject', '').lower() == partner.lower():
                    partner_threats.add(triplet.get('object', ''))
            
            common_threats = focal_threats.intersection(partner_threats)
            
            if common_threats:
                shared_threats.append({
                    'species': partner,
                    'shared_threats': list(common_threats),
                    'threat_count': len(common_threats)
                })
        
        return sorted(shared_threats, key=lambda x: x['threat_count'], reverse=True) 