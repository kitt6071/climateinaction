import logging

logger = logging.getLogger(__name__)

class SystemicRiskAnalyzer:
    def __init__(self):
        self.species_data = {}
        self.threat_graph = {}
        
    def build_ecological_network(self, species_threats_data):
        network = {
            'nodes': [],
            'links': [],
            'metrics': {}
        }
        
        species_set = set()
        threat_set = set()
        
        for species, threats in species_threats_data.items():
            species_set.add(species)
            for threat in threats:
                threat_set.add(threat)
        
        for species in species_set:
            network['nodes'].append({
                'id': species,
                'type': 'species',
                'group': 1,
                'size': len(species_threats_data.get(species, []))
            })
        
        for threat in threat_set:
            network['nodes'].append({
                'id': threat,
                'type': 'threat',
                'group': 2,
                'size': sum(1 for threats in species_threats_data.values() if threat in threats)
            })
        
        for species, threats in species_threats_data.items():
            for threat in threats:
                network['links'].append({
                    'source': species,
                    'target': threat,
                    'value': 1
                })
        
        network['metrics'] = {
            'species_count': len(species_set),
            'threat_count': len(threat_set),
            'connection_density': len(network['links']) / (len(species_set) * len(threat_set)) if species_set and threat_set else 0,
            'avg_threats_per_species': sum(len(threats) for threats in species_threats_data.values()) / len(species_set) if species_set else 0
        }
        
        return network
    
    def find_indirect_impacts(self, focal_species, species_threats_data):
        direct_threats = species_threats_data.get(focal_species, [])
        indirect_impacts = []
        
        for species, threats in species_threats_data.items():
            if species != focal_species:
                shared_threats = set(direct_threats) & set(threats)
                if shared_threats:
                    impact_chain = {
                        'target_species': species,
                        'shared_threats': list(shared_threats),
                        'risk_level': len(shared_threats) / max(len(direct_threats), 1),
                        'pathway': f"{focal_species} → {list(shared_threats)} → {species}"
                    }
                    indirect_impacts.append(impact_chain)
        
        indirect_impacts.sort(key=lambda x: x['risk_level'], reverse=True)
        
        return indirect_impacts[:10]
    
    def calculate_systemic_metrics(self, species_threats_data):
        if not species_threats_data:
            return {}
        
        threat_counts = {}
        for threats in species_threats_data.values():
            for threat in threats:
                threat_counts[threat] = threat_counts.get(threat, 0) + 1
        
        total_species = len(species_threats_data)
        
        vulnerability_scores = {}
        for species, threats in species_threats_data.items():
            vulnerability_scores[species] = len(threats) / 10
        
        avg_connectivity = sum(len(threats) for threats in species_threats_data.values()) / total_species if total_species > 0 else 0
        
        return {
            'most_common_threats': sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'vulnerability_distribution': vulnerability_scores,
            'network_resilience': min(1.0, avg_connectivity / 5),
            'systemic_risk_score': sum(vulnerability_scores.values()) / total_species if total_species > 0 else 0
        } 