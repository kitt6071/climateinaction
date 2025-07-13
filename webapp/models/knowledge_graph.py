import networkx as nx
import logging

logger = logging.getLogger(__name__)

class EnhancedKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.semantic_analyzer = None
        self.ecological_integrator = None
        self.node_embeddings = {}
        self.edge_embeddings = {}
        self.ecological_data_loaded = False
        self.triplets_data = []
        self.threat_clusters_cache = None
        self.embeddings_cache = {}
    
    def build_enriched_graph(self, triplets_data, load_ecological=False):
        self.triplets_data = triplets_data
        
        species_set = set()
        threats_set = set()
        predicates_set = set()
        
        for triplet in triplets_data:
            subject = triplet.get('subject', '')
            obj = triplet.get('object', '')
            predicate = triplet.get('predicate', '')
            
            if subject:
                species_set.add(subject)
            if obj:
                threats_set.add(obj)
            if predicate:
                predicates_set.add(predicate)
        
        for i, triplet in enumerate(triplets_data):
            subject = triplet.get('subject', '')
            obj = triplet.get('object', '')
            predicate = triplet.get('predicate', '')
            
            if subject and not self.graph.has_node(subject):
                self.graph.add_node(subject, 
                                  node_type='species',
                                  scientific_name=subject,
                                  interactions=[],
                                  semantic_processed=False)
            
            if obj and not self.graph.has_node(obj):
                self.graph.add_node(obj,
                                  node_type='threat',
                                  threat_text=obj,
                                  semantic_processed=False)
            
            if subject and obj:
                self.graph.add_edge(subject, obj,
                                  predicate=predicate,
                                  doi=triplet.get('doi', ''),
                                  triplet_id=i,
                                  semantic_processed=False)
        
        return {
            'graph': self.graph,
            'threat_clusters': [],
            'embedding_dimensions': 0,
            'species_count': len(species_set),
            'threat_count': len(threats_set),
            'ecological_loaded': False,
            'semantic_processed': False
        }
    
    def _ensure_semantic_analyzer(self):
        if self.semantic_analyzer is None:
            from .semantic_analyzer import SemanticThreatAnalyzer
            self.semantic_analyzer = SemanticThreatAnalyzer()
    
    def _ensure_ecological_integrator(self):
        if self.ecological_integrator is None:
            from .ecological_integrator import EcologicalContextIntegrator
            self.ecological_integrator = EcologicalContextIntegrator()
    
    def get_threat_clusters(self, force_refresh=False):
        if self.threat_clusters_cache is None or force_refresh:
            self._ensure_semantic_analyzer()
            
            threat_texts = []
            for node, data in self.graph.nodes(data=True):
                if data.get('node_type') == 'threat':
                    threat_texts.append(data.get('threat_text', node))
            
            if threat_texts:
                try:
                    threat_clusters, threat_cluster_info = self.semantic_analyzer.cluster_threats(threat_texts)
                    self.threat_clusters_cache = threat_cluster_info
                    
                    for i, (node, data) in enumerate(self.graph.nodes(data=True)):
                        if data.get('node_type') == 'threat' and i < len(threat_clusters):
                            cluster_id = threat_clusters[i]
                            cluster_label = threat_cluster_info[cluster_id]['label'] if cluster_id < len(threat_cluster_info) else 'Unknown'
                            self.graph.nodes[node]['cluster_id'] = cluster_id
                            self.graph.nodes[node]['cluster_label'] = cluster_label
                            self.graph.nodes[node]['semantic_processed'] = True
                    
                except Exception as e:
                    logger.error(f"Error computing threat clusters: {e}")
                    self.threat_clusters_cache = []
            else:
                self.threat_clusters_cache = []
        
        return self.threat_clusters_cache
    
    def get_species_semantic_analysis(self, species_name):
        if species_name not in self.embeddings_cache:
            self._ensure_semantic_analyzer()
            
            species_threats = []
            for _, neighbor, edge_data in self.graph.edges(species_name, data=True):
                if self.graph.nodes[neighbor].get('node_type') == 'threat':
                    species_threats.append(neighbor)
            
            if species_threats:
                try:
                    embeddings = self.semantic_analyzer.generate_embeddings(species_threats, f"species_{species_name}")
                    clusters, cluster_info = self.semantic_analyzer.cluster_threats(species_threats)
                    
                    self.embeddings_cache[species_name] = {
                        'threats': species_threats,
                        'embeddings': embeddings,
                        'clusters': clusters,
                        'cluster_info': cluster_info
                    }
                except Exception as e:
                    logger.error(f"Error computing semantic analysis for {species_name}: {e}")
                    self.embeddings_cache[species_name] = {
                        'threats': species_threats,
                        'embeddings': [],
                        'clusters': [],
                        'cluster_info': []
                    }
            else:
                self.embeddings_cache[species_name] = {
                    'threats': [],
                    'embeddings': [],
                    'clusters': [],
                    'cluster_info': []
                }
        
        return self.embeddings_cache[species_name] 