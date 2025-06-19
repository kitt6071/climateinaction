from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time
import logging
import re
import torch
import spacy
from urllib.parse import unquote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

app = Flask(__name__)
CORS(app)

DATA_PATH = "backend/data_with_embeddings.json"

class SemanticThreatAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.threat_embeddings = {}
        self.impact_embeddings = {}
        self.threat_clusters = {}
        self.impact_clusters = {}
        
    def generate_embeddings(self, texts, cache_key=None):
        if cache_key and cache_key in self.threat_embeddings:
            return self.threat_embeddings[cache_key]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        if cache_key:
            self.threat_embeddings[cache_key] = embeddings
        return embeddings
    
    def cluster_threats(self, threat_texts, method='kmeans', n_clusters=None):
        if len(threat_texts) < 2:
            return [0] * len(threat_texts), [{'label': 'Single Threat', 'keywords': [], 'size': len(threat_texts)}]
        
        embeddings = self.generate_embeddings(threat_texts, 'threats')
        
        if n_clusters is None:
            n_clusters = min(max(2, len(threat_texts) // 5), 8)
        
        cluster_labels = None
        cluster_info = []
        
        try:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(embeddings)
            elif method == 'gmm':
                clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(embeddings)
            elif method == 'hdbscan' and HDBSCAN_AVAILABLE:
                min_samples = max(2, len(threat_texts) // 10)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples, metric='euclidean')
                cluster_labels = clusterer.fit_predict(embeddings)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            elif method == 'dbscan':
                eps = self._estimate_eps(embeddings)
                clusterer = DBSCAN(eps=eps, min_samples=2)
                cluster_labels = clusterer.fit_predict(embeddings)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            else:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(embeddings)
            
            cluster_info = self._generate_cluster_info(threat_texts, cluster_labels, n_clusters)
            
            if len(set(cluster_labels)) > 1:
                silhouette_score(embeddings, cluster_labels)
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            cluster_labels = [0] * len(threat_texts)
            cluster_info = [{'label': 'All Threats', 'keywords': [], 'size': len(threat_texts)}]
        
        return cluster_labels, cluster_info
    
    def _estimate_eps(self, embeddings):
        from sklearn.neighbors import NearestNeighbors
        k = min(4, len(embeddings) - 1)
        if k <= 0:
            return 0.5
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(embeddings)
        distances, _ = neighbors.kneighbors(embeddings)
        distances = np.sort(distances[:, k-1], axis=0)
        return np.median(distances)
    
    def _generate_cluster_info(self, texts, labels, n_clusters):
        cluster_info = []
        for cluster_id in range(n_clusters):
            cluster_texts = [texts[i] for i, label in enumerate(labels) if label == cluster_id]
            if not cluster_texts:
                continue
            keywords = self._extract_cluster_keywords(cluster_texts)
            cluster_label = self._generate_cluster_label(keywords, cluster_texts)
            cluster_info.append({
                'label': cluster_label,
                'keywords': keywords[:5],
                'size': len(cluster_texts),
                'sample_threats': cluster_texts[:3]
            })
        return cluster_info
    
    def _extract_cluster_keywords(self, cluster_texts):
        if len(cluster_texts) == 1:
            text = cluster_texts[0].lower()
            words = re.findall(r'\b\w+\b', text)
            return [w for w in words if len(w) > 3][:5]
        try:
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            return keywords
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def _generate_cluster_label(self, keywords, sample_texts):
        if not keywords:
            return "Mixed Threats"
        category_patterns = {
            'Habitat Loss': ['habitat', 'deforestation', 'land', 'forest', 'development'],
            'Climate Change': ['climate', 'temperature', 'warming', 'weather', 'drought'],
            'Pollution': ['pollution', 'chemical', 'contamination', 'toxic', 'waste'],
            'Human Activity': ['human', 'anthropogenic', 'disturbance', 'recreation'],
            'Disease/Pathogens': ['disease', 'pathogen', 'infection', 'virus', 'bacteria'],
            'Invasive Species': ['invasive', 'alien', 'introduced', 'exotic'],
            'Resource Exploitation': ['hunting', 'fishing', 'harvest', 'extraction', 'logging'],
            'Infrastructure': ['development', 'urban', 'construction', 'infrastructure', 'road']
        }
        category_scores = {}
        combined_text = ' '.join(keywords + sample_texts).lower()
        for category, patterns in category_patterns.items():
            score = sum(1 for pattern in patterns if pattern in combined_text)
            if score > 0:
                category_scores[category] = score
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            meaningful_keywords = [k for k in keywords if len(k) > 3]
            return meaningful_keywords[0].title() + " Related" if meaningful_keywords else "Unclassified Threats"


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
            self.semantic_analyzer = SemanticThreatAnalyzer()
    
    def _ensure_ecological_integrator(self):
        if self.ecological_integrator is None:
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

class EcologicalKnowledgeProcessor:
    def __init__(self):
        self.iucn_threats = self._initialize_iucn_classification()
        self.iucn_stresses = self._initialize_stress_classification()
        self.impact_patterns = self._initialize_impact_patterns()
        
    def _initialize_iucn_classification(self):
        return {
            '1': {'category': 'Residential & commercial development', 'subcategories': {
                '1.1': 'Housing & urban areas',
                '1.2': 'Commercial & industrial areas',
                '1.3': 'Tourism & recreation areas'
            }},
            '2': {'category': 'Agriculture & aquaculture', 'subcategories': {
                '2.1': 'Annual & perennial non-timber crops',
                '2.2': 'Wood & pulp plantations',
                '2.3': 'Livestock farming & ranching',
                '2.4': 'Marine & freshwater aquaculture'
            }},
            '3': {'category': 'Energy production & mining', 'subcategories': {
                '3.1': 'Oil & gas drilling',
                '3.2': 'Mining & quarrying',
                '3.3': 'Renewable energy'
            }},
            '4': {'category': 'Transportation & service corridors', 'subcategories': {
                '4.1': 'Roads & railroads',
                '4.2': 'Utility & service lines',
                '4.3': 'Shipping lanes',
                '4.4': 'Flight paths'
            }},
            '5': {'category': 'Biological resource use', 'subcategories': {
                '5.1': 'Hunting & collecting terrestrial animals',
                '5.2': 'Gathering terrestrial plants',
                '5.3': 'Logging & wood harvesting',
                '5.4': 'Fishing & harvesting aquatic resources'
            }},
            '6': {'category': 'Human intrusions & disturbance', 'subcategories': {
                '6.1': 'Recreational activities',
                '6.2': 'War, civil unrest & military exercises',
                '6.3': 'Work & other activities'
            }},
            '7': {'category': 'Natural system modifications', 'subcategories': {
                '7.1': 'Fire & fire suppression',
                '7.2': 'Dams & water management/use',
                '7.3': 'Other ecosystem modifications'
            }},
            '8': {'category': 'Invasive & other problematic species', 'subcategories': {
                '8.1': 'Invasive non-native/alien species/diseases',
                '8.2': 'Problematic native species/diseases',
                '8.3': 'Introduced genetic material',
                '8.4': 'Problematic species/diseases of unknown origin',
                '8.5': 'Viral/prion-induced diseases',
                '8.6': 'Diseases of unknown cause'
            }},
            '9': {'category': 'Pollution', 'subcategories': {
                '9.1': 'Domestic & urban waste water',
                '9.2': 'Industrial & military effluents',
                '9.3': 'Agricultural & forestry effluents',
                '9.4': 'Garbage & solid waste',
                '9.5': 'Air-borne pollutants',
                '9.6': 'Excess energy'
            }},
            '10': {'category': 'Geological events', 'subcategories': {
                '10.1': 'Volcanoes',
                '10.2': 'Earthquakes/tsunamis',
                '10.3': 'Avalanches/landslides'
            }},
            '11': {'category': 'Climate change & severe weather', 'subcategories': {
                '11.1': 'Habitat shifting & alteration',
                '11.2': 'Droughts',
                '11.3': 'Temperature extremes',
                '11.4': 'Storms & flooding',
                '11.5': 'Other impacts'
            }},
            '12': {'category': 'Other options', 'subcategories': {
                '12.1': 'Other threat'
            }}
        }
    
    def _initialize_stress_classification(self):
        return {
            'ecosystem_conversion': 'Complete habitat loss',
            'ecosystem_degradation': 'Habitat quality decline',
            'indirect_ecosystem_effects': 'Secondary habitat impacts',
            'species_mortality': 'Direct species killing',
            'species_disturbance': 'Behavioral disruption',
            'reduced_reproductive_success': 'Breeding impacts',
            'reduced_recruit_survival': 'Juvenile survival',
            'competition': 'Interspecific competition',
            'predation': 'Predation pressure',
            'poisoning': 'Toxic exposure',
            'disease': 'Pathogenic impacts',
            'genetic_effects': 'Genetic diversity loss',
            'hybridization': 'Genetic pollution'
        }
    
    def _initialize_impact_patterns(self):
        return {
            'magnitude_patterns': [
                r'\b(severe|major|significant|substantial|extensive|massive|dramatic)\b',
                r'\b(moderate|limited|minor|slight|small|reduced)\b',
                r'\b(complete|total|entire|whole|full)\b',
                r'\b(partial|some|certain|specific)\b'
            ],
            'causality_patterns': [
                r'\b(cause[sd]?|lead[s]? to|result[s]? in|trigger[s]?|induce[s]?)\b',
                r'\b(due to|because of|as a result of|owing to)\b',
                r'\b(contribute[s]? to|influence[s]?|affect[s]?|impact[s]?)\b'
            ],
            'temporal_patterns': [
                r'\b(immediate|instant|rapid|quick|sudden)\b',
                r'\b(gradual|slow|progressive|chronic|long-term)\b',
                r'\b(annual|seasonal|periodic|cyclic)\b',
                r'\b(historic|past|recent|current|ongoing)\b'
            ],
            'directness_patterns': [
                r'\b(direct[ly]?|immediate[ly]?|straight|explicit)\b',
                r'\b(indirect[ly]?|secondary|consequent|mediated)\b',
                r'\b(cascading|knock-on|ripple|downstream)\b'
            ]
        }

    def classify_threat_to_iucn(self, threat_text):
        threat_lower = threat_text.lower()
        
        category_keywords = {
            '1': ['urban', 'housing', 'development', 'commercial', 'industrial', 'tourism', 'recreation'],
            '2': ['agriculture', 'farming', 'livestock', 'aquaculture', 'plantation', 'crops'],
            '3': ['mining', 'oil', 'gas', 'drilling', 'renewable', 'energy', 'quarrying'],
            '4': ['road', 'railroad', 'transport', 'shipping', 'utility', 'corridor'],
            '5': ['hunting', 'fishing', 'harvesting', 'logging', 'collecting', 'exploitation'],
            '6': ['recreation', 'disturbance', 'human', 'war', 'military'],
            '7': ['fire', 'dam', 'water management', 'ecosystem modification'],
            '8': ['invasive', 'alien', 'disease', 'pathogen', 'introduced'],
            '9': ['pollution', 'contamination', 'waste', 'chemical', 'toxic'],
            '10': ['volcano', 'earthquake', 'landslide', 'geological'],
            '11': ['climate', 'temperature', 'weather', 'drought', 'storm', 'flood'],
            '12': ['other', 'unknown', 'unspecified']
        }
        
        best_match = None
        best_score = 0
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in threat_lower)
            if score > best_score:
                best_score = score
                best_match = category
        
        confidence = min(best_score / 3.0, 1.0)
        
        return {
            'category': best_match,
            'confidence': confidence,
            'category_name': self.iucn_threats.get(best_match, {}).get('category', 'Unknown') if best_match else 'Unclassified'
        }

    def analyze_impact_statement(self, statement):
        if not statement:
            return {}
        
        statement_lower = statement.lower()
        
        magnitude = self._extract_magnitude(statement)
        
        causality = self._extract_causality(statement)
        
        temporality = self._extract_temporality(statement)
        
        directness = self._extract_directness(statement)
        
        mechanisms = self._extract_mechanisms(statement)
        
        confidence = self._calculate_analysis_confidence(statement)
        
        impact_outcomes = self._extract_impact_outcomes(statement)
        
        return {
            'magnitude': magnitude,
            'causality': causality,
            'temporality': temporality,
            'directness': directness,
            'mechanisms': mechanisms,
            'impact_outcomes': impact_outcomes,
            'confidence': confidence,
            'iucn_classification': self.classify_threat_to_iucn(statement),
            'processed_statement': statement
        }
    
    def _extract_impact_outcomes(self, text):
        text_lower = text.lower()
        outcomes = []
        
        outcome_patterns = {
            'population_decline': ['population decline', 'population decrease', 'population reduction', 'decline in population'],
            'mortality': ['mortality', 'death', 'killing', 'die', 'died', 'dies'],
            'habitat_loss': ['habitat loss', 'habitat destruction', 'habitat degradation', 'loss of habitat'],
            'breeding_failure': ['breeding failure', 'nesting failure', 'reproduction failure', 'failed breeding'],
            'displacement': ['displacement', 'forced migration', 'abandonment', 'relocate'],
            'stress': ['stress', 'physiological stress', 'behavioral stress'],
            'reduced_fitness': ['reduced fitness', 'fitness decline', 'lower fitness'],
            'extinction': ['extinction', 'extirpation', 'local extinction']
        }
        
        for outcome_type, patterns in outcome_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    outcomes.append({
                        'type': outcome_type,
                        'pattern': pattern,
                        'confidence': 0.8
                    })
                    break
        
        return outcomes
    
    def _extract_magnitude(self, text):
        high_magnitude = ['severe', 'major', 'significant', 'substantial', 'extensive', 'massive', 'dramatic', 'complete', 'total']
        medium_magnitude = ['moderate', 'noticeable', 'considerable', 'partial']
        low_magnitude = ['minor', 'slight', 'small', 'limited', 'minimal']
        
        for word in high_magnitude:
            if word in text:
                return {'level': 'high', 'indicators': [word]}
        for word in medium_magnitude:
            if word in text:
                return {'level': 'medium', 'indicators': [word]}
        for word in low_magnitude:
            if word in text:
                return {'level': 'low', 'indicators': [word]}
        
        return {'level': 'unknown', 'indicators': []}
    
    def _extract_causality(self, text):
        strong_causal = ['cause', 'lead to', 'result in', 'trigger', 'induce']
        weak_causal = ['contribute to', 'influence', 'affect', 'impact']
        
        for phrase in strong_causal:
            if phrase in text:
                return {'strength': 'strong', 'indicators': [phrase]}
        for phrase in weak_causal:
            if phrase in text:
                return {'strength': 'weak', 'indicators': [phrase]}
        
        return {'strength': 'unknown', 'indicators': []}
    
    def _extract_temporality(self, text):
        immediate = ['immediate', 'instant', 'rapid', 'quick', 'sudden']
        gradual = ['gradual', 'slow', 'progressive', 'chronic', 'long-term']
        periodic = ['annual', 'seasonal', 'periodic', 'cyclic']
        
        for word in immediate:
            if word in text:
                return {'pattern': 'immediate', 'indicators': [word]}
        for word in gradual:
            if word in text:
                return {'pattern': 'gradual', 'indicators': [word]}
        for word in periodic:
            if word in text:
                return {'pattern': 'periodic', 'indicators': [word]}
        
        return {'pattern': 'unknown', 'indicators': []}
    
    def _extract_directness(self, text):
        direct_indicators = ['direct', 'immediate', 'straight', 'explicit']
        indirect_indicators = ['indirect', 'secondary', 'consequent', 'mediated', 'cascading', 'knock-on', 'ripple', 'downstream']
        
        for word in direct_indicators:
            if word in text:
                return {'type': 'direct', 'confidence': 0.8, 'indicators': [word]}
        for word in indirect_indicators:
            if word in text:
                return {'type': 'indirect', 'confidence': 0.8, 'indicators': [word]}
        
        if any(phrase in text for phrase in ['kill', 'death', 'mortality', 'destroy']):
            return {'type': 'direct', 'confidence': 0.6, 'indicators': ['implicit']}
        
        return {'type': 'ambiguous', 'confidence': 0.3, 'indicators': []}
    
    def _extract_mechanisms(self, text):
        mechanisms = {
            'habitat_loss': ['habitat loss', 'deforestation', 'destruction', 'clearance'],
            'pollution': ['pollution', 'contamination', 'chemical', 'toxic', 'pesticide'],
            'climate_change': ['temperature', 'warming', 'climate', 'weather', 'precipitation'],
            'disease': ['disease', 'pathogen', 'virus', 'infection', 'parasite'],
            'competition': ['competition', 'compete', 'displacement', 'outcompete'],
            'predation': ['predation', 'predator', 'prey', 'hunting', 'consumption'],
            'disturbance': ['disturbance', 'noise', 'light', 'traffic', 'human activity']
        }
        
        identified_mechanisms = []
        for mechanism, keywords in mechanisms.items():
            if any(keyword in text for keyword in keywords):
                identified_mechanisms.append(mechanism)
        
        return identified_mechanisms
    
    def _calculate_analysis_confidence(self, statement):
        if not statement:
            return 0.0
        
        confidence_factors = [
            len(statement.split()) > 5,
            any(word in statement.lower() for word in ['study', 'research', 'observed', 'measured']),
            bool(re.search(r'\d+', statement)),
            any(word in statement.lower() for word in ['significant', 'p <', 'correlation', 'analysis'])
        ]
        
        base_confidence = 0.3
        bonus = sum(confidence_factors) * 0.15
        
        return min(base_confidence + bonus, 1.0)

class SpeciesAnalyzer:
    def __init__(self, triplets_data):
        self.triplets_data = triplets_data
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

logger.info(f"Loading data from {DATA_PATH}...")
try:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        app_data = json.load(f)
    triplets_data = app_data.get("triplets", [])
    for triplet in triplets_data:
        if 'embedding' in triplet and triplet['embedding'] is not None:
            triplet['embedding_tensor'] = torch.tensor(triplet['embedding'])
        else:
            triplet['embedding_tensor'] = None
    logger.info(f"Data loaded: {len(triplets_data)} triplets available.")
except FileNotFoundError:
    logger.error(f"Data file not found: {DATA_PATH}")
    triplets_data = []
except json.JSONDecodeError:
    logger.error(f"Could not decode JSON from {DATA_PATH}")
    triplets_data = []

logger.info("Initializing knowledge graph...")
enhanced_kg = EnhancedKnowledgeGraph()
kg_results = enhanced_kg.build_enriched_graph(triplets_data, load_ecological=False)
logger.info("Knowledge graph initialized.")

analyzer = SpeciesAnalyzer(triplets_data)
logger.info("SpeciesAnalyzer initialized.")

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

def get_triplet_by_id(triplet_id):
    for triplet in triplets_data:
        if triplet.get('id') == triplet_id:
            return triplet
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/triplets', methods=['GET'])
def get_all_triplets():
    if not triplets_data:
        return jsonify({"error": "No triplet data loaded, check server logs."}), 500
    display_triplets = [
        {k: v for k, v in t.items() if k not in ['embedding', 'embedding_tensor']}
        for t in triplets_data
    ]
    return jsonify(display_triplets)

@app.route('/api/similar_threats', methods=['GET'])
def find_similar_threats():
    if not triplets_data:
        return jsonify({"error": "No triplet data loaded, check server logs."}), 500
        
    target_triplet_id = request.args.get('id')
    if not target_triplet_id:
        return jsonify({"error": "Missing 'id' parameter for target triplet"}), 400

    target_triplet = get_triplet_by_id(target_triplet_id)
    if not target_triplet or target_triplet.get('embedding_tensor') is None:
        return jsonify({"error": "Target triplet not found or has no embedding"}), 404

    target_embedding = target_triplet['embedding_tensor']
    
    similarities = []
    for triplet in triplets_data:
        if triplet.get('id') == target_triplet_id or triplet.get('embedding_tensor') is None:
            continue
        
        current_embedding = triplet['embedding_tensor']
        similarity_score = util.cos_sim(target_embedding, current_embedding).item()
        
        similarities.append({
            "id": triplet.get('id'),
            "threat_sentence": triplet.get('threat_sentence'),
            "subject": triplet.get('subject'),
            "predicate": triplet.get('predicate'),
            "object": triplet.get('object'),
            "doi": triplet.get('doi'),
            "score": similarity_score
        })
    
    similarities.sort(key=lambda x: x['score'], reverse=True)
    top_n = int(request.args.get('top_n', 5))

    return jsonify(similarities[:top_n])

@app.route('/api/species_analysis', methods=['POST'])
def analyze_species():
    try:
        data = request.get_json()
        species_name = data.get('species_name')
        
        if not species_name:
            return jsonify({'error': 'Species name required'}), 400
        
        analysis_result = analyzer.analyze_species(species_name)
        
        species_threats = []
        for triplet in triplets_data:
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
        
        species_triplets = [t for t in triplets_data if t.get('subject', '').lower() == species_name.lower()]
        
        for triplet in species_triplets:
            threat_text = triplet.get('object', '')
            impact_text = triplet.get('predicate', '').lower()
            category = analyzer.categorize_threat_node(threat_text)
            
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

systemic_analyzer = SystemicRiskAnalyzer()

@app.route('/api/network_analysis', methods=['POST'])
def network_analysis():
    try:
        data = request.get_json()
        analysis_type = data.get('analysis_type', 'shared_threats')
        species_list = data.get('species_list', [])
        
        if not triplets_data:
            return jsonify({'success': False, 'error': 'No triplet data loaded'}), 500
        
        species_threats_data = {}
        
        if not species_list:
            species_list = list(set([triplet.get('subject', '') for triplet in triplets_data]))
        
        for triplet in triplets_data:
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
        
        if analysis_type == 'shared_threats':
            network = systemic_analyzer.build_ecological_network(species_threats_data)
        else:
            network = systemic_analyzer.build_ecological_network(species_threats_data)
        
        return jsonify({
            'success': True,
            'network': network,
            'analysis_type': analysis_type,
            'species_count': len(species_threats_data),
            'species_included': list(species_threats_data.keys())
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/indirect_impacts', methods=['POST'])
def indirect_impacts():
    try:
        data = request.get_json()
        focal_species = data.get('focal_species')
        
        if not focal_species:
            return jsonify({'success': False, 'error': 'Focal species required'}), 400
        
        species_threats_data = {}
        for triplet in triplets_data:
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

@app.route('/api/knowledge_graph_query', methods=['POST'])
def knowledge_graph_query():
    try:
        data = request.get_json()
        query_type = data.get('query_type')
        custom_query = data.get('custom_query', '')
        
        if query_type == 'shared_threats':
            species_threats = {}
            for triplet in triplets_data:
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
            threats = list(set([t.get('object', '') for t in triplets_data if t.get('object')]))
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
            graph = analyzer.enhanced_kg.graph
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

@app.route('/api/systemic_metrics', methods=['GET'])
def systemic_metrics():
    try:
        species_threats_data = {}
        for triplet in triplets_data:
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

@app.route('/api/threat_embeddings', methods=['GET'])
def get_threat_embeddings():
    try:
        if not triplets_data:
            return jsonify({'success': False, 'error': 'No triplet data loaded'}), 500
        
        general_subject_terms_to_filter = {'aves', 'bird', 'birds', 'afrotropical bird',
                                          'seabird', 'seabirds', 'waterbird', 'waterbirds',
                                          'passerine', 'passerines', 'raptor', 'raptors',
                                          'forest bird', 'forest birds'}
        
        threat_embeddings = []
        threat_id = 0
        valid_embeddings = 0
        invalid_embeddings = 0
        filtered_count = 0
        
        for triplet in triplets_data:
            subject = triplet.get('subject', '')
            if subject.lower() in general_subject_terms_to_filter:
                filtered_count += 1
                continue
                
            predicate = triplet.get('predicate', '')
            obj = triplet.get('object', '')
            
            threat_text = f"{subject} {predicate} {obj}".strip()
            
            if not threat_text or threat_text == "":
                threat_text = triplet.get('threat_sentence', '') or triplet.get('predicate', '')
            
            embedding = triplet.get('embedding', [])
            
            if isinstance(embedding, str):
                try:
                    import json
                    embedding = json.loads(embedding)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse embedding string for threat {threat_id}: {e}")
                    embedding = []
            
            valid_embedding = (
                embedding and 
                isinstance(embedding, list) and 
                len(embedding) > 0 and 
                all(isinstance(x, (int, float)) and not (isinstance(x, float) and (x != x or x == float('inf') or x == float('-inf'))) for x in embedding)
            )
            
            if valid_embedding and threat_text.strip():
                threat_embeddings.append({
                    'id': threat_id,
                    'text': threat_text,
                    'embedding': embedding,
                    'species': triplet.get('subject', ''),
                    'impact': triplet.get('object', ''),
                    'predicate': triplet.get('predicate', ''),
                    'category': triplet.get('category', 'Unknown'),
                    'doi': triplet.get('doi', '')
                })
                valid_embeddings += 1
            else:
                invalid_embeddings += 1
                
            threat_id += 1
        
        if len(threat_embeddings) == 0:
            return jsonify({'success': False, 'error': 'No valid embeddings found in data'}), 500
        
        return jsonify({
            'success': True,
            'embeddings': threat_embeddings,
            'total_count': len(threat_embeddings),
            'valid_count': valid_embeddings,
            'invalid_count': invalid_embeddings
        })
        
    except Exception as e:
        logger.error(f"Error in get_threat_embeddings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dimensionality_reduction', methods=['POST'])
def perform_dimensionality_reduction():
    try:
        data = request.get_json()
        embeddings = data.get('embeddings', [])
        method = data.get('method', 'tsne').lower()
        
        if not embeddings:
            return jsonify({'success': False, 'error': 'No embeddings provided'}), 400
        
        embeddings_array = np.array(embeddings)
        
        if embeddings_array.shape[0] < 2:
            return jsonify({'success': False, 'error': 'Need at least 2 embeddings for dimensionality reduction'}), 400
        
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_array)
        
        if method == 'tsne':
            perplexity = min(data.get('perplexity', 30), embeddings_array.shape[0] - 1)
            
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                n_iter=1000,
                learning_rate='auto',
                init='pca'
            )
            reduced_embeddings = tsne.fit_transform(embeddings_scaled)
            
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                return jsonify({'success': False, 'error': 'UMAP not available. Install with: pip install umap-learn'}), 400
            
            n_neighbors = min(data.get('n_neighbors', 15), embeddings_array.shape[0] - 1)
            min_dist = data.get('min_dist', 0.1)
            
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
                metric='cosine'
            )
            reduced_embeddings = reducer.fit_transform(embeddings_array)
            
        elif method == 'pca':
            pca = PCA(n_components=2, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings_scaled)
            
        else:
            return jsonify({'success': False, 'error': f'Unknown method: {method}. Use tsne, umap, or pca'}), 400
        
        reduced_list = reduced_embeddings.tolist()
        
        return jsonify({
            'success': True,
            'method': method,
            'reduced_embeddings': reduced_list,
            'original_count': embeddings_array.shape[0],
            'original_dimensions': embeddings_array.shape[1],
            'reduced_dimensions': 2
        })
        
    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/species-deepdive/<species_name>')
def get_species_deepdive_analysis(species_name):
    try:
        species_name = unquote(species_name)
        
        if not triplets_data:
            return jsonify({'error': 'No triplet data loaded, check server logs.'}), 500
        
        df = pd.DataFrame(triplets_data)
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        species_data = df[df['subject'].str.contains(species_name, case=False, na=False)]
        
        if species_data.empty:
            return jsonify({'error': f'No data found for species: {species_name}'}), 404
        
        kg_processor = EcologicalKnowledgeProcessor()
        
        analysis = build_enhanced_species_analysis(species_data, kg_processor)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error in species deepdive analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

def build_enhanced_species_analysis(species_data, kg_processor):
    metadata = extract_species_metadata(species_data)
    
    kg = build_species_knowledge_graph(species_data, kg_processor)
    
    nlp_analysis = perform_comprehensive_nlp_analysis(species_data, kg_processor)
    
    network_analysis = analyze_kg_network_structure(kg)
    
    ontology_mapping = map_to_ontologies(species_data, kg_processor)
    
    knowledge_gaps = identify_knowledge_gaps(species_data, nlp_analysis, ontology_mapping)
    
    causal_chains = extract_causal_chains(kg, nlp_analysis)
    
    temporal_patterns = analyze_temporal_patterns(species_data, nlp_analysis)
    
    return {
        'species': metadata['scientificName'],
        'metadata': metadata,
        'knowledgeGraph': kg,
        'impacts': nlp_analysis,
        'networkAnalysis': network_analysis,
        'ontologyMapping': ontology_mapping,
        'knowledgeGaps': knowledge_gaps,
        'causalChains': causal_chains,
        'temporalPatterns': temporal_patterns,
        'mechanisms': extract_mechanisms_analysis(nlp_analysis),
        'directness': extract_directness_analysis(nlp_analysis),
        'sources': extract_sources_analysis(species_data),
        'ecology': extract_ecological_context(species_data, kg),
        'threats': extract_threat_profile(species_data, ontology_mapping),
        'vulnerability': assess_vulnerability(species_data, nlp_analysis, ontology_mapping)
    }

def build_species_knowledge_graph(species_data, kg_processor):
    nodes = {}
    edges = []
    node_types = defaultdict(int)
    
    for _, row in species_data.iterrows():
        subject = row['subject']
        predicate = row['predicate']
        threat = row['object']
        doi = row.get('doi', '')
        
        species_id = f"species_{subject.replace(' ', '_')}"
        if species_id not in nodes:
            nodes[species_id] = {
                'id': species_id,
                'type': 'Species',
                'properties': {
                    'scientificName': subject,
                    'displayName': subject
                }
            }
            node_types['Species'] += 1
        
        threat_id = f"threat_{threat.replace(' ', '_')[:50]}"
        if threat_id not in nodes:
            threat_classification = kg_processor.classify_threat_to_iucn(threat)
            nodes[threat_id] = {
                'id': threat_id,
                'type': 'Threat',
                'properties': {
                    'threatName': threat,
                    'iucnCategory': threat_classification.get('category', 'unknown'),
                    'iucnCategoryName': threat_classification.get('category_name', 'Unknown'),
                    'classificationConfidence': threat_classification.get('confidence', 0.0)
                }
            }
            node_types['Threat'] += 1
        
        impact_id = f"impact_{len(nodes)}"
        impact_analysis = kg_processor.analyze_impact_statement(predicate)
        nodes[impact_id] = {
            'id': impact_id,
            'type': 'ImpactStatement',
            'properties': {
                'predicate': predicate,
                'magnitude': impact_analysis.get('magnitude', {}),
                'causality': impact_analysis.get('causality', {}),
                'temporality': impact_analysis.get('temporality', {}),
                'directness': impact_analysis.get('directness', {}),
                'mechanisms': impact_analysis.get('mechanisms', []),
                'confidence': impact_analysis.get('confidence', 0.0),
                'doi': doi
            }
        }
        node_types['ImpactStatement'] += 1
        
        edges.append({
            'source': species_id,
            'target': impact_id,
            'type': 'experiences',
            'properties': {'strength': impact_analysis.get('confidence', 0.5)}
        })
        
        edges.append({
            'source': impact_id,
            'target': threat_id,
            'type': 'caused_by',
            'properties': {
                'directness': impact_analysis.get('directness', {}).get('type', 'ambiguous'),
                'strength': impact_analysis.get('causality', {}).get('strength', 'unknown')
            }
        })
    
    graph_metrics = calculate_graph_metrics(nodes, edges)
    
    return {
        'nodes': list(nodes.values()),
        'edges': edges,
        'nodeCount': len(nodes),
        'edgeCount': len(edges),
        'nodeTypes': dict(node_types),
        'connectivity': graph_metrics
    }

def calculate_graph_metrics(nodes, edges):
    if not nodes or not edges:
        return {'averageDegree': 0, 'density': 0}
    
    degree_count = defaultdict(int)
    for edge in edges:
        degree_count[edge['source']] += 1
        degree_count[edge['target']] += 1
    
    avg_degree = sum(degree_count.values()) / len(nodes) if nodes else 0
    max_edges = len(nodes) * (len(nodes) - 1)
    density = len(edges) / max_edges if max_edges > 0 else 0
    
    return {
        'averageDegree': avg_degree,
        'density': density
    }

def perform_comprehensive_nlp_analysis(species_data, kg_processor):
    impacts = {
        'directness': {'direct': 0, 'indirect': 0, 'ambiguous': 0},
        'confidence': {'high': 0, 'medium': 0, 'low': 0},
        'magnitude': {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0},
        'temporality': {'immediate': 0, 'gradual': 0, 'periodic': 0, 'unknown': 0},
        'mechanisms': defaultdict(int),
        'categories': defaultdict(int)
    }
    
    detailed_analyses = []
    
    for _, row in species_data.iterrows():
        predicate = row['predicate']
        analysis = kg_processor.analyze_impact_statement(predicate)
        detailed_analyses.append(analysis)
        
        directness = analysis.get('directness', {}).get('type', 'ambiguous')
        impacts['directness'][directness] += 1
        
        confidence = analysis.get('confidence', 0)
        if confidence > 0.7:
            impacts['confidence']['high'] += 1
        elif confidence > 0.4:
            impacts['confidence']['medium'] += 1
        else:
            impacts['confidence']['low'] += 1
        
        magnitude = analysis.get('magnitude', {}).get('level', 'unknown')
        impacts['magnitude'][magnitude] += 1
        
        temporality = analysis.get('temporality', {}).get('pattern', 'unknown')
        impacts['temporality'][temporality] += 1
        
        for mechanism in analysis.get('mechanisms', []):
            impacts['mechanisms'][mechanism] += 1
        
        if 'death' in predicate.lower() or 'mortality' in predicate.lower():
            impacts['categories']['mortality'] += 1
        elif 'habitat' in predicate.lower():
            impacts['categories']['habitat_change'] += 1
        elif 'breed' in predicate.lower() or 'reproduction' in predicate.lower():
            impacts['categories']['reproductive'] += 1
        else:
            impacts['categories']['other'] += 1
    
    return impacts

def analyze_kg_network_structure(kg):
    nodes = kg['nodes']
    edges = kg['edges']
    
    if not nodes or not edges:
        return {
            'pathways': [],
            'centralNodes': [],
            'clusters': [],
            'connectivity': 0
        }
    
    adjacency = defaultdict(list)
    for edge in edges:
        adjacency[edge['source']].append(edge['target'])
        adjacency[edge['target']].append(edge['source'])
    
    pathways = []
    species_nodes = [n for n in nodes if n['type'] == 'Species']
    threat_nodes = [n for n in nodes if n['type'] == 'Threat']
    
    for species in species_nodes:
        for threat in threat_nodes:
            path = find_simple_path(adjacency, species['id'], threat['id'])
            if path:
                pathways.append({
                    'source': species['id'],
                    'target': threat['id'],
                    'path': path,
                    'length': len(path) - 1
                })
    
    degree_count = defaultdict(int)
    for edge in edges:
        degree_count[edge['source']] += 1
        degree_count[edge['target']] += 1
    
    central_nodes = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)[:5]
    central_nodes = [{'nodeId': node_id, 'degree': degree} for node_id, degree in central_nodes]
    
    return {
        'pathways': pathways,
        'centralNodes': central_nodes,
        'clusters': [],
        'connectivity': kg['connectivity']['averageDegree']
    }

def find_simple_path(graph, start, end, visited=None):
    if visited is None:
        visited = set()
    
    if start == end:
        return [start]
    
    if start in visited:
        return None
    
    visited.add(start)
    
    for neighbor in graph.get(start, []):
        path = find_simple_path(graph, neighbor, end, visited.copy())
        if path:
            return [start] + path
    
    return None

def map_to_ontologies(species_data, kg_processor):
    iucn_threats = defaultdict(lambda: {'category': '', 'threats': [], 'count': 0})
    stress_categories = defaultdict(int)
    unmapped_threats = []
    total_threats = len(species_data)
    mapped_count = 0
    
    for _, row in species_data.iterrows():
        threat = row['object']
        predicate = row['predicate']
        
        classification = kg_processor.classify_threat_to_iucn(threat)
        
        if classification['category'] and classification['confidence'] > 0.3:
            category = classification['category']
            iucn_threats[category]['category'] = classification['category_name']
            iucn_threats[category]['threats'].append(threat)
            iucn_threats[category]['count'] += 1
            mapped_count += 1
        else:
            unmapped_threats.append({
                'description': threat,
                'predicate': predicate
            })
        
        impact_analysis = kg_processor.analyze_impact_statement(predicate)
        mechanisms = impact_analysis.get('mechanisms', [])
        
        for mechanism in mechanisms:
            if mechanism in kg_processor.iucn_stresses:
                stress_categories[mechanism] += 1
        
        predicate_lower = predicate.lower()
        if 'mortality' in predicate_lower or 'death' in predicate_lower:
            stress_categories['species_mortality'] += 1
        elif 'habitat' in predicate_lower:
            if 'loss' in predicate_lower or 'destruction' in predicate_lower:
                stress_categories['ecosystem_conversion'] += 1
            else:
                stress_categories['ecosystem_degradation'] += 1
        elif 'disturb' in predicate_lower:
            stress_categories['species_disturbance'] += 1
        elif 'breed' in predicate_lower or 'reproduction' in predicate_lower:
            stress_categories['reduced_reproductive_success'] += 1
    
    mapping_confidence = mapped_count / total_threats if total_threats > 0 else 0
    
    return {
        'iucnThreats': dict(iucn_threats),
        'stressCategories': dict(stress_categories),
        'unmappedThreats': unmapped_threats,
        'mappingConfidence': mapping_confidence,
        'totalThreats': total_threats,
        'mappedThreats': mapped_count
    }

def identify_knowledge_gaps(species_data, nlp_analysis, ontology_mapping):
    gaps = {
        'missingMagnitudes': 0,
        'lowConfidenceStatements': 0,
        'unmappedThreats': len(ontology_mapping['unmappedThreats']),
        'temporalGaps': 0,
        'mechanismGaps': 0,
        'suggestions': []
    }
    
    for _, row in species_data.iterrows():
        predicate = row['predicate']
        
        if not any(word in predicate.lower() for word in ['severe', 'major', 'minor', 'significant', 'substantial', 'complete', 'partial']):
            gaps['missingMagnitudes'] += 1
        
        if not any(word in predicate.lower() for word in ['immediate', 'gradual', 'long-term', 'short-term', 'chronic', 'acute']):
            gaps['temporalGaps'] += 1
        
        if not any(word in predicate.lower() for word in ['habitat', 'pollution', 'disease', 'predation', 'competition']):
            gaps['mechanismGaps'] += 1
    
    gaps['lowConfidenceStatements'] = nlp_analysis['confidence']['low']
    
    suggestions = []
    
    if gaps['missingMagnitudes'] > len(species_data) * 0.5:
        suggestions.append("Consider quantifying impact magnitudes in threat assessments")
    
    if gaps['temporalGaps'] > len(species_data) * 0.7:
        suggestions.append("Add temporal context to threat descriptions (immediate vs long-term)")
    
    if gaps['unmappedThreats'] > len(species_data) * 0.3:
        suggestions.append("Standardize threat terminology using IUCN classification")
    
    if gaps['mechanismGaps'] > len(species_data) * 0.6:
        suggestions.append("Include biological mechanisms in impact descriptions")
    
    if gaps['lowConfidenceStatements'] > len(species_data) * 0.4:
        suggestions.append("Improve statement specificity and include quantitative data")
    
    gaps['suggestions'] = suggestions
    
    return gaps

def extract_causal_chains(kg, nlp_analysis):
    causal_chains = []
    
    nodes = kg['nodes']
    edges = kg['edges']
    
    species_nodes = {n['id']: n for n in nodes if n['type'] == 'Species'}
    impact_nodes = {n['id']: n for n in nodes if n['type'] == 'ImpactStatement'}
    threat_nodes = {n['id']: n for n in nodes if n['type'] == 'Threat'}
    
    for edge1 in edges:
        if edge1['type'] == 'experiences' and edge1['source'] in species_nodes:
            for edge2 in edges:
                if (edge2['type'] == 'caused_by' and 
                    edge2['source'] == edge1['target'] and 
                    edge2['target'] in threat_nodes):
                    
                    species = species_nodes[edge1['source']]
                    impact = impact_nodes[edge1['target']]
                    threat = threat_nodes[edge2['target']]
                    
                    confidence = (
                        edge1['properties'].get('strength', 0.5) * 
                        impact['properties'].get('confidence', 0.5)
                    )
                    
                    causal_chains.append({
                        'cause': species,
                        'impact': impact,
                        'effect': threat,
                        'confidence': confidence,
                        'directness': impact['properties'].get('directness', {}).get('type', 'ambiguous')
                    })
    
    causal_chains.sort(key=lambda x: x['confidence'], reverse=True)
    
    return causal_chains[:10]

def analyze_temporal_patterns(species_data, nlp_analysis):
    years = []
    for _, row in species_data.iterrows():
        doi = row.get('doi', '')
        if doi:
            year_match = re.search(r'20\d{2}', doi)
            if year_match:
                years.append(int(year_match.group()))
    
    temporal_analysis = {
        'temporalityDistribution': dict(nlp_analysis['temporality']),
        'publicationYears': years,
        'yearRange': {
            'min': min(years) if years else None,
            'max': max(years) if years else None,
            'span': max(years) - min(years) if years else 0
        },
        'trends': []
    }
    
    return temporal_analysis

def extract_mechanisms_analysis(nlp_analysis):
    return {
        'uniqueMechanisms': len(nlp_analysis['mechanisms']),
        'mechanismDistribution': dict(nlp_analysis['mechanisms']),
        'dominantMechanism': max(nlp_analysis['mechanisms'].items(), key=lambda x: x[1])[0] if nlp_analysis['mechanisms'] else None
    }

def extract_directness_analysis(nlp_analysis):
    directness = nlp_analysis['directness']
    total = sum(directness.values())
    
    return {
        'summary': {
            'directCount': directness['direct'],
            'indirectCount': directness['indirect'],
            'uncertainCount': directness['ambiguous'],
            'directPercentage': (directness['direct'] / total * 100) if total > 0 else 0
        },
        'distribution': directness
    }

def extract_sources_analysis(species_data):
    dois = species_data['doi'].dropna().unique() if 'doi' in species_data.columns else []
    
    years = []
    for doi in dois:
        year_match = re.search(r'20\d{2}', str(doi))
        if year_match:
            years.append(int(year_match.group()))
    
    return {
        'sourceCount': len(dois),
        'uniqueSources': len(dois),
        'yearRange': {
            'min': min(years) if years else None,
            'max': max(years) if years else None,
            'span': max(years) - min(years) if years else 0
        } if years else None
    }

def extract_ecological_context(species_data, kg):
    habitat_mentions = 0
    ecosystem_types = set()
    
    for _, row in species_data.iterrows():
        predicate = row['predicate'].lower()
        
        if any(term in predicate for term in ['habitat', 'ecosystem', 'forest', 'marine', 'terrestrial']):
            habitat_mentions += 1
        
        if 'forest' in predicate:
            ecosystem_types.add('forest')
        if 'marine' in predicate or 'ocean' in predicate:
            ecosystem_types.add('marine')
        if 'freshwater' in predicate or 'river' in predicate:
            ecosystem_types.add('freshwater')
    
    return {
        'habitatMentions': habitat_mentions,
        'ecosystemTypes': list(ecosystem_types),
        'habitatDiversity': len(ecosystem_types),
        'ecologicalComplexity': 'high' if len(ecosystem_types) > 2 else 'medium' if len(ecosystem_types) > 0 else 'low'
    }

def extract_threat_profile(species_data, ontology_mapping):
    threats = species_data['object'].value_counts().to_dict()
    
    return {
        'totalThreats': len(threats),
        'topThreats': dict(list(threats.items())[:5]),
        'threatDiversity': len(threats),
        'iucnCoverage': len(ontology_mapping['iucnThreats']),
        'unmappedThreats': len(ontology_mapping['unmappedThreats'])
    }

def assess_vulnerability(species_data, nlp_analysis, ontology_mapping):
    factors = {
        'threatDiversity': min(len(species_data['object'].unique()) / 10, 1.0),
        'impactSeverity': nlp_analysis['magnitude'].get('high', 0) / len(species_data) if len(species_data) > 0 else 0,
        'directnessRatio': nlp_analysis['directness'].get('direct', 0) / len(species_data) if len(species_data) > 0 else 0,
        'knowledgeConfidence': nlp_analysis['confidence'].get('high', 0) / len(species_data) if len(species_data) > 0 else 0,
        'iucnMapping': ontology_mapping['mappingConfidence']
    }
    
    overall_score = sum(factors.values()) / len(factors) if len(factors) > 0 else 0
    
    if overall_score > 0.7:
        vulnerability_level = 'High'
    elif overall_score > 0.4:
        vulnerability_level = 'Medium'
    else:
        vulnerability_level = 'Low'
    
    return {
        'overall': vulnerability_level,
        'score': overall_score,
        'factors': factors,
        'recommendations': generate_vulnerability_recommendations(factors)
    }

def generate_vulnerability_recommendations(factors):
    recommendations = []
    
    if factors['threatDiversity'] > 0.8:
        recommendations.append("High threat diversity indicates need for comprehensive conservation strategy")
    
    if factors['impactSeverity'] > 0.7:
        recommendations.append("Severe impacts detected - immediate intervention may be required")
    
    if factors['knowledgeConfidence'] < 0.5:
        recommendations.append("Low confidence in impact assessments - additional research needed")
    
    if factors['iucnMapping'] < 0.6:
        recommendations.append("Improve threat classification using standardized taxonomies")
    
    return recommendations

def extract_species_metadata(species_data):
    scientific_name = species_data['subject'].iloc[0] if not species_data.empty else 'Unknown'
    
    return {
        'scientificName': scientific_name,
        'totalThreats': len(species_data),
        'uniqueThreats': len(species_data['object'].unique()),
        'sourceCount': len(species_data['doi'].dropna().unique()) if 'doi' in species_data.columns else 0,
        'dataPoints': len(species_data)
    }

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

@app.route('/api/species/<species_name>/deepdive')
def get_species_deepdive(species_name):
    try:
        result = analyzer.analyze_species(species_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/semantic/threat-landscape')
def get_semantic_threat_landscape():
    try:
        landscape = analyzer.enhanced_kg.get_semantic_threat_landscape()
        return jsonify(landscape)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ecological/interactions/<species_name>')
def get_species_interactions(species_name):
    try:
        integrator = analyzer.enhanced_kg.ecological_integrator
        interactions = integrator.fetch_species_interactions(species_name)
        network_analysis = integrator.analyze_interaction_network(interactions)
        
        return jsonify({
            'species': species_name,
            'interactions': interactions[:20],
            'network_analysis': network_analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/semantic/clusters/<species_name>')
def get_species_threat_clusters(species_name):
    try:
        semantic_analysis = analyzer.analyze_semantic_threats(species_name)
        return jsonify(semantic_analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge-graph/stats')
def get_knowledge_graph_stats():
    try:
        graph = analyzer.enhanced_kg.graph
        
        stats = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'species_nodes': len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'species']),
            'threat_nodes': len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'threat']),
            'threat_clusters': len(analyzer.threat_clusters),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        }
        
        cluster_summary = [
            {
                'label': cluster['label'],
                'size': cluster['size'],
                'keywords': cluster['keywords'][:3]
            }
            for cluster in analyzer.threat_clusters
        ]
        
        stats['cluster_summary'] = cluster_summary
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shared-threats/<species_name>')
def get_shared_threats(species_name):
    try:
        if not analyzer.enhanced_kg.graph.has_node(species_name):
            return jsonify({'shared_threats': [], 'message': 'Species not found in knowledge graph'})
        
        node_data = analyzer.enhanced_kg.graph.nodes[species_name]
        interactions = node_data.get('interactions', [])
        
        partners = []
        for interaction in interactions[:15]:
            partner_name = interaction.get('target_name', '') or interaction.get('source_name', '')
            if partner_name and partner_name != species_name:
                partners.append(partner_name)
        
        shared_threats = analyzer.enhanced_kg.ecological_integrator.identify_shared_threats(
            species_name, partners, analyzer.triplets_data
        )
        
        return jsonify({
            'species': species_name,
            'interaction_partners': len(partners),
            'shared_threats': shared_threats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/centrality/<species_name>')
def get_species_centrality(species_name):
    try:
        graph = analyzer.enhanced_kg.graph
        
        if not graph.has_node(species_name):
            return jsonify({'error': 'Species not found in knowledge graph'}), 404
        
        centrality_measures = {}
        
        try:
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

@app.route('/api/knowledge_transfer', methods=['POST'])
def knowledge_transfer_analysis():
    try:
        data = request.get_json()
        target_species = data.get('target_species')
        similarity_threshold = data.get('similarity_threshold', 0.7)
        min_evidence_count = data.get('min_evidence_count', 3)
        
        if not target_species:
            return jsonify({'error': 'Target species name required'}), 400
        
        target_threats = []
        target_triplets = []
        for triplet in triplets_data:
            if triplet.get('subject', '').lower() == target_species.lower():
                target_threats.append(triplet.get('object', ''))
                target_triplets.append(triplet)
        
        target_threat_set = set(target_threats)
        
        similar_species = {}
        
        for triplet in triplets_data:
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
        
        for triplet in triplets_data:
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
            
            if combined_similarity >= similarity_threshold:
                transferable_threats = []
                
                for unique_threat in data_dict['unique_threats']:
                    evidence_triplets = [t for t in data_dict['triplets'] if t.get('object') == unique_threat]
                    
                    if len(evidence_triplets) >= min_evidence_count:
                        threat_analysis = analyze_threat_transferability(
                            unique_threat, evidence_triplets, target_species, target_triplets
                        )
                        
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
        
        knowledge_transfer_candidates.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
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
            }
        })
        
    except Exception as e:
        logger.error(f"Error in knowledge transfer analysis: {e}")
        return jsonify({'error': str(e)}), 500

def analyze_threat_transferability(threat, evidence_triplets, target_species, target_triplets):
    transferability_score = 0.0
    reasoning_factors = []
    
    threat_lower = threat.lower()
    if any(keyword in threat_lower for keyword in ['climate change', 'habitat loss', 'pollution', 'pesticide', 'chemical', 'contamination']):
        transferability_score += 0.3
        reasoning_factors.append("Broad environmental threat likely affects multiple species")
    
    evidence_strength = min(len(evidence_triplets) / 10.0, 0.3)
    transferability_score += evidence_strength
    reasoning_factors.append(f"Strong evidence base with {len(evidence_triplets)} documented cases")
    
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
    
    recent_studies = 0
    for triplet in evidence_triplets:
        doi = triplet.get('doi', '')
        if doi and any(year in doi for year in ['2020', '2021', '2022', '2023', '2024']):
            recent_studies += 1
    
    if recent_studies > 0:
        transferability_score += 0.1
        reasoning_factors.append(f"{recent_studies} recent studies provide current evidence")
    
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
    gaps = {
        'threat_categories_missing': [],
        'impact_mechanisms_understudied': [],
        'geographic_coverage_gaps': [],
        'temporal_coverage_gaps': [],
        'methodological_gaps': []
    }
    
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
    
    for candidate in transfer_candidates:
        for threat_info in candidate['transferable_threats']:
            threat = threat_info['threat'].lower()
            for category in missing_categories:
                if category.lower().replace(' ', '') in threat.replace(' ', ''):
                    if category not in gaps['threat_categories_missing']:
                        gaps['threat_categories_missing'].append(category)
    
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
    recommendations = []
    
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
    
    if gap_analysis['threat_categories_missing']:
        recommendations.append({
            'type': 'field_study',
            'priority': 'Medium',
            'title': 'Field Assessment of Understudied Threat Categories',
            'description': f'Conduct field studies to assess {target_species} exposure to these threat categories',
            'specific_actions': [f"Assess exposure to {category}" for category in gap_analysis['threat_categories_missing'][:3]]
        })
    
    return recommendations

if __name__ == '__main__':
    logger.info("Starting Climate Inaction Analysis Server...")
    logger.info(f"Data loaded: {len(triplets_data)} triplets.")
    if kg_results:
        logger.info(f"KG: {kg_results.get('species_count', 0)} species, {kg_results.get('threat_count', 0)} threats.")
    logger.info("Server running at http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)