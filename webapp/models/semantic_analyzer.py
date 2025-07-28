import re
import logging
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)

class SemanticThreatAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
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
            elif method == 'hdbscan':
                try:
                    import hdbscan
                    min_samples = max(2, len(threat_texts) // 10)
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples, metric='euclidean')
                    cluster_labels = clusterer.fit_predict(embeddings)
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                except ImportError:
                    logger.warning("HDBSCAN not available, falling back to KMeans")
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = clusterer.fit_predict(embeddings)
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