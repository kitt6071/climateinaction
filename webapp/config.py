import os
import logging

# Data source configuration
DATA_SOURCES = [
    "/data/data_with_embeddings.json",  # Railway volume
    "backend/data_with_embeddings.json",  # Local file
    "https://storage.googleapis.com/climateinaction/data_with_embeddings.json"  # Google Cloud Storage
]
DATA_PATH = "data_with_embeddings.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

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

triplets_data = []
enhanced_kg = None
kg_results = None
analyzer = None
data_loaded = False

def initialize_analyzer():
    global analyzer
    if analyzer is None and triplets_data:
        from models import SpeciesAnalyzer
        analyzer = SpeciesAnalyzer(triplets_data)
