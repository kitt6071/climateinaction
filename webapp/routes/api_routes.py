import random
from flask import Blueprint, jsonify, request
import config
from config import logger
from utils import load_data_if_needed, get_triplet_by_id
import os
import polars as pl
from pathlib import Path
from datetime import datetime
import json

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

api_bp = Blueprint('api', __name__)

@api_bp.route('/load-data', methods=['POST'])
def manual_load_data():
    try:
        if load_data_if_needed():
            return jsonify({
                "success": True,
                "message": "Data loaded successfully",
                "triplets_count": len(config.triplets_data),
                "kg_stats": {
                    "species_count": config.kg_results.get('species_count', 0) if config.kg_results else 0,
                    "threat_count": config.kg_results.get('threat_count', 0) if config.kg_results else 0
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to load data. Check if data file exists at /data/data_with_embeddings.json"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error loading data: {str(e)}"
        }), 500

@api_bp.route('/triplets', methods=['GET'])
def get_all_triplets():
    if not load_data_if_needed():
        return jsonify({"error": "No triplet data loaded, check server logs."}), 500
    
    # Remove embedding data for efficiency
    display_triplets = [
        {k: v for k, v in t.items() if k not in ['embedding', 'embedding_tensor']}
        for t in config.triplets_data
    ]
    return jsonify(display_triplets)

@api_bp.route('/similar_threats', methods=['GET'])
def find_similar_threats():
    if not config.triplets_data:
        return jsonify({"error": "No triplet data loaded, check server logs."}), 500
        
    target_triplet_id = request.args.get('id')
    if not target_triplet_id:
        return jsonify({"error": "Missing 'id' parameter for target triplet"}), 400

    target_triplet = get_triplet_by_id(target_triplet_id)
    if not target_triplet or target_triplet.get('embedding_tensor') is None:
        return jsonify({"error": "Target triplet not found or has no embedding"}), 404

    target_embedding = target_triplet['embedding_tensor']
    
    similarities = []
    for triplet in config.triplets_data:
        if triplet.get('id') == target_triplet_id or triplet.get('embedding_tensor') is None:
            continue
        
        current_embedding = triplet['embedding_tensor']
        similarity_score = torch.cosine_similarity(target_embedding, current_embedding, dim=0).item()
        
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

@api_bp.route('/threat_embeddings', methods=['GET'])
def get_threat_embeddings():
    try:
        if not config.triplets_data:
            return jsonify({'success': False, 'error': 'No triplet data loaded'}), 500
        
        # Filter out general subject terms
        general_subject_terms_to_filter = {
            'aves', 'bird', 'birds', 'afrotropical bird',
            'seabird', 'seabirds', 'waterbird', 'waterbirds',
            'passerine', 'passerines', 'raptor', 'raptors',
            'forest bird', 'forest birds'
        }
        
        threat_embeddings = []
        threat_id = 0
        valid_embeddings = 0
        invalid_embeddings = 0
        
        for triplet in config.triplets_data:
            subject = triplet.get('subject', '')
            if subject.lower() in general_subject_terms_to_filter:
                continue
                
            predicate = triplet.get('predicate', '')
            obj = triplet.get('object', '')
            
            threat_text = f"{subject} {predicate} {obj}".strip()
            if not threat_text:
                threat_text = triplet.get('threat_sentence', '') or triplet.get('predicate', '')
            
            embedding = triplet.get('embedding', [])
            
            # Validate embedding
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

@api_bp.route('/dimensionality_reduction', methods=['POST'])
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
        
        # Scale the embeddings
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
            n_neighbors = min(data.get('n_neighbors', 15), embeddings_array.shape[0] - 1)
            
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=data.get('min_dist', 0.1),
                metric=data.get('metric', 'cosine'),
                random_state=42
            )
            reduced_embeddings = reducer.fit_transform(embeddings_scaled)
            
        elif method == 'pca':
            pca = PCA(n_components=2, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings_scaled)
            
        else:
            return jsonify({'success': False, 'error': f'Unknown method: {method}'}), 400
        
        # Convert to list for JSON serialization
        reduced_embeddings_list = reduced_embeddings.tolist()
        
        return jsonify({
            'success': True,
            'reduced_embeddings': reduced_embeddings_list,
            'method': method,
            'original_dimensions': embeddings_array.shape[1],
            'reduced_dimensions': 2,
            'n_samples': embeddings_array.shape[0]
        })
        
    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/random-triplet', methods=['GET'])
def get_random_triplet():
    try:
        if not load_data_if_needed() or config.abstracts_df is None or not config.triplets_data:
            return jsonify({"success": False, "message": "Data not loaded or unavailable."}), 500

        abstract_dois = set(config.abstracts_df['doi_lower'].to_list())
        triplet_dois = {t.get('doi', '').lower().strip() for t in config.triplets_data}
        valid_dois = list(abstract_dois.intersection(triplet_dois))

        if not valid_dois:
            return jsonify({"success": False, "message": "No matching DOIs found between abstracts and triplets."}), 404

        selected_doi_lower = random.choice(valid_dois)
        
        abstract_row = config.abstracts_df.filter(pl.col("doi_lower") == selected_doi_lower).row(0, named=True)
        
        associated_triplets = [
            {k: v for k, v in t.items() if k != 'embedding_tensor'}
            for t in config.triplets_data 
            if t.get('doi', '').lower().strip() == selected_doi_lower
        ]

        abstract_text = abstract_row.get("abstract", "")
        if abstract_text:
            import re
            abstract_text = re.sub(r'<[^>]+>', '', abstract_text).strip()
            abstract_text = re.sub(r'\s+', ' ', abstract_text).strip()

        response_data = {
            "doi": abstract_row.get("doi"),
            "title": abstract_row.get("title", "Title not available"),
            "abstract": abstract_text,
            "triplets": associated_triplets
        }
        
        return jsonify({
            "success": True,
            "group": response_data
        })
        
    except Exception as e:
        logger.error(f"Error getting random triplet group: {e}", exc_info=True)
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@api_bp.route('/submit-review', methods=['POST'])
def submit_review():
    try:
        review_data = request.get_json()
        
        if not review_data:
            return jsonify({"success": False, "message": "No review data provided"}), 400
        
        required_fields = ['group_doi', 'triplets', 'rating', 'validation', 'reviewer']
        if not all(field in review_data for field in required_fields):
            return jsonify({"success": False, "message": "Missing required fields for group review"}), 400
        
        if not isinstance(review_data['triplets'], list) or len(review_data['triplets']) == 0:
            return jsonify({"success": False, "message": "Triplets must be a non-empty list"}), 400
            
        review_data['server_timestamp'] = datetime.utcnow().isoformat()
        
        reviews_dir = "/webapp/data/reviews"
        os.makedirs(reviews_dir, exist_ok=True)
        
        file_path = os.path.join(reviews_dir, "triplet_reviews.jsonl")
        
        with open(file_path, 'a') as f:
            f.write(json.dumps(review_data) + '\\n')
        
        reviewer_name = review_data.get('reviewer', {}).get('name', 'Anonymous')
        session_id = review_data.get('reviewer', {}).get('session_id', 'N/A')
        logger.info(f"Review for DOI {review_data['group_doi']} by {reviewer_name} (Session: {session_id}) saved to {file_path}")
        
        return jsonify({"success": True, "message": "Review submitted successfully"})
        
    except Exception as e:
        logger.error(f"Error submitting review: {e}", exc_info=True)
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500 