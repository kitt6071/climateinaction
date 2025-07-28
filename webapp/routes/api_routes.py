from flask import Blueprint, jsonify, request
import config
from config import logger
from utils import load_data_if_needed, get_triplet_by_id

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
        if not load_data_if_needed():
            return jsonify({
                "success": False,
                "message": "No triplet data loaded"
            }), 500
        
        if not config.triplets_data:
            return jsonify({
                "success": False,
                "message": "No triplets available"
            }), 500
        
        import random
        import polars as pl
        from pathlib import Path
        
        parquet_path = Path(config.PROJECT_ROOT) / "Lent_Init" / "shorebirds.parquet"
        logger.info(f"Attempting to load abstracts from: {parquet_path}")

        if not parquet_path.exists():
            logger.error(f"FATAL: Parquet file not found at expected path: {parquet_path}")
        
        random_triplet = None
        triplet_data = None
        abstract_found = False
        
        if parquet_path.exists():
            try:
                df = pl.read_parquet(parquet_path)
                parquet_dois = set(df['doi'].to_list())
                
                triplets_with_abstracts = [
                    t for t in config.triplets_data 
                    if t.get('doi', '') in parquet_dois
                ]
                
                if triplets_with_abstracts:
                    random_triplet = random.choice(triplets_with_abstracts)
                    logger.info(f"Selected triplet with available abstract: {random_triplet.get('doi')}")
                else:
                    random_triplet = random.choice(config.triplets_data)
                    logger.warning("No triplets with available abstracts, using random triplet")
                
                doi = random_triplet.get('doi', '')
                
                triplet_data = {
                    "subject": random_triplet.get('subject', ''),
                    "predicate": random_triplet.get('predicate', ''),
                    "object": random_triplet.get('object', ''),
                    "doi": doi,
                    "abstract": "",
                    "title": "",
                    "id": random_triplet.get('id', '')
                }
                
                if doi:
                    matching_rows = df.filter(
                        pl.col("doi").str.to_lowercase() == doi.lower()
                    )
                    
                    if len(matching_rows) > 0:
                        row = matching_rows.row(0, named=True)
                        triplet_data["abstract"] = row.get("abstract", "")
                        triplet_data["title"] = row.get("title", "")
                        abstract_found = True
                        logger.info(f"Successfully loaded abstract for DOI: {doi}")
                    else:
                        logger.warning(f"No matching row found for DOI: {doi}")
                        
            except Exception as e:
                logger.error(f"Error reading parquet file: {e}")
                random_triplet = random.choice(config.triplets_data)
                doi = random_triplet.get('doi', '')
                triplet_data = {
                    "subject": random_triplet.get('subject', ''),
                    "predicate": random_triplet.get('predicate', ''),
                    "object": random_triplet.get('object', ''),
                    "doi": doi,
                    "abstract": "",
                    "title": "",
                    "id": random_triplet.get('id', '')
                }
        else:
            random_triplet = random.choice(config.triplets_data)
            doi = random_triplet.get('doi', '')
            triplet_data = {
                "subject": random_triplet.get('subject', ''),
                "predicate": random_triplet.get('predicate', ''),
                "object": random_triplet.get('object', ''),
                "doi": doi,
                "abstract": "",
                "title": "",
                "id": random_triplet.get('id', '')
            }
        
        if triplet_data["abstract"]:
            import re
            clean_abstract = re.sub(r'<[^>]+>', '', triplet_data["abstract"])
            clean_abstract = re.sub(r'\s+', ' ', clean_abstract).strip()
            triplet_data["abstract"] = clean_abstract
        
        if not abstract_found or not triplet_data["abstract"]:
            try:
                if parquet_path.exists():
                    df = pl.read_parquet(parquet_path)
                    parquet_dois = set(df['doi'].to_list())
                    matching_triplets = len([t for t in config.triplets_data if t.get('doi', '') in parquet_dois])
                else:
                    matching_triplets = 0
                triplet_data["abstract"] = f"Abstract not available for DOI: {doi}\n\nNote: Only {matching_triplets} out of {len(config.triplets_data)} triplets have abstracts available in the database."
            except:
                triplet_data["abstract"] = f"Abstract not available for DOI: {doi}"
            triplet_data["title"] = "Title not available"
        
        return jsonify({
            "success": True,
            "triplet": triplet_data
        })
        
    except Exception as e:
        logger.error(f"Error getting random triplet: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

@api_bp.route('/submit-review', methods=['POST'])
def submit_review():
    try:
        review_data = request.get_json()
        
        if not review_data:
            return jsonify({
                "success": False,
                "message": "No review data provided"
            }), 400
        
        required_fields = ['triplet', 'rating']
        for field in required_fields:
            if field not in review_data:
                return jsonify({
                    "success": False,
                    "message": f"Missing required field: {field}"
                }), 400
        
        import json
        import os
        from datetime import datetime
        
        reviews_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'reviews')
        os.makedirs(reviews_dir, exist_ok=True)
        
        reviews_file = os.path.join(reviews_dir, 'triplet_reviews.jsonl')
        review_data['server_timestamp'] = datetime.utcnow().isoformat()        
        reviewer_info = review_data.get('reviewer', {})
        reviewer_name = reviewer_info.get('name', 'anonymous')
        session_id = reviewer_info.get('session_id', 'no_session')        
        with open(reviews_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(review_data) + '\n')
        
        logger.info(f"Review submitted by {reviewer_name} (session: {session_id}) for triplet: {review_data.get('triplet', {}).get('subject', 'unknown')} - Rating: {review_data.get('rating', 'unknown')}")
        
        return jsonify({
            "success": True,
            "message": "Review submitted successfully"
        })
        
    except Exception as e:
        logger.error(f"Error submitting review: {e}")
        return jsonify({
            "success": False,
            "message": f"Error submitting review: {str(e)}"
        }), 500 