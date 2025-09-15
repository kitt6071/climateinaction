import random
from flask import Blueprint, jsonify, request, Response
import config
from config import logger
from utils import load_data_if_needed, get_triplet_by_id
import os
import polars as pl
from pathlib import Path
from datetime import datetime
import json

import csv
from io import StringIO
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

api_bp = Blueprint('api', __name__)

ASSIGNMENTS_FILE = "/webapp/data/reviews/assignments.json"
ASSIGNMENTS_TIMEOUT = 3600

def get_reviewed_dois():
    try:
        reviews_file = "/webapp/data/reviews/triplet_reviews.jsonl"
        if not os.path.exists(reviews_file):
            return set()
        
        reviewed = set()
        with open(reviews_file, 'r') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    doi = review.get('group_doi', '').lower().strip()
                    if doi:
                        reviewed.add(doi)
                except json.JSONDecodeError:
                    continue
        return reviewed
    except Exception as e:
        logger.error(f"Error reading reviewed DOIs: {e}")
        return set()

def get_assigned_dois():
    try:
        if not os.path.exists(ASSIGNMENTS_FILE):
            return set()
        
        with open(ASSIGNMENTS_FILE, 'r') as f:
            assignments = json.load(f)
        
        current_time = datetime.utcnow().timestamp()
        active_assignments = {}
        
        for session_id, session_data in assignments.items():
            active_dois = []
            for assignment in session_data.get('dois', []):
                if current_time - assignment['timestamp'] < ASSIGNMENTS_TIMEOUT:
                    active_dois.append(assignment)
            if active_dois:
                active_assignments[session_id] = {'dois': active_dois}
        
        os.makedirs(os.path.dirname(ASSIGNMENTS_FILE), exist_ok=True)
        with open(ASSIGNMENTS_FILE, 'w') as f:
            json.dump(active_assignments, f)
        
        assigned = set()
        for session_data in active_assignments.values():
            for assignment in session_data.get('dois', []):
                doi_val = str(assignment.get('doi', '')).lower().strip()
                if doi_val:
                    assigned.add(doi_val)
        
        return assigned
    except Exception as e:
        logger.error(f"Error reading assignments: {e}")
        return set()

def get_session_assigned_dois(session_id):
    try:
        if not os.path.exists(ASSIGNMENTS_FILE):
            return set()
        
        with open(ASSIGNMENTS_FILE, 'r') as f:
            assignments = json.load(f)
        
        session_data = assignments.get(session_id, {})
        return {assignment['doi'] for assignment in session_data.get('dois', [])}
    except Exception as e:
        logger.error(f"Error reading session assignments: {e}")
        return set()

def track_doi_assignment(session_id, doi):
    try:
        os.makedirs(os.path.dirname(ASSIGNMENTS_FILE), exist_ok=True)
        
        assignments = {}
        if os.path.exists(ASSIGNMENTS_FILE):
            with open(ASSIGNMENTS_FILE, 'r') as f:
                assignments = json.load(f)
        
        if session_id not in assignments:
            assignments[session_id] = {'dois': []}

        assignments[session_id]['dois'] = [{
            'doi': str(doi).lower().strip(),
            'timestamp': datetime.utcnow().timestamp()
        }]
        
        with open(ASSIGNMENTS_FILE, 'w') as f:
            json.dump(assignments, f)
            
        logger.info(f"DOI {doi} assigned to session {session_id}")
    except Exception as e:
        logger.error(f"Error tracking assignment: {e}")

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
        if not load_data_if_needed() or not config.triplets_data:
            return jsonify({"success": False, "message": "Triplet data not loaded or unavailable."}), 500
        
        if config.abstracts_df is None:
            return jsonify({"success": False, "message": "Abstract data not loaded. Review functionality requires parquet data."}), 500

        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"success": False, "message": "Session ID required for review tracking."}), 400

        valid_dois = config.abstracts_df['doi_lower'].to_list()

        if not valid_dois:
            return jsonify({"success": False, "message": "No matching DOIs found between abstracts and triplets."}), 404

        reviewed_dois = get_reviewed_dois()
        assigned_dois = get_assigned_dois()
        
        unreviewed_dois = [
            doi for doi in valid_dois 
            if doi not in reviewed_dois and doi not in assigned_dois
        ]

        if not unreviewed_dois:
            return jsonify({"success": False, "message": "All available abstracts have been reviewed or assigned."}), 200
            
        selected_doi = random.choice(unreviewed_dois)
        track_doi_assignment(session_id, selected_doi)
        
        abstract_row_df = config.abstracts_df.filter(pl.col("doi_lower") == selected_doi)
        
        abstract_row = {}
        if len(abstract_row_df) > 0:
            abstract_row = abstract_row_df.row(0, named=True)

        group_triplets = [t for t in config.triplets_data if t.get('doi', '').lower() == selected_doi.lower()]

        clean_triplets = []
        for triplet in group_triplets:
            clean_triplet = {k: v for k, v in triplet.items() if k not in ['embedding_tensor', 'embedding']}
            clean_triplets.append(clean_triplet)

        response_data = {
            "success": True,
            "group": {
                "doi": selected_doi,
                "title": abstract_row.get("title", "Title not found"),
                "abstract": abstract_row.get("abstract", "Abstract not found."),
                "triplets": clean_triplets
            }
        }
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error getting random triplet group: {e}", exc_info=True)
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@api_bp.route('/submit-review', methods=['POST'])
def submit_review():
    try:
        review_data = request.get_json()
        
        if not review_data:
            return jsonify({"success": False, "message": "No review data provided"}), 400
        
        required_fields = ['group_doi', 'triplets', 'comments', 'reviewer']
        if not all(field in review_data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in review_data]
            return jsonify({"success": False, "message": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        if not isinstance(review_data['triplets'], list) or len(review_data['triplets']) == 0:
            return jsonify({"success": False, "message": "Triplets must be a non-empty list"}), 400
            
        review_data['server_timestamp'] = datetime.utcnow().isoformat()
        
        reviews_dir = "/webapp/data/reviews"
        os.makedirs(reviews_dir, exist_ok=True)
        
        file_path = os.path.join(reviews_dir, "triplet_reviews.jsonl")
        
        with open(file_path, 'a') as f:
            f.write(json.dumps(review_data) + '\n')
        
        session_id = review_data.get('reviewer', {}).get('session_id')
        reviewed_doi = review_data['group_doi'].lower().strip()
        
        if session_id:
            try:
                if os.path.exists(ASSIGNMENTS_FILE):
                    with open(ASSIGNMENTS_FILE, 'r') as f:
                        assignments = json.load(f)
                    
                    # Remove from this session's assignments
                    if session_id in assignments:
                        before_count = len(assignments[session_id].get('dois', []))
                        assignments[session_id]['dois'] = [
                            assignment for assignment in assignments[session_id]['dois']
                            if assignment['doi'] != reviewed_doi
                        ]
                        after_count = len(assignments[session_id]['dois'])
                        
                        for sid, session_data in assignments.items():
                            session_data['dois'] = [
                                assignment for assignment in session_data.get('dois', [])
                                if assignment['doi'] != reviewed_doi
                            ]
                        
                        with open(ASSIGNMENTS_FILE, 'w') as f:
                            json.dump(assignments, f)
                        
                        logger.info(f"Cleared assignment for DOI {reviewed_doi} from session {session_id} (removed {before_count - after_count} assignments)")
            except Exception as e:
                logger.error(f"Error clearing assignment: {e}")
        
        reviewer_name = review_data.get('reviewer', {}).get('name', 'Anonymous')
        session_id_display = review_data.get('reviewer', {}).get('session_id', 'N/A')
        logger.info(f"Review for DOI {review_data['group_doi']} by {reviewer_name} (Session: {session_id_display}) saved to {file_path}")
        
        return jsonify({"success": True, "message": "Review submitted successfully"})
        
    except Exception as e:
        logger.error(f"Error submitting review: {e}", exc_info=True)
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@api_bp.route('/review-progress', methods=['GET'])
def get_review_progress():
    """Get review progress statistics"""
    if not config.data_loaded:
        return jsonify({"success": False, "message": "Data not loaded"}), 500

    try:
        total_dois = len(config.abstracts_df)

        reviewed_dois = get_reviewed_dois()
        assigned_dois = get_assigned_dois()
        
        reviewed_set = {d.lower().strip() for d in reviewed_dois}
        assigned_set = {d.lower().strip() for d in assigned_dois}
        clean_assigned = assigned_set - reviewed_set
        
        return jsonify({
            "success": True,
            "total_abstracts": total_dois,
            "reviewed": len(reviewed_set),
            "assigned": len(clean_assigned),
            "available": total_dois - len(reviewed_set) - len(clean_assigned),
            "progress_percentage": round((len(reviewed_set) / total_dois * 100), 1) if total_dois > 0 else 0
        })
        
    except Exception as e:
        logger.error(f"Error getting review progress: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@api_bp.route('/reviews/stats', methods=['GET'])
def get_review_stats():
    try:
        reviews_file_path = "/webapp/data/reviews/triplet_reviews.jsonl"
        count = 0
        if os.path.exists(reviews_file_path):
            with open(reviews_file_path, 'r') as f:
                count = sum(1 for line in f if line.strip())
        return jsonify({"success": True, "reviews_completed": count})
    except Exception as e:
        logger.error(f"Error getting review stats: {e}", exc_info=True)
        return jsonify({"success": False, "message": f"Error getting stats: {str(e)}"}), 500

@api_bp.route('/clear-reviews', methods=['POST'])
def clear_reviews():
    try:
        reviews_file_path = "/webapp/data/reviews/triplet_reviews.jsonl"
        assignments_file_path = ASSIGNMENTS_FILE
        
        if os.path.exists(reviews_file_path):
            os.remove(reviews_file_path)
            logger.info(f"All reviews have been cleared. File deleted: {reviews_file_path}")
        else:
            logger.info("Clear reviews called, but no review file to delete.")

        if os.path.exists(assignments_file_path):
            os.remove(assignments_file_path)
            logger.info(f"All assignments have been cleared. File deleted: {assignments_file_path}")

        return jsonify({"success": True, "message": "All reviews and assignments have been cleared."})
            
    except Exception as e:
        logger.error(f"Error clearing reviews: {e}", exc_info=True)
        return jsonify({"success": False, "message": f"An error occurred while clearing reviews: {str(e)}"}), 500
        

@api_bp.route('/export-reviews', methods=['GET'])
def export_reviews():
    try:
        reviews_file_path = "/webapp/data/reviews/triplet_reviews.jsonl"
        
        if not os.path.exists(reviews_file_path):
            return "No reviews found to export.", 404

        output = StringIO()
        writer = csv.writer(output)

        header = [
            'review_timestamp', 'server_timestamp', 'reviewer_name', 'session_id',
            'review_comments', 'group_doi', 'triplet_id', 
            'triplet_subject', 'is_subject_valid',
            'triplet_predicate', 'is_predicate_valid',
            'triplet_object', 'is_object_valid'
        ]
        writer.writerow(header)
        with open(reviews_file_path, 'r') as f:
            for line in f:
                review = json.loads(line)
                
                for triplet in review.get('triplets', []):
                    validity = triplet.get('validity', {})
                    row = [
                        review.get('timestamp'),
                        review.get('server_timestamp'),
                        review.get('reviewer', {}).get('name'),
                        review.get('reviewer', {}).get('session_id'),
                        review.get('comments'),
                        review.get('group_doi'),
                        triplet.get('id'),
                        triplet.get('subject'), validity.get('subject'),
                        triplet.get('predicate'), validity.get('predicate'),
                        triplet.get('object'), validity.get('object')
                    ]
                    writer.writerow(row)

        output.seek(0)
        
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=triplet_reviews.csv"}
        )

    except Exception as e:
        logger.error(f"Error exporting reviews: {e}", exc_info=True)
        return f"Error generating CSV file: {str(e)}", 500 