import os
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import config
# Import configuration and global state
from config import logger, ML_LIBS_LOADED

# Import utilities
from utils import load_data_if_needed

# Import route blueprints
from routes import register_blueprints

def create_app():
    """Application factory pattern"""
    app = Flask(__name__, static_folder='static', template_folder='templates')
    CORS(app)
    
    # Register all route blueprints
    register_blueprints(app)
    
    return app

# Create the Flask app
app = create_app()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "Climate Analysis API is running",
        "data_loaded": config.data_loaded,
        "triplets_count": len(config.triplets_data) if config.data_loaded else 0,
        "ml_available": ML_LIBS_LOADED
    })

# Backward compatibility endpoints for frontend
@app.route('/load-data', methods=['POST'])
def load_data():
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
                "message": "Failed to load data"
            }), 500
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Climate Analysis Server")
    
    if ML_LIBS_LOADED:
        logger.info("All ML libraries loaded - advanced features available")
    else:
        logger.info("ML libraries not available - some features will be disabled")
    
    logger.info("Data will be loaded from cloud storage when accessed")
    logger.info(f"Server running at http://0.0.0.0:{port}")
    
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port) 