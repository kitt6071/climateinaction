from .api_routes import api_bp
from .analysis_routes import analysis_bp
from .network_routes import network_bp
from .knowledge_transfer_routes import knowledge_transfer_bp

def register_blueprints(app):
    """Register all route blueprints with the Flask app"""
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(analysis_bp, url_prefix='/api')  
    app.register_blueprint(network_bp, url_prefix='/api')
    app.register_blueprint(knowledge_transfer_bp, url_prefix='/api')

__all__ = [
    'api_bp',
    'analysis_bp', 
    'network_bp',
    'knowledge_transfer_bp',
    'register_blueprints'
] 