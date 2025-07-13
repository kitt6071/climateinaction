import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
    exit(1)

def load_training_data_v2():
    training_file = Path('data_to_review/training_data.json')
    if training_file.exists():
        logger.info("Loading combined training data from training_data.json")
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        formatted_data = []
        for item in training_data:
            formatted_data.append({
                'text': item.get('text', ''),
                'label': item.get('label', 0),
                'title': item.get('title', 'Generated')
            })
        
        positives = sum(1 for item in formatted_data if item['label'] == 1)
        negatives = len(formatted_data) - positives
        
        logger.info(f"Loaded {positives} positive and {negatives} negative examples")
        logger.info(f"Total training data: {len(formatted_data)} examples")
        
        return formatted_data
    
    logger.info("Loading separate positive and negative files")
    
    with open('data_to_review_v2/shorebird_positives.json', 'r') as f:
        positives_raw = json.load(f)
    
    with open('data_to_review_v2/shorebird_negatives.json', 'r') as f:
        negatives_raw = json.load(f)
    
    training_data = []
    
    if isinstance(positives_raw, list):
        for item in positives_raw:
            if isinstance(item, str):
                training_data.append({
                    'text': item,
                    'label': 1,
                    'title': 'Generated Shorebird Abstract'
                })
            elif isinstance(item, dict):
                training_data.append({
                    'text': item.get('text', item.get('abstract', '')),
                    'label': 1,
                    'title': item.get('title', 'Generated Shorebird Abstract')
                })
    
    if isinstance(negatives_raw, list):
        for item in negatives_raw:
            if isinstance(item, str):
                training_data.append({
                    'text': item,
                    'label': 0,
                    'title': 'Generated Non-Shorebird Abstract'
                })
            elif isinstance(item, dict):
                training_data.append({
                    'text': item.get('text', item.get('abstract', '')),
                    'label': 0,
                    'title': item.get('title', 'Generated Non-Shorebird Abstract')
                })
    
    positives = sum(1 for item in training_data if item['label'] == 1)
    negatives = len(training_data) - positives
    
    logger.info(f"Loaded {positives} positive examples and {negatives} negative examples")
    logger.info(f"Total training data: {len(training_data)} examples")
    
    return training_data

def train_classifier(training_data, test_size=0.2):
    texts = [item['text'] for item in training_data if item['text'].strip()]
    labels = [item['label'] for item in training_data if item['text'].strip()]
    
    if len(texts) != len(training_data):
        logger.warning(f"Filtered out {len(training_data) - len(texts)} empty text examples")
    
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    logger.info(f"Training set: {len(X_train_texts)} examples")
    logger.info(f"Test set: {len(X_test_texts)} examples")
    
    logger.info("Loading sentence transformer model")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    logger.info("Generating embeddings for training data")
    X_train_embeddings = embedding_model.encode(X_train_texts, show_progress_bar=True)
    
    logger.info("Generating embeddings for test data")
    X_test_embeddings = embedding_model.encode(X_test_texts, show_progress_bar=True)
    
    logger.info("Training logistic regression classifier")
    classifier = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    classifier.fit(X_train_embeddings, y_train)
    
    logger.info("Evaluating classifier")
    y_pred = classifier.predict(X_test_embeddings)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Non-Relevant', 'Relevant'])
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")
    
    current_script_dir = Path(__file__).parent
    models_base_dir = current_script_dir / "trained_relevance_models_central"
    model_name = "synthetic_shorebird_classifier_v2" 
    models_dir = models_base_dir / model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    
    classifier_path = models_dir / "embedding_classifier.pkl"
    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    logger.info(f"Saved classifier to {classifier_path}")
    
    model_info = {
        'embedding_model': 'all-mpnet-base-v2',
        'classifier_type': 'LogisticRegression',
        'test_accuracy': accuracy,
        'training_examples': len(training_data),
        'positive_examples': sum(labels),
        'negative_examples': len(labels) - sum(labels),
        'training_method': 'synthetic_shorebird_data_v2'
    }
    
    info_path = models_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Saved model info to {info_path}")
    
    metrics_path = models_dir / "evaluation_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Test Set Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    logger.info(f"Saved evaluation metrics to {metrics_path}")
    
    training_data_path = models_dir / f"collected_training_data_{sum(labels)}R_{len(labels) - sum(labels)}I.json"
    with open(training_data_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2)
    logger.info(f"Saved training data to {training_data_path}")
    
    logger.info(f"Classifier saved in trained_relevance_models_central structure at: {models_dir}")
    
    return classifier, embedding_model

def main():
    logger.info("Starting shorebird relevance classifier training with V2 data...")
    
    training_data = load_training_data_v2()
    
    if not training_data:
        logger.error("No training data loaded. Check that files exist in data_to_review_v2/")
        return
    
    classifier, embedding_model = train_classifier(training_data)
    
    logger.info("Training completed successfully!")
    logger.info("You can now use the trained classifier to score new abstracts.")

if __name__ == "__main__":
    main() 