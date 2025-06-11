import json
from sentence_transformers import SentenceTransformer


TRIPLETS_FILE_PATH = "/Users/kittsonhamill/Desktop/dissertation/climate_inaction/Lent_Init/runs/second run/deepseek_deepseek-r1-0528_10000/results/enriched_triplets.json"
OUTPUT_SUBDIR = "backend/"
OUTPUT_FILENAME = "data_with_embeddings.json" 
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
MAX_K_TO_TEST_FOR_CLUSTERING = 20
MIN_K_FOR_CLUSTERING = 2
RANDOM_STATE = 42
TSNE_N_COMPONENTS = 3
TSNE_ITERATIONS = 300
TSNE_PERPLEXITY = 30

def construct_threat_sentence(triplet):
    subject = triplet.get("subject", "Unknown species")
    predicate = triplet.get("predicate", "unknown predicate")
    obj = triplet.get("object", "unknown object")
    
    sentence = f"{subject} {predicate} related to {obj}."
    return sentence

def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded.")

    print(f"Loading data from {TRIPLETS_FILE_PATH}")
    with open(TRIPLETS_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    triplets = data.get("triplets", [])
    print(f"Loaded {len(triplets)} triplets.")

    processed_triplets = []
    all_threat_sentences = []

    for i, triplet in enumerate(triplets):
        threat_sentence = construct_threat_sentence(triplet)
        triplet['id'] = f"triplet_{i}" 
        triplet['threat_sentence'] = threat_sentence
        all_threat_sentences.append(threat_sentence)
        processed_triplets.append(triplet)
    
    if all_threat_sentences:
        print(f"Generating embeddings for {len(all_threat_sentences)} sentences")
        embeddings = model.encode(all_threat_sentences, show_progress_bar=True)
        print("Embeddings generated.")

        for triplet, embedding in zip(processed_triplets, embeddings):
            triplet['embedding'] = embedding.tolist()
    else:
        print("No threat sentences found to embed.")

    output_data = {"triplets": processed_triplets, "taxonomy": data.get("taxonomy", {})}
    
    print(f"Saving processed data with embeddings to {OUTPUT_SUBDIR}")
    with open(OUTPUT_SUBDIR + OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print("Preprocessing complete. Data saved.")

if __name__ == "__main__":
    main()