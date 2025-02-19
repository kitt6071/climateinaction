

def perform_basic_clustering(graph):
    # Community detection
    communities = nx.community.louvain_communities(graph.to_undirected())
    
    print(f"Number of clusters: {len(communities)}")
    for i, community in enumerate(communities):
        print(f"Cluster {i} (Size: {len(community)}):")
        print(community)  # prints all of the nodes in the cluster
    
    return communities

def analyze_clusters_with_llm(communities, graph, llm_setup, min_size=5):  # adjust min_size as needed

    # Try to get the largest clusters and describe them with llm
    sorted_communities = sorted(communities, key=len, reverse=True)
    
    for i, community in enumerate(sorted_communities):
        # set min to 20 
        if len(community) < min_size:
            continue
            
        # Get all concepts in this community
        concepts = list(community)
        
        print(f"\nAnalyzing Cluster {i} (Size: {len(community)})")
        
        prompt = f"""Analyze these related scientific concepts and identify the main theme:
        Concepts: {', '.join(concepts)}
        
        What is the main theme connecting these concepts? Be concise."""
        
        llm_setup['rate_limiter'].wait()
        response = llm_setup['client'].chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        print(f"Theme: {response.choices[0].message.content}")
from sentence_transformers import SentenceTransformer

import threading

def embed_documents(df):
    # generate basic embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Combine title and abstract
    texts = [f"{row['title']} {row['abstract']}" for row in df.iter_rows(named=True)]
    
    # Generate embeddings
    embeddings = model.encode(texts)
    return embeddings



    print("Creating embeddings...")
    embeddings = embed_documents(df)
    
    print("Clustering...")
    communities = perform_basic_clustering(global_graph)
    
    print("Analyzing clusters...")
    analyze_clusters_with_llm(communities, global_graph, llm_setup)