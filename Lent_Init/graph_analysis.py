import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import pickle
from pathlib import Path
import os
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import logging
from .iucn_refinement import cache_enriched_triples, parse_and_validate_object
from .batch_ingesting import EMBEDDINGS_AVAILABLE

# Setup logger
logger = logging.getLogger("pipeline")

# quick debug helper
def print_enriched_triple(subject, predicate, obj, doi, taxonomy):
    print(f"\nSubject: {subject}")
    if taxonomy:
        for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']:
            if taxonomy.get(level):
                print(f"  {level.title()}: {taxonomy[level]}")
    print(f"Predicate: {predicate}")
    print(f"Object: {obj}")
    print(f"DOI: {doi}")
    print("-" * 50)

def build_global_graph(all_triplets):
    # simple directed graph construction
    global_graph = nx.DiGraph()
    for triplet in all_triplets:
        subject, predicate, obj, _doi = triplet
        global_graph.add_node(subject)
        global_graph.add_node(obj)
        # merge multiple relationships between same nodes
        global_graph.add_edge(subject, obj, relation=predicate)
    return global_graph

def analyze_graph_detailed(graph, figures_dir):
    # Create output dir
    figures_path = figures_dir
    figures_path.mkdir(exist_ok=True)
    
    undirected_graph = graph.to_undirected()
    
    # Main visualization - Figure 2 style
    plt.figure(figsize=(15, 5))
    
    # Full graph view
    plt.subplot(131)
    pos = nx.spring_layout(graph) # force directed layout? fruchterman-reingold algorithm
    # it calculates node positions based on simulated attractive and repulsive forces which leads to a visual clustering of nodes.
    nx.draw(graph, pos, node_size=20, alpha=0.6, with_labels=False)
    plt.title("Global Graph")
    
    # Zoomed section similar to paper
    plt.subplot(132)
    subset_nodes = list(graph.nodes())[:10]  # takes the first 10 nodes to zoom into the graph
    subgraph = graph.subgraph(subset_nodes)
    pos_sub = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos_sub, with_labels=True, node_size=500)
    plt.title("Zoomed Section")
    
    # print what we're looking at
    print("\n--- Triplets in Zoomed Section ---")
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'related_to')
        print(f"{u} | {relation} | {v}")
    print("-" * 30)
    
    # highlight most connected node
    plt.subplot(133)
    if graph.nodes():
        max_degree_node = max(graph.degree, key=lambda x: x[1])[0]
        neighbors = list(graph.neighbors(max_degree_node)) + [max_degree_node]
        highlight_graph = graph.subgraph(neighbors)
        pos_highlight = nx.spring_layout(highlight_graph)
        
        nx.draw(highlight_graph, pos_highlight, node_color='lightgrey', with_labels=True, node_size=500)
        nx.draw_networkx_nodes(highlight_graph, pos_highlight, 
                            nodelist=[max_degree_node], 
                            node_color='red', 
                            node_size=700)
        plt.title(f"Connections of '{max_degree_node}'")
        
        print(f"\n--- Triplets for {max_degree_node} ---")
        for u, v, data in highlight_graph.edges(data=True):
            relation = data.get('relation', 'related_to')
            print(f"{u} | {relation} | {v}")
        print("-" * 30)
    else:
        plt.title("Empty Graph")
    
    plt.savefig(figures_path / "graph_analysis.png", 
                bbox_inches='tight', dpi=300, format='png')
    plt.close()
    print(f"\nGraph analysis visualization saved to {figures_path / 'graph_analysis.png'}")

    # basic stats
    plt.figure(figsize=(15, 5))
    
    # Need to update
    plt.subplot(131)
    degrees = [d for n, d in graph.degree()]
    if not degrees:
        print("Graph is empty, cannot plot degree distribution.")
    else:
        degree_count = {}
        for d in degrees:
            degree_count[d] = degree_count.get(d, 0) + 1
        
        plt.loglog(list(degree_count.keys()), list(degree_count.values()), 'bo-')
        plt.xlabel('Degree (log)')
        plt.ylabel('Frequency (log)')
        plt.title('Degree Distribution')
    
    # some basic numbers
    print(f"\nNodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    if degrees:
        print(f"Avg degree: {sum(degrees)/len(degrees):.2f}")
        print(f"Max degree: {max(degrees)}")
        print(f"Min degree: {min(degrees)}")
    print(f"Density: {nx.density(graph):.5f}")
    
    # Similarly do the same with the giant_component that ignores the sparsely connected nodes
    if graph.number_of_nodes() > 0:
        try:
            undirected_graph_for_components = graph.to_undirected()
            if undirected_graph_for_components.number_of_edges() > 0:
                 connected_components = list(nx.connected_components(undirected_graph_for_components))
                 if connected_components:
                     giant_component_nodes = max(connected_components, key=len)
                     giant_subgraph = graph.subgraph(giant_component_nodes)
                     print(f"\nGiant component - Nodes: {giant_subgraph.number_of_nodes()}")
                     print(f"Giant component - Edges: {giant_subgraph.number_of_edges()}")
                     gc_degrees = [d for n, d in giant_subgraph.degree()]
                     if gc_degrees:
                         print(f"Average node degree: {sum(gc_degrees)/len(gc_degrees):.2f}")
                         print(f"Maximum node degree: {max(gc_degrees)}")
                         print(f"Minimum node degree: {min(gc_degrees)}")
                         print(f"Median node degree: {sorted(gc_degrees)[len(gc_degrees)//2]}")
                     else:
                         print("Average node degree: isolated gc nodes)")
                     print(f"Density: {nx.density(giant_subgraph):.5f}")
                 else:
                    print("\nGiant Component: No connected components")
            else:
                print("\nGiant Component: Graph is empty")
        except Exception as e:
            print(f"Giant component failed: {e}")

def analyze_hub_node(graph, figures_dir):
    figures_path = figures_dir
    figures_path.mkdir(exist_ok=True)

    if not graph.nodes():
        print("Empty graph, no hub analysis")
        return

    # finds the node with the highest degree by iterating over node and degree and using the lambda to compare and return the second element of the tuple, degree, for each node
    max_degree_node = max(graph.degree, key=lambda x: x[1])[0] # [0] gets the node itself to pass to degree
    degree = graph.degree[max_degree_node]
    
    # Get all neighbors of the degree node
    neighbors = list(graph.neighbors(max_degree_node))
    predecessors = list(graph.predecessors(max_degree_node))
    
    print(f"\nHub: '{max_degree_node}' (degree: {degree})")
    print("Outgoing:")
    if neighbors:
        for neighbor in neighbors:
            edge_data = graph.get_edge_data(max_degree_node, neighbor)
            relation = edge_data.get('relation', 'related_to')
            print(f"  {max_degree_node} | {relation} | {neighbor}")
    else:
        print("  None")
    
    print("Incoming:")
    if predecessors:
        for predecessor in predecessors:
            edge_data = graph.get_edge_data(predecessor, max_degree_node)
            relation = edge_data.get('relation', 'related_to')
            print(f"  {predecessor} | {relation} | {max_degree_node}")
    else:
        print("  None")
        
    # plot the hub
    plt.figure(figsize=(12, 8))
    hub_nodes_to_plot = list(set([max_degree_node] + neighbors + predecessors))
    hub_subgraph = graph.subgraph(hub_nodes_to_plot)
    pos = nx.spring_layout(hub_subgraph, k=1, iterations=50)
    
    nx.draw(hub_subgraph, pos, 
           node_color='lightblue',
           node_size=2000,
           with_labels=True,
           font_size=8,
           font_weight='bold')
    
    nx.draw_networkx_nodes(hub_subgraph, pos,
                           nodelist=[max_degree_node],
                           node_color='red',
                           node_size=3000)
    
    edge_labels = nx.get_edge_attributes(hub_subgraph, 'relation')
    nx.draw_networkx_edge_labels(hub_subgraph, pos, 
                                edge_labels=edge_labels,
                                font_size=6)
    
    plt.title(f"Hub Node: {max_degree_node}")
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(figures_path / "hub_node_analysis.png", 
                bbox_inches='tight', dpi=300, format='png')
    plt.close()
    print(f"Hub pic saved: {figures_path / 'hub_node_analysis.png'}")

def enrich_graph_with_embeddings(graph, model, results_dir):
    """Add embeddings to graph nodes"""
    if not EMBEDDINGS_AVAILABLE or model is None:
        print("No embeddings available")
        return []
        
    print(f"Enriching {len(graph.nodes())} nodes with embeddings...")
    
    try:
        node_texts = list(graph.nodes())
        node_embeddings = model.encode(node_texts, show_progress_bar=True)
        
        # add as node attributes
        for i, node in enumerate(node_texts):
            graph.nodes[node]['embedding'] = node_embeddings[i]
        
        print("Looking for potential connections...")
        potential_connections = []
        
        # sample if too large
        if len(graph.nodes()) > 1000:
            import random
            sample_nodes = random.sample(list(graph.nodes()), 1000)
        else:
            sample_nodes = list(graph.nodes())
        
        # find similar unconnected nodes
        for i, node1 in enumerate(sample_nodes):
            if i % 100 == 0:
                print(f"  {i}/{len(sample_nodes)}")
                
            embed1 = graph.nodes[node1]['embedding'].reshape(1, -1)
            
            for node2 in sample_nodes:
                if node1 != node2 and not graph.has_edge(node1, node2) and not graph.has_edge(node2, node1):
                    embed2 = graph.nodes[node2]['embedding'].reshape(1, -1)
                    similarity = sklearn_cosine_similarity(embed1, embed2)[0][0]
                    
                    if similarity > 0.85:
                        potential_connections.append((node1, node2, similarity))
        
        potential_connections.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(potential_connections)} potential connections")
        return potential_connections[:100]
    except Exception as e:
        print(f"Embedding enrichment failed: {e}")
        return []

def create_embedding_visualization(graph, model, figures_dir):
    """Create embedding-based graph viz"""
    if not EMBEDDINGS_AVAILABLE or model is None:
        logger.warning("No embedding model available")
        return False
        
    logger.info("Creating embedding visualization...")
    try:
        nodes = list(graph.nodes())
        
        # get embeddings
        embeddings = []
        for node in nodes:
            if 'embedding' in graph.nodes[node]:
                embeddings.append(graph.nodes[node]['embedding'])
            else:
                embeddings.append(model.encode([node])[0])
        
        embeddings = np.array(embeddings)
        
        # reduce to 2D
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            node_positions = reducer.fit_transform(embeddings)
            
            plt.figure(figsize=(20, 20))
            
            # draw edges
            for u, v in graph.edges():
                i = nodes.index(u)
                j = nodes.index(v)
                plt.plot([node_positions[i, 0], node_positions[j, 0]],
                         [node_positions[i, 1], node_positions[j, 1]],
                         'k-', alpha=0.1, linewidth=0.5)
            
            # draw nodes
            plt.scatter(node_positions[:, 0], node_positions[:, 1], s=10, alpha=0.8)
            
            # label top nodes by degree
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
            
            for node, _ in top_nodes:
                i = nodes.index(node)
                plt.text(node_positions[i, 0], node_positions[i, 1], node, 
                        fontsize=8, alpha=0.7)
            
            figures_dir.mkdir(parents=True, exist_ok=True)
            png_path = figures_dir / "embedding_graph.png"
            html_path = figures_dir / "embedding_graph.html"

            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Static embedding viz: {png_path}")
            
            # try interactive version
            try:
                from pyvis.network import Network
                
                net = Network(height="800px", width="100%", notebook=False)
                
                for i, node_label in enumerate(nodes):
                    x_coord = float(node_positions[i, 0])
                    y_coord = float(node_positions[i, 1])
                    net.add_node(i, label=node_label, x=x_coord*100, y=y_coord*100)
                    
                for u, v in graph.edges():
                    i = nodes.index(u)
                    j = nodes.index(v)
                    net.add_edge(i, j)
                    
                net.save_graph(str(html_path)) 
                logger.info(f"Interactive viz: {html_path}")
                
            except ImportError:
                logger.warning("pyvis not available - no interactive viz")
            except Exception as e:
                logger.exception(f"Interactive viz failed: {e}")
                
        except ImportError:
            logger.warning("UMAP not available")
            return False
        except Exception as e:
            logger.exception(f"UMAP/plotting failed: {e}")
            return False
            
        return True
    except Exception as e:
        logger.exception(f"Embedding viz failed: {e}")
        return False

async def visualize_triplet_sentence_embeddings_batch_ingest(graph, embedding_model_instance, output_figures_dir, filename="triplet_sentences_tsne_batch_ingest.png", tsne_n_components=2, tsne_perplexity=30.0, tsne_n_iter=300, n_clusters_sentences=5, random_state=42):
    """t-SNE viz of triplet sentences for batch ingestion"""
    if graph is None:
        logger.warning("No graph for t-SNE")
        return None, None
    if not embedding_model_instance:
        logger.warning("No embedding model for t-SNE")
        return None, None

    logger.info(f"t-SNE for {graph.number_of_nodes()}N, {graph.number_of_edges()}E graph")

    sentences = []
    triplet_details = []

    for u, v, data in graph.edges(data=True):
        subject_node_id = str(u)
        object_node_full_string = str(v)
        predicate = data.get('relation', 'is_related_to') 

        obj_desc_for_sentence, _iucn_code, _iucn_name, _is_valid = parse_and_validate_object(object_node_full_string)
        if not obj_desc_for_sentence:
            obj_desc_for_sentence = object_node_full_string

        sentence = f"{subject_node_id} {predicate} {obj_desc_for_sentence}"
        sentences.append(sentence)
        triplet_details.append({
            'subject': subject_node_id,
            'predicate': predicate,
            'object_full_string': object_node_full_string,
            'object_description_used': obj_desc_for_sentence,
            'sentence': sentence
        })

    if not sentences:
        logger.warning("No sentences for t-SNE")
        return None, None

    logger.info(f"Generated {len(sentences)} sentences for t-SNE")

    try:
        sentence_embeddings = embedding_model_instance.encode(sentences, show_progress_bar=True)
        num_samples = sentence_embeddings.shape[0]
        effective_perplexity = min(tsne_perplexity, max(1.0, float(num_samples - 1)))

        if num_samples <= 1 or effective_perplexity <= 0:
            logger.warning(f"Not enough samples for t-SNE: {num_samples}")
            return None, None

        tsne_reducer = TSNE(n_components=tsne_n_components,
                            perplexity=effective_perplexity,
                            n_iter=tsne_n_iter,
                            init='pca', learning_rate='auto',
                            random_state=random_state)
        reduced_embeddings = tsne_reducer.fit_transform(sentence_embeddings)

        effective_n_clusters = min(n_clusters_sentences, num_samples)
        if effective_n_clusters <=0: effective_n_clusters = 1

        cluster_labels = np.zeros(num_samples, dtype=int)
        if num_samples >= effective_n_clusters and effective_n_clusters > 0:
            from sklearn.cluster import KMeans 
            kmeans = KMeans(n_clusters=effective_n_clusters, random_state=random_state, n_init='auto')
            cluster_labels = kmeans.fit_predict(sentence_embeddings)
        else:
            logger.warning(f"KMeans won't work: {effective_n_clusters} clusters vs {num_samples} samples")
        
        df_data = {
            'sentence': sentences,
            'cluster': cluster_labels
        }
        if tsne_n_components >= 1: df_data['tsne_x'] = reduced_embeddings[:, 0]
        if tsne_n_components >= 2: df_data['tsne_y'] = reduced_embeddings[:, 1]
        if tsne_n_components >= 3: df_data['tsne_z'] = reduced_embeddings[:, 2]
        df = pd.DataFrame(df_data)

        plt.figure(figsize=(12, 10))
        if tsne_n_components == 2:
            sns.scatterplot(data=df, x='tsne_x', y='tsne_y', hue='cluster', palette='tab10', s=50, alpha=0.7)
            plt.xlabel("t-SNE Dim 1")
            plt.ylabel("t-SNE Dim 2")
        elif tsne_n_components == 3:
            ax = plt.gcf().add_subplot(111, projection='3d')
            scatter_plot = ax.scatter(df['tsne_x'], df['tsne_y'], df['tsne_z'], c=df['cluster'], cmap='tab10', s=30, alpha=0.7)
            ax.set_xlabel("t-SNE Dim 1")
            ax.set_ylabel("t-SNE Dim 2")
            ax.set_zlabel("t-SNE Dim 3")
            legend = ax.legend(*scatter_plot.legend_elements(), title="Cluster")
            ax.add_artist(legend)
        else:
            sns.stripplot(data=df, x='cluster', y='tsne_x', hue='cluster', palette='tab10', alpha=0.7)
            plt.xlabel("Cluster")
            plt.ylabel("t-SNE Dim 1")

        plt.title(f"t-SNE of Triplet Sentences ({len(sentences)} sentences, {effective_n_clusters} clusters)")
        plt.tight_layout()

        if filename:
            output_figures_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_figures_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"t-SNE saved: {save_path}")
        
        return plt.gcf(), df

    except Exception as e:
        logger.error(f"t-SNE failed: {e}", exc_info=True)
        return None, None
