
def analyze_graph(graph: nx.DiGraph):
    # Convert to undirected graph for certain analyses.
    undirected_graph = graph.to_undirected()
    
    # 5.1 Degree Distribution
    degrees = [degree for _, degree in undirected_graph.degree()]
    plt.figure()
    plt.hist(degrees, bins=range(1, max(degrees)+1), log=True, color="skyblue")
    plt.title("Degree Distribution (log scale)")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()
    
    # 5.2 Identify Giant Component
    giant_component_nodes = max(nx.connected_components(undirected_graph), key=len)
    giant_component = undirected_graph.subgraph(giant_component_nodes)
    print("Giant Component nodes:", len(giant_component.nodes()))
    print("Giant Component edges:", len(giant_component.edges()))
    
    # 5.3 Graph Density
    density = nx.density(graph)
    print("Graph Density:", density)
    
    # 5.4 Clustering Coefficient
    clustering = nx.average_clustering(undirected_graph)
    print("Average Clustering Coefficient:", clustering)
    
    # 5.5 Betweenness Centrality
    betweenness = nx.betweenness_centrality(graph)
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top nodes by betweenness centrality:", top_betweenness)



def analyze_cross_abstract_connections(graph: nx.DiGraph, all_triplets: list):
    """
    Analyze how species and threats are connected across different abstracts
    """
    # Create dictionaries to track appearances
    species_connections = {}  # species -> list of threats
    threat_connections = {}   # threat -> list of species
    
    # Build connection dictionaries
    for triplet in all_triplets:
        species, _, threat = triplet
        
        # Track species -> threats
        if species not in species_connections:
            species_connections[species] = set()
        species_connections[species].add(threat)
        
        # Track threat -> species
        if threat not in threat_connections:
            threat_connections[threat] = set()
        threat_connections[threat].add(species)
    
    # Analyze species affected by multiple threats
    print("\nSpecies affected by multiple threats:")
    multi_threat_species = {
        species: threats 
        for species, threats in species_connections.items() 
        if len(threats) > 1
    }
    for species, threats in sorted(multi_threat_species.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{species} is affected by {len(threats)} threats:")
        for threat in threats:
            print(f"- {threat}")
    
    # Analyze threats affecting multiple species
    print("\nThreats affecting multiple species:")
    multi_species_threats = {
        threat: species 
        for threat, species in threat_connections.items() 
        if len(species) > 1
    }
    for threat, species in sorted(multi_species_threats.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{threat} affects {len(species)} species:")
        for s in species:
            print(f"- {s}")
    
    # Visualize the most connected threats and species
    plt.figure(figsize=(15, 5))
    
    # Plot species by number of threats
    plt.subplot(121)
    species_threat_counts = [len(threats) for threats in species_connections.values()]
    plt.hist(species_threat_counts, bins=range(max(species_threat_counts)+2))
    plt.title("Distribution of Threats per Species")
    plt.xlabel("Number of Threats")
    plt.ylabel("Number of Species")
    
    # Plot threats by number of species affected
    plt.subplot(122)
    threat_species_counts = [len(species) for species in threat_connections.values()]
    plt.hist(threat_species_counts, bins=range(max(threat_species_counts)+2))
    plt.title("Distribution of Species per Threat")
    plt.xlabel("Number of Species Affected")
    plt.ylabel("Number of Threats")
    
    plt.tight_layout()
    plt.show()
    
    return species_connections, threat_connections

    print("\nAnalyzing cross-abstract connections...")
    species_connections, threat_connections = analyze_cross_abstract_connections(global_graph, all_triplets)