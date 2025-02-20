import json
import time
import polars as pl
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
import openai
import hashlib
import pickle
from pathlib import Path
import os
from dotenv import load_dotenv

def load_data(file_path: str) -> pl.DataFrame:
    df = pl.read_csv(file_path)
    # Csv has columns title, abstract, year, and doi, working only title and abstract for analysis
    df = df.drop_nulls(["title", "abstract"])
    return df

# follows the following: https://thedataschool.com/salome-grasland/using-the-isbndb-api-with-python/
# Uses the same idea/setup of rate limiting, but per minute instead of calls_per_secpnd like in the example I followed
class RateLimiter:
    # program is single-threaded for now, may have to check this for race conditions later?
    def __init__(self, rpm: int = 500):
        # initialize the values of self
        self.rpm = rpm 
        self.last_call = 0 
        self.interval = 60.0 / self.rpm # allows 500 requests per minute (60 seconds)
        
    def wait(self):
        now = time.time()
        elapsed = now - self.last_call # gets the time elapsed from the init call to now
        if self.interval - elapsed > 0: # checks if the interval (0.12 seconds) - the elapsed time from the call is positive
                                        # if positive, not enough time has passed yet, if negative (elapsed bigger than interval) too much time has passed
            time.sleep(self.interval - elapsed) # not enough time has passed and the program sleeps for the remainder of the interval
        
        self.last_call = time.time() # updates the last call with the new init call

#caches the responses from the api so I don't have to keep calling with the same prompt when testing other things
class Cache:
    # initializes the directory for the cache as cache, if directory doesn't exist it makes the directory
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    # makes a unique hash key to store then get each abstract and generated summary
    def make_hash_key(self, abstract: str, gen_summary: str) -> str:
        return hashlib.md5(f"{gen_summary}:{abstract}".encode()).hexdigest()
        
    def get(self, abstract: str, gen_summary: str):
        # uses the abstract and gen summary to make a unique hash
        cache_key = self.make_hash_key(abstract, gen_summary)
        # sets the dir for the cache_file to be in the generated unique hash .pkl file
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # if that .pkl file exists, then that means there exists the exact same abstract in the cache
        if cache_file.exists():
            try:
                #loads the cache data 
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                # should return either summary or triplets depending on what we are pulling
                print(f"Cache file found for {gen_summary}")
                return cached_data
            except Exception as e:
                print(f"Error reading the cache: {e}")
                return None
        return None
        
    def set(self, abstract: str, gen_summary: str, result):
        # does the same process as the getting, but instead creates the file
        cache_key = self.make_hash_key(abstract, gen_summary)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Opens the file name it made in writing binary mode so we can pickle dump
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Error writing the cache: {e}")

# initializes the api client, openai gpt4 in this case
def setup_llm():
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return {
        'client': openai.Client(api_key=api_key),
        'rate_limiter': RateLimiter(rpm=500),
        'cache': Cache()
    }

### Paper Used:
##The analyses presented here are conducted with different generative models, including fine-tuned models targeted for
#materials and biological materials:
#• X-LoRA, a fine-tuned, dynamic dense mixture-of-experts large language model with strong biological materials,
#math, chemistry, logic and reasoning capabilities [24] that uses two forward passes (details see reference and
#discussion in main text)
#• BioinspiredLLM-Mixtral, a fine-tuned mixture-of-experts (MoE) model based on the original BioinspiredLLM
#model [11] but using a mixture-of-expert approach basde on the Mixtral model [50]
#We also use general-purpose models, including:
#• Mistral-7B-OpenOrca [51, 52, 53] (used for text distillation into a heading, summary and bulleted list of
#detailed mechanisms and reasoning)
#• Zephyr-7B-β [54] built on top of the Mistral-7B model[5] (used for original graph generation due efficient
#compute and local hosting)
#• GPT-4 (gpt-4-0125-preview), at the time of the writing of this paper, this is the latest GPT model by
#OpenAI [3] (for some less complex tasks, specificall graph augmentation, we use GPT 3.5)
#• GPT-4V (gpt-4-vision-preview), a multimodal vision-text model by OpenAI [55, 56], for some use cases
#accessed via https://chat.openai.com/
#• Claude-3 Opus and Sonnet [57], accessed via https://claude.ai/chats


# Based on the knowledge graph paper, they did a multi step process of getting a summary with key facts and a title then extracting triples
# Following that, this asks the llm to generate a summary, though more basic, and send it off to generate triples
def convert_to_summary(abstract: str, llm_setup) -> str:
    cached = llm_setup["cache"].get(abstract, "summary")
    if cached:
        return cached

    # This prompt attempts to get a concise retelling of the abstract that focuses on the 
    # relationships between species and threats
    system_prompt = (
        """
        You are a scientific knowledge summarizer. Please convert the following text 
        into a structured summary that clearly states scientific relationships, causal connections, 
        and factual findings. Emphasize how relationships work, not just what happened, emphasize causation.
        """
    )

    #Output is generally inconsistent in format and provides either a very succint or very lenghty response:
    # example output:
 ### Summary of Scientific Relationships and Causal Connections

#1. **Urbanization and Human Activities Impact on Biodiversity:**
 #  - Increasing urbanization and recreational activities in biodiversity hotspots necessitate understanding the impacts of human disturbance on multiple species. This understanding is crucial for reducing negative impacts on biodiversity.

#2. **Anti-Predator Behaviour and Human Disturbance:**
 #  - Expanding knowledge on anti-predator behaviour can help predict how different species might respond to human disturbances. This approach provides a theoretical basis for understanding interspecies variation in response to humans.    

#3. **Limited Focus on Multiple Species in Studies:**
 #  - A review of literature revealed that only 21% of studies focusing on human disturbance through a behavioural approach considered multiple species. These studies highlighted several potential predictive variables for understanding species' responses.

#4. **Simulation Model Findings:**
 #  - A developed simulation model showed that fitness-related responses (e.g., quantity of food consumed) are sensitive to the distance at which animals detect humans and the frequency of human disturbance. These responses are less affected by other characteristics, indicating specific factors that significantly influence species' reactions to human presence.

#5. **Avian Alert Distance Study:**
 #  - An examination of avian alert distance across 150 species found that larger species have greater alert distances than smaller species. This suggests that body size influences how species perceive and react to threats, including human disturbance, potentially affecting their habitat suitability as human visitation increases.

#6. **Body Size as a Predictor of Response to Human Disturbance:**
 #  - Body size is suggested as a potential predictor of species' responses to human disturbance. This finding can aid conservation managers in making informed decisions regarding human visitation levels at protected sites.

#7. **Recommendations for Developing Predictive Models:**
 #  - To develop effective predictive models of species' responses to human disturbance, it is recommended to:
  #   a. Study multiple indicators of disturbance to identify those with lower intraspecific variation.
   #  b. Acknowledge the species-specific nature of responses to disturbance.
    # c. Assess life history, natural history, and other correlates of these species-specific responses.

### Conclusion
#Understanding and predicting the impacts of human disturbance on biodiversity requires a multifaceted approach that considers species-specific responses, the role of body size, and the interaction between humans and wildlife. By focusing on these aspects, conservation efforts can be more effectively tailored to protect biodiversity in areas of increasing human activity.


    try:
        # ensures rate is limited if necessary
        llm_setup["rate_limiter"].wait()
        # uses the gpt 4 for setup, the system has the above instruction and the user provides the info, abstract in this case
        response = llm_setup["client"].chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": abstract},
            ],
            temperature=0.1, # lower temperature for more consistent responses with same input
        )
        summary = response.choices[0].message.content.strip()
        print(f"Generated summary: {summary}\n")
        # caches the summary for later retrieval to save time
        llm_setup["cache"].set(abstract, "summary", summary)

        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""


def extract_triplets(summary: str, llm_setup) -> List[Tuple[str, str, str]]:

    cached = llm_setup["cache"].get(summary, "triplets")
    if cached:
        return cached

    # The paper prompt for triple extraction is more detailed and explicitly asks the model to extract taxonomy/ontology elements (nodes and their relations) using category theory as guidance.
    # • "node_1": "A concept from extracted ontology"
    # • "node_2": "A related concept from extracted ontology"
    # • "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"

    # attempts to get json response of triples for analysis, works well with gpt 4
    system_prompt = ("""
        You are a scientific knowledge graph builder focusing on species and their threats. 
        Extract subject-predicate-object triplets from the supplied summary such that:
        The subject represents a species, animal, or group of animals.
        The object is the threat or risk factor affecting the species.
        The predicate describes how the threat affects the species.
        Please format your response as a JSON array where each object has subject, predicate, and object keys.
        
        Example format:
        [
            {"subject": "sea birds", "predicate": "threatened by", "object": "plastic pollution"},
            {"subject": "arctic foxes", "predicate": "experience population decline due to", "object": "climate change"}
        ]
        """
    )

    #### Open AI structured outputs, json schema, guarantees the output is conforming to schema


    # same idea as the summary where gpt 4 is used and the system is set up with the above prompt, but 
    # the summary is provided as content instead of the abstract
    try:
        llm_setup["rate_limiter"].wait()
        response = llm_setup["client"].chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summary},
            ],
            temperature=0.1,
        )
        response_text = response.choices[0].message.content
        triplets_data = json.loads(response_text)
        triplets = [
            (item["subject"], item["predicate"], item["object"]) 
            for item in triplets_data
        ]
        print(f"Extracted triplets: {triplets}\n")
        ## example output:
        #Extracted triplets: [('multiple species', 'negatively impacted by', 'urbanization and recreational activities in biodiversity hotspots'), 
        # ('species', 'respond to', 'human disturbances based on anti-predator behaviour'), 
        # ('species', 'have fitness-related responses sensitive to', 'distance at which animals detect humans and the frequency of human disturbance'), 
        # ('avian species', 'have alert distances influenced by', 'body size, affecting habitat suitability as human visitation increases'), 
        # ('species', 'response to human disturbance predicted by', 'body size')]
        # also ('ground-nesting birds', 'experience reduced pre-fledgling survival due to', 'human disturbance')
        # also ('birds', 'face mortality due to', 'collisions with wind turbines')

        # few shot give some examples, and then see, the objects can be very similar and the 

        #############
        ######## D3 observable, typescript, force tree LINK: https://observablehq.com/@d3/force-directed-tree
        #############

        llm_setup["cache"].set(summary, "triplets", triplets)
        return triplets
    except json.JSONDecodeError as je:
        # catches error with any of the json triplet setup
        print(f"JSON decoding error: {je}\nResponse: {response_text}")
        return []
    except Exception as e:
        print(f"Error extracting triplets: {e}")
        return []

def build_global_graph(all_triplets: list) -> nx.DiGraph:
    # builds directed graph
    global_graph = nx.DiGraph()
    for triplet in all_triplets:
        subject, predicate, obj = triplet
        global_graph.add_node(subject)
        global_graph.add_node(obj)
        # You can later add logic to merge multiple relationships between the same nodes.
        global_graph.add_edge(subject, obj, relation=predicate)
    return global_graph

# attempt to get same statistics as paper to analyze and compare based on the paper's description
def analyze_graph_detailed(graph: nx.DiGraph):

    # Convert to undirected graph for certain analyses
    undirected_graph = graph.to_undirected()
    
    # Figure 2 equivalent - Graph Visualization
    plt.figure(figsize=(15, 5))
    
    # Full graph
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
    
    # Single node highlight
    plt.subplot(133)
    # Find most connected node
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
    
    plt.tight_layout()
    plt.show()

    # Graph stats from paper 
    plt.figure(figsize=(15, 5))
    
    # Need to update
    plt.subplot(131)
    degrees = [d for n, d in graph.degree()]
    degree_count = {}
    for d in degrees:
        degree_count[d] = degree_count.get(d, 0) + 1
    
    plt.loglog(list(degree_count.keys()), list(degree_count.values()), 'bo-')
    plt.xlabel('Degree (log)')
    plt.ylabel('Frequency (log)')
    plt.title('Degree Distribution')
    
    # Print statistics similar to table 1 in the knowledge graph paper
    print("\nGraph Statistics:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Average node degree: {sum(degrees)/len(degrees):.2f}")
    print(f"Maximum node degree: {max(degrees)}")
    print(f"Minimum node degree: {min(degrees)}")
    print(f"Median node degree: {sorted(degrees)[len(degrees)//2]}")
    print(f"Density: {nx.density(graph):.5f}")
    
    # Similarly, do the same with the giant_component that ignores the sparsely connected notdes
    giant_component = max(nx.connected_components(undirected_graph), key=len) # selects the set with the largest component in the graph
    giant_subgraph = graph.subgraph(giant_component)
    print("\nGiant Component Statistics:")
    gc_degrees = [d for n, d in giant_subgraph.degree()]
    print(f"Number of nodes: {giant_subgraph.number_of_nodes()}")
    print(f"Number of edges: {giant_subgraph.number_of_edges()}")
    print(f"Average node degree: {sum(gc_degrees)/len(gc_degrees):.2f}")
    print(f"Maximum node degree: {max(gc_degrees)}")
    print(f"Minimum node degree: {min(gc_degrees)}")
    print(f"Median node degree: {sorted(gc_degrees)[len(gc_degrees)//2]}")
    print(f"Density: {nx.density(giant_subgraph):.5f}")

# my results:

#Graph Statistics:
#Number of nodes: 181 # from 33 abstracts
#Number of edges: 139
#Average node degree: 1.54
#Maximum node degree: 33 # one node has 33 connections
#Minimum node degree: 1
#Median node degree: 1 # more than hald of the nodes appear in only one relationship, very sparse network
#Density: 0.00427

#Giant Component Statistics:
#Number of nodes: 45
#Number of edges: 45
#Average node degree: 2.00
#Maximum node degree: 33
#Minimum node degree: 1
#Median node degree: 1

#the paper results: graph vs giant component
#Number of nodes 12319 11878
#Number of edges 15752 15396
#Average node degree 2.56 2.59
#Maximum node degree 171 171
#Minimum node degree 1 1
#Median node degree 1 1
#Density 0.00021 0.00022
#Number of communities 109 80
#Density: 0.02273

def analyze_hub_node(graph: nx.DiGraph):

    # finds the node with the highest degree by iterating over node and degree and using the lambda to compare and return the second element of the tuple, degree, for each node
    max_degree_node = max(graph.degree, key=lambda x: x[1])[0] # [0] gets the node itself to pass to degree
    degree = graph.degree[max_degree_node]
    
    # Get all neighbors of the degree node
    neighbors = list(graph.neighbors(max_degree_node))
    
    # gets the relation from the neighbors of the max node 
    relationships = []
    for neighbor in neighbors:
        edge_data = graph.get_edge_data(max_degree_node, neighbor)
        relationships.append({
            'from': max_degree_node,
            'to': neighbor,
            'relation': edge_data['relation']
        })
    
    print(f"\nHub Node Analysis:")
    print(f"Most connected node: '{max_degree_node}'")
    print(f"Degree: {degree}")
    print("\nConnections:")
    for rel in relationships:
        print(f"- {rel['from']} --[{rel['relation']}]--> {rel['to']}")
        
    # Output snippet:
    
    #Hub Node Analysis:
    #Most connected node: 'birds'
    #Degree: 33

    #Connections:
    #- birds --[mistake for water sources and are killed by]--> oil pits and tanks
    #- birds --[estimated to die annually in the US due to]--> oil pits
    #- birds --[affected by]--> oil production operations
    #- birds --[negatively impacted by]--> wildlife viewing
    #- birds --[negatively impacted by]--> hiking
    #- birds --[negatively impacted by]--> running
    #- birds --[negatively impacted by]--> cycling

    # plotting for the hub and its connections
    plt.figure(figsize=(12, 8))
    hub_subgraph = graph.subgraph([max_degree_node] + neighbors)
    pos = nx.spring_layout(hub_subgraph)
    
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
    
    plt.title(f"Hub Node: {max_degree_node} and its connections")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    llm_setup = setup_llm()
    
    print("Loading csv...")
    df = load_data("bird_threats_abstracts.csv")
    
    print("Processing abstracts...")
    all_triplets = []
    for row in df.iter_rows(named=True):
        # Process entire abstract at once
        summary = convert_to_summary(row["abstract"], llm_setup)
        if not summary:
            return []
        triplets =  extract_triplets(summary, llm_setup)
        all_triplets.extend(triplets)
    
    global_graph = build_global_graph(all_triplets)
    
    print("Creating visualizations...")
    analyze_graph_detailed(global_graph)

    print("Analyzing hub node...")
    analyze_hub_node(global_graph)
if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    main()