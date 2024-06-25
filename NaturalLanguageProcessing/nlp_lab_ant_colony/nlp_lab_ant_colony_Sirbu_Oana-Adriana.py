import nltk
import random
import math
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from collections import defaultdict


stop_words = set(stopwords.words('english'))

    
class IDMapper:
    def __init__(self):
        self.word_to_id_map = {}
        self.unique_id = 1

    def extract_word_ids(self, sense):
        """
        I start by word tokenizing the definition of the sense, followed by removing the stopwords.
        I then extract the unique IDs for each word in the definition, and sort them, as per instructions.
        """
        word_ids = []
        gloss = sense.definition()
        def_words = nltk.word_tokenize(gloss)
        def_words = [word for word in def_words if word.lower() not in stop_words]
        for word in def_words:
            if word not in self.word_to_id_map:
                self.word_to_id_map[word] = self.unique_id
                self.unique_id += 1
            word_ids.append(self.word_to_id_map[word])
        word_ids.sort()
        return word_ids


def compute_common_words_score(sense_1, sense_2, mapper):
    """
    Compute the relatedness score between two senses based on the number of common words in their definitions.
    """
    gloss_1_ids = mapper.extract_word_ids(sense_1)
    gloss_2_ids = mapper.extract_word_ids(sense_2)
    common_ids = set(gloss_1_ids).intersection(gloss_2_ids)
    return len(common_ids)

def extended_lesk_similarity(vector_1, vector_2):
    common_elements = set(vector_1).intersection(vector_2)
    return len(common_elements)


class Graph:
    """
    Class representing the entire graph, with nodes and edges.
    We have the possibility to add nodes and edges to the graph, depending on the flow of the algorithm.
    """
    def __init__(self, mapper):
        self.nodes = {}
        self.edges = defaultdict(GraphEdge)
        self.mapper = mapper
        
    def add_node(self, sense, is_nest_node):
        self.nodes[sense] = GraphNode(sense, is_nest_node, self.mapper)
    
    def add_edge(self, sense_1, sense_2):
        self.edges[(sense_1, sense_2)] = GraphEdge()

class GraphNode:
    """
    Create the Node class to represent a node in the graph.
    The energy level of a simple node is 0, while the energy level of a nest node is a random integer between 5 and 60.
    This list of IDs of the words from a definitions represents the "odour" of the node based on the gloss.
    If the node is indeed a nest node, we must assign a True value to the is_nest_node attribute.
    """
    def __init__(self, sense, is_nest_node, mapper):
        self.sense = sense
        self.is_nest_node = is_nest_node
        self.energy_level = random.randint(5, 60) if is_nest_node else 0
        self.odour_signature = mapper.extract_word_ids(sense) if is_nest_node else []

class GraphEdge:
    """
    We know that the edge between two nodes is represented by the pheromone level. 
    It is initialized with a value of 0, but it will be updated during the algorithm.
    """
    def __init__(self):
        self.pheromone_level = 0

class SearchAnt:
    """
    Class to describe an ant in terms of its current node, lifespan, energy collected, and path taken.
    We also allow this ant to move to a new node.
    """
    def __init__(self, initial_node, lifespan):
        self.current_node = initial_node
        self.lifespan = lifespan
        self.stored_energy = 0
        self.path_taken = [initial_node]
        self.return_mode = False
    
    def move_to_node(self, node):
        self.path_taken.append(node)
        self.current_node = node

class ACO_WSD:
    """
    The ant colony optimization algorithm for word sense disambiguation has a specified number of cycles, 
    and also predefined values for omega (ant life-span), theta (quantity of pheromone), and delta (evaporation rate of pheromone).
    """
    def __init__(self, input_text, cycles=50, omega=26, theta=1, delta=0.3577):
        self.input_text = input_text
        self.cycles = cycles
        self.mapper = IDMapper()
        self.graph = Graph(self.mapper)
        self.ants = []
        self.omega = omega
        self.theta = theta
        self.delta = delta
        self.initialize_graph()

    def initialize_graph(self):
        """
        We consider an ambiguous word to be a word that has more than one synset.
        """
        words = nltk.word_tokenize(self.input_text)
        ambiguous_words = [word for word in words if len(wn.synsets(word)) > 1]
        for word in ambiguous_words:
            senses = wn.synsets(word)
            for sense in senses:
                self.graph.add_node(sense, True)
        for sense1 in self.graph.nodes:
            for sense2 in self.graph.nodes:
                if sense1 != sense2:
                    self.graph.add_edge(sense1, sense2)
    
    def main(self):
        """
        Runs the ACO algorithm for a specified number of cycles, as specified in the lecture (following the next steps):
        1. Remove dead ants and bridges with no pheromone.
        2. For each nest, potentially spawn a new ant.
        3. For each ant, determine its mode (energy search or return), make it move, potentially create a bridge, and update the path.
        4. Update the pheromone levels on the edges, odour vectors and energy levels of nodes.
        """
        for cycle in range(self.cycles):
            self.remove_dead_ants()
            self.spawn_new_ants()
            self.move_ants()
            self.update_pheromones()

    def remove_dead_ants(self):
        """
        Removes ants that exceeded their lifespan. Also update the energy of the nodes.
        """
        for ant in self.ants:
            ant.lifespan -= 1
            if ant.lifespan <= 0:
                self.graph.nodes[ant.current_node].energy_level += ant.stored_energy
                self.ants.remove(ant)
        for edge in list(self.graph.edges.keys()):
            if self.graph.edges[edge].pheromone_level <= 0:
                del self.graph.edges[edge]
    
    def spawn_new_ants(self):
        """
        Function to create new ants based on the energy level of nest nodes.
        """
        for sense in self.graph.nodes:
            node = self.graph.nodes[sense]
            if node.is_nest_node:
                probability = math.atan(node.energy_level) / math.pi + 0.5
                if random.random() < probability:
                    self.ants.append(SearchAnt(sense, self.omega))

    def move_ants(self):
        """
        Helps us determine the mode of the ant (energy search or return), and then the ant moves.
        """
        for ant in self.ants:
            if ant.return_mode:
                self.return_to_nest(ant)
            else:
                self.search_for_energy(ant)
    
    def search_for_energy(self, ant):
        """
        Function which focuses on exploring nodes with higher energy levels. 
        It uses evaluation based on energy levels and pheromone avoidance.
        """
        current_node = self.graph.nodes[ant.current_node]
        neighbors = [neighbor for neighbor in self.graph.nodes if neighbor != ant.current_node]
        probabilities = []
        for neighbor in neighbors:
            edge = self.graph.edges[(ant.current_node, neighbor)]
            relatedness = compute_common_words_score(current_node.sense, self.graph.nodes[neighbor].sense, self.mapper)
            energy_eval = self.graph.nodes[neighbor].energy_level / sum(self.graph.nodes[n].energy_level for n in neighbors)
            pheromone_eval = 1 - edge.pheromone_level
            evalf = energy_eval * pheromone_eval * relatedness
            probabilities.append(evalf)
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
            chosen_neighbor = random.choices(neighbors, probabilities)[0]
            ant.move_to_node(chosen_neighbor)
            if (ant.current_node, chosen_neighbor) not in self.graph.edges:
                self.graph.add_edge(ant.current_node, chosen_neighbor)
            if self.graph.nodes[ant.current_node].energy_level > 0:
                ant.return_mode = True

    def return_to_nest(self, ant):
        """
        Method to handle the logic of ants moving towards nodes with the highest pheromone levels and closest odor vectors.
        It focuses on returning to the nest with the collected energy and also uses evaluation 
        based on pheromone levels and similarity of odor vectors.
        """
        current_node = self.graph.nodes[ant.current_node]
        neighbors = [neighbor for neighbor in self.graph.nodes if neighbor != ant.current_node]
        probabilities = []
        for neighbor in neighbors:
            edge = self.graph.edges[(ant.current_node, neighbor)]
            similarity = extended_lesk_similarity(current_node.odour_signature, self.graph.nodes[neighbor].odour_signature)
            pheromone_eval = edge.pheromone_level
            evalf = pheromone_eval * (similarity / sum(extended_lesk_similarity(self.graph.nodes[n].odour_signature, current_node.odour_signature) for n in neighbors))
            probabilities.append(evalf)
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
            chosen_neighbor = random.choices(neighbors, probabilities)[0]
            ant.move_to_node(chosen_neighbor)
    
    def update_pheromones(self):
        """
        Updates the pheromone levels on edges based on ant movements.
        First it increases with theta, then it decays (as time passes, basically it evaporates). 
        """
        for ant in self.ants:
            for i in range(len(ant.path_taken) - 1):
                edge = self.graph.edges[(ant.path_taken[i], ant.path_taken[i + 1])]
                edge.pheromone_level += self.theta
        for edge in self.graph.edges.values():
            edge.pheromone_level *= (1 - self.delta)

def identify_best_senses(graph):
    """
    We consider the best senses those with the highest energy levels in the graph.
    """
    best_senses = {}
    for sense in graph.nodes:
        node = graph.nodes[sense]
        if node.is_nest_node:
            if sense not in best_senses or node.energy_level > best_senses[sense].energy_level:
                best_senses[sense] = node
    return best_senses

my_text = "Astronomers have discovered a new planet outside our solar system that might have conditions suitable for life"

aco_wsd = ACO_WSD(my_text)
aco_wsd.main()

best_senses = identify_best_senses(aco_wsd.graph)
print("The best senses proved to be:", best_senses)
