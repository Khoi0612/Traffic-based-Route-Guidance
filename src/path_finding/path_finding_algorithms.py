"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.
"""

import sys
from collections import deque

from .utils import *

import glob
# from graph_gui import *
import time

def is_in(x, lst):
    return x in lst

class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


def depth_first_graph_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack
    explored = set()
    start_time = time.perf_counter()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node, len(explored), (time.perf_counter() - start_time) * 1000
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None, len(explored), (time.perf_counter() - start_time) * 1000


def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    start_time = time.perf_counter()
    node = Node(problem.initial)
    explored = set()
    if problem.goal_test(node.state):
        return node, len(explored), (time.perf_counter() - start_time) * 1000
    frontier = deque([node])
    
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child, len(explored), (time.perf_counter() - start_time) * 1000
                frontier.append(child)
    return None, len(explored), (time.perf_counter() - start_time) * 1000

def uniform_cost_search(problem):
    """Expands the node with the lowest total path cost."""
    start_time = time.perf_counter()
    node = Node(problem.initial)
    explored = set()
    if problem.goal_test(node.state):
        return node, 0, (time.perf_counter() - start_time) * 1000

    frontier = []
    heapq.heappush(frontier, (node.path_cost, node))
    nodes_expanded = 0

    while frontier:
        cost, node = heapq.heappop(frontier)
        nodes_expanded += 1

        if problem.goal_test(node.state):
            return node, nodes_expanded, (time.perf_counter() - start_time) * 1000 

        explored.add(node.state)

        for child in node.expand(problem):
            if child.state not in explored and all(child.state != existing.state for _, existing in frontier):
                heapq.heappush(frontier, (child.path_cost, child))
            else:
                # Replace node in frontier if child has a lower cost
                for i, (c, existing) in enumerate(frontier):
                    if existing.state == child.state and child.path_cost < existing.path_cost:
                        frontier[i] = (child.path_cost, child)
                        heapq.heapify(frontier)
                        break

    return None, nodes_expanded, (time.perf_counter() - start_time) * 1000

def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    start_time = time.perf_counter()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node, len(explored), (time.perf_counter() - start_time) * 1000
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None, len(explored), (time.perf_counter() - start_time) * 1000

# ______________________________________________________________________________
# Informed (Heuristic) Search


greedy_best_first_graph_search = best_first_graph_search


# Greedy best-first search is accomplished by specifying f(n) = h(n).


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


# ______________________________________________________________________________
# A* heuristics 

# ______________________________________________________________________________


def iterative_deepening_astar_search(problem, h):

    h = memoize(h or problem.h, 'h')

    def f(node):
        return node.path_cost + h(node)
    
    nodes_explored = [0]
    
    def recursive_dls_with_f_value(node, problem, f_limit):
        try:
            nodes_explored[0] += 1
            f_value = f(node)       
            if f_value > f_limit:
                return 'cutoff', f_value           
            if problem.goal_test(node.state):
                return node, None
                
            next_f = float('inf')
            
            for child in node.expand(problem):
                result, new_f = recursive_dls_with_f_value(child, problem, f_limit)
                
                if result == 'cutoff':
                    next_f = min(next_f, new_f if new_f is not None else float('inf'))
                elif result is not None:
                    return result, None
                    
            return 'cutoff', next_f if next_f < float('inf') else None
    
        except RecursionError:
            return None, None  # Signal recursion failure
    
    initial_node = Node(problem.initial)
    threshold = f(initial_node)
    total_nodes_explored = 0
    start_time = time.perf_counter()
    
    if problem.goal_test(initial_node.state):
        return initial_node, total_nodes_explored, (time.perf_counter() - start_time) * 1000

    while True:
        nodes_explored[0] = 0
        result, next_threshold = recursive_dls_with_f_value(initial_node, problem, threshold)
        total_nodes_explored += nodes_explored[0]

        if result != 'cutoff':
            return result, total_nodes_explored, (time.perf_counter() - start_time) * 1000
            
        if result is None or next_threshold is None:
            return None, total_nodes_explored, (time.perf_counter() - start_time) * 1000
            
        threshold = next_threshold


# ______________________________________________________________________________
# IDA* 

# ______________________________________________________________________________


# ______________________________________________________________________________
# Graphs and Graph Problems


class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def load_graph_from_file(filename):

    # Set up data structures
    nodes = {}
    edges = {}
    origin = None
    destinations = []
    graph_map = Graph()

    # Open and read the text file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Divide the lines from the text file into sections and name each section by its component
    section = None
    for line in lines:
        line = line.strip()
        if line == "Nodes:":
            section = "nodes"
            continue
        elif line == "Edges:":
            section = "edges"
            continue
        elif line == "Origin:":
            section = "origin"
            continue
        elif line == "Destinations:":
            section = "destinations"
            continue

        if not line:
            continue

        # Assign each section's lines with each element in the corresponding data structure
        if section == "nodes":
            # e.g. 0970: (-37.867303089430855,145.09151138211365)
            parts = line.split(":")
            node = parts[0]  # remove leading zeros like '0970' → 970
            coords = list(map(float, parts[1].strip(" ()").split(',')))  # support float
            nodes[node] = coords
        elif section == "edges":
            parts = line.split(":")
            n1, n2 = map(lambda x: x, parts[0].strip(" ()").split(','))
            cost = float(parts[1])          
            edges.setdefault((n1, n2), cost)
        elif section == "origin":
            origin = line
        elif section == "destinations":
            destinations = map(int, line.split(';'))

    # Create the graph from these sections
    for node in nodes:
        for n, cost in edges.items():
            if n[0] == node:
                graph_map.connect1(n[0], n[1], cost) # One way connection
    graph_map.locations = nodes       

    return graph_map, origin, destinations

class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if isinstance(self.goal, list):
                if type(node) is str:
                    return min(int(distance(locs[node], locs[goal])) for goal in self.goal)
                else:
                    return min(int(distance(locs[node.state], locs[goal])) for goal in self.goal)
            else:
                if type(node) is str:
                    return int(distance(locs[node], locs[self.goal]))

                return int(distance(locs[node.state], locs[self.goal]))
         
        else:
            return np.inf


def run_algorithm(method, problem):

    if method == "DFS":
        result_node, explored, runtime_ms = depth_first_graph_search(problem)
    elif method == "BFS":
        result_node, explored, runtime_ms = breadth_first_graph_search(problem)
    elif method == "GBFS":
        result_node, explored, runtime_ms = best_first_graph_search(problem, lambda n: problem.h(n), display=True)
    elif method == "AS":
        result_node, explored, runtime_ms = astar_search(problem, lambda n: problem.h(n), display=True)
    elif method == "CUS1":
        result_node, explored, runtime_ms = uniform_cost_search(problem)
    elif method == "CUS2":
        result_node, explored, runtime_ms = iterative_deepening_astar_search(problem, lambda n: problem.h(n))
    else:
        raise ValueError(f"Unsupported method: {method}")

    return result_node, explored, runtime_ms

