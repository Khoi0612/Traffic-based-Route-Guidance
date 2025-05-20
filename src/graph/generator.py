from src.data.data_processor import *
from .traffic_graph import TrafficGraph
from .visualizer import GraphVisualizer

class TrafficGraphGenerator:
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.scats_data = None
        self.graph = None
    
    def run(self, origin, destination, ml_model_type, target_dt):
        # Load and process data
        self.scats_data = DataProcessor(self.data_file)
        nodes = self.scats_data.extract_node_coordinates()
        connections = self.scats_data.extract_street_connections()
        
        # Create graph
        self.graph = TrafficGraph(nodes, connections)
        self.graph.generate_edges('models', ml_model_type.lower(), target_dt)
        
        # Generate and export graph content
        graph_content = self.graph.format_graph_data(origin, destination)
        print(graph_content)
        
        # Export to file
        target_dt_str = target_dt.strftime("%Y-%m-%d_%H-%M-%S")
        filename = self.graph.export_to_file(
            graph_content, 
            base_name=f"test_from_{origin}_to_{destination}_using_{ml_model_type}_at_{target_dt_str}"
        )

        # Visualize the graph
        visualizer = GraphVisualizer()
        fig, _, _, _ = visualizer.draw_graph_on_map(self.graph.nodes, self.graph.edges, origin, destination)
        fig.show()

        return filename