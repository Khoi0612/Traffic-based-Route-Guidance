from src.models import *
import plotly.graph_objects as go

class GraphVisualizer:
    
    @staticmethod
    def draw_graph_on_map(nodes, edges, origin, destination):
        # Create a figure
        fig = go.Figure()

        # Extract the scats number, lattitude, and longtitude from the nodes dictionary
        scats_no = list(nodes.keys())
        scats_lat = [nodes[name][0] for name in scats_no]
        scats_lon = [nodes[name][1] for name in scats_no]

        # Add markers for all scat sites
        fig.add_trace(go.Scattermap(
            lat=scats_lat,
            lon=scats_lon,
            mode='markers+text',
            text=scats_no,
            textposition='top right',
            marker=dict(size=10, color='skyblue'),
            name='Locations'
        ))

        # Draw edges connecting 2 nodes
        for (n1, n2), cost in edges.items():
            # Extract the lattitude, and longtitude from the nodes dictionary with each edge's key
            lat_pair = [nodes[n1][0], nodes[n2][0]]
            lon_pair = [nodes[n1][1], nodes[n2][1]]

            # Add lines for each edge
            fig.add_trace(go.Scattermap(
                lat=lat_pair,
                lon=lon_pair,
                mode='lines+markers',
                line=dict(width=2, color='black'),
                marker=dict(size=6),
                name=f"{cost:.0f} secs"
            ))

        # Add origin
        fig.add_trace(go.Scattermap(
            lat=[nodes[origin][0]],
            lon=[nodes[origin][1]],
            mode='markers+text',
            text=origin,
            textposition='top right',
            marker=dict(size=10, color='limegreen'),
            name='Origin'
        ))

        # Add destination
        fig.add_trace(go.Scattermap(
            lat=[nodes[destination][0]],
            lon=[nodes[destination][1]],
            mode='markers+text',
            text=destination[0] if isinstance(destination, list) else destination,
            textposition='top right',
            marker=dict(size=10, color='orange'),
            name='Destination'
        ))

        center_lat = sum(scats_lat)/len(scats_lat)
        center_lon = sum(scats_lon)/len(scats_lon)
        zoom_level = 11.5

        legend_item_count = len([trace for trace in fig.data if trace.showlegend])
        legend_height_adjustment = 0.5 * legend_item_count / zoom_level  # Adjust based on zoom level
        
        adjusted_center_lat = center_lat - legend_height_adjustment/2

        fig.update_layout(
            autosize=True,
            hovermode='closest',
            showlegend=True,
            map=dict(
                bearing=0,
                center=dict(lat=adjusted_center_lat, lon=center_lon),
                zoom=zoom_level,
                style='open-street-map'
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.02,
                xanchor="center",
                x=0.5
            )
        )   

        return fig, adjusted_center_lat, center_lon, zoom_level
    
    @staticmethod
    def draw_solution_on_map(graph_map, origin, destination, solution_path):
        nodes = graph_map.locations
        edges = {}
        for node1 in graph_map.graph_dict:
            for node2, cost in graph_map.graph_dict[node1].items():
                edges[(node1, node2)] = cost

        fig, adjusted_center_lat, center_lon, zoom_level = GraphVisualizer.draw_graph_on_map(nodes, edges, origin, destination)

        # Draw solution path edges in red
        for i in range(len(solution_path) - 1):
            n1 = solution_path[i]
            n2 = solution_path[i + 1]
            
            # Get coordinates
            lat1, lon1 = nodes[n1]
            lat2, lon2 = nodes[n2]
            
            # Draw the solution edge line in red with increased width
            fig.add_trace(go.Scattermap(
                lat=[lat1, lat2],
                lon=[lon1, lon2],
                mode='lines',
                line=dict(width=5, color='red'),
                showlegend=False if i > 0 else True,
                name='Solution Path' if i == 0 else None
            ))

        fig.update_layout(
            autosize=True,
            hovermode='closest',
            showlegend=True,
            map=dict(
                bearing=0,
                center=dict(lat=adjusted_center_lat, lon=center_lon),
                zoom=zoom_level,
                style='open-street-map'
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.02,
                xanchor="center",
                x=0.5
            )
        ) 

        # Show the map
        fig.show()