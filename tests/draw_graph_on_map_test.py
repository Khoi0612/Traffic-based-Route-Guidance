import plotly.graph_objects as go

def load_graph_from_file(filename):

    # Set up data structures
    nodes = {}
    edges = {}
    origin = None
    destinations = []

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
            # e.g. {1:(2,3)}
            parts = line.split(":")
            node = int(parts[0])
            coords = list(map(float, parts[1].strip(" ()").split(',')))
            nodes[node] = coords
        elif section == "edges":
            # e.g. {(1,2):3}
            parts = line.split(":")
            n1, n2 = map(int, parts[0].strip(" ()").split(','))
            cost = round(float(parts[1]))         
            edges.setdefault((n1, n2), cost)
        elif section == "origin":
            origin = int(line)
        elif section == "destinations":
            destinations = list(map(int, line.split(';')))      

    return nodes, edges, origin, destinations

nodes, edges, origin, destination = load_graph_from_file('test_xgb_1.txt')

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
        mode='lines+markers+text',
        line=dict(width=2, color='black'),
        marker=dict(size=6),
        text=[None, cost],  # label at the end point
        textposition='top right',
        name=cost
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
    lat=[nodes[destination[0]][0]],
    lon=[nodes[destination[0]][1]],
    mode='markers+text',
    text=destination[0],
    textposition='top right',
    marker=dict(size=10, color='orange'),
    name='Destination'
))

# Map layout
fig.update_layout(
    mapbox=dict(
        style='open-street-map',
        center=dict(lat=sum(scats_lat)/len(scats_lat), lon=sum(scats_lon)/len(scats_lon)),
        zoom=4
    ),
    margin={"r":0,"t":0,"l":0,"b":0}
)

# Show the map
fig.show()

