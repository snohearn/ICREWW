#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

stat_sig_results = pd.read_csv("stat_sig_kendall_tau_results")
raw_data = pd.read_csv("raw_data.csv")

# Create species mapping using common_name
species_mapping = raw_data.set_index("species_id")["common_name"].to_dict()

# Create site mapping
site_mapping = raw_data.set_index("site_id")["site_name"].to_dict()

# Create phenophase mapping using phenophase_id and phenophase_description
phenophase_mapping = raw_data.set_index("phenophase_id")["phenophase_description"].to_dict()

# Apply mappings to the stat_sig_results dataset
stat_sig_results["species_a_name"] = stat_sig_results["species_a"].map(species_mapping)
stat_sig_results["species_b_name"] = stat_sig_results["species_b"].map(species_mapping)
stat_sig_results["site_name"] = stat_sig_results["site_id"].map(site_mapping)
stat_sig_results["phenophase_a"] = stat_sig_results["phenophase_a"].map(phenophase_mapping)
stat_sig_results["phenophase_b"] = stat_sig_results["phenophase_b"].map(phenophase_mapping)

# Save the updated dataset
stat_sig_results.to_csv("updated_stat_sig_results.csv", index=False)




# In[2]:


practice_df = pd.read_csv("Fake_Data_Test.csv")


# In[3]:


import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import threading
import webbrowser

# Load your actual data
data = stat_sig_results  # Update the path if needed

# Initialize Dash app
app = Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    dcc.Dropdown(
        id="site-dropdown",
        options=[{"label": site, "value": site} for site in data["site_name"].unique()],
        placeholder="Select a site",
        clearable=True
    ),
    dcc.Graph(id="network-graph", config={"scrollZoom": True}),
    dcc.Store(id="selected-nodes", data=[]),
    dcc.Graph(id="detailed-graph", config={"scrollZoom": True}, style={"display": "none"})
])

# Function to create species interaction graph
def create_species_graph(filtered_data):
    G = nx.Graph()
    
    for _, row in filtered_data.iterrows():
        species_a = row["species_a_name"]
        species_b = row["species_b_name"]
        G.add_edge(species_a, species_b)

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color="gray"),
        mode="lines",
        hoverinfo="none"
    )

    node_x, node_y, node_names = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_names.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_names,
        hoverinfo="text", marker=dict(size=10, color="blue"),
        name="Species"
    )

    return go.Figure(data=[edge_trace, node_trace], layout = go.Layout(
    title="Your Graph Title Here",
    showlegend=False,
    hovermode="closest",
    margin=dict(b=0, l=0, r=0, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Hides numbers
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Hides numbers
)
)

# Function to create phenophase graph (Adds dots at hover locations)
def create_phenophase_graph(filtered_data, selected_species):
    species_a, species_b = selected_species
    pos = {species_a: (-1, 0), species_b: (1, 0)}

    traces = []
    offset = 0.1  # Small vertical shift for multiple edges

    for i, (_, row) in enumerate(filtered_data.iterrows()):
        if {row["species_a_name"], row["species_b_name"]} == set(selected_species):
            phenophase_a = row["phenophase_a"]
            phenophase_b = row["phenophase_b"]
            tau = row["kendall_tau"]
            p_value = row["p_value"]

            hover_text = f"{phenophase_a} → {phenophase_b}<br>Kendall Tau: {tau}<br>P-value: {p_value}"

            x0, y0 = pos[species_a]
            x1, y1 = pos[species_b]

            # Offset for curved edges
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2 + (offset * (i - 0.5))

            # Create a trace for the edge
            edge_trace = go.Scatter(
                x=[x0, mid_x, x1, None],
                y=[y0, mid_y, y1, None],
                mode="lines",
                line=dict(width=2, color="red"),
                hoverinfo="text",
                text=hover_text,
                name=""
            )

            # Add visible dots at hover points
            hover_dots = go.Scatter(
                x=[mid_x], y=[mid_y],
                mode="markers",
                marker=dict(size=8, color="black"),
                hoverinfo="text",
                text=hover_text,
                name=""
            )

            traces.append(edge_trace)
            traces.append(hover_dots)

    # Create two species nodes
    node_trace = go.Scatter(
        x=[pos[species_a][0], pos[species_b][0]], 
        y=[pos[species_a][1], pos[species_b][1]],
        mode="markers+text",
        text=[species_a, species_b],
        hoverinfo="text",
        marker=dict(size=10, color="blue"),
        name=""
    )

    traces.append(node_trace)

    return go.Figure(data=traces, layout = go.Layout(
    title="Your Graph Title Here",
    showlegend=False,
    hovermode="closest",
    margin=dict(b=0, l=0, r=0, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Hides numbers
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Hides numbers
)
)

# Callback to update species graph
@app.callback(
    Output("network-graph", "figure"),
    Input("site-dropdown", "value")
)
def update_species_graph(selected_site):
    filtered_data = data if not selected_site else data[data["site_name"] == selected_site]
    return create_species_graph(filtered_data)

# Callback to track clicked nodes
@app.callback(
    Output("selected-nodes", "data"),
    Input("network-graph", "clickData"),
    State("selected-nodes", "data"),
)
def store_clicked_nodes(click_data, selected_nodes):
    if not click_data or "points" not in click_data:
        return selected_nodes

    clicked_node = click_data["points"][0]["text"]

    if clicked_node in selected_nodes:
        selected_nodes.remove(clicked_node)  # Deselect if clicked again
    else:
        selected_nodes.append(clicked_node)

    if len(selected_nodes) > 2:
        selected_nodes = [selected_nodes[-1]]  # Reset if more than two nodes are clicked

    return selected_nodes

# Callback to update phenophase graph
@app.callback(
    Output("detailed-graph", "figure"),
    Output("detailed-graph", "style"),
    Input("selected-nodes", "data"),
    Input("site-dropdown", "value")
)
def update_phenophase_graph(selected_nodes, selected_site):
    if len(selected_nodes) != 2:
        return go.Figure(), {"display": "none"}

    filtered_data = data if not selected_site else data[data["site_name"] == selected_site]
    return create_phenophase_graph(filtered_data, selected_nodes), {"display": "block"}

# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)


# In[4]:


pip install gunicorn


# In[5]:


pip install --upgrade pip


# In[6]:


pip freeze > requirements.txt


# In[ ]:





# In[ ]:




