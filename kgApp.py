import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="KG Genie Prototype", layout="wide")
st.title("🧠 KG Genie: AI-Driven Knowledge Graph for Cancer")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("triplet_df_extended.csv")
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filter")
gene_options = sorted(df[df['relation'].isin(['targets', 'associated_with', 'involved_in'])]['target'].unique())
drug_options = sorted(df[df['relation'] == 'targets']['source'].unique())
disease_options = sorted(df[df['relation'] == 'associated_with']['target'].unique())
pathway_options = sorted(df[df['relation'] == 'involved_in']['target'].unique())

gene_options = sorted(df[df['relation'].isin(['targets', 'associated_with', 'involved_in'])]['target'].unique())
drug_options = sorted(df[df['relation'] == 'targets']['source'].unique())
disease_options = sorted(df[df['relation'] == 'associated_with']['target'].unique())
pathway_options = sorted(df[df['relation'] == 'involved_in_pathway']['target'].unique())

filter_type = st.sidebar.radio("Filter by:", ["Gene", "Drug", "Disease", "Pathway"])
selected = None

if filter_type == "Gene":
    selected = st.sidebar.selectbox("Select Gene", ["All"] + gene_options)
    if selected == "All":
        filtered_df = df
    else:
        # Step 1: Drug ➝ Gene (reverse lookup)
        drugs_df = df[(df['target'] == selected) & (df['relation'] == 'targets')]

        # Step 2: Gene ➝ Disease
        disease_df = df[(df['source'] == selected) & (df['relation'] == 'associated_with')]

        # Step 3: Gene ➝ Pathway
        pathway_df = df[(df['source'] == selected) & (df['relation'] == 'involved_in_pathway')]

        # Combine all
        filtered_df = pd.concat([drugs_df, disease_df, pathway_df], ignore_index=True).drop_duplicates()

elif filter_type == "Drug":
    selected = st.sidebar.selectbox("Select Drug", ["All"] + drug_options)
    if selected == "All":
        filtered_df = df
    else:
        # Step 1: Direct Drug ➝ Gene
        direct_df = df[(df['source'] == selected) & (df['relation'] == 'targets')]
        target_genes = direct_df['target'].unique()

        # Step 2: Gene ➝ Disease and Gene ➝ Pathway
        disease_df = df[(df['source'].isin(target_genes)) & (df['relation'] == 'associated_with')]
        pathway_df = df[(df['source'].isin(target_genes)) & (df['relation'] == 'involved_in_pathway')]

        # Step 3: Combine all
        filtered_df = pd.concat([direct_df, disease_df, pathway_df], ignore_index=True).drop_duplicates()

elif filter_type == "Disease":
    selected = st.sidebar.selectbox("Select Disease", ["All"] + disease_options)
    if selected == "All":
        filtered_df = df
    else:
        # Step 1: Gene ⇐ Disease
        gene_df = df[(df['target'] == selected) & (df['relation'] == 'associated_with')]
        disease_genes = gene_df['source'].unique().tolist()

        # Step 2: Drug ⇨ Gene
        drug_df = df[(df['target'].isin(disease_genes)) & (df['relation'] == 'targets')]

        # Step 3: Gene ⇨ Pathway
        pathway_df = df[(df['source'].isin(disease_genes)) & (df['relation'] == 'involved_in_pathway')]

        # Combine all
        filtered_df = pd.concat([gene_df, drug_df, pathway_df], ignore_index=True).drop_duplicates()


elif filter_type == "Pathway":
    selected = st.sidebar.selectbox("Select Pathway", ["All"] + pathway_options)
    if selected == "All":
        filtered_df = df
    else:
        # Step 1: Gene ⇐ Pathway
        gene_df = df[(df['target'] == selected) & (df['relation'] == 'involved_in_pathway')]
        pathway_genes = gene_df['source'].unique().tolist()

        # Step 2: Drug ⇨ Gene
        drug_df = df[(df['target'].isin(pathway_genes)) & (df['relation'] == 'targets')]

        # Step 3: Gene ⇨ Disease
        disease_df = df[(df['source'].isin(pathway_genes)) & (df['relation'] == 'associated_with')]

        # Combine all
        filtered_df = pd.concat([gene_df, drug_df, disease_df], ignore_index=True).drop_duplicates()


# Prompt area



# Show filtered table
st.subheader("📋 Knowledge Graph Triples")
st.dataframe(filtered_df)

# Build the graph
G = nx.DiGraph()

for _, row in filtered_df.iterrows():
    G.add_edge(row['source'], row['target'], label=row['relation'])

# Layout with fixed seed
pos = nx.spring_layout(G, k=0.5, seed=42)

edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x, node_y, node_text, node_color = [], [], [], []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

    if node in drug_options:
        node_color.append('orange')
    elif node in disease_options:
        node_color.append('red')
    elif node in gene_options:
        node_color.append('skyblue')
    elif node in pathway_options:
        node_color.append('purple')
    else:
        node_color.append('lightgreen')

# Draw the graph
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none', mode='lines'
)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="top center",
    marker=dict(color=node_color, size=15, line_width=2),
    hoverinfo='text'
)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title="📊 Knowledge Graph",
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
)

st.plotly_chart(fig, use_container_width=True)

# Optional footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, NetworkX, and Plotly")

# Show KG HTML
#st.subheader("🧬 Interactive Knowledge Graph")
#with open("kg_genie_graph.html", "r", encoding="utf-8") as f:
    #html_code = f.read()
#st.components.v1.html(html_code, height=600, scrolling=True)

