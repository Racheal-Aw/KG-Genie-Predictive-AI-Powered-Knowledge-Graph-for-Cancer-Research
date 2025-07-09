import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import ast
import pickle
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="KG Genie Prototype", layout="wide")
st.title("🧠  KG Genie – Activity Predictor + Network Context")

BEST_THRESH = 0.50          # ← replace with your Youden threshold
MODEL_PATH  = "New_RF_pipeline.sav" 
SCALER_PATH = None          # or "scaler.pkl" if you saved one
FEATURE_CSV = "triplet_df_cleaned_good.csv" 

import re

# 🔑 fixed lengths
FINGERPRINT_LEN = 2048   # adjust if you used a different size
EMBED_LEN       = 64

# ---- robust parser that pads / truncates ----
def parse_vector(cell, target_len):
    if isinstance(cell, (list, np.ndarray)):
        arr = np.asarray(cell, dtype=float)
    else:
        tokens = re.findall(r"[-+]?\d*\.\d+|\d+", str(cell))
        arr = np.asarray([float(t) for t in tokens], dtype=float)

    if arr.size < target_len:                       # pad with zeros
        arr = np.hstack([arr, np.zeros(target_len - arr.size)])
    elif arr.size > target_len:                     # truncate
        arr = arr[:target_len]
    return arr


# ------------------------------------------------------------------
# 🚀  Load artefacts *once* (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts(feature_csv, model_pkl, scaler_pkl=None):
    # features
    df_feat = pd.read_csv(feature_csv)
    # parse & fix lengths
    df_feat["morgan_fp_array"]   = df_feat["morgan_fp_array"].apply(lambda x: parse_vector(x, FINGERPRINT_LEN))
    df_feat["graph_embed"] = df_feat["graph_embed"].apply(lambda x: parse_vector(x, EMBED_LEN))

    # sanity‑check
    assert all(v.size == FINGERPRINT_LEN for v in df_feat["morgan_fp_array"])
    assert all(v.size == EMBED_LEN       for v in df_feat["graph_embed"])

    # model
    with open(model_pkl, "rb") as f:
        model = pickle.load(f)

    # scaler (if you saved one).  Otherwise create fresh—less ideal but works.
    if scaler_pkl:
        with open(scaler_pkl, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler().fit(
            np.vstack(
                df_feat.apply(lambda r: np.hstack((r["morgan_fp_array"], r["graph_embed"])), axis=1)
            )
        )

    return df_feat, model, scaler


df_feat, model, scaler = load_artifacts(
    feature_csv="triplet_df_cleaned_good.csv",
    model_pkl="New_RF_pipeline.sav"
)

# ----------------------------------------------------------------
# ❹  If the choice is a drug row → run prediction
drug_choices = df_feat["source"] if "source" in df_feat else df_feat["smiles"]
choice = st.selectbox("Select drug", drug_choices)


# Fetch the row
row = df_feat.loc[df_feat["source"] == choice] if "source" in df_feat else \
      df_feat.loc[df_feat["smiles"] == choice]

if row.empty:
    st.error("Drug not found!")
else:
    row = row.iloc[0]  # convert to Series
    morgan_fp = row["morgan_fp_array"]
    graph_emb = row["graph_embed"]

    # Combine and scale
    features = np.hstack((morgan_fp, graph_emb)).reshape(1, -1)
    features_scaled = scaler.transform(features)

    if st.button("🔮 Predict"):
        pred = int(model.predict(features_scaled)[0])
        prob = model.predict_proba(features_scaled)[0][pred]
        st.success(f"**Prediction:** {pred}  (prob = {prob:.2f})")

df = pd.read_csv("triplet_df_cleaned_good.csv")

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

