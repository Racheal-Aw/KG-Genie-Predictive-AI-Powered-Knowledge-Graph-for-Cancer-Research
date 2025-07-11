# 🧬 KG Genie: AI-Powered Knowledge Graph for Cancer Research
KG Genie is an interactive, AI-enhanced platform for constructing and exploring biomedical knowledge graphs focused on cancer-related entities such as genes, pathways, diseases, and drug-target interactions. Built with Streamlit, it enables researchers to visualize complex relationships across oncogenes, cancer types, signalling pathways, and therapeutic active compounds.
# 🌐 **Visualise**
complex relationships (oncogenes ⇄ cancer types ⇄ signalling pathways ⇄ therapeutic compounds). ** 
# 🔮**Predict 
In a single click, determine whether a selected drug is *active* or *inactive* against its target, using a machine-learning model trained on molecular fingerprints and graph embeddings.**
## 🌟 Key Features
| Category | Highlights |
|----------|------------|
| **Knowledge Graph** | **Triplet‑based KG built from ChEMBL, DisGeNET, KEGG & more |**
| **Drug‑Activity Prediction** | One‑click predict to estimate *Active / Inactive* status with **88 % accuracy** |
| **Data Breadth** | **Gene‑pathway‑disease‑drug mapping, SMILES strings, binding affinities & approval status |**
| **Interactive UI** |** Sidebar filters, dropdown drug, gene, pathway and disease selector, zoomable graph, probability read‑outs |**
| **Modular** | **Easily extend to other therapeutic areas or data sources |**
---
## 🧠 Example Walk‑Through
1. **Select a drug** 
2. **Toggle “Run prediction?”
   * **Prediction**: *Active* (1) or *Inactive* (0)  
   * **Probability** (e.g. *0.92*)  
3. **Explore the KG** to see how that drug connects to:
   * Associated diseases  
   * Implicated genes & pathways  
   * Related approved therapies  
4. Swap to another drug—and repeat!
## 🚀 How to Use
| Action | What to Do |
|--------|------------|
| **Try it online** | Click the **“Live Demo”** link below—no installs needed. |
| **Local run** | `git clone …`, `pip install -r requirements.txt`, `streamlit run kggApp.py` |
| **Select & Predict** | In the sidebar: choose a drug → set **“Run prediction?”** to **Yes** → view result & probability. |
| **Graph navigation** | Pan/zoom, highlight neighbours, or filter entities to focus your discussion. |
# Live Demo
Try the app live here: 🏡 [Streamlit App](https://kg-genie-ai-powered-knowledge-graph-for-cancer-research-evzhhj.streamlit.app/)
# Setup (if running locally)
◘ Clone the repo
◘ Install dependencies: pip install -r requirements.txt
◘ Run the app: streamlit run kggApp.py
## 🙏 Acknowledgements
* **DisGeNET** – gene‑disease associations  
* **ChEMBL** – bioactivity data  
* **KEGG** – pathways  
* **Streamlit**, **NetworkX**, **RDKit** for the tech stack

---

*Happy exploring & predicting!* 🎉

