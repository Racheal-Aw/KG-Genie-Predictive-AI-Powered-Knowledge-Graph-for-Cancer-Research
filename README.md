# 🧬 KG Genie: AI-Powered Knowledge Graph for Cancer Research
KG Genie is an interactive, AI-enhanced platform for constructing and exploring biomedical knowledge graphs focused on cancer-related entities such as genes, pathways, diseases, and drug-target interactions. Built with Streamlit, it enables researchers to visualize complex relationships across oncogenes, cancer types, signalling pathways, and therapeutic active compounds.
# 🌟 Key Features
🔎 Triplet-based knowledge graph from public biomedical databases (ChEMBL, DisGeNET, etc.)
🧠 Gene-pathway-disease-drug mapping using curated and programmatically extracted data
🧪 Bioactivity insights with SMILES, binding affinities, and drug approval status
🖼️ Interactive Streamlit interface for querying and visualizing cancer mechanisms
📚 Modular design for easy extension to other diseases or datasets
# 🧠 Example Use Case
A user-friendly interface allows you to input house features easily.
Explore connections such as:
•	Select a disease from the dropdown to view its associated pathway, gene targets, and FDA-approved cancer drugs for treatment.
•	Choose a pathway to see related diseases, gene targets, and approved therapies.
Examples include:
◘ BRCA1 → Homologous recombination pathway → Triple Negative Breast Cancer
◘ EGFR → PI3K-Akt signaling → Lung Cancer and targeted treatments from ChEMBL
◘ TP53 → p53 signaling → Glioblastoma, Breast, and Colorectal cancers
# How to Use
This app is designed for everyone to explore various types of cancer, learn about the targets of cancer drugs, and understand the pathways they use to inhibit cancer growth. No programming skills are needed — just open the live app URL, enter the features, and receive an estimated price instantly.
# Live Demo
Try the app live here: 🏡 [Streamlit App](https://kg-genie-ai-powered-knowledge-graph-for-cancer-research-evzhhj.streamlit.app/)
# Setup (if running locally)
Clone the repo
Install dependencies: pip install -r requirements.txt
Run the app: streamlit run kggApp.py
# 🙏 Acknowledgments
◘ DisGeNET
◘ ChEMBL
◘ KEGG
◘ Streamlit, NetworkX, and RDKit (for chemical structures)

