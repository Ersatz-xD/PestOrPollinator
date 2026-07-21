
# 🌿 PestOrPollinator: Ecological Latent Space Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning pipeline and Streamlit web application designed to classify insect species, retrieve visual neighbors using deep latent embeddings, and map high-dimensional visual clusters. 

This tool operates as an ecological monitor, differentiating between beneficial pollinators and harmful agricultural pests by analyzing the mathematical representations of their visual archetypes.

---

## System Architecture & Pipeline

The system bypasses standard end-to-end image classification in favor of a modular, feature-extraction-first approach. This allows for multi-modal analysis (classification, retrieval, and topological mapping) from a single forward pass.

1.  **Deep Feature Extraction:** A pre-trained **ResNet50** convolutional neural network (classification head removed) processes the input image into a dense 2,048-dimensional embedding vector.
2.  **Dimensionality Reduction Engine:** **PCA** (Principal Component Analysis) is utilized to compress the 2,048D space down to 100D, retaining maximal variance while cutting computational overhead by ~95% for rapid inference.
3.  **Classification Engine:** A high-dimensional **Logistic Regression** model classifies the embeddings into specific biological species.
4.  **Retrieval Engine:** A reverse-image search functionality implemented via **Cosine Similarity** to locate exact visual neighbors within the latent space:
    $$\text{similarity}(A, B) = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}}$$
5.  **Latent Topology:** **UMAP** (Uniform Manifold Approximation and Projection) projects the high-dimensional embeddings into an interactive 2D map, clustering mathematical similarities.

---

## Supervised Classification Performance

During model training, linear boundaries significantly outperformed tree-based partitioning (Random Forest, XGBoost) in the sparse 2,048D continuous space.

| Architecture | Feature Space | Peak Accuracy | Compute Trade-off |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 2,048D (Raw) | 88.6% | Baseline |
| **Linear SVM** | 2,048D (Raw) | 87.2% | Slower Training |
| **Random Forest** | 2,048D (Raw) | 74.1% | Poor High-D Scaling |
| **Logistic Regression + PCA** | 100D (Compressed)| 84.5% | 95% Memory Reduction |

**The Biological Ceiling:** Hyperparameter tuning peaked at an 88.6% ceiling. Confusion matrix analysis revealed that ResNet50 struggles to separate specific species pairs that share identical visual textures and background foliage, lacking the microscopic biological context to differentiate them.

---

## Unsupervised Visual Reality

To understand how the model "sees" the dataset natively, we bypassed the biological true labels and ran a **KMeans ($k=40$)** algorithm directly on the raw 2048D embeddings, projecting the results onto a UMAP 2D scatter plot.

**Key Finding:** The unsupervised KMeans clusters do *not* map perfectly to true biological species. Instead, the model grouped images into **visual archetypes** (e.g., "orange wings on dirt" vs "tiny white bugs on green leaves"). This visually proves why the supervised classifier peaked at 88.6%: a general-purpose vision model prioritizes visual texture and background context over strict biological taxonomy.

---

## Local Development & Setup

### 1. Repository Structure
Ensure your downloaded `embeddings.npy`, `labels.npy`, `umap_df.csv`, and `.pkl` model files are placed in their respective directories before running the application:

```text
📦 PestOrPollinator
 ┣ 📂 embeddings_and_labels
 ┃ ┣ 📜 embeddings.npy
 ┃ ┣ 📜 labels.npy
 ┃ ┗ 📜 umap_df.csv
 ┣ 📂 models
 ┃ ┣ 📜 best_insect_model_2048D.pkl
 ┃ ┣ 📜 best_insect_model_100D.pkl
 ┃ ┗ 📜 pca_transformer_100D.pkl
 ┗ 📜 app.py

```

### 2. Environment Setup

It is highly recommended to run this project in an isolated virtual environment to prevent dependency conflicts.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install core dependencies
pip install streamlit torch torchvision scikit-learn pandas plotly Pillow

```

### 3. Execution

```bash
streamlit run app.py

```

---

## ⚠️ Edge Cases & Limitations

* **Scale Invariance:** Images taken from extreme distances (where the subject occupies <10% of the frame) heavily skew the similarity search toward background texture matching rather than subject matching.
* **Out-of-Distribution Overconfidence:** If fed a non-insect image (e.g., a green leaf), the classifier will forcibly assign it to the nearest class based on background color, occasionally yielding high confidence scores. *Mitigation:* The secondary Retrieval Engine exposes these anomalies by returning low ($<0.80$) cosine similarity scores despite high classification confidence.

---

*Developed by Ayaan Ahmed Khan*



