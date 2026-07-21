
# Project Findings: Butterfly Species Classification & Similarity Search

This document summarizes the core findings from analyzing ResNet50 embeddings extracted from a 40-species butterfly image dataset. The project explored both Supervised Learning (Classification) and Unsupervised Learning (Similarity Search & Clustering) to understand how deep learning models interpret biological data.

---

## 1. Supervised Classification Performance

We tested multiple machine learning architectures to classify the 2,048-dimensional ResNet50 embeddings. 

*   **Best Classifier:** Logistic Regression (Default Parameters)
*   **Peak Accuracy:** 88.6%
*   **Key Insight on Linear vs. Complex Boundaries:** Linear models significantly outperformed tree-based models (Random Forest, XGBoost). In a high-dimensional continuous space like 2048D, data points are sparse. Drawing a single hyperplane (Linear SVM / Logistic Regression) is far more effective than trying to box off boundaries (Trees).

### The Biological Ceiling & Confusion Matrix
Hyperparameter tuning (via GridSearchCV) failed to push accuracy past the 88.6% ceiling. The Confusion Matrix revealed why:
*   **Repeated Errors:** The model persistently confused specific species pairs, most notably misclassifying **Class 79 as Class 36** up to 9 times.
*   **Conclusion:** This is an embedding-level issue, not a classifier issue. ResNet50 placed the high-dimensional coordinates of these two species completely on top of each other. The model lacks the microscopic biological context to separate them.

---

## 2. Dimensionality Reduction (PCA Optimization)

To optimize the model for potential edge-device deployment, we applied Principal Component Analysis (PCA) to compress the feature space.

*   **Original Space (2,048 Dimensions):** 88.6% Accuracy
*   **Compressed Space (100 Dimensions):** 84.5% Accuracy
*   **Conclusion:** We sacrificed ~4% accuracy to achieve a **95% reduction in memory and compute requirements**. This tradeoff is highly viable for real-world mobile applications where speed and resource constraints are critical.

---

## 3. Reverse Image Search (Similarity Search)

We built a similarity search engine utilizing **Cosine Similarity** to measure the angular distance between vectors, which effectively ignores lighting and brightness magnitudes to focus on pure pattern matching.

**Search Observations:**
When querying specific images against the dataset, the engine reliably returned visual neighbors:
*   **Querying Class 44:** Returned two Class 44 matches and three Class 86 matches. The similarity score delta between the correct species and the incorrect species was less than `0.0008`.
*   **Querying Class 30:** Returned a mix of Classes 30, 77, 61, 32, and 2, all clustered within tight similarity bounds (`0.852` down to `0.837`).
*   **Conclusion:** The engine successfully acts as a "visual neighbor" locator, confirming that ResNet50 groups mathematically similar images tightly together, even if they span across different true species.

---

## 4. Unsupervised Clustering & Visual Mapping

To visualize the high-dimensional data, we utilized **UMAP** to project the 2,048D space down to a 2D map, preserving both local neighborhoods and global structure.

### Biological Reality vs. Visual Reality
We compared two mapping strategies:
1.  **Colored by True Labels (Biological):** The UMAP plot revealed distinct "islands" for highly unique species, but a massive, overlapping central "continent" where dozens of species visually merged.
2.  **Colored by KMeans Clusters (Visual, k=40):** We ran KMeans clustering on the raw 2,048D data, ignoring true labels, to see if the model could "discover" the 40 species on its own. 

**Final Insight:** 
The KMeans clusters did *not* perfectly align with true biological species. Instead of separating the overlapping central continent into species, KMeans carved it into solid "visual territories." 

ResNet50 grouped images based on **visual archetypes** (e.g., shared color palettes, similar wing shapes, or specific background foliage) rather than biological taxonomy. This completely explains the supervised classification ceiling: to a general-purpose vision model trained on ImageNet, several distinct biological species are mathematically indistinguishable based on visual features alone.

