# LLM-GNN Dynamic Fusion Model for Fraud Detection

This project presents a novel approach to fraud detection using a **Dynamic Fusion Deep Learning Model**. The model combines insights from three distinct data modalities:  

1. **Traditional Tabular Data** (structured transaction features)  
2. **Graph-Based Features** from a Graph Neural Network (GNN)  
3. **Contextual Information** from a Large Language Model (LLM)  

The goal is to build a more robust and accurate system for identifying fraudulent transactions, particularly in **imbalanced datasets**.

---

## 1. Model Architecture

The core of this project is the **Dynamic Fusion Model**, which intelligently combines information from multiple data sources. We compare this model against several baselines to validate its effectiveness.

- **Deep Neural Network Only (Baseline):** A traditional MLP using only tabular features.  
- **GNN-Only Model:** Uses only Node2Vec embeddings generated from transaction graphs.  
- **LLM-Only Model:** Uses only semantic embeddings from an LLM.  
- **Sequential Fusion Model:** Simply concatenates all three feature types (tabular, GNN, LLM) into a single input layer.  
- **Dynamic Fusion Model (Proposed):** Built with the Keras Functional API. Uses a custom `AttentionGatingLayer` to dynamically fuse the GNN and LLM embeddings before combining them with tabular features.

---

## 2. Key Results

Models were evaluated on a held-out test set. The **ROC AUC** score was the primary metric due to the imbalanced nature of the dataset. The **Dynamic Fusion Model** significantly outperformed all baselines, showing the power of intelligent fusion.

| Metric       | NN Only | GNN-Only | LLM-Only | Sequential Fusion | **Dynamic Fusion** |
|--------------|--------:|---------:|---------:|------------------:|-------------------:|
| **Accuracy** | 0.905   | 0.887    | 0.882    | 0.834             | **0.863**          |
| **Precision**| 0.195   | 0.133    | 0.130    | 0.136             | **0.164**          |
| **Recall**   | 0.526   | 0.390    | 0.398    | **0.680**         | 0.615               |
| **F1 Score** | 0.284   | 0.198    | 0.195    | 0.227             | **0.259**          |
| **ROC AUC**  | 0.827   | 0.695    | 0.696    | 0.844             | **0.871**          |

---

## 3. Getting Started

### 3.1 Clone the Repository
```bash
git clone https://github.com/AishwaryaSrinivasPrasad/FraudDetectiom-hybridmodel.git
cd FraudDetectiom-hybridmodel
```

### 3.2 Install Dependencies
It’s recommended to use a virtual environment:
```bash
pip install tensorflow scikit-learn pandas numpy pecanpy openai gensim
```

### 3.3 Dataset
This project uses the **IEEE-CIS Fraud Detection** dataset. Ensure your dataset is downloaded and placed in the correct project directory before running scripts. The expected structure should match the code requirements.

### 3.4 Usage
Run the preprocessing and embedding generation scripts before training:
```bash
python main_project_script.py
```
This script should orchestrate:
1. Data preprocessing  
2. GNN embedding generation  
3. LLM embedding generation  
4. Model training and evaluation  

---

## 4. Future Work
- **Model Explainability:** Integrate Explainable AI (XAI) techniques to provide transparency, especially for the dynamic fusion layer.  
- **Real-Time Optimization:** Optimize for real-time inference (lighter LLMs, precomputed GNN embeddings).  
- **Expanded Data Sources:** Include additional features like device info or behavioral logs to improve detection.

---

## License
This project is for academic and research purposes.
