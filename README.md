
# Receipt Product Categorization: NLP & Transformers Prototype

**Status:** Prototype / PoC  
**Stack:** Python, Scikit-Learn, HuggingFace Transformers, PyTorch  
**Goal:** Clean and classify messy retail receipt text into standardized categories.

## The Business Problem
In the CPG (Consumer Packaged Goods) industry, receipt data is invaluable but "dirty". Text comes from OCR or legacy POS systems with inconsistent formatting:
* Real Product: "Leche Lala Entera 1 Litro"
* Receipt Text: LECH. LALA ENT 1L

Standard keyword matching fails here. This project compares probabilistic approaches to solve this ambiguity.

## Methodology

### 1. Data Simulation
Since real receipt data is proprietary, I built a *Data Augmentation Engine* (src/data_gen.py) that generates synthetic "messy" text by simulating:
* Vowel removal (OCR errors).
* Aggressive abbreviation (Retail slang).
* Truncation.

### 2. Model Comparison
We benchmarked two architectures to find the optimal balance between accuracy and computational cost:


| Approach      | Technology                                | Pros                                           | Cons                                           |
|--------------|-------------------------------------------|-----------------------------------------------|-----------------------------------------------|
| *Baseline*   | TF-IDF + Random Forest                   | Ultra-fast inference (<5ms), Low CPU usage.  | Struggles with completely unseen abbreviations. |
| *Challenger* | DistilBERT Embeddings + Logistic Regression | Context-aware, Semantic understanding.       | High latency (~50ms), Requires heavy RAM/GPU. |

## Key Findings
* The **DistilBERT** model achieved slightly higher accuracy (+3%) by understanding semantic context.
* However, for a real-time high-throughput system, the *TF-IDF Baseline* is recommended due to its superior efficiency (20x faster).

## Project Structure
* notebooks/: Contains the main analysis and visualization logic.
* src/: Modular code for text cleaning, data generation, and model training.

## How to Run
1.  Install requirements: pip install -r requirements.txt
2.  Open the prototype: jupyter notebook notebooks/Receipt_Classification_Prototyping.ipynb
