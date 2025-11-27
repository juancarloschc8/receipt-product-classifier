import time
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# --- ENFOQUE 1: CLÁSICO (TF-IDF) ---
def train_classic_model(df, text_col, label_col):
    """Entrena un Random Forest con TF-IDF."""
    print(">>> Entrenando Modelo Clásico (TF-IDF + Random Forest)...")
    start = time.time()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=0.2, random_state=42
    )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Evaluar
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return pipeline, {
        'accuracy': acc,
        'time': train_time,
        'report': classification_report(y_test, y_pred, output_dict=True),
        'X_test': X_test,
        'y_test': y_test
    }

# --- ENFOQUE 2: MODERNO (TRANSFORMERS) ---
def get_transformer_embeddings(text_list, model_name='distilbert-base-uncased', batch_size=32):
    """
    Genera embeddings usando un modelo pre-entrenado (Feature Extraction).
    No hacemos fine-tuning completo para mantenerlo ligero en el prototipo.
    """
    print(f">>> Generando Embeddings con {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Mover a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_embeddings = []
    
    # Procesar por lotes para no saturar memoria
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                           max_length=64, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Usamos el token CLS (primera posición) como vector de la frase
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(batch_embeddings)
        
    return np.vstack(all_embeddings)

def train_transformer_head(embeddings, labels):
    """Entrena un clasificador ligero (Logistic Reg) sobre los embeddings."""
    print(">>> Entrenando Clasificador sobre Embeddings...")
    start = time.time()
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    # Usamos Regresión Logística porque los embeddings ya son linealmente separables
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return clf, {
        'accuracy': acc,
        'time': train_time, # Nota: Esto no incluye el tiempo de generación de embeddings
        'report': classification_report(y_test, y_pred, output_dict=True)
    }