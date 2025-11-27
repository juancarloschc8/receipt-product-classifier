import pandas as pd
import numpy as np
import random
import re

def degrade_text(text):
    """
    Simula la entropía de un ticket de compra:
    - Eliminación de vocales (LECHE -> LCH)
    - Truncamiento (LALA -> LA)
    - Abreviaturas numéricas (1 LITRO -> 1L)
    """
    text = text.upper()
    
    # Simular abreviaturas comunes en Retail
    replacements = {
        'LITRO': 'L', 'LITROS': 'L', 'MILILITROS': 'ML', 
        'GRAMOS': 'GR', 'KILOGRAMO': 'KG', 'PIEZA': 'PZA',
        'ENTERA': 'ENT', 'DESLACTOSADA': 'DESLAC', 'BOTELLA': 'BOT'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    words = text.split()
    new_words = []
    for w in words:
        # 30% de probabilidad de perder vocales si la palabra es larga
        if len(w) > 3 and random.random() < 0.3:
            w = re.sub(r'[AEIOU]', '', w)
        # 20% de probabilidad de truncarse
        if len(w) > 4 and random.random() < 0.2:
            w = w[:4]
        new_words.append(w)
    
    return " ".join(new_words)

def generate_dataset(n_samples=2000):
    """Genera un dataset balanceado de productos y sus versiones 'ticket'."""
    
    # Jerarquía: Categoría -> Lista de Productos Base
    taxonomy = {
        'LACTEOS': [
            'LECHE LALA ENTERA 1 LITRO', 'LECHE ALPURA DESLACTOSADA', 
            'YOGURT YOPLAIT FRESA 1KG', 'QUESO PANELA 400 GRAMOS',
            'MANTEQUILLA SIN SAL 90 GRAMOS', 'CREMA LALA ACIDA'
        ],
        'BEBIDAS': [
            'COCA COLA 600 MILILITROS', 'SPRITE 2 LITROS', 
            'AGUA BONAFONT 1 LITRO', 'JUGO DEL VALLE MANZANA',
            'CERVEZA MODELO ESPECIAL BOTELLA', 'PEPSI LIGHT'
        ],
        'LIMPIEZA': [
            'DETERGENTE ARIEL 1 KILOGRAMO', 'JABON ZOTE ROSA', 
            'CLORO CLORALEX 1 LITRO', 'SUAVITEL MOMENTOS MAGICOS',
            'LAVATRASTES SALVO LIMON', 'ESPONJA SCOTCH BRITE'
        ],
        'ABARROTES': [
            'ATUN TUNY EN AGUA', 'MAYONESA MCCORMICK', 
            'SOPA NISSI VASO CAMARON', 'FRIJOLES LA SIERRA BAYOS',
            'ARROZ VERDE VALLE 1KG', 'ACEITE 123 GIRASOL'
        ]
    }
    
    data = []
    for _ in range(n_samples):
        cat = random.choice(list(taxonomy.keys()))
        clean_name = random.choice(taxonomy[cat])
        dirty_name = degrade_text(clean_name)
        
        data.append({
            'clean_product': clean_name,
            'receipt_text': dirty_name,  # Input sucio (X)
            'category': cat              # Target (y)
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset(10)
    print(df.head())