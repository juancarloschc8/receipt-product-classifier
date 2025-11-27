import pandas as pd
import numpy as np
import random
import re

def degrade_text(text):
    """
    Versión AGRESIVA de degradación de texto.
    """
    text = text.upper()
    
    # 1. Abreviaturas confusas
    replacements = {
        'LITRO': 'L', 'LITROS': 'L', 'MILILITROS': 'ML', 
        'GRAMOS': 'G', 'KILOGRAMO': 'KG', 'PIEZA': 'PZ',
        'ENTERA': 'ENT', 'DESLACTOSADA': 'DES', 'BOTELLA': 'BOT',
        'CHOCOLATE': 'CHOC', 'LIMON': 'LIM', 'JABON': 'JBN'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    words = text.split()
    new_words = []
    for w in words:
        # 50% de probabilidad de perder vocales (Mucha suciedad)
        if len(w) > 3 and random.random() < 0.5:
            w = re.sub(r'[AEIOU]', '', w)
        # 30% de probabilidad de truncarse
        if len(w) > 4 and random.random() < 0.3:
            w = w[:3] # Cortar agresivamente
        new_words.append(w)
    
    # A veces pegar palabras (Error de ticket común) "LECHELALA"
    if len(new_words) > 1 and random.random() < 0.2:
        idx = random.randint(0, len(new_words)-2)
        new_words[idx] = new_words[idx] + new_words[idx+1]
        del new_words[idx+1]

    return " ".join(new_words)

def generate_dataset(n_samples=2000):
    # Taxonomía con "Trampas" (Items ambiguos)
    taxonomy = {
        'LACTEOS': [
            'LECHE LALA ENTERA', 'LECHE CHOCOLATE LALA', 
            'YOGURT FRESA', 'QUESO OAXACA', 'CREMA LALA',
            'LECHE ALMENDRA SILK' # Confuso con abarrotes
        ],
        'BEBIDAS': [
            'COCA COLA', 'AGUA MINERAL', 'JUGO MANZANA', 
            'CERVEZA MODELO', 'BEBIDA DE ALMENDRA', # Confuso con lacteos
            'AGUA DE COCO' # Confuso con fruta/jabon
        ],
        'LIMPIEZA': [
            'DETERGENTE ARIEL', 'JABON ZOTE', 'CLORO CLORALEX', 
            'SUAVITEL AROMA FRESA', # Confuso con yogurt fresa
            'JABON LIQUIDO COCO', # Confuso con comida
            'AROMA LIMON'
        ],
        'ABARROTES': [
            'GALLETAS CHOCOLATE', # Confuso con leche chocolate
            'ALMENDRAS TOSTADAS', # Confuso con leche almendra
            'COCO RALLADO', # Confuso con jabon/agua
            'SOPA DE CODITOS'
        ]
    }
    
    data = []
    for _ in range(n_samples):
        cat = random.choice(list(taxonomy.keys()))
        clean_name = random.choice(taxonomy[cat])
        
        # Agregamos variación numérica aleatoria para que no sean idénticos
        qty = random.choice(['1L', '600ML', '1KG', '500G', 'PACK', ''])
        full_clean = f"{clean_name} {qty}".strip()
        
        dirty_name = degrade_text(full_clean)
        
        data.append({
            'clean_product': full_clean,
            'receipt_text': dirty_name,
            'category': cat
        })
        
    return pd.DataFrame(data)