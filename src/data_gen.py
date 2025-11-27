import pandas as pd
import numpy as np
import random
import re

def degrade_text(text):
    """
    Simula tickets de muy baja calidad.
    """
    text = text.upper()
    
    # 1. Abreviaturas agresivas de unidades y conectores
    replacements = {
        'LITRO': 'L', 'ML': '', 'GRAMOS': 'G', 'DE': '', 'CON': '',
        'SABOR': '', 'TIPO': '', 'PIEZA': '', 'BOTELLA': 'BOT'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    words = text.split()
    
    # 2. Destrucción de caracteres
    new_words = []
    for w in words:
        # Si la palabra es clave (ej: LECHE), 30% de probabilidad de borrarla completamente
        # Esto simula un ticket mal impreso donde falta la palabra principal
        if random.random() < 0.15: 
            continue
            
        # Eliminar vocales (Ruido OCR clásico)
        if random.random() < 0.6:
            w = re.sub(r'[AEIOU]', '', w)
            
        # Truncar al final (JABON -> JAB)
        if len(w) > 3 and random.random() < 0.4:
            w = w[:3]
            
        new_words.append(w)
    
    # Si borramos todo, regresamos al menos un caracter basura para no romper el código
    if not new_words:
        return "ITEM_DESC"
        
    return " ".join(new_words)

def generate_dataset(n_samples=5000):
    """
    Genera dataset con AMBIGÜEDAD SEMÁNTICA.
    La misma palabra clave (ej: COCO) aparece en múltiples categorías.
    """
    
    # Notarás que 'COCO', 'FRESA', 'MANZANA' están en todas partes.
    taxonomy = {
        'LACTEOS': [
            'LECHE DE COCO', 'YOGURT SABOR FRESA', 'LECHE DE ALMENDRA', 
            'BATIDO DE CHOCOLATE', 'CREMA DE AVELLANA', 'HELADO DE VAINILLA'
        ],
        'BEBIDAS': [
            'AGUA DE COCO', 'JUGO DE FRESA', 'BEBIDA DE ALMENDRA', 
            'REFRESCO CHOCOLATE', 'AGUA SABOR VAINILLA', 'LICOR DE AVELLANA'
        ],
        'LIMPIEZA': [
            'JABON AROMA COCO', 'DETERGENTE OLO FRESA', 'SHAMPOO DE ALMENDRA', 
            'SUAVITEL AROMA CHOCOLATE', 'LIMPIADOR VAINILLA', 'JABON AVELLANA'
        ],
        'ABARROTES': [
            'COCO RALLADO SECO', 'MERMELADA DE FRESA', 'ALMENDRAS ENTERAS', 
            'BARRA DE CHOCOLATE', 'ESENCIA DE VAINILLA', 'CREMA DE AVELLANA'
        ]
    }
    
    data = []
    for _ in range(n_samples):
        cat = random.choice(list(taxonomy.keys()))
        base_product = random.choice(taxonomy[cat])
        
        # Generar texto sucio
        dirty_name = degrade_text(base_product)
        
        data.append({
            'clean_product': base_product, # Texto original (para referencia humana)
            'receipt_text': dirty_name,    # Input del modelo (muy sucio)
            'category': cat                # Target
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset(10)
    print(df[['clean_product', 'receipt_text', 'category']])