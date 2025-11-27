import re
import unicodedata

def normalize_text(text):
    """
    Pipeline de limpieza para normalizar texto de tickets.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. A minusculas
    text = text.lower()
    
    # 2. Eliminar acentos
    text = "".join(c for c in unicodedata.normalize("NFD", text) 
                   if unicodedata.category(c) != "Mn")
    
    # 3. Regex: Mantener solo letras, números, % y puntos (para 1.5L)
    # Se eliminan símbolos extraños que a veces traen los OCRs
    text = re.sub(r'[^a-z0-9\s\.\%]', '', text)
    
    # 4. Colapsar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text