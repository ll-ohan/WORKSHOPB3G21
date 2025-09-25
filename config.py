import os
from pathlib import Path

# Configuration des chemins
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR.parent / "storage" if (BASE_DIR.parent / "storage").exists() else BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CACHE_DIR = BASE_DIR / "data" / "cache"

# Création des dossiers s'ils n'existent pas
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# API
API_BASE_URL = "http://localhost:8000"
VECTORIZATION_TIMEOUT = 300  # 5 minutes


# Limites fichiers
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]

# Limites fichiers
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]

# Configuration du thème
THEME_CONFIG = {
    "primaryColor": "#FFDD35",
    "backgroundColor": "#0E1117", 
    "secondaryBackgroundColor": "#262730",
    "textColor": "#FAFAFA"
}

# Configuration Streamlit
STREAMLIT_CONFIG = {
    "page_title": "BatMap",
    "page_icon":"🗺",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Messages d'interface
MESSAGES = {
    "welcome": "BATMAP - TROUVER SON CHEMIN",
    "mission": "Mission : Numérisation sécurisée des cartes post-Ultron",
    "api_online": "✅ SYSTÈME OPÉRATIONNEL - API de vectorisation active",
    "api_offline": "❌ SYSTÈME HORS LIGNE - Impossible de joindre l'API de vectorisation",
    "upload_success": "✅ Carte uploadée avec succès",
    "upload_error": "❌ Erreur lors de l'upload",
    "processing": "🔄 Traitement en cours...",
    "processing_complete": "✅ Traitement terminé"
}

# Configuration de sécurité (mode post-apocalyptique)
SECURITY_CONFIG = {
    "offline_mode": True,
    "encryption_enabled": True,
    "ultron_detection": False,
    "resistance_level": "maximum"
}
