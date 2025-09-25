import streamlit as st
import requests
import json
from pathlib import Path
import time
import pandas as pd
from PIL import Image
import io
import base64
import os

# Configuration
from config import API_BASE_URL, THEME_CONFIG, MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from shared.constants import VECTORIZATION_TIMEOUT

# Configuration de la page
st.set_page_config(
    page_title="BatMap",
    page_icon="🗺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le thème post-apocalyptique
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 50%, #FFD23F 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .mission-box {
        background-color: #1E1E1E;
        border-left: 5px solid #FF6B35;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .alert-box {
        background-color: #2D1B1B;
        border: 2px solid #FF4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #1B2D1B;
        border: 2px solid #44FF44;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #FF6B35;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'uploaded_maps' not in st.session_state:
        st.session_state.uploaded_maps = []
    if 'vectorization_results' not in st.session_state:
        st.session_state.vectorization_results = {}
    if 'api_status' not in st.session_state:
        st.session_state.api_status = "unknown"
    if 'binary_results' not in st.session_state:
        st.session_state.binary_results = None
    if 'processed_maps' not in st.session_state:
        st.session_state.processed_maps = []

def check_api_status():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = "online"
            return True
        else:
            st.session_state.api_status = "error"
            return False
    except requests.exceptions.RequestException:
        st.session_state.api_status = "offline"
        return False

def display_header():
    st.markdown("""
    <div class="main-header">
        <h1>🗺 BATMAP - TROUVE TON CHEMIN</h1>
    </div>
    """, unsafe_allow_html=True)

def display_mission_brief():
    with st.expander("Notre Mission", expanded=False):
        st.markdown("""
        <div class="mission-box">
            <h3>BatMap c'est quoi?</h3>
            <ul>
                <li>Une Application qui numérise les cartes physiques restantes</li>
                <li>...</li>
            </ul>
            <p><strong> ATTENTION :</strong> Système 100% offline - aucune connexion Internet autorisée</p>
        </div>
        """, unsafe_allow_html=True)

def display_api_status():
    api_online = check_api_status()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if api_online:
            st.markdown("""
            <div class="success-box">
                ✅ <strong>SYSTÈME OPÉRATIONNEL</strong> - API de vectorisation active
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box">
                ❌ <strong>SYSTÈME HORS LIGNE</strong> - Impossible de joindre l'API de vectorisation
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Vérifier Status"):
            with st.spinner("Vérification..."):
                time.sleep(1)
                st.rerun()
    
    with col3:
        status_color = "🟢" if api_online else "🔴"
        st.metric("Status API", f"{status_color} {st.session_state.api_status.upper()}")

def upload_map_interface():
    st.header("Ajouter une carte")
    
    with st.form("upload_form"):
        st.subheader("1. Sélection et Paramètres")
        
        uploaded_file = st.file_uploader(
            "Sélectionnez votre carte",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=False,
            help=f"Taille maximum : {MAX_FILE_SIZE // (1024*1024)}MB par fichier"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            scale = st.number_input("Échelle de la carte", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        with col2:
            city = st.text_input("Ville/Zone", placeholder="Ex: Paris, Zone Nord...")
        
        if uploaded_file:
            st.subheader("Fichier Sélectionné")
            df = pd.DataFrame([{
                "Nom": uploaded_file.name,
                "Taille": f"{uploaded_file.size / 1024:.1f} KB",
                "Type": uploaded_file.type,
                "Status": "Prêt à traiter"
            }])
            st.dataframe(df, use_container_width=True)
            
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Prévisualisation de la carte", use_column_width=True)
                uploaded_file.seek(0)
            except Exception as e:
                st.warning(f"Impossible d'afficher l'aperçu: {e}")
        
        submitted = st.form_submit_button("Numériser", type="primary")
        if submitted:
            if not uploaded_file:
                st.error("❌ Veuillez sélectionner une carte")
            elif not city:
                st.error("❌ Veuillez indiquer la ville/zone")
            else:
                process_map_with_your_api(uploaded_file, scale, city)

def process_map_with_your_api(uploaded_file, scale, city):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(f"🔄 Numérisation de {uploaded_file.name}...")
        progress_bar.progress(0.2)
        
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"scale": scale, "city": city}
        
        progress_bar.progress(0.4)
        status_text.text("Veuillez patienter...")
        
        response = requests.post(
            f"{API_BASE_URL}/map/binaryse",
            files=files,
            data=data,
            timeout=VECTORIZATION_TIMEOUT
        )
        
        progress_bar.progress(0.8)
        
        if response.status_code == 200:
            st.session_state.binary_results = response.json()
            progress_bar.progress(1.0)
            status_text.text("✅ Traitement terminé!")
            st.success(f"{uploaded_file.name} Opération réussie!")
            st.info("👇 Choisissez une carte.")
            time.sleep(2)
            st.rerun()
        else:
            progress_bar.progress(1.0)
            status_text.text("❌ Erreur de traitement")
            st.error(f"❌ Erreur lors du traitement: {response.status_code}")
            if response.text:
                st.error(f"Détail: {response.text}")
                
    except requests.exceptions.RequestException as e:
        progress_bar.progress(1.0)
        status_text.text("❌ Erreur de connexion")
        st.error(f"❌ Erreur de connexion: {str(e)}")
    except Exception as e:
        progress_bar.progress(1.0)
        status_text.text("❌ Erreur inattendue")
        st.error(f"❌ Erreur inattendue: {str(e)}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

def display_binary_selection():
    if not st.session_state.binary_results:
        return
    
    st.header("Sélectionner ma carte")
    st.subheader("2. Choisissez la meilleure extraction")
    
    results = st.session_state.binary_results.get('results', {})
    if not results:
        st.warning("Aucun résultat trouvé dans la réponse de l'API.")
        return
    
    st.info(f"Carte: **{st.session_state.binary_results.get('filename')}** - Ville: **{st.session_state.binary_results.get('city')}** - Échelle: **{st.session_state.binary_results.get('scale')}**")
    
    options = []
    cols = st.columns(min(3, len(results)))
    
    for idx, (path, base64_img) in enumerate(results.items()):
        with cols[idx % len(cols)]:
            try:
                img_data = base64.b64decode(base64_img)
                img = Image.open(io.BytesIO(img_data))
                st.image(img, caption=f"Option {idx + 1}", use_column_width=True)
                method_name = "Otsu" if "method_0" in path else "Adaptive Mean" if "method_1" in path else "Autre"
                st.caption(f"Méthode: {method_name}")
                options.append(path)
            except Exception as e:
                st.error(f"Erreur d'affichage pour {path}: {e}")
    
    if options:
        selected_idx = st.radio("Sélectionnez la meilleure extraction:", range(len(options)), format_func=lambda x: f"Option {x + 1}", key="binary_choice", horizontal=True)
        
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("✅ Valider ce Choix", type="primary", use_container_width=True):
                finalize_map_choice(options[selected_idx])
        
        with col3:
            if st.button("🔄 Nouveau Traitement"):
                st.session_state.binary_results = None
                st.rerun()

def finalize_map_choice(selected_path):
    with st.spinner("🔄 Finalisation en cours..."):
        try:
            data = {
                "map_choosen": selected_path,
                "city": st.session_state.binary_results['city'],
                "scale": str(st.session_state.binary_results['scale']),
                "original_filename": st.session_state.binary_results['filename']
            }

            response = requests.post(
                f"{API_BASE_URL}/map/add",
                data=data,
                timeout=VECTORIZATION_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                map_info = {
                    "filename": st.session_state.binary_results['filename'],
                    "city": st.session_state.binary_results['city'],
                    "scale": st.session_state.binary_results['scale'],
                    "selected_path": selected_path,
                    "timestamp": time.time(),
                    "api_result": result
                }
                st.session_state.processed_maps.append(map_info)
                st.session_state.binary_results = None
                st.success("Carte ajoutée avec succès!")
                with st.expander("Détails de la carte ajoutée", expanded=True):
                    st.json(result)
                time.sleep(3)
                st.rerun()
            else:
                st.error(f"❌ Erreur lors de la finalisation: {response.status_code}")
                if response.text:
                    st.error(f"Détail: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Erreur de connexion: {str(e)}")
        except Exception as e:
            st.error(f"❌ Erreur inattendue: {str(e)}")

def display_processed_maps_summary():
    if 'processed_maps' in st.session_state and st.session_state.processed_maps:
        st.header("Cartes récemment ajoutées.")
        df = pd.DataFrame([{
            "Fichier": m["filename"],
            "Ville": m["city"],
            "Échelle": m["scale"],
            "Horodatage": time.strftime("%H:%M:%S", time.localtime(m["timestamp"])),
            "Status": "✅ Finalisé"
        } for m in st.session_state.processed_maps])
        st.dataframe(df, use_container_width=True)
        
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"batmap_session_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def display_statistics():
    st.header("Statistiques")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>🗺️</h3><h2>{len(st.session_state.vectorization_results)}</h2><p>Cartes Traitées</p></div>", unsafe_allow_html=True)
    with col2:
        success_rate = len([r for r in st.session_state.vectorization_results.values() if r.get('success')]) / max(len(st.session_state.vectorization_results), 1) * 100
        st.markdown(f"<div class='metric-card'><h2>{success_rate:.1f}%</h2><p>Taux de Réussite</p></div>", unsafe_allow_html=True)
    with col3:
        total_size = sum([len(str(r)) for r in st.session_state.vectorization_results.values()]) / 1024
        st.markdown(f"<div class='metric-card'><h2>{total_size:.1f}KB</h2><p>Données Sauvées</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><h2>100%</h2><p>Sécurité Offline</p></div>", unsafe_allow_html=True)

def main():
    initialize_session_state()
    display_header()
    display_mission_brief()
    display_api_status()
    tab1, tab2 = st.tabs(["Ajout", "Toutes les cartes"])
    with tab1:
        upload_map_interface()
        if st.session_state.binary_results:
            st.divider()
            display_binary_selection()
    with tab2:
        display_processed_maps_summary()
        
        try:
            response = requests.get(f"{API_BASE_URL}/maps", timeout=10)
            if response.status_code == 200:
                maps = response.json()
                if maps:
                    df = pd.DataFrame(maps)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Aucune carte disponible dans l'API.")
            else:
                st.error(f"Erreur lors de la récupération: {response.status_code}")
                if response.text:
                    st.error(f"Détail: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API: {str(e)}")

if __name__ == "__main__":
    main()
