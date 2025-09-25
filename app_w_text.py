import streamlit as st
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
import heapq
from PIL import Image
import io
import math

def load_image_from_bytes(image_bytes):
    """Charger une image depuis des bytes"""
    image = Image.open(io.BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def load_mask_from_bytes(image_bytes):
    """Charger un masque depuis des bytes"""
    image = Image.open(io.BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

def barreRoute(mask, pts):
    """Barrer une route sur le masque"""
    if len(pts) == 2:
        x1, y1 = map(int, pts[0])
        x2, y2 = map(int, pts[1])
        mask_barr = mask.copy()
        cv2.line(mask_barr, (x1, y1), (x2, y2), 0, thickness=8)
        st.success(f'Route barrée entre ({x1},{y1}) et ({x2},{y2}) sur le masque.')
        return mask_barr
    else:
        st.warning('Deux points nécessaires pour barrer une route.')
        return mask

def findNearestRoutePoints(mask, points, k=100):
    """Trouver les points de route les plus proches"""
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, ncomponents = label(mask == 255, structure=structure)
    route_pixels = np.column_stack(np.where(mask == 255))  # (y, x)
    route_pixels_xy = np.fliplr(route_pixels)  # (x, y)
    kdtree = cKDTree(route_pixels_xy)
    all_candidates = []
    for pt in points:
        dists, idxs = kdtree.query(pt, k=min(k, len(route_pixels_xy)))
        if np.isscalar(idxs):
            idxs = [idxs]
        candidates = [tuple(int(x) for x in route_pixels_xy[i]) for i in idxs]
        all_candidates.append(candidates)
    return all_candidates

def computeShortestPath(mask, all_candidates):
    """Calculer le chemin le plus court"""
    grid = (mask == 255).astype(np.uint8)
    height, width = grid.shape

    def neighbors(x, y):
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny, nx]:
                    yield nx, ny

    def dijkstra(start, end):
        dist = np.full((height, width), np.inf)
        prev = np.full((height, width, 2), -1, dtype=int)
        dist[start[1], start[0]] = 0
        heap = [(0, start)]
        while heap:
            d, (x, y) = heapq.heappop(heap)
            if (x, y) == end:
                break
            for nx, ny in neighbors(x, y):
                alt = d + 1
                if alt < dist[ny, nx]:
                    dist[ny, nx] = alt
                    prev[ny, nx] = [x, y]
                    heapq.heappush(heap, (alt, (nx, ny)))
        path = []
        x, y = end
        while (x, y) != start:
            path.append((x, y))
            x, y = prev[y, x]
            if x == -1:
                return []
        path.append(start)
        path.reverse()
        return path

    found = False
    for start in all_candidates[0]:
        for end in all_candidates[1]:
            start = tuple(map(int, start))
            end = tuple(map(int, end))
            path = dijkstra(start, end)
            if path:
                found = True
                break
        if found:
            break

    if not found or not path:
        return None, None, []
    return start, end, path

def drawPath(img, path, start, end):
    """Dessiner le chemin sur l'image"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for (x, y) in path:
        cv2.circle(img_rgb, (x, y), 1, (255,0,0), -1)
    cv2.circle(img_rgb, start, 4, (0,255,0), -1)
    
    # Dessiner un drapeau à la fin
    flag_size = 12
    flag_base = np.array([
        [end[0], end[1]],
        [end[0] - flag_size//2, end[1] + flag_size],
        [end[0] + flag_size//2, end[1] + flag_size]
    ], np.int32)
    cv2.fillPoly(img_rgb, [flag_base], (255,0,0))
    cv2.line(img_rgb, (end[0], end[1]), (end[0], end[1] + flag_size), (0,0,0), 2)
    return img_rgb

def pixels_to_distance(length_pixels, scale_m_per_cm):
    """Convertir les pixels en distance réelle selon l'échelle
    
    Étapes :
    1. pixels -> cm : pixels * 0.02646
    2. cm -> mètres réels : produit en croix selon l'échelle
    
    Exemple : 600 pixels, échelle 1cm = 200m
    - 600 * 0.02646 = 15.88 cm
    - 15.88 cm * 200 m/cm = 3176 m
    """
    cm = length_pixels * 0.02646
    meters = cm * scale_m_per_cm
    return meters

import math

# def distance(p1, p2):
#     """Distance euclidienne entre deux points"""
#     return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

# def angle_direction(p1, p2, p3):
#     """Détermine l'angle et la direction (gauche/droite)"""
#     v1 = (p2[0]-p1[0], p2[1]-p1[1])
#     v2 = (p3[0]-p2[0], p3[1]-p2[1])
    
#     # Produit vectoriel pour savoir gauche/droite
#     cross = v1[0]*v2[1] - v1[1]*v2[0]
    
#     # Produit scalaire pour l'angle
#     dot = v1[0]*v2[0] + v1[1]*v2[1]
#     norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
#     norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
#     angle = math.degrees(math.acos(dot/(norm1*norm2)))
    
#     if cross > 0:
#         direction = "gauche"
#     elif cross < 0:
#         direction = "droite"
#     else:
#         direction = "tout droit"
    
#     return angle, direction

# def generate_route_instructions(points, distance_reelle_km, distance_pixels):
#     """Génère un texte d'itinéraire"""
#     # Conversion px -> mètres
#     pixel_to_m = (distance_reelle_km * 1000) / distance_pixels
    
#     instructions = ["🚩 Départ"]
#     cumule_px = 0
    
#     for i in range(1, len(points)-1):
#         d = distance(points[i-1], points[i])
#         cumule_px += d
#         dist_m = cumule_px * pixel_to_m
        
#         angle, direction = angle_direction(points[i-1], points[i], points[i+1])
        
#         # Formater la distance
#         if dist_m >= 1000:
#             texte_dist = f"📍Après {dist_m/1000:.1f} km"
#         else:
#             texte_dist = f"📍Après {int(dist_m)} m"
        
#         instructions.append(f"{texte_dist}, {direction}")
    
#     instructions.append("🏁 Arrivée")
#     return instructions


# def generate_route_instructions(path, scale_m_per_cm, 
#                                 min_distance_between_turns=80, 
#                                 angle_threshold=60, 
#                                 max_turns=20):
#     """
#     Générer les instructions d'itinéraire avec des distances le long du chemin,
#     pour que la somme des distances corresponde à la distance totale.
#     """
#     if len(path) < 3:
#         return ["Itinéraire trop court pour générer des instructions."]

#     # Détection des virages
#     key_indices = [0]
#     for i in range(1, len(path)-1):
#         angle = calculate_angle(path[i-1], path[i], path[i+1])
#         if abs(angle) >= angle_threshold:
#             key_indices.append(i)
#     key_indices.append(len(path)-1)

#     instructions = ["🚩 Départ"]

#     for i in range(1, len(key_indices)-1):
#         idx_prev = key_indices[i-1]
#         idx = key_indices[i]
#         idx_next = key_indices[i+1]

#         # Distance parcourue le long du chemin depuis le dernier changement
#         dist_m = distance_along_path(path, idx_prev, idx, scale_m_per_cm)

#         if dist_m < min_distance_between_turns:
#             continue

#         angle = calculate_angle(path[idx_prev], path[idx], path[idx_next])
#         direction = get_direction_instruction(angle)

#         if "tournez" in direction:
#             distance_str = f"{dist_m:.0f}m" if dist_m < 1000 else f"{dist_m/1000:.1f}km"
#             instructions.append(f"📍 Après {distance_str}, {direction}")

#     # Dernier tronçon (jusqu'à l'arrivée)
#     idx_last = key_indices[-2]
#     idx_end = key_indices[-1]
#     dist_m = distance_along_path(path, idx_last, idx_end, scale_m_per_cm)
#     distance_str = f"{dist_m:.0f}m" if dist_m < 1000 else f"{dist_m/1000:.1f}km"
#     instructions.append(f"📍 Continuez sur {distance_str}")

#     instructions.append("🏁 Arrivée")
#     return instructions

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Calculateur de chemin",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Calculateur de chemin le plus court")
st.markdown("---")

# Initialisation des variables de session
if 'points' not in st.session_state:
    st.session_state.points = []
if 'barrier_points' not in st.session_state:
    st.session_state.barrier_points = []
if 'path_calculated' not in st.session_state:
    st.session_state.path_calculated = False
if 'route_instructions' not in st.session_state:
    st.session_state.route_instructions = []

# Interface pour charger les images
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Image originale")
    original_file = st.file_uploader("Chargez l'image originale", type=['jpg', 'jpeg', 'png'], key="original")

with col2:
    st.subheader("🎭 Masque des routes")
    mask_file = st.file_uploader("Chargez le masque des routes", type=['jpg', 'jpeg', 'png'], key="mask")

if original_file and mask_file:
    # Charger les images
    img = load_image_from_bytes(original_file.getvalue())
    mask = load_mask_from_bytes(mask_file.getvalue())
    
    # Convertir l'image pour affichage
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.markdown("---")
    
    # Interface pour sélectionner les points
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🎯 Sélection des points")
        
        # Mode de sélection
        mode = st.radio(
            "Mode de sélection :",
            ["🗺️ Points d'itinéraire", "🚧 Points de barrière"],
            key="selection_mode"
        )
        
        # Instructions dynamiques
        if mode == "🗺️ Points d'itinéraire":
            points_needed = 2 - len(st.session_state.points)
            if points_needed > 0:
                st.info(f"Cliquez sur l'image pour sélectionner le point {len(st.session_state.points) + 1}/{2}")
            else:
                st.success("✅ Deux points d'itinéraire sélectionnés")
        else:
            barrier_needed = 2 - len(st.session_state.barrier_points)
            if barrier_needed > 0:
                st.info(f"Cliquez sur l'image pour sélectionner le point de barrière {len(st.session_state.barrier_points) + 1}/{2}")
            else:
                st.success("✅ Deux points de barrière sélectionnés")
        
        # Afficher les points sélectionnés
        if st.session_state.points:
            st.write("**Points d'itinéraire :**")
            for i, point in enumerate(st.session_state.points):
                st.write(f"🎯 Point {i+1}: ({int(point[0])}, {int(point[1])})")
        
        if st.session_state.barrier_points:
            st.write("**Points de barrière :**")
            for i, point in enumerate(st.session_state.barrier_points):
                st.write(f"🚧 Point {i+1}: ({int(point[0])}, {int(point[1])})")
        
        # Boutons de contrôle
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Reset itinéraire"):
                st.session_state.points = []
                st.session_state.path_calculated = False
                st.rerun()
        
        with col_btn2:
            if st.button("🔄 Reset barrière"):
                st.session_state.barrier_points = []
                st.session_state.path_calculated = False
                st.rerun()
        
        # Configuration de l'échelle
        st.subheader("📏 Configuration de l'échelle")
        
        col_scale1, col_scale2 = st.columns(2)
        with col_scale1:
            scale_cm = st.number_input("1 cm sur la carte =", min_value=1, max_value=10000, value=200, step=10, key="scale_cm")
        with col_scale2:
            scale_unit = st.selectbox("Unité", ["mètres", "kilomètres"], key="scale_unit")
        
        # Calculer l'échelle en m/cm
        if scale_unit == "kilomètres":
            scale_m_per_cm = scale_cm * 1000
        else:
            scale_m_per_cm = scale_cm
            
        st.info(f"Échelle : 1 cm = {scale_cm} {scale_unit}")
        
        st.markdown("---")
        
        # Calcul du chemin
        if len(st.session_state.points) == 2:
            if st.button("🧭 Calculer le chemin le plus court"):
                with st.spinner("Calcul en cours..."):
                    # Appliquer la barrière si des points sont sélectionnés
                    working_mask = mask.copy()
                    if len(st.session_state.barrier_points) == 2:
                        working_mask = barreRoute(working_mask, st.session_state.barrier_points)
                    
                    # Trouver les candidats
                    all_candidates = findNearestRoutePoints(working_mask, st.session_state.points, k=100)
                    
                    # Calculer le chemin
                    start, end, path = computeShortestPath(working_mask, all_candidates)
                    
                    if path:
                        st.session_state.path_result = {
                            'path': path,
                            'start': start,
                            'end': end,
                            'mask': working_mask
                        }
                        st.session_state.path_calculated = True
                        st.success(f"Chemin trouvé !")
                        
                        # Calcul de la distance
                        length_pixels = len(path)
                        length_cm = length_pixels * 0.02646
                        length_m = pixels_to_distance(length_pixels, scale_m_per_cm)
                        
                        # Affichage des métriques
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        with col_metric1:
                            st.metric("Distance (pixels)", f"{length_pixels}", "pixels")
                        
                        with col_metric2:
                            st.metric("Distance (carte)", f"{length_cm:.2f}", "cm")
                        
                        with col_metric3:
                            if length_m < 1000:
                                st.metric("Distance réelle", f"{length_m:.0f}", "m")
                            else:
                                st.metric("Distance réelle", f"{length_m/1000:.2f}", "km")
                        
                        # Temps estimé
                        time_hours = length_m / 1000 / 6  # 5 km/h à pied
                        if time_hours < 1:
                            time_str = f"{time_hours * 60:.0f} min"
                        else:
                            time_str = f"{time_hours:.1f}h"
                        
                        st.info(f"⏱️ **Temps estimé à pied (6 km/h) :** {time_str}")
                        
                        # Générer les instructions d'itinéraire
                        # st.session_state.route_instructions = generate_route_instructions(
                        #     path, scale_m_per_cm, length_pixels
                        # )
                    else:
                        st.error("Aucun chemin trouvé entre les deux points.")
        else:
            st.info("Sélectionnez 2 points sur l'image pour calculer le chemin.")
    
    with col1:
        # Affichage de l'image avec interaction
        if st.session_state.path_calculated and 'path_result' in st.session_state:
            # Créer deux colonnes pour l'image et les instructions
            col_img, col_instructions = st.columns([3, 2])
            
            with col_img:
                # Afficher l'image avec le chemin
                result_img = drawPath(img, 
                                    st.session_state.path_result['path'], 
                                    st.session_state.path_result['start'], 
                                    st.session_state.path_result['end'])
                
                st.image(result_img, caption="Chemin le plus court calculé", use_container_width=True)
                
                # Bouton de téléchargement
                result_pil = Image.fromarray(result_img)
                buf = io.BytesIO()
                result_pil.save(buf, format='PNG')
                
                st.download_button(
                    label="📥 Télécharger l'image avec le chemin",
                    data=buf.getvalue(),
                    file_name="chemin_superpose.png",
                    mime="image/png"
                )
            
            with col_instructions:
                st.subheader("🧭 Instructions d'itinéraire")
                
                if hasattr(st.session_state, 'route_instructions'):
                    # Afficher les instructions dans un conteneur stylé
                    instructions_container = st.container()
                    with instructions_container:
                        for i, instruction in enumerate(st.session_state.route_instructions):
                            if i == 0:  # Départ
                                st.success(instruction)
                            elif i == len(st.session_state.route_instructions) - 1:  # Arrivée
                                st.success(instruction)
                            else:  # Instructions intermédiaires
                                st.info(instruction)
                    
                    # Résumé de l'itinéraire
                    st.markdown("---")
                    st.subheader("📋 Résumé")
                    total_instructions = len(st.session_state.route_instructions) - 2  # Exclure départ et arrivée
                    st.write(f"**Nombre de changements de direction :** {total_instructions}")
                    
                    # Bouton pour télécharger les instructions
                    instructions_text = "\n".join([f"{i+1}. {inst}" for i, inst in enumerate(st.session_state.route_instructions)])
                    st.download_button(
                        label="📄 Télécharger les instructions",
                        data=instructions_text,
                        file_name="instructions_itineraire.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("Recalculez l'itinéraire pour voir les instructions détaillées.")
        else:
            # Afficher l'image originale avec les points sélectionnés
            display_img = img_display.copy()
            
            # Dessiner les points de barrière en rouge
            for i, point in enumerate(st.session_state.barrier_points):
                cv2.circle(display_img, (int(point[0]), int(point[1])), 7, (255, 0, 0), -1)
                cv2.circle(display_img, (int(point[0]), int(point[1])), 7, (255, 255, 255), 2)
                cv2.putText(display_img, f"B{i+1}", (int(point[0])+10, int(point[1])-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Dessiner les points d'itinéraire en vert
            for i, point in enumerate(st.session_state.points):
                cv2.circle(display_img, (int(point[0]), int(point[1])), 7, (0, 255, 0), -1)
                cv2.circle(display_img, (int(point[0]), int(point[1])), 7, (255, 255, 255), 2)
                cv2.putText(display_img, str(i+1), (int(point[0])+10, int(point[1])-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Utiliser streamlit-image-coordinates pour la sélection de points
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                
                # Créer une clé unique basée sur l'état actuel pour forcer la mise à jour
                image_key = f"image_coords_{len(st.session_state.points)}_{len(st.session_state.barrier_points)}_{mode}"
                
                coordinates = streamlit_image_coordinates(
                    display_img,
                    key=image_key
                )
                
                if coordinates is not None:
                    x, y = coordinates["x"], coordinates["y"]
                    
                    # Gestion selon le mode sélectionné
                    if mode == "🗺️ Points d'itinéraire":
                        if len(st.session_state.points) < 2:
                            # Vérifier que ce n'est pas un double clic (même point)
                            if not st.session_state.points or \
                               abs(st.session_state.points[-1][0] - x) > 5 or \
                               abs(st.session_state.points[-1][1] - y) > 5:
                                st.session_state.points.append((x, y))
                                st.success(f"Point {len(st.session_state.points)} sélectionné : ({x}, {y})")
                                # Reset du chemin calculé
                                st.session_state.path_calculated = False
                                st.rerun()
                        else:
                            st.warning("Deux points d'itinéraire déjà sélectionnés. Utilisez 'Reset itinéraire' pour recommencer.")
                    
                    elif mode == "🚧 Points de barrière":
                        if len(st.session_state.barrier_points) < 2:
                            # Vérifier que ce n'est pas un double clic (même point)
                            if not st.session_state.barrier_points or \
                               abs(st.session_state.barrier_points[-1][0] - x) > 5 or \
                               abs(st.session_state.barrier_points[-1][1] - y) > 5:
                                st.session_state.barrier_points.append((x, y))
                                st.success(f"Point de barrière {len(st.session_state.barrier_points)} sélectionné : ({x}, {y})")
                                # Reset du chemin calculé
                                st.session_state.path_calculated = False
                                st.rerun()
                        else:
                            st.warning("Deux points de barrière déjà sélectionnés. Utilisez 'Reset barrière' pour recommencer.")
                        
            except ImportError:
                st.error("Le package 'streamlit-image-coordinates' n'est pas installé. Installez-le avec : pip install streamlit-image-coordinates")
                st.image(display_img, caption="Image originale (interaction non disponible)", use_container_width=True)

else:
    st.info("👆 Veuillez charger l'image originale et le masque des routes pour commencer.")
    
    st.markdown("""
    ### Comment utiliser cette application :
    
    1. **Chargez les images** : L'image originale et le masque des routes
    2. **Sélectionnez les points** : Cliquez sur l'image pour sélectionner 2 points (départ et arrivée)
    3. **Barrez une route (optionnel)** : Sélectionnez 2 points pour créer un obstacle
    4. **Calculez le chemin** : L'algorithme trouve le chemin le plus court
    5. **Téléchargez le résultat** : L'image avec le chemin superposé
    
    ### Installation requise :
    ```bash
    pip install streamlit opencv-python scipy pillow streamlit-image-coordinates
    ```
    """)