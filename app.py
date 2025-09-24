import cv2
import numpy as np
import streamlit as st
from scipy.spatial import cKDTree
from scipy.ndimage import label
import heapq
from PIL import Image, ImageDraw

def selectPoints(img):
    st.image(img, caption="Cliquez sur deux points de départ et d'arrivée", use_container_width =True)
    points = st.session_state.get("points", [])
    if "reset_points" not in st.session_state:
        st.session_state.reset_points = False

    if st.button("Réinitialiser la sélection"):
        st.session_state.points = []
        st.session_state.reset_points = True
        points = []

    # click = st.image(img, caption="Cliquez sur l'image pour sélectionner les points", use_container_width =True)
    coords = st.experimental_get_query_params().get("coords", [])
    if coords:
        coords = [tuple(map(int, c.split(','))) for c in coords]
        st.session_state.points = coords
        points = coords

    st.write("Sélectionnez deux points sur l'image (x, y) :")
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("x1", min_value=0, max_value=img.shape[1]-1, value=0)
        y1 = st.number_input("y1", min_value=0, max_value=img.shape[0]-1, value=0)
    with col2:
        x2 = st.number_input("x2", min_value=0, max_value=img.shape[1]-1, value=img.shape[1]-1)
        y2 = st.number_input("y2", min_value=0, max_value=img.shape[0]-1, value=img.shape[0]-1)
    if st.button("Valider les points"):
        st.session_state.points = [(x1, y1), (x2, y2)]
        points = st.session_state.points
    if len(points) == 2:
        st.success(f"Points sélectionnés : {points}")
    return points

def barreRoute(mask, img, pts=None):
    st.image(img, caption="Tracez un trait pour barrer une route", use_container_width =True)
    st.write("Entrez les coordonnées de début et de fin du trait à tracer :")
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("x début", min_value=0, max_value=img.shape[1]-1, value=0, key="barre_x1")
        y1 = st.number_input("y début", min_value=0, max_value=img.shape[0]-1, value=0, key="barre_y1")
    with col2:
        x2 = st.number_input("x fin", min_value=0, max_value=img.shape[1]-1, value=img.shape[1]-1, key="barre_x2")
        y2 = st.number_input("y fin", min_value=0, max_value=img.shape[0]-1, value=img.shape[0]-1, key="barre_y2")
    if st.button("Tracer le trait"):
        mask_barr = mask.copy()
        cv2.line(mask_barr, (int(x1), int(y1)), (int(x2), int(y2)), 0, thickness=8)
        st.success(f'Route barrée entre ({x1},{y1}) et ({x2},{y2}) sur le masque.')
        st.image(mask_barr, caption="Masque après barrière", use_container_width =True, channels="GRAY")
        return mask_barr
    return mask

def findNearestRoutePoints(mask, points, k=100):
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
        st.error('Aucun chemin trouvé pour les k plus proches points.')
        return None, None, []
    st.success(f'Chemin trouvé de {start} à {end}, longueur : {len(path)}')
    return start, end, path

def plotPath(img, path, start, end):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for (x, y) in path:
        draw.point((x, y), fill=(255,0,0))
    draw.ellipse((start[0]-4, start[1]-4, start[0]+4, start[1]+4), fill=(0,255,0))
    flag_size = 12
    flag_base = [
        (end[0], end[1]),
        (end[0] - flag_size//2, end[1] + flag_size),
        (end[0] + flag_size//2, end[1] + flag_size)
    ]
    draw.polygon(flag_base, fill=(255,0,0))
    draw.line([(end[0], end[1]), (end[0], end[1] + flag_size)], fill=(0,0,0), width=2)
    st.image(img_pil, caption="Chemin le plus court superposé", use_container_width =True)
    return img_pil

def pixels_to_cm(length_pixels):
    return length_pixels * 0.02646

def main():
    st.title("Itinéraire interactif sur image (Streamlit)")
    uploaded_img = st.file_uploader("Chargez l'image originale (jpg/png)", type=["jpg", "png"])
    uploaded_mask = st.file_uploader("Chargez le masque des routes (png)", type=["png"])
    if uploaded_img and uploaded_mask:
        file_bytes_img = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes_img, 1)
        file_bytes_mask = np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8)
        mask = cv2.imdecode(file_bytes_mask, 0)

        st.header("1. Sélection des points")
        points = selectPoints(img)
        if len(points) == 2:
            st.header("2. Barrer une route (optionnel)")
            mask_mod = barreRoute(mask, img)
            st.header("3. Recherche des points de route les plus proches")
            all_candidates = findNearestRoutePoints(mask_mod, points, k=100)
            st.write("Points candidats trouvés.")
            st.header("4. Calcul du chemin le plus court")
            start, end, path = computeShortestPath(mask_mod, all_candidates)
            if path:
                st.header("5. Affichage du chemin")
                img_with_path = plotPath(img, path, start, end)
                length_pixels = len(path)
                length_cm = pixels_to_cm(length_pixels)
                st.success(f"Longueur du chemin : {length_pixels} pixels, soit {length_cm:.2f} cm")
    else:
        st.info("Veuillez charger une image et un masque pour commencer.")

if __name__ == "__main__":
    main()