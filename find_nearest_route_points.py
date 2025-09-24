
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label

# Charger le masque des routes
mask = cv2.imread('mask_routes.png', 0)

# Trouver les composantes connexes sur le masque
structure = np.ones((3, 3), dtype=np.int32)
labeled, ncomponents = label(mask == 255, structure=structure)

# Charger les points sélectionnés
with open('selected_points.txt', 'r') as f:
    points = [tuple(map(float, line.strip().split(','))) for line in f.readlines()]

# Trouver les coordonnées des pixels de route (valeur blanche)
route_pixels = np.column_stack(np.where(mask == 255))  # (y, x)
route_pixels_xy = np.fliplr(route_pixels)  # (x, y)

# Construire un k-d tree pour les routes
kdtree = cKDTree(route_pixels_xy)

# Pour chaque point sélectionné, on va sauvegarder les k plus proches points de route (pour test itératif)
k = 100  # nombre de voisins à tester
all_candidates = []

for pt in points:
    dists, idxs = kdtree.query(pt, k=k)
    # idxs peut être un scalaire si moins de k points
    if np.isscalar(idxs):
        idxs = [idxs]
    # Convertir en int natifs Python pour la sérialisation JSON
    candidates = [tuple(int(x) for x in route_pixels_xy[i]) for i in idxs]
    all_candidates.append(candidates)

# Sauvegarder tous les candidats pour chaque point
import json
with open('all_route_candidates.json', 'w') as f:
    json.dump(all_candidates, f)

# Pour compatibilité, on garde le plus proche dans nearest_route_points.txt
nearest_route_points = [cands[0] for cands in all_candidates]
with open('nearest_route_points.txt', 'w') as f:
    for pt in nearest_route_points:
        f.write(f"{pt[0]},{pt[1]}\n")
