import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import heapq

def selectPoints():
    # Charger l'image originale et le masque
    img = cv2.imread('original.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('mask_routes.png', 0)

    # Afficher l'image et permettre à l'utilisateur de sélectionner deux points
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title('Cliquez pour sélectionner deux points')
    points = plt.ginput(2, timeout=0)
    plt.close()

    print('Points sélectionnés :', points)

    # Sauvegarder les points sélectionnés
    with open('selected_points.txt', 'w') as f:
        for pt in points:
            f.write(f"{pt[0]},{pt[1]}\n")

def barreRoute():
    # Charger l'image originale et le masque
    img = cv2.imread('original.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('mask_routes.png', 0)

    # Afficher l'image et permettre à l'utilisateur de tracer un trait
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_title('Tracez un trait pour barrer une route (clic début et fin)')
    pts = plt.ginput(2, timeout=0)
    plt.close()

    if len(pts) == 2:
        x1, y1 = map(int, pts[0])
        x2, y2 = map(int, pts[1])
        # Tracer un trait noir sur le masque
        mask_barr = mask.copy()
        cv2.line(mask_barr, (x1, y1), (x2, y2), 0, thickness=8)  # épaisseur ajustable
        cv2.imwrite('mask_routes.png', mask_barr)
        print(f'Route barrée entre ({x1},{y1}) et ({x2},{y2}) sur le masque.')
    else:
        print('Trait non tracé, aucune modification.')

def findNearestRoutePoints():
    # Charger le masque des routes
    mask = cv2.imread('mask_routes.png', 0)

    # Charger les points sélectionnés
    with open('selected_points.txt', 'r') as f:
        points = [tuple(map(float, line.strip().split(','))) for line in f.readlines()]

    # Trouver les coordonnées des pixels de route (valeur blanche)
    route_pixels = np.column_stack(np.where(mask == 255))  # (y, x)
    route_pixels_xy = np.fliplr(route_pixels)  # (x, y)

    # Construire un k-d tree pour les routes
    kdtree = cKDTree(route_pixels_xy)

    # Trouver les pixels de route les plus proches des points sélectionnés
    nearest_route_points = []
    for pt in points:
        dist, idx = kdtree.query(pt)
        nearest_route_points.append(tuple(route_pixels_xy[idx]))

    print('Points de route les plus proches :', nearest_route_points)

    # Sauvegarder les points de route trouvés
    with open('nearest_route_points.txt', 'w') as f:
        for pt in nearest_route_points:
            f.write(f"{pt[0]},{pt[1]}\n")

def computeShortestPath():
    # Charger le masque et les points de départ/arrivée
    mask = cv2.imread('mask_routes.png', 0)
    with open('nearest_route_points.txt', 'r') as f:
        points = [tuple(map(int, line.strip().split(','))) for line in f.readlines()]
    start, end = points

    # Créer une grille de poids (0 = obstacle, 1 = route)
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
        # Reconstruire le chemin
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

    # Calculer le chemin le plus court
    path = dijkstra(start, end)
    if not path:
        print('Aucun chemin trouvé.')
    else:
        print(f'Chemin trouvé de {start} à {end}, longueur : {len(path)}')

        # Afficher le chemin sur l'image originale
        img = cv2.imread('original.jpg')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for (x, y) in path:
            cv2.circle(img_rgb, (x, y), 1, (255,0,0), -1)
        plt.figure(figsize=(10,8))
        plt.imshow(img_rgb)
        plt.title('Chemin le plus court superposé')
        plt.show()

        # Sauvegarder l'image avec le chemin
        img_out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite('chemin_superpose.png', img_out)
        print('Image avec chemin sauvegardée sous chemin_superpose.png')

if __name__ == "__main__":
    # Étape 1 : Sélectionner les points
    selectPoints()

    # Étape 2 : Barrer une route si nécessaire
    barreRoute()

    # Étape 3 : Trouver les points de route les plus proches
    findNearestRoutePoints()

    # Étape 4 : Calculer le chemin le plus court entre les deux points
    computeShortestPath()