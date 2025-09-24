
import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt

def neighbors(x, y, width, height, grid):
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < width and 0 <= ny < height:
            if grid[ny, nx]:
                yield nx, ny

def dijkstra(start, end, grid):
    height, width = grid.shape
    dist = np.full((height, width), np.inf)
    prev = np.full((height, width, 2), -1, dtype=int)
    dist[start[1], start[0]] = 0
    heap = [(0, start)]
    while heap:
        d, (x, y) = heapq.heappop(heap)
        if (x, y) == end:
            break
        for nx, ny in neighbors(x, y, width, height, grid):
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

def main():
    # Charger le masque
    mask = cv2.imread('mask_routes.png', 0)
    # Charger tous les candidats pour chaque point
    import json
    with open('all_route_candidates.json', 'r') as f:
        all_candidates = json.load(f)
    if len(all_candidates) != 2:
        print('Erreur : il faut deux listes de candidats.')
        return
    # Créer une grille de poids (0 = obstacle, 1 = route)
    grid = (mask == 255).astype(np.uint8)
    height, width = grid.shape

    # Essayer tous les couples de candidats (ordre croissant de distance)
    found = False
    for start in all_candidates[0]:
        for end in all_candidates[1]:
            start = tuple(map(int, start))
            end = tuple(map(int, end))
            path = dijkstra(start, end, grid)
            if path:
                found = True
                break
        if found:
            break

    if not found or not path:
        print('Aucun chemin trouvé pour les k plus proches points.')
        return
    print(f'Chemin trouvé de {start} à {end}, longueur : {len(path)}')

    # Afficher le chemin sur l'image originale
    img = cv2.imread('original.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for (x, y) in path:
        cv2.circle(img_rgb, (x, y), 1, (255,0,0), -1)

    # Afficher le point de départ en vert
    cv2.circle(img_rgb, start, 4, (0,255,0), -1)

    # Afficher un drapeau pour l'arrivée (triangle rouge)
    flag_size = 12
    flag_base = np.array([
        [end[0], end[1]],
        [end[0] - flag_size//2, end[1] + flag_size],
        [end[0] + flag_size//2, end[1] + flag_size]
    ], np.int32)
    cv2.fillPoly(img_rgb, [flag_base], (255,0,0))
    cv2.line(img_rgb, (end[0], end[1]), (end[0], end[1] + flag_size), (0,0,0), 2)

    plt.figure(figsize=(10,8))
    plt.imshow(img_rgb)
    plt.title('Chemin le plus court superposé')
    plt.show()

    # Sauvegarder l'image avec le chemin
    img_out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('chemin_superpose.png', img_out)
    print('Image avec chemin sauvegardée sous chemin_superpose.png')

if __name__ == "__main__":
    main()
