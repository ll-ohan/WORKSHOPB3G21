import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
import heapq

def barreRoute(mask, pts):
    """Barrer une route sur le masque"""
    if len(pts) == 2:
        x1, y1 = map(int, pts[0])
        x2, y2 = map(int, pts[1])
        mask_barr = mask.copy()
        cv2.line(mask_barr, (x1, y1), (x2, y2), 0, thickness=8)
        return mask_barr
    else:
        return mask

def findNearestRoutePoints(mask, points, k=200):
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

def pixels_to_cm(length_pixels):
    return length_pixels * 0.02646
