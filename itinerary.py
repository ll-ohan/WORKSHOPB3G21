import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
import heapq
from math import hypot, sqrt

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

def computeShortestPath(mask, all_candidates, smooth=True, epsilon=3):
    """Calculer le chemin le plus court, avec lissage optionnel."""
    grid = (mask == 255).astype(np.uint8)
    height, width = grid.shape

    def neighbors(x, y, allow_diagonals=True):
        dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
        if not allow_diagonals:
            for dx, dy in dirs4:
                nx, ny = x+dx, y+dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny, nx]:
                    yield nx, ny, dx, dy
        else:
            dirs8 = dirs4 + [(-1,-1),(1,-1),(-1,1),(1,1)]
            for dx, dy in dirs8:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if grid[ny, nx] == 0:
                    continue
                # anti corner-cutting : si diagonal, les deux cases adjacentes doivent être libres
                if dx != 0 and dy != 0:
                    if grid[y, x+dx] == 0 or grid[y+dy, x] == 0:
                        continue
                yield nx, ny, dx, dy

    # --- Rasterisation 4-connexe d'un segment (variant Bresenham) ---
    def bresenham4(a, b):
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        yield (x, y)
        while (x, y) != (x1, y1):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
                yield (x, y)          # on émet l'étape horizontale/verticale
            if (x, y) == (x1, y1):
                break
            if e2 < dx:
                err += dx
                y += sy
                yield (x, y)          # on émet l'autre axe

    def densify_waypoints(waypoints, allow_diagonals=True):
        if len(waypoints) <= 1:
            return waypoints
        out = [waypoints[0]]
        for i in range(len(waypoints)-1):
            x, y = waypoints[i]
            x1, y1 = waypoints[i+1]
            while (x, y) != (x1, y1):
                dx = 0 if x1 == x else (1 if x1 > x else -1)
                dy = 0 if y1 == y else (1 if y1 > y else -1)

                # tente d'abord la diagonale (si autorisée et sans corner-cutting)
                moved = False
                if allow_diagonals and dx != 0 and dy != 0:
                    if grid[y, x+dx] and grid[y+dy, x]:  # anti coin
                        x += dx; y += dy; moved = True

                # sinon, avance sur l’axe dominant
                if not moved:
                    if abs(x1 - x) >= abs(y1 - y):
                        x += dx if dx != 0 else 0
                    else:
                        y += dy if dy != 0 else 0

                out.append((x, y))
        return out

    def has_line_of_sight(a, b, allow_diagonals=True):
        # On “densifie” virtuellement le segment et on vérifie chaque case
        x, y = a; x1, y1 = b
        while (x, y) != (x1, y1):
            dx = 0 if x1 == x else (1 if x1 > x else -1)
            dy = 0 if y1 == y else (1 if y1 > y else -1)
            moved = False
            if allow_diagonals and dx != 0 and dy != 0:
                if grid[y, x+dx] and grid[y+dy, x]:
                    x += dx; y += dy; moved = True
            if not moved:
                if abs(x1 - x) >= abs(y1 - y):
                    x += dx if dx != 0 else 0
                else:
                    y += dy if dy != 0 else 0
            if not (0 <= x < width and 0 <= y < height) or grid[y, x] == 0:
                return False
        return True


    # --- Simplification par "shortcut" (string pulling sur grille) ---
    def visibility_shortcut(path):
        if len(path) <= 2:
            return path
        res = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = i + 1
            last_ok = j
            # pousse j aussi loin que possible tant que la visibilité tient
            while j < len(path) and has_line_of_sight(path[i], path[j]):
                last_ok = j
                j += 1
            res.append(path[last_ok])
            i = last_ok
        return res

    # --- RDP "obstacle-aware" : tolérance géo + check visibilité ---
    def rdp_los(points, eps):
        if len(points) < 3:
            return points

        def perp_dist(p, a, b):
            ax, ay = a; bx, by = b; px, py = p
            vx, vy = (bx - ax), (by - ay)
            if vx == 0 and vy == 0:
                return hypot(px - ax, py - ay)
            t = ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)
            t = max(0.0, min(1.0, t))
            cx, cy = ax + t * vx, ay + t * vy
            return hypot(px - cx, py - cy)

        def recurse(i, j):
            if i + 1 >= j:
                return [points[i], points[j]]
            a = points[i]; b = points[j]
            # point le plus éloigné du segment [a,b]
            idx, dmax = -1, -1.0
            for k in range(i + 1, j):
                d = perp_dist(points[k], a, b)
                if d > dmax:
                    dmax, idx = d, k
            # si proche du segment ET visibilité ok -> garder [a,b]
            if dmax <= eps and has_line_of_sight(a, b):
                return [a, b]
            # sinon on découpe
            left = recurse(i, idx)
            right = recurse(idx, j)
            return left[:-1] + right

        return recurse(0, len(points) - 1)

    def dijkstra(start, end):
        dist = np.full((height, width), np.inf)
        prev = np.full((height, width, 2), -1, dtype=int)
        dist[start[1], start[0]] = 0
        heap = [(0, start)]
        while heap:
            d, (x, y) = heapq.heappop(heap)
            if (x, y) == end:
                break
            if d != dist[y, x]:
                continue
            for nx, ny, dx, dy in neighbors(x, y, allow_diagonals=True):
                step_cost = sqrt(2.0) if (dx != 0 and dy != 0) else 1.0
                alt = d + step_cost
                if alt < dist[ny, nx]:
                    dist[ny, nx] = alt
                    prev[ny, nx] = [x, y]
                    heapq.heappush(heap, (alt, (nx, ny)))
        path = []
        x, y = end
        while (x, y) != start:
            path.append((x, y))
            px, py = prev[y, x]
            if px == -1:
                return []
            x, y = px, py
        path.append(start)
        path.reverse()
        return path

    found = False
    path = []
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

    # --- LISSAGE avant retour ---
    if smooth:
        # 1) on “tire la ficelle” (shortcut)
        waypoints = visibility_shortcut(path)
        # 2) on simplifie encore si possible, en respectant obstacles
        waypoints = rdp_los(waypoints, eps=epsilon)
        # 3) on re-densifie en chemin 4-connexe (cases adjacentes)
        path = densify_waypoints(waypoints)

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

import math
from typing import List, Tuple

Point = Tuple[int, int]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])

def _dot(u, v) -> float:
    return u[0]*v[0] + u[1]*v[1]

def _norm(u) -> float:
    return math.hypot(u[0], u[1])

def _angle_deg(a: Point, b: Point, c: Point) -> float:
    # angle en B entre BA et BC
    u = (a[0]-b[0], a[1]-b[1])
    v = (c[0]-b[0], c[1]-b[1])
    nu, nv = _norm(u), _norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    cosv = max(-1.0, min(1.0, _dot(u, v)/(nu*nv)))
    return math.degrees(math.acos(cosv))

def _point_seg_dist(p: Point, a: Point, c: Point) -> float:
    # distance P -> segment [A,C]
    ac = (c[0]-a[0], c[1]-a[1])
    ap = (p[0]-a[0], p[1]-a[1])
    ac2 = ac[0]*ac[0] + ac[1]*ac[1]
    if ac2 == 0:
        return _dist(p, a)
    t = _dot(ap, ac) / ac2
    t = max(0.0, min(1.0, t))
    proj = (a[0] + t*ac[0], a[1] + t*ac[1])
    return _dist(p, proj)

def smooth_path_preserve_corners(
    path: List[Point],
    scale_m_per_px: float,
    rel_threshold: float = 0.15,      # 15% de l_max
    abs_threshold_m: float = 20.0,    # 20 m
    straight_deg: float = 12.0,       # rectitude locale tolérée
    curv_eps_m: float = 3.0,          # écart mini à la corde pour préserver l’arc
    max_passes: int = 20              # sécurité pour éviter des boucles longues
) -> List[Point]:
    """
    Supprime B si (AB < T OU BC < T) ET que B n’est ni un coin (angle > straight_deg),
    ni porteur de courbure (distance B->AC > curv_eps_m).
    """
    n = len(path)
    if n < 3:
        return path[:]

    pts = path[:]
    px_per_m = 1.0 / max(scale_m_per_px, 1e-12)
    curv_eps_px = curv_eps_m * px_per_m

    for _ in range(max_passes):
        if len(pts) < 3:
            break

        seg_lengths = [_dist(pts[i], pts[i+1]) for i in range(len(pts)-1)]
        if not seg_lengths:
            break
        l_max = max(seg_lengths)
        threshold_px = max(rel_threshold * l_max, abs_threshold_m * px_per_m)

        changed = False
        i = 1  # on préserve les extrémités
        while i < len(pts) - 1:
            A, B, C = pts[i-1], pts[i], pts[i+1]
            AB = _dist(A, B)
            BC = _dist(B, C)

            # Garde-fous géométriques
            ang = _angle_deg(A, B, C)           # angle au point B
            d_perp = _point_seg_dist(B, A, C)   # courbure locale via écart à la corde

            # règle de suppression : petite(s) arête(s) + quasi-rectiligne + faible courbure
            if (AB < threshold_px or BC < threshold_px) and (ang <= straight_deg) and (d_perp <= curv_eps_px):
                del pts[i]
                changed = True
                # ne pas incrémenter i pour réévaluer au même index
            else:
                i += 1

        if not changed:
            break

    return pts

# ---- Génération d'itinéraire inchangée, avec un hook de lissage par défaut ----
def textual_itinerary(path: List[Point],
                      scale_m_per_px: float,
                      angle_threshold_deg: float = 30.0,
                      smooth: bool = True,
                      **smooth_kwargs) -> List[str]:
    def dist(a, b) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def dot(u, v) -> float:
        return u[0]*v[0] + u[1]*v[1]

    def norm(u) -> float:
        return math.hypot(u[0], u[1])

    def cross_z(u, v) -> float:
        return u[0]*v[1] - u[1]*v[0]

    def angle_deg(u, v) -> float:
        nu, nv = norm(u), norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        c = max(-1.0, min(1.0, dot(u, v) / (nu * nv)))
        return math.degrees(math.acos(c))

    def fmt_m(m: float) -> str:
        return f"{m:.1f}"

    if not path or len(path) < 2:
        return []

    if smooth:
        path = smooth_path_preserve_corners(path, scale_m_per_px, **smooth_kwargs)
        if len(path) < 2:
            return []

    itinerary: List[str] = []
    seg_len_px: float = 0.0

    for i in range(1, len(path) - 1):
        p_prev, p_cur, p_next = path[i-1], path[i], path[i+1]
        seg_len_px += dist(p_prev, p_cur)

        v1 = (p_cur[0] - p_prev[0], p_cur[1] - p_prev[1])
        v2 = (p_next[0] - p_cur[0], p_next[1] - p_cur[1])

        ang = angle_deg(v1, v2)
        if ang >= angle_threshold_deg:
            seg_len_m = seg_len_px * scale_m_per_px
            itinerary.append(f"avancer sur {fmt_m(seg_len_m)} m")

            cz = cross_z(v1, v2)  # repère image (y vers le bas)
            if cz > 0:
                itinerary.append("tourner à droite")
            elif cz < 0:
                itinerary.append("tourner à gauche")
            seg_len_px = 0.0

    seg_len_px += dist(path[-2], path[-1])
    seg_len_m = seg_len_px * scale_m_per_px
    itinerary.append(f"avancer sur {fmt_m(seg_len_m)} m")
    return itinerary
