import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
import heapq
from math import hypot, sqrt
from typing import Optional, Callable

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

def findNearestRoutePoints(mask, points, k=50):
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

import math
from PIL import Image
import heapq
import numpy as np
from math import hypot, sqrt
from typing import List, Tuple, Optional

def computeShortestPath(
    mask,
    all_candidates,
    smooth: bool = True,
    epsilon: float = 3.0,
    *,
    # --- nouveautés pour le post-traitement "itinéraire" ---
    scale_m_per_px: float = 1.0,
    nav_angle_threshold_deg: float = 30.0,     # seuil d'angle en-dessous duquel on “redresse”
    nav_min_segment_floor_m: float = 20.0,     # longueur mini absolue (m)
    nav_min_segment_ratio: float = 0.15,       # ou 15% du plus long tronçon
):
    """
    Calculer le chemin le plus court sur un masque binaire (255 = praticable).
    Optionnel: lissage géométrique (string-pulling + RDP) et post-traitement
    spécial pour la génération d'instructions (nav_path).

    Returns
    -------
    (start, end, path)                          si return_nav_path=False
    (start, end, path, nav_path)                si return_nav_path=True
    where:
      - path: chemin 4-connexe (dense) utilisable pour tracé/affichage
      - nav_path: chemin “épuré pour instructions” (moins de points parasites)
    """
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
                # anti corner-cutting : si diagonal, les deux cellules adjacentes doivent être libres
                if dx != 0 and dy != 0:
                    if grid[y, x+dx] == 0 or grid[y+dy, x] == 0:
                        continue
                yield nx, ny, dx, dy

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
                moved = False
                if allow_diagonals and dx != 0 and dy != 0:
                    if grid[y, x+dx] and grid[y+dy, x]:  # anti coin
                        x += dx; y += dy; moved = True
                if not moved:
                    if abs(x1 - x) >= abs(y1 - y):
                        x += dx if dx != 0 else 0
                    else:
                        y += dy if dy != 0 else 0
                out.append((x, y))
        return out

    def has_line_of_sight(a, b, allow_diagonals=True):
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

    def visibility_shortcut(path):
        if len(path) <= 2:
            return path
        res = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = i + 1
            last_ok = j
            while j < len(path) and has_line_of_sight(path[i], path[j]):
                last_ok = j
                j += 1
            res.append(path[last_ok])
            i = last_ok
        return res

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
            idx, dmax = -1, -1.0
            for k in range(i + 1, j):
                d = perp_dist(points[k], a, b)
                if d > dmax:
                    dmax, idx = d, k
            if dmax <= eps and has_line_of_sight(a, b):
                return [a, b]
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

    # --- recherche d’un couple (start, end) valide ---
    found = False
    path: List[Tuple[int, int]] = []
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

    # --- lissage “géométrique” (optionnel) ---
    if smooth:
        waypoints = visibility_shortcut(path)
        waypoints = rdp_los(waypoints, eps=epsilon)
        path = densify_waypoints(waypoints)

    # --- post-traitement spécial “itinéraire textuel” ---
    def _angle_deg(u, v) -> float:
        nu = hypot(u[0], u[1]); nv = hypot(v[0], v[1])
        if nu == 0 or nv == 0:
            return 0.0
        c = max(-1.0, min(1.0, (u[0]*v[0] + u[1]*v[1]) / (nu * nv)))
        return math.degrees(math.acos(c))

    def _smooth_nav_path_for_instructions(poly: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(poly) <= 2:
            return poly[:]

        # seuil de fusion “court” en mètres
        seg_lengths_px = [hypot(poly[i+1][0]-poly[i][0], poly[i+1][1]-poly[i][1]) for i in range(len(poly)-1)]
        if not seg_lengths_px:
            return poly[:]
        Lmax_m = max(seg_lengths_px) * scale_m_per_px
        Lmin_m = max(nav_min_segment_floor_m, nav_min_segment_ratio * Lmax_m)

        pts: List[Tuple[int, int]] = [poly[0]]
        i = 1
        while i < len(poly) - 1:
            a = pts[-1]; b = poly[i]; c = poly[i+1]
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            ang = _angle_deg(v1, v2)
            short_middle = min(hypot(b[0]-a[0], b[1]-a[1]),
                               hypot(c[0]-b[0], c[1]-b[1])) * scale_m_per_px < Lmin_m

            # Si le “coude” est faible OU que le tronçon central est très court,
            # et que la visibilité est OK, on saute le point b.
            if has_line_of_sight(a, c) and (ang < nav_angle_threshold_deg or short_middle):
                i += 1
                continue

            pts.append(b)
            i += 1

        pts.append(poly[-1])

        # Passe 2 : enlève les quasi-colinéarités résiduelles sous 1°
        j = 1
        while j < len(pts) - 1:
            a, b, c = pts[j-1], pts[j], pts[j+1]
            v1 = (b[0]-a[0], b[1]-a[1]); v2 = (c[0]-b[0], c[1]-b[1])
            if _angle_deg(v1, v2) < 1.0 and has_line_of_sight(a, c):
                del pts[j]
                continue
            j += 1

        return pts

    nav_path = _smooth_nav_path_for_instructions(path)

    return start, end, path, nav_path

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

def get_image_dpi(path: str) -> float | None:
    """
    Retourne le DPI horizontal de l'image si disponible.
    """
    with Image.open(path) as img:
        dpi = img.info.get("dpi")  # (dpi_x, dpi_y) si dispo
        if dpi:
            return dpi[0]  # on prend la valeur horizontale
    raise Exception("DPI non disponible dans les métadonnées de l'image")

def pixels_to_cm(length_pixels, dpi):
    return (length_pixels / dpi) * 2.54

import math
from typing import List, Tuple

Point = Tuple[int, int]

# ---- Génération d'itinéraire inchangée, avec un hook de lissage par défaut ----

def textual_itinerary(
    nav_path: List[Tuple[int, int]],
    *,
    scale_m_per_px: float,          # ⚠️ obligatoire: échelle carte (m par pixel)
    angle_eps_deg: float = 10.0,     # en-dessous: "tout droit"
    min_seg_m: float = 0.05,        # ignorer segments < 5 cm
    use_cm_below_1m: bool = True,   # formater < 1 m en cm
) -> str:
    """
    Génère des instructions textuelles à partir d'un nav_path.
    Règles d'angle (u=i0->i1, v=i1->i2):
      - 0°–45°   : "prendre le virage très serré à ..."
      - 45°–135° : "tourner à ..."
      - 135°–180°: "tourner légèrement à ..."
    """
    def normalize_path(nav_path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Renvoie le chemin converti dans un repère cartésien (y vers le haut).
        Détection automatique selon la majorité des mouvements.
        """
        ups, downs = 0, 0
        for (x0, y0), (x1, y1) in zip(nav_path, nav_path[1:]):
            dy = y1 - y0
            if dy < 0:
                ups += 1
            elif dy > 0:
                downs += 1
        y_is_down = ups >= downs  # si "haut" correspond à dy<0 → repère image

        if y_is_down:
            # convertir tous les points en inversant Y
            return [(x, -y) for (x, y) in nav_path]
        else:
            return nav_path

    
    if not nav_path or len(nav_path) < 2:
        return ""
    
    nav_path = normalize_path(nav_path)
    # 1) dédoublonne les points consécutifs identiques
    clean = [nav_path[0]]
    for p in nav_path[1:]:
        if p != clean[-1]:
            clean.append(p)
    nav_path = clean
    if len(nav_path) < 2:
        return ""

    def dist_m(a, b) -> float:
        return math.hypot(b[0]-a[0], b[1]-a[1]) * scale_m_per_px

    def fmt_distance(m: float) -> str:
        if use_cm_below_1m and m < 1.0:
            cm = int(round(m * 100))
            return f"{cm} cm"
        m_int = int(round(m))
        if m_int == 0:
            m_int = 1  # éviter "0 m", annoncer au moins 1 m
        return f"{m_int} {'mètre' if m_int == 1 else 'mètres'}"

    def angle_deg(u, v) -> Optional[float]:
        ux, uy = u; vx, vy = v
        nu = math.hypot(ux, uy); nv = math.hypot(vx, vy)
        if nu == 0 or nv == 0:
            return None
        c = max(-1.0, min(1.0, (ux*vx + uy*vy) / (nu*nv)))
        return math.degrees(math.acos(c))
    
    def turn_side(u, v) -> Optional[str]:
        ux, uy = u; vx, vy = v
        cross = ux*vy - uy*vx
        if cross > 0: return "gauche"
        if cross < 0: return "droite"
        return None
    
    def classify_turn(ang: float) -> str:
        if 0.0 <= ang < 45.0:
            return "Tourner légèrement à"
        elif 45.0 <= ang <= 135.0:
            return "Tourner à"
        else:  # 135–180
            return "Prendre le virage très serré à"

    instructions = []

    # Étapes: "Avancer de d puis [tourner/continuer]"
    for i in range(len(nav_path) - 2):
        i0, i1, i2 = nav_path[i], nav_path[i+1], nav_path[i+2]
        d = dist_m(i0, i1)
        if d < min_seg_m:
            # segment trop court -> on l'ignore (pas d'instruction "0 m")
            continue

        v1 = (i1[0]-i0[0], i1[1]-i0[1])
        v2 = (i2[0]-i1[0], i2[1]-i1[1])
        ang = angle_deg(v1, v2)
        side = turn_side(v1, v2) if ang is not None else None

        if ang is None or ang < angle_eps_deg or side is None:
            instructions.append(f"Avancer de {fmt_distance(d)} puis continuer tout droit")
        else:
            action = classify_turn(ang)
            instructions.append(f"Avancer de {fmt_distance(d)} puis {action} {side}")

    # Dernier tronçon
    last_d = dist_m(nav_path[-2], nav_path[-1])
    if last_d >= min_seg_m:
        instructions.append(f"Avancer de {fmt_distance(last_d)} avant destination")

    return ". ".join(instructions) + "."