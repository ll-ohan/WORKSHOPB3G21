import cv2
import numpy as np
from scipy.spatial import cKDTree
import heapq
import math
from math import hypot, sqrt
from PIL import Image
from typing import List, Tuple, Optional


ROUTE_COLOR_BGR = (225, 220, 142)  # hex #8EDCE1
START_MARKER_COLOR_BGR = ROUTE_COLOR_BGR
END_MARKER_COLOR_BGR = (69, 23, 255)  # hex #ff1745
MARKER_BORDER_COLOR_BGR = (255, 255, 255)
MARKER_RADIUS_PX = 10
MARKER_BORDER_WIDTH_PX = 3
ROUTE_THICKNESS_PX = 5

def barreRoute(mask, pts):
    """Ajoute une barrière rouge semi-transparente sur le mask RGBA"""
    if len(pts) == 2:
        x1, y1 = map(int, pts[0])
        x2, y2 = map(int, pts[1])
        mask_barr = mask.copy()

        # rouge Batmap opaque (B=69, G=23, R=255, A=255)
        cv2.line(mask_barr, (x1, y1), (x2, y2), (69, 23, 255, 255), thickness=8)
        return mask_barr
    return mask

def findNearestRoutePoints(mask, points, k=50):
    """Trouver les points de route les plus proches (= pixels blancs)"""
    route_pixels = np.column_stack(np.where(mask == 255)) #extrait coordonées des pixels blancs
    route_pixels_xy = np.fliplr(route_pixels)  #inverser colonnes pour (x,y)
    kdtree = cKDTree(route_pixels_xy) # création d'un arbre pour recherche rapide
    all_candidates = []
    for pt in points:
        _, idxs = kdtree.query(pt, k=min(k, len(route_pixels_xy))) #k plus proches voisins
        if np.isscalar(idxs): #verifier si idxs est scalaire car k peut être > nombre de pixels
            idxs = [idxs]
        candidates = [tuple(int(x) for x in route_pixels_xy[i]) for i in idxs] #extraction des coordonnées
        all_candidates.append(candidates)
    return all_candidates

def computeShortestPath(
    mask,
    all_candidates,
    scale_m_per_px: float,
    *,
    smooth: bool = True,
    epsilon: float = 3.0,
    nav_angle_threshold_deg: float = 30.0,     # seuil d'angle en-dessous duquel on “redresse”
    nav_min_segment_floor_m: float = 20.0,     # longueur mini absolue (m)
    nav_min_segment_ratio: float = 0.15,       # ou 15% du plus long tronçon
):
    """
    Calculer le chemin le plus court sur un masque binaire (255 = praticable).
    """
    grid = (mask == 255).astype(np.uint8) # 1 = praticable, 0 = obstacle
    height, width = grid.shape # dimensions de la grille

    def neighbors(x, y, allow_diagonals=True):
        dirs4 = [(-1,0),(1,0),(0,-1),(0,1)] 
        if not allow_diagonals:
            for dx, dy in dirs4: # 4 directions
                nx, ny = x+dx, y+dy #coordonnées du voisin
                if 0 <= nx < width and 0 <= ny < height and grid[ny, nx]: #si hors de la grille
                    yield nx, ny, dx, dy
        else:
            dirs8 = dirs4 + [(-1,-1),(1,-1),(-1,1),(1,1)] # 8 directions
            for dx, dy in dirs8:
                nx, ny = x+dx, y+dy #coordonnées du voisin
                if not (0 <= nx < width and 0 <= ny < height): # si hors de la grille classique
                    continue
                if grid[ny, nx] == 0:
                    continue
                #ne pas couper le chemin: si diagonal, les deux cellules adjacentes doivent être libres
                if dx != 0 and dy != 0:
                    if grid[y, x+dx] == 0 or grid[y+dy, x] == 0:
                        continue
                yield nx, ny, dx, dy

    def densify_waypoints(waypoints, allow_diagonals=True):
        if len(waypoints) <= 1:
            return waypoints #waypoint is a coordinate point
        out = [waypoints[0]] 
        for i in range(len(waypoints)-1): #pour chaque segment entre deux waypoints (-1 car on regarde i et i+1)
            x, y = waypoints[i]
            x1, y1 = waypoints[i+1]
            while (x, y) != (x1, y1): #tant qu'on n'a pas atteint le waypoint suivant
                dx = 0 if x1 == x else (1 if x1 > x else -1) #direction x
                dy = 0 if y1 == y else (1 if y1 > y else -1) #direction y
                moved = False #indicateur de mouvement
                if allow_diagonals and dx != 0 and dy != 0:
                    if grid[y, x+dx] and grid[y+dy, x]: 
                        x += dx; y += dy; moved = True #déplacement diagonal
                if not moved: #si pas de déplacement diagonal
                    if abs(x1 - x) >= abs(y1 - y): #priorité à l'axe le plus long
                        x += dx if dx != 0 else 0 
                    else:
                        y += dy if dy != 0 else 0 
                out.append((x, y))
        return out

    def has_line_of_sight(a, b, allow_diagonals=True): #verifier si chemin praticable entre a et b
        x, y = a; x1, y1 = b
        while (x, y) != (x1, y1): #tant qu'on n'a pas atteint le point b
            dx = 0 if x1 == x else (1 if x1 > x else -1) 
            dy = 0 if y1 == y else (1 if y1 > y else -1)
            #déplacement
            moved = False
            if allow_diagonals and dx != 0 and dy != 0:
                if grid[y, x+dx] and grid[y+dy, x]:
                    x += dx; y += dy; moved = True
            if not moved:
                if abs(x1 - x) >= abs(y1 - y):
                    x += dx if dx != 0 else 0
                else:
                    y += dy if dy != 0 else 0
            if not (0 <= x < width and 0 <= y < height) or grid[y, x] == 0: #verifier si hors de la grille ou obstacle
                return False 
        return True

    def visibility_shortcut(path): #simplification du chemin en sautant les points intermédiaires si visibilité directe
        if len(path) <= 2:
            return path
        res = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = i + 1
            last_ok = j
            while j < len(path) and has_line_of_sight(path[i], path[j]): #tant qu'il y a une ligne de vue
                last_ok = j #mettre à jour le dernier point valide
                j += 1
            res.append(path[last_ok])
            i = last_ok #avancer i au dernier point valide
        return res

    def rdp_los(points, eps): #lissage Ramer-Douglas-Peucker avec ligne de vue
        if len(points) < 3:
            return points

        def perp_dist(p, a, b): #distance perpendiculaire du point p à la ligne ab
            ax, ay = a; bx, by = b; px, py = p #décomposition des tuples
            vx, vy = (bx - ax), (by - ay) #vecteur ab
            if vx == 0 and vy == 0: # a et b sont identiques
                return hypot(px - ax, py - ay) #distance euclidienne
            t = ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy) #projection scalaire
            t = max(0.0, min(1.0, t)) #clamp entre 0 et 1
            cx, cy = ax + t * vx, ay + t * vy #point projeté sur la ligne
            return hypot(px - cx, py - cy) #distance euclidienne

        def recurse(i, j): #recurse entre les indices i et j
            if i + 1 >= j: #si un ou aucun point entre i et j
                return [points[i], points[j]]
            a = points[i]; b = points[j] #points de début et fin
            idx, dmax = -1, -1.0 
            for k in range(i + 1, j): 
                d = perp_dist(points[k], a, b) #distance perpendiculaire
                if d > dmax: 
                    dmax, idx = d, k 
            if dmax <= eps and has_line_of_sight(a, b): #si distance max < eps et ligne de vue
                return [a, b]
            left = recurse(i, idx) #recurse gauche
            right = recurse(idx, j) #recurse droite
            return left[:-1] + right #fusionner en évitant le doublon

        return recurse(0, len(points) - 1)

    def dijkstra(start, end):
        dist = np.full((height, width), np.inf) #distance infinie initiale
        prev = np.full((height, width, 2), -1, dtype=int) #prédécesseur
        dist[start[1], start[0]] = 0 
        heap = [(0, start)] 
        while heap:
            d, (x, y) = heapq.heappop(heap) #extraire le noeud avec la plus petite distance
            if (x, y) == end: 
                break
            if d != dist[y, x]: #si la distance extraite n'est pas à jour
                continue
            for nx, ny, dx, dy in neighbors(x, y, allow_diagonals=True): 
                step_cost = sqrt(2.0) if (dx != 0 and dy != 0) else 1.0 #coût du pas
                alt = d + step_cost #distance alternative
                if alt < dist[ny, nx]: 
                    dist[ny, nx] = alt
                    prev[ny, nx] = [x, y]
                    heapq.heappush(heap, (alt, (nx, ny))) #ajouter au tas
        path = []
        x, y = end
        while (x, y) != start: #reconstruction du chemin à partir des prédécesseurs
            path.append((x, y))
            px, py = prev[y, x]
            if px == -1:
                return []
            x, y = px, py
        path.append(start)
        path.reverse()
        return path

    #lissage du chemin pour les instructions
    def _angle_deg(u, v) -> float:
        nu = hypot(u[0], u[1]); nv = hypot(v[0], v[1]) #normes
        if nu == 0 or nv == 0:
            return 0.0
        c = max(-1.0, min(1.0, (u[0]*v[0] + u[1]*v[1]) / (nu * nv))) #produit scalaire
        return math.degrees(math.acos(c)) #angle en degrés

    def _smooth_nav_path_for_instructions(poly: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(poly) <= 2:
            return poly[:]

        # seuil de fusion “court” en mètres
        seg_lengths_px = [hypot(poly[i+1][0]-poly[i][0], poly[i+1][1]-poly[i][1]) for i in range(len(poly)-1)] #longueurs des segments
        if not seg_lengths_px:
            return poly[:]
        Lmax_m = max(seg_lengths_px) * scale_m_per_px #longueur max en mètres
        Lmin_m = max(nav_min_segment_floor_m, nav_min_segment_ratio * Lmax_m) #longueur min en mètres

        pts: List[Tuple[int, int]] = [poly[0]]
        i = 1
        while i < len(poly) - 1:
            a = pts[-1]; b = poly[i]; c = poly[i+1]
            v1 = (b[0] - a[0], b[1] - a[1]) #vecteurs
            v2 = (c[0] - b[0], c[1] - b[1])
            ang = _angle_deg(v1, v2) 
            short_middle = min(hypot(b[0]-a[0], b[1]-a[1]), 
                               hypot(c[0]-b[0], c[1]-b[1])) * scale_m_per_px < Lmin_m #vérifier si segment court
            if has_line_of_sight(a, c) and (ang < nav_angle_threshold_deg or short_middle):
                i += 1
                continue

            pts.append(b)
            i += 1

        pts.append(poly[-1])

        #enlèver les vecteurs quasi-colinéraire
        j = 1
        while j < len(pts) - 1:
            a, b, c = pts[j-1], pts[j], pts[j+1]
            v1 = (b[0]-a[0], b[1]-a[1]); v2 = (c[0]-b[0], c[1]-b[1]) 
            if _angle_deg(v1, v2) < 1.0 and has_line_of_sight(a, c): #condition de colinéarité
                del pts[j]
                continue
            j += 1

        return pts


    #recherche d'un couple de point valide
    found = False
    path: List[Tuple[int, int]] = []
    for start in all_candidates[0]:
        for end in all_candidates[1]:
            start = tuple(map(int, start))
            end = tuple(map(int, end))
            path = dijkstra(start, end) #calcul du plus court chemin
            if path:
                found = True
                break
        if found:
            break

    if not found or not path:
        return None, None, []

    #lissage
    if smooth:
        waypoints = visibility_shortcut(path)
        waypoints = rdp_los(waypoints, eps=epsilon)
        path = densify_waypoints(waypoints)

    nav_path = _smooth_nav_path_for_instructions(path)

    return start, end, path, nav_path

def drawPath(img, path, start, end):
    """Dessiner le chemin sur l'image"""
    img_bgr = img.copy()

    if path:
        points = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
        if len(points) > 1:
            cv2.polylines(
                img_bgr,
                [points],
                isClosed=False,
                color=ROUTE_COLOR_BGR,
                thickness=ROUTE_THICKNESS_PX,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.circle(
                img_bgr,
                tuple(points[0][0]),
                MARKER_RADIUS_PX // 2,
                ROUTE_COLOR_BGR,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    def draw_marker(center, fill_color):
        cv2.circle(
            img_bgr,
            center,
            MARKER_RADIUS_PX,
            fill_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            img_bgr,
            center,
            MARKER_RADIUS_PX,
            MARKER_BORDER_COLOR_BGR,
            thickness=MARKER_BORDER_WIDTH_PX,
            lineType=cv2.LINE_AA,
        )

    draw_marker(start, START_MARKER_COLOR_BGR)
    draw_marker(end, END_MARKER_COLOR_BGR)

    return img_bgr

def get_image_dpi(path: str) -> float | None:
    """
    Retourne le DPI horizontal de l'image si disponible.
    """
    with Image.open(path) as img:
        dpi = img.info.get("dpi")  #si dispo
        if dpi:
            return dpi[0]
    raise Exception("DPI non disponible dans les métadonnées de l'image")

def pixels_to_cm(length_pixels, dpi):
    return (length_pixels / dpi) * 2.54 #car 1 inch = 2.54 cm



def textual_itinerary(
    nav_path: List[Tuple[int, int]],
    *,
    scale_m_per_px: float,
    angle_eps_deg: float = 10.0,     #seuille minimale pour le "tout droit"
    min_seg_m: float = 0.05,        #ignorer segments < 5 cm
) -> str:
    """
    Génère des instructions textuelles à partir d'un nav_path.
    """
    def normalize_path(nav_path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Renvoie le chemin converti dans un repère cartésien (y vers le haut).
        Détection automatique selon la majorité des mouvements.
        """
        ups, downs = 0, 0
        for (x0, y0), (x1, y1) in zip(nav_path, nav_path[1:]): #pour chaque segment
            dy = y1 - y0 
            if dy < 0: #si déplacement vers le haut
                ups += 1
            elif dy > 0: #si déplacement vers le bas
                downs += 1
        y_is_down = ups >= downs  #si y augmente vers le bas

        if y_is_down:
            # convertir tous les points en inversant Y
            return [(x, -y) for (x, y) in nav_path]
        else:
            return nav_path

    def dist_m(a, b) -> float:
        return math.hypot(b[0]-a[0], b[1]-a[1]) * scale_m_per_px

    def fmt_distance(m: float) -> str: #formater la distance en mètres
        m_int = int(round(m))
        if m_int == 0:
            m_int = 1
        return f"{m_int} {'mètre' if m_int == 1 else 'mètres'}"

    def angle_deg(u, v) -> Optional[float]: #angle entre deux vecteurs
        ux, uy = u; vx, vy = v
        nu = math.hypot(ux, uy); nv = math.hypot(vx, vy) #normes
        if nu == 0 or nv == 0:
            return None
        c = max(-1.0, min(1.0, (ux*vx + uy*vy) / (nu*nv))) #produit scalaire
        return math.degrees(math.acos(c)) #angle en degrés
    
    def turn_side(u, v) -> Optional[str]: #déterminer le côté du virage
        ux, uy = u; vx, vy = v
        cross = ux*vy - uy*vx #produit vectoriel 2D
        if cross > 0: return "gauche"
        if cross < 0: return "droite"
        return None
    
    def classify_turn(ang: float) -> str: #classifier le type de virage
        if 0.0 <= ang < 45.0:
            return "Tourner légèrement à"
        elif 45.0 <= ang <= 135.0:
            return "Tourner à"
        else:  # 135–180
            return "Prendre le virage très serré à"

    if not nav_path or len(nav_path) < 2:
        return ""
    
    nav_path = normalize_path(nav_path)

    #nettoyer les points consécutifs identiques
    clean = [nav_path[0]]
    for p in nav_path[1:]:
        if p != clean[-1]:
            clean.append(p)
    nav_path = clean
    if len(nav_path) < 2:
        return ""
    instructions = []

    for i in range(len(nav_path) - 2): #pour chaque segment sauf le dernier (-2 car on regarde i, i+1, i+2)
        i0, i1, i2 = nav_path[i], nav_path[i+1], nav_path[i+2] #points consécutifs
        d = dist_m(i0, i1)
        if d < min_seg_m:
            continue

        v1 = (i1[0]-i0[0], i1[1]-i0[1]) #vecteurs
        v2 = (i2[0]-i1[0], i2[1]-i1[1])
        ang = angle_deg(v1, v2)
        side = turn_side(v1, v2) if ang is not None else None

        if ang is None or ang < angle_eps_deg or side is None:
            # On cherche à "fusionner" les segments tout droit
            j = i + 2
            total_d = d
            v_ref = v1
            while j < len(nav_path) - 1:
                v_next = (nav_path[j+1][0] - i1[0], nav_path[j+1][1] - i1[1])
                ang_next = angle_deg(v_ref, v_next)
                side_next = turn_side(v_ref, v_next) if ang_next is not None else None
                d_next = dist_m(nav_path[j], nav_path[j+1])
                if ang_next is None or ang_next < angle_eps_deg or side_next is None:
                    total_d += d_next
                    j += 1
                else:
                    break
            action = classify_turn(ang)
            instructions.append(f"Avancer de {fmt_distance(total_d)} puis {action} {side_next}")
            i = j - 2
        else:
            action = classify_turn(ang)
            instructions.append(f"Avancer de {fmt_distance(d)} puis {action} {side}")

    last_d = dist_m(nav_path[-2], nav_path[-1]) #dernier segment
    if last_d >= min_seg_m:
        instructions.append(f"Avancer de {fmt_distance(last_d)} avant destination")

    return ". ".join(instructions) + "."
