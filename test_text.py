import math

def distance(p1, p2):
    """Distance euclidienne entre deux points"""
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def angle_direction(p1, p2, p3):
    """Détermine l'angle et la direction (gauche/droite)"""
    v1 = (p2[0]-p1[0], p2[1]-p1[1])
    v2 = (p3[0]-p2[0], p3[1]-p2[1])
    
    # Produit vectoriel pour savoir gauche/droite
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    
    # Produit scalaire pour l'angle
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
    norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
    angle = math.degrees(math.acos(dot/(norm1*norm2)))
    
    if cross > 0:
        direction = "gauche"
    elif cross < 0:
        direction = "droite"
    else:
        direction = "tout droit"
    
    return angle, direction

def generer_itineraire(points, distance_reelle_km, distance_pixels):
    """Génère un texte d'itinéraire"""
    # Conversion px -> mètres
    pixel_to_m = (distance_reelle_km * 1000) / distance_pixels
    
    print("Départ")
    cumule_px = 0
    
    for i in range(1, len(points)-1):
        d = distance(points[i-1], points[i])
        cumule_px += d
        dist_m = cumule_px * pixel_to_m
        
        angle, direction = angle_direction(points[i-1], points[i], points[i+1])
        
        # Formater la distance
        if dist_m >= 1000:
            texte_dist = f"Après {dist_m/1000:.1f} km"
        else:
            texte_dist = f"Après {int(dist_m)} m"
        
        print(f"{texte_dist}, tournez à {direction}")
    
    print("Arrivée")

# Exemple d'utilisation
points = [(0,0), (0,40), (20,40), (20,270), (60,270), (60,460)]  # coordonnées fictives
generer_itineraire(points, distance_reelle_km=3.33, distance_pixels=630)
