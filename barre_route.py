import cv2
import matplotlib.pyplot as plt
import numpy as np

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
