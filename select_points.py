import cv2
import matplotlib.pyplot as plt

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
