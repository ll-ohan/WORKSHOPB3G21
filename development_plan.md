# Plan de Développement pour le Projet de Traitement d'Image de Carte

## Objectif
Créer un programme qui traite une image de carte pour supprimer les éléments textuels, générer un masque contenant uniquement les routes, et calculer le chemin le plus court entre deux points sélectionnés par l'utilisateur.

---

### Étapes de Développement

#### 1. Préparation du Projet
- [X] Configurer l'environnement de développement (Python, bibliothèques nécessaires)
- [X] Installer les bibliothèques nécessaires (OpenCV, NumPy, matplotlib, etc.)

#### 2. Traitement de l'Image
- [X] Générer un masque/squelette de l'image contenant uniquement les routes
  - [X] Extraire les routes à l'aide de la détection de contours
  - [X] Créer un masque représentant uniquement les routes (blanc pour les routes, noir pour le reste)

#### 3. Sélection des Points par l'Utilisateur
- [X] Afficher l'image initiale avec un outil interactif pour sélectionner deux points sur l'image
- [X] Convertir les coordonnées de ces points sur l'image en coordonnées sur le masque/squelette

#### 4. Identification des Points les Plus Proches sur les Routes
- [X] Utiliser les coordonnées des points sélectionnés par l'utilisateur
- [X] Identifier les points de la route les plus proches des points sélectionnés sur le masque
  - [X] Utiliser des algorithmes de recherche de plus proches voisins (par exemple, k-d tree)

#### 5. Calcul du Chemin le Plus Court
- [X] Calculer le chemin le plus court entre les deux points sur le masque
  - [X] Utiliser l'algorithme de Dijkstra ou A* pour trouver le chemin
  - [X] Appliquer cet algorithme sur la topologie des routes du masque

#### 6. Affichage du Chemin sur l'Image Originale
- [ ] Superposer le chemin calculé sur l'image originale
- [ ] Mettre en évidence le chemin (couleur différente, épaisseur de ligne, etc.)

#### 7. Finalisation
- [ ] Vérifier la précision du chemin calculé et son superposition sur la carte
- [ ] Tester avec différentes images et ajuster les paramètres si nécessaire
- [ ] Créer une interface utilisateur conviviale pour la sélection des points et l'affichage des résultats

### Conclusion
Une fois terminé, ce projet permettra de charger une image de carte, d'en extraire uniquement les routes, de permettre à l'utilisateur de sélectionner deux points, et de calculer et afficher le chemin le plus court entre ces points sur l'image originale.
