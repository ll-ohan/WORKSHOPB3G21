# BatMap - Guide Utilisateur actuel

## Etape 1 : installer les dépendances
1. Ouvrez un terminal
2. Placez vous dans le dossier du projet avec :
```python
cd "C:\chemin\Workshop2025-26-B3g21-sources"
```
3. Pour chaque dossier (front et back) :
    - placez vous dans le dossier 
    ```python
    cd back
    ```
    OU
    ```python
    cd front
    ```
    - exécutez la commande suivante :
    ```python
    pip install -r requirements.txt
    ```

## Etape 2 : binariser l'image
1. Ouvrez un terminal
2. Placez vous dans le dossier back :
```python
cd back
```
3. Lancer l'api avec la commande suivante: 
```python 
python api.py
```
4. Ouvrez un autre terminal et placez vous dans le dossier front :
```python
cd front
```
5. Lancez la page HTML avec la commande suivante :
```
.\binary.html
```
6. Importez votre image, renseignez les éléments, envoyez votre réponse et choisissez le résultat de carte le plus adapté 
7. Validez votre choix, la carte binaire est enregistré dans la base de données
8. Pour arrêter l'API, retournez dans le terminal et cliquez "Ctrl+C"

## Etape 3 : créer l'itinéraire
1. Ouvrez un terminal
2. Placez vous dans le dossier front :
```python
cd front
```
3. Lancer la page avec la commande suivante: 
```python 
streamlit run itinerary.py
```
4. Définissez :
- Les **Points de départ et d'arrivée** : Cliquez sur la carte pour définir vos points
- L'**Échelle** : Renseignez l'échelle de votre carte
5. Cliquez sur "Calculer l'itinéraire"
6. Si vous souhaitez barrer une route (optionnel), sélectionnez "Points de barrière"
7. Définissez les points de début et fin de la section à bloquer
8. Recalculez l'itinéraire pour obtenir un nouveau tracé
9. L'itinéraire calculé est affiché sur l'image 
10. Les distances (picels, centimètre, kilomètres) et le temps estimé sont affichées sur l'interface
11. Pour arrêter l'application, retournez dans le terminal et cliquez "Ctrl+C"

## Etape 4 : créer les instructions textuelles de l'itinéraire
1. Ouvrez un terminal
2. Placez vous dans le dossier back :
```python
cd back
```
3. Lancer l'api avec la commande suivante: 
```python 
python api.py
```
4. Ouvrez un autre terminal et placez vous aussi dans le dossier back
5. Appelez le fonction correspondante avec la commande suivante : 
```python 
curl -X POST "http://localhost:8000/itinerary/route" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "map_id=1" \
  -d "start=156,561" \
  -d "end=454,221"
```
Remarques : map_id représente l'échelle, start le point de départ et end le point d'arrivée
6. Défilez le résultat jusqu'en bas et retrouvez les instructions textuelles de l'itinéraire réalisé entre les 2 points selectionnés
7. Pour arrêter l'API, retournez dans le terminal et cliquez "Ctrl+C"

## Remarques
Vous trouverez le guide utilisateur final (cas du projet fini) :
- au format WORD dans le google drive du projet
- dans le fichier Final_guide.md à la racine du dossier