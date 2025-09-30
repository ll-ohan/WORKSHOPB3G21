# Batmap

## Contexte du projet

Ce projet est né dans le cadre d’un challenge académique d’une semaine, réunissant cinq étudiants issus de parcours complémentaires. L’objectif était de concevoir, en équipe, une solution innovante répondant à une consigne commune, en mobilisant à la fois nos compétences techniques, créatives et organisationnelles.

Le scénario proposé plaçait l’humanité dans un futur dystopique : après une guerre victorieuse contre l’intelligence artificielle « Ultron », celle-ci, loin d’avoir disparu, a muté sous la forme d’un virus numérique omniprésent. Internet, jadis pilier des communications et des infrastructures, est désormais profondément corrompu. Chaque action en ligne, de la simple consultation d’information au téléchargement d’un fichier, représente une menace potentielle : intrusion virale, manipulation des données, prise de contrôle des systèmes critiques. La vie quotidienne en est radicalement transformée : réseaux électriques instables, signalisations routières défaillantes, systèmes financiers compromis, jusqu’aux satellites et services de géolocalisation rendus inexploitables.

Dans un tel monde, la perte d’Internet ne signifie pas seulement la fin de l’information instantanée, mais aussi l’effondrement silencieux des outils invisibles qui régissent nos vies : hôpitaux incapables de synchroniser leurs données, transports paralysés par des signalisations corrompues, échanges commerciaux figés par des systèmes bancaires inopérants. L’absence de confiance dans les flux numériques fait peser un climat de suspicion permanent, où chaque fichier téléchargé ou message reçu peut être porteur d’une menace invisible. Le quotidien devient plus lent, plus dangereux, et la coopération entre individus se retrouve entravée.

Dans ce contexte, la mobilité devient un enjeu vital mais précaire. L’accès au carburant est rare, les routes sont dégradées, parfois abandonnées ou obstruées, et les trajets motorisés demeurent dangereux. Se déplacer à pied ou à vélo constitue bien souvent la seule option réaliste. C’est dans ce contexte que notre équipe a choisi de concentrer son projet sur un besoin essentiel : **permettre à chacun de naviguer plus sereinement dans cet environnement incertain**.

La contrainte principale était claire : tout devait fonctionner **hors ligne** et exclusivement à partir de ressources **open source**. Il était interdit de s’appuyer sur des services en ligne, des API externes ou l’entraînement de nouveaux modèles d’IA nécessitant Internet. Dès lors, il nous a fallu réinventer la roue en concevant un système capable de fournir un service de navigation robuste, sans dépendance à des données corrompues ou inaccessibles.

Notre réponse fut **Batmap** : un GPS alternatif et collaboratif. L’utilisateur capture une simple photo ou scanne un plan papier. L’application analyse automatiquement l’image, en extrait les routes, puis calcule un itinéraire optimal en tenant compte de l’échelle et de la topologie relevée. L’utilisateur peut, de manière participative, signaler des zones impraticables en posant des « barrières » sur la carte, garantissant ainsi que la communauté bénéficie d’une cartographie plus réaliste et évolutive.

En somme, Batmap incarne une tentative de redonner un sens au mot « orientation » dans un monde privé de ses infrastructures numériques. Le projet ne se limite pas à une prouesse technique : il s’inscrit dans une réflexion plus large sur la **résilience des sociétés humaines** face à la dépendance aux technologies connectées. Comment continuer à se repérer, à se déplacer et à coopérer lorsque les outils qui structuraient nos vies deviennent soudainement inopérants, voire hostiles ? Batmap propose une première réponse, simple, pragmatique et surtout réappropriable par tous.

## Démarrage rapide avec Docker

### Prérequis

- Docker Desktop ou Docker Engine ≥ 24
- Plugin Docker Compose v2

### Lancement standard

1. Cloner ce dépôt puis se placer à sa racine.
2. Construire et démarrer la stack : `docker compose up --build`.
3. Attendre que `batmap-api` affiche `Uvicorn running on http://0.0.0.0:8000` et que `batmap-web` signale `nginx entered RUNNING state`.
4. Ouvrir `http://localhost:8080` pour le client web et `http://localhost:8000/docs` pour la documentation interactive de l’API.

### Services exposés

- API FastAPI : `http://localhost:8000` (OpenAPI, traitement d’images, calculs d’itinéraires).
- Client web statique : `http://localhost:8080` (front Nginx servant le dossier `web-client`).
- Volumes persistants :
  - `./map_store` → stockage des cartes finales, des binaires et des masques RGBA.
  - `./data.db` → base SQLite embarquant les métadonnées.

### Commandes utiles

- Arrêt propre : `docker compose down`.
- Reconstruction forcée après modification du code : `docker compose build`.
- Inspection des logs : `docker compose logs -f api` ou `docker compose logs -f web`.
- Purge totale (attention, supprime les données locales) : `docker compose down -v`.

### Personnalisation rapide

- Proxy API côté Nginx : variable `API_PROXY_PASS` (par défaut `http://api:8000/`).
- Ports d’exposition : modifier les sections `ports` du `docker-compose.yml` si collision.
- Base front : le script `web-client/assets/js/api-base.js` détecte automatiquement l’origine (`window.location.origin`) et ajoute `/api`; pour un backend distant, définir `window.BATMAP_API_BASE` dans `index.html` avant le chargement des scripts.

## Guide utilisateur

- **Accueil** : écran d’introduction avec animation parallax, accès aux parcours « Ajouter une carte » et « Consulter mes cartes ».
- **Ajouter une nouvelle carte** (`new_map.html`) :
  - Importer un plan (PNG, JPEG, TIFF) - avec métadonnée DPI puis saisir la ville associée.
  - Renseigner l’échelle papier → terrain via la double saisie unitaire (conversion automatique en mètres).
  - L’API retourne deux binarisations : choisir la meilleure et finaliser. La carte est persistée et un masque transparent est généré pour annoter des obstacles.
- **Sélectionner une carte enregistrée** (`select_map.html`) :
  - Choisir une ville dans la liste dynamique alimentée par `GET /map/cities`.
  - Parcourir les vignettes (base64) retournées par `GET /map/list` puis ouvrir le détail voulu.
- **Consulter une carte** (`map_detail.html`) :
  - Cliquer pour définir un point de départ (cyan) et un point d’arrivée (magenta).
  - Lancer le calcul d’itinéraire : la réponse de `POST /itinerary/route` superpose l’itinéraire, affiche distance métrique, durée de marche estimée et instructions textuelles.
  - Basculer entre vue carte et vue itinéraire via le panneau pivotant.
- **Gestion des obstacles temporaires** :
  - Mode barrière pour tracer une zone impraticable (`POST /map/modify_bin`), directement fusionnée dans le masque RGBA.
  - Réinitialisation complète du masque depuis la vue détail (`DELETE /map/{id}/bin/reset`).
- **Administration légère** :
  - Nettoyage automatique des téléversements abandonnés via tâche d’arrière-plan (`cleanup_task`).
  - L’interface affiche des messages contextuels (succès/erreur) via les composants `flash-message`.

## Déploiement détaillé

### API FastAPI

#### Environnement local sans Docker

1. Créer un environnement virtuel : `python -m venv .venv && source .venv/bin/activate`.
2. Installer les dépendances : `pip install --upgrade pip && pip install -r requirements.txt`.
3. Vérifier la présence du dossier `map_store` (créé automatiquement au premier lancement) et du fichier `data.db` (auto-création via `init_db`).
4. Démarrer le serveur : `uvicorn api:app --reload --host 0.0.0.0 --port 8000`.
5. Accéder à `http://127.0.0.1:8000/docs` pour tester les endpoints.

> Dépendances système : OpenCV nécessite `libgl1` et `libglib2.0-0` (déjà installées dans le Docker, à prévoir manuellement sous Linux minimal).

#### Environnement local avec Docker

1. Construire l’image : `docker build -f Dockerfile.api -t batmap-api .`.
2. Lancer en standalone :
   - `docker run --rm -it -p 8000:8000 \`
     `-v $(pwd)/map_store:/app/map_store \`
     `-v $(pwd)/data.db:/app/data.db \`
     `batmap-api`.
3. Les volumes montés garantissent la persistance hors du conteneur.
4. Ajuster `--reload` (activé par défaut) selon le contexte de prod.

### Client web

#### Environnement local sans Docker

1. Construire les assets statiques (déjà présents dans `web-client`).
2. Servir le dossier avec un serveur de fichiers, par exemple : `cd web-client && python -m http.server 8080`.
3. Si l’API tourne sur une autre origine, définir `window.BATMAP_API_BASE = "http://ADRESSE:8000";` dans `index.html` avant l’inclusion des scripts.
4. Pour un hébergement plus avancé, utiliser un serveur statique (Nginx, Caddy) ou une extension VS Code type Live Server.

#### Environnement local avec Docker

1. Construire l’image : `docker build -f Dockerfile.web -t batmap-web .`.
2. Lancer en autonome :
   - `docker run --rm -it -p 8080:80 \`
     `-e API_PROXY_PASS=http://192.168.0.10:8000/ \`
     `batmap-web`.
3. Le script `docker/nginx/entrypoint.sh` injecte dynamiquement `API_PROXY_PASS` dans la configuration Nginx.
4. Monter `-v $(pwd)/web-client:/usr/share/nginx/html:ro` pour tester des modifications à chaud.

### Déploiement réseau local sécurisé (deux machines)

#### Hypothèses

- Machine A (API) : adresse statique `192.168.10.21`, exécutant FastAPI.
- Machine B (client web) : adresse statique `192.168.10.22`, exposant le front.
- Pare-feu ouverts uniquement sur les ports nécessaires (TCP 8000 pour l’API, TCP 80/443 pour le front).

#### Étapes côté API (Machine A)

1. Cloner le dépôt et préparer les dossiers `map_store` et `data.db` (ou monter un volume chiffré).
2. Lancer en production :
   - Option Docker : `docker run --name batmap-api -d -p 8000:8000 -v /srv/batmap/map_store:/app/map_store -v /srv/batmap/data.db:/app/data.db batmap-api`.
   - Option bare-metal : `uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4` derrière un reverse proxy.
3. Restreindre l’accès via pare-feu (`ufw allow from 192.168.10.22 to any port 8000`).
4. Placer Nginx en frontal pour activer TLS interne (certificat auto-signé ou AC interne).

#### Étapes côté client (Machine B)

1. Déployer l’image `batmap-web` ou copier le dossier `web-client` sur un serveur Nginx.
2. Configurer la résolution API :
   - Docker : `docker run --name batmap-web -d -p 80:80 -e API_PROXY_PASS=http://192.168.10.21:8000/ batmap-web`.
   - Serveur statique : éditer `index.html` pour définir `window.BATMAP_API_BASE = "http://192.168.10.21:8000";`.
3. Activer HTTPS si le front est diffusé au-delà du LAN (Let’s Encrypt ou certificats d’entreprise).
4. Limiter l’accès au port 80/443 aux sous-réseaux autorisés (`ufw allow from 192.168.10.0/24 to any port 80`).

#### Synchronisation et maintenance

- Sauvegardes périodiques :
  - `map_store` (fichiers binaires) → stratégie incrémentale.
  - `data.db` (SQLite) → copie chaude via `sqlite3 data.db ".backup data_$(date +%F).db"`.
- Surveillance :
  - Santé API via `/docs` ou `/redoc` (restreindre par IP ou authentification).
  - Journaux centralisés (`docker logs`, journalctl).
- Rotation des obstacles : script cron `DELETE /map/{id}/bin/reset` selon politique.

## Algorithmes

### Traitement d’image (Python)

- **Extraction des voies (`map_mader.py`)** : pipeline paramétrable combinant débruitage non-local (`cv2.fastNlMeansDenoising`), opérations morphologiques elliptiques et binarisation multi-méthodes (Otsu, adaptative moyenne/gaussienne). Cette orchestration de techniques offre plusieurs variantes exploitables en fonction des contraintes du terrain.
- **Gestion des formats** : lecture depuis flux mémoire ou fichiers, conversion systématique en niveaux de gris et export PNG homogène (`cv2.imwrite`), avec inversion automatique lorsque le contraste source est atypique.

### Calcul d’itinéraire et navigation (Python)

- **Projection géométrique (`itinerary.py`)** : association rapide des clics utilisateur aux zones praticables grâce à un KD-tree (`scipy.spatial.cKDTree`) sur masque binaire.
- **Recherche du plus court chemin** : Dijkstra 8-connecté avec diagonales contrôlées et garde-fous contre le corner-cutting. Les waypoints sont densifiés pour garantir un tracé continu et fluide.
- **Lissage et lisibilité** : simplification double (Ramer-Douglas-Peucker) et redressement heuristique des segments courts, afin de restituer des indications naturelles à l’échelle humaine.
- **Interprétation métrique** : passage des pixels aux centimètres via les DPI embarqués, projection à l’échelle déclarée et estimation du temps de parcours (1.4 m/s).
- **Résilience** : prise en charge RGBA pour obstacles semi-transparents, réécriture de masques via `barreRoute`, et nettoyage disque asynchrone (`cleanup_task`).

### Interaction frontale (JavaScript)

- **Initialisation adaptative (`web-client/assets/js/api-base.js`)** : sélection dynamique de la base API selon le contexte, avec fallback HTTP hors ligne.
- **Expérience utilisateur** :
  - Effets parallax légers, accessibles et limités pour préserver la lisibilité.
  - Messages inline injectés dynamiquement pour une cohérence de feedback.
  - Conversion multi-unités (`computeScale`) pour rendre l’échelle universelle.
- **Navigation métier** :
  - Téléversement progressif avec prévisualisation, complétions automatiques et choix de méthode de binarisation.
  - Galerie filtrée par ville, accès aux détails et rendu canvas synchronisé avec l’itinéraire.

### Gouvernance des données et architecture

- **Persistance légère** : base SQLite unique (`data.db`) avec indexation par ville et gestion d’historique via timestamps.
- **Organisation fichiers** : répertoires dédiés `map_store/{color,bin,mask,temp}` séparant originaux, binaires, masques éditables et temporaires.
- **API contract-first** : endpoints REST standardisés (`/map/*`, `/itinerary/route`) décrits par OpenAPI et réutilisables par d’autres clients (mobile, CLI).
- **Observabilité** : journalisation ciblée des opérations sensibles (suppression, parsing) afin de simplifier l’audit et le débogage.
