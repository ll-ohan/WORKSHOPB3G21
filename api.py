from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import cv2
import numpy as np
import base64
import sqlite3
from pathlib import Path
from typing import Dict
import shutil
import uuid
from map_mader import RoadExtractor
from itinerary import findNearestRoutePoints, computeShortestPath, drawPath, pixels_to_cm, barreRoute, textual_itinerary

DB_PATH = Path("data.db")
DB_TABLE_NAME = "map_data"

def init_db():
    """Initialise la base SQLite et crée la table si besoin."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            map TEXT NOT NULL,          -- chemin de l’image originale
            original_bin TEXT,          -- chemin version binaire 1 (nullable)
            use_bin TEXT,               -- chemin version binaire 2 (nullable)
            temp_bin_1 TEXT,            -- chemin temporaire binaire 1 (nullable)
            temp_bin_2 TEXT,            -- chemin temporaire binaire 2 (nullable)
            city TEXT NOT NULL,
            scale REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()
    print("✅ Base SQLite initialisée.")

def init_dirs():
    """Crée l’arborescence map_store/ si elle n’existe pas."""
    subdirs = ["temp", "color", "bin"]
    for sub in subdirs:
        path = Path("map_store") / sub
        path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Répertoires initialisés dans map_store/")
import asyncio
import datetime

async def cleanup_task():
    """Supprime les entrées temporaires vieilles de plus de 60 min."""
    while True:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Sélection des entrées à supprimer
            cutoff = (datetime.datetime.utcnow() - datetime.timedelta(minutes=60)).isoformat(" ")
            cursor.execute(f"""
                SELECT id, map, temp_bin_1, temp_bin_2
                FROM {DB_TABLE_NAME}
                WHERE use_bin IS NULL AND created_at < ?
            """, (cutoff,))
            rows = cursor.fetchall()

            # Suppression fichiers + DB
            for entry_id, map_path, temp1, temp2 in rows:
                for f in [map_path, temp1, temp2]:
                    if f and Path(f).exists():
                        try:
                            Path(f).unlink()
                        except Exception as e:
                            print(f"⚠️ Erreur suppression {f}: {e}")

                cursor.execute(f"DELETE FROM {DB_TABLE_NAME} WHERE id = ?", (entry_id,))

            conn.commit()
            conn.close()

            if rows:
                print(f"🧹 Nettoyage effectué, {len(rows)} entrées supprimées")

        except Exception as e:
            print(f"⚠️ Erreur dans cleanup_task: {e}")

        await asyncio.sleep(300)  # relance toutes les 5 minutes

app = FastAPI(title="Batmap API", version="1.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Code exécuté quand l’API démarre et se termine."""
    init_db()
    init_dirs()
    task = asyncio.create_task(cleanup_task())
    yield
    task.cancel()

app = FastAPI(title="Batmap API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: limiter aux origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_image(path: Path, image: np.ndarray):
    """Sauvegarde une image OpenCV en PNG."""
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"Erreur lors de l’enregistrement de {path}")

def encode_image_to_base64(img) -> str:
    """Encode une image numpy en base64 (format PNG)."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Erreur lors de l'encodage de l'image en PNG.")
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/map/binaryse")
async def binaryse(
    file: UploadFile = File(...),
    scale: float = Form(...),
    city: str = Form(...)
):
    """
    Upload d'une image avec champs scale & city.
    Sauvegarde l’image originale + les résultats dans map_store/temp,
    insère les infos en DB, et renvoie les résultats encodés en base64.
    """
    try:
        contents = await file.read()

        # Sauvegarde image originale dans temp
        temp_dir = Path("map_store/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        original_path = temp_dir / f"original_{file.filename}"
        with open(original_path, "wb") as f:
            f.write(contents)

        # Exécution du traitement
        extractor = RoadExtractor(contents)
        results = extractor.run()  # dict {method: np.ndarray}

        # Sauvegarde et encodage des résultats
        result_files: Dict[str, str] = {}
        for method, img in results.items():
            out_path = temp_dir / f"{method}_{file.filename}"
            save_image(out_path, img)
            result_files[str(out_path)] = encode_image_to_base64(img)

        # Insertion en DB (on garde chemins, pas base64)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {DB_TABLE_NAME}(map, temp_bin_1, temp_bin_2, city, scale)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(original_path),
            str(temp_dir / f"method_0_{file.filename}"),
            str(temp_dir / f"method_1_{file.filename}"),
            city,
            scale
        ))
        conn.commit()
        conn.close()

        # Réponse avec images encodées
        return JSONResponse(content={
            "filename": file.filename,
            "city": city,
            "scale": scale,
            "results": result_files  # base64 des images
        })

    except Exception as e:
        print(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/map/add")
async def add_map(
    map: str = Form(...),
    map_choosen: str = Form(...),
    city: str = Form(...),
    scale: float = Form(...)
):
    """
    Finalise une carte en déplaçant l'original et le binaire choisi
    depuis map_store/temp vers map_store/color et map_store/bin.
    Met à jour la DB avec les nouveaux chemins.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Vérifie si la ligne existe
        cursor.execute(f"""
            SELECT id, map, temp_bin_1, temp_bin_2
            FROM {DB_TABLE_NAME} WHERE map = ? AND city = ? AND scale = ?
        """, (map, city, scale))
        row = cursor.fetchone()

        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Entrée non trouvée dans la base")

        entry_id, original_path, temp1, temp2 = row

        # Vérifie existence fichiers
        orig_file = Path(original_path)
        chosen_file = Path(map_choosen)

        if not orig_file.exists():
            conn.close()
            raise HTTPException(status_code=404, detail=f"Fichier original introuvable: {orig_file}")
        if not chosen_file.exists():
            conn.close()
            raise HTTPException(status_code=404, detail=f"Fichier binaire choisi introuvable: {chosen_file}")

        # Génération noms uniques
        uid = uuid.uuid4().hex[:8]
        new_orig_path = Path("map_store/color") / f"map_{uid}.png"
        new_bin_path = Path("map_store/bin") / f"bin_{uid}.png"

        # Déplacement fichiers
        shutil.move(str(orig_file), new_orig_path)
        shutil.move(str(chosen_file), new_bin_path)

        # Supprimer le fichiers non choisis
        for temp_file in [temp1, temp2]:
            if temp_file and Path(temp_file) != chosen_file:
                try:
                    Path(temp_file).unlink()
                except FileNotFoundError:
                    pass  # au cas où il aurait déjà été supprimé

        # Mise à jour DB
        cursor.execute(f"""
            UPDATE {DB_TABLE_NAME}
            SET map = ?, original_bin = ?, use_bin = ?, temp_bin_1 = NULL, temp_bin_2 = NULL
            WHERE id = ?
        """, (str(new_orig_path), str(new_bin_path), str(new_bin_path), entry_id))

        conn.commit()
        conn.close()

        return JSONResponse(content={
            "id": entry_id,
            "map": str(new_orig_path),
            "original_bin": str(new_bin_path),
            "use_bin": str(new_bin_path),
            "city": city,
            "scale": scale
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/map/cities")
async def list_cities():
    """
    Retourne la liste des villes présentes dans la base de données.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT DISTINCT city
            FROM {DB_TABLE_NAME}
            WHERE use_bin IS NOT NULL
            ORDER BY city ASC
        """)
        rows = cursor.fetchall()
        conn.close()

        cities = [row[0] for row in rows]

        return JSONResponse(content={"cities": cities})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/map/list")
async def list_maps(city: str):
    """
    Récupère la liste des cartes filtrées par ville.
    Retourne les chemins et les images encodées en base64.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT id, map, use_bin, city, scale, created_at
            FROM {DB_TABLE_NAME}
            WHERE city = ? AND use_bin IS NOT NULL
            ORDER BY created_at DESC
        """, (city,))
        rows = cursor.fetchall()
        conn.close()

        results = []
        for entry_id, map_path, use_bin_path, city, scale, created_at in rows:
            # Charger et encoder en base64 (si fichier existe)
            map_b64 = None
            bin_b64 = None

            if map_path and Path(map_path).exists():
                img = cv2.imread(map_path)
                if img is not None:
                    map_b64 = encode_image_to_base64(img)


            results.append({
                "id": entry_id,
                "city": city,
                "scale": scale,
                "created_at": created_at,
                "map": map_b64,
            })

        return JSONResponse(content={"results": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/map/{map_id}")
async def get_map(map_id: int):
    """
    Récupère une carte unique par son identifiant.
    Retourne les chemins et les images encodées en base64 si disponibles.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT id, map, use_bin, city, scale, created_at
            FROM {DB_TABLE_NAME}
            WHERE id = ? AND use_bin IS NOT NULL
        """, (map_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Carte introuvable")

        entry_id, map_path, use_bin_path, city, scale, created_at = row

        map_b64 = None
        bin_b64 = None

        if map_path and Path(map_path).exists():
            img = cv2.imread(map_path)
            if img is not None:
                map_b64 = encode_image_to_base64(img)

        if use_bin_path and Path(use_bin_path).exists():
            img = cv2.imread(use_bin_path)
            if img is not None:
                bin_b64 = encode_image_to_base64(img)

        return JSONResponse(content={
            "id": entry_id,
            "city": city,
            "scale": scale,
            "created_at": created_at,
            "map": map_b64,
            "use_bin": bin_b64
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import heapq
from scipy.ndimage import label
from scipy.spatial import cKDTree

@app.post("/itinerary/route")
async def itinerary_route(
    map_id: int = Form(...),
    start: str = Form(...),  # format: "x,y"
    end: str = Form(...)
):
    """
    Calcule l'itinéraire le plus court entre deux points (start, end)
    sur la carte binaire associée à `map_id`.
    Retourne l'image avec le trajet dessiné et la distance parcourue (mètres).
    """
    try:
        # Parsing start et end
        try:
            start_pt = tuple(map(int, start.split(",")))
            end_pt = tuple(map(int, end.split(",")))
        except Exception as e:
            print(f"Erreur parsing points: {e}")
            raise HTTPException(status_code=400, detail="Format invalide pour start ou end (attendu: 'x,y')")

        # Récupération de la carte
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT map, use_bin, scale
            FROM {DB_TABLE_NAME}
            WHERE id = ?
        """, (map_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Carte introuvable")
        map_path, use_bin_path, scale = row

        if not Path(use_bin_path).exists():
            raise HTTPException(status_code=404, detail=f"Fichier binaire introuvable: {use_bin_path}")

        # Chargement du masque binaire
        mask = cv2.imread(use_bin_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if mask is None:
            raise HTTPException(status_code=500, detail="Impossible de charger le masque binaire")

        # Trouver candidats
        all_candidates = findNearestRoutePoints(mask, [start_pt, end_pt])

        # Calcul plus court chemin
        start_real, end_real, path = computeShortestPath(mask, all_candidates)
        if not path:
            raise HTTPException(status_code=404, detail="Aucun chemin trouvé entre les deux points")

        # Dessiner chemin sur image couleur
        img = cv2.imread(map_path)  # pour avoir une base
        img_path = drawPath(img, path, start_real, end_real)

        # Distance en mètres (pixels → cm → m) avec l'échelle
        length_pixels = len(path)
        length_cm = pixels_to_cm(length_pixels)
        length_m = (length_cm) * scale  # échelle appliquée

        scale_m_per_px = pixels_to_cm(1) * scale  # m par pixel
        steps = textual_itinerary(path, scale_m_per_px)  # liste[str]

        return JSONResponse(content={
            "map_id": map_id,
            "distance_m": round(length_m, 2),
            "image": encode_image_to_base64(img_path),
            "itinerary": steps
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/map/modify_bin")
async def modify_bin(
    map_id: int = Form(...),
    start: str = Form(...),  # format: "x,y"
    end: str = Form(...)
):
    """
    Modifie le masque binaire associé à une carte en "barrant" une route
    entre deux points. Le masque modifié est enregistré dans un NOUVEAU fichier
    nommé <original>_user_modified.png, et la base est mise à jour.
    """
    try:
        # Parsing start et end
        try:
            start_pt = tuple(map(int, start.split(",")))
            end_pt = tuple(map(int, end.split(",")))
        except Exception:
            raise HTTPException(status_code=400, detail="Format invalide pour start ou end (attendu: 'x,y')")

        # Récupération du chemin du binaire
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT use_bin
            FROM {DB_TABLE_NAME}
            WHERE id = ?
        """, (map_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Carte introuvable")

        use_bin_path = row[0]
        conn.close()

        if not Path(use_bin_path).exists():
            raise HTTPException(status_code=404, detail=f"Fichier binaire introuvable: {use_bin_path}")

        # Chargement du masque
        mask = cv2.imread(use_bin_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(status_code=500, detail="Impossible de charger le masque binaire")

        # Barrer la route
        mask_barr = barreRoute(mask, [start_pt, end_pt])

        # Nouveau chemin : on ajoute _user_modified avant l’extension
        orig_path = Path(use_bin_path)
        new_name = orig_path.stem + "_user_modified.png"
        new_path = orig_path.parent / new_name

        # Sauvegarde du nouveau masque
        cv2.imwrite(str(new_path), mask_barr)

        # Mise à jour DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE {DB_TABLE_NAME}
            SET use_bin = ?
            WHERE id = ?
        """, (str(new_path), map_id))
        conn.commit()
        conn.close()

        return JSONResponse(content={
            "map_id": map_id,
            "new_use_bin": str(new_path),
            "message": f"Route barrée entre {start_pt} et {end_pt}. Nouveau fichier créé."
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/map/{map_id}/bin/reset")
async def reset_bin(map_id: int):
    """
    Réinitialise le masque binaire (use_bin) en le remplaçant par l'original_bin.
    Supprime le fichier use_bin actuel s'il est différent.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT original_bin, use_bin
            FROM {DB_TABLE_NAME}
            WHERE id = ?
        """, (map_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Carte introuvable")

        original_bin, use_bin = row

        if use_bin == original_bin:
            raise HTTPException(status_code=409, detail="Le masque est déjà à l'état original")

        # Supprimer l'ancien fichier use_bin (si différent et existant)
        if use_bin and Path(use_bin).exists() and use_bin != original_bin:
            try:
                Path(use_bin).unlink()
            except Exception as e:
                print(f"⚠️ Impossible de supprimer {use_bin}: {e}")

        # Mise à jour DB → use_bin = original_bin
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE {DB_TABLE_NAME}
            SET use_bin = ?
            WHERE id = ?
        """, (original_bin, map_id))
        conn.commit()
        conn.close()

        return JSONResponse(content={
            "map_id": map_id,
            "use_bin": original_bin,
            "message": "Le masque a été réinitialisé à sa version originale"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Lancer avec : python api.py
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
