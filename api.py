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
from itinerary import findNearestRoutePoints, computeShortestPath, drawPath, pixels_to_cm, barreRoute, textual_itinerary, get_image_dpi
import asyncio
import datetime
import math

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

def init_dirs():
    """Crée l’arborescence map_store/ si elle n’existe pas."""
    subdirs = ["temp", "color", "bin", "mask"]
    for sub in subdirs:
        path = Path("map_store") / sub
        path.mkdir(parents=True, exist_ok=True)

async def cleanup_task():
    """Supprime les entrées temporaires vieilles de plus de 60 min."""
    while True:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            #Selection des entrées à supprimer
            cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=60)).isoformat(" ")
            cursor.execute(f"""
                SELECT id, map, temp_bin_1, temp_bin_2
                FROM {DB_TABLE_NAME}
                WHERE use_bin IS NULL AND created_at < ?
            """, (cutoff,))
            rows = cursor.fetchall()

            #Suppression fichiers + DB
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

        except Exception as e:
            print(f"⚠️ Erreur dans cleanup_task: {e}")

        await asyncio.sleep(5*60)

def ensure_rgba(img):
    """Force une image à être RGBA (ajoute canal alpha si besoin)."""
    if img is None:
        return None
    if len(img.shape) == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:  # BGR → BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 4:  # déjà OK
        pass
    else:
        raise ValueError("Format image inattendu")
    return img


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
    allow_origins=["*"],  # TODO: limiter aux origines
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


def compute_path_length_m(path, scale_m_per_px):
    """Calcule la longueur réelle d'un chemin discret en mètres."""
    if not path or len(path) < 2:
        return 0.0
    total_px = 0.0
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        total_px += math.hypot(x1 - x0, y1 - y0)
    return total_px * scale_m_per_px


def format_walking_time(length_m, speed_m_s=1.4):
    """Retourne une estimation de durée de marche à vitesse constante."""
    if length_m <= 0:
        return "1 min"

    total_minutes = length_m / speed_m_s / 60
    if total_minutes < 1:
        return "1 min"

    hours = int(total_minutes // 60)
    minutes = int(round(total_minutes % 60))

    if minutes == 60:
        hours += 1
        minutes = 0

    if hours:
        return f"{hours}h" if minutes == 0 else f"{hours}h {minutes:02d} min"
    return f"{minutes} min"


@app.post("/map/binaryse")
async def binaryse(
    file: UploadFile = File(...),
    scale: float = Form(...),
    city: str = Form(...)
):
    """
    Upload d'une map: Sauvegarde l’image originale + les résultats dans map_store/temp, insère les infos en DB, et renvoie les résultats encodés en base64.
    """
    try:
        contents = await file.read()

        #Sauvegarde image original dans temp
        temp_dir = Path("map_store/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        original_path = temp_dir / f"original_{file.filename}"
        with open(original_path, "wb") as f:
            f.write(contents)
            dpi = get_image_dpi(original_path)
            if dpi is None:
                original_path.unlink()
                raise HTTPException(status_code=400, detail="DPI introuvable dans l'image. Assurez-vous que l'image contient des informations DPI valides.")
            



        #Exec du traitement
        extractor = RoadExtractor(contents)
        results = extractor.run()

        #Sauvegarde
        result_files: Dict[str, str] = {}
        for method, img in results.items():
            out_path = temp_dir / f"{method}_{file.filename}"
            save_image(out_path, img)
            result_files[str(out_path)] = encode_image_to_base64(img)

        #Insertion à la DB
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

        return JSONResponse(content={
            "filename": file.filename,
            "city": city,
            "scale": scale,
            "results": result_files
        })
    except HTTPException:
        raise

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
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT id, map, temp_bin_1, temp_bin_2
            FROM {DB_TABLE_NAME} WHERE map = ? AND city = ? AND scale = ?
        """, (map, city, scale))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Entrée non trouvée dans la base")

        entry_id, original_path, temp1, temp2 = row
        orig_file = Path(original_path)
        chosen_file = Path(map_choosen)
        if not orig_file.exists():
            conn.close()
            raise HTTPException(status_code=404, detail=f"Fichier original introuvable: {orig_file}")
        if not chosen_file.exists():
            conn.close()
            raise HTTPException(status_code=404, detail=f"Fichier binaire choisi introuvable: {chosen_file}")

        uid = uuid.uuid4().hex[:8]
        new_orig_path = Path("map_store/color") / f"map_{uid}.png"
        new_bin_path = Path("map_store/bin") / f"bin_{uid}.png"
        new_mask_path = Path("map_store/mask") / f"mask_{uid}.png"

        shutil.move(str(orig_file), new_orig_path)
        shutil.move(str(chosen_file), new_bin_path)

        for temp_file in [temp1, temp2]:
            if temp_file and Path(temp_file) != chosen_file:
                try:
                    Path(temp_file).unlink()
                except FileNotFoundError:
                    pass

        # Créer mask transparent de même taille que new_orig_path
        img_ref = cv2.imread(str(new_orig_path))
        h, w = img_ref.shape[:2]
        mask_empty = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA transparent
        cv2.imwrite(str(new_mask_path), mask_empty)

        cursor.execute(f"""
            UPDATE {DB_TABLE_NAME}
            SET map = ?, original_bin = ?, use_bin = ?, temp_bin_1 = NULL, temp_bin_2 = NULL
            WHERE id = ?
        """, (str(new_orig_path), str(new_bin_path), str(new_mask_path), entry_id))

        conn.commit()
        conn.close()

        return JSONResponse(content={
            "id": entry_id,
            "map": str(new_orig_path),
            "original_bin": str(new_bin_path),
            "use_bin": str(new_mask_path),
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

        #requete
        cursor.execute(f"""
            SELECT DISTINCT city
            FROM {DB_TABLE_NAME}
            WHERE use_bin IS NOT NULL
            ORDER BY city ASC
        """)
        rows = cursor.fetchall()
        conn.close()

        cities = [row[0] for row in rows] #car rows=liste de tuples

        return JSONResponse(content={"cities": cities})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/map/list")
async def list_maps(city: str):
    """
    Récupère la liste des cartes filtrées par ville.
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
            if use_bin_path and Path(use_bin_path).exists():
                if map_path and Path(map_path).exists():
                    img = cv2.imread(map_path)
                    if img is not None:
                        map_b64 = encode_image_to_base64(img)
            else:
                pass #map non finaliséé

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
    Récupère une carte par son identifiant.
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

        if map_path and Path(map_path).exists():
            img = cv2.imread(map_path)
            mask_rgba = cv2.imread(use_bin_path, cv2.IMREAD_UNCHANGED)
            mask_rgba = ensure_rgba(mask_rgba)
            if mask_rgba is not None:
                alpha = mask_rgba[:,:,3] / 255.0
                for c in range(3):
                    img[:,:,c] = (1-alpha) * img[:,:,c] + alpha * mask_rgba[:,:,c]
            map_b64 = encode_image_to_base64(img)

        return JSONResponse(content={
            "id": entry_id,
            "city": city,
            "scale": scale,
            "created_at": created_at,
            "map": map_b64,
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/itinerary/route")
async def itinerary_route(
    map_id: int = Form(...),
    start: str = Form(...),
    end: str = Form(...)
):
    """
    Calcule l'itinéraire le plus court entre deux points (start, end) sur la carte. Retourne l'image avec le trajet dessiné et la distance parcourue (mètres).
    """
    try:
        try:
            start_pt = tuple(map(int, start.split(","))) #récupère les points
            end_pt = tuple(map(int, end.split(",")))
        except Exception as e:
            print(f"Erreur parsing points: {e}")
            raise HTTPException(status_code=400, detail="Format invalide pour start ou end (attendu: 'x,y')")

        #récupération de la carte
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT map, original_bin, use_bin, scale
            FROM {DB_TABLE_NAME}
            WHERE id = ?
        """, (map_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Carte introuvable")
        map_path, original_bin_path, use_bin_path, scale = row

        if not Path(use_bin_path).exists():
            raise HTTPException(status_code=404, detail=f"Fichier binaire introuvable: {use_bin_path}")

        #chargement du masque binaire
        try:
            mask_rgba = cv2.imread(use_bin_path, cv2.IMREAD_UNCHANGED)
            mask_rgba = ensure_rgba(mask_rgba)
            violet_mask = np.zeros(mask_rgba.shape[:2], dtype=np.uint8)
            if mask_rgba is not None and mask_rgba.shape[2] == 4:
                violet_mask[np.where((mask_rgba[:,:,0]==255) & (mask_rgba[:,:,2]==255))] = 0  # noir
                violet_mask[np.where((mask_rgba[:,:,0]==0) & (mask_rgba[:,:,1]==0) & (mask_rgba[:,:,2]==0) & (mask_rgba[:,:,3]==0))] = 255  # blanc ailleurs
            orig_bin = cv2.imread(original_bin_path, cv2.IMREAD_GRAYSCALE)
            _, orig_bin = cv2.threshold(orig_bin, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(orig_bin, violet_mask)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Erreur lors de l'utilisation d'obstacles. Veuillez réinitialiser les obstacles.")
        if mask is None:
            raise HTTPException(status_code=500, detail="Impossible de charger le masque binaire")

        #trouver candidats
        all_candidates = findNearestRoutePoints(mask, [start_pt, end_pt])

        # Distance en mètres (pixels → cm → m) avec l'échelle
        dpi = get_image_dpi(map_path)  #DPI de l'image
        scale_m_per_px = pixels_to_cm(1, dpi)/100 * scale  # m par pixel
        
        #Calcul plus court chemin
        start_real, end_real, path, nav_path = computeShortestPath(mask, all_candidates, scale_m_per_px=scale_m_per_px)
        if not path:
            raise HTTPException(status_code=404, detail="Aucun chemin trouvé entre les deux points")

        #Dessiner chemin sur map
        img = cv2.imread(map_path)
        if mask_rgba is not None:
            alpha = mask_rgba[:,:,3] / 255.0
            for c in range(3):
                img[:,:,c] = (1-alpha) * img[:,:,c] + alpha * mask_rgba[:,:,c]
        img_path = drawPath(img, path, start_real, end_real)

        
        steps = textual_itinerary(nav_path, scale_m_per_px=scale_m_per_px)

        length_m = compute_path_length_m(path, scale_m_per_px)
        time_str = format_walking_time(length_m)
        return JSONResponse(content={
            "map_id": map_id,
            "distance_m": round(length_m, 2),
            "image": encode_image_to_base64(img_path),
            "itinerary": steps,
            "estimated_time": time_str
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/map/modify_bin")
async def modify_bin(map_id: int = Form(...), start: str = Form(...), end: str = Form(...)):
    try:
        try:
            start_pt = tuple(map(int, start.split(",")))
            end_pt = tuple(map(int, end.split(",")))
        except Exception:
            raise HTTPException(status_code=400, detail="Format invalide pour start ou end (attendu: 'x,y')")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT use_bin FROM {DB_TABLE_NAME} WHERE id = ?", (map_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Carte introuvable")
        use_bin_path = row[0]
        conn.close()

        if not Path(use_bin_path).exists():
            raise HTTPException(status_code=404, detail=f"Fichier mask introuvable: {use_bin_path}")

        mask = cv2.imread(use_bin_path, cv2.IMREAD_UNCHANGED)  # RGBA
        if mask is None or mask.shape[2] != 4:
            raise HTTPException(status_code=500, detail="Impossible de charger le mask RGBA")

        mask_barr = barreRoute(mask, [start_pt, end_pt])
        cv2.imwrite(use_bin_path, mask_barr)

        return JSONResponse(content={
            "map_id": map_id,
            "use_bin": str(use_bin_path),
            "message": f"Barrière ajoutée entre {start_pt} et {end_pt}"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/map/{map_id}/bin/reset")
async def reset_bin(map_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT original_bin, map, use_bin FROM {DB_TABLE_NAME} WHERE id = ?", (map_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Carte introuvable")

        original_bin, map_path, use_bin = row

        # recrée un masque transparent vierge
        img_ref = cv2.imread(map_path)
        h, w = img_ref.shape[:2]
        mask_empty = np.zeros((h, w, 4), dtype=np.uint8)
        cv2.imwrite(use_bin, mask_empty)

        return JSONResponse(content={
            "map_id": map_id,
            "use_bin": use_bin,
            "message": "Le masque a été réinitialisé"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
