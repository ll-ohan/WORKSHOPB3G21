from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import base64, time
from pathlib import Path
import cv2
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from map_mader import RoadExtractor

app = FastAPI()

BASE_DIR = Path(__file__).parent
MAP_STORE = BASE_DIR / "map_store"
TEMP_DIR = MAP_STORE / "temp"
for d in (MAP_STORE, TEMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Stockage en mémoire des cartes ajoutées
STORED_MAPS = []

def numpy_to_b64_png(img):
    success, png = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("encodage PNG échoué")
    return base64.b64encode(png.tobytes()).decode('ascii')

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/map/binaryse")
async def map_binaryse(file: UploadFile = File(...), scale: str = Form(...), city: str = Form(...)):
    try:
        filename = f"original_{int(time.time())}_{file.filename}"
        temp_path = TEMP_DIR / filename
        with temp_path.open("wb") as f:
            f.write(await file.read())
        extractor = RoadExtractor(str(temp_path))
        results = extractor.run()  # dict of numpy arrays
        encoded = {k: numpy_to_b64_png(img) for k, img in results.items()}
        return JSONResponse({
            "filename": file.filename,
            "city": city,
            "scale": scale,
            "results": encoded
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/map/add")
async def map_add(map_choosen: str = Form(...), city: str = Form(...), scale: str = Form(...), original_filename: str = Form(...)):
    """Ajoute la carte choisie en mémoire"""
    card_data = {
        "timestamp": int(time.time()),
        "city": city,
        "scale": scale,
        "map_choosen": map_choosen,
        "original_filename": original_filename
    }
    STORED_MAPS.append(card_data)
    return {"status": "ok", "map": card_data}

@app.get("/maps")
async def list_maps():
    """Liste toutes les cartes ajoutées en mémoire"""
    return {"stored_maps": STORED_MAPS}
