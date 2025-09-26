import cv2
import numpy as np
from skimage.filters import threshold_otsu
from pathlib import Path


class RoadExtractor:
    def __init__(self, image_source):
        self.image_source = image_source
        self.original = None
        self.gray = None

        self.params = {
            'denoise_h': 10,
            'morph_kernel_size': 0,
            'erosion_iterations': 0,
            'dilation_iterations': 0,
            'adaptive_block_size': 15,
            'adaptive_c': 0,
            'manual_threshold': 0
        }

    def load_image(self):
        if isinstance(self.image_source, (str, Path)):
            path = Path(self.image_source)
            if not path.exists():
                raise FileNotFoundError(f"L'image {path} n'existe pas")
            self.original = cv2.imread(str(path))
        elif isinstance(self.image_source, (bytes, bytearray)):
            data = np.frombuffer(self.image_source, np.uint8) #wrap du fichier en bytes
            self.original = cv2.imdecode(data, cv2.IMREAD_COLOR) #decode les bytes en image
        else:
            raise TypeError("image_source doit être un chemin (str/Path) ou des bytes")

        if self.original is None:
            raise ValueError("Impossible de charger l'image (format non supporté ou corrompu)")

        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY) #charle l'image 
        print(f"✓ Image chargée: {self.original.shape}")

    def denoise_image(self, image, h=10):
        return cv2.fastNlMeansDenoising(image, None, h, 7, 21) #retire le bruit de l'image

    def morphology_operations(self, image, kernel_size=0, erosion_iter=0, dilation_iter=0):
        if kernel_size <= 0 or (erosion_iter == 0 and dilation_iter == 0): #bypass si pas d'opérations
            return image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)) #kernel elliptique
        eroded = cv2.erode(image, kernel, iterations=erosion_iter) if erosion_iter > 0 else image #érosion
        dilated = cv2.dilate(eroded, kernel, iterations=dilation_iter) if dilation_iter > 0 else eroded #dilatation
        return cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel) #ouverture et double nettoyage

    def binarize_image(self, image, method=0, block_size=15, c=0, manual_thresh=0):
        if method == 0:
            thresh_val = threshold_otsu(image) #seuil d'Otsu
            _, binary = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY) #binarisation
        elif method == 1:
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, block_size, c) #binarisation adaptative par moyenne
        elif method == 2:
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, c) #binarisation adaptative gaussienne
        else:  
            _, binary = cv2.threshold(image, manual_thresh, 255, cv2.THRESH_BINARY) #binarisation manuelle

        if np.mean(binary) > 127:  
            binary = cv2.bitwise_not(binary) #inversion si fond clair
        return binary

    def process(self, method): #manage le traitement d'image
        #méthode 0 = Otsu, 1 = adaptative moyenne, 2 = adaptative gaussienne, 3 = manuelle
        denoised = self.denoise_image(self.gray, self.params['denoise_h'])
        morphed = self.morphology_operations(
            denoised,
            self.params['morph_kernel_size'],
            self.params['erosion_iterations'],
            self.params['dilation_iterations']
        )
        return self.binarize_image(
            morphed,
            method,
            self.params['adaptive_block_size'],
            self.params['adaptive_c'],
            self.params['manual_threshold']
        )

    def run(self): #orchestration generale
        self.load_image()
        results = {}
        for method in [0, 1]: 
            results[f"method_{method}"] = self.process(method)
        print("\n✓ Traitement terminé avec succès!")
        return results
