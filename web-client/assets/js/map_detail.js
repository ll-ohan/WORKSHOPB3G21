"use strict";

const API_BASE = window.BATMAP_API_BASE;
const START_MARKER_COLOR = "#8EDCE1";
const END_MARKER_COLOR = "#ff1745";
const MARKER_RADIUS = 10;
const MARKER_OUTLINE_WIDTH = 3;

let mapId = null;
let originalImage = null;
let currentImage = null;
let startPoint = null;
let endPoint = null;
let mapFlipCardElement = null;
let toggleItineraryButton = null;
let resultsSection = null;
let isShowingItinerary = false;
let pendingLayoutFrame = null;

let obstacleMode = false;
let obstacleStart = null;
let obstacleEnd = null;
let obstacleToggle = true;

function showInlineMessage(text, variant = "error") {
  let container = document.getElementById("message-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "message-container";
    container.setAttribute("aria-live", "polite");
    document.body.appendChild(container);
  }

  const message = document.createElement("div");
  message.className = "flash-message";
  message.dataset.variant = variant;
  message.textContent = text;
  container.appendChild(message);

  const removeMessage = () => {
    message.remove();
    if (!container.childElementCount) {
      container.remove();
    }
  };

  message.addEventListener("animationend", removeMessage, { once: true });
}

function showErrorMessage(text) {
  showInlineMessage(text, "error");
}

function showSuccessMessage(text) {
  showInlineMessage(text, "success");
}

/**
 * Aligne l’état visuel de la carte/itinéraire selon le bouton de bascule.
 */
function applyFlipState() {
  if (!mapFlipCardElement) return;
  if (isShowingItinerary) {
    mapFlipCardElement.classList.add("is-flipped");
  } else {
    mapFlipCardElement.classList.remove("is-flipped");
  }
}

/**
 * Planifie une synchronisation de layout à la prochaine frame.
 */
function scheduleItineraryLayoutSync() {
  if (pendingLayoutFrame) {
    cancelAnimationFrame(pendingLayoutFrame);
  }
  pendingLayoutFrame = requestAnimationFrame(() => {
    pendingLayoutFrame = null;
    syncItineraryLayout();
  });
}

/**
 * Calque dynamiquement la hauteur du flip-card et du panneau résultats
 */
function syncItineraryLayout() {
  const canvas = document.getElementById("map-canvas");
  const flipCard = document.getElementById("map-flip-card");
  const frontFace = document.querySelector(".map-face.map-front");
  const backFace = document.querySelector(".map-face.map-back");
  const results = document.getElementById("results");

  if (!canvas || !flipCard || !frontFace || !backFace || !results) return;

  const { height } = canvas.getBoundingClientRect();
  if (!height) return;

  const targetHeight = `${height}px`;
  flipCard.style.height = targetHeight;
  frontFace.style.height = targetHeight;
  backFace.style.height = targetHeight;
  results.style.height = targetHeight;
}

/**
 * Met à jour le libellé du bouton (carte ↔ itinéraire)
 */
function syncItineraryToggleLabel() {
  if (!toggleItineraryButton) return;
  toggleItineraryButton.textContent = isShowingItinerary
    ? "AFFICHER LA CARTE"
    : "AFFICHER L’ITINÉRAIRE";
}

/**
 * Masque le panneau d’itinéraire et réinitialise ses champs d’affichage.
 * Désactive le bouton de bascule
 */
function hideItineraryResults() {
  if (resultsSection && !resultsSection.classList.contains("hidden")) {
    resultsSection.classList.add("hidden");
  }
  const distanceElem = document.getElementById("distance");
  if (distanceElem) distanceElem.textContent = "";
  const timeElem = document.getElementById("time");
  if (timeElem) timeElem.textContent = "";
  const stepsDiv = document.getElementById("itinerary-steps");
  if (stepsDiv) stepsDiv.innerHTML = "";
  if (toggleItineraryButton) {
    toggleItineraryButton.disabled = true;
    toggleItineraryButton.classList.add("hidden");
  }
  isShowingItinerary = false;
  applyFlipState();
  syncItineraryToggleLabel();
}

/**
 * Affiche le panneau d’itinéraire et réactive la bascule.
 */
function showItineraryResults() {
  if (resultsSection) {
    resultsSection.classList.remove("hidden");
  }
  if (toggleItineraryButton) {
    toggleItineraryButton.disabled = false;
    toggleItineraryButton.classList.remove("hidden");
  }
  applyFlipState();
  syncItineraryToggleLabel();
}

/*Chargement de carte */

/**
 * Récupère la carte et ses métadonnées, instancie l’image, puis déclenche le rendu sur canvas.
 */
async function loadMap(id) {
  hideItineraryResults();
  try {
    const res = await fetch(`${API_BASE}/map/${id}`);
    const data = await res.json();

    const cityElement = document.getElementById("map-city");
    const normalizedCity =
      typeof data.city === "string" ? data.city.trim() : "";
    const displayCity = normalizedCity.toUpperCase();
    if (cityElement) {
      cityElement.textContent = displayCity;
      cityElement.dataset.originalCity = normalizedCity;
    }

    // Affichage d’échelle: conserve le formatage d’origine
    const scaleEl = document.getElementById("map-scale");
    if (scaleEl) {
      scaleEl.textContent = `1 cm / ${data.scale / 100} m`;
    }

    originalImage = new Image();
    originalImage.src = `data:image/png;base64,${data.map}`;
    originalImage.onload = () => {
      currentImage = originalImage;
      drawCanvas();
    };
  } catch (err) {
    console.error("Erreur chargement carte:", err);
  }
}

/**
 * Convertit un clic utilisateur (coordonnées viewport) en coordonnées pixel.
 */
function handleCanvasClick(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  const x = Math.round((event.clientX - rect.left) * scaleX);
  const y = Math.round((event.clientY - rect.top) * scaleY);

  if (obstacleMode) {
    // Saisie en 2 temps: start → end; si déjà 2 points, repartir sur un nouveau start
    if (!obstacleStart) {
      obstacleStart = [x, y];
      console.log("➡️ Start défini :", obstacleStart);
    } else if (!obstacleEnd) {
      obstacleEnd = [x, y];
      console.log("➡️ End défini :", obstacleEnd);
    } else {
      obstacleStart = [x, y];
      obstacleEnd = null;
      console.log("🔄 Nouveau start défini :", obstacleStart);
    }
    drawCanvas();
    return;
  }

  // Mode itinéraire: trois clics ou plus => remplace simplement l’arrivée
  if (!startPoint) {
    startPoint = [x, y];
  } else if (!endPoint) {
    endPoint = [x, y];
  } else {
    endPoint = [x, y];
  }

  drawCanvas();
  const calcBtn = document.getElementById("btn-calc");
  if (calcBtn) calcBtn.disabled = !(startPoint && endPoint);
}

/**
 * Redessine le canva
 */
function drawCanvas() {
  const canvas = document.getElementById("map-canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = currentImage.width;
  canvas.height = currentImage.height;
  ctx.drawImage(currentImage, 0, 0);

  // Points de l’itinéraire
  if (startPoint)
    drawMarker(ctx, startPoint[0], startPoint[1], START_MARKER_COLOR);
  if (endPoint) drawMarker(ctx, endPoint[0], endPoint[1], END_MARKER_COLOR);

  // Obstacles (points + trait si les deux existent)
  if (obstacleStart)
    drawSquare(ctx, obstacleStart[0], obstacleStart[1], "black");
  if (obstacleEnd) drawSquare(ctx, obstacleEnd[0], obstacleEnd[1], "black");
  if (obstacleStart && obstacleEnd) {
    drawObstacleLine(ctx, obstacleStart, obstacleEnd);
  }

  scheduleItineraryLayoutSync();
}

/**
 * Marqueur rond arrivé /départ.
 */
function drawMarker(ctx, x, y, color) {
  ctx.beginPath();
  ctx.arc(x, y, MARKER_RADIUS, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "white";
  ctx.lineWidth = MARKER_OUTLINE_WIDTH;
  ctx.stroke();
}

/**
 * Petit carré pour les obstacle.
 */
function drawSquare(ctx, x, y, color) {
  ctx.fillStyle = color;
  ctx.fillRect(x - 5, y - 5, 10, 10);
}

/**
 * Trait entre les deux points d’obstacle.
 */
function drawObstacleLine(ctx, start, end) {
  ctx.beginPath();
  ctx.moveTo(start[0], start[1]);
  ctx.lineTo(end[0], end[1]);
  ctx.strokeStyle = "black";
  ctx.lineWidth = 3;
  ctx.stroke();
}

/**
 * Envoie les points départ/arrivée à l’API, remplace l’image par celle de l’itinéraire
 * et met à jour les informations (distance, temps, étapes).
 */
async function sendItinerary() {
  if (!startPoint || !endPoint) {
    showErrorMessage("Veuillez définir un départ et une arrivée.");
    return;
  }

  try {
    const formData = new FormData();
    formData.append("map_id", mapId);
    formData.append("start", `${startPoint[0]},${startPoint[1]}`);
    formData.append("end", `${endPoint[0]},${endPoint[1]}`);

    const res = await fetch(`${API_BASE}/itinerary/route`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (res.ok) {
      const routeImage = new Image();
      routeImage.src = `data:image/png;base64,${data.image}`;
      routeImage.onload = () => {
        currentImage = routeImage;
        drawCanvas();
      };

      // Affichage dynamique m / km
      const distanceElem = document.getElementById("distance");
      if (distanceElem) {
        if (data.distance_m >= 1000) {
          distanceElem.textContent =
            (data.distance_m / 1000).toFixed(2) + " km";
        } else {
          distanceElem.textContent = data.distance_m + " m";
        }
      }
      const timeElem = document.getElementById("time");
      if (timeElem) {
        timeElem.textContent = data.estimated_time;
      }

      // Génération de la liste d’étapes
      const stepsDiv = document.getElementById("itinerary-steps");
      let steps = [];
      if (typeof data.itinerary === "string") {
        steps = data.itinerary.split(". ").filter((line) => line.trim() !== "");
      }
      stepsDiv.innerHTML = steps.length
        ? "<ol>" + steps.map((s) => `<li>${s}</li>`).join("") + "</ol>"
        : "<p>Aucune instruction disponible.</p>";

      showItineraryResults();
    } else {
      showErrorMessage(`Erreur API: ${data.detail || "inconnue"}`);
    }
  } catch (err) {
    console.error("Erreur envoi itinéraire:", err);
    showErrorMessage("Impossible de calculer l'itinéraire.");
  }
}

function resetPoints() {
  startPoint = null;
  endPoint = null;
  currentImage = originalImage;
  const calcBtn = document.getElementById("btn-calc");
  if (calcBtn) calcBtn.disabled = true;
  drawCanvas();
  hideItineraryResults();
}

function swapPoints() {
  if (!startPoint || !endPoint) {
    showErrorMessage("Il faut d'abord définir un départ et une arrivée.");
    return;
  }
  [startPoint, endPoint] = [endPoint, startPoint];
  drawCanvas();
}

function enableObstacleMode() {
  obstacleMode = true;
  obstacleStart = null;
  obstacleEnd = null;
  console.log("Obstacle mode activé");

  const addBtn = document.getElementById("btn-add-obstacle");
  if (addBtn) addBtn.style.display = "none";

  const obstacleActions = document.getElementById("obstacle-actions");
  if (obstacleActions) obstacleActions.style.display = "grid";
}

async function validateObstacle() {
  if (!obstacleStart || !obstacleEnd) {
    showErrorMessage("Veuillez définir 2 points pour l’obstacle.");
    return;
  }

  try {
    const formData = new FormData();
    formData.append("map_id", mapId);
    formData.append("start", `${obstacleStart[0]},${obstacleStart[1]}`);
    formData.append("end", `${obstacleEnd[0]},${obstacleEnd[1]}`);

    const res = await fetch(`${API_BASE}/map/modify_bin`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (res.ok) {
      showSuccessMessage(data.message);
      cancelObstacle();
      await loadMap(mapId); // Rechargement après modification serveur
    } else {
      showErrorMessage(`Erreur API: ${data.detail || "inconnue"}`);
    }
  } catch (err) {
    console.error("Erreur validation obstacle:", err);
    showErrorMessage("Impossible de modifier la carte.");
  }
}

function cancelObstacle() {
  obstacleMode = false;
  obstacleStart = null;
  obstacleEnd = null;
  console.log("Obstacle annulé");

  const addBtn = document.getElementById("btn-add-obstacle");
  if (addBtn) addBtn.style.display = "";

  const obstacleActions = document.getElementById("obstacle-actions");
  if (obstacleActions) obstacleActions.style.display = "none";

  drawCanvas();
}

async function resetObstacles() {
  try {
    const res = await fetch(`${API_BASE}/map/${mapId}/bin/reset`, {
      method: "DELETE",
    });
    const data = await res.json();
    if (res.ok) {
      showSuccessMessage(data.message);
      obstacleToggle = true; // inchangé
      await loadMap(mapId);
    } else {
      showErrorMessage(`Erreur API: ${data.detail || "inconnue"}`);
    }
  } catch (err) {
    console.error("Erreur reset obstacles:", err);
    showErrorMessage("Impossible de réinitialiser les obstacles.");
  }
}
/**
 * Léger déplacement en fonction de la souris pour donner de la profondeur.
 */
function initParallaxHover() {
  const elements = document.querySelectorAll(".header h1, .header svg, .btn");
  if (!elements.length) return;

  const intensity = 10;
  const maxMove = intensity * 0.8;

  document.addEventListener("mousemove", (e) => {
    elements.forEach((el) => {
      const rect = el.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      const x = (e.clientX - centerX) / rect.width;
      const y = (e.clientY - centerY) / rect.height;

      let moveX = x * intensity;
      let moveY = y * intensity;

      moveX = Math.max(-maxMove, Math.min(maxMove, moveX));
      moveY = Math.max(-maxMove, Math.min(maxMove, moveY));

      el.style.transform = `translate(${moveX}px, ${moveY}px)`;
      el.style.transition = "all 0.1s ease-out";
    });
  });

  document.addEventListener("mouseleave", () => {
    elements.forEach((el) => {
      el.style.transform = "translate(0, 0)";
      el.style.transition = "all 0.5s ease-out";
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  // Récupération ID carte depuis l’URL
  const urlParams = new URLSearchParams(window.location.search);
  mapId = urlParams.get("id");
  if (!mapId) {
    showErrorMessage("Aucune carte spécifiée");
    return;
  }

  // Raccourcis d’éléments UI
  mapFlipCardElement = document.getElementById("map-flip-card");
  toggleItineraryButton = document.getElementById("btn-toggle-itinerary");
  resultsSection = document.getElementById("results");

  // Bascule carte/itinéraire
  if (toggleItineraryButton) {
    toggleItineraryButton.addEventListener("click", () => {
      if (toggleItineraryButton.disabled) return;
      isShowingItinerary = !isShowingItinerary;
      applyFlipState();
      syncItineraryToggleLabel();
    });
  }

  hideItineraryResults();

  // Canvas + interaction
  const canvas = document.getElementById("map-canvas");
  if (canvas) {
    canvas.addEventListener("click", (e) => handleCanvasClick(e, canvas));
  }

  // Chargement initial de la carte
  loadMap(mapId);

  // Boutons principaux
  const btnCalc = document.getElementById("btn-calc");
  const btnReset = document.getElementById("btn-reset");
  const btnSwap = document.getElementById("btn-swap");
  if (btnCalc) btnCalc.addEventListener("click", sendItinerary);
  if (btnReset) btnReset.addEventListener("click", resetPoints);
  if (btnSwap) btnSwap.addEventListener("click", swapPoints);

  // Obstacles
  const btnAddObst = document.getElementById("btn-add-obstacle");
  const btnValidObst = document.getElementById("btn-validate-obstacle");
  const btnCancelObst = document.getElementById("btn-cancel-obstacle");
  const btnResetObst = document.getElementById("btn-reset-obstacles");
  if (btnAddObst) btnAddObst.addEventListener("click", enableObstacleMode);
  if (btnValidObst) btnValidObst.addEventListener("click", validateObstacle);
  if (btnCancelObst) btnCancelObst.addEventListener("click", cancelObstacle);
  if (btnResetObst) btnResetObst.addEventListener("click", resetObstacles);

  // Sync layout au resize
  window.addEventListener("resize", scheduleItineraryLayoutSync);

  initParallaxHover();
});
