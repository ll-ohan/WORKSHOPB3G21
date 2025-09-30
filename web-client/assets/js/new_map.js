"use strict";

const API_BASE = "http://localhost:8000";

let uploadedFile = null;
let chosenBinPath = null;
let originalMapPath = null;
let currentCity = "";
let currentScale = 0;

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
 * Récupère la liste des villes et alimente le <datalist>.
 */
async function loadCities() {
  try {
    const res = await fetch(`${API_BASE}/map/cities`);
    const data = await res.json();
    const datalist = document.getElementById("cities-list");
    if (!datalist) return;

    if (data.cities) {
      data.cities.forEach((city) => {
        const option = document.createElement("option");
        option.value = city;
        datalist.appendChild(option);
      });
    }
  } catch (err) {
    console.error("Erreur chargement villes:", err);
  }
}

/**
 * Soumet l’upload au backend, affiche les options binaires et masque le formulaire.
 */
async function handleUpload(e) {
  e.preventDefault();

  const fileInput = document.getElementById("map-file");
  const cityInput = document.getElementById("city");
  const scaleInput = document.getElementById("scale"); // conservé même si non lu directement

  if (!fileInput?.files?.[0]) {
    showErrorMessage("Veuillez sélectionner une carte.");
    return;
  }

  const file = fileInput.files[0];
  const validTypes = ["image/png", "image/jpeg", "image/tiff"];
  if (!validTypes.includes(file.type)) {
    showErrorMessage("Format invalide. Formats acceptés: PNG, JPEG, TIFF.");
    return;
  }

  currentCity = cityInput.value.trim();
  if (!currentCity) {
    showErrorMessage("Veuillez entrer une ville.");
    return;
  }

  // Conversion échelle via les champs gauche/droite (conservée)
  currentScale = computeScale();
  if (!currentScale) {
    showErrorMessage("Veuillez entrer une échelle valide.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("scale", currentScale);
  formData.append("city", currentCity);

  try {
    const res = await fetch(`${API_BASE}/map/binaryse`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (res.ok) {
      uploadedFile = data.filename;
      originalMapPath = `map_store/temp/original_${data.filename}`;
      displayBinOptions(data.results);

      const form = document.getElementById("new-map-form");
      if (form) form.classList.add("hidden");
    } else {
      showErrorMessage(`Erreur API: ${data.detail || "inconnue"}`);
    }
  } catch (err) {
    console.error("Erreur upload:", err);
    showErrorMessage("Erreur lors de l'upload de la carte.");
  }
}

/**
 * Affiche les vignettes binaires renvoyées et active la sélection d’une option.
 * le bouton "finalize" est (dé)bloqué.
 */
function displayBinOptions(results) {
  const binContainer = document.getElementById("bin-options");
  if (!binContainer) return;

  binContainer.innerHTML = "";

  Object.entries(results).forEach(([path, base64]) => {
    const card = document.createElement("div");
    card.classList.add("map-card");
    card.innerHTML = `<img src="data:image/png;base64,${base64}" alt="Option binaire" class="map-bin" />`;

    card.addEventListener("click", () => {
      document
        .querySelectorAll(".map-card")
        .forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      chosenBinPath = path;

      const finalizeBtn = document.getElementById("btn-finalize");
      if (finalizeBtn) finalizeBtn.disabled = false;
    });

    binContainer.appendChild(card);
  });

  const previewZone = document.getElementById("preview");
  if (previewZone) previewZone.style.display = "flex";
}

/**
 * Envoie la sélection finale au backend et redirige vers la page détail.
 */
async function finalizeMap() {
  if (!chosenBinPath) {
    showErrorMessage("Veuillez choisir une version binaire.");
    return;
  }

  const formData = new FormData();
  formData.append("map", originalMapPath);
  formData.append("map_choosen", chosenBinPath);
  formData.append("city", currentCity);
  formData.append("scale", currentScale);

  try {
    const res = await fetch(`${API_BASE}/map/add`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (res.ok) {
      showSuccessMessage("Carte ajoutée avec succès !");
      setTimeout(() => {
        window.location.href = `map_detail.html?id=${data.id}`;
      }, 600);
    } else {
      showErrorMessage(`Erreur API: ${data.detail || "inconnue"}`);
    }
  } catch (err) {
    console.error("Erreur finalisation:", err);
    showErrorMessage("Erreur lors de l'enregistrement de la carte.");
  }
}

/**
 * Convertit une valeur vers des mètres selon l’unité.
 */
function convertToMeters(value, unit) {
  switch (unit) {
    case "mm":
      return value / 1000;
    case "cm":
      return value / 100;
    case "m":
      return value;
    case "km":
      return value * 1000;
    case "inch":
    case "pouce":
      return value * 0.0254;
    case "mile":
      return value * 1609.34;
    case "nm":
      return value * 1852;
    case "px":
      return value * (0.0254 / 300);
    default:
      return value;
  }
}

/**
 * Calcule le ratio d’échelle "monde / papier"
 */
function computeScale() {
  const leftEl = document.getElementById("scale-left");
  const leftUnitEl = document.getElementById("scale-left-unit");
  const rightEl = document.getElementById("scale-right");
  const rightUnitEl = document.getElementById("scale-right-unit");
  const preview = document.getElementById("scale-preview");

  if (!leftEl || !leftUnitEl || !rightEl || !rightUnitEl) return null;

  const leftValue = parseFloat(leftEl.value);
  const leftUnit = leftUnitEl.value;
  const rightValue = parseFloat(rightEl.value);
  const rightUnit = rightUnitEl.value;

  if (isNaN(leftValue) || isNaN(rightValue)) return null;

  const leftMeters = convertToMeters(leftValue, leftUnit);
  const rightMeters = convertToMeters(rightValue, rightUnit);
  if (leftMeters <= 0) return null;

  const ratio = rightMeters / leftMeters;

  // Aperçu "1:XXXX" conservé
  if (preview) {
    preview.textContent = `1:${Math.round(ratio)}`;
  }

  return ratio;
}

/**
 * clic sur le cadre -> ouvre le sélecteur; aperçu image dès sélection.
 */
function initFileUploadPreview() {
  const fileUpload = document.getElementById("file-upload");
  const fileInput = document.getElementById("map-file");
  const fileText = document.getElementById("file-text");
  const filePreview = document.getElementById("file-preview");

  if (!fileUpload || !fileInput || !fileText || !filePreview) return;

  fileUpload.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        filePreview.src = e.target.result;
        filePreview.style.display = "block";
        fileText.style.display = "none";
      };
      reader.readAsDataURL(file);
    } else {
      filePreview.style.display = "none";
      fileText.style.display = "block";
      fileText.textContent = "Appuyer pour ajouter une carte";
    }
  });
}

/**
 * Placeholder factice.
 */
function initCityPlaceholder() {
  const placeholder = document.querySelector(".fake-placeholder");
  const input = document.querySelector("#city");
  if (!placeholder || !input) return;

  input.addEventListener("input", () => {
    placeholder.style.opacity = input.value ? "0" : "1";
  });
  input.addEventListener("focus", () => {
    placeholder.classList.add("focused");
  });
  input.addEventListener("blur", () => {
    placeholder.classList.remove("focused");
  });
}

function initParallaxHover() {
  const elements = document.querySelectorAll(
    ".header h1, .header svg, .map-card, .fake-placeholder, #file-text, .map-bin, #file-preview, #scale-left, #scale-right, #scale-left-unit, #scale-right-unit"
  );
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

      // Clamp du déplacement
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
  // Données initiales
  loadCities();

  // Formulaire upload
  const form = document.getElementById("new-map-form");
  if (form) form.addEventListener("submit", handleUpload);

  // Finalisation
  const finalizeBtn = document.getElementById("btn-finalize");
  if (finalizeBtn) finalizeBtn.addEventListener("click", finalizeMap);

  // Mise à jour auto de l’échelle (input + change sur les unités)
  ["scale-left", "scale-right"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", computeScale);
  });
  ["scale-left-unit", "scale-right-unit"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", computeScale);
  });

  computeScale();

  initFileUploadPreview();
  initCityPlaceholder();
  initParallaxHover();
});
