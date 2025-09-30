"use strict";

const API_BASE = "http://localhost:8000";
/**
 * Récupère les villes et peuple le <select>.
 */
async function loadCities(cityDropdown) {
  try {
    const res = await fetch(`${API_BASE}/map/cities`);
    const data = await res.json();
    if (!Array.isArray(data?.cities)) return;

    data.cities.forEach((city) => {
      const normalizedCity = String(city).trim();
      const displayCity = normalizedCity.toUpperCase();
      const option = document.createElement("option");
      option.value = normalizedCity;
      option.textContent = displayCity;
      cityDropdown.appendChild(option);
    });
  } catch (err) {
    console.error("Erreur lors du chargement des villes:", err);
  }
}

/**
 * Charge les cartes pour une ville et les affiche dans la galerie.
 */
async function loadMapsForCity(city, mapGallery) {
  mapGallery.innerHTML = ""; // reset
  if (!city) return;

  try {
    const res = await fetch(
      `${API_BASE}/map/list?city=${encodeURIComponent(city)}`
    );
    const data = await res.json();

    if (!Array.isArray(data?.results)) return;

    data.results.forEach((map) => {
      if (!map?.map) return;
      const card = createMapCard(map);
      mapGallery.appendChild(card);
    });
  } catch (err) {
    console.error("Erreur lors du chargement des cartes:", err);
  }
}

/**
 * Construit la carte cliquable menant à la page détail.
 */
function createMapCard(map) {
  const card = document.createElement("div");
  card.classList.add("map-card");
  card.innerHTML = `<img src="data:image/png;base64,${map.map}" alt="Carte ${map.id}" />`;
  card.addEventListener("click", () => {
    window.location.href = `map_detail.html?id=${map.id}`;
  });
  return card;
}

function initParallaxHover() {
  const elements = document.querySelectorAll(
    ".header h1, .header svg, .map-card, #city-dropdown"
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

      // Clamp des valeurs pour rester dans une zone confortable
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
  const cityDropdown = document.getElementById("city-dropdown");
  const mapGallery = document.getElementById("map-gallery");
  if (!cityDropdown || !mapGallery) return;

  // Villes
  loadCities(cityDropdown);

  // Quand une ville est sélectionnée
  cityDropdown.addEventListener("change", () => {
    const city = cityDropdown.value;
    loadMapsForCity(city, mapGallery);
  });

  // Parallax
  initParallaxHover();
});
