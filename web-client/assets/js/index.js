"use strict";

document.addEventListener("DOMContentLoaded", () => {
  initNavigationButtons();
  initSplashScreen();
  initParallaxHover();
});

/**
 * Initialise les boutons de navigation d’accueil.
 */
function initNavigationButtons() {
  const btnNewMap = document.getElementById("btn-new-map");
  const btnSelectMap = document.getElementById("btn-select-map");

  if (btnNewMap) {
    btnNewMap.addEventListener("click", () => {
      window.location.href = "new_map.html";
    });
  }

  if (btnSelectMap) {
    btnSelectMap.addEventListener("click", () => {
      window.location.href = "select_map.html";
    });
  }
}

/**
 * Gère la disparition du splash-screen après le chargement complet.
 */
function initSplashScreen() {
  const splash = document.getElementById("splash-screen");
  if (!splash) return; // garde-fou si l'élément est absent sur certaines pages

  window.addEventListener("load", () => {
    setTimeout(() => {
      splash.classList.add("hidden");
    }, 4500); //Temps minimum d'affichage
  });
}

/**
 * Applique un léger déplacement (type "parallax" 2D) en fonction de la position de la souris
 *
 * - La position de la souris est normalisée par la taille de chaque élément (x,y ≈ [-0.5 ; +0.5]).
 * - Une intensité fixe est utilisée, puis "clampée" avec une borne à 80% pour éviter les sauts visuels.
 * - La logique reste volontairement simple et synchrone (pas de throttle/debounce pour ne pas changer le comportement).
 */
function initParallaxHover() {
  const elements = document.querySelectorAll(
    ".header h1, .header svg, .actions .btn"
  );
  if (!elements.length) return;

  const intensity = 10; // amplitude de déplacement « nominale »
  const maxMove = intensity * 0.8; // borne 80% pour éviter des translations trop visibles

  document.addEventListener("mousemove", (e) => {
    elements.forEach((el) => {
      const rect = el.getBoundingClientRect();

      // Centre géométrique de l’élément
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      // Normalisation de la position curseur relativement à l’élément
      const x = (e.clientX - centerX) / rect.width;
      const y = (e.clientY - centerY) / rect.height;

      // Déplacement brut proportionnel à l’intensité
      let moveX = x * intensity;
      let moveY = y * intensity;

      // Clamp pour contenir le déplacement dans une zone confortable
      moveX = Math.max(-maxMove, Math.min(maxMove, moveX));
      moveY = Math.max(-maxMove, Math.min(maxMove, moveY));

      el.style.transform = `translate(${moveX}px, ${moveY}px)`;
      el.style.transition = "all 0.1s ease-out";
    });
  });

  // Retour progressif à l’état neutre lorsque la souris quitte le document
  document.addEventListener("mouseleave", () => {
    elements.forEach((el) => {
      el.style.transform = "translate(0, 0)";
      el.style.transition = "all 0.5s ease-out";
    });
  });
}
