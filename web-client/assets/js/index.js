document.addEventListener("DOMContentLoaded", () => {
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
});
