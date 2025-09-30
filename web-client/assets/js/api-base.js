(function configureApiBase() {
  const normalize = (value) => value.replace(/\/+$/, "");

  if (window.BATMAP_API_BASE && window.BATMAP_API_BASE.trim()) {
    window.BATMAP_API_BASE = normalize(window.BATMAP_API_BASE.trim());
    return;
  }

  const isHttp = /^https?:$/i.test(window.location.protocol);
  if (isHttp) {
    window.BATMAP_API_BASE = normalize(`${window.location.origin}/api`);
  } else {
    window.BATMAP_API_BASE = normalize("http://localhost:8000");
  }
})();
