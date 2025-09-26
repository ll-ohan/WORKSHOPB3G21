/* ===== BatMap Front (2-form wizard) ===== */
const API_BASE = "http://localhost:8000";
const qs = (sel, el=document)=>el.querySelector(sel);
const qsa = (sel, el=document)=>[...el.querySelectorAll(sel)];
const toastEl = qs('#toast');

/* ---------- Router (onglets + ?mapId=) ---------- */
window.addEventListener('DOMContentLoaded', () => {
  bindTabs();
  initUseView();
  initNewView();
  initDetailView();

  const url = new URL(window.location.href);
  const mapId = url.searchParams.get('mapId');
  if (mapId) {
    showView('detail');
    loadMapDetail(parseInt(mapId, 10));
  } else {
    showView('use');
  }
});

function bindTabs(){
  qsa('.tab-btn').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const tab = btn.dataset.tab;
      if (tab === 'use') {
        const url = new URL(window.location.href);
        url.searchParams.delete('mapId');
        history.pushState({}, '', url);
      }
      showView(tab);
    });
  });

  window.addEventListener('popstate', ()=>{
    const url = new URL(window.location.href);
    const mapId = url.searchParams.get('mapId');
    if (mapId) {
      showView('detail'); loadMapDetail(parseInt(mapId,10));
    } else {
      showView('use');
    }
  });
}

function showView(key){
  qsa('.tab-btn').forEach(b=>b.classList.toggle('active', b.dataset.tab===key));
  ['use','new','detail'].forEach(k=>qs(`#view-${k}`).classList.add('hidden'));
  qs(`#view-${key}`).classList.remove('hidden');
}

/* ---------- Utiliser une carte ---------- */
const citySelect = qs('#citySelect');
const refreshCityBtn = qs('#refreshCity');
const mapsGrid = qs('#mapsGrid');
const emptyStateUse = qs('#emptyStateUse');

function initUseView(){
  refreshCityBtn.addEventListener('click', loadCities);
  citySelect.addEventListener('change', ()=>{
    const city = citySelect.value;
    if (city) {
      localStorage.setItem('batmap:lastCity', city);
      loadMapsForCity(city);
    } else {
      mapsGrid.innerHTML = '';
      emptyStateUse.classList.remove('hidden');
    }
  });

  loadCities().then(()=>{
    const last = localStorage.getItem('batmap:lastCity');
    if (last && [...citySelect.options].some(o=>o.value===last)) {
      citySelect.value = last;
      loadMapsForCity(last);
    } else if (citySelect.options.length>0) {
      citySelect.selectedIndex = 0;
      citySelect.dispatchEvent(new Event('change'));
    }
  });
}

async function loadCities(){
  citySelect.innerHTML = '';
  try{
    const data = await GET('/map/cities');
    const cities = data.cities || [];
    if (cities.length===0){
      emptyStateUse.classList.remove('hidden');
      mapsGrid.innerHTML='';
      return;
    }
    emptyStateUse.classList.add('hidden');
    for (const c of cities){
      const opt = document.createElement('option');
      opt.value = c; opt.textContent = c;
      citySelect.appendChild(opt);
    }
    return cities;
  }catch(e){
    notify("Impossible de charger les villes", true);
  }
}

async function loadMapsForCity(city){
  mapsGrid.innerHTML = '';
  try{
    const data = await GET(`/map/list?city=${encodeURIComponent(city)}`);
    const results = data.results || [];
    if (results.length===0){
      emptyStateUse.classList.remove('hidden'); return;
    }
    emptyStateUse.classList.add('hidden');

    for (const r of results){
      const card = document.createElement('div');
      card.className = 'card';
      card.tabIndex = 0;
      card.addEventListener('click', ()=>openMapDetail(r.id));
      card.addEventListener('keyup', (ev)=>{ if(ev.key==='Enter'){ openMapDetail(r.id); } });

      const img = document.createElement('img');
      img.alt = `${r.city} – #${r.id}`;
      img.src = `data:image/png;base64,${r.map}`;

      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.innerHTML = `
        <div><strong>${r.city}</strong> – #${r.id}</div>
        <div>Échelle: ${r.scale}</div>
        <div><small>${new Date(r.created_at.replace(' ','T')).toLocaleString()}</small></div>
      `;

      card.appendChild(img);
      card.appendChild(meta);
      mapsGrid.appendChild(card);
    }
  }catch(e){
    notify("Erreur lors du chargement des cartes", true);
  }
}

function openMapDetail(id){
  const url = new URL(window.location.href);
  url.searchParams.set('mapId', String(id));
  history.pushState({}, '', url);
  showView('detail');
  loadMapDetail(id);
}

/* ---------- Nouvelle carte (2 formulaires) ---------- */
const formUpload   = qs('#formUpload');
const formConfirm  = qs('#formConfirm');
const btnUpload    = qs('#btnUpload');
const btnConfirm   = qs('#btnConfirm');
const btnBack      = qs('#btnBack');
const binaryChoices= qs('#binaryChoices');

let lastBinaryseCtx = null; // { filename, city, scale, results: {path:b64,...} }

function initNewView(){
  // SUBMIT formulaire 1
  formUpload.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const city = qs('#cityInput').value.trim();
    const scale = parseFloat(qs('#scaleInput').value);
    const file  = qs('#fileInput').files[0];
    if (!city || !scale || !file){
      notify("Remplis tous les champs.", true); return;
    }

    try{
      btnUpload.disabled = true;
      notify("Extraction des routes en cours…");
      const fd = new FormData();
      fd.append('file', file);
      fd.append('scale', String(scale));
      fd.append('city', city);

      const res = await fetch(`${API_BASE}/map/binaryse`, { method:'POST', body:fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      lastBinaryseCtx = data;

      renderBinaryChoices(data);
      // Passe à la "page" 2 (toujours dans l’onglet Nouvelle carte)
      formUpload.classList.add('hidden');
      formConfirm.classList.remove('hidden');
      formConfirm.scrollIntoView({ behavior:'smooth', block:'start' });
      notify("Choisis la version binaire à enregistrer.");
    }catch(err){
      console.error(err);
      notify("Erreur pendant l'extraction binaire.", true);
    }finally{
      btnUpload.disabled = false;
    }
  });

  // SUBMIT formulaire 2
  formConfirm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    if (!lastBinaryseCtx){ notify("Aucun contexte d’upload.", true); return; }
    const chosen = qs('input[name="binPick"]:checked', binaryChoices);
    if (!chosen){ notify("Sélectionne une image binaire.", true); return; }

    const map_choosen = decodeURIComponent(chosen.value);
    const original_path = `map_store/temp/original_${lastBinaryseCtx.filename}`;
    const fd = new FormData();
    fd.append('map', original_path);
    fd.append('map_choosen', map_choosen);
    fd.append('city', lastBinaryseCtx.city);
    fd.append('scale', String(lastBinaryseCtx.scale));

    try{
      btnConfirm.disabled = true;
      notify("Enregistrement de la carte…");
      const res = await fetch(`${API_BASE}/map/add`, { method:'POST', body:fd });
      if (!res.ok) throw new Error(await res.text());
      const saved = await res.json();

      // reset wizard
      resetConfirm();
      formConfirm.classList.add('hidden');
      formUpload.reset();
      formUpload.classList.remove('hidden');

      // redirection vers "Utiliser une carte"
      showView('use');
      await loadCities();
      citySelect.value = saved.city;
      citySelect.dispatchEvent(new Event('change'));
      notify(`Carte #${saved.id} enregistrée pour ${saved.city} ✔️`);
    }catch(err){
      console.error(err);
      notify("Erreur lors de l’enregistrement de la carte.", true);
    }finally{
      btnConfirm.disabled = false;
    }
  });

  // Retour à l’étape 1 (sans enregistrer)
  btnBack.addEventListener('click', ()=>{
    resetConfirm();
    formConfirm.classList.add('hidden');
    formUpload.classList.remove('hidden');
  });
}

function renderBinaryChoices(data){
  binaryChoices.innerHTML = '';
  const entries = Object.entries(data.results || {});
  if (entries.length===0){
    notify("Aucune image binaire reçue.", true);
    btnConfirm.disabled = true;
    return;
  }
  for (const [path, b64] of entries){
    const div = document.createElement('div');
    div.className = 'bin';
    div.innerHTML = `
      <label class="pick">
        <input type="radio" name="binPick" value="${encodeURIComponent(path)}">
        Choisir
      </label>
      <img alt="Binaire" src="data:image/png;base64,${b64}">
      <div class="meta" style="padding:8px;color:#9aa4b2;font-size:.85rem">${path}</div>
    `;
    binaryChoices.appendChild(div);
  }
  binaryChoices.addEventListener('change', ()=>{
    btnConfirm.disabled = !qs('input[name="binPick"]:checked', binaryChoices);
  }, { once:true });
  btnConfirm.disabled = true;
}

function resetConfirm(){
  binaryChoices.innerHTML = '';
  lastBinaryseCtx = null;
  btnConfirm.disabled = true;
}

/* ---------- Détail carte ---------- */
const backToListBtn = qs('#backToList');
const detailTitle = qs('#detailTitle');
const mapImage = qs('#mapImage');
const clickLayer = qs('#clickLayer');
const routeStartEl = qs('#routeStart');
const routeEndEl = qs('#routeEnd');
const blockStartEl = qs('#blockStart');
const blockEndEl = qs('#blockEnd');
const btnRoute = qs('#btnRoute');
const btnBlock = qs('#btnBlock');
const btnReset = qs('#btnReset');
const routeMeta = qs('#routeMeta');
const resultPanel = qs('#resultPanel');
const routeImage = qs('#routeImage');
const itineraryList = qs('#itineraryList');

let currentMapId = null;
let pickMode = 'route';
let routePts = { start:null, end:null };
let blockPts = { start:null, end:null };

function initDetailView(){
  backToListBtn.addEventListener('click', ()=>{
    const url = new URL(window.location.href);
    url.searchParams.delete('mapId');
    history.pushState({}, '', url);
    showView('use');
  });

  qsa('input[name="pickMode"]').forEach(r => {
    r.addEventListener('change', ()=>{
      pickMode = r.value; drawMarkers();
    });
  });

  mapImage.addEventListener('load', syncCanvasSize);
  window.addEventListener('resize', syncCanvasSize);

  mapImage.addEventListener('click', (ev)=>{
    const pt = eventToImageCoords(ev, mapImage);
    if (!pt) return;

    if (pickMode==='route'){
      if (!routePts.start) routePts.start = pt;
      else if (!routePts.end) routePts.end = pt;
      else { routePts = { start: pt, end: null }; }
    } else {
      if (!blockPts.start) blockPts.start = pt;
      else if (!blockPts.end) blockPts.end = pt;
      else { blockPts = { start: pt, end: null }; }
    }
    updateCoordLabels();
    drawMarkers();
  });

  btnRoute.addEventListener('click', onComputeRoute);
  btnBlock.addEventListener('click', onBlockRoute);
  btnReset.addEventListener('click', onResetBin);
}

function eventToImageCoords(ev, imgEl){
  const rect = imgEl.getBoundingClientRect();
  const xDisp = ev.clientX - rect.left;
  const yDisp = ev.clientY - rect.top;
  if (xDisp<0 || yDisp<0 || xDisp>rect.width || yDisp>rect.height) return null;

  const scaleX = imgEl.naturalWidth / rect.width;
  const scaleY = imgEl.naturalHeight / rect.height;
  const x = Math.round(xDisp * scaleX);
  const y = Math.round(yDisp * scaleY);
  return { x, y };
}

function syncCanvasSize(){
  const rect = mapImage.getBoundingClientRect();
  clickLayer.width = rect.width;
  clickLayer.height = rect.height;
  clickLayer.style.width = rect.width + 'px';
  clickLayer.style.height = rect.height + 'px';
  drawMarkers();
}

function updateCoordLabels(){
  routeStartEl.textContent = routePts.start ? `${routePts.start.x},${routePts.start.y}` : '(clic…)';
  routeEndEl.textContent   = routePts.end   ? `${routePts.end.x},${routePts.end.y}`   : '(clic…)';
  blockStartEl.textContent = blockPts.start ? `${blockPts.start.x},${blockPts.start.y}` : '(clic…)';
  blockEndEl.textContent   = blockPts.end   ? `${blockPts.end.x},${blockPts.end.y}`   : '(clic…)';
}

function drawMarkers(){
  const ctx = clickLayer.getContext('2d');
  ctx.clearRect(0,0,clickLayer.width, clickLayer.height);

  const rect = mapImage.getBoundingClientRect();
  const sx = rect.width / mapImage.naturalWidth;
  const sy = rect.height / mapImage.naturalHeight;

  const drawDot = (pt, color) => {
    const x = Math.round(pt.x * sx);
    const y = Math.round(pt.y * sy);
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI*2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000000aa';
    ctx.stroke();
  };

  if (routePts.start) drawDot(routePts.start, getCSS('--dot-blue'));
  if (routePts.end)   drawDot(routePts.end,   getCSS('--dot-blue'));
  if (blockPts.start) drawDot(blockPts.start, getCSS('--dot-red'));
  if (blockPts.end)   drawDot(blockPts.end,   getCSS('--dot-red'));
}

function getCSS(varName){
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

async function loadMapDetail(id){
  currentMapId = id;
  detailTitle.textContent = `Carte #${id}`;
  routePts = { start:null, end:null };
  blockPts = { start:null, end:null };
  updateCoordLabels();
  routeMeta.classList.add('hidden');
  resultPanel.classList.add('hidden');

  try{
    notify("Chargement de la carte…");
    const data = await GET(`/map/${id}`);
    if (!data || !data.map){
      notify("Impossible de charger cette carte.", true);
      return;
    }
    mapImage.src = `data:image/png;base64,${data.map}`;
  }catch(e){
    notify("Erreur lors du chargement de la carte.", true);
  }
}

async function onComputeRoute(){
  if (!currentMapId) return;
  if (!routePts.start || !routePts.end){
    notify("Sélectionne d’abord un départ et une arrivée.", true);
    return;
  }
  const fd = new FormData();
  fd.append('map_id', String(currentMapId));
  fd.append('start', `${routePts.start.x},${routePts.start.y}`);
  fd.append('end',   `${routePts.end.x},${routePts.end.y}`);

  try{
    btnRoute.disabled = true;
    notify("Calcul du plus court chemin…");
    const res = await fetch(`${API_BASE}/itinerary/route`, { method:'POST', body:fd });
    if (!res.ok){
      let msg = 'Erreur lors du calcul d’itinéraire.';
      try{ msg += ' ' + (await res.json()).detail; }catch(_){}
      throw new Error(msg);
    }
    const data = await res.json();
    routeImage.src = `data:image/png;base64,${data.image}`;
    itineraryList.innerHTML = '';
    const raw = (data.itinerary || '').toString();
    raw.split('.').map(s=>s.trim()).filter(s=>s.length>0).forEach(step=>{
      const li = document.createElement('li'); li.textContent = step; itineraryList.appendChild(li);
    });

    routeMeta.innerHTML = `
      <div><strong>Distance :</strong> ${data.distance_m} m</div>
      <div><strong>Temps estimé :</strong> ${data.estimated_time}</div>
    `;
    routeMeta.classList.remove('hidden');
    resultPanel.classList.remove('hidden');
    notify("Itinéraire prêt ✅");
  }catch(err){
    console.error(err);
    notify(err.message || "Échec du calcul d’itinéraire.", true);
  }finally{
    btnRoute.disabled = false;
  }
}

async function onBlockRoute(){
  if (!currentMapId) return;
  if (!blockPts.start || !blockPts.end){
    notify("Sélectionne deux points à barrer.", true);
    return;
  }
  const fd = new FormData();
  fd.append('map_id', String(currentMapId));
  fd.append('start', `${blockPts.start.x},${blockPts.start.y}`);
  fd.append('end',   `${blockPts.end.x},${blockPts.end.y}`);

  try{
    btnBlock.disabled = true;
    notify("Application de la restriction sur la carte…");
    const res = await fetch(`${API_BASE}/map/modify_bin`, { method:'POST', body:fd });
    if (!res.ok) throw new Error((await res.json()).detail || 'Erreur API');
    await res.json();
    notify("Route barrée. Les prochains calculs utiliseront la carte modifiée ✅");
  }catch(err){
    console.error(err);
    notify("Impossible de barrer cette route.", true);
  }finally{
    btnBlock.disabled = false;
  }
}

async function onResetBin(){
  if (!currentMapId) return;
  try{
    btnReset.disabled = true;
    notify("Restauration de la carte originale…");
    const res = await fetch(`${API_BASE}/map/${currentMapId}/bin/reset`, { method:'DELETE' });
    if (!res.ok) throw new Error((await res.json()).detail || 'Erreur API');
    await res.json();
    notify("Carte restaurée sur le binaire original ✔️");
  }catch(err){
    console.error(err);
    notify("Échec de la restauration.", true);
  }finally{
    btnReset.disabled = false;
  }
}

/* ---------- Utils ---------- */
async function GET(path){
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

let toastTimer = null;
function notify(msg, isError=false){
  toastEl.textContent = msg;
  toastEl.classList.remove('hidden');
  toastEl.style.borderColor = isError ? '#823b32' : 'var(--border)';
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>toastEl.classList.add('hidden'), 3500);
}
