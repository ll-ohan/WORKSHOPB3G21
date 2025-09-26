/* ===== BatMap Front (2 formulaires, zéro reload & aucune redirection auto) ===== */
const API_BASE = "http://localhost:8000";

const qs  = (s, el=document)=>el.querySelector(s);
const qsa = (s, el=document)=>[...el.querySelectorAll(s)];
const toastEl = qs('#toast');

/* ---------- Tabs + routing (?mapId=) ---------- */
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
    if (mapId) { showView('detail'); loadMapDetail(parseInt(mapId,10)); }
    else { showView('use'); }
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
    if (city) { localStorage.setItem('batmap:lastCity', city); loadMapsForCity(city); }
    else { mapsGrid.innerHTML = ''; emptyStateUse.classList.remove('hidden'); }
  });

  loadCities().then(()=>{
    const last = localStorage.getItem('batmap:lastCity');
    if (last && [...citySelect.options].some(o=>o.value===last)) {
      citySelect.value = last; loadMapsForCity(last);
    } else if (citySelect.options.length>0) {
      citySelect.selectedIndex = 0; citySelect.dispatchEvent(new Event('change'));
    }
  });
}

async function loadCities(){
  citySelect.innerHTML = '';
  try{
    const data = await GET('/map/cities');
    const cities = data.cities || [];
    if (cities.length===0){ emptyStateUse.classList.remove('hidden'); mapsGrid.innerHTML=''; return; }
    emptyStateUse.classList.add('hidden');
    for (const c of cities){ const o=document.createElement('option'); o.value=c; o.textContent=c; citySelect.appendChild(o); }
    return cities;
  }catch{ notify("Impossible de charger les villes", true); }
}

async function loadMapsForCity(city){
  mapsGrid.innerHTML = '';
  try{
    const data = await GET(`/map/list?city=${encodeURIComponent(city)}`);
    const results = data.results || [];
    if (results.length===0){ emptyStateUse.classList.remove('hidden'); return; }
    emptyStateUse.classList.add('hidden');
    for (const r of results){
      const card = document.createElement('div'); card.className='card'; card.tabIndex=0;
      card.addEventListener('click', ()=>openMapDetail(r.id));
      card.addEventListener('keyup', ev=>{ if(ev.key==='Enter') openMapDetail(r.id); });

      const img = document.createElement('img');
      img.alt = `${r.city} – #${r.id}`;
      img.src = `data:image/png;base64,${r.map}`;

      const meta = document.createElement('div'); meta.className='meta';
      meta.innerHTML = `<div><strong>${r.city}</strong> – #${r.id}</div>
                        <div>Échelle: ${r.scale}</div>
                        <div><small>${new Date(r.created_at.replace(' ','T')).toLocaleString()}</small></div>`;
      card.append(img, meta); mapsGrid.appendChild(card);
    }
  }catch{ notify("Erreur lors du chargement des cartes", true); }
}

function openMapDetail(id){
  const url = new URL(window.location.href);
  url.searchParams.set('mapId', String(id));
  history.pushState({}, '', url);
  showView('detail');
  loadMapDetail(id);
}

/* ---------- Nouvelle carte (2 formulaires, anti-reload) ---------- */
const formUpload   = qs('#formUpload');
const formConfirm  = qs('#formConfirm');
const btnUpload    = qs('#btnUpload');
const btnConfirm   = qs('#btnConfirm');
const btnBack      = qs('#btnBack');
const btnOpenSaved = qs('#btnOpenSaved');
const binaryChoices= qs('#binaryChoices');

let lastBinaryCtx = null; // { filename, city, scale, results: { path:b64, ... } }
let lastSavedId = null;

function initNewView(){
  // Form 1 : upload → /map/binaryse
  formUpload.addEventListener('submit', async (e)=>{
    e.preventDefault(); e.stopPropagation(); // ⛔️ pas de reload, jamais
    const file  = qs('#fileInput').files[0];
    const scale = parseFloat(qs('#scaleInput').value);
    const city  = qs('#cityInput').value.trim();
    if (!file || !scale || !city){ notify("Remplis tous les champs.", true); return; }

    try{
      btnUpload.disabled = true;
      notify("Extraction des routes en cours…");
      const fd = new FormData();
      fd.append('file', file);
      fd.append('scale', String(scale));
      fd.append('city', city);

      const resp = await fetch(`${API_BASE}/map/binaryse`, { method:'POST', body: fd });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      lastBinaryCtx = data;
      renderBinaryChoices(data);

      // Affiche l'étape 2 (sans changer d’onglet, sans rediriger)
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

  // Form 2 : /map/add (aucune redirection auto)
  formConfirm.addEventListener('submit', async (e)=>{
    e.preventDefault(); e.stopPropagation(); // ⛔️ pas de reload
    if (!lastBinaryCtx){ notify("Aucun contexte d’upload.", true); return; }
    const chosen = qs('input[name="binPick"]:checked', binaryChoices);
    if (!chosen){ notify("Sélectionne une image binaire.", true); return; }

    const map_choosen = decodeURIComponent(chosen.value);
    const original_path = `map_store/temp/original_${lastBinaryCtx.filename}`;

    const fd = new FormData();
    fd.append('map', original_path);
    fd.append('map_choosen', map_choosen);
    fd.append('city', lastBinaryCtx.city);
    fd.append('scale', String(lastBinaryCtx.scale));

    try{
      btnConfirm.disabled = true;
      notify("Enregistrement de la carte…");
      const resp = await fetch(`${API_BASE}/map/add`, { method:'POST', body: fd });
      if (!resp.ok) throw new Error(await resp.text());
      const saved = await resp.json();
      lastSavedId = saved.id;

      // On reste sur place. On propose un bouton pour ouvrir la carte créée (à la demande).
      btnOpenSaved.classList.remove('hidden');
      btnOpenSaved.onclick = ()=>{
        const url = new URL(window.location.href);
        url.searchParams.set('mapId', String(lastSavedId));
        history.pushState({}, '', url);
        showView('detail');
        loadMapDetail(lastSavedId);
      };

      notify(`Carte #${saved.id} enregistrée pour ${saved.city} ✔️ (pas de redirection automatique)`);
      // Option : garder la prévisualisation visible. Sinon, décommente pour revenir à l’étape 1 :
      // resetConfirm(); formConfirm.classList.add('hidden'); formUpload.reset(); formUpload.classList.remove('hidden');
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
  if (entries.length === 0){
    btnConfirm.disabled = true;
    notify("Aucune image binaire reçue.", true);
    return;
  }
  let i = 0;
  for (const [path, b64] of entries){
    const container = document.createElement('div');
    container.className = 'bin';
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${b64}`;
    img.alt = `Binaire ${++i}`;

    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'binPick';
    radio.value = encodeURIComponent(path);
    if (i === 1) radio.checked = true;

    const label = document.createElement('label');
    label.className = 'pick';
    label.append(radio, document.createTextNode(' Choisir'));

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.style = 'padding:8px;color:#9aa4b2;font-size:.85rem';
    meta.textContent = path;

    container.append(label, img, meta);
    binaryChoices.appendChild(container);
  }
  btnConfirm.disabled = !qs('input[name="binPick"]:checked', binaryChoices);
  binaryChoices.addEventListener('change', ()=>{
    btnConfirm.disabled = !qs('input[name="binPick"]:checked', binaryChoices);
  }, { once:true });
}

function resetConfirm(){
  binaryChoices.innerHTML = '';
  lastBinaryCtx = null;
  lastSavedId = null;
  btnConfirm.disabled = true;
  btnOpenSaved.classList.add('hidden');
  btnOpenSaved.onclick = null;
}

/* ---------- Page détail carte ---------- */
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
    r.addEventListener('change', ()=>{ pickMode = r.value; drawMarkers(); });
  });

  mapImage.addEventListener('load', syncCanvasSize);
  window.addEventListener('resize', syncCanvasSize);

  mapImage.addEventListener('click', (ev)=>{
    const pt = eventToImageCoords(ev, mapImage);
    if (!pt) return;
    if (pickMode==='route'){
      if (!routePts.start) routePts.start = pt;
      else if (!routePts.end) routePts.end = pt;
      else routePts = { start: pt, end: null };
    } else {
      if (!blockPts.start) blockPts.start = pt;
      else if (!blockPts.end) blockPts.end = pt;
      else blockPts = { start: pt, end: null };
    }
    updateCoordLabels(); drawMarkers();
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
  const sx = imgEl.naturalWidth / rect.width;
  const sy = imgEl.naturalHeight / rect.height;
  return { x: Math.round(xDisp*sx), y: Math.round(yDisp*sy) };
}

function syncCanvasSize(){
  const rect = mapImage.getBoundingClientRect();
  clickLayer.width = rect.width; clickLayer.height = rect.height;
  clickLayer.style.width = rect.width+'px'; clickLayer.style.height = rect.height+'px';
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

  const dot = (pt, cssVar)=>{
    const x = Math.round(pt.x * sx), y = Math.round(pt.y * sy);
    ctx.beginPath(); ctx.arc(x,y,6,0,Math.PI*2);
    ctx.fillStyle = getCSS(cssVar); ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = '#000000aa'; ctx.stroke();
  };
  if (routePts.start) dot(routePts.start, '--dot-blue');
  if (routePts.end)   dot(routePts.end,   '--dot-blue');
  if (blockPts.start) dot(blockPts.start, '--dot-red');
  if (blockPts.end)   dot(blockPts.end,   '--dot-red');
}

function getCSS(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

async function onComputeRoute(){
  if (!currentMapId) return;
  if (!routePts.start || !routePts.end){ notify("Sélectionne d’abord un départ et une arrivée.", true); return; }

  const fd = new FormData();
  fd.append('map_id', String(currentMapId));
  fd.append('start', `${routePts.start.x},${routePts.start.y}`);
  fd.append('end',   `${routePts.end.x},${routePts.end.y}`);

  try{
    btnRoute.disabled = true; notify("Calcul du plus court chemin…");
    const res = await fetch(`${API_BASE}/itinerary/route`, { method:'POST', body: fd });
    if (!res.ok){
      let msg='Erreur lors du calcul d’itinéraire.'; try{ msg+=' '+(await res.json()).detail; }catch{}
      throw new Error(msg);
    }
    const data = await res.json();
    routeImage.src = `data:image/png;base64,${data.image}`;
    itineraryList.innerHTML = '';
    const steps = (data.itinerary||'').toString().split('.').map(s=>s.trim()).filter(Boolean);
    steps.forEach(step=>{ const li=document.createElement('li'); li.textContent=step; itineraryList.appendChild(li); });
    routeMeta.innerHTML = `<div><strong>Distance :</strong> ${data.distance_m} m</div>
                           <div><strong>Temps estimé :</strong> ${data.estimated_time}</div>`;
    routeMeta.classList.remove('hidden'); resultPanel.classList.remove('hidden');
    notify("Itinéraire prêt ✅");
  }catch(err){ console.error(err); notify(err.message || "Échec du calcul d’itinéraire.", true); }
  finally{ btnRoute.disabled = false; }
}

async function onBlockRoute(){
  if (!currentMapId) return;
  if (!blockPts.start || !blockPts.end){ notify("Sélectionne deux points à barrer.", true); return; }
  const fd = new FormData();
  fd.append('map_id', String(currentMapId));
  fd.append('start', `${blockPts.start.x},${blockPts.start.y}`);
  fd.append('end',   `${blockPts.end.x},${blockPts.end.y}`);

  try{
    btnBlock.disabled = true; notify("Application de la restriction sur la carte…");
    const res = await fetch(`${API_BASE}/map/modify_bin`, { method:'POST', body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || 'Erreur API');
    await res.json();
    notify("Route barrée. Les prochains calculs utiliseront la carte modifiée ✅");
  }catch(err){ console.error(err); notify("Impossible de barrer cette route.", true); }
  finally{ btnBlock.disabled = false; }
}

async function onResetBin(){
  if (!currentMapId) return;
  try{
    btnReset.disabled = true; notify("Restauration de la carte originale…");
    const res = await fetch(`${API_BASE}/map/${currentMapId}/bin/reset`, { method:'DELETE' });
    if (!res.ok) throw new Error((await res.json()).detail || 'Erreur API');
    await res.json();
    notify("Carte restaurée sur le binaire original ✔️");
  }catch(err){ console.error(err); notify("Échec de la restauration.", true); }
  finally{ btnReset.disabled = false; }
}

/* ---------- Utils ---------- */
async function GET(path){
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

let toastTimer=null;
function notify(msg, isError=false){
  toastEl.textContent = msg;
  toastEl.classList.remove('hidden');
  toastEl.style.borderColor = isError ? '#823b32' : 'var(--border)';
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>toastEl.classList.add('hidden'), 3500);
}
