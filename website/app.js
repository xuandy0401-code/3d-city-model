/**
 * IC Campus 3D Explorer
 *
 * Buildings are rendered with THREE.ExtrudeGeometry from OSM polygon footprints.
 * Each building is a single SOLID mesh with a top cap, bottom cap, and side walls.
 * No OBJ file required.
 *
 * Coordinate system (matches the pipeline's local UTM):
 *   +X = East (metres from campus centre)
 *   +Y = Height / Up
 *   +Z = South  (−North in UTM)
 */

import * as THREE                from 'three';
import { OrbitControls }         from 'three/addons/controls/OrbitControls.js';

// ── Config ────────────────────────────────────────────────────────────────────
const DATA_URL  = './building_data.json';
const IMG_BASE  = './IC_campus_streetview/';

const HEADING_NAMES = {0:'N',45:'NE',90:'E',135:'SE',180:'S',225:'SW',270:'W',315:'NW'};

// ── State ─────────────────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let raycaster, mouse;
let buildingData   = {};
let buildingMeshes = new Map();   // bid → THREE.Mesh
let allMeshes      = [];          // flat list for raycasting
let hoveredBid     = null;
let selectedBid    = null;
let mouseDownXY    = null;

let lbImages = [], lbIndex = 0;

// ── Colour helpers ────────────────────────────────────────────────────────────
function heightToColor(h) {
  const t = Math.min(h / 65, 1);
  // blue(0.62) → teal(0.48) → yellow-green(0.30) → orange-red(0.05)
  const hue  = 0.62 - t * 0.57;
  const sat  = 0.50 + t * 0.28;
  const lit  = 0.42 + (1 - t) * 0.12;
  return new THREE.Color().setHSL(hue, sat, lit);
}

function brighten(c, amount = 0.18) {
  const h = c.clone();
  h.offsetHSL(0, 0, amount);
  return h;
}

const SEL_COLOR = new THREE.Color(0x88ccff);

// ── Scene ─────────────────────────────────────────────────────────────────────
function setupScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080c18);
  scene.fog = new THREE.FogExp2(0x080c18, 0.0005);

  camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 1, 4000);
  camera.position.set(350, 420, 850);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
  document.getElementById('canvas-container').appendChild(renderer.domElement);

  raycaster = new THREE.Raycaster();
  mouse     = new THREE.Vector2();
}

function setupLighting() {
  // Soft sky fill
  scene.add(new THREE.AmbientLight(0x8090b8, 0.85));

  // Main directional (sun, NE above)
  const sun = new THREE.DirectionalLight(0xfff4d8, 1.6);
  sun.position.set(450, 700, 350);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, {near:1, far:2200, left:-900, right:900, top:900, bottom:-900});
  sun.shadow.bias = -0.0008;
  scene.add(sun);

  // Secondary fill from west
  const fill = new THREE.DirectionalLight(0x6070a8, 0.45);
  fill.position.set(-400, 200, -300);
  scene.add(fill);
}

function setupControls() {
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping   = true;
  controls.dampingFactor   = 0.07;
  controls.screenSpacePanning = false;
  controls.minDistance     = 25;
  controls.maxDistance     = 2200;
  controls.maxPolarAngle   = Math.PI / 2.05;
  controls.target.set(-60, 8, 90);
  controls.update();
}

function addGround() {
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(3000, 3000),
    new THREE.MeshPhongMaterial({ color: 0x090d1c })
  );
  plane.rotation.x = -Math.PI / 2;
  plane.receiveShadow = true;
  scene.add(plane);

  // Subtle grid
  const grid = new THREE.GridHelper(2200, 55, 0x111c38, 0x0d1428);
  grid.position.y = 0.15;
  scene.add(grid);
}

// ── Building geometry (ExtrudeGeometry) ───────────────────────────────────────
function createBuildings() {
  const entries = Object.entries(buildingData);
  const total   = entries.length;
  let done = 0, nBuildings = 0, nWithImg = 0;

  for (const [bidStr, bdat] of entries) {
    done++;
    if (done % 50 === 0) {
      const pct = Math.round(done / total * 100);
      document.getElementById('loading-bar').style.width = pct + '%';
      document.getElementById('loading-status').textContent =
        `正在构建三维几何体… ${pct}%`;
    }

    const fp = bdat.footprint;
    if (!fp || fp.length < 3) continue;

    const bid      = parseInt(bidStr);
    const height   = Math.max(bdat.height_m || 15, 3);
    const hasImg   = bdat.images && bdat.images.length > 0;
    const baseCol  = heightToColor(height);

    // ── Shape in local East-North (X-Y) plane ────────────────────────────────
    const shape = new THREE.Shape();
    shape.moveTo(fp[0][0], fp[0][1]);
    for (let i = 1; i < fp.length; i++) shape.lineTo(fp[i][0], fp[i][1]);
    shape.closePath();

    if (bdat.holes) {
      for (const hc of bdat.holes) {
        if (hc.length < 3) continue;
        const hole = new THREE.Path();
        hole.moveTo(hc[0][0], hc[0][1]);
        for (let i = 1; i < hc.length; i++) hole.lineTo(hc[i][0], hc[i][1]);
        hole.closePath();
        shape.holes.push(hole);
      }
    }

    // ── Extrude upward ───────────────────────────────────────────────────────
    let geo;
    try {
      geo = new THREE.ExtrudeGeometry(shape, { depth: height, bevelEnabled: false });
    } catch (e) {
      console.warn(`Building ${bid}: geometry failed`, e);
      continue;
    }
    // Rotate so the East-North plane becomes the ground (X-Z) and extrusion goes Y-up
    geo.rotateX(-Math.PI / 2);
    geo.computeVertexNormals();   // recompute after rotation

    // ── Material ─────────────────────────────────────────────────────────────
    const mat = new THREE.MeshPhongMaterial({
      color:     baseCol.clone(),
      shininess: 30,
      specular:  new THREE.Color(0x101828),
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow    = true;
    mesh.receiveShadow = true;
    mesh.userData      = { bid, baseColor: baseCol.clone(), hasImg };

    scene.add(mesh);
    buildingMeshes.set(bid, mesh);
    allMeshes.push(mesh);
    nBuildings++;
    if (hasImg) nWithImg++;
  }

  document.getElementById('stat-buildings').textContent = nBuildings;
  document.getElementById('stat-with-img').textContent  = nWithImg;
  hideLoading();
}

function hideLoading() {
  const el = document.getElementById('loading');
  el.style.opacity = '0';
  setTimeout(() => (el.style.display = 'none'), 500);
}

// ── Highlight helpers ─────────────────────────────────────────────────────────
function applyColor(bid, col) {
  const m = buildingMeshes.get(bid);
  if (!m) return;
  m.material.color.set(col);
  m.material.emissive.set(col === null ? 0x000000 : 0x0d2040);
}

function restoreColor(bid) {
  const m = buildingMeshes.get(bid);
  if (!m) return;
  m.material.color.set(m.userData.baseColor);
  m.material.emissive.set(0x000000);
}

function selectBuilding(bid) {
  if (selectedBid !== null) restoreColor(selectedBid);
  selectedBid = bid;
  if (bid !== null) applyColor(bid, SEL_COLOR);
}

// ── Mouse events ──────────────────────────────────────────────────────────────
function onMouseMove(e) {
  mouse.x =  (e.clientX / innerWidth)  * 2 - 1;
  mouse.y = -(e.clientY / innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(allMeshes, false);

  if (hits.length > 0) {
    const { bid, hasImg, baseColor } = hits[0].object.userData;

    renderer.domElement.style.cursor = hasImg ? 'pointer' : 'default';

    if (bid !== hoveredBid) {
      if (hoveredBid !== null && hoveredBid !== selectedBid) restoreColor(hoveredBid);
      hoveredBid = bid;
      if (bid !== selectedBid) applyColor(bid, brighten(baseColor));
    }
    showTooltip(e, bid);
  } else {
    if (hoveredBid !== null && hoveredBid !== selectedBid) restoreColor(hoveredBid);
    hoveredBid = null;
    hideTooltip();
    renderer.domElement.style.cursor = 'default';
  }
}

function onMouseDown(e) { mouseDownXY = [e.clientX, e.clientY]; }

function onMouseUp(e) {
  if (!mouseDownXY || e.button !== 0) return;
  const dx = e.clientX - mouseDownXY[0];
  const dy = e.clientY - mouseDownXY[1];
  mouseDownXY = null;
  if (Math.sqrt(dx * dx + dy * dy) > 5) return;   // drag → skip

  mouse.x =  (e.clientX / innerWidth)  * 2 - 1;
  mouse.y = -(e.clientY / innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(allMeshes, false);

  if (hits.length > 0) {
    const bid = hits[0].object.userData.bid;
    if (bid === selectedBid) { selectBuilding(null); closePanel(); }
    else                     { selectBuilding(bid);  openPanel(bid); }
  } else {
    selectBuilding(null);
    closePanel();
  }
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function showTooltip(e, bid) {
  const b = buildingData[bid];
  if (!b) return;
  document.getElementById('tooltip').classList.remove('hidden');
  document.getElementById('tt-name').textContent = b.name;
  document.getElementById('tt-sub').textContent  =
    `高度 ${b.height_m}m · ` +
    (b.images.length ? `${b.images.length} 张街景` : '无图片');
  const tt = document.getElementById('tooltip');
  tt.style.left = (e.clientX + 14) + 'px';
  tt.style.top  = (e.clientY - 8) + 'px';
}
function hideTooltip() {
  document.getElementById('tooltip').classList.add('hidden');
}

// ── Side panel ────────────────────────────────────────────────────────────────
function openPanel(bid) {
  const b = buildingData[bid];
  if (!b) return;
  document.getElementById('panel-cat').textContent =
    b.has_name ? '命名建筑' : '未命名建筑';
  document.getElementById('panel-name').textContent     = b.name;
  document.getElementById('panel-height').textContent   = `高度 ${b.height_m} m`;
  document.getElementById('panel-imgcount').textContent =
    b.images.length ? `${b.images.length} 张街景图` : '暂无图片';
  buildGallery(b.images);
  document.getElementById('side-panel').classList.remove('hidden');
}

window.closePanel = function () {
  document.getElementById('side-panel').classList.add('hidden');
  if (selectedBid !== null) { restoreColor(selectedBid); selectedBid = null; }
};

// ── Gallery ───────────────────────────────────────────────────────────────────
function buildGallery(images) {
  const gallery = document.getElementById('image-gallery');
  const tabs    = document.getElementById('dir-tabs');
  gallery.innerHTML = '';
  tabs.innerHTML    = '';

  if (!images || !images.length) {
    gallery.innerHTML = `<div class="gallery-empty">
      <div class="icon">🌐</div>
      该建筑暂无分类街景图片<br/>
      <span style="font-size:.72rem;color:#203040">可能位于采集路线之外</span>
    </div>`;
    return;
  }

  // Group by heading
  const byH = {};
  images.forEach(img => (byH[img.heading] = byH[img.heading] || []).push(img));

  const makeTab = (text, active, onClick) => {
    const b = document.createElement('button');
    b.className   = 'dir-tab' + (active ? ' active' : '');
    b.textContent = text;
    b.onclick = () => {
      tabs.querySelectorAll('.dir-tab').forEach(t => t.classList.remove('active'));
      b.classList.add('active');
      onClick();
    };
    return b;
  };

  tabs.appendChild(makeTab('全部', true, () => renderAll(images, byH)));
  Object.keys(byH).map(Number).sort((a,b)=>a-b).forEach(h => {
    const lbl = HEADING_NAMES[h] || h + '°';
    tabs.appendChild(makeTab(`${lbl} (${byH[h].length})`, false,
      () => { gallery.innerHTML=''; gallery.appendChild(buildSection(lbl,byH[h],byH[h])); }));
  });

  renderAll(images, byH);
}

function renderAll(images, byH) {
  const gallery = document.getElementById('image-gallery');
  gallery.innerHTML = '';
  Object.keys(byH).map(Number).sort((a,b)=>a-b).forEach(h => {
    gallery.appendChild(buildSection(HEADING_NAMES[h]||h+'°', byH[h], images));
  });
}

function buildSection(label, imgs, allImgs) {
  const sec  = document.createElement('div');
  sec.className = 'heading-section';
  sec.innerHTML = `<div class="heading-label">↗ ${label}</div>`;
  const grid = document.createElement('div');
  grid.className = 'img-grid';

  imgs.forEach(img => {
    const lbIdx = allImgs.indexOf(img);
    const card  = document.createElement('div');
    card.className = 'img-card';
    card.onclick   = () => openLightbox(allImgs, lbIdx >= 0 ? lbIdx : 0);

    const sk = document.createElement('div');
    sk.className = 'img-skeleton';
    card.appendChild(sk);

    const imgEl = new Image();
    imgEl.loading = 'lazy';
    imgEl.alt     = label;
    imgEl.onload  = () => sk.remove();
    imgEl.src     = IMG_BASE + img.filename;
    card.appendChild(imgEl);

    const overlay = document.createElement('div');
    overlay.className = 'img-overlay';
    overlay.innerHTML = `
      <span class="img-dist">${img.distance_m ? img.distance_m+'m' : ''}</span>
      <span class="img-dir">${HEADING_NAMES[img.heading]||img.heading+'°'}</span>`;
    card.appendChild(overlay);

    // Per-card download button
    const dlBtn = document.createElement('button');
    dlBtn.className   = 'card-dl';
    dlBtn.title       = '下载图片';
    dlBtn.innerHTML   = '⬇';
    dlBtn.onclick     = (e) => { e.stopPropagation(); triggerDownload(img.filename); };
    card.appendChild(dlBtn);

    grid.appendChild(card);
  });

  sec.appendChild(grid);
  return sec;
}

// ── Lightbox ──────────────────────────────────────────────────────────────────
function openLightbox(imgs, idx) {
  lbImages = imgs; lbIndex = idx;
  document.getElementById('lightbox').classList.remove('hidden');
  refreshLightbox();
}

function refreshLightbox() {
  const img = lbImages[lbIndex];
  document.getElementById('lb-img').src = IMG_BASE + img.filename;
  document.getElementById('lb-caption').textContent =
    `${HEADING_NAMES[img.heading]||img.heading+'°'} · ` +
    (img.distance_m ? `距建筑 ${img.distance_m}m · ` : '') +
    `${lbIndex+1} / ${lbImages.length}`;
}

window.lbNav = function (dir) {
  lbIndex = (lbIndex + dir + lbImages.length) % lbImages.length;
  refreshLightbox();
};
window.closeLightbox = function () {
  document.getElementById('lightbox').classList.add('hidden');
};

// Keyboard navigation
document.addEventListener('keydown', e => {
  if (document.getElementById('lightbox').classList.contains('hidden')) return;
  if (e.key === 'ArrowLeft')  window.lbNav(-1);
  if (e.key === 'ArrowRight') window.lbNav(1);
  if (e.key === 'Escape')     window.closeLightbox();
});
document.getElementById('lightbox').addEventListener('click', e => {
  if (e.target === e.currentTarget) window.closeLightbox();
});

// ── Download helpers ──────────────────────────────────────────────────────────
function triggerDownload(filename) {
  const a = document.createElement('a');
  a.href     = IMG_BASE + filename;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

window.downloadCurrentImage = function () {
  if (!lbImages.length) return;
  triggerDownload(lbImages[lbIndex].filename);
};

// ── Animation & resize ────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function init() {
  setupScene();
  setupLighting();
  setupControls();
  addGround();

  document.getElementById('loading-status').textContent = '正在加载建筑数据…';
  const resp = await fetch(DATA_URL);
  buildingData = await resp.json();

  // Attach mouse events AFTER renderer canvas exists
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('mousedown', onMouseDown);
  renderer.domElement.addEventListener('mouseup',   onMouseUp);

  createBuildings();
  animate();
}

init();
