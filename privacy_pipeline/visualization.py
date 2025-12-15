import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _scene_root(image_path: Path, levels_up: int) -> Path:
    levels_up = max(1, levels_up)
    current = image_path
    for _ in range(levels_up):
        current = current.parent
    return current


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _derive_scene_images(
    source_records: Iterable[Dict], scene_level: int
) -> Dict[str, List[str]]:
    scene_to_images: Dict[str, List[str]] = defaultdict(list)
    for record in source_records:
        image_path = Path(record.get("image_path", ""))
        if not image_path:
            continue
        scene_path = str(_scene_root(image_path, scene_level))
        scene_to_images[scene_path].append(str(image_path))
    for scene, images in scene_to_images.items():
        images.sort()
        scene_to_images[scene] = images
    return scene_to_images


def _flatten_text(obj) -> str:
    texts: List[str] = []

    def _walk(node):
        if isinstance(node, dict):
            if "text" in node and isinstance(node["text"], str):
                texts.append(node["text"])
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(obj)
    return "\n".join(t for t in texts if t)


def _extract_categories(obj) -> List[str]:
    categories = set()

    def _walk(node):
        if isinstance(node, dict):
            if "categories" in node and isinstance(node["categories"], list):
                for entry in node["categories"]:
                    if isinstance(entry, str):
                        categories.add(entry)
                    elif isinstance(entry, dict):
                        name = entry.get("name") or entry.get("category")
                        if isinstance(name, str):
                            categories.add(name)
            if "category" in node and isinstance(node["category"], str):
                categories.add(node["category"])
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(obj)
    return sorted(categories)


def generate_yoloe_viewer(
    yolo_output: Path,
    output_dir: Path,
    scene_level: int = 1,
    index_jsonl: Optional[Path] = None,
    title: str = "YOLOE Viewer",
) -> Path:
    yolo_records = list(_load_jsonl(yolo_output))
    source_records = list(_load_jsonl(index_jsonl)) if index_jsonl else yolo_records
    scene_to_all_images = _derive_scene_images(source_records, scene_level)

    images_payload: List[Dict] = []
    scenes_payload: Dict[str, Dict] = {}
    for record in yolo_records:
        image_path = Path(record.get("image_path", ""))
        if not image_path:
            continue
        scene_path = str(_scene_root(image_path, scene_level))
        detections = record.get("detections", []) or []
        display_path = record.get("visualization_path") or str(image_path)
        attributes = record.get("attributes", {}) or {}

        images_payload.append(
            {
                "image_path": str(image_path),
                "display_path": display_path,
                "scene_path": scene_path,
                "attributes": attributes,
                "detections": detections,
                "has_visualization": record.get("visualization_path") is not None,
            }
        )

        scene_entry = scenes_payload.setdefault(
            scene_path,
            {
                "scene_path": scene_path,
                "attributes": attributes,
                "detected_images": [],
                "all_images": scene_to_all_images.get(scene_path, []),
            },
        )
        scene_entry["detected_images"].append(display_path)

    payload = {
        "title": title,
        "scene_level": scene_level,
        "images": images_payload,
        "scenes": list(scenes_payload.values()),
    }

    data_path = output_dir / "yoloe_viewer_data.json"
    _write_json(data_path, payload)

    html_path = output_dir / "yoloe_viewer.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_YOLOE_HTML)
    return html_path


def generate_gemini_viewer(
    gemini_output: Path,
    output_dir: Path,
    scene_level: int = 1,
    index_jsonl: Optional[Path] = None,
    yolo_output: Optional[Path] = None,
    title: str = "Gemini OCR Viewer",
) -> Path:
    gemini_records = list(_load_jsonl(gemini_output))

    scene_source_records: List[Dict] = []
    if index_jsonl:
        scene_source_records.extend(_load_jsonl(index_jsonl))
    elif yolo_output:
        scene_source_records.extend(_load_jsonl(yolo_output))
    scene_to_all_images = _derive_scene_images(scene_source_records, scene_level)

    scenes: List[Dict] = []
    for record in gemini_records:
        scene_path = record.get("scene_path")
        if not scene_path:
            continue
        response = record.get("gemini_response")
        categories = _extract_categories(response)
        ocr_text = _flatten_text(response)
        scenes.append(
            {
                "scene_path": scene_path,
                "attributes": record.get("attributes", {}) or {},
                "categories": categories,
                "ocr_text": ocr_text,
                "images": scene_to_all_images.get(scene_path, []),
            }
        )

    payload = {
        "title": title,
        "scene_level": scene_level,
        "scenes": scenes,
    }

    data_path = output_dir / "gemini_viewer_data.json"
    _write_json(data_path, payload)

    html_path = output_dir / "gemini_viewer.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_GEMINI_HTML)
    return html_path


_YOLOE_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>YOLOE Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
    header { padding: 16px; background: #0b3954; color: white; }
    main { padding: 16px; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
    .controls label { display: flex; flex-direction: column; font-size: 12px; }
    .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(var(--thumb-size, 200px), 1fr)); }
    .card { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .card img { width: 100%; display: block; }
    .card .meta { padding: 8px; font-size: 12px; }
    .pill { display: inline-block; padding: 2px 6px; margin: 2px; background: #f1f5f9; border-radius: 4px; }
    .hidden { display: none; }
    .scene-view { margin-top: 16px; }
    .scene-images { display: grid; gap: 8px; grid-template-columns: repeat(auto-fill, minmax(var(--thumb-size, 160px), 1fr)); }
    .topbar { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
    button { cursor: pointer; }
  </style>
</head>
<body>
  <header>
    <h1 id=\"title\">YOLOE Viewer</h1>
  </header>
  <main>
    <div class=\"controls\">
      <label>Class filter
        <select id=\"classFilter\" multiple size=\"4\"></select>
      </label>
      <label>Confidence ≥ <span id=\"confVal\">0.0</span>
        <input type=\"range\" id=\"confSlider\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0\" />
      </label>
      <label>Group by scene
        <input type=\"checkbox\" id=\"groupToggle\" />
      </label>
      <label>Scene level
        <input type=\"number\" id=\"sceneLevel\" min=\"1\" value=\"1\" />
      </label>
      <label>Thumb size (px)
        <input type=\"number\" id=\"thumbSize\" min=\"80\" value=\"200\" />
      </label>
      <label>Cards per page
        <input type=\"number\" id=\"pageSize\" min=\"1\" value=\"50\" />
      </label>
      <div id=\"attrFilters\"></div>
    </div>

    <div id=\"grid\" class=\"grid\"></div>
    <div id=\"sceneView\" class=\"scene-view hidden\">
      <div class=\"topbar\">
        <button id=\"backBtn\">Back to grid</button>
        <div id=\"sceneInfo\"></div>
      </div>
      <div id=\"sceneImages\" class=\"scene-images\"></div>
    </div>
  </main>

  <script>
    const state = { data: null, grouped: false, pageSize: 50, thumbSize: 200, conf: 0, sceneLevel: 1, attrFilters: {} };

    async function loadData() {
      const res = await fetch('yoloe_viewer_data.json');
      state.data = await res.json();
      document.getElementById('title').textContent = state.data.title || 'YOLOE Viewer';
      state.sceneLevel = state.data.scene_level || 1;
      document.getElementById('sceneLevel').value = state.sceneLevel;
      initFilters();
      render();
    }

    function initFilters() {
      const classes = new Set();
      const attrKeys = new Set();
      state.data.images.forEach(img => {
        (img.detections || []).forEach(det => classes.add(det.class));
        Object.keys(img.attributes || {}).forEach(k => attrKeys.add(k));
      });
      const classSelect = document.getElementById('classFilter');
      classes.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c; opt.textContent = c; classSelect.appendChild(opt);
      });

      const attrContainer = document.getElementById('attrFilters');
      attrContainer.innerHTML = '';
      attrKeys.forEach(key => {
        const values = _unique(Array.from(state.data.images.map(i => i.attributes?.[key]).filter(Boolean)));
        const label = document.createElement('label');
        label.textContent = key;
        const select = document.createElement('select');
        select.multiple = true; select.size = 4; select.dataset.key = key;
        values.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v; opt.textContent = v; select.appendChild(opt);
        });
        label.appendChild(select);
        attrContainer.appendChild(label);
      });

      document.getElementById('confSlider').addEventListener('input', e => { state.conf = parseFloat(e.target.value); document.getElementById('confVal').textContent = state.conf.toFixed(2); render(); });
      document.getElementById('groupToggle').addEventListener('change', e => { state.grouped = e.target.checked; render(); });
      document.getElementById('thumbSize').addEventListener('input', e => { state.thumbSize = parseInt(e.target.value || '200'); document.documentElement.style.setProperty('--thumb-size', state.thumbSize + 'px'); render(); });
      document.getElementById('pageSize').addEventListener('input', e => { state.pageSize = parseInt(e.target.value || '50'); render(); });
      document.getElementById('sceneLevel').addEventListener('input', e => { state.sceneLevel = parseInt(e.target.value || '1'); render(); });
      document.getElementById('classFilter').addEventListener('change', render);
      attrContainer.addEventListener('change', e => { if (e.target.tagName === 'SELECT') render(); });
      document.getElementById('backBtn').addEventListener('click', () => { document.getElementById('sceneView').classList.add('hidden'); });
    }

    function _unique(arr) { return Array.from(new Set(arr)).sort(); }

    function gatherAttrFilters() {
      const filters = {};
      document.querySelectorAll('#attrFilters select').forEach(sel => {
        const selected = Array.from(sel.selectedOptions).map(o => o.value);
        if (selected.length) filters[sel.dataset.key] = selected;
      });
      return filters;
    }

    function imagePasses(img, classes, conf, attrFilters) {
      if (classes.length) {
        const hasClass = (img.detections || []).some(det => classes.includes(det.class) && det.confidence >= conf);
        if (!hasClass) return false;
      } else {
        const meetsConf = (img.detections || []).some(det => det.confidence >= conf);
        if (conf > 0 && !meetsConf) return false;
      }
      for (const [key, values] of Object.entries(attrFilters)) {
        if (!values.includes(img.attributes?.[key])) return false;
      }
      return true;
    }

    function render() {
      state.attrFilters = gatherAttrFilters();
      const classes = Array.from(document.getElementById('classFilter').selectedOptions).map(o => o.value);
      const conf = state.conf;
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      const filtered = state.grouped ? buildSceneCards(classes, conf) : buildImageCards(classes, conf);
      const limited = filtered.slice(0, state.pageSize);
      limited.forEach(card => grid.appendChild(card));
    }

    function buildImageCards(classes, conf) {
      const cards = [];
      state.data.images.forEach(img => {
        if (!_matchesSceneLevel(img.scene_path)) return;
        if (!imagePasses(img, classes, conf, state.attrFilters)) return;
        const card = document.createElement('div'); card.className = 'card';
        const imageEl = document.createElement('img'); imageEl.loading = 'lazy'; imageEl.src = img.display_path; card.appendChild(imageEl);
        const meta = document.createElement('div'); meta.className = 'meta';
        meta.innerHTML = `<div><strong>${img.image_path}</strong></div>`;
        const dets = (img.detections || []).filter(d => (!classes.length || classes.includes(d.class)) && d.confidence >= conf);
        if (dets.length) meta.innerHTML += '<div>' + dets.map(d => `<span class="pill">${d.class} (${d.confidence.toFixed(2)})</span>`).join(' ') + '</div>';
        const attrs = Object.entries(img.attributes || {}).map(([k,v]) => `<span class="pill">${k}: ${v}</span>`).join(' ');
        if (attrs) meta.innerHTML += '<div>' + attrs + '</div>';
        const btn = document.createElement('button'); btn.textContent = 'View scene'; btn.onclick = () => showScene(img.scene_path); meta.appendChild(btn);
        card.appendChild(meta);
        cards.push(card);
      });
      return cards;
    }

    function buildSceneCards(classes, conf) {
      const cards = [];
      state.data.scenes.forEach(scene => {
        if (!_matchesSceneLevel(scene.scene_path)) return;
        const anyMatch = state.data.images.some(img => img.scene_path === scene.scene_path && imagePasses(img, classes, conf, state.attrFilters));
        if (!anyMatch) return;
        const thumb = scene.detected_images[0] || scene.all_images[0];
        if (!thumb) return;
        const card = document.createElement('div'); card.className = 'card';
        const imageEl = document.createElement('img'); imageEl.loading = 'lazy'; imageEl.src = thumb; card.appendChild(imageEl);
        const meta = document.createElement('div'); meta.className = 'meta';
        meta.innerHTML = `<div><strong>${scene.scene_path}</strong></div>`;
        const attrs = Object.entries(scene.attributes || {}).map(([k,v]) => `<span class="pill">${k}: ${v}</span>`).join(' ');
        if (attrs) meta.innerHTML += '<div>' + attrs + '</div>';
        const btn = document.createElement('button'); btn.textContent = 'Open scene'; btn.onclick = () => showScene(scene.scene_path); meta.appendChild(btn);
        card.appendChild(meta);
        cards.push(card);
      });
      return cards;
    }

    function _matchesSceneLevel(scenePath) {
      if (!scenePath) return false;
      const parts = scenePath.split(/[\\/]/).filter(Boolean);
      return parts.length >= state.sceneLevel;
    }

    function showScene(scenePath) {
      const view = document.getElementById('sceneView');
      const info = document.getElementById('sceneInfo');
      const container = document.getElementById('sceneImages');
      view.classList.remove('hidden');
      info.textContent = scenePath;
      container.innerHTML = '';
      const images = (state.data.scenes.find(s => s.scene_path === scenePath)?.all_images) ||
        state.data.images.filter(i => i.scene_path === scenePath).map(i => i.display_path);
      images.forEach(p => {
        const img = document.createElement('img'); img.loading = 'lazy'; img.src = p; container.appendChild(img);
      });
    }

    loadData();
  </script>
</body>
</html>
"""


_GEMINI_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Gemini OCR Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
    header { padding: 16px; background: #093e6a; color: white; }
    main { padding: 16px; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
    .controls label { display: flex; flex-direction: column; font-size: 12px; }
    .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(var(--thumb-size, 220px), 1fr)); }
    .card { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .card img { width: 100%; display: block; }
    .meta { padding: 8px; font-size: 12px; }
    .pill { display: inline-block; padding: 2px 6px; margin: 2px; background: #f1f5f9; border-radius: 4px; }
    .hidden { display: none; }
    .scene-view { margin-top: 16px; }
    .scene-images { display: grid; gap: 8px; grid-template-columns: repeat(auto-fill, minmax(var(--thumb-size, 180px), 1fr)); }
    .ocr { white-space: pre-wrap; background: #f8fafc; padding: 8px; border-radius: 6px; margin-bottom: 8px; }
    .topbar { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  </style>
</head>
<body>
  <header>
    <h1 id=\"title\">Gemini OCR Viewer</h1>
  </header>
  <main>
    <div class=\"controls\">
      <label>Categories
        <select id=\"categoryFilter\" multiple size=\"4\"></select>
      </label>
      <label>Show OCR under thumbnails
        <input type=\"checkbox\" id=\"ocrToggle\" />
      </label>
      <label>Thumb size (px)
        <input type=\"number\" id=\"thumbSize\" min=\"80\" value=\"220\" />
      </label>
      <label>Cards per page
        <input type=\"number\" id=\"pageSize\" min=\"1\" value=\"50\" />
      </label>
      <div id=\"attrFilters\"></div>
    </div>

    <div id=\"grid\" class=\"grid\"></div>
    <div id=\"sceneView\" class=\"scene-view hidden\">
      <div class=\"topbar\">
        <button id=\"backBtn\">Back to grid</button>
        <div id=\"sceneInfo\"></div>
      </div>
      <div id=\"sceneOcr\" class=\"ocr\"></div>
      <div id=\"sceneImages\" class=\"scene-images\"></div>
    </div>
  </main>

  <script>
    const state = { data: null, thumbSize: 220, pageSize: 50, showOcr: false, attrFilters: {} };

    async function loadData() {
      const res = await fetch('gemini_viewer_data.json');
      state.data = await res.json();
      document.getElementById('title').textContent = state.data.title || 'Gemini OCR Viewer';
      initFilters();
      render();
    }

    function initFilters() {
      const categories = new Set();
      const attrKeys = new Set();
      state.data.scenes.forEach(scene => {
        (scene.categories || []).forEach(c => categories.add(c));
        Object.keys(scene.attributes || {}).forEach(k => attrKeys.add(k));
      });
      const catSelect = document.getElementById('categoryFilter');
      categories.forEach(c => { const opt = document.createElement('option'); opt.value = c; opt.textContent = c; catSelect.appendChild(opt); });

      const attrContainer = document.getElementById('attrFilters');
      attrContainer.innerHTML = '';
      attrKeys.forEach(key => {
        const values = Array.from(new Set(state.data.scenes.map(s => s.attributes?.[key]).filter(Boolean))).sort();
        const label = document.createElement('label'); label.textContent = key;
        const select = document.createElement('select'); select.multiple = true; select.size = 4; select.dataset.key = key;
        values.forEach(v => { const opt = document.createElement('option'); opt.value = v; opt.textContent = v; select.appendChild(opt); });
        label.appendChild(select); attrContainer.appendChild(label);
      });

      document.getElementById('thumbSize').addEventListener('input', e => { state.thumbSize = parseInt(e.target.value || '220'); document.documentElement.style.setProperty('--thumb-size', state.thumbSize + 'px'); render(); });
      document.getElementById('pageSize').addEventListener('input', e => { state.pageSize = parseInt(e.target.value || '50'); render(); });
      document.getElementById('ocrToggle').addEventListener('change', e => { state.showOcr = e.target.checked; render(); });
      document.getElementById('categoryFilter').addEventListener('change', render);
      attrContainer.addEventListener('change', e => { if (e.target.tagName === 'SELECT') render(); });
      document.getElementById('backBtn').addEventListener('click', () => document.getElementById('sceneView').classList.add('hidden'));
    }

    function gatherAttrFilters() {
      const filters = {};
      document.querySelectorAll('#attrFilters select').forEach(sel => {
        const selected = Array.from(sel.selectedOptions).map(o => o.value);
        if (selected.length) filters[sel.dataset.key] = selected;
      });
      return filters;
    }

    function scenePasses(scene, cats, attrFilters) {
      if (cats.length) {
        const hasCat = (scene.categories || []).some(c => cats.includes(c));
        if (!hasCat) return false;
      }
      for (const [key, values] of Object.entries(attrFilters)) {
        if (!values.includes(scene.attributes?.[key])) return false;
      }
      return true;
    }

    function render() {
      state.attrFilters = gatherAttrFilters();
      const cats = Array.from(document.getElementById('categoryFilter').selectedOptions).map(o => o.value);
      const grid = document.getElementById('grid'); grid.innerHTML = '';
      const cards = [];
      state.data.scenes.forEach(scene => {
        if (!scenePasses(scene, cats, state.attrFilters)) return;
        const thumb = scene.images?.[0]; if (!thumb) return;
        const card = document.createElement('div'); card.className = 'card';
        const img = document.createElement('img'); img.loading = 'lazy'; img.src = thumb; card.appendChild(img);
        const meta = document.createElement('div'); meta.className = 'meta';
        meta.innerHTML = `<div><strong>${scene.scene_path}</strong></div>`;
        const attrs = Object.entries(scene.attributes || {}).map(([k,v]) => `<span class="pill">${k}: ${v}</span>`).join(' ');
        if (attrs) meta.innerHTML += '<div>' + attrs + '</div>';
        const catsRow = (scene.categories || []).map(c => `<span class="pill">${c}</span>`).join(' ');
        if (catsRow) meta.innerHTML += '<div>' + catsRow + '</div>';
        if (state.showOcr && scene.ocr_text) meta.innerHTML += `<div class="ocr">${scene.ocr_text}</div>`;
        const btn = document.createElement('button'); btn.textContent = 'Open scene'; btn.onclick = () => showScene(scene); meta.appendChild(btn);
        card.appendChild(meta);
        cards.push(card);
      });
      cards.slice(0, state.pageSize).forEach(c => grid.appendChild(c));
    }

    function showScene(scene) {
      const view = document.getElementById('sceneView'); view.classList.remove('hidden');
      document.getElementById('sceneInfo').textContent = scene.scene_path;
      document.getElementById('sceneOcr').textContent = scene.ocr_text || '';
      const container = document.getElementById('sceneImages'); container.innerHTML = '';
      (scene.images || []).forEach(p => { const img = document.createElement('img'); img.loading = 'lazy'; img.src = p; container.appendChild(img); });
    }

    loadData();
  </script>
</body>
</html>
"""
