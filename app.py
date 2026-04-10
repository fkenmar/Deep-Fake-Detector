import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./model/face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)


# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "./model" if os.path.exists("./model") else "prithivMLmods/Deep-Fake-Detector-v2-Model"
print(f"Loading model from: {MODEL_ID}")
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID)
model.eval()

# ── MediaPipe Face Mesh ──────────────────────────────────────────────────────

# ── Uncanny Valley Analysis ──────────────────────────────────────────────────
def analyze_uncanny(bgr_img):
    """Run uncanny valley heuristics on a BGR image. Returns dict of metrics."""
    results = {}
    h, w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # --- 1. Facial Symmetry ---
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = face_landmarker.detect(mp_image)
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        # Wrap landmarks so we can access .x / .y by index
        class _LM:
            def __init__(self, lst):
                self._lst = lst
            def __getitem__(self, idx):
                return self._lst[idx]
        lm = _LM(landmarks)

        # Compare left vs right side landmarks
        # Left eye: 33, Right eye: 263, Nose tip: 1
        pairs = [(33, 263), (133, 362), (70, 300), (105, 334), (107, 336)]
        nose_x = lm[1].x
        diffs = []
        for li, ri in pairs:
            left_dist = abs(lm[li].x - nose_x)
            right_dist = abs(lm[ri].x - nose_x)
            if max(left_dist, right_dist) > 0:
                diffs.append(abs(left_dist - right_dist) / max(left_dist, right_dist))
        symmetry = 1.0 - (sum(diffs) / len(diffs)) if diffs else 1.0
        results["symmetry"] = round(symmetry * 100, 1)

        # --- 2. Eye Reflection Consistency ---
        # Extract left and right eye regions and compare their brightness histograms
        def eye_region(indices):
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))
            if x2 <= x1 or y2 <= y1:
                return None
            return gray[y1:y2, x1:x2]

        left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263]
        le = eye_region(left_eye_idx)
        re = eye_region(right_eye_idx)

        if le is not None and re is not None and le.size > 0 and re.size > 0:
            le_resized = cv2.resize(le, (32, 16))
            re_resized = cv2.resize(re, (32, 16))
            # Compare histograms
            h_left = cv2.calcHist([le_resized], [0], None, [32], [0, 256])
            h_right = cv2.calcHist([re_resized], [0], None, [32], [0, 256])
            cv2.normalize(h_left, h_left)
            cv2.normalize(h_right, h_right)
            eye_corr = cv2.compareHist(h_left, h_right, cv2.HISTCMP_CORREL)
            results["eye_consistency"] = round(max(0, eye_corr) * 100, 1)
        else:
            results["eye_consistency"] = None
    else:
        results["symmetry"] = None
        results["eye_consistency"] = None

    # --- 3. Skin Texture (FFT frequency analysis) ---
    # Real skin has more high-frequency micro-texture; deepfakes are smoother
    gray_f = np.float32(gray)
    dft = cv2.dft(gray_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude = np.log(magnitude + 1)

    # Ratio of high-freq to total energy
    cy, cx = h // 2, w // 2
    radius = min(cy, cx) // 3
    total_energy = magnitude.sum()
    # Mask out center (low freq)
    mask = np.ones_like(magnitude)
    cv2.circle(mask, (cx, cy), radius, 0, -1)
    high_freq_energy = (magnitude * mask).sum()
    texture_score = high_freq_energy / total_energy if total_energy > 0 else 0
    # Normalize to 0-100 range (empirically tuned)
    results["texture"] = round(min(texture_score * 130, 100), 1)

    # --- 4. Edge Artifacts (Laplacian variance along face boundary) ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = laplacian.var()
    # Higher variance = more natural edges; low = over-smoothed blending
    # Normalize: typical range 100-3000
    edge_score = min(lap_var / 20, 100)
    results["edge_natural"] = round(edge_score, 1)

    # --- 5. Color Consistency (check lighting direction via face halves) ---
    left_half = bgr_img[:, :w // 2]
    right_half = bgr_img[:, w // 2:]
    left_mean = np.mean(left_half, axis=(0, 1))
    right_mean = np.mean(right_half, axis=(0, 1))
    # Compare color channel ratios between halves
    color_diff = np.abs(left_mean - right_mean)
    # Normalize: small diff = consistent lighting
    color_consistency = max(0, 100 - np.mean(color_diff) * 2)
    results["color_consistency"] = round(color_consistency, 1)

    return results

# ── HTML UI ───────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Deepfake Detector</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0a0a0a;
      color: #f0f0f0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 40px 20px;
    }
    h1 { font-size: 2.2rem; margin-bottom: 4px; letter-spacing: 1px; }
    p.sub { color: #666; margin-bottom: 32px; font-size: 0.95rem; }
    .card {
      background: #141414;
      border: 1px solid #222;
      border-radius: 20px;
      padding: 32px;
      width: 100%;
      max-width: 500px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
    }
    .drop-zone {
      width: 100%;
      height: 220px;
      border: 2px dashed #333;
      border-radius: 14px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.25s;
      overflow: hidden;
      position: relative;
    }
    .drop-zone:hover { border-color: #555; background: #1a1a1a; }
    .drop-zone.drag-over { border-color: #fff; background: #1a1a1a; }
    .drop-zone img { max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 12px; }
    .drop-zone .icon { font-size: 2rem; margin-bottom: 8px; opacity: 0.3; }
    .drop-zone span { color: #555; font-size: 0.85rem; }
    input[type="file"] { display: none; }

    .btn-row { display: flex; gap: 10px; width: 100%; }
    button {
      flex: 1;
      padding: 14px;
      background: #fff;
      color: #000;
      border: none;
      border-radius: 10px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    button:hover { opacity: 0.85; transform: translateY(-1px); }
    button:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
    .btn-secondary { background: #222; color: #aaa; }
    .btn-secondary:hover { background: #2a2a2a; opacity: 1; }

    .result {
      width: 100%;
      padding: 20px;
      border-radius: 14px;
      text-align: center;
      display: none;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .result.realism { background: #0a2618; border: 1px solid #1a4a32; }
    .result.deepfake { background: #2a0a0a; border: 1px solid #4a1a1a; }
    .verdict { font-size: 1.2rem; font-weight: 700; margin-bottom: 12px; }
    .result.realism .verdict { color: #4caf82; }
    .result.deepfake .verdict { color: #e05c5c; }

    .conf-bar-wrap {
      width: 100%;
      height: 8px;
      background: #1a1a1a;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 8px;
    }
    .conf-bar {
      height: 100%;
      border-radius: 4px;
      transition: width 0.5s ease;
    }
    .result.realism .conf-bar { background: linear-gradient(90deg, #2d7a53, #4caf82); }
    .result.deepfake .conf-bar { background: linear-gradient(90deg, #8a2c2c, #e05c5c); }
    .conf-text { font-size: 0.8rem; color: #888; }

    .analysis { width: 100%; margin-top: 4px; display: none; }
    .analysis h4 { font-size: 0.75rem; color: #555; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; }
    .metric {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    .metric-label {
      font-size: 0.8rem;
      color: #999;
      width: 130px;
      flex-shrink: 0;
    }
    .metric-bar-wrap {
      flex: 1;
      height: 6px;
      background: #1a1a1a;
      border-radius: 3px;
      overflow: hidden;
    }
    .metric-bar {
      height: 100%;
      border-radius: 3px;
      transition: width 0.6s ease;
    }
    .metric-bar.good { background: linear-gradient(90deg, #1a5c3a, #4caf82); }
    .metric-bar.warn { background: linear-gradient(90deg, #5c4a1a, #cfaa3e); }
    .metric-bar.bad  { background: linear-gradient(90deg, #5c1a1a, #e05c5c); }
    .metric-val {
      font-size: 0.75rem;
      color: #666;
      width: 40px;
      text-align: right;
      flex-shrink: 0;
    }
    .metric-desc {
      font-size: 0.7rem;
      color: #444;
      margin: -6px 0 10px 140px;
    }

    .history { width: 100%; max-width: 500px; margin-top: 24px; }
    .history h3 { font-size: 0.85rem; color: #444; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; }
    .history-list { display: flex; flex-direction: column; gap: 8px; }
    .history-item {
      display: flex;
      align-items: center;
      gap: 12px;
      background: #141414;
      border: 1px solid #222;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 0.85rem;
      animation: fadeIn 0.3s ease;
    }
    .history-item img { width: 36px; height: 36px; border-radius: 6px; object-fit: cover; }
    .history-item .name { flex: 1; color: #aaa; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .history-item .badge {
      padding: 3px 10px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .badge.realism { background: #0a2618; color: #4caf82; }
    .badge.deepfake { background: #2a0a0a; color: #e05c5c; }
  </style>
</head>
<body>
  <h1>Deepfake Detector</h1>
  <p class="sub">Upload a face image to check if it's real or AI-generated</p>
  <div class="card">
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">+</div>
      <span>Click or drag & drop an image</span>
    </div>
    <input type="file" id="fileInput" accept="image/*" onchange="previewFile(event)" />
    <div class="btn-row">
      <button class="btn-secondary" id="clearBtn" onclick="clearImage()" disabled>Clear</button>
      <button id="analyzeBtn" onclick="analyze()" disabled>Analyze</button>
    </div>
    <div class="result" id="result">
      <div class="verdict" id="verdict"></div>
      <div class="conf-bar-wrap"><div class="conf-bar" id="confBar"></div></div>
      <div class="conf-text" id="confidence"></div>
      <div class="analysis" id="analysis">
        <h4>Uncanny Valley Analysis</h4>
        <div id="metrics"></div>
      </div>
    </div>
  </div>

  <div class="history" id="historySection" style="display:none">
    <h3>History</h3>
    <div class="history-list" id="historyList"></div>
  </div>

  <script>
    let selectedFile = null;
    let thumbData = null;

    const dropZone = document.getElementById('dropZone');

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) loadFile(file);
    });

    function previewFile(e) { loadFile(e.target.files[0]); }

    function loadFile(file) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = ev => {
        thumbData = ev.target.result;
        dropZone.innerHTML = '<img src="' + thumbData + '" />';
      };
      reader.readAsDataURL(file);
      document.getElementById('analyzeBtn').disabled = false;
      document.getElementById('clearBtn').disabled = false;
      document.getElementById('result').style.display = 'none';
    }

    function clearImage() {
      selectedFile = null;
      thumbData = null;
      dropZone.innerHTML = '<div class="icon">+</div><span>Click or drag & drop an image</span>';
      document.getElementById('analyzeBtn').disabled = true;
      document.getElementById('clearBtn').disabled = true;
      document.getElementById('result').style.display = 'none';
      document.getElementById('fileInput').value = '';
    }

    async function analyze() {
      if (!selectedFile) return;
      const btn = document.getElementById('analyzeBtn');
      btn.disabled = true;
      btn.textContent = 'Analyzing...';

      const form = new FormData();
      form.append('image', selectedFile);

      try {
        const res = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json();

        const resultEl = document.getElementById('result');
        const cls = data.label.toLowerCase();
        resultEl.className = 'result ' + cls;

        document.getElementById('verdict').textContent =
          cls === 'realism' ? 'REAL IMAGE' : 'DEEPFAKE DETECTED';

        const bar = document.getElementById('confBar');
        bar.style.width = '0%';
        setTimeout(() => { bar.style.width = data.confidence + '%'; }, 50);

        document.getElementById('confidence').textContent = data.confidence + '% confidence';
        resultEl.style.display = 'block';

        // Render uncanny valley metrics
        renderMetrics(data.uncanny);

        addHistory(selectedFile.name, cls, data.confidence, thumbData);
      } catch (err) {
        alert('Analysis failed: ' + err.message);
      }

      btn.disabled = false;
      btn.textContent = 'Analyze';
    }

    const metricInfo = {
      symmetry:          { label: 'Facial Symmetry',     desc: 'How balanced left vs right face features are' },
      eye_consistency:   { label: 'Eye Reflections',     desc: 'Whether both eyes reflect light consistently' },
      texture:           { label: 'Skin Texture',        desc: 'Presence of natural micro-texture (FFT analysis)' },
      edge_natural:      { label: 'Edge Naturalness',    desc: 'Quality of edges around facial boundaries' },
      color_consistency: { label: 'Lighting Consistency', desc: 'Whether lighting is uniform across the face' },
    };

    function renderMetrics(uncanny) {
      const container = document.getElementById('metrics');
      const panel = document.getElementById('analysis');
      container.innerHTML = '';
      if (!uncanny) { panel.style.display = 'none'; return; }
      panel.style.display = 'block';

      for (const [key, info] of Object.entries(metricInfo)) {
        const val = uncanny[key];
        if (val === null || val === undefined) continue;

        const grade = val >= 70 ? 'good' : val >= 40 ? 'warn' : 'bad';
        const row = document.createElement('div');
        row.innerHTML =
          '<div class="metric">' +
            '<span class="metric-label">' + info.label + '</span>' +
            '<div class="metric-bar-wrap"><div class="metric-bar ' + grade + '" style="width:0%"></div></div>' +
            '<span class="metric-val">' + val + '%</span>' +
          '</div>' +
          '<div class="metric-desc">' + info.desc + '</div>';
        container.appendChild(row);

        // Animate bar
        setTimeout(() => {
          row.querySelector('.metric-bar').style.width = val + '%';
        }, 100);
      }
    }

    function addHistory(name, cls, conf, thumb) {
      const section = document.getElementById('historySection');
      section.style.display = 'block';
      const list = document.getElementById('historyList');
      const item = document.createElement('div');
      item.className = 'history-item';
      item.innerHTML =
        '<img src="' + thumb + '" />' +
        '<span class="name">' + name + '</span>' +
        '<span class="badge ' + cls + '">' +
          (cls === 'realism' ? 'Real' : 'Fake') + ' ' + conf + '%</span>';
      list.prepend(item);
    }
  </script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    arr = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "Could not decode image"}), 400
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probs).item())

    label = model.config.id2label[predicted_idx]
    confidence = round(float(probs[predicted_idx].item()) * 100, 1)

    # Uncanny valley analysis
    uncanny = analyze_uncanny(bgr)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "uncanny": uncanny,
    })

if __name__ == "__main__":
    app.run(debug=True)
