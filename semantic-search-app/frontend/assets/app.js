// ---------------------------------------------------------------------
// Nearest — frontend logic
// ---------------------------------------------------------------------

const API = {
  status: "/api/status",
  search: "/api/search",
  upload: "/api/upload",
};

const els = {
  statusPill: document.getElementById("statusPill"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  searchForm: document.getElementById("searchForm"),
  queryInput: document.getElementById("queryInput"),
  topKSelect: document.getElementById("topKSelect"),
  searchBtn: document.getElementById("searchBtn"),
  heroMeta: document.getElementById("heroMeta"),
  resultsGrid: document.getElementById("resultsGrid"),
  resultsHead: document.getElementById("resultsHead"),
  resultsTiming: document.getElementById("resultsTiming"),
  emptyState: document.getElementById("emptyState"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  dzTitle: document.getElementById("dzTitle"),
  dzSub: document.getElementById("dzSub"),
  uploadStatus: document.getElementById("uploadStatus"),
};

let modelReady = false;

// ----------------------------- Status ----------------------------- //

async function refreshStatus() {
  setStatus("busy", "Checking model…");
  try {
    const res = await fetch(API.status);
    const data = await res.json();
    modelReady = data.ready;
    if (data.ready) {
      setStatus("ready", `Ready · ${data.num_documents} documents`);
    } else {
      setStatus("error", "No dataset loaded");
    }
  } catch (e) {
    setStatus("error", "Backend unreachable");
  }
}

function setStatus(kind, text) {
  els.statusPill.className = "status-pill " + kind;
  els.statusText.textContent = text;
}

// ----------------------------- Search ----------------------------- //

els.searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = els.queryInput.value.trim();
  if (!query) return;

  if (!modelReady) {
    els.heroMeta.textContent = "No dataset loaded yet — upload a CSV below first.";
    els.heroMeta.classList.add("error");
    return;
  }

  els.heroMeta.classList.remove("error");
  els.heroMeta.textContent = "Embedding your query…";
  els.searchBtn.disabled = true;
  els.searchBtn.textContent = "Searching…";

  try {
    const res = await fetch(API.search, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: parseInt(els.topKSelect.value, 10),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Search failed");
    }

    const data = await res.json();
    renderResults(data);
    pulseSpace(data.results.length);
  } catch (err) {
    els.heroMeta.textContent = err.message;
    els.heroMeta.classList.add("error");
  } finally {
    els.searchBtn.disabled = false;
    els.searchBtn.textContent = "Search";
  }
});

function renderResults(data) {
  els.heroMeta.textContent = "";
  els.emptyState.hidden = true;
  els.resultsHead.hidden = false;
  els.resultsTiming.textContent = `${data.results.length} results · ${data.took_ms} ms`;

  els.resultsGrid.innerHTML = "";
  data.results.forEach((r, i) => {
    const pct = Math.max(0, Math.min(100, r.score * 100));
    const card = document.createElement("article");
    card.className = "result-card";
    card.style.animationDelay = `${i * 60}ms`;

    const keywords = (r.keywords || "")
      .split(",")
      .map((k) => k.trim())
      .filter(Boolean)
      .slice(0, 6)
      .map((k) => `<span class="kw-chip">${escapeHtml(k)}</span>`)
      .join("");

    card.innerHTML = `
      <div class="result-rank">${String(i + 1).padStart(2, "0")}</div>
      <div class="result-body">
        <h3>${escapeHtml(r.title || "Untitled")}</h3>
        <p>${escapeHtml(truncate(r.content || "", 220))}</p>
        <div class="result-keywords">${keywords}</div>
      </div>
      <div class="result-score">
        <span class="score-value">${pct.toFixed(1)}%</span>
        <div class="score-bar"><div class="score-bar-fill"></div></div>
      </div>
    `;
    els.resultsGrid.appendChild(card);

    requestAnimationFrame(() => {
      const fill = card.querySelector(".score-bar-fill");
      requestAnimationFrame(() => (fill.style.width = pct + "%"));
    });
  });
}

function truncate(text, n) {
  return text.length > n ? text.slice(0, n).trim() + "…" : text;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ----------------------------- Upload ----------------------------- //

els.dropzone.addEventListener("click", () => els.fileInput.click());

["dragenter", "dragover"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("drag-over");
  })
);
els.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleUpload(file);
});
els.fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleUpload(file);
});

async function handleUpload(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    setUploadStatus("error", "Please choose a .csv file.");
    return;
  }

  els.dzTitle.textContent = `Training on ${file.name}…`;
  els.dzSub.textContent = "This can take a few seconds for larger datasets";
  setUploadStatus("", "Uploading and training Word2Vec model…");
  setStatus("busy", "Training…");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(API.upload, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Upload failed");

    setUploadStatus(
      "success",
      `Trained on ${data.num_documents} documents · vocab ${data.vocab_size} words`
    );
    els.dzTitle.textContent = "Drop a CSV here, or click to browse";
    els.dzSub.textContent = ".csv — Title, Content, Keywords columns";
    modelReady = true;
    setStatus("ready", `Ready · ${data.num_documents} documents`);
    els.emptyState.textContent = "";
    els.heroMeta.classList.remove("error");
    els.heroMeta.textContent = "Dataset loaded — try a search above.";
  } catch (err) {
    setUploadStatus("error", err.message);
    els.dzTitle.textContent = "Drop a CSV here, or click to browse";
    els.dzSub.textContent = ".csv — Title, Content, Keywords columns";
    refreshStatus();
  }
}

function setUploadStatus(kind, text) {
  els.uploadStatus.className = "upload-status " + kind;
  els.uploadStatus.textContent = text;
}

// ------------------------- Embedding space canvas ------------------------- //

const canvas = document.getElementById("spaceCanvas");
const ctx = canvas.getContext("2d");
let nodes = [];
let dpr = window.devicePixelRatio || 1;

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  initNodes(rect.width, rect.height);
}

function initNodes(w, h) {
  const count = Math.max(18, Math.floor((w * h) / 9000));
  nodes = Array.from({ length: count }, () => ({
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * 0.18,
    vy: (Math.random() - 0.5) * 0.18,
    r: 1.6 + Math.random() * 1.8,
    glow: 0,
  }));
}

function pulseSpace(k) {
  const w = canvas.clientWidth,
    h = canvas.clientHeight;
  const cx = w / 2,
    cy = h / 2;

  // pick k random nodes to represent "results"
  const chosen = [...nodes].sort(() => Math.random() - 0.5).slice(0, Math.min(k, nodes.length));
  chosen.forEach((n, i) => {
    setTimeout(() => {
      n.glow = 1;
      n.pulseFrom = { x: cx, y: cy };
    }, i * 90);
  });
}

function drawSpace() {
  const w = canvas.clientWidth,
    h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  // connections
  ctx.lineWidth = 1;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i],
        b = nodes[j];
      const dx = a.x - b.x,
        dy = a.y - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 90) {
        ctx.strokeStyle = `rgba(124,108,255,${0.08 * (1 - dist / 90)})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
  }

  // nodes
  nodes.forEach((n) => {
    n.x += n.vx;
    n.y += n.vy;
    if (n.x < 0 || n.x > w) n.vx *= -1;
    if (n.y < 0 || n.y > h) n.vy *= -1;

    if (n.glow > 0) {
      ctx.beginPath();
      const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, 14 * n.glow + n.r);
      grad.addColorStop(0, `rgba(255,184,107,${0.55 * n.glow})`);
      grad.addColorStop(1, "rgba(255,184,107,0)");
      ctx.fillStyle = grad;
      ctx.arc(n.x, n.y, 14 * n.glow + n.r, 0, Math.PI * 2);
      ctx.fill();
      n.glow *= 0.985;
      if (n.glow < 0.02) n.glow = 0;
    }

    ctx.beginPath();
    ctx.fillStyle = n.glow > 0.05 ? "#FFB86B" : "rgba(231,233,245,0.55)";
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fill();
  });

  requestAnimationFrame(drawSpace);
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
requestAnimationFrame(drawSpace);

// ambient pulses so the space feels alive before the first search
setInterval(() => {
  if (nodes.length) pulseSpace(1 + Math.floor(Math.random() * 2));
}, 3200);

// ----------------------------- Init ----------------------------- //

refreshStatus();
