// ====================== 配置与状态 ======================
const CATEGORY_LABELS = {
  ABC: "入门（ABC）",
  easy: "简单（easy）",
  middling: "中等困难（middling）",
  hard: "困难（hard）",
};
const CATEGORY_ORDER = ["ABC", "easy", "middling", "hard"];
const CONTENT_BASE = "content"; // 与 index.html 同目录

const state = {
  loaded: false,
  items: {
    ABC: [],
    easy: [],
    middling: [],
    hard: [],
  },
  active: null, // {category, name, source: 'remote', url}
};

// ====================== DOM 工具 ======================
const $ = (sel, root = document) => root.querySelector(sel);
const createEl = (tag, cls) => {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  return el;
};
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const extIsMd = (name) => /\.md$/i.test(name);

// ====================== 渲染函数 ======================
function renderMarkdown(mdText) {
  marked.setOptions({ breaks: true, gfm: true, mangle: false, headerIds: true });
  const html = marked.parse(mdText || "");
  const article = $("#article");
  article.innerHTML = html;
  document.querySelectorAll('pre code').forEach(block => {
    try { hljs.highlightElement(block); } catch(e){}
  });
}

function updateToolbar() {
  const bc = $("#breadcrumb");
  const openRawBtn = $("#openRawBtn");
  const copyLinkBtn = $("#copyLinkBtn");
  if (!state.active) {
    bc.textContent = "请选择左侧题目";
    openRawBtn.disabled = true;
    copyLinkBtn.disabled = true;
    return;
  }
  const { category, name, url } = state.active;
  bc.textContent = `${CATEGORY_LABELS[category]} / ${name}`;
  openRawBtn.disabled = false;
  openRawBtn.onclick = () => window.open(url, "_blank");
  copyLinkBtn.disabled = false;
  copyLinkBtn.onclick = async () => {
    const hash = `#/${encodeURIComponent(category)}/${encodeURIComponent(name)}`;
    history.replaceState(null, "", hash);
    await navigator.clipboard.writeText(location.href);
    copyLinkBtn.textContent = "已复制 ✔";
    await sleep(1200);
    copyLinkBtn.textContent = "🔗";
  };
}

function highlightActive(category, name) {
  document.querySelectorAll(".section-list .item").forEach(li => {
    li.classList.toggle("active",
      li.dataset.category === category && li.dataset.name === name
    );
  });
}

function buildItemLi(category, item) {
  const li = createEl("div", "item");
  li.dataset.category = category;
  li.dataset.name = item.name;
  li.innerHTML = `📄 <span class="title">${item.name}</span>`;
  li.onclick = () => openItem(category, item.name);
  return li;
}

function renderTOC() {
  const toc = $("#toc");
  toc.innerHTML = "";
  CATEGORY_ORDER.forEach(cat => {
    const section = createEl("div", "section");
    const header = createEl("div", "section-header");
    header.innerHTML = `
      <h3>${CATEGORY_LABELS[cat]}</h3>
      <span class="section-count">${state.items[cat].length}</span>
      <span class="chev">▾</span>
    `;
    const list = createEl("div", "section-list");
    state.items[cat]
      .sort((a,b) => a.name.localeCompare(b.name, 'zh-Hans-CN'))
      .forEach(item => list.appendChild(buildItemLi(cat, item)));

    header.onclick = () => section.classList.toggle("open");
    section.appendChild(header);
    section.appendChild(list);
    if (state.items[cat].length) section.classList.add("open");
    toc.appendChild(section);
  });
}

function setupSearch() {
  const input = $("#searchInput");
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    document.querySelectorAll(".section-list .item").forEach(li => {
      const title = li.querySelector(".title")?.textContent?.toLowerCase() || "";
      li.style.display = title.includes(q) ? "" : "none";
    });
  });
}

// ====================== 打开题目 ======================
async function openItem(category, name) {
  const entry = state.items[category].find(it => it.name === name);
  if (!entry) return;

  let text = "";
  try {
    const res = await fetch(entry.url);
    if (!res.ok) throw new Error(`加载失败: ${res.status}`);
    text = await res.text();
  } catch (e) {
    text = `> 无法加载该题目：${e.message}`;
  }

  state.active = { category, name, source: "remote", url: entry.url };
  renderMarkdown(text);
  updateToolbar();
  highlightActive(category, name);
}

function pickRandom() {
  const all = CATEGORY_ORDER.flatMap(cat => state.items[cat].map(it => ({...it, category: cat})));
  if (!all.length) {
    alert("未发现可用题目，请确认服务器允许目录索引或提供 content/manifest.json。");
    return;
  }
  const idx = Math.floor(Math.random() * all.length);
  const { category, name } = all[idx];
  openItem(category, name);
  document.querySelectorAll(".section").forEach(sec => {
    const h3 = sec.querySelector("h3")?.textContent || "";
    if (h3.includes(CATEGORY_LABELS[category])) sec.classList.add("open");
  });
}

function tryOpenFromHash() {
  if (!location.hash.startsWith("#/")) return;
  const [, catEnc, nameEnc] = location.hash.split("/");
  const cat = decodeURIComponent(catEnc || "");
  const name = decodeURIComponent(nameEnc || "");
  if (CATEGORY_ORDER.includes(cat) && name) {
    let tries = 0;
    const t = setInterval(() => {
      tries++;
      const has = state.items[cat]?.some(it => it.name === name);
      if (has) { clearInterval(t); openItem(cat, name); }
      if (tries > 40) clearInterval(t);
    }, 200);
  }
}

// ====================== 自动扫描 /content/ ======================
// 策略：manifest.json -> 目录索引解析 -> 失败提示
async function loadFromContent() {
  const hasManifest = await tryLoadManifest();
  if (hasManifest) return true;

  const viaIndex = await tryLoadByDirectoryIndex();
  if (viaIndex) return true;

  // 失败：显示提示
  $("#article").innerHTML = `
    <div class="empty">
      <h2>没有找到题目</h2>
      <p>请确认以下任意条件：</p>
      <ol>
        <li>在 <code>/content/</code> 下提供 <code>manifest.json</code>，格式：
          <pre><code>{
  "ABC": ["题目A.md", "题目B.md"],
  "easy": [],
  "middling": [],
  "hard": []
}</code></pre>
        </li>
        <li>或启用目录索引（Nginx/Apache/Vite/Live Server）以便页面能解析出 <code>.md</code> 文件。</li>
      </ol>
    </div>`;
  return false;
}

// 方案一：manifest.json
async function tryLoadManifest() {
  try {
    const res = await fetch(`${CONTENT_BASE}/manifest.json?ts=${Date.now()}`);
    if (!res.ok) return false;
    const manifest = await res.json();
    const next = { ABC: [], easy: [], middling: [], hard: [] };
    for (const cat of CATEGORY_ORDER) {
      (manifest[cat] || []).filter(n => extIsMd(n)).forEach(filename => {
        next[cat].push({
          name: filename.replace(/\.md$/i, ""),
          url: `${CONTENT_BASE}/${cat}/${encodeURIComponent(filename)}`
        });
      });
    }
    state.items = next;
    state.loaded = true;
    renderTOC();
    updateCounts();
    return true;
  } catch {
    return false;
  }
}

// 方案二：解析目录索引（HTML）
async function tryLoadByDirectoryIndex() {
  const next = { ABC: [], easy: [], middling: [], hard: [] };
  let anyFound = false;

  for (const cat of CATEGORY_ORDER) {
    const ok = await parseIndexForCategory(cat, next);
    anyFound = anyFound || ok;
  }
  if (!anyFound) return false;

  state.items = next;
  state.loaded = true;
  renderTOC();
  updateCounts();
  return true;
}

async function parseIndexForCategory(cat, next) {
  try {
    // 请求目录本身，例如 /content/ABC/ —— 若服务器开启索引，会返回一个包含链接的 HTML
    const url = `${CONTENT_BASE}/${cat}/`;
    const res = await fetch(url, { method: "GET" });
    if (!res.ok) return false;
    const text = await res.text();

    // 尝试从 HTML 中提取 .md 文件链接（兼容多种目录索引格式）
    const parser = new DOMParser();
    const doc = parser.parseFromString(text, "text/html");
    const links = Array.from(doc.querySelectorAll("a"))
      .map(a => a.getAttribute("href"))
      .filter(Boolean);

    const mdNames = links
      .map(href => decodeURIComponent(href))
      .filter(href => extIsMd(href))
      .map(href => href.split("/").pop());

    mdNames.forEach(filename => {
      next[cat].push({
        name: filename.replace(/\.md$/i, ""),
        url: `${CONTENT_BASE}/${cat}/${encodeURIComponent(filename)}`
      });
    });

    // 兼容某些纯文本索引或无 <a> 的情况，做一次兜底正则
    if (!mdNames.length) {
      const rx = />([^<>"]+?\.md)</gi;
      let m; const found = [];
      while ((m = rx.exec(text))) found.push(m[1]);
      found.forEach(filename => {
        next[cat].push({
          name: filename.replace(/\.md$/i, ""),
          url: `${CONTENT_BASE}/${cat}/${encodeURIComponent(filename)}`
        });
      });
    }

    return next[cat].length > 0;
  } catch {
    return false;
  }
}

function updateCounts() {
  document.querySelectorAll(".section").forEach((sec, i) => {
    const cat = CATEGORY_ORDER[i];
    const count = state.items[cat]?.length || 0;
    sec.querySelector(".section-count")?.textContent = count;
  });
}

// ====================== 初始化 ======================
async function init() {
  setupSearch();
  $("#randomBtn").addEventListener("click", pickRandom);

  await loadFromContent();
  renderTOC();
  updateCounts();
  tryOpenFromHash();

  const firstSec = $(".section");
  if (firstSec) firstSec.classList.add("open");

  $("#breadcrumb").textContent = state.loaded
    ? "请选择左侧题目"
    : "未加载到题目，请检查 /content/ 配置";
}

document.addEventListener("DOMContentLoaded", init);
