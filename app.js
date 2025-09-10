/**********************
 *  强韧版 app.js
 *  - 只依赖 manifest.json
 *  - 任何失败都在右侧可见提示
 *  - 3s 内没加载成功 => 超时兜底错误
 *  - 详细 console 日志 & 浏览器直达调试链接
 **********************/

const CATEGORY_LABELS = {
  ABC: "入门（ABC）",
  easy: "简单（easy）",
  middling: "中等困难（middling）",
  hard: "困难（hard）",
};
const CATEGORY_ORDER = ["ABC", "easy", "middling", "hard"];
const CONTENT_BASE = "content"; // 相对路径，适配 GitHub Pages 子路径
const DEBUG = true;

const state = {
  loaded: false,
  items: { ABC: [], easy: [], middling: [], hard: [] },
  active: null, // {category, name, url}
};

// ------------- 小工具 -------------
const $ = (sel, root = document) => root.querySelector(sel);
const createEl = (tag, cls) => { const el = document.createElement(tag); if (cls) el.className = cls; return el; };
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const extIsMd = (name) => /\.md$/i.test(name);

function log(...args){ if (DEBUG) console.log("[题库]", ...args); }
function warn(...args){ if (DEBUG) console.warn("[题库]", ...args); }
function error(...args){ console.error("[题库]", ...args); }

// ------------- 右侧状态渲染（可见）-------------
function renderStatusHTML(title, lines = [], kind = "info"){
  const colors = {
    info:   "#a0a0a0",
    warn:   "#e6c07b",
    error:  "#ff8a8a",
  };
  const color = colors[kind] || colors.info;
  $("#article").innerHTML = `
    <div class="empty" style="color:${color}">
      <h2 style="margin:0 0 8px 0;">${title}</h2>
      ${lines.length ? `<ul style="margin:8px 0 0 16px;">${
        lines.map(x => `<li style="margin:6px 0">${x}</li>`).join("")
      }</ul>` : ""}
    </div>
  `;
}

function setLoading() {
  $("#breadcrumb").textContent = "正在读取题目清单（manifest.json）…";
  renderStatusHTML("正在加载…", [
    "请稍候。如果超过 3 秒仍无结果，会显示错误信息以便排查。"
  ], "info");
}

// ------------- Markdown 渲染 -------------
function renderMarkdown(mdText) {
  marked.setOptions({ breaks: true, gfm: true, mangle: false, headerIds: true });
  $("#article").innerHTML = marked.parse(mdText || "");
  document.querySelectorAll('pre code').forEach(block => { try { hljs.highlightElement(block); } catch(e){} });
}

// ------------- 工具栏 & 目录 -------------
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
    li.classList.toggle("active", li.dataset.category === category && li.dataset.name === name);
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
    state.items[cat].sort((a,b) => a.name.localeCompare(b.name, 'zh-Hans-CN'))
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

// ------------- 打开题目 -------------
async function openItem(category, name) {
  const entry = state.items[category].find(it => it.name === name);
  if (!entry) return;
  let text = "";
  try {
    const res = await fetch(entry.url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}：${entry.url}`);
    text = await res.text();
  } catch (e) {
    const debugUrl = location.origin + location.pathname.replace(/\/[^/]*$/, "/") + entry.url;
    renderStatusHTML("无法加载题目", [
      `请求：<code>${entry.url}</code>`,
      `直达（复制到地址栏）：<code>${debugUrl}</code>`,
      `错误：<code>${String(e)}</code>`,
      "检查：文件是否真的存在？文件名大小写是否一致？"
    ], "error");
    error("加载题目失败：", e);
    return;
  }
  state.active = { category, name, url: entry.url };
  renderMarkdown(text);
  updateToolbar();
  highlightActive(category, name);
}

function pickRandom() {
  const all = CATEGORY_ORDER.flatMap(cat => state.items[cat].map(it => ({...it, category: cat})));
  if (!all.length) {
    renderStatusHTML("未发现可用题目", [
      `需要：<code>${CONTENT_BASE}/manifest.json</code> 列出各目录下的 .md 文件。`,
      "你可以用我之前给的脚本一键生成。"
    ], "warn");
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

// ------------- 只走 manifest 的加载 -------------
async function loadFromManifestOnly() {
  const url = `${CONTENT_BASE}/manifest.json?ts=${Date.now()}`;
  const debugUrl = location.origin + location.pathname.replace(/\/[^/]*$/, "/") + url;

  log("尝试读取 manifest：", url, "（可直接访问调试）=>", debugUrl);

  try {
    const res = await fetch(url, { cache: "no-store" });
    log("manifest 响应状态：", res.status, res.statusText);
    if (!res.ok) throw new Error(`HTTP ${res.status}：${url}`);

    const manifest = await res.json();
    log("manifest 内容：", manifest);

    const next = { ABC: [], easy: [], middling: [], hard: [] };
    for (const cat of CATEGORY_ORDER) {
      const arr = manifest[cat] || [];
      arr.filter(n => extIsMd(n)).forEach(filename => {
        next[cat].push({
          name: filename.replace(/\.md$/i, ""),
          url: `${CONTENT_BASE}/${cat}/${encodeURIComponent(filename)}`
        });
      });
    }

    const total = Object.values(next).reduce((s, a) => s + a.length, 0);
    if (total === 0) {
      renderStatusHTML("manifest 已读取，但未包含任何 .md", [
        "请检查 manifest.json 是否正确填入文件名（区分大小写）。",
        `调试直达：<code>${debugUrl}</code>`
      ], "warn");
    }

    state.items = next;
    state.loaded = true;
    renderTOC();
    updateCounts();
    $("#breadcrumb").textContent = "请选择左侧题目";
    return true;

  } catch (e) {
    error("读取 manifest 失败：", e);
    renderStatusHTML("没有加载到题目", [
      `未找到或无法读取：<code>${url}</code>`,
      `可在地址栏直接访问验证：<code>${debugUrl}</code>`,
      "若在 GitHub Pages：确认文件已随站点一起发布（同一分支/目录），并放置在与 index.html 同级的 content/ 下。",
      "确认路径为相对路径（不要以 /content 开头）。",
      "必要时在仓库发布根放置一个空的 <code>.nojekyll</code> 文件。"
    ], "error");
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

// ------------- 全局错误可视化（兜底）-------------
window.addEventListener("error", (ev) => {
  renderStatusHTML("脚本运行错误", [
    `消息：<code>${ev.message}</code>`,
    `源：<code>${ev.filename}:${ev.lineno}</code>`,
    "打开浏览器控制台（F12）→ Console 查看详细堆栈。"
  ], "error");
});
window.addEventListener("unhandledrejection", (ev) => {
  renderStatusHTML("未处理的 Promise 拒绝", [
    `原因：<code>${String(ev.reason)}</code>`,
    "打开浏览器控制台（F12）→ Console 查看详细堆栈。"
  ], "error");
});

// ------------- 初始化 -------------
async function init() {
  setupSearch();
  $("#randomBtn").addEventListener("click", pickRandom);

  setLoading();

  // 3s 超时兜底：如果仍未 loaded，就给出可见错误
  const timeout = setTimeout(() => {
    if (!state.loaded) {
      const url = `${CONTENT_BASE}/manifest.json`;
      const debugUrl = location.origin + location.pathname.replace(/\/[^/]*$/, "/") + url;
      renderStatusHTML("加载超时", [
        "3 秒内没有成功读取题目清单。",
        `请直接访问：<code>${debugUrl}</code> 看看是否能打开。`,
        "若 404：请确认 manifest.json 已发布、路径大小写正确、与 index.html 同级的 content/ 下。",
        "若 200：请按 F12 → Network 看看前端是否有其他报错。"
      ], "error");
      warn("3 秒超时兜底触发");
    }
  }, 3000);

  await loadFromManifestOnly();
  clearTimeout(timeout);

  renderTOC();
  updateCounts();
  tryOpenFromHash();

  const firstSec = $(".section");
  if (firstSec) firstSec.classList.add("open");
}

document.addEventListener("DOMContentLoaded", init);
