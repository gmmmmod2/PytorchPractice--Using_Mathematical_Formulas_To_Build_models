/* ====== 极简稳健版 app.js（只依赖 manifest.json）====== */
"use strict";

const CATEGORY_LABELS = {
  ABC: "入门",
  easy: "简单",
  middling: "中等",
  hard: "困难",
};
const CATEGORY_ORDER = ["ABC", "easy", "middling", "hard"];
const CONTENT_BASE = "content"; // 相对路径，适配 GitHub Pages 子路径
const DEBUG = true;

const state = {
  loaded: false,
  items: { ABC: [], easy: [], middling: [], hard: [] },
  active: null, // {category, name, url}
};

// ---------- 小工具 ----------
const $ = (sel, root = document) => root.querySelector(sel);
const createEl = (tag, cls) => { const el = document.createElement(tag); if (cls) el.className = cls; return el; };
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const extIsMd = (name) => /\.md$/i.test(name);
const log = (...a) => DEBUG && console.log("[题库]", ...a);
const warn = (...a) => DEBUG && console.warn("[题库]", ...a);
const error = (...a) => console.error("[题库]", ...a);

// 计算“当前目录下某文件”的可直接打开链接（供调试展示）
function makeDebugUrl(relPath) {
  // 把当前页面路径替换为“目录路径”，再拼相对文件路径
  const baseDir = location.pathname.replace(/\/[^/]*$/, "/");
  return location.origin + baseDir + relPath.replace(/^.\//, "");
}

// ---------- 右侧状态渲染 ----------
function renderStatusHTML(title, lines, kind) {
  const colors = { info: "#a0a0a0", warn: "#e6c07b", error: "#ff8a8a" };
  const color = colors[kind || "info"];
  const items = (lines || []).map(x => `<li style="margin:6px 0">${x}</li>`).join("");
  const listHtml = items ? `<ul style="margin:8px 0 0 16px;">${items}</ul>` : "";
  $("#article").innerHTML =
    `<div class="empty" style="color:${color}">
       <h2 style="margin:0 0 8px 0;">${title}</h2>
       ${listHtml}
     </div>`;
}

function setLoading() {
  $("#breadcrumb").textContent = "正在读取题目清单（manifest.json）…";
  renderStatusHTML("正在加载…", ["请稍候。如果超过 3 秒仍无结果，会显示错误信息以便排查。"], "info");
}

// ---------- Markdown 渲染 ----------
function preprocessMathBlocks(md) {
  // 统一换行
  const s = (md || "").replace(/\r\n?/g, "\n");

  // 规则：行首 $$、中间任意内容（可跨多行）、行尾 $$
  // 用 div.math-block 包住内容，后续用 katex.render() 渲染
  return s.replace(/(^|\n)\$\$([\s\S]*?)\$\$(\n|$)/g, (m, pre, body, post) => {
    return `${pre}<div class="math-block">\n${body.trim()}\n</div>${post}`;
  });
}

function renderMarkdown(mdText) {
  // 先把 $$...$$ 块替换成占位容器，避免被 Marked 拆段
  const preprocessed = preprocessMathBlocks(mdText);

  // 正常 Markdown -> HTML
  marked.setOptions({ breaks: true, gfm: true, mangle: false, headerIds: true });
  $("#article").innerHTML = marked.parse(preprocessed || "");

  // 代码高亮
  document.querySelectorAll("pre code").forEach(block => {
    try { hljs.highlightElement(block); } catch(e){}
  });

  // 先渲染块级数学（我们自己包的 .math-block）
  try {
    document.querySelectorAll("#article .math-block").forEach(el => {
      const tex = el.textContent.trim();
      if (tex) {
        katex.render(tex, el, { displayMode: true, throwOnError: false });
      }
    });
  } catch (e) {
    console.warn("KaTeX block render failed:", e);
  }

  // 再渲染行内数学（交给 KaTeX auto-render）
  try {
    if (window.renderMathInElement) {
      window.renderMathInElement($("#article"), {
        delimiters: [
          // 注意：这里不再包含 $$...$$，因为上面已处理
          { left: "$",  right: "$",  display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true }  // 可选：支持 \[...\] 行间
        ],
        throwOnError: false,
        ignoredTags: ["script","noscript","style","textarea","pre","code"]
      });
    }
  } catch (e) {
    console.warn("KaTeX inline render failed:", e);
  }
}

// ---------- 工具栏 & 目录 ----------
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
  bc.textContent = CATEGORY_LABELS[category] + " / " + name;
  openRawBtn.disabled = false;
  openRawBtn.onclick = () => window.open(url, "_blank");
  copyLinkBtn.disabled = false;
  copyLinkBtn.onclick = async () => {
    const hash = "#/" + encodeURIComponent(category) + "/" + encodeURIComponent(name);
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
  li.innerHTML = "📄 <span class=\"title\">" + item.name + "</span>";
  li.onclick = () => openItem(category, item.name);
  return li;
}

function renderTOC() {
  const toc = $("#toc");
  toc.innerHTML = "";
  CATEGORY_ORDER.forEach(cat => {
    const section = createEl("div", "section");
    const header = createEl("div", "section-header");
    header.innerHTML =
      "<h3>" + CATEGORY_LABELS[cat] + "</h3>" +
      "<span class=\"section-count\">" + state.items[cat].length + "</span>" +
      "<span class=\"chev\">▾</span>";
    const list = createEl("div", "section-list");
    state.items[cat]
      .slice()
      .sort((a,b) => a.name.localeCompare(b.name, "zh-Hans-CN"))
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
      const title = (li.querySelector(".title")?.textContent || "").toLowerCase();
      li.style.display = title.includes(q) ? "" : "none";
    });
  });
}

// ---------- 打开题目 ----------
async function openItem(category, name) {
  const entry = state.items[category].find(it => it.name === name);
  if (!entry) return;
  let text = "";
  try {
    const res = await fetch(entry.url, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status + "：" + entry.url);
    text = await res.text();
  } catch (e) {
    const dbg = makeDebugUrl(entry.url);
    renderStatusHTML("无法加载题目", [
      "请求：" + `<code>${entry.url}</code>`,
      "直达（复制到地址栏）：" + `<code>${dbg}</code>`,
      "错误：" + `<code>${String(e)}</code>`,
      "检查：文件是否存在？文件名大小写是否一致？"
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
      "需要 <code>content/manifest.json</code> 列出各目录下的 .md 文件。"
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
  const parts = location.hash.split("/");
  const cat = decodeURIComponent(parts[1] || "");
  const name = decodeURIComponent(parts[2] || "");
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

// ---------- 只走 manifest 的加载 ----------
async function loadFromManifestOnly() {
  const url = CONTENT_BASE + "/manifest.json?ts=" + Date.now();
  const debugUrl = makeDebugUrl(CONTENT_BASE + "/manifest.json");

  log("尝试读取 manifest：", url, "（可直接访问调试）=>", debugUrl);

  try {
    const res = await fetch(url, { cache: "no-store" });
    log("manifest 响应状态：", res.status, res.statusText);
    if (!res.ok) throw new Error("HTTP " + res.status + "：" + url);

    const manifest = await res.json();
    log("manifest 内容：", manifest);

    const next = { ABC: [], easy: [], middling: [], hard: [] };
    CATEGORY_ORDER.forEach(cat => {
      const arr = Array.isArray(manifest[cat]) ? manifest[cat] : [];
      arr.filter(n => extIsMd(n)).forEach(filename => {
        next[cat].push({
          name: filename.replace(/\.md$/i, ""),
          url: CONTENT_BASE + "/" + cat + "/" + encodeURIComponent(filename)
        });
      });
    });

    const total = Object.values(next).reduce((s, a) => s + a.length, 0);
    if (total === 0) {
      const tip = "manifest 已读取，但未包含任何 .md；请检查文件名是否正确（注意大小写）。";
      renderStatusHTML("清单为空", [tip, "调试直达：" + `<code>${debugUrl}</code>`], "warn");
    }

    state.items = next;
    state.loaded = true;
    renderTOC();
    updateCounts();
    $("#breadcrumb").textContent = "请选择左侧题目";

    // 成功后替换掉“正在加载...”提示
    renderStatusHTML("欢迎使用题库 🎉", [
      "从左侧选择一个题目即可开始阅读；",
      "或者点击上方的 <strong>随机一题 🎲</strong> 按钮试试运气！"
    ], "info");
    return true;

  } catch (e) {
    renderStatusHTML("没有加载到题目", [
      "未找到或无法读取：" + `<code>${CONTENT_BASE}/manifest.json</code>`,
      "可在地址栏直接访问验证：" + `<code>${debugUrl}</code>`,
      "若在 GitHub Pages：确认文件已随站点一起发布，并放在与 <code>index.html</code> 同级的 <code>content/</code> 下。",
      "确认路径为相对路径（不要以 <code>/content</code> 开头）。",
      "必要时在仓库发布根放置一个空的 <code>.nojekyll</code> 文件。"
    ], "error");
    error("读取 manifest 失败：", e);
    return false;
  }
}

function updateCounts() {
  document.querySelectorAll(".section").forEach((sec, i) => {
    const cat = CATEGORY_ORDER[i];
    const count = state.items[cat]?.length || 0;
    const el = sec.querySelector(".section-count");
    if (el) el.textContent = String(count);
  });
}

// ---------- 全局错误可视化（兜底）----------
window.addEventListener("error", (ev) => {
  const msg = String(ev.message || "");
  const src = String(ev.filename || "");
  const loc = (ev.lineno || 0) + ":" + (ev.colno || 0);
  renderStatusHTML("脚本运行错误", [
    "消息：" + `<code>${msg}</code>`,
    "源：" + `<code>${src}:${loc}</code>`,
    "打开浏览器控制台（F12）→ Console 查看详细堆栈。"
  ], "error");
});
window.addEventListener("unhandledrejection", (ev) => {
  renderStatusHTML("未处理的 Promise 拒绝", [
    "原因：" + `<code>${String(ev.reason)}</code>`,
    "打开浏览器控制台（F12）→ Console 查看详细堆栈。"
  ], "error");
});

// ---------- 初始化 ----------
async function init() {
  setupSearch();
  const randomBtn = $("#randomBtn");
  if (randomBtn) randomBtn.addEventListener("click", pickRandom);

  setLoading();

  // 3s 超时兜底：如果仍未 loaded，就给出可见错误
  const timeoutId = setTimeout(() => {
    if (!state.loaded) {
      const dbg = makeDebugUrl(CONTENT_BASE + "/manifest.json");
      renderStatusHTML("加载超时", [
        "3 秒内没有成功读取题目清单。",
        "请直接访问：" + `<code>${dbg}</code>`,
        "若 404：请确认 manifest.json 已发布、路径大小写正确、与 index.html 同级的 content/ 下。",
        "若 200：按 F12 → Network 查看是否有其他报错。"
      ], "error");
      warn("3 秒超时兜底触发");
    }
  }, 3000);

  await loadFromManifestOnly();
  clearTimeout(timeoutId);

  renderTOC();
  updateCounts();
  tryOpenFromHash();

  const firstSec = $(".section");
  if (firstSec) firstSec.classList.add("open");
}

document.addEventListener("DOMContentLoaded", init);
