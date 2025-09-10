// 单文件 JS：配置 + GitHub API + UI + 逻辑 + Giscus
(function () {
  'use strict';

  /* ================= 配置 ================= */
  const CONFIG = {
    github: {
      owner: "gmmmmod2",
      repo: "PytorchPractice--Using_Mathematical_Formulas_To_Build_models",
      branch: "main",
      basePath: "content",
    },
    categories: [
      { key: "rumen", label: "入门",   folder: "入门" },
      { key: "jichu", label: "基础",   folder: "基础" },
      { key: "zhongdeng", label: "中等", folder: "中等" },
      { key: "kunnan", label: "困难",   folder: "困难" },
    ],
    giscus: {
      // 到 https://giscus.app 绑定你的仓库后，填入以下四项
      repo: "gmmmmod2/PytorchPractice--Using_Mathematical_Formulas_To_Build_models",
      repoId: "",      // 形如 R_kgDOxxxxxx
      category: "General",
      categoryId: "",  // 形如 DIC_kwDOxxxxxx4Cxxxx
    },
  };

  /* =============== GitHub API =============== */
  const API_BASE = (owner, repo) => `https://api.github.com/repos/${owner}/${repo}/contents`;

  async function listDir(path, ref) {
    const url = `${API_BASE(CONFIG.github.owner, CONFIG.github.repo)}/${encodeURIComponent(path)}?ref=${encodeURIComponent(ref)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GitHub API 读取目录失败: ${res.status}`);
    const data = await res.json();
    if (!Array.isArray(data)) return [];
    return data
      .filter(x => x.type === 'file' && /\.md$/i.test(x.name))
      .map(x => ({
        name: x.name,
        path: x.path,
        sha: x.sha,
        download_url: x.download_url,
      }));
  }

  async function fetchMarkdownRaw(downloadUrl) {
    const res = await fetch(downloadUrl);
    if (!res.ok) throw new Error(`读取题目内容失败: ${res.status}`);
    return await res.text();
  }

  async function getCatalog() {
    const { basePath, branch } = CONFIG.github;
    const result = {};
    for (const c of CONFIG.categories) {
      const dirPath = `${basePath}/${c.folder}`;
      try {
        const files = await listDir(dirPath, branch);
        result[c.key] = files;
      } catch (e) {
        console.warn('目录读取失败', dirPath, e);
        result[c.key] = [];
      }
    }
    return result;
  }

  /* =============== UI helpers =============== */
  function buildTabs(activeKey, onChange) {
    const tabs = document.getElementById('tabs');
    tabs.innerHTML = '';
    CONFIG.categories.forEach(c => {
      const btn = document.createElement('button');
      btn.className = 'tab' + (c.key === activeKey ? ' active' : '');
      btn.textContent = c.label;
      btn.addEventListener('click', () => onChange(c.key));
      tabs.appendChild(btn);
    });
  }

  function buildFileList(files, activePath, onClick) {
    const list = document.getElementById('fileList');
    list.innerHTML = '';
    files.forEach(f => {
      const btn = document.createElement('button');
      btn.className = 'file-item' + (f.path === activePath ? ' active' : '');
      btn.innerHTML = f.name.replace(/\.md$/i, '');
      btn.addEventListener('click', () => onClick(f));
      list.appendChild(btn);
    });
  }

  function setContentTitle(title) {
    const el = document.getElementById('contentTitle');
    if (el) el.textContent = title;
  }

  function setViewRawLink(url) {
    const a = document.getElementById('viewRawLink');
    if (!a) return;
    if (url) {
      a.href = url;
      a.style.display = 'inline';
    } else {
      a.style.display = 'none';
    }
  }

  function renderMarkdown(mdText) {
    const html = marked.parse(mdText ?? '');
    return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
  }

  /* =============== Giscus（评论） =============== */
  function mountComments(termPath) {
    const container = document.getElementById('comments');
    if (!container) return;
    container.innerHTML = '';

    // 如果还没配置 repoId/categoryId，则给出提示，但不加载 giscus
    if (!CONFIG.giscus.repoId || !CONFIG.giscus.categoryId) {
      const tip = document.createElement('div');
      tip.className = 'hint';
      tip.textContent = '（提示：尚未配置 Giscus repoId/categoryId，前往 https://giscus.app 绑定仓库后填入 app.js 顶部配置即可启用评论。）';
      container.appendChild(tip);
      return;
    }

    // 动态插入 giscus 脚本
    const script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.async = true;
    script.crossOrigin = 'anonymous';

    script.setAttribute('data-repo', CONFIG.giscus.repo);
    script.setAttribute('data-repo-id', CONFIG.giscus.repoId);
    script.setAttribute('data-category', CONFIG.giscus.category);
    script.setAttribute('data-category-id', CONFIG.giscus.categoryId);
    script.setAttribute('data-mapping', 'specific');
    script.setAttribute('data-term', termPath);
    script.setAttribute('data-strict', '1');
    script.setAttribute('data-reactions-enabled', '1');
    script.setAttribute('data-emit-metadata', '0');
    script.setAttribute('data-input-position', 'top');
    script.setAttribute('data-theme', 'dark');
    script.setAttribute('data-lang', 'zh-CN');

    container.appendChild(script);
  }

  /* =============== 应用逻辑 =============== */
  let catalog = null;
  let activeCatKey = CONFIG.categories[0].key;
  let activeFile = null;

  function allFilesFlat() {
    if (!catalog) return [];
    return Object.values(catalog).flat();
  }

  function pickRandom() {
    const all = allFilesFlat();
    if (!all.length) return null;
    return all[Math.floor(Math.random() * all.length)];
  }

  function updateHash(path) {
    if (path) window.location.hash = encodeURIComponent(path);
  }

  async function openFile(file) {
    if (!file) return;
    activeFile = file;
    setContentTitle(file.name.replace(/\.md$/i, ''));
    setViewRawLink(file.download_url);

    const body = document.getElementById('contentBody');
    body.innerHTML = '<div class="hint">正在加载内容…</div>';
    try {
      const md = await fetchMarkdownRaw(file.download_url);
      body.innerHTML = renderMarkdown(md);
    } catch (e) {
      body.innerHTML = '<div class="hint">读取失败，请稍后再试。</div>';
    }

    mountComments(file.path);
    updateHash(file.path);
    const cat = CONFIG.categories.find(c => file.path.includes(`/${c.folder}/`));
    if (cat) activeCatKey = cat.key;
    refreshSidebar();
  }

  function refreshSidebar() {
    buildTabs(activeCatKey, (key) => {
      activeCatKey = key;
      renderFileList();
    });
    renderFileList();
  }

  function renderFileList() {
    const list = (catalog && catalog[activeCatKey]) ? catalog[activeCatKey] : [];
    const qEl = document.getElementById('searchInput');
    const q = (qEl.value || '').trim().toLowerCase();
    const filtered = q ? list.filter(f => f.name.toLowerCase().includes(q)) : list;
    buildFileList(filtered, activeFile && activeFile.path, openFile);
  }

  function tryOpenFromHash() {
    const h = decodeURIComponent((window.location.hash || '').replace(/^#/, ''));
    if (!h) return false;
    const found = allFilesFlat().find(x => x.path === h);
    if (found) { openFile(found); return true; }
    return false;
  }

  async function main() {
    document.getElementById('year').textContent = String(new Date().getFullYear());

    try {
      catalog = await getCatalog();
      refreshSidebar();
      if (!tryOpenFromHash()) {
        // 初始不打开题
      }
    } catch (err) {
      console.error(err);
      const body = document.getElementById('contentBody');
      body.innerHTML = '<div class="hint">初始化失败，请检查网络或仓库配置（确保仓库为 Public 且存在 content/四个目录）。</div>';
    }

    document.getElementById('searchInput').addEventListener('input', renderFileList);
    document.getElementById('randomBtn').addEventListener('click', () => {
      const f = pickRandom();
      if (f) openFile(f);
    });
    window.addEventListener('hashchange', () => { tryOpenFromHash(); });
  }

  main();
})();
