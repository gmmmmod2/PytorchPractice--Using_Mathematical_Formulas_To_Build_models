import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Loader2, Shuffle, Github, Folder, FileText, ArrowLeft } from "lucide-react";
import Giscus from "@giscus/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

/**
 * NLP 练习题目录页 & 渲染
 * 
 * ✅ 双栏布局：左侧目录（入门/基础/中等/困难），右侧渲染题目
 * ✅ 随机练习按钮：从所有题目随机抽一题
 * ✅ 评论区：基于 GitHub（Giscus）
 * ✅ 黑色背景：深色主题
 * ✅ 自动从 GitHub 仓库读取 md 文件列表与内容
 * 
 * 使用方法：
 * 1) 将你的题目按目录结构放在 GitHub 仓库，例如：
 *    content/
 *      rumen/  (入门)
 *      jichu/  (基础)
 *      zhongdeng/ (中等)
 *      kunnan/ (困难)
 *    每个目录内若干 .md 文件，文件名即题目名称（去掉扩展名）。
 * 
 * 2) 填写下面的 GITHUB 与 GISCUS 配置。
 * 
 * 3) 直接在任意 React 环境中使用该组件，或部署到 GitHub Pages。
 */

/*************************
 *      配置信息           *
 *************************/
const GITHUB = {
  owner: "gmmmmod2",
  repo: "PytorchPractice--Using_Mathematical_Formulas_To_Build_models",
  branch: "main", // 或者 "master"
  basePath: "content", // 你的题目根目录（见上）
};

// 目录与路径映射
const CATEGORIES: { key: CategoryKey; label: string; folder: string }[] = [
  { key: "rumen", label: "入门", folder: "rumen" },
  { key: "jichu", label: "基础", folder: "jichu" },
  { key: "zhongdeng", label: "中等", folder: "zhongdeng" },
  { key: "kunnan", label: "困难", folder: "kunnan" },
];

// Giscus 评论配置（登录 GitHub 即可评论）。
// 你需要在 https://giscus.app 按照你的仓库生成以下参数。
const GISCUS = {
  repo: "your-gh-username/your-repo-name",
  repoId: "REPO_ID",
  category: "General",
  categoryId: "CATEGORY_ID",
};

/*************************
 *      类型定义           *
 *************************/
 type CategoryKey = "rumen" | "jichu" | "zhongdeng" | "kunnan";
 type FileEntry = {
  name: string; // 文件名（含 .md）
  path: string; // 仓库内路径
  sha: string;
  download_url: string; // 原始 raw 链接（GitHub API 会给出）
 };

 type Catalog = Record<CategoryKey, FileEntry[]>;

/*************************
 *     工具函数/Hook       *
 *************************/
const apiBase = (owner: string, repo: string) =>
  `https://api.github.com/repos/${owner}/${repo}/contents`;

async function fetchDir(owner: string, repo: string, path: string, ref: string) {
  const url = `${apiBase(owner, repo)}/${encodeURIComponent(path)}?ref=${encodeURIComponent(ref)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GitHub API 读取目录失败: ${res.status}`);
  const data = (await res.json()) as any[];
  // 仅保留 .md 文件
  return data
    .filter((x) => x.type === "file" && /\.md$/i.test(x.name))
    .map((x) => ({
      name: x.name,
      path: x.path,
      sha: x.sha,
      download_url: x.download_url as string,
    })) as FileEntry[];
}

function fileNameToTitle(name: string) {
  return name.replace(/\.md$/i, "");
}

function useCatalog() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const results = await Promise.all(
          CATEGORIES.map((c) =>
            fetchDir(GITHUB.owner, GITHUB.repo, `${GITHUB.basePath}/${c.folder}`, GITHUB.branch)
              .then((files) => ({ key: c.key, files }))
          )
        );
        const cat: Partial<Catalog> = {};
        results.forEach((r) => (cat[r.key as CategoryKey] = r.files));
        setCatalog(cat as Catalog);
      } catch (e: any) {
        setError(e?.message ?? String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return { catalog, loading, error };
}

async function fetchMarkdownRaw(downloadUrl: string) {
  const res = await fetch(downloadUrl);
  if (!res.ok) throw new Error(`读取题目内容失败: ${res.status}`);
  return await res.text();
}

/*************************
 *     主组件 UI           *
 *************************/
export default function NLPPytorchPractice() {
  const { catalog, loading, error } = useCatalog();

  const [activeCat, setActiveCat] = useState<CategoryKey>("rumen");
  const [activeFile, setActiveFile] = useState<FileEntry | null>(null);
  const [content, setContent] = useState<string>("");
  const [isFetchingMd, setIsFetchingMd] = useState(false);
  const [query, setQuery] = useState("");

  // 所有题目扁平化，用于随机抽题
  const allFiles = useMemo(() => {
    if (!catalog) return [] as FileEntry[];
    return CATEGORIES.flatMap((c) => catalog[c.key] ?? []);
  }, [catalog]);

  // 根据 URL hash（#路径）进行直达
  useEffect(() => {
    const h = decodeURIComponent(window.location.hash.replace(/^#/, ""));
    if (h) {
      // 尝试在所有文件中找到 path 匹配的项
      const f = allFiles.find((x) => x.path === h);
      if (f) {
        setActiveFile(f);
        const cat = CATEGORIES.find((c) => h.includes(`/${c.folder}/`));
        if (cat) setActiveCat(cat.key);
      }
    }
  }, [allFiles.length]);

  // 读取题目内容
  useEffect(() => {
    (async () => {
      if (!activeFile) return;
      try {
        setIsFetchingMd(true);
        const md = await fetchMarkdownRaw(activeFile.download_url);
        setContent(md);
        // 更新 hash 以便分享/刷新直达
        window.location.hash = encodeURIComponent(activeFile.path);
      } catch (e) {
        setContent(`# 读取失败\n\n请稍后再试。`);
      } finally {
        setIsFetchingMd(false);
      }
    })();
  }, [activeFile?.path]);

  function handleRandom() {
    if (!allFiles.length) return;
    const idx = Math.floor(Math.random() * allFiles.length);
    setActiveFile(allFiles[idx]);
    const cat = CATEGORIES.find((c) => allFiles[idx].path.includes(`/${c.folder}/`));
    if (cat) setActiveCat(cat.key);
  }

  const filteredList = useMemo(() => {
    const list = catalog?.[activeCat] ?? [];
    if (!query.trim()) return list;
    const q = query.trim().toLowerCase();
    return list.filter((f) => fileNameToTitle(f.name).toLowerCase().includes(q));
  }, [catalog, activeCat, query]);

  return (
    <div className="min-h-screen bg-black text-gray-100">
      <div className="max-w-7xl mx-auto p-4 md:p-6">
        <header className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Github className="w-6 h-6" />
            <h1 className="text-xl md:text-2xl font-semibold">NLP · PyTorch 练习题</h1>
          </div>
          <div className="flex items-center gap-2">
            <Button onClick={handleRandom} variant="secondary" className="rounded-2xl shadow-md">
              <Shuffle className="w-4 h-4 mr-2" /> 随机练习
            </Button>
          </div>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
          {/* 左侧目录 */}
          <Card className="bg-zinc-900/60 border-zinc-800 md:col-span-4">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2"><Folder className="w-5 h-5"/>目录</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs value={activeCat} onValueChange={(v) => setActiveCat(v as CategoryKey)}>
                <TabsList className="grid grid-cols-4 bg-zinc-800">
                  {CATEGORIES.map((c) => (
                    <TabsTrigger key={c.key} value={c.key} className="data-[state=active]:bg-zinc-700">
                      {c.label}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>

              <div className="mt-3">
                <Input
                  placeholder="搜索题目…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>

              <Separator className="my-3 bg-zinc-800" />

              {loading && (
                <div className="flex items-center gap-2 text-sm text-zinc-400"><Loader2 className="w-4 h-4 animate-spin"/> 正在加载目录…</div>
              )}
              {error && (
                <div className="text-red-400 text-sm">加载目录失败：{error}</div>
              )}

              <ScrollArea className="h-[50vh] mt-2 pr-2">
                <ul className="space-y-1">
                  {filteredList.map((f) => (
                    <li key={f.sha}>
                      <Button
                        variant={activeFile?.path === f.path ? "secondary" : "ghost"}
                        className="w-full justify-start font-normal"
                        onClick={() => setActiveFile(f)}
                      >
                        <FileText className="w-4 h-4 mr-2" /> {fileNameToTitle(f.name)}
                      </Button>
                    </li>
                  ))}
                  {!loading && filteredList.length === 0 && (
                    <div className="text-sm text-zinc-500">暂无题目</div>
                  )}
                </ul>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* 右侧渲染 */}
          <div className="md:col-span-8 flex flex-col gap-4">
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg flex items-center gap-2">
                    {activeFile ? (
                      <>
                        <Button variant="ghost" size="sm" className="mr-1" onClick={() => window.history.back()}>
                          <ArrowLeft className="w-4 h-4"/>
                        </Button>
                        {fileNameToTitle(activeFile.name)}
                      </>
                    ) : (
                      <span className="text-zinc-300">选择左侧题目开始练习</span>
                    )}
                  </CardTitle>
                  {activeFile && (
                    <a
                      href={activeFile.download_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sm text-zinc-400 hover:text-zinc-200"
                    >
                      在 GitHub 查看原文 →
                    </a>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {!activeFile && (
                  <div className="text-zinc-400 text-sm">
                    提示：点击左侧任意题目，或点“随机练习”快速开始。
                  </div>
                )}
                {activeFile && isFetchingMd && (
                  <div className="flex items-center gap-2 text-sm text-zinc-400"><Loader2 className="w-4 h-4 animate-spin"/> 正在加载内容…</div>
                )}
                {activeFile && !isFetchingMd && (
                  <article className="prose prose-invert max-w-none prose-pre:bg-zinc-800 prose-pre:border prose-pre:border-zinc-700">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                      {content}
                    </ReactMarkdown>
                  </article>
                )}
              </CardContent>
            </Card>

            {/* 评论区（Giscus） */}
            {activeFile && (
              <Card className="bg-zinc-900/60 border-zinc-800">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">评论区</CardTitle>
                </CardHeader>
                <CardContent>
                  <Giscus
                    id="comments"
                    repo={GISCUS.repo}
                    repoId={GISCUS.repoId}
                    category={GISCUS.category}
                    categoryId={GISCUS.categoryId}
                    mapping="specific"
                    term={activeFile.path} // 每道题唯一
                    reactionsEnabled="1"
                    emitMetadata="0"
                    inputPosition="top"
                    theme="dark"
                    lang="zh-CN"
                    loading="lazy"
                  />
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        <footer className="mt-6 text-xs text-zinc-500">
          © {new Date().getFullYear()} PyTorch 练习 · 支持深色主题。
        </footer>
      </div>
    </div>
  );
}
