import { AlertTriangle, BookMarked, Workflow } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, ask } from "./api/client";
import type { AskResponse, Filters } from "./api/types";
import { AnswerCard } from "./components/AnswerCard";
import { AskPanel } from "./components/AskPanel";
import { EvidenceLadder } from "./components/EvidenceBadge";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { LoadingWorkout } from "./components/LoadingWorkout";
import { MythStrip } from "./components/MythStrip";
import { PipelinePanel } from "./components/PipelinePanel";
import { RecentList } from "./components/RecentList";
import { SourcesPanel } from "./components/SourcesPanel";
import { useHealth } from "./hooks/useHealth";
import { useLocalStorage } from "./hooks/useLocalStorage";
import { COPY, type Lang } from "./i18n";

type Theme = "dark" | "light";
type Tab = "sources" | "pipeline";
const MAX_RECENT = 5;

function initialTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export default function App() {
  const [lang, setLang] = useLocalStorage<Lang>("antibros.lang", "vi");
  const [theme, setTheme] = useLocalStorage<Theme>("antibros.theme", initialTheme());
  const [recent, setRecent] = useLocalStorage<string[]>("antibros.recent", []);
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [filters, setFilters] = useState<Filters>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>("sources");
  const healthy = useHealth();
  const controller = useRef<AbortController | null>(null);
  const t = COPY[lang];

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.lang = lang;
  }, [theme, lang]);

  const submit = useCallback(
    async (q: string) => {
      controller.current?.abort();
      const current = new AbortController();
      controller.current = current;
      setLoading(true);
      setError(null);
      setResult(null);
      setActiveCitation(null);
      setTab("sources");
      setAsked(q);
      setRecent([q, ...recent.filter((r) => r !== q)].slice(0, MAX_RECENT));
      try {
        setResult(await ask(q, filters, current.signal));
      } catch (e) {
        if (current.signal.aborted) return;
        setError(e instanceof ApiError ? `${t.apiError} (${e.status}): ${e.message}` : t.apiDown);
      } finally {
        if (controller.current === current) setLoading(false);
      }
    },
    [filters, recent, setRecent, t],
  );

  const pick = (q: string) => {
    setQuestion(q);
    void submit(q);
  };
  const reset = () => {
    controller.current?.abort();
    setLoading(false);
    setResult(null);
    setError(null);
    setQuestion("");
    window.scrollTo?.({ top: 0, behavior: "smooth" });
  };
  const onCitation = (id: number) => {
    setTab("sources");
    setActiveCitation(id);
  };

  const idle = !loading && !result;

  return (
    <div className="app">
      <div className="bg-glow" aria-hidden="true" />
      <Header
        t={t}
        lang={lang}
        onLang={setLang}
        theme={theme}
        onTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
        healthy={healthy}
        onHome={reset}
      />
      <main className="container">
        {idle && <Hero t={t} />}
        <AskPanel
          t={t}
          lang={lang}
          question={question}
          onQuestion={setQuestion}
          filters={filters}
          onFilters={setFilters}
          onSubmit={(q) => void submit(q)}
          loading={loading}
          compact={!idle}
        />
        <RecentList t={t} items={recent} onPick={pick} onClear={() => setRecent([])} disabled={loading} />

        {error && (
          <div className="error-banner" role="alert">
            <AlertTriangle size={18} aria-hidden="true" /> {error}
          </div>
        )}

        {loading && <LoadingWorkout t={t} />}

        {result && (
          <div className="result-grid">
            <AnswerCard
              t={t}
              result={result}
              question={asked}
              activeCitation={activeCitation}
              onCitation={onCitation}
              onReset={reset}
            />
            <aside className="side-panel">
              <div className="tabs" role="tablist">
                <button
                  role="tab"
                  aria-selected={tab === "sources"}
                  className={tab === "sources" ? "active" : ""}
                  onClick={() => setTab("sources")}
                >
                  <BookMarked size={16} aria-hidden="true" /> {t.sources}
                  <span className="count-badge">{result.citations.length}</span>
                </button>
                <button
                  role="tab"
                  aria-selected={tab === "pipeline"}
                  className={tab === "pipeline" ? "active" : ""}
                  onClick={() => setTab("pipeline")}
                >
                  <Workflow size={16} aria-hidden="true" /> {t.pipeline}
                </button>
              </div>
              <div role="tabpanel">
                {tab === "sources" ? (
                  <>
                    <SourcesPanel t={t} citations={result.citations} active={activeCitation} />
                    <EvidenceLadder t={t} />
                  </>
                ) : (
                  <PipelinePanel t={t} result={result} />
                )}
              </div>
            </aside>
          </div>
        )}

        {idle && <MythStrip t={t} lang={lang} onPick={pick} disabled={loading} />}
      </main>
      <footer className="site-footer">
        <p>{t.footer}</p>
      </footer>
    </div>
  );
}
