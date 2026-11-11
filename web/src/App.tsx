import { useEffect, useState } from "react";
import { api, type Health } from "./api";
import { ExperimentsPage } from "./pages/Experiments";
import { LabPage } from "./pages/Lab";
import { PipelinePage } from "./pages/Pipeline";
import { VerifyPage } from "./pages/Verify";

const PAGES = {
  verify: { label: "Verify", el: VerifyPage },
  lab: { label: "DIP Lab", el: LabPage },
  experiments: { label: "Experiments", el: ExperimentsPage },
  pipeline: { label: "Pipeline", el: PipelinePage },
} as const;
type PageKey = keyof typeof PAGES;

const fromHash = (): PageKey => {
  const h = location.hash.slice(1);
  return h in PAGES ? (h as PageKey) : "verify";
};

export function App() {
  const [page, setPage] = useState<PageKey>(fromHash);
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const on = () => setPage(fromHash());
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setDown(true));
  }, []);

  const Page = PAGES[page].el;
  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span>eKYC · DIP</span>
          <small>face · document · liveness</small>
        </div>
        <nav className="tabs">
          {(Object.keys(PAGES) as PageKey[]).map((k) => (
            <button key={k} aria-current={k === page ? "page" : undefined} onClick={() => (location.hash = k)}>
              {PAGES[k].label}
            </button>
          ))}
        </nav>
        <span className="spacer" />
        {down ? (
          <span className="badge bad">API offline: run `ekyc serve`</span>
        ) : health ? (
          <span className="row small">
            <span className={`badge ${health.tesseract ? "ok" : "warn"}`}>tesseract</span>
            <span className={`badge ${health.trained.fusion ? "ok" : "neutral"}`}>fusion LR</span>
            <span className={`badge ${health.trained.liveness_lr ? "ok" : "neutral"}`}>liveness LR</span>
          </span>
        ) : null}
      </header>
      <main>
        <Page />
      </main>
    </>
  );
}
