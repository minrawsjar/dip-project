import { useEffect, useState } from "react";
import { api, type Experiment } from "../api";
import { DataTable } from "../components/DataTable";
import { Gallery } from "../components/Gallery";

const command = (id: string) =>
  id === "face_summary" ? "ekyc eval face" : id === "document_summary" ? "ekyc eval document" : `ekyc ablate --exp ${id}`;

/** Results tables and figures written by `ekyc eval` / `ekyc ablate` into results/. */
export function ExperimentsPage() {
  const [items, setItems] = useState<Experiment[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [onlyDone, setOnlyDone] = useState(false);

  useEffect(() => {
    api.experiments().then(setItems).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!items) return <div className="card empty">Loading…</div>;
  const shown = items.filter((e) => !onlyDone || e.available);
  const done = items.filter((e) => e.available).length;

  return (
    <div className="stack">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <h2>Experiments</h2>
          <span className="muted small">{done} / {items.length} have results · numbers are read live from results/*.csv</span>
        </div>
        <div className="chips">
          <button aria-pressed={!onlyDone} onClick={() => setOnlyDone(false)}>All</button>
          <button aria-pressed={onlyDone} onClick={() => setOnlyDone(true)}>With results</button>
        </div>
      </div>
      {shown.map((e) => (
        <div key={e.id} className="card stack" id={e.id}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>
              <span className="mono muted">{e.id}</span> · {e.title}
            </h3>
            {e.available ? <span className="badge ok">results</span> : <code className="badge neutral">{command(e.id)}</code>}
          </div>
          {e.description && <span className="muted small">{e.description}</span>}
          {e.rows.length > 0 && <DataTable columns={e.columns} rows={e.rows} />}
          {e.figures.length > 0 && <Gallery items={e.figures.map((f) => ({ label: f.split("/").pop() ?? f, src: f }))} />}
          {!e.available && (
            <span className="muted small">
              Not run yet.{" "}
              {["E10", "E11"].includes(e.id) && "Needs own-capture liveness clips under data/own_capture/."}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
