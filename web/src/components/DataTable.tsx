type Cell = string | number | boolean | null | undefined;

const num = (v: number) => {
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(1) : a >= 1 ? v.toFixed(3) : a < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(4);
};
const fmt = (v: Cell) => (v == null ? "—" : typeof v === "number" ? num(v) : String(v));

/** Plain table for experiment CSVs; numbers right-aligned in mono. */
export function DataTable({ columns, rows }: { columns: string[]; rows: Record<string, Cell>[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c} className={typeof r[c] === "number" ? "num" : undefined}>{fmt(r[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Flatten one level of nested objects into "parent.child" keys; string lists joined. */
const flatten = (data: Record<string, unknown>) =>
  Object.entries(data).flatMap(([k, v]): [string, unknown][] =>
    v && typeof v === "object" && !Array.isArray(v) ? Object.entries(v).map(([c, cv]) => [`${k}.${c}`, cv]) : [[k, v]],
  );

const show = (v: unknown) =>
  typeof v === "number" ? fmt(v) : Array.isArray(v) ? (v.length ? v.join("; ") : "—") : v && typeof v === "object" ? JSON.stringify(v) : String(v);

/** Definition list of key/value pairs. */
export function KeyValues({ data }: { data: Record<string, unknown> }) {
  return (
    <dl className="kv">
      {flatten(data).map(([k, v]) => (
        <div key={k} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd>{show(v)}</dd>
        </div>
      ))}
    </dl>
  );
}
