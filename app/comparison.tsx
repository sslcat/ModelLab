"use client";
import { useEffect, useState } from "react";
import { api, Metric, money, Run } from "./types";

type Example = {
  id?: string;
  prompt: string;
  input_prompt?: string;
  expected: string;
  prediction: string;
  correct: boolean;
  template?: string;
  latency_ms?: number;
  scores: { errors: string[] } | null;
  content_scores?: { errors: string[] };
};
type Summary = {
  n: number;
  valid: number;
  schema: number;
  exact: number;
  content: number;
  fields: Record<string, number>;
  content_fields: Record<string, number>;
  unseen_n: number;
  unseen_exact: number;
  errors: Record<string, number>;
  latency_n: number;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
};
type Analysis = {
  arms: string[];
  arm: string | null;
  partitions: string[];
  partition: string | null;
  summary: Summary | null;
  examples: Example[];
  groups: Record<string, { n: number; exact: number }>;
  bundle_hash: string;
  correct: number | null;
  total: number | null;
};
const names: Record<string, string> = {
  base: "Base",
  prompted: "Base · improved prompt",
  base_constrained: "Base · constrained",
  tuned: "Fine-tuned",
  tuned_constrained: "Fine-tuned · constrained",
};
const ratio = (n: number, d: number) =>
  d ? n + " / " + d + " (" + ((100 * n) / d).toFixed(1) + "%)" : "—";
const num = (v: number | undefined | null, d = 2) =>
  v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: d });
function useAnalysis(id: string, initial: string, status: string) {
  const [arm, select] = useState(initial),
    [partition, split] = useState("locked"),
    [value, set] = useState<Analysis | null>(null),
    [error, fail] = useState("");
  useEffect(() => {
    let alive = true;
    set(null);
    fail("");
    api<Analysis>(
      "runs/" + id + "/analysis?" + new URLSearchParams({ arm, partition }),
    )
      .then((v) => {
        if (alive) set(v);
      })
      .catch((e) => {
        if (alive) fail(String(e));
      });
    return () => {
      alive = false;
    };
  }, [id, status, arm, partition]);
  return { value, error, select, split };
}
type Pick = ReturnType<typeof useAnalysis>;
function Controls({
  pick,
  run,
  label,
}: {
  pick: Pick;
  run: Run;
  label: string;
}) {
  const v = pick.value;
  return (
    <fieldset className="researchpick">
      <legend>
        {label} · {run.id.slice(0, 6)}
      </legend>
      <b>{run.config.name}</b>
      {pick.error ? (
        <p role="alert">{pick.error}</p>
      ) : v?.arms.length ? (
        <div className="pickfields">
          <label>
            Partition
            <select
              aria-label={label + " partition"}
              value={v.partition ?? ""}
              onChange={(e) => pick.split(e.target.value)}
            >
              {v.partitions.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
          </label>
          <label>
            Inference arm
            <select
              aria-label={label + " arm"}
              value={v.arm ?? ""}
              onChange={(e) => pick.select(e.target.value)}
            >
              {v.arms.map((a) => (
                <option value={a} key={a}>
                  {names[a] ?? a}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : (
        <p>{v ? "No recorded predictions" : "Loading predictions…"}</p>
      )}
    </fieldset>
  );
}
function metrics(v: Analysis | null) {
  const s = v?.summary;
  const row = (key: "valid" | "schema" | "content") =>
    s ? ratio(s[key], s.n) : "—";
  return {
    "JSON validity": row("valid"),
    "Schema compliance": row("schema"),
    "Complete record exact match":
      v?.total != null && v.correct != null ? ratio(v.correct, v.total) : "—",
    "Record match after one fence removal": row("content"),
    ...Object.fromEntries(
      ["item", "quantity", "site"].map((k) => [
        "Field · " + k,
        s ? ratio(s.fields[k], s.n) : "—",
      ]),
    ),
    ...Object.fromEntries(
      ["item", "quantity", "site"].map((k) => [
        "After fence removal · " + k,
        s ? ratio(s.content_fields[k], s.n) : "—",
      ]),
    ),
    "Unseen-template exact match": s ? ratio(s.unseen_exact, s.unseen_n) : "—",
    "Generation latency p50 (ms)": num(s?.latency_p50_ms),
    "Generation latency p95 (ms)": num(s?.latency_p95_ms),
  };
}
function Table({
  left,
  right,
  labels,
}: {
  left: Record<string, string>;
  right: Record<string, string>;
  labels: [string, string];
}) {
  return (
    <div className="researchscroll">
      <table>
        <thead>
          <tr>
            <th>Measure</th>
            <th>{labels[0]}</th>
            <th>{labels[1]}</th>
          </tr>
        </thead>
        <tbody>
          {Object.keys({ ...left, ...right }).map((k) => (
            <tr key={k}>
              <td>{k}</td>
              <td>{left[k] ?? "—"}</td>
              <td>{right[k] ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function Comparison({ left, right }: { left: Run; right: Run }) {
  const a = useAnalysis(
      left.id,
      left.id === right.id ? "base" : "tuned",
      left.status,
    ),
    b = useAnalysis(right.id, "tuned", right.status);
  const [only, setOnly] = useState("All"),
    [query, search] = useState("");
  const av = a.value,
    bv = b.value;
  const labels: [string, string] = [
    (names[av?.arm ?? ""] ?? "A") + " · " + left.id.slice(0, 6),
    (names[bv?.arm ?? ""] ?? "B") + " · " + right.id.slice(0, 6),
  ];
  const same =
    av &&
    bv &&
    av.bundle_hash === bv.bundle_hash &&
    av.partition === bv.partition;
  const pairedKey = (r: Example) => JSON.stringify([r.prompt, r.expected]);
  const bMap = new Map((bv?.examples ?? []).map((x) => [pairedKey(x), x]));
  let rows: { left?: Example; right?: Example }[] = [];
  if (same) {
    const seen = new Set<string>();
    rows = (av?.examples ?? []).map((x) => {
      seen.add(pairedKey(x));
      return { left: x, right: bMap.get(pairedKey(x)) };
    });
    rows.push(
      ...(bv?.examples ?? [])
        .filter((x) => !seen.has(pairedKey(x)))
        .map((x) => ({ right: x })),
    );
  } else
    rows = [
      ...(av?.examples ?? []).map((x) => ({ left: x })),
      ...(bv?.examples ?? []).map((x) => ({ right: x })),
    ];
  const paired = rows.filter((p) => p.left && p.right);
  const gained = paired.filter(
    (p) => !p.left!.correct && p.right!.correct,
  ).length;
  const lost = paired.filter(
    (p) => p.left!.correct && !p.right!.correct,
  ).length;
  const filtered = rows.filter((pair) => {
    const r = pair.left ?? pair.right!;
    const both = [pair.left, pair.right].filter((x): x is Example => !!x);
    return (
      (r.prompt + " " + r.expected)
        .toLowerCase()
        .includes(query.toLowerCase()) &&
      (only === "All" ||
        (only === "Any error" && both.some((x) => !x.correct)) ||
        (only === "Disagreement" &&
          pair.left &&
          pair.right &&
          pair.left.correct !== pair.right.correct) ||
        both.some((x) => x.scores?.errors.includes(only)))
    );
  });
  const errors = Array.from(
    new Set([
      ...Object.keys(av?.summary?.errors ?? {}),
      ...Object.keys(bv?.summary?.errors ?? {}),
    ]),
  );
  const errorCounts = (v: Analysis | null) =>
    Object.fromEntries(
      errors.map((k) => [
        k,
        v?.summary ? String(v.summary.errors[k] ?? 0) : "—",
      ]),
    );
  const groups = (v: Analysis | null) =>
    Object.fromEntries(
      Object.entries(v?.groups ?? {}).map(([k, g]) => [k, ratio(g.exact, g.n)]),
    );
  return (
    <div className="panel analysispanel">
      <h2>Prediction comparison</h2>
      <div className="researchpicks">
        <Controls label="A" run={left} pick={a} />
        <Controls label="B" run={right} pick={b} />
      </div>
      <Table left={metrics(av)} right={metrics(bv)} labels={labels} />
      {paired.length > 0 && (
        <p>
          Across {paired.length} matched notes, B gained {gained} exact matches
          and lost {lost} versus A.
        </p>
      )}
      {!!Object.keys({ ...av?.groups, ...bv?.groups }).length && (
        <>
          <h3>Generalization by split</h3>
          <Table left={groups(av)} right={groups(bv)} labels={labels} />
        </>
      )}
      {errors.length > 0 && (
        <>
          <h3>Error categories</h3>
          <Table
            left={errorCounts(av)}
            right={errorCounts(bv)}
            labels={labels}
          />
        </>
      )}
      {rows.length > 0 && (
        <>
          <h3>Inspect predictions</h3>
          {!same && av?.examples.length && bv?.examples.length ? (
            <p>
              Different partitions or datasets. Each prediction is shown against
              its own reference.
            </p>
          ) : null}
          <div className="pickfields">
            <label>
              Find note, item or site
              <input
                value={query}
                onChange={(e) => search(e.target.value)}
                placeholder="e.g. brackets"
              />
            </label>
            <label>
              Show
              <select value={only} onChange={(e) => setOnly(e.target.value)}>
                {["All", "Any error", "Disagreement", ...errors].map((k) => (
                  <option key={k}>{k}</option>
                ))}
              </select>
            </label>
          </div>
          <p>
            {filtered.length} of {rows.length} records
          </p>
          {filtered.map((p, i) => {
            const r = p.left ?? p.right!;
            return (
              <details key={i + pairedKey(r)}>
                <summary>
                  {r.template ? r.template + " · " : ""}
                  {r.prompt.split("Note: ").at(-1)}
                </summary>
                <p>
                  Reference <code>{r.expected}</code>
                </p>
                {[p.left, p.right].map(
                  (x, j) =>
                    x && (
                      <section key={j}>
                        <b>
                          {j ? "B" : "A"} · {labels[j]} ·{" "}
                          {x.correct
                            ? "Exact match"
                            : x.scores?.errors.join(", ") || "Mismatch"}
                        </b>
                        <pre>{x.prediction}</pre>
                        {x.content_scores &&
                          x.scores?.errors.includes("Markdown fence") && (
                            <small>
                              After fence removal ·{" "}
                              {x.content_scores.errors.join(", ") ||
                                "Exact match"}
                            </small>
                          )}
                        {x.input_prompt && x.input_prompt !== x.prompt && (
                          <small>{x.input_prompt}</small>
                        )}
                        {x.latency_ms != null && (
                          <small>{num(x.latency_ms)} ms</small>
                        )}
                      </section>
                    ),
                )}
              </details>
            );
          })}
        </>
      )}
    </div>
  );
}
function telemetry(run: Run, rows: Metric[]) {
  const values = (k: string) => rows.map((r) => r[k]).filter(Number.isFinite);
  const last = (k: string, d = 3) => num(values(k).at(-1), d);
  const peak = (k: string) =>
    values(k).length ? num(Math.max(...values(k))) : "—";
  return {
    "Mean training loss": last("train_loss", 6),
    "Final validation loss": last("eval_loss", 6),
    "Learning rate (last logged step)": last("learning_rate", 9),
    "Trainer runtime (s)": last("train_runtime"),
    "Overall tokens / second": last("train_tokens_per_second"),
    "Examples / second": last("train_samples_per_second"),
    "Peak PyTorch allocation (MiB)": peak("peak_allocated_mib"),
    "Peak sampled VRAM (MiB)": peak("vram_mib"),
    "Peak sampled GPU utilization (%)": peak("gpu_percent"),
    "Estimated cloud usage":
      run.estimated_usage == null ? "—" : money(run.estimated_usage),
    "Provider charges": run.actual ? money(run.actual.totalAmount) : "—",
  };
}
export function TrainingComparison({
  run,
  other,
  rows,
}: {
  run: Run;
  other?: Run;
  rows: Metric[];
}) {
  const runs = [
    { r: run, rows },
    ...(other ? [{ r: other, rows: other.metrics }] : []),
  ];
  const series = runs.flatMap(({ r, rows }, j) =>
    ["loss", "eval_loss"].map((key, i) => ({
      name: r.id.slice(0, 6) + " " + (i ? "validation" : "training"),
      color: j ? "#c56826" : "#496acb",
      dash: i ? "5 4" : undefined,
      points: rows
        .filter((m) => Number.isFinite(m.step) && Number.isFinite(m[key]))
        .map((m) => ({ x: m.step, y: m[key] })),
    })),
  );
  const pts = series.flatMap((s) => s.points),
    mx = Math.max(1, ...pts.map((p) => p.x)),
    my = Math.max(0.001, ...pts.map((p) => p.y)) * 1.05;
  const x = (n: number) => 55 + (n / mx) * 565,
    y = (n: number) => 149 - (n / my) * 132;
  return (
    <div className="panel">
      <h2>Training telemetry</h2>
      {pts.length ? (
        <div className="plot">
          <svg
            viewBox="0 0 650 178"
            role="img"
            aria-label="Overlaid training and validation loss by optimizer step"
          >
            {[0, 0.5, 1].map((t) => (
              <g key={t}>
                <path d={"M 55 " + y(t * my) + " h 565"} stroke="#cdd7e3" />
                <text x="2" y={y(t * my) + 3}>
                  {num(t * my, 4)}
                </text>
              </g>
            ))}
            {series.map((s) => (
              <g key={s.name} stroke={s.color}>
                <polyline
                  fill="none"
                  strokeWidth="2"
                  strokeDasharray={s.dash}
                  points={s.points.map((p) => x(p.x) + "," + y(p.y)).join(" ")}
                />
                {s.points.map((p, i) => (
                  <circle
                    key={i}
                    cx={x(p.x)}
                    cy={y(p.y)}
                    r={s.dash ? 4 : 2}
                    fill={s.dash ? "white" : s.color}
                  >
                    <title>
                      {s.name} · step {p.x} · {p.y}
                    </title>
                  </circle>
                ))}
              </g>
            ))}
            <text x="55" y="172">
              Step 0
            </text>
            <text x="568" y="172">
              Step {mx}
            </text>
          </svg>
          <div className="curvelegend">
            {series
              .filter((s) => s.points.length)
              .map((s) => (
                <span style={{ color: s.color }} key={s.name}>
                  {s.dash ? "○" : "━"} {s.name}
                </span>
              ))}
          </div>
        </div>
      ) : (
        <p>No measured loss samples.</p>
      )}
      {other ? (
        <Table
          left={telemetry(run, rows)}
          right={telemetry(other, other.metrics)}
          labels={[run.id.slice(0, 6), other.id.slice(0, 6)]}
        />
      ) : (
        <dl>
          {Object.entries(telemetry(run, rows)).map(([k, v]) => (
            <div className="dlrow" key={k}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
