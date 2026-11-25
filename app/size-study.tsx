"use client";
import { useEffect, useState } from "react";
import { api, Run } from "./types";

type Axis = "parameters" | "adapter_bytes" | "latency" | "cost";
type Seed = {
  own_base: {
    gains: number;
    losses: number;
    ties: number;
    delta: number;
    lower: number;
    upper: number;
  } | null;
  arm: string;
  seed: number | null;
  n: number;
  correct: number;
  schema: number;
  fields: Record<string, number>;
  latency: number;
  p95: number;
  generation_tps: number;
  generation_peak_mib: number | null;
  train_tps: number | null;
  train_seconds: number | null;
  peak_mib: number | null;
  groups: Record<string, { n: number; correct: number }>;
};
type Point = {
  id: string;
  model: string;
  accuracy: number;
  parameters: number;
  adapter_bytes: number;
  latency: number;
  cost: number | null;
  seed_sd: number | null;
  trainable: number;
  improvement_per_million: number | null;
  seed_rows: Seed[];
};
type Study = {
  points: Point[];
  frontiers: Record<Axis, string[]>;
  joint_frontier?: string[];
  exports?: string[];
  sessions: { model: string; status: string }[];
  pairs: {
    model: string;
    arm: string;
    gains: number;
    losses: number;
    ties: number;
    delta: number;
    lower: number;
    upper: number;
    exact_p: number;
    leave_family_out: [number, number] | null;
  }[];
};
type Case = {
  id: string;
  prompt: string;
  expected: string;
  prediction: string;
  template: string;
  difficulty?: string;
  errors: string[];
  latency_ms: number;
};
const exportNames: Record<string, string> = {
  findings: "Study findings",
  figure: "Results figure",
  seeds: "Seed measurements",
  difficulty: "Difficulty scores",
  paired: "Paired comparisons",
  frontiers: "Pareto frontiers",
  billing: "RunPod charges",
  history: "Training history",
  adapters: "Adapter inventory",
  agreement: "Seed agreement",
  matched: "Matched model sizes",
  primary: "Primary prediction pairs",
  checksums: "Archive checksums",
};
const axes: Record<Axis, string> = {
  parameters: "Model size (B parameters)",
  adapter_bytes: "Adapter size (MB)",
  latency: "Inference latency (ms)",
  cost: "Posted session cost (USD)",
};
const short = (model: string) =>
  model.split("2.5-")[1]?.replace("-Instruct", "") ?? model;
const display = (n: number | null, d = 2) =>
  n === null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: d });
const axisValue = (n: number, a: Axis) =>
  a === "parameters" ? n / 1e9 : a === "adapter_bytes" ? n / 1e6 : n;

function Chart({
  axis,
  points,
  front,
  choose,
}: {
  axis: Axis;
  points: Point[];
  front: string[];
  choose: (id: string) => void;
}) {
  const visible = points.filter((p) => p[axis] !== null);
  if (!visible.length)
    return (
      <div className="sizechart">
        <h4>{axes[axis]}</h4>
        <p>Awaiting measured values.</p>
      </div>
    );
  const maximum =
    (Math.max(...visible.map((p) => axisValue(p[axis]!, axis))) || 1) * 1.08;
  return (
    <figure className="sizechart">
      <figcaption>{axes[axis]}</figcaption>
      <svg
        viewBox="0 0 334 152"
        role="img"
        aria-label={"Complete-record accuracy versus " + axes[axis]}
      >
        {[0, 0.5, 1].map((y) => (
          <g key={y}>
            <line
              x1="40"
              x2="315"
              y1={116 - y * 88}
              y2={116 - y * 88}
              stroke="#d8dce5"
            />
            <text x="34" y={120 - y * 88} textAnchor="end">
              {y * 100}%
            </text>
          </g>
        ))}
        {[0, 0.5, 1].map((x) => (
          <text key={x} x={40 + x * 265} y="140" textAnchor="middle">
            {display(x * maximum, axis === "cost" ? 3 : 2)}
          </text>
        ))}
        {visible.map((p) => {
          const x = 40 + (265 * axisValue(p[axis]!, axis)) / maximum,
            y = 116 - p.accuracy * 88;
          return (
            <g
              key={p.id}
              role="button"
              tabIndex={0}
              aria-label={"Inspect " + short(p.model)}
              onClick={() => choose(p.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  choose(p.id);
                }
              }}
            >
              <title>
                {short(p.model)} · {display(100 * p.accuracy)}% ·{" "}
                {display(axisValue(p[axis]!, axis), 3)}
              </title>
              <circle
                cx={x}
                cy={y}
                r="9"
                fill="transparent"
                stroke={front.includes(p.id) ? "#087668" : "none"}
                strokeWidth="2"
              />
              <circle cx={x} cy={y} r="4.4" fill="#384faa" />
              <text x={x} y={y - 12} textAnchor="middle">
                {short(p.model)}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

function Inspector({
  run,
  arm,
  partition,
}: {
  run: string;
  arm: string;
  partition: string;
}) {
  const [fail, setFail] = useState(true),
    [offset, setOffset] = useState(0),
    [page, setPage] = useState<{ total: number; examples: Case[] } | null>(
      null,
    ),
    [err, setErr] = useState("");
  useEffect(() => {
    let live = true;
    setPage(null);
    setErr("");
    api<{ total: number; examples: Case[] }>(
      "research/model-size/" +
        run +
        "/cases?" +
        new URLSearchParams({
          partition,
          arm,
          failures: String(fail),
          offset: String(offset),
        }),
    )
      .then((v) => {
        if (live) setPage(v);
      })
      .catch((e) => {
        if (live) setErr(String(e));
      });
    return () => {
      live = false;
    };
  }, [run, arm, partition, fail, offset]);
  return (
    <div>
      <h3>Prediction inspection</h3>
      <label className="check">
        <input
          type="checkbox"
          checked={fail}
          onChange={(e) => {
            setFail(e.target.checked);
            setOffset(0);
          }}
        />{" "}
        Failures only
      </label>
      {err && <p role="alert">{err}</p>}
      {page && (
        <>
          <p>{page.total} matching predictions</p>
          {page.examples.map((e) => (
            <details key={e.id}>
              <summary>
                {e.id} · {e.difficulty ?? e.template} ·{" "}
                {e.errors.join(", ") || "Correct"}
              </summary>
              <p>{e.prompt}</p>
              <strong>Expected</strong>
              <pre>{e.expected}</pre>
              <strong>Generated</strong>
              <pre>{e.prediction}</pre>
              <small>
                {display(e.latency_ms)} ms · {e.template}
              </small>
            </details>
          ))}
          <div className="pills">
            <button
              disabled={offset === 0}
              onClick={() => setOffset(offset - 20)}
            >
              Previous
            </button>
            <button
              disabled={offset + 20 >= page.total}
              onClick={() => setOffset(offset + 20)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export function SizeStudy({ runs }: { runs: Run[] }) {
  const [benchmark, B] = useState("mission3"),
    [part, P] = useState("locked"),
    [condition, C] = useState("tuned"),
    [selected, S] = useState(""),
    [arm, A] = useState(""),
    [result, R] = useState<Study | null>(null),
    [err, E] = useState("");
  const signature = runs
    .map((r) => [r.id, r.status, r.actual?.totalAmount].join(" "))
    .join("|");
  useEffect(() => {
    let live = true;
    R(null);
    E("");
    api<Study>(
      "research/model-size?" +
        new URLSearchParams({ benchmark, partition: part, condition }),
    )
      .then((v) => {
        if (live) R(v);
      })
      .catch((e) => {
        if (live) E(String(e));
      });
    return () => {
      live = false;
    };
  }, [benchmark, part, condition, signature]);
  const pt = result?.points.find((p) => p.id === selected) ?? result?.points.find(p=>p.model.includes("1.5B-")) ?? result?.points[0],
    seed = pt?.seed_rows.find((r) => r.arm === arm) ?? pt?.seed_rows.find(r=>r.seed===42) ?? pt?.seed_rows[0],
    pair = result?.pairs.find(
      (x) => x.model === pt?.model && x.arm === seed?.arm,
    );
  return (
    <section className="panel analysispanel">
      <h2>Model size and specialization</h2>
      <div className="fields">
        <label>
          Size benchmark
          <select value={benchmark} onChange={(e) => B(e.target.value)}>
            <option value="mission3">Independent receipt study</option>
            <option value="historical64">Historical rank-4 benchmark</option>
          </select>
        </label>
        <label>
          Size partition
          <select value={part} onChange={(e) => P(e.target.value)}>
            <option value="locked">Locked test</option>
            <option value="development">Development</option>
          </select>
        </label>
        <label>
          Weight condition
          <select value={condition} onChange={(e) => C(e.target.value)}>
            <option value="tuned">Rank-4 adapters</option>
            <option value="base">Constrained base</option>
          </select>
        </label>
      </div>
      {err && <p role="alert">{err}</p>}
      {!result && !err && <p>Loading measurements…</p>}
      {result && !pt && (
        <>
          <p>
            Awaiting verified model sessions. Accuracy and resource charts will
            appear after artifact retrieval.
          </p>
          <div className="pills">
            {result.sessions.map((s) => (
              <span key={s.model}>
                {short(s.model)} · {s.status}
              </span>
            ))}
          </div>
        </>
      )}
      {result && pt && seed && (
        <>
          <p>
            Accuracy is the mean complete-record accuracy across available
            training seeds. Rings mark the Pareto frontier for each plotted
            trade-off. Cost covers the entire model session.
          </p>
          <div className="sizecharts">
            {(Object.keys(axes) as Axis[]).map((a) => (
              <Chart
                key={a}
                axis={a}
                points={result.points}
                front={result.frontiers[a]}
                choose={S}
              />
            ))}
          </div>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  {[
                    "Model",
                    "Accuracy",
                    "Seed SD (pp)",
                    "Trainable",
                    "Gain (pp / M trainable)",
                    "Adapter bytes",
                    "Session USD",
                  ].map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.points.map((p) => (
                  <tr key={p.id}>
                    <th>
                      <button onClick={() => S(p.id)}>{short(p.model)}</button>
                    </th>
                    <td>{display(p.accuracy * 100)}%</td>
                    <td>
                      {display(p.seed_sd === null ? null : p.seed_sd * 100)}
                    </td>
                    <td>{display(p.trainable, 0)}</td>
                    <td>{display(p.improvement_per_million)}</td>
                    <td>{display(p.adapter_bytes, 0)}</td>
                    <td>{display(p.cost, 5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="fields">
            <label>
              Inspect model
              <select value={pt.id} onChange={(e) => S(e.target.value)}>
                {result.points.map((p) => (
                  <option key={p.id} value={p.id}>
                    {short(p.model)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Training seed
              <select value={seed.arm} onChange={(e) => A(e.target.value)}>
                {pt.seed_rows.map((r) => (
                  <option key={r.arm} value={r.arm}>
                    {r.seed ?? "Base"}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p>
            {seed.correct}/{seed.n} complete records · {seed.schema}/{seed.n}{" "}
            schema passes · item {seed.fields.item}, quantity{" "}
            {seed.fields.quantity}, site {seed.fields.site}
          </p>
          <p>
            Generation p50 / p95 {display(seed.latency)} / {display(seed.p95)}{" "}
            ms · {display(seed.generation_tps)} generated tokens/s
            {seed.generation_peak_mib != null && (
              <>
                {" "}
                · peak inference allocation {display(
                  seed.generation_peak_mib,
                )}{" "}
                MiB
              </>
            )}
          </p>
          {seed.seed !== null && (
            <p>
              Training {display(seed.train_seconds)} s ·{" "}
              {display(seed.train_tps)} tokens/s · peak {display(seed.peak_mib)}{" "}
              MiB
            </p>
          )}
          {seed.own_base && (
            <p>
              Against the same-size base, {seed.own_base.gains} gains and{" "}
              {seed.own_base.losses} losses. Paired difference{" "}
              {display(100 * seed.own_base.delta)} pp (95% interval{" "}
              {display(100 * seed.own_base.lower)} to{" "}
              {display(100 * seed.own_base.upper)}).
            </p>
          )}
          {pair && (
            <p>
              Paired against the 7B base on these notes: {pair.gains} gains,{" "}
              {pair.losses} losses, {pair.ties} ties. Difference{" "}
              {display(100 * pair.delta)} pp (95% interval{" "}
              {display(100 * pair.lower)} to {display(100 * pair.upper)}), exact
              p {display(pair.exact_p, 5)}.{" "}
              {pair.leave_family_out && (
                <>
                  Leaving out one template family at a time gives{" "}
                  {display(100 * pair.leave_family_out[0])} to{" "}
                  {display(100 * pair.leave_family_out[1])} pp.
                </>
              )}
            </p>
          )}
          <div className="pills">
            {Object.entries(seed.groups).map(([g, v]) => (
              <span key={g}>
                {g} · {v.correct}/{v.n}
              </span>
            ))}
          </div>
          {!runs.some(r => r.historical) && benchmark === "mission3" && seed.seed !== null && (
            <div className="pills">
              {["weights", "config"].map((k) => (
                <a
                  className="button"
                  key={k}
                  href={
                    "/api/research/model-size/" +
                    pt.id +
                    "/adapter/" +
                    seed.seed +
                    "/" +
                    k
                  }
                >
                  Adapter {k}
                </a>
              ))}
            </div>
          )}
          {!!result.joint_frontier?.length && (
            <p>
              Accuracy, latency and posted cost frontier ·{" "}
              {result.points
                .filter((p) => result.joint_frontier?.includes(p.id))
                .map((p) => short(p.model))
                .join(", ")}
            </p>
          )}
          {!!result.exports?.length && (
            <details className="study-downloads">
              <summary>Study reports and data</summary>
              <div className="pills">
                {result.exports.map((name) => (
                  <a key={name} className="button" href={"/api/research/size-files/" + name}>
                    {exportNames[name]}
                  </a>
                ))}
              </div>
            </details>
          )}
          <Inspector
            key={[pt.id, seed.arm, part].join("/")}
            run={pt.id}
            arm={seed.arm}
            partition={part}
          />
        </>
      )}
    </section>
  );
}
