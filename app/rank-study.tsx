"use client";
import { useEffect, useState } from "react";
import { api, money, Run } from "./types";
type Pair = {
  n: number;
  gains: number;
  losses: number;
  ties: number;
  delta: number;
  lower: number;
  upper: number;
  exact_p: number;
};
type Result = {
  id: string;
  rank: number;
  alpha: number;
  status: string;
  summary: null | {
    n: number;
    valid: number;
    schema: number;
    exact: number;
    content: number;
    fields: Record<string, number>;
    latency_p50_ms: number | null;
    latency_p95_ms: number | null;
  };
  groups: Record<string, { n: number; exact: number; content?: number }>;
  paired: Pair | null;
  resources: {
    trainable: number | null;
    adapter_bytes: number | null;
    train_loss: number | null;
    validation_loss: number | null;
    training_seconds: number | null;
    tokens_per_second: number | null;
    cuda_peak_mib: number | null;
    estimated_usd: number | null;
    posted_usd: number | null;
  };
};
type Study = { arms: string[]; ranks: Result[]; exports?: string[] };
const labels: Record<string, string> = {
  base: "Base",
  prompted: "Base · improved prompt",
  base_constrained: "Base · constrained",
  tuned: "Tuned",
  tuned_constrained: "Tuned · constrained",
};
const n = (v: number | null | undefined, d = 2) =>
  v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: d });
const pp = (v: number) => (v > 0 ? "+" : "") + (100 * v).toFixed(1);
export function RankStudy({
  runs,
  open,
}: {
  runs: Run[];
  open: (id: string) => void;
}) {
  const [split, chooseSplit] = useState("locked"),
    [arm, chooseArm] = useState("tuned_constrained"),
    [ref, chooseRef] = useState("base_constrained");
  const [result, setResult] = useState<Study | null>(null),
    [error, setError] = useState("");
  const signature = runs
    .map((r) => [r.id, r.status, r.actual?.totalAmount])
    .join("|");
  useEffect(() => {
    let alive = true;
    setError("");
    setResult(null);
    api<Study>(
      "research/study?" +
        new URLSearchParams({ partition: split, arm, reference: ref }),
    )
      .then((r) => {
        if (alive) setResult(r);
      })
      .catch((e) => {
        if (alive) setError(String(e));
      });
    return () => {
      alive = false;
    };
  }, [split, arm, ref, signature]);
  if (error) return <p role="alert">{error}</p>;
  if (!result?.ranks.length) return null;
  const rows = result.ranks,
    groups = Array.from(new Set(rows.flatMap((r) => Object.keys(r.groups))));
  return (
    <section className="panel analysispanel">
      <h2>Controlled rank study</h2>
      <p>
        Primary comparison · Rank 8, locked notes, tuned and base weights under
        the same grammar.
      </p>
      <div className="fields">
        <label>
          Study partition
          <select value={split} onChange={(e) => chooseSplit(e.target.value)}>
            <option value="locked">Locked test</option>
            <option value="development">Development</option>
          </select>
        </label>
        <label>
          Study arm
          <select value={arm} onChange={(e) => chooseArm(e.target.value)}>
            {result.arms.map((a) => (
              <option key={a} value={a}>
                {labels[a]}
              </option>
            ))}
          </select>
        </label>
        <label>
          Paired reference
          <select value={ref} onChange={(e) => chooseRef(e.target.value)}>
            {result.arms.map((a) => (
              <option key={a} value={a}>
                {labels[a]}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="researchscroll">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Valid JSON</th>
              <th>Schema</th>
              <th>Strict record</th>
              <th>After fence removal</th>
              <th>Item / qty / site correct</th>
              <th>Latency p50 / p95 (ms)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const s = r.summary;
              return (
                <tr key={r.id}>
                  <th>
                    <button className="button" onClick={() => open(r.id)}>
                      {r.rank}
                    </button>
                  </th>
                  {[s?.valid, s?.schema, s?.exact, s?.content].map((v, i) => (
                    <td key={i}>{v == null ? "—" : v + "/" + s!.n}</td>
                  ))}
                  <td>
                    {s
                      ? [s.fields.item, s.fields.quantity, s.fields.site].join(
                          " / ",
                        )
                      : "—"}
                  </td>
                  <td>
                    {s
                      ? n(s.latency_p50_ms, 0) + " / " + n(s.latency_p95_ms, 0)
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <h3>Paired record accuracy</h3>
      <div className="researchscroll">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Gained / lost / tied</th>
              <th>Difference (pp)</th>
              <th>95% interval (pp)</th>
              <th>Exact paired p</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <th>{r.rank}</th>
                <td>
                  {r.paired
                    ? [r.paired.gains, r.paired.losses, r.paired.ties].join(
                        " / ",
                      )
                    : "—"}
                </td>
                <td>{r.paired ? pp(r.paired.delta) : "—"}</td>
                <td>
                  {r.paired
                    ? pp(r.paired.lower) + " to " + pp(r.paired.upper)
                    : "—"}
                </td>
                <td>{n(r.paired?.exact_p, 5)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <small>
        Intervals use matched notes and conservative exact binomial bounds.
        Repeated synthetic templates and one training seed limit generalization.
      </small>
      {groups.length > 0 && (
        <>
          <h3>Exact matches by template and item–site pair</h3>
          <small>
            Each cell shows strict matches, then content matches after removing
            one enclosing Markdown fence.
          </small>
          <div className="researchscroll">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  {groups.map((g) => (
                    <th key={g}>{g}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id}>
                    <th>{r.rank}</th>
                    {groups.map((g) => (
                      <td key={g}>
                        {r.groups[g]
                          ? r.groups[g].exact +
                            "/" +
                            r.groups[g].n +
                            (r.groups[g].content == null
                              ? ""
                              : " · " +
                                r.groups[g].content +
                                "/" +
                                r.groups[g].n)
                          : "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      <h3>Training and cloud usage</h3>
      <div className="researchscroll">
        <table>
          <thead>
            <tr>
              <th>Rank / alpha</th>
              <th>Trainable</th>
              <th>Adapter bytes</th>
              <th>Train / val loss</th>
              <th>Train seconds</th>
              <th>Tokens/s</th>
              <th>CUDA peak (MiB)</th>
              <th>Estimated</th>
              <th>Posted</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const v = r.resources;
              return (
                <tr key={r.id}>
                  <th>
                    {r.rank} / {r.alpha}
                  </th>
                  <td>{n(v.trainable, 0)}</td>
                  <td>{n(v.adapter_bytes, 0)}</td>
                  <td>
                    {n(v.train_loss, 6)} / {n(v.validation_loss, 6)}
                  </td>
                  <td>{n(v.training_seconds)}</td>
                  <td>{n(v.tokens_per_second)}</td>
                  <td>{n(v.cuda_peak_mib, 0)}</td>
                  <td>
                    {v.estimated_usd == null ? "—" : money(v.estimated_usd)}
                  </td>
                  <td>
                    {v.posted_usd == null ? "Pending" : money(v.posted_usd)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!!result.exports?.length && (
        <>
          <h3>Study exports</h3>
          <div className="pills">
            {result.exports.map((k) => (
              <a className="button" key={k} href={"/api/research/files/" + k}>
                {
                  (
                    {
                      findings: "Findings",
                      tables: "Measurement tables",
                      figure: "Results figure",
                      scores: "Arm scores",
                      paired: "Paired differences",
                      strata: "Generalization",
                      failures: "Failure index",
                      costs: "Cloud closeout",
                      training: "Training history",
                      resources: "Resource measurements",
                      "rank-contrasts": "Rank comparisons",
                      repeatability: "Base repeatability",
                    } as Record<string, string>
                  )[k]
                }
              </a>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
