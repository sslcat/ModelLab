"use client";
import { useEffect, useState } from "react";
import { ArrowDownToLine, ArrowRight } from "lucide-react";
import { active, api, Metric, money, Run } from "./types";
import { Comparison, TrainingComparison } from "./comparison";
import { RankStudy } from "./rank-study";
import { Blank, Props } from "./views";

export function Experiments({
  data,
  act,
  refresh,
  open,
  selected,
}: { selected: string } & Props) {
  const [other, compare] = useState(""),
    [pulse, setPulse] = useState<{ id: string; metrics: Metric[] } | null>(
      null,
    );
  const r =
    data.runs.find((x) => x.id === selected) ??
    data.runs.find((x) => x.status === "completed") ??
    data.runs[0];
  useEffect(() => {
    if (!r || !active(r.status)) return;
    const s = new EventSource("/api/runs/" + r.id + "/events");
    s.onmessage = (e) =>
      setPulse({ id: r.id, metrics: JSON.parse(e.data).metrics });
    return () => s.close();
  }, [r?.id, r?.status]);
  if (!r)
    return (
      <div className="panel">
        <Blank title="No experiments yet">
          Save a configuration in Training to create your first experiment.
        </Blank>
      </div>
    );
  const rows =
    pulse?.id === r.id && active(r.status) ? pulse.metrics : r.metrics;
  const q = r.quote,
    cmp = data.runs.find((x) => x.id === other && x.id !== r.id);
  const post = (suffix: string, body = {}) =>
    act(async () => {
      await api("runs/" + r.id + "/" + suffix, body);
      await refresh();
    });
  return (
    <>
      <div className="runselect">
        <label>
          Experiment
          <select value={r.id} onChange={(e) => open(e.target.value)}>
            {data.runs.map((x) => (
              <option key={x.id} value={x.id}>
                {x.config.name} · {x.id.slice(0, 6)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Compare experiment
          <select value={other} onChange={(e) => compare(e.target.value)}>
            <option value="">Select another experiment</option>
            {data.runs
              .filter((x) => x.id !== r.id)
              .map((x) => (
                <option key={x.id} value={x.id}>
                  {x.config.name} · {x.id.slice(0, 6)}
                </option>
              ))}
          </select>
        </label>
      </div>
      <div className="panel">
        <div className="sectionline">
          <div>
            <div className="eyebrow">{r.id.slice(0, 10)}</div>
            <h2>{r.config.name}</h2>
          </div>
          <span className="tag">{r.status}</span>
        </div>
        <p>
          {r.config.model} · {r.config.method} · {r.dataset_name}
        </p>
        {r.detail && <p className="notice">{r.detail}</p>}
        {r.status === "draft" && (
          <button onClick={() => post("quote")}>
            Get live quote <ArrowRight size={14} />
          </button>
        )}
        {q && (
          <div className="offer">
            <h3>RunPod cost review</h3>
            <p>
              {q.gpu} · {q.stock} stock · {money(q.gpu_hourly)}/h GPU +{" "}
              {"$" + q.storage_hourly.toFixed(4)}/h storage. Limit {q.minutes}{" "}
              min. Recovery disk {"$" + q.stopped_hourly.toFixed(4)}/h.
            </p>
            <p>
              Account inventory at quote time · {q.account.pods} pods,{" "}
              {q.account.volumes} volumes, {q.account.endpoints} endpoints.
            </p>
            <strong>Estimated maximum {money(q.max_usd)}</strong>
            {r.status === "draft" && (
              <button
                className="primary"
                disabled={Date.now() > q.expires * 1000}
                onClick={() =>
                  post("approve", { token: q.token, approve: true })
                }
              >
                Approve {money(q.max_usd)} and launch
              </button>
            )}
            <small>
              Includes a recovery allowance and contingency. Provider charges
              can exceed this estimate.
            </small>
          </div>
        )}
        {active(r.status) && (
          <button onClick={() => post("cancel")}>
            Cancel training & retrieve artifacts
          </button>
        )}
        {r.pod && (
          <>
            <p>
              Pod {r.pod} · estimated usage {money(r.estimated_usage ?? 0)} ·
              recorded charges{" "}
              {r.actual ? money(r.actual.totalAmount) : "awaiting provider"} ·
              stopped disk estimate{" "}
              {"$" + (r.stopped_storage_estimate ?? 0).toFixed(4)}
            </p>
            <button onClick={() => post("billing")}>Refresh billing</button>
          </>
        )}
      </div>
      <TrainingComparison run={r} other={cmp} rows={rows} />
      <Comparison key={r.id + (cmp?.id ?? r.id)} left={r} right={cmp ?? r} />
      <div className="panel">
        <h2>{cmp ? "Configuration comparison" : "Run configuration"}</h2>
        <div className="pills">
          <a className="button" href={"/api/runs/" + r.id + "/configuration"}>
            Export configuration · {r.id.slice(0, 6)}
          </a>
          {cmp && (
            <a
              className="button"
              href={"/api/runs/" + cmp.id + "/configuration"}
            >
              Export configuration · {cmp.id.slice(0, 6)}
            </a>
          )}
        </div>
        <table>
          <thead>
            <tr>
              <th>Parameter</th>
              <th>{r.id.slice(0, 6)}</th>
              {cmp && <th>{cmp.id.slice(0, 6)}</th>}
            </tr>
          </thead>
          <tbody>
            {Object.entries(r.config).map(([k, v]) => (
              <tr key={k}>
                <td>{k}</td>
                <td>{k === "dataset" ? r.dataset_name : String(v)}</td>
                {cmp && (
                  <td
                    className={
                      cmp.config[k as keyof typeof cmp.config] !== v
                        ? "diff"
                        : ""
                    }
                  >
                    {k === "dataset"
                      ? cmp.dataset_name
                      : String(cmp.config[k as keyof typeof cmp.config])}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {r.archived && <Exports r={r} />}
      <RankStudy runs={data.runs} open={open} />
    </>
  );
}
export function Exports({ r }: { r: Run }) {
  return (
    <div className="panel">
      <h3>Verified exports</h3>
      <p>
        {r.historical ? "Public research records" : r.archive + " · " + r.sync}
      </p>
      <div className="pills">
        {[
          "report",
          "metrics",
          "manifest",
          ...(r.adapter_available ? ["adapter", "adapter-config"] : []),
        ].map((k) => (
          <a
            className="button"
            key={k}
            href={"/api/runs/" + r.id + "/export/" + k}
          >
            <ArrowDownToLine size={15} />
            {k}
          </a>
        ))}
      </div>
    </div>
  );
}
export function Evaluation({ data, open }: Props) {
  const entries = data.runs.filter((r) => r.evaluation),
    [pick, setPick] = useState("");
  const r = entries.find((x) => x.id === pick) ?? entries[0],
    ev = r?.evaluation;
  if (!ev)
    return (
      <div className="panel">
        <Blank title="No paired evaluations yet">
          The worker runs the frozen base model and its trained adapter against
          the same held-out prompts. Scores and individual predictions appear
          here after retrieval.
        </Blank>
      </div>
    );
  return (
    <>
      <label>
        Evaluation run
        <select value={r.id} onChange={(e) => setPick(e.target.value)}>
          {entries.map((x) => (
            <option key={x.id} value={x.id}>
              {x.config.name}
            </option>
          ))}
        </select>
      </label>
      <RankStudy runs={data.runs} open={open} />
      <Comparison key={r.id} left={r} right={r} />
      <p>{ev.limitations}</p>
      <Exports r={r} />
    </>
  );
}
