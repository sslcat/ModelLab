"use client";
import { useState } from "react";
import {
  ArrowRight,
  Upload,
  Database,
  Plus,
  RefreshCw,
  ChevronRight,
  Cloud,
  Cpu,
  Folder,
  KeyRound,
  ShieldCheck,
} from "lucide-react";
import { api, Dataset, GPU, Studio, money } from "./types";

type Action = (work: () => Promise<void>) => void;
export type Props = {
  data: Studio;
  act: Action;
  refresh: () => Promise<void>;
  open: (id: string) => void;
};
export function Blank({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="blank">
      <span className="emptyicon">
        <Database size={21} />
      </span>
      <strong>{title}</strong>
      <p>{children}</p>
    </div>
  );
}
export function Datasets({ data, act, refresh }: Props) {
  const [selected, select] = useState(data.datasets[0]?.id),
    [adding, add] = useState(false);
  const ds = data.datasets.find((d) => d.id === selected) ?? data.datasets[0];
  return (
    <>
      <div className="sectionline">
        <h2>
          Dataset library <small>{data.datasets.length}</small>
        </h2>
        <button hidden={data.mode === "demo"} className="primary" onClick={() => add(!adding)}>
          <Plus size={16} />
          Import dataset
        </button>
      </div>
      {adding && (
        <form
          className="panel formpanel"
          onSubmit={(e) => {
            e.preventDefault();
            const f = new FormData(e.currentTarget);
            act(async () => {
              const file = f.get("file") as File;
              if (file.size > 2097152)
                throw Error("Choose a file under 2 MiB.");
              const d = await api<Dataset>("datasets", {
                name: f.get("name"),
                license: f.get("license"),
                format: file.name.split(".").pop()?.toLowerCase(),
                content: await file.text(),
                seed: Number(f.get("seed")),
                fraction: Number(f.get("fraction")),
                public_data: f.get("public") === "on",
              });
              await refresh();
              select(d.id);
              add(false);
            });
          }}
        >
          <h3>Import a dataset</h3>
          <div className="fields">
            <label>
              Dataset name
              <input name="name" required maxLength={80} />
            </label>
            <label>
              License or original authorship
              <input
                name="license"
                required
                placeholder="e.g. Original · MIT"
                minLength={2}
              />
            </label>
            <label>
              UTF-8 CSV or JSONL
              <input name="file" type="file" accept=".csv,.jsonl" required />
            </label>
            <label>
              Split seed
              <input name="seed" type="number" defaultValue={42} min="0" />
            </label>
            <label>
              Validation fraction
              <input
                name="fraction"
                type="number"
                step=".01"
                min=".05"
                max=".45"
                defaultValue=".15"
              />
            </label>
          </div>
          <p>
            Each row needs a prompt and a completion. Identical pairs are
            deduplicated before splitting.
          </p>
          <label className="check">
            <input name="public" type="checkbox" required />
            This dataset is licensed for this use and contains only public,
            nonconfidential data.
          </label>
          <button className="primary">
            <Upload size={16} />
            Validate and import
          </button>
        </form>
      )}
      <div className="split">
        <div className="panel list">
          {data.datasets.map((d) => (
            <button
              className={d.id === ds?.id ? "chosen" : ""}
              key={d.id}
              onClick={() => select(d.id)}
            >
              <Database size={18} />
              <span>
                <b>{d.name}</b>
                <small>
                  {d.train + d.validation + d.test} records ·{" "}
                  {d.staged ? d.license : "Re-import to stage"}
                </small>
              </span>
              <ChevronRight size={16} />
            </button>
          ))}
        </div>
        {ds && (
          <div className="panel">
            <div className="eyebrow">DATASET PROFILE</div>
            <h2>{ds.name}</h2>
            <div className="pills">
              <span>{ds.train} train</span>
              <span>{ds.validation} validation</span>
              {ds.test > 0 && <span>{ds.test} held out</span>}
            </div>
            <dl>
              <dt>Duplicate rows removed</dt>
              <dd>{ds.duplicates}</dd>
              <dt>Reference tokens · mean / p95 / max</dt>
              <dd>
                {ds.tokens.mean} / {ds.tokens.p95} / {ds.tokens.max}
              </dd>
              <dt>Split seed</dt>
              <dd>{ds.seed ?? "Versioned hash streams"}</dd>
            </dl>
            <p>
              Reference counts use words and punctuation. The worker records
              model-tokenizer lengths on RunPod.
            </p>
            <label>
              Content SHA-256<code>{ds.fingerprint}</code>
            </label>
            <label>
              Split SHA-256<code>{ds.split_id}</code>
            </label>
            <h3>Preview</h3>
            {ds.preview.map((r, i) => (
              <div className="sample" key={i}>
                <p>{r.prompt}</p>
                <code>{r.completion}</code>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

export const controls = [
  ["learning_rate", "Learning rate", 0.0002, 0.000001, 0.01, 0.000001],
  ["epochs", "Epochs", 1, 1, 12, 1],
  ["batch_size", "Batch size", 1, 1, 8, 1],
  ["grad_accum", "Gradient accumulation", 8, 1, 64, 1],
  ["rank", "LoRA rank", 8, 2, 64, 1],
  ["alpha", "LoRA alpha", 16, 1, 128, 1],
  ["dropout", "LoRA dropout", 0.05, 0, 0.5, 0.01],
  ["seq_len", "Sequence length", 256, 64, 2048, 1],
  ["seed", "Random seed", 42, 0, 2147483647, 1],
  ["minutes", "GPU time limit (min)", 90, 15, 180, 1],
] as const;
export function Training({ data, act, refresh, open }: Props) {
  const [chosen, select] = useState("rank-4");
  if (data.mode === "demo") {
    const run=data.runs.find(r=>r.id===chosen) ?? data.runs[0];
    return <section className="panel formpanel"><div className="eyebrow">RECORDED CONFIGURATION</div><h2>Training configuration</h2><label>Experiment<select value={run.id} onChange={e=>select(e.target.value)}>{data.runs.map(r=><option key={r.id} value={r.id}>{r.config.name}</option>)}</select></label><p>{run.config.model} · {run.dataset_name} · {run.config.method}</p><dl>{controls.map(([key,label])=><div className="dlrow" key={key}><dt>{label}</dt><dd>{run.config[key]}</dd></div>)}</dl><a className="button" href={"/api/runs/"+run.id+"/configuration"}>Download configuration</a></section>;
  }
  return (
    <form
      className="panel formpanel"
      onSubmit={(e) => {
        e.preventDefault();
        const f = new FormData(e.currentTarget);
        const body = Object.fromEntries(f.entries());
        act(async () => {
          const cfg = {
            ...body,
            ...Object.fromEntries(controls.map(([k]) => [k, Number(f.get(k))])),
          };
          const r = await api<{ id: string }>("runs", cfg);
          await refresh();
          open(r.id);
        });
      }}
    >
      <div className="eyebrow">RUN CONFIGURATION</div>
      <h2>New experiment</h2>
      <div className="fields">
        <label>
          Experiment name
          <input
            name="name"
            defaultValue="Receipt extraction · pilot"
            required
            maxLength={80}
          />
        </label>
        <label>
          Dataset
          <select name="dataset">
            {data.datasets.filter(d => d.protocol !== "size-v1").map((d) => (
              <option key={d.id} value={d.id} disabled={!d.staged}>
                {d.name} · {d.rows} rows
              </option>
            ))}
          </select>
        </label>
        <label>
          Base model
          <select name="model">
            {Object.keys(data.models).map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          Method
          <select name="method">
            <option>QLoRA</option>
            <option>LoRA</option>
          </select>
        </label>
        <label className="wide">
          RunPod GPU identifier
          <input name="gpu" defaultValue="NVIDIA GeForce RTX 4090" required />
        </label>
      </div>
      <h3>Hyperparameters</h3>
      <div className="fields three">
        {controls.map(([k, label, val, min, max, step]) => (
          <label key={k}>
            {label}
            <input
              type="number"
              required
              name={k}
              min={min}
              max={max}
              step={step}
              defaultValue={val}
            />
          </label>
        ))}
      </div>
      <div className="formfoot">
        <p>Saving a draft is free. Review a live quote before launch.</p>
        <button className="primary">
          Save experiment <ArrowRight size={16} />
        </button>
      </div>
    </form>
  );
}
type Inventory = {
  pods: { id: string; name: string; status: string; cost: number }[];
  volumes: {
    id: string;
    name: string;
    size: number;
    monthly_estimate: number | null;
  }[];
  endpoints: { id: string; name: string }[];
  abandoned: { id: string; name: string }[];
  balance: { balanceAmount: number };
};
export function Resources({ data, act }: Props) {
  const [catalog, setCatalog] = useState<{
      gpus: GPU[];
      source: string;
      at: number;
    } | null>(null),
    [account, setAccount] = useState<Inventory | null>(null);
  if (data.mode === "demo") return <section className="panel"><div className="eyebrow">RECONCILED RUNPOD USAGE</div><h2>Historical cloud charges</h2><p>Compute and disk charges across the three studies, including the initial failed pilot attempt.</p><div className="billing-total"><strong>${data.reserved.toFixed(5)}</strong><span> of the original ${data.budget} budget</span></div><div className="tablewrap"><table><thead><tr><th>Session</th><th>Status</th><th>Disk (USD)</th><th>Total (USD)</th></tr></thead><tbody>{data.runs.map(r=><tr key={r.id}><td>{r.config.name}</td><td>{r.status}</td><td>{r.actual?.diskAmount.toFixed(5)}</td><td>{r.actual?.totalAmount.toFixed(5)}</td></tr>)}</tbody></table></div><p>Each amount covers a complete pod session. Application estimates cannot enforce a provider billing cap.</p></section>;
  return (
    <>
      <div className="panel">
        <div className="sectionline">
          <div>
            <div className="eyebrow">SECURE CLOUD</div>
            <h2>GPU catalog</h2>
          </div>
          <button
            onClick={() => act(async () => setCatalog(await api("gpus")))}
          >
            <RefreshCw size={15} />
            Refresh catalog
          </button>
        </div>
        {catalog ? (
          <>
            <p>
              {catalog.source} ·{" "}
              {new Date(catalog.at * 1000).toLocaleTimeString()}
            </p>
            <div className="scrolltable">
              <table>
                <thead>
                  <tr>
                    <th>Accelerator</th>
                    <th>VRAM</th>
                    <th>Stock</th>
                    <th>GPU / hour</th>
                  </tr>
                </thead>
                <tbody>
                  {catalog.gpus.map((g) => (
                    <tr key={g.id}>
                      <td>
                        {g.name}
                        <small>{g.id}</small>
                      </td>
                      <td>{g.vram} GB</td>
                      <td>{g.stock}</td>
                      <td>{money(g.hourly)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p>Refresh to read current GPU rates and availability from RunPod.</p>
        )}
      </div>
      <div className="panel">
        <div className="sectionline">
          <h2>Account inventory</h2>
          <button
            onClick={() => act(async () => setAccount(await api("inventory")))}
          >
            <Cloud size={15} />
            Inspect account
          </button>
        </div>
        {account ? (
          <>
            <p>
              Prepaid balance {money(account.balance.balanceAmount)} ·{" "}
              {account.pods.length} pods · {account.volumes.length} network
              volumes · {account.endpoints.length} endpoints
            </p>
            {account.pods.map((p) => (
              <p key={p.id}>
                {p.name} · {p.id} · {p.status} · {money(p.cost)}/h listed
                compute
              </p>
            ))}
            {account.volumes.map((v) => (
              <p key={v.id}>
                {v.name} · {v.size} GB ·{" "}
                {v.monthly_estimate === null
                  ? "Tier-specific rate"
                  : money(v.monthly_estimate) + "/month"}
              </p>
            ))}
            {account.endpoints.map((v) => (
              <p key={v.id}>
                {v.name} · {v.id}
              </p>
            ))}
            <h3>Abandoned ModelLab resources</h3>
            {account.abandoned.length ? (
              account.abandoned.map((p) => (
                <p key={p.id}>
                  {p.name} · {p.id}
                </p>
              ))
            ) : (
              <p>None found in this inventory read.</p>
            )}
          </>
        ) : (
          <p>
            {data.credential
              ? "Ready for an authenticated inventory read."
              : "Add a RunPod API key to inspect account resources and reconcile billing."}
          </p>
        )}
      </div>
    </>
  );
}
export function Settings({ data }: { data: Studio }) {
  return (
    <div className="panel">
      <div className="eyebrow">LOCAL WORKSPACE</div>
      <h2>Connections & storage</h2>
      {[
        [
          KeyRound,
          "RunPod account",
          data.mode === "demo" ? "Historical read-only workspace" : data.credential
            ? "Credential available for account checks"
            : "Set RUNPOD_API_KEY in the environment that starts the studio. Restart the service to pick it up.",
        ],
        [Folder, "Artifact destination", data.artifact_root],
        [
          ShieldCheck,
          "Mission budget",
          money(data.budget) +
            " including compute and storage. Quotes include a 24-hour recovery allowance and a 20% contingency.",
        ],
        [
          Cpu,
          "Training environment",
          "RunPod Secure Cloud · PyTorch 2.8 · CUDA 12.8. The local computer runs the controller and SQLite.",
        ],
      ].map(([Icon, title, value]) => {
        const I = Icon as typeof Folder;
        return (
          <div className="setting" key={String(title)}>
            <I size={19} />
            <div>
              <h3>{String(title)}</h3>
              <p>{String(value)}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
