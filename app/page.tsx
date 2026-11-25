"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Boxes,
  ChartNoAxesCombined,
  ChevronRight,
  Cloud,
  Cpu,
  Database,
  FlaskConical,
  Layers3,
  LayoutGrid,
  Menu,
  Plus,
  Settings2,
  SlidersHorizontal,
  Terminal,
  Wallet,
  X,
} from "lucide-react";
import { api, money, Studio } from "./types";
import { Datasets, Training, Resources, Settings } from "./views";
import { Evaluation, Experiments, Exports } from "./results";
import { SizeStudy } from "./size-study";
import sizeModels from "../research/size_models.json";

const nav = [
  ["Overview", LayoutGrid],
  ["Datasets", Database],
  ["Training", SlidersHorizontal],
  ["Experiments", FlaskConical],
  ["Models", Boxes],
  ["Evaluation", ChartNoAxesCombined],
  ["Cloud Resources", Cloud],
  ["Settings", Settings2],
] as const;
type Page = (typeof nav)[number][0];
function Terrain() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const draw = () => {
      const ctx = c.getContext("2d");
      if (!ctx) return;
      const w = c.clientWidth,
        h = c.clientHeight,
        dpr = window.devicePixelRatio || 1;
      c.width = w * dpr;
      c.height = h * dpr;
      ctx.scale(dpr, dpr);
      const pt = (x: number, y: number) => {
        const z =
          Math.exp(-((x - 0.6) ** 2 * 8 + (y - 0.5) ** 2 * 10)) * 0.8 +
          Math.exp(-((x - 0.25) ** 2 * 22 + (y - 0.7) ** 2 * 26)) * 0.29 +
          0.06 * Math.sin(x * 12 + y * 4);
        return [
          w * (0.1 + x * 0.93 + (y - 0.5) * 0.43),
          h * (0.66 + y * 0.43 - z * 0.62 - x * 0.11),
        ];
      };
      for (let y = 0; y <= 1; y += 0.033) {
        ctx.beginPath();
        ctx.strokeStyle = "rgba(165,185,255," + (0.22 + 0.6 * y) + ")";
        ctx.lineWidth = 0.9;
        for (let x = -0.12; x <= 1.14; x += 0.016) {
          const [a, b] = pt(x, y);
          if (x === -0.12) ctx.moveTo(a, b);
          else ctx.lineTo(a, b);
        }
        ctx.stroke();
      }
      for (let x = -0.12; x < 1.15; x += 0.035) {
        ctx.beginPath();
        ctx.strokeStyle = "rgba(137,166,250,.25)";
        for (let y = 0; y <= 1; y += 0.024) {
          const [a, b] = pt(x, y);
          if (!y) ctx.moveTo(a, b);
          else ctx.lineTo(a, b);
        }
        ctx.stroke();
      }
    };
    const obs = new ResizeObserver(draw);
    obs.observe(c);
    draw();
    return () => obs.disconnect();
  }, []);
  return <canvas aria-hidden="true" ref={ref} />;
}
function Overview({
  data,
  go,
  open,
}: {
  data: Studio;
  go: (p: Page) => void;
  open: (id: string) => void;
}) {
  const ds = data.datasets.find((d) => d.test > 0) ?? data.datasets[0];
  const cards = [
    [
      Database,
      "Datasets",
      data.datasets.length,
      "Validated datasets",
      "Datasets",
    ],
    [
      FlaskConical,
      "Experiments",
      data.runs.length,
      data.active + " active allocations",
      "Experiments",
    ],
    [
      Wallet,
      data.mode === "demo" ? "Recorded spend" : "Budget available",
      money(data.mode === "demo" ? data.reserved : Math.max(0, data.budget - data.reserved)),
      "Of " + money(data.budget) + " mission budget",
      "Cloud Resources",
    ],
  ] as const;
  return (
    <>
      <div className="hero">
        <Terrain />
        <div className="herocopy">
          <div className="herokicker">
            <span className="lightdot" /> {data.mode === "demo" ? "HISTORICAL RESEARCH" : "RUNPOD WORKSPACE"}
          </div>
          <h2>{data.mode === "demo" ? "Receipt extraction studies" : "Remote training lab"}</h2>
          <p>
            {data.mode === "demo" ? "Explore the configurations, predictions and bills from three completed RunPod studies." : "Configure a fine-tuning run, review its cost, and track the results from one workspace."}
          </p>
          <button onClick={() => go(data.mode === "demo" ? "Evaluation" : "Training")}>
            <Plus size={16} />
            {data.mode === "demo" ? "Explore results" : "New experiment"}
          </button>
        </div>
      </div>
      <div className="cards">
        {cards.map(([I, k, v, n, target]) => (
          <button key={k} className="stat" onClick={() => go(target)}>
            <div>
              <I size={17} />
              <span>{k}</span>
              <ArrowUpRight size={14} />
            </div>
            <strong>{v}</strong>
            <small>{n}</small>
          </button>
        ))}
      </div>
      <div className="overviewlower">
        <section className="panel">
          <div className="sectionline">
            <h2>{data.mode === "demo" ? "Recorded experiments" : "Recent experiments"}</h2>
            <button
              className="icon"
              title="All experiments"
              onClick={() => go("Experiments")}
            >
              <ArrowUpRight size={17} />
            </button>
          </div>
          {data.runs.length ? (
            data.runs.slice(0, 4).map((r) => (
              <button className="runrow" key={r.id} onClick={() => open(r.id)}>
                <FlaskConical size={18} />
                <span>
                  <b>{r.config.name}</b>
                  <small>
                    {r.config.method} · {r.dataset_name}
                  </small>
                </span>
                <i>{r.status}</i>
              </button>
            ))
          ) : (
            <div className="emptyruns">
              <span className="emptyicon">
                <FlaskConical size={23} />
              </span>
              <h3>No experiments yet</h3>
              <p>
                Your saved configurations and measured runs will appear here.
              </p>
              <button className="textbtn" onClick={() => go("Training")}>
                Configure a run <ArrowRight size={14} />
              </button>
            </div>
          )}
        </section>
        {ds && (
          <section className="panel ready">
            <div className="sectionline">
              <h2>{data.mode === "demo" ? "Research dataset" : "Ready for training"}</h2>
              <Database size={17} />
            </div>
            <div className="datasetart" aria-hidden="true">
              <Layers3 size={38} strokeWidth={1.35} />
              <span>JSONL</span>
            </div>
            <h3>{ds.name}</h3>
            <p>
              Original receipt notes paired with structured inventory records.
            </p>
            <div className="sectionline">
              <span className="minitext">
                {ds.train} train · {ds.validation} validation · {ds.test} held out
              </span>
              <button
                className="icon"
                title="Open dataset library"
                onClick={() => go("Datasets")}
              >
                <ArrowRight size={16} />
              </button>
            </div>
          </section>
        )}
      </div>
    </>
  );
}
export default function Home() {
  const [page, setPage] = useState<Page>("Overview"),
    [data, setData] = useState<Studio | null>(null),
    [selected, setSelected] = useState(""),
    [message, msg] = useState(""),
    [busy, setBusy] = useState(false),
    [mobile, menu] = useState(false);
  const refresh = useCallback(
    async () => setData(await api<Studio>("studio")),
    [],
  );
  useEffect(() => {
    const read = () => {
      const name = decodeURIComponent(location.hash.slice(1));
      setPage(nav.find((n) => n[0] === name)?.[0] ?? "Overview");
    };
    read();
    window.addEventListener("hashchange", read);
    refresh().catch((e) => msg(e.message));
    return () => window.removeEventListener("hashchange", read);
  }, [refresh]);
  useEffect(() => {
    if (!data?.active) return;
    const timer = setInterval(
      () => refresh().catch((e) => msg(e.message)),
      4000,
    );
    return () => clearInterval(timer);
  }, [data?.active, refresh]);
  function go(p: Page) {
    location.hash = p;
    setPage(p);
    menu(false);
    msg("");
    window.scrollTo(0, 0);
  }
  function open(id: string) {
    setSelected(id);
    go("Experiments");
  }
  function act(work: () => Promise<void>) {
    if (busy) return;
    setBusy(true);
    msg("");
    work()
      .catch((e) => msg(e.message))
      .finally(() => setBusy(false));
  }
  const props = data ? { data, act, refresh, open } : null;
  return (
    <div className="shell">
      <aside className={mobile ? "side mobileopen" : "side"}>
        <div className="brand">
          <span className="brandicon">
            <Layers3 size={23} />
          </span>
          ModelLab
          <button
            aria-label="Close menu"
            className="mobclose"
            onClick={() => menu(false)}
          >
            <X size={18} />
          </button>
        </div>
        <div className="navlabel">WORKSPACE</div>
        <nav>
          {nav.map(([name, I], idx) => (
            <a
              key={name}
              href={"#" + name}
              className={
                (page === name ? "current " : "") + (idx === 6 ? "divider" : "")
              }
              aria-current={page === name ? "page" : undefined}
              onClick={() => go(name)}
            >
              <I size={17} />
              {name}
            </a>
          ))}
        </nav>
        <div className="sidefoot">
          <Terminal size={16} />
          <span>
            Local studio<small>ModelLab v0.1</small>
          </span>
        </div>
      </aside>
      <main>
        <header>
          <button
            aria-label="Open menu"
            className="hamb"
            onClick={() => menu(true)}
          >
            <Menu size={19} />
          </button>
          <span>Workspace</span>
          <ChevronRight size={14} />
          <b>{page}</b>
          <div className="host">
            <i /> {data?.mode === "demo" ? "Historical · read-only" : "Local workspace"}
          </div>
        </header>
        <div className="body">
          <div className="pagetitle">
            <div className="eyebrow">TRAINING STUDIO</div>
            <h1>{page === "Overview" ? "Workspace overview" : page}</h1>
          </div>
          {message && (
            <div className="error" role="alert">
              {message}
              <button onClick={() => msg("")} aria-label="Dismiss">
                <X size={15} />
              </button>
            </div>
          )}
          {busy && (
            <div className="progress" role="status">
              Request in progress…
            </div>
          )}
          {!data || !props ? (
            <div className="panel">
              Connecting to the local controller…{" "}
              <button onClick={() => act(refresh)}>Reconnect</button>
            </div>
          ) : (
            <>
              {page === "Overview" && (
                <Overview data={data} go={go} open={open} />
              )}
              {page === "Datasets" && <Datasets {...props} />}
              {page === "Training" && <Training {...props} />}
              {(page === "Experiments" || page === "Evaluation") && <SizeStudy runs={data.runs} />}
              {page === "Experiments" && (
                <Experiments {...props} selected={selected} />
              )}
              {page === "Models" && (
                <>
                  <div className="sectionline">
                    <h2>Base model registry</h2>
                    <span className="sub">Pinned revisions</span>
                  </div>
                  <div className="modelgrid">
                    {Object.entries(data.models).map(([m, rev]) => (
                      <div key={m} className="panel">
                        <span className="emptyicon">
                          <Cpu size={22} />
                        </span>
                        <h2>{m.split("/")[1]}</h2>
                        <p>
                          {m.startsWith("Qwen")
                            ? ((sizeModels.find(x => x.model === m)?.total ?? 0) / 1e9).toFixed(2) + "B parameters · Qwen"
                            : "7.25B parameters · Mistral · comparison candidate"}
                        </p>
                        <div className="pills">
                          <span>{m === "Qwen/Qwen2.5-3B-Instruct" ? "Qwen Research License" : "Apache 2.0"}</span>
                          <span>Remote weights</span>
                        </div>
                        <code>{rev}</code>
                      </div>
                    ))}
                  </div>
                  <h2>{data.mode === "demo" ? "Recorded adapters" : "Exported adapters"}</h2>
                  {data.mode === "demo" && <p>The public evidence includes adapter hashes, byte sizes and trainable counts. Adapter weights remain in the research archive.</p>}
                  {data.mode === "demo" ? <div className="panel researchscroll"><table><thead><tr><th>Session</th><th>Rank</th><th>Seed</th><th>Trainable</th><th>Adapter bytes</th><th>SHA-256</th></tr></thead><tbody>{data.recorded_adapters?.map(a=><tr key={a.run+"-"+a.seed}><td><a href={"/api/evidence/"+a.run}>{a.run}</a></td><td>{a.rank}</td><td>{a.seed}</td><td>{a.trainable.toLocaleString()}</td><td>{a.bytes.toLocaleString()}</td><td><details><summary>Digest</summary><code>{a.sha256}</code></details></td></tr>)}</tbody></table></div> : data.runs.some((r) => r.adapter_available) ? (
                    data.runs
                      .filter((r) => r.adapter_available)
                      .map((r) => <Exports key={r.id} r={r} />)
                  ) : (
                    <div className="panel">
                      <p>
                        Verified adapters will appear here after remote
                        training.
                      </p>
                    </div>
                  )}
                </>
              )}
              {page === "Evaluation" && <Evaluation {...props} />}
              {page === "Cloud Resources" && <Resources {...props} />}
              {page === "Settings" && <Settings data={data} />}
            </>
          )}
        </div>
      </main>
    </div>
  );
}
