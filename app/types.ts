export type Dataset = {
  protocol?: string;
  id: string;
  name: string;
  staged: boolean;
  license: string;
  rows: number;
  train: number;
  validation: number;
  test: number;
  duplicates: number;
  fingerprint: string;
  split_id: string;
  seed: number | null;
  tokens: Record<string, number>;
  preview: { prompt: string; completion: string }[];
};
export type Config = {
  name: string;
  dataset: string;
  model: string;
  method: string;
  gpu: string;
  minutes: number;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  grad_accum: number;
  rank: number;
  alpha: number;
  dropout: number;
  seq_len: number;
  seed: number;
};
export type Quote = {
  gpu: string;
  gpu_hourly: number;
  storage_hourly: number;
  stopped_hourly: number;
  minutes: number;
  max_usd: number;
  token: string;
  expires: number;
  stock: string;
  account: { pods: number; volumes: number; endpoints: number };
};
export type Metric = Record<string, number>;
export type Arm = {
  correct: number;
  total: number;
  examples: {
    prompt: string;
    prediction: string;
    expected: string;
    correct: boolean;
  }[];
};
export type Run = {
  historical?: boolean;
  id: string;
  created: number;
  status: string;
  config: Config;
  dataset_name: string;
  detail: string;
  pod: string | null;
  metrics: Metric[];
  quote: Quote | null;
  actual: { totalAmount: number; diskAmount: number } | null;
  estimated_usage?: number;
  stopped_storage_estimate?: number;
  evaluation?: { base: Arm; tuned?: Arm; split: string; limitations: string };
  archived: boolean;
  adapter_available?: boolean;
  sync?: string;
  archive?: string;
};
export type Studio = {
  mode?: "demo" | "live";
  recorded_adapters?: {run:string;model:string;rank:number;seed:number;bytes:number;trainable:number;sha256:string}[];
  datasets: Dataset[];
  runs: Run[];
  models: Record<string, string>;
  budget: number;
  reserved: number;
  credential: boolean;
  one_drive: string;
  artifact_root: string;
  active: number;
};
export type GPU = {
  id: string;
  name: string;
  vram: number;
  hourly: number;
  stock: string;
};
export const active = (status: string) =>
  [
    "authorized",
    "creating",
    "allocated",
    "running",
    "archiving",
    "deleting",
    "stopping",
    "recovery",
    "uncertain",
  ].includes(status);
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(
    "/api/" + path,
    body === undefined
      ? { cache: "no-store" }
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-ModelLab": "studio",
          },
          body: JSON.stringify(body),
        },
  );
  const data = await res.json();
  if (!res.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : Array.isArray(data.detail)
          ? data.detail.map((i: { msg: string }) => i.msg).join(" · ")
          : "Request failed",
    );
  return data;
}
export const money = (v: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(v);
