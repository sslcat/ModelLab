/* Offline browser acceptance. Uses installed Chromium + Node's CDP pipe, no test GPU or cloud requests. */
import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createServer } from "node:net";
import { once } from "node:events";
import assert from "node:assert/strict";

const demo = process.argv.includes("--demo");
const dir = await mkdtemp(join(tmpdir(), "modellab-ui-"));
const delay = (ms) => new Promise((done) => setTimeout(done, ms));
const server = createServer();
server.listen(0, "127.0.0.1");
await once(server, "listening");
const port = server.address().port;
await new Promise((done) => server.close(done));
const base = "http://127.0.0.1:" + port;
const py =
  process.env.MODELLAB_PYTHON ||
  (process.platform === "win32"
    ? ".venv/Scripts/python.exe"
    : ".venv/bin/python");
const backend = spawn(
  py,
  [
    "-B",
    "-m",
    "uvicorn",
    demo ? "studio.demo:app" : "studio.api:app",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
  ],
  {
    env: { ...process.env, MODELLAB_OFFLINE: "1", RUNPOD_API_KEY: "", MODELLAB_STATE: join(dir, "db") },
    stdio: "ignore",
    windowsHide: true,
  },
);
let browser;
try {
  for (let i = 0; i < 350; i++) {
    try {
      if ((await fetch(base + "/api/studio")).ok) break;
    } catch {}
    await delay(100);
  }
  const exe =
    process.env.CHROME_BIN ||
    [
      "C:/Program Files/Google/Chrome/Application/chrome.exe",
      "/usr/bin/chromium",
      "/usr/bin/google-chrome",
    ].find(existsSync);
  assert(exe, "Set CHROME_BIN to your installed Chromium executable");
  browser = spawn(
    exe,
    [
      "--headless=new",
      "--disable-gpu",
      "--disable-extensions",
      "--disable-background-networking",
      "--no-first-run",
      "--no-default-browser-check",
      "--remote-debugging-pipe",
      "--user-data-dir=" + join(dir, "browser"),
      "about:blank",
    ],
    {
      stdio: ["ignore", "ignore", "ignore", "pipe", "pipe"],
      windowsHide: true,
    },
  );
  const pending = new Map(),
    errors = [];
  let sequence = 0,
    buffer = "";
  browser.stdio[4].setEncoding("utf8");
  browser.stdio[4].on("data", (chunk) => {
    buffer += chunk;
    let at;
    while ((at = buffer.indexOf("\0")) >= 0) {
      const msg = JSON.parse(buffer.slice(0, at));
      buffer = buffer.slice(at + 1);
      if (msg.id && pending.has(msg.id)) {
        const { done, fail, timer } = pending.get(msg.id);
        pending.delete(msg.id);
        clearTimeout(timer);
        msg.error ? fail(Error(msg.error.message)) : done(msg.result);
      }
      if (msg.method === "Runtime.exceptionThrown")
        errors.push(msg.params.exceptionDetails.text);
    }
  });
  function cdp(method, params = {}, sessionId) {
    return new Promise((done, fail) => {
      const id = ++sequence;
      const timer = setTimeout(() => {
        pending.delete(id);
        fail(Error("Timeout in " + method));
      }, 60000);
      pending.set(id, { done, fail, timer });
      browser.stdio[3].write(
        JSON.stringify({ id, method, params, sessionId }) + "\0",
      );
    });
  }
  const { targetId } = await cdp("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await cdp("Target.attachToTarget", {
    targetId,
    flatten: true,
  });
  const send = (m, p) => cdp(m, p, sessionId);
  async function evaluate(expression) {
    const r = await send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (r.exceptionDetails)
      throw Error(
        r.exceptionDetails.exception?.description || r.exceptionDetails.text,
      );
    return r.result.value;
  }
  async function wait(expression) {
    for (let i = 0; i < 350; i++) {
      if (await evaluate(expression)) return;
      await delay(80);
    }
    throw Error("Missing UI state: " + expression);
  }
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1365,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  if (demo) {
    const {verifyDemo}=await import("./demo-pages.mjs");
    await verifyDemo({base,send,evaluate,wait});
  } else {
  await send("Page.navigate", { url: base });
  await wait('document.body.innerText.includes("Stockroom receipts")');
  assert(await evaluate('document.querySelector("canvas").width>0'));
  if (process.env.MODELLAB_SCREENSHOT)
    await writeFile(
      process.env.MODELLAB_SCREENSHOT,
      Buffer.from((await send("Page.captureScreenshot")).data, "base64"),
    );
  for (const [page, expected] of [
    ["Datasets", "Dataset library"],
    ["Training", "Hyperparameters"],
    ["Experiments", "No experiments yet"],
    ["Models", "Pinned revisions"],
    ["Evaluation", "No paired evaluations yet"],
    ["Cloud Resources", "Account inventory"],
    ["Settings", "Artifact destination"],
  ]) {
    await evaluate(
      'document.querySelector("nav a[href=\\"#' + page + '\\"]").click()',
    );
    await wait(
      "document.body.innerText.includes(" + JSON.stringify(expected) + ")",
    );
  }
  await evaluate(
    'document.querySelector("nav a[href=\\"#Datasets\\"]").click()',
  );
  await wait('document.body.innerText.includes("Import dataset")');
  await evaluate(
    '[...document.querySelectorAll("button")].find(b=>b.textContent.includes("Import dataset")).click()',
  );
  await wait('!!document.querySelector("input[name=file]")');
  await evaluate(
    '(()=>{const f=document.querySelector("form");f.elements.name.value="UI acceptance";f.elements.license.value="MIT";const dt=new DataTransfer();dt.items.add(new File(["prompt,completion\\na,b\\nc,d\\ne,f"],"notes.csv",{type:"text/csv"}));f.elements.file.files=dt.files;f.elements.public.checked=true;f.requestSubmit()})()',
  );
  await wait(
    'document.body.innerText.includes("UI acceptance") && !document.querySelector("input[name=file]")',
  );
  await evaluate(
    'document.querySelector("nav a[href=\\"#Training\\"]").click()',
  );
  await wait('!!document.querySelector("input[name=epochs]")');
  await evaluate('document.querySelector("form").requestSubmit()');
  await wait('document.body.innerText.includes("Get live quote")');
  assert.equal(
    (await (await fetch(base + "/api/studio")).json()).runs.length,
    1,
  );
  await send("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await evaluate(
    'document.querySelector("button[aria-label=\\"Open menu\\"]").click()',
  );
  await wait('!!document.querySelector(".mobileopen")');
  await evaluate(
    'document.querySelector("nav a[href=\\"#Overview\\"]").click()',
  );
  await wait('document.body.innerText.includes("Remote training lab")');
  assert(
    await evaluate("document.documentElement.scrollWidth<=window.innerWidth+1"),
    "Mobile overflow",
  );
  }
  assert.deepEqual(errors, []);
  if (!demo) console.log(
    "Passed: eight sections, real CSV import, persisted draft, empty telemetry, responsive navigation, no browser exceptions.",
  );
  await cdp("Browser.close").catch(() => {});
} finally {
  if (browser?.exitCode === null) {
    browser.kill();
    await Promise.race([once(browser, "exit"), delay(1500)]);
  }
  if (backend.exitCode === null) {
    backend.kill();
    await Promise.race([once(backend, "exit"), delay(1500)]);
  }
  await rm(dir, {
    recursive: true,
    force: true,
    maxRetries: 12,
    retryDelay: 250,
  });
}
