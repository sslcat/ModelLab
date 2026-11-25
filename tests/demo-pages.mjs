import assert from "node:assert/strict";
import {mkdir,writeFile} from "node:fs/promises";

export async function verifyDemo({base,send,evaluate,wait}) {
  const pictures=process.argv.includes("--screenshots");
  if(pictures) await mkdir("docs/screenshots",{recursive:true});
  const open=async(name,check)=>{
    await send("Page.navigate",{url:base+"/#"+encodeURIComponent(name)});
    await wait("document.body.innerText.includes("+JSON.stringify(check)+")");
    await wait('document.body.innerText.includes("Historical · read-only")');
  };
  const panel=title=>"[...document.querySelectorAll('h2,h3')].find(h=>h.textContent==="+JSON.stringify(title)+").parentElement";
  const choose=async(selector,value)=>evaluate("(()=>{const e="+selector+";e.value="+JSON.stringify(value)+";e.dispatchEvent(new Event('change',{bubbles:true}));})()");
  async function capture(name,element,limit=850) {
    if(!pictures)return;
    await evaluate("document.fonts.ready.then(()=>true)");
    let clip;
    if(element) clip=await evaluate("(()=>{const e="+element+";const b=e.getBoundingClientRect();return {x:Math.max(0,b.x+scrollX-8),y:Math.max(0,b.y+scrollY-8),width:b.width+16,height:Math.min(b.height+16,"+limit+"),scale:1}})()");
    const {data}=await send("Page.captureScreenshot",{format:"png",captureBeyondViewport:!!clip,...(clip?{clip}:{})});
    await writeFile("docs/screenshots/"+name+".png",Buffer.from(data,"base64"));
  }
  await open("Overview","Receipt extraction studies");
  await capture("overview");
  await open("Datasets","Content SHA-256");
  await capture("dataset");
  await open("Training","Download configuration");
  await capture("training");
  await open("Experiments","Controlled rank study");
  await wait('document.querySelectorAll(".sizechart svg").length===4');
  const sizePanel=panel("Model size and specialization");
  const sizeHeight=await evaluate("(()=>{const e="+sizePanel+";return Math.ceil(e.querySelector(\"table\").getBoundingClientRect().bottom-e.getBoundingClientRect().top+20)})()");
  await capture("model-size",sizePanel,sizeHeight);
  await capture("rank",panel("Controlled rank study"),763);
  await choose('document.querySelectorAll(".runselect select")[0]',"rank-4");
  await choose('document.querySelectorAll(".runselect select")[1]',"rank-8");
  await wait('document.querySelectorAll(".plot polyline").length===4');
  await capture("metrics",panel("Training telemetry"),820);
  const inspect=panel("Prediction inspection");
  await wait("!!("+inspect+").querySelector('details summary')");
  await evaluate("("+inspect+").querySelector('details summary').click()");
  await wait("("+inspect+").innerText.includes('Generated')");
  await capture("prediction",inspect,540);
  await open("Cloud Resources","Historical cloud charges");
  await capture("cost");
  const state=await(await fetch(base+"/api/studio")).json();
  assert.equal(state.mode,"demo");
  assert.equal(state.active,0);
  assert.equal(state.runs.length,10);
  assert.equal(state.recorded_adapters.length,17);
  assert.equal((await fetch(base+"/api/runs",{method:"POST"})).status,405);
  await send("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});
  for(const [name,label] of [["Overview","Receipt extraction studies"],["Datasets","Preview"],["Training","Download configuration"],["Experiments","Training telemetry"],["Models","Pinned revisions"],["Evaluation","Prediction comparison"],["Cloud Resources","Historical cloud charges"],["Settings","Artifact destination"]]) {
    await open(name,label);
    assert(await evaluate("document.documentElement.scrollWidth<=innerWidth+1"),name+" overflows mobile viewport");
    if(name==="Overview") await capture("mobile");
  }
  console.log("Verified historical desktop and mobile sections, measured charts, failure inspection and read-only API.");
}
