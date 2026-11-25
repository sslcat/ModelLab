// Run against an isolated MathJax 3 installation supplied by the caller.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {createRequire} = require("node:module");
const load = createRequire(path.resolve(process.argv[2], "package.json"));
const {mathjax}=load("mathjax-full/js/mathjax.js");
const {TeX}=load("mathjax-full/js/input/tex.js");
const {SVG}=load("mathjax-full/js/output/svg.js");
const {liteAdaptor}=load("mathjax-full/js/adaptors/liteAdaptor.js");
const {RegisterHTMLHandler}=load("mathjax-full/js/handlers/html.js");
const {AllPackages}=load("mathjax-full/js/input/tex/AllPackages.js");
const adaptor=liteAdaptor();
RegisterHTMLHandler(adaptor);
const input=new TeX({packages:AllPackages,formatError:(_jax,error)=>{throw error}});
const doc=mathjax.document("",{InputJax:input,OutputJax:new SVG({fontCache:"none"})});
const src=fs.readFileSync("docs/mathematics.md","utf8").replaceAll("\r\n","\n");
let count=0;
const rest=src.replace(/\$\$\n([\s\S]*?)\n\$\$|\$([^$\n]+)\$/g,(_,block,inline)=>{
  const svg=adaptor.outerHTML(doc.convert(block??inline,{display:block!==undefined}));
  assert(svg.includes("<svg")&&!svg.includes("merror"));
  count++;
  return "";
});
assert(!rest.includes("$"),"Unbalanced math delimiters");
assert(count>20);
console.log("Rendered "+count+" inline and display expressions to SVG with MathJax.");
