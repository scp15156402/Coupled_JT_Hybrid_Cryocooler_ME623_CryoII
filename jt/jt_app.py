#!/usr/bin/env python3
"""Local browser interface for the JT cooler project (standard library only).

    python jt_app.py            # opens http://127.0.0.1:8765
    python jt_app.py --port 9000 --no-browser

Tabs
    1  Validate : data check + full reproduction of Zhang & Dang 2021 + scorecard   (workflow 1)
    2  Solve    : fill in or load a case (other paper, or your own inputs), solve    (workflows 2, 3)
    3  Cases    : the saved case files in cases/

It is a thin layer over jt_run.py: the same functions, the same files in results/.
Runs only on your own machine (binds to 127.0.0.1).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import jt_run as jr  # noqa: E402

CASES = ROOT / "cases"
SCRATCH = ROOT / "results" / "_app"
_fluids: dict = {}
_job = {"running": False, "log": "", "rc": None, "kind": ""}
_lock = threading.Lock()


# ---------------------------------------------------------------- actions
def api_check(_):
    return {"items": [{"status": s, "text": t} for s, t in jr.check_he3()]}


def api_cases(_):
    out = []
    for p in sorted(CASES.glob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8-sig"))
            out.append({"file": p.name, "case": {k: v for k, v in raw.items() if not k.startswith("_")}})
        except Exception as exc:  # show broken files rather than hide them
            out.append({"file": p.name, "error": str(exc)})
    return {"cases": out}


def _fluid(case):
    key = (case["fluid"], case.get("singlephase_csv"), case.get("saturation_csv"))
    if key not in _fluids:
        _fluids[key] = jr.make_case_fluid(case)
    return _fluids[key]


def api_solve(body):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = SCRATCH / "case.json"
    tmp.write_text(json.dumps(body), encoding="utf-8")
    case = jr.load_case(tmp)                      # same validation as the command line
    fluid = _fluid(case)
    base = {k: case[k] for k in ("T_C", "P_U", "T_H", "T_pre", "eta", "d_um")}
    res = jr.solve_point(fluid, base, True)
    met = jr.cycle_metrics(res, base)
    out = {"metrics": met, "flags": res.flags, "states": jr.state_table(res, len(base["eta"]))}
    if case["reported"]:
        out["comparison"] = jr.compare_reported(met, case["reported"])
    sweep = case["sweep"]
    if sweep:
        import itertools
        keys, rows = list(sweep), []
        for combo in itertools.product(*(sweep[k] for k in keys)):
            over = dict(zip(keys, combo))
            row = dict(over)
            try:
                p = jr._apply(base, over)
                r = jr.solve_point(fluid, p, True)
                row.update(jr.cycle_metrics(r, p))
                row["flags"] = "; ".join(r.flags)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        out["sweep"] = rows
    return jr._clean(out)


def api_save(body):
    tmp = SCRATCH / "case.json"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(body), encoding="utf-8")
    case = jr.load_case(tmp)                      # refuse to save an invalid case
    CASES.mkdir(exist_ok=True)
    path = CASES / (jr._slug(case["name"]) + ".json")
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return {"saved": path.name}


def _run_job(kind, cmd):
    with _lock:
        if _job["running"]:
            return {"error": "a job is already running"}
        _job.update(running=True, log="", rc=None, kind=kind)

    def work():
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            _job["log"] += line
        _job["rc"] = p.wait()
        _job["running"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"started": kind}


def api_validate(body):
    cmd = [sys.executable, "-u", "jt_run.py", "validate"]
    if body.get("quick"):
        cmd.append("--quick")
    return _run_job("validate", cmd)


def api_job(_):
    out = {"running": _job["running"], "rc": _job["rc"], "kind": _job["kind"], "log": _job["log"]}
    sc = ROOT / "results" / "validation" / "scorecard.json"
    if not _job["running"] and _job["kind"] == "validate" and sc.is_file():
        out["scorecard"] = json.loads(sc.read_text(encoding="utf-8"))
    return out


ROUTES = {("POST", "/api/check"): api_check, ("GET", "/api/cases"): api_cases,
          ("POST", "/api/solve"): api_solve, ("POST", "/api/save"): api_save,
          ("POST", "/api/validate"): api_validate, ("GET", "/api/job"): api_job}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, data, ctype="application/json"):
        raw = data if isinstance(data, bytes) else json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _handle(self, method):
        if method == "GET" and self.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        fn = ROUTES.get((method, self.path))
        if not fn:
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if method == "POST" else {}
            self._send(200, fn(body))
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>JT cooler</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#dcdcd5;--acc:#1f5fbf;--ok:#17803d;--warn:#a15c00;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--mut:#9a9a94;--card:#1f1f1e;--line:#33332f;--acc:#6ea4f4;--ok:#5fcf86;--warn:#f0b050;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 12px}
nav{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:16px}
nav button{background:none;border:0;padding:8px 14px;color:var(--mut);font:inherit;cursor:pointer;border-bottom:2px solid transparent}
nav button.on{color:var(--fg);border-color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:14px}
label{display:block;font-size:12px;color:var(--mut);margin:8px 0 2px}
input,select,textarea{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
textarea{font-family:ui-monospace,monospace;font-size:13px;min-height:60px}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0 12px}
button.p{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:8px 14px;font:inherit;cursor:pointer;margin:12px 8px 0 0}
button.s{background:none;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px 14px;font:inherit;cursor:pointer;margin:12px 8px 0 0}
button:disabled{opacity:.5}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
.wrap{overflow-x:auto}pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;max-height:340px;overflow:auto;font-size:12px;white-space:pre-wrap}
.PASS,.ok{color:var(--ok)}.WARN,.warn{color:var(--warn)}.FAIL,.bad{color:var(--bad)}.INFO,.NA{color:var(--mut)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.kv div{border:1px solid var(--line);border-radius:6px;padding:8px}
.kv b{display:block;font-size:18px}.kv span{font-size:12px;color:var(--mut)}.hide{display:none}.err{color:var(--bad);margin-top:10px;white-space:pre-wrap}
</style></head><body><main>
<h1>He-3 JT cooler</h1>
<nav><button data-t="val" class="on">1 · Validate vs paper</button><button data-t="sol">2 · Solve a case</button><button data-t="cas">Cases</button></nav>

<section id="val">
 <div class="card"><b>Data check</b><div id="chk" style="margin-top:8px"></div>
  <button class="s" id="bchk">Re-check data</button></div>
 <div class="card"><b>Reproduce Zhang &amp; Dang 2021</b>
  <p style="color:var(--mut);margin:6px 0 0">Data check, unit tests, fits, Figs 5/7/15/16, mass flux, then a scorecard.</p>
  <label><input type="checkbox" id="quick" style="width:auto"> quick mode (coarser grids, minutes → seconds)</label>
  <button class="p" id="bval">Run validation</button><span id="vst" style="color:var(--mut)"></span>
  <div id="score" class="wrap" style="margin-top:12px"></div>
  <pre id="vlog" class="hide"></pre></div>
</section>

<section id="sol" class="hide">
 <div class="card"><label>Load a saved case</label><select id="pick"><option value="">— blank / template —</option></select>
  <div class="g">
   <div><label>Name</label><input id="name" value="my cooler"></div>
   <div><label>Fluid ('he3' or CoolProp name)</label><input id="fluid" value="he3"></div>
   <div><label>T_C [K] (evaporator)</label><input id="T_C" value="1.0"></div>
   <div><label>P_U [MPa] (high pressure)</label><input id="P_U" value="0.36"></div>
   <div><label>T_H [K] (warm inlet)</label><input id="T_H" value="300"></div>
   <div><label>d [µm] (blank = no mass flow)</label><input id="d_um" value="20"></div>
  </div>
  <div class="g">
   <div><label>T_pre [K], warm→cold (one fewer than η)</label><input id="T_pre" value="80, 25, 8"></div>
   <div><label>η per CHEX, warm→cold</label><input id="eta" value="1, 1, 1, 1"></div>
  </div>
  <label>Source / notes (optional)</label><input id="source" placeholder="paper, table, or 'own design'">
  <label>Reported values to compare (JSON, optional) e.g. {"Q_C_mW": 14.7, "mdot_mg_s": 2.34}</label><textarea id="reported"></textarea>
  <label>Sweep (JSON, optional) e.g. {"P_U":[0.3,0.36,0.5],"eta_last":[1,0.97]}. Allowed: T_C, P_U, T_H, d_um, eta_last, T_pre_last</label><textarea id="sweep"></textarea>
  <button class="p" id="bsolve">Solve</button><button class="s" id="bsave">Save as case file</button><span id="sst" style="color:var(--mut)"></span>
  <div id="serr" class="err"></div></div>
 <div id="out" class="hide">
  <div class="card"><b>Result</b><div class="kv" id="met" style="margin-top:8px"></div><div id="flags" style="margin-top:8px"></div></div>
  <div class="card hide" id="cmpc"><b>Versus reported</b><div class="wrap" id="cmp"></div></div>
  <div class="card hide" id="swc"><b>Sweep</b><div class="wrap" id="sw"></div></div>
  <div class="card"><b>States</b><div class="wrap" id="sts"></div></div>
 </div>
</section>

<section id="cas" class="hide"><div class="card" id="clist">Loading…</div></section>

<script>
const $=id=>document.getElementById(id);
const api=async(m,u,b)=>{const r=await fetch(u,{method:m,headers:{'Content-Type':'application/json'},body:m=='POST'?JSON.stringify(b||{}):undefined});
 const j=await r.json();if(j.error)throw new Error(j.error);return j};
const fmt=v=>typeof v=='number'?(Math.abs(v)>=1e5||(v!==0&&Math.abs(v)<1e-3)?v.toExponential(3):+v.toPrecision(5)):(v==null?'':String(v));
function table(rows,cols,cls){cols=cols||[...new Set(rows.flatMap(r=>Object.keys(r)))];
 const t=document.createElement('table'),h=t.createTHead().insertRow();
 cols.forEach(c=>{const th=document.createElement('th');th.textContent=c;h.appendChild(th)});
 const b=t.createTBody();rows.forEach(r=>{const tr=b.insertRow();cols.forEach(c=>{const td=tr.insertCell();td.textContent=fmt(r[c]);
  if(cls&&c==cls)td.className=String(r[c]).replace('/','')});});return t}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
 ['val','sol','cas'].forEach(s=>$(s).classList.toggle('hide',s!=b.dataset.t));if(b.dataset.t=='cas')loadCases()});

async function check(){$('chk').textContent='…';try{const j=await api('POST','/api/check');
 $('chk').replaceChildren(...j.items.map(i=>{const d=document.createElement('div');d.innerHTML='<b class="'+i.status+'"></b> ';
  d.firstChild.textContent=i.status;d.append(i.text);return d}))}catch(e){$('chk').innerHTML='<span class="bad"></span>';$('chk').firstChild.textContent=e.message}}
$('bchk').onclick=check;

let poll;
$('bval').onclick=async()=>{try{await api('POST','/api/validate',{quick:$('quick').checked})}catch(e){$('vst').textContent=e.message;return}
 $('bval').disabled=true;$('vlog').classList.remove('hide');$('score').replaceChildren();
 poll=setInterval(async()=>{const j=await api('GET','/api/job');$('vlog').textContent=j.log;$('vlog').scrollTop=1e9;
  $('vst').textContent=j.running?' running…':' finished (exit '+j.rc+')';
  if(!j.running){clearInterval(poll);$('bval').disabled=false;
   if(j.scorecard){const n=s=>j.scorecard.filter(r=>r.status==s).length;
    const p=document.createElement('p');p.innerHTML='<b class="ok">'+n('PASS')+' pass</b> · <b class="warn">'+n('WARN')+' warn</b> · <b class="bad">'+n('FAIL')+' fail</b> · '+n('N/A')+' n/a (tolerances are provisional)';
    $('score').replaceChildren(p,table(j.scorecard,['section','check','ours','paper','tol','unit','status'],'status'))}}},1500)};

function readCase(){const num=id=>$(id).value.trim()===''?null:parseFloat($(id).value);
 const list=id=>$(id).value.split(',').map(s=>s.trim()).filter(Boolean).map(parseFloat);
 const js=id=>$(id).value.trim()?JSON.parse($(id).value):{};
 const c={name:$('name').value.trim()||'my cooler',fluid:$('fluid').value.trim(),T_C:num('T_C'),P_U:num('P_U'),T_H:num('T_H'),
  T_pre:list('T_pre'),eta:list('eta'),d_um:num('d_um'),reported:js('reported'),sweep:js('sweep')};
 if($('source').value.trim())c.source=$('source').value.trim();return c}
function fillCase(c){$('name').value=c.name||'';$('fluid').value=c.fluid||'he3';$('T_C').value=c.T_C??'';$('P_U').value=c.P_U??'';
 $('T_H').value=c.T_H??300;$('d_um').value=c.d_um??'';$('T_pre').value=(c.T_pre||[]).join(', ');$('eta').value=(c.eta||[]).join(', ');
 $('source').value=c.source||'';$('reported').value=c.reported?JSON.stringify(c.reported):'';$('sweep').value=c.sweep?JSON.stringify(c.sweep):''}
let CASES=[];
async function loadCases(){const j=await api('GET','/api/cases');CASES=j.cases;
 $('pick').replaceChildren(new Option('— blank / template —',''),...CASES.filter(c=>c.case&&!c.file.startsWith('_')).map((c,i)=>new Option(c.case.name+'  ['+c.file+']',c.file)));
 const box=$('clist');box.replaceChildren();CASES.forEach(c=>{const d=document.createElement('div');d.style.margin='6px 0';
  if(c.error){d.className='bad';d.textContent=c.file+': '+c.error}else{const b=document.createElement('button');b.className='s';b.style.margin='0 8px 0 0';
   b.textContent='Open';b.onclick=()=>{fillCase(c.case);document.querySelector('[data-t=sol]').click()};d.append(b,c.file+' — '+(c.case.name||'')+(c.case.source?' · '+c.case.source:''))}
  box.appendChild(d)})}
$('pick').onchange=()=>{const c=CASES.find(x=>x.file==$('pick').value);if(c&&c.case)fillCase(c.case)};

$('bsolve').onclick=async()=>{$('serr').textContent='';$('sst').textContent=' solving…';$('bsolve').disabled=true;
 try{const j=await api('POST','/api/solve',readCase());$('out').classList.remove('hide');
  const m=j.metrics,lab={q_C_kJ_kg:['q_C','kJ/kg'],mdot_mg_s:['ṁ','mg/s'],Q_C_mW:['Q_C','mW'],T_JT_inlet_K:['T before JT','K'],P_D_kPa:['P_D','kPa'],G_kg_m2_s:['G','kg/m²/s']};
  $('met').replaceChildren(...Object.entries(lab).filter(([k])=>m[k]!=null).map(([k,[n,u]])=>{const d=document.createElement('div');
   d.innerHTML='<span></span><b></b>';d.firstChild.textContent=n+' ['+u+']';d.lastChild.textContent=fmt(m[k]);return d}));
  Object.keys(m).filter(k=>/^Q_pre/.test(k)).forEach(k=>{const d=document.createElement('div');d.innerHTML='<span></span><b></b>';d.firstChild.textContent=k.replace('_',' ')+' [mW]';d.lastChild.textContent=fmt(m[k]);$('met').appendChild(d)});
  $('flags').innerHTML=j.flags.length?'<span class="warn">Flags:</span> ':'<span class="ok">No flags.</span>';
  if(j.flags.length)$('flags').append(j.flags.join(' · '));
  $('cmpc').classList.toggle('hide',!j.comparison);if(j.comparison)$('cmp').replaceChildren(table(j.comparison,['metric','ours','reported','difference','difference_percent']));
  $('swc').classList.toggle('hide',!j.sweep);if(j.sweep)$('sw').replaceChildren(table(j.sweep));
  $('sts').replaceChildren(table(j.states))}catch(e){$('serr').textContent=e.message}
 $('sst').textContent='';$('bsolve').disabled=false};
$('bsave').onclick=async()=>{$('serr').textContent='';try{const j=await api('POST','/api/save',readCase());$('sst').textContent=' saved cases/'+j.saved;loadCases()}catch(e){$('serr').textContent=e.message}};
check();loadCases();
</script></main></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Local browser interface for the JT cooler project.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"JT cooler interface at {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
