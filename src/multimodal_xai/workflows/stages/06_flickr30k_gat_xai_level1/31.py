import io, inspect, base64, html as _html, urllib.request
from IPython.display import HTML, display  # import lại tại chỗ để cell chạy độc lập

TOP_K_DEMO = 5
_CLIP_TEXT = {}

def encode_free_text(text):
    """Mã hóa truy vấn bằng đúng CLIP mà nhóm đã dùng để trích đặc trưng (đóng băng)."""
    if "model" not in _CLIP_TEXT:
        try:
            import open_clip
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
            import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        _CLIP_TEXT["model"] = m.eval().to(DEVICE)
        _CLIP_TEXT["tok"] = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    with torch.no_grad():
        t = _CLIP_TEXT["model"].encode_text(_CLIP_TEXT["tok"]([text]).to(DEVICE)).float()
    return F.normalize(t, dim=-1).cpu().squeeze(0)

def search_and_explain(q_raw, k=TOP_K_DEMO, name=MAIN_RUN, query_sentence=None, query_image_index=None):
    model, a = MODELS[name], ALPHA[name]
    q_proj = model.encode_query(q_raw[None].to(DEVICE)).float().cpu().squeeze(0)
    cosine = T_IMG @ q_raw
    graph = GEMB_TEST[name] @ q_proj
    final = (1 - a) * cosine + a * graph
    results = []
    for idx in final.topk(k).indices.tolist():
        labels = region_labels(test, idx, query_sentence if idx == query_image_index else None)
        ex = explain(model, test_graphs_for(name)[idx], q_proj, a, K_STAR, labels)
        assert abs(ex["S_full"] - float(graph[idx])) < 1e-4      # giải thích khớp đúng điểm đã dùng để xếp hạng
        results.append({"image_index": idx, "image_id": test["image_ids"][idx], "cosine": float(cosine[idx]),
                        "clip_rank": int((cosine > cosine[idx]).sum()) + 1,
                        "graph": float(graph[idx]), "final": float(final[idx]), "explanation": ex, "labels": labels})
    return results

# ---------- Trang kết quả: 2 tab (Graph | Ranking), đồ thị tương tác kiểu Obsidian (thư viện force-graph) ----------
_FG_URLS = ["https://cdn.jsdelivr.net/npm/force-graph@1.52.0/dist/force-graph.min.js",
            "https://unpkg.com/force-graph@1.52.0/dist/force-graph.min.js"]
_FG_JS = {}

def _force_graph_js():
    if "js" not in _FG_JS:
        local = os.environ.get("XAI_FORCEGRAPH_JS")
        cache = WORK / "force-graph.min.js"
        if local and Path(local).exists():
            _FG_JS["js"] = Path(local).read_text(encoding="utf-8")
        elif cache.exists():
            _FG_JS["js"] = cache.read_text(encoding="utf-8")
        else:
            _FG_JS["js"] = None
            for url in _FG_URLS:
                try:
                    js = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
                    cache.write_text(js, encoding="utf-8"); _FG_JS["js"] = js; break
                except Exception:
                    continue
    js = _FG_JS["js"]
    return f"<script>{js}</script>" if js else f'<script src="{_FG_URLS[0]}"></script>'

_REL_VI = {3: "chồng lấn", 4: "nằm bên trái", 5: "nằm bên phải", 6: "nằm phía trên", 7: "nằm phía dưới"}

def _region_name(labels, n):
    """Tên nút vùng, bắt đầu bằng đúng mã R trên khung ảnh (R0, R1, ...)."""
    parts = labels.get(n, f"R{n - 1}").split(":")
    if len(parts) > 2:
        return f"{parts[0]} · {parts[2]}"
    return f"{parts[0]} · {parts[1].rstrip('*')}" if len(parts) > 1 else parts[0]

def _graph_payload(r):
    ex, labels = r["explanation"], r["labels"]
    rels = [x for x in ex["relations"] if x["in_explanation"]]
    nodes = {0: {"id": 0, "name": f"ẢNH {r['image_id']}", "kind": "image", "delta": 0.0}}
    links = []
    for x in rels:
        u, v = x["nodes"]
        for n in (u, v):
            if n and n not in nodes:
                nodes[n] = {"id": n, "name": _region_name(labels, n), "kind": "region", "delta": 0.0}
        for n in {u, v} - {0}:
            nodes[n]["delta"] += max(x["delta_graph"], 0.0)
        if x["kind"] == "star":
            links.append({"source": v, "target": 0, "rel": "thuộc", "delta": x["delta_graph"], "hi": True,
                          "tip": f"{_region_name(labels, v)} → thuộc → ảnh · Δgraph {x['delta_graph']:+.3f}"})
        else:
            t = x["type_ids"].get(f"{u}->{v}")
            links.append({"source": u, "target": v, "rel": _REL_VI.get(t, "?"), "delta": x["delta_graph"], "hi": True,
                          "tip": f"{_region_name(labels, u)} → {_REL_VI.get(t, '?')} → {_region_name(labels, v)} · "
                                 f"Δgraph {x['delta_graph']:+.3f}"})
    star_nodes = {l["source"] for l in links if l["target"] == 0}
    for n in nodes:
        if n and n not in star_nodes:      # đường đi tiếp về ảnh qua cạnh vùng–ảnh (không nằm trong top-K)
            links.append({"source": n, "target": 0, "rel": "thuộc", "delta": 0.0, "hi": False,
                          "tip": f"{nodes[n]['name']} → thuộc → ảnh (đường đi tiếp, không nằm trong giải thích)"})
    vmax = max([nd["delta"] for nd in nodes.values()] + [1e-9])
    for nd in nodes.values():
        nd["val"] = 9 if nd["kind"] == "image" else 1.5 + 6 * nd["delta"] / vmax
        nd["tip"] = nd["name"] + ("" if nd["kind"] == "image" else f" · tổng Δ {nd['delta']:+.3f}")
    return {"nodes": list(nodes.values()), "links": links}

def _slide(r, rank):
    meta = test["graph_metadata"][r["image_index"]]
    graph = _graph_payload(r)
    img = _open_image(r["image_id"])
    src = None
    if img is not None:
        img.thumbnail((720, 720)); buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
        src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    boxes = []
    for nd in graph["nodes"]:
        if nd["kind"] == "region":
            x1, y1, x2, y2 = meta["regions"][nd["id"] - 1]["box_normalized"]
            boxes.append({"n": nd["id"], "tag": nd["name"].split(" · ")[0], "name": nd["name"],
                          "l": x1 * 100, "t": y1 * 100, "w": (x2 - x1) * 100, "h": (y2 - y1) * 100})
    return {"rank": rank, "id": r["image_id"], "final": r["final"], "cosine": r["cosine"], "graph": r["graph"],
            "clip_rank": r["clip_rank"], "img": src, "ar": meta["width"] / meta["height"], "boxes": boxes,
            "nrel": sum(l["hi"] for l in graph["links"]), "g": graph}

_PAGE_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#1e1e1e;color:#dadada;font:13px system-ui,-apple-system,'Segoe UI',sans-serif}
.tabs{display:flex;gap:2px;padding:8px 12px 0;border-bottom:1px solid #333}
.tab{background:none;border:0;color:#8a8a8a;padding:8px 16px;font:inherit;font-weight:600;cursor:pointer;
     border-bottom:2px solid transparent}
.tab.on{color:#e6e6e6;border-bottom-color:#a882ff}
.pane{display:none;padding:12px}.pane.on{display:block}
.gv{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:12px;height:540px}
.viewer{display:flex;flex-direction:column;min-width:0}
.vhead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
.vhead b{color:#fff;font-size:14px}.sc{color:#a882ff}.mut{color:#8a8a8a}
.stage{flex:1;position:relative;background:#171717;border-radius:8px;display:flex;align-items:center;
       justify-content:center;min-height:0;overflow:hidden}
.imgbox{position:relative;line-height:0;max-width:100%;max-height:100%}
.imgbox img{display:block;max-width:100%;max-height:420px}
.ph{background:#2a2a2a;color:#777;display:flex;align-items:center;justify-content:center;font-size:12px;line-height:normal}
.box{position:absolute;border:2px solid #a882ff;transition:background .12s,border-color .12s;cursor:pointer}
.box .tag{position:absolute;left:-2px;top:-2px;background:#a882ff;color:#1e1e1e;font:700 11px/1 sans-serif;
          padding:2px 4px;border-radius:0 0 4px 0}
.box.on{border-color:#fff;background:rgba(168,130,255,.28);z-index:2}.box.on .tag{background:#fff}
.nav{position:absolute;top:50%;transform:translateY(-50%);background:rgba(30,30,30,.75);border:1px solid #444;
     color:#e6e6e6;width:34px;height:34px;border-radius:50%;cursor:pointer;font-size:18px;line-height:30px;z-index:3}
.nav:hover{border-color:#a882ff}.nav.prev{left:8px}.nav.next{right:8px}
.thumbs{display:flex;gap:6px;margin-top:8px}
.thumbs div{flex:1;height:52px;border-radius:5px;background:#2a2a2a center/cover no-repeat;cursor:pointer;
            border:2px solid transparent;opacity:.55;position:relative}
.thumbs div.on{border-color:#a882ff;opacity:1}
.thumbs span{position:absolute;left:3px;top:2px;font-size:11px;font-weight:700;color:#fff;text-shadow:0 0 3px #000}
.gwrap{position:relative;border-radius:8px;overflow:hidden;background:#1e1e1e;border:1px solid #2c2c2c;min-width:0}
#graph{position:absolute;inset:0}
.empty{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;
       color:#8a8a8a;padding:20px}
.gear{position:absolute;top:8px;right:8px;z-index:4;background:#2b2b2b;border:1px solid #444;color:#dadada;
      border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit}
.gear:hover{border-color:#a882ff}
.panel{position:absolute;top:42px;right:8px;width:260px;z-index:4;background:#262626;border:1px solid #3a3a3a;
       border-radius:8px;padding:10px 12px;display:none;font-size:12px}
.panel.on{display:block}
.panel h4{margin:8px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8a8a8a}
.panel h4:first-child{margin-top:0}
.panel label{display:flex;justify-content:space-between;align-items:center;margin:5px 0;gap:8px}
.panel input[type=range]{width:110px;accent-color:#a882ff}.panel input[type=checkbox]{accent-color:#a882ff}
.panel button{width:100%;margin-top:8px;background:#333;border:1px solid #444;color:#dadada;border-radius:5px;
              padding:5px;cursor:pointer;font:inherit}
.legend{position:absolute;left:10px;bottom:8px;z-index:3;font-size:11px;color:#8a8a8a;line-height:1.6;
        background:rgba(30,30,30,.8);padding:4px 8px;border-radius:6px;pointer-events:none}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px -1px 0}
.ln{display:inline-block;width:18px;margin:0 4px 3px 8px;border-top:2px solid #a882ff}
.ln.d{border-top:2px dotted #666}
table{width:100%;border-collapse:collapse}
th{text-align:left;color:#8a8a8a;font-weight:600;font-size:12px;padding:8px 10px;border-bottom:1px solid #333}
td{padding:8px 10px;border-bottom:1px solid #2a2a2a;vertical-align:middle}
tbody tr{cursor:pointer}tbody tr:hover{background:#262626}
td.num{font-variant-numeric:tabular-nums}
td .th{width:84px;height:56px;border-radius:4px;background:#2a2a2a center/cover no-repeat;margin-bottom:3px}
td .mut{font-size:11px}
.up{color:#7bd88f}.down{color:#ff7a7a}
.float-tooltip-kap{background:#2b2b2b!important;color:#e6e6e6!important;border:1px solid #444;font-size:12px!important}
"""

_PAGE_JS = r"""
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const S={labels:true,paths:true,arrows:true,names:true,size:1,width:1,repel:120,dist:80};
let cur=0, hover=null, nb={}, fg=null;
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const gel=$('#graph');

function tab(t){ $$('.tab').forEach(b=>b.classList.toggle('on',b.dataset.t===t));
  $$('.pane').forEach(p=>p.classList.toggle('on',p.id==='p-'+t));
  if(t==='graph'&&fg){ fg.width(gel.clientWidth).height(gel.clientHeight); } }
$$('.tab').forEach(b=>b.onclick=()=>tab(b.dataset.t));

function initGraph(){
  if(fg||!window.ForceGraph) return;
  fg=ForceGraph()(gel).backgroundColor('#1e1e1e').autoPauseRedraw(false).nodeId('id')
    .nodeVal(n=>n.val).nodeRelSize(4).nodeLabel(n=>n.tip).linkLabel(l=>l.tip)
    .linkVisibility(l=>l.hi||S.paths)
    .linkColor(l=>{const a=hover===null||l.source.id===hover||l.target.id===hover;
       return l.hi?(a?'rgba(168,130,255,0.85)':'rgba(168,130,255,0.15)'):(a?'#666':'#333');})
    .linkWidth(l=>(l.hi?1+40*Math.max(l.delta,0):0.6)*S.width).linkLineDash(l=>l.hi?null:[2,3])
    .linkDirectionalArrowLength(l=>S.arrows&&l.hi&&l.rel!=='chồng lấn'?5:0).linkDirectionalArrowRelPos(1)
    .linkDirectionalArrowColor(()=> '#a882ff')
    .linkCanvasObjectMode(()=> 'after')
    .linkCanvasObject((l,ctx,s)=>{ if(!S.labels||!l.hi||l.rel==='thuộc'||s<0.5) return;
       const x=(l.source.x+l.target.x)/2,y=(l.source.y+l.target.y)/2,f=10/s;
       ctx.font=f+'px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
       const w=ctx.measureText(l.rel).width; ctx.fillStyle='#1e1e1e'; ctx.fillRect(x-w/2-2,y-f/2-1,w+4,f+2);
       ctx.fillStyle='#a882ff'; ctx.fillText(l.rel,x,y); })
    .nodeCanvasObject((n,ctx,s)=>{
       const r=Math.sqrt(n.val)*4*S.size, act=hover===null||(nb[hover]&&nb[hover].has(n.id));
       ctx.beginPath(); ctx.arc(n.x,n.y,r,0,2*Math.PI);
       ctx.fillStyle=n.kind==='image'?(act?'#a882ff':'rgba(168,130,255,0.3)')
                    :(n.id===hover?'#ffffff':(act?'#b3b3b3':'rgba(179,179,179,0.25)'));
       ctx.fill();
       if(!S.names&&n.id!==hover) return;
       const f=Math.max(11/s,3); ctx.font=(n.kind==='image'?'bold ':'')+f+'px sans-serif';
       ctx.textAlign='center'; ctx.textBaseline='top'; ctx.fillStyle=act?'#dadada':'rgba(218,218,218,0.3)';
       ctx.fillText(n.name,n.x,n.y+r+2); })
    .nodePointerAreaPaint((n,c,ctx)=>{ctx.fillStyle=c;ctx.beginPath();
       ctx.arc(n.x,n.y,Math.sqrt(n.val)*4*S.size+3,0,2*Math.PI);ctx.fill();})
    .onNodeHover(n=>{ setHover(n?n.id:null); gel.style.cursor=n?'pointer':null; })
    .cooldownTicks(150).onEngineStop(()=>fg.zoomToFit(300,50));
  forces();
}
function forces(){ if(!fg) return; fg.d3Force('charge').strength(-S.repel);
  fg.d3Force('link').distance(l=>l.hi?S.dist:S.dist*1.3).strength(l=>l.hi?1:0.4);
  fg.nodeRelSize(4*S.size); fg.d3ReheatSimulation(); }
function setHover(id){ hover=(id===undefined?null:id);
  $$('.box').forEach(b=>b.classList.toggle('on',hover!==null&&+b.dataset.n===hover)); }

function show(i){
  cur=(i+D.items.length)%D.items.length; const it=D.items[cur]; hover=null;
  $('#vh').innerHTML='<b>#'+it.rank+' · '+it.id+'.jpg</b>';
  const boxes=it.boxes.map(b=>'<div class="box" data-n="'+b.n+'" title="'+esc(b.name)+'" style="left:'+b.l+'%;top:'+b.t+
    '%;width:'+b.w+'%;height:'+b.h+'%"><span class="tag">'+esc(b.tag)+'</span></div>').join('');
  const pic=it.img?'<img src="'+it.img+'">'
    :'<div class="ph" style="width:'+Math.min(420*it.ar,440)+'px;height:'+Math.min(420,440/it.ar)+'px">chưa có ảnh gốc</div>';
  $('#imgbox').innerHTML=pic+boxes;
  $$('.box').forEach(b=>{ b.onmouseenter=()=>setHover(+b.dataset.n); b.onmouseleave=()=>setHover(null); });
  $$('.thumbs div').forEach((t,k)=>t.classList.toggle('on',k===cur));
  const has=it.nrel>0; $('.empty').style.display=has?'none':'flex';
  initGraph();
  if(!fg) return;
  const g=JSON.parse(JSON.stringify(it.g));
  nb={}; g.nodes.forEach(n=>{nb[n.id]=new Set([n.id]); if(n.kind==='image'){n.fx=0;n.fy=0;}});
  g.links.forEach(l=>{nb[l.source].add(l.target); nb[l.target].add(l.source);});
  fg.width(gel.clientWidth).height(gel.clientHeight).graphData(has?g:{nodes:[],links:[]});
}

$('#thumbs').innerHTML=D.items.map((it,k)=>'<div style="background-image:url('+(it.img||'')+')" onclick="show('+k+
  ')"><span>#'+it.rank+'</span></div>').join('');
$('.prev').onclick=()=>show(cur-1); $('.next').onclick=()=>show(cur+1);
document.addEventListener('keydown',e=>{ if(!$('#p-graph').classList.contains('on')) return;
  if(e.key==='ArrowLeft') show(cur-1); if(e.key==='ArrowRight') show(cur+1); });
$('.gear').onclick=()=>$('.panel').classList.toggle('on');
$$('.panel input').forEach(inp=>inp.oninput=()=>{
  const k=inp.dataset.k; S[k]=inp.type==='checkbox'?inp.checked:+inp.value;
  if(['repel','dist','size'].includes(k)) forces(); });
$('#fit').onclick=()=>fg&&fg.zoomToFit(300,50);
$('#rows').innerHTML=D.items.map((it,k)=>{ const d=it.clip_rank-it.rank;
  return '<tr onclick="tab(\'graph\');show('+k+')"><td class="num"><b>#'+it.rank+'</b></td><td><div class="th" '+
  'style="background-image:url('+(it.img||'')+')"></div><span class="mut">'+it.id+'</span></td><td class="num sc">'+
  it.final.toFixed(3)+'</td><td class="num">'+it.cosine.toFixed(3)+'</td><td class="num">'+it.graph.toFixed(3)+
  '</td><td class="num">#'+it.clip_rank+(d>0?' <span class="up">▲'+d+'</span>':d<0?' <span class="down">▼'+(-d)+
  '</span>':'')+'</td></tr>'; }).join('');
window.addEventListener('resize',()=>fg&&fg.width(gel.clientWidth).height(gel.clientHeight));
show(0);
"""

_PAGE_BODY = """
<div class="tabs"><button class="tab on" data-t="graph">Graph</button><button class="tab" data-t="rank">Ranking</button>
</div>
<div class="pane on" id="p-graph"><div class="gv">
  <div class="viewer"><div class="vhead" id="vh"></div>
    <div class="stage"><button class="nav prev">‹</button><div class="imgbox" id="imgbox"></div>
      <button class="nav next">›</button></div>
    <div class="thumbs" id="thumbs"></div></div>
  <div class="gwrap"><div id="graph"></div>
    <div class="empty">Không có quan hệ nào làm tăng điểm Graph của ảnh này</div>
    <button class="gear">⚙ Tùy chỉnh</button>
    <div class="panel">
      <h4>Hiển thị</h4>
      <label>Nhãn quan hệ trên cạnh<input type="checkbox" data-k="labels" checked></label>
      <label>Tên nút<input type="checkbox" data-k="names" checked></label>
      <label>Mũi tên hướng quan hệ<input type="checkbox" data-k="arrows" checked></label>
      <label>Đường đi về ảnh (chấm)<input type="checkbox" data-k="paths" checked></label>
      <label>Kích thước nút<input type="range" data-k="size" min="0.5" max="2" step="0.1" value="1"></label>
      <label>Độ dày cạnh<input type="range" data-k="width" min="0.3" max="3" step="0.1" value="1"></label>
      <h4>Lực</h4>
      <label>Lực đẩy<input type="range" data-k="repel" min="20" max="600" step="10" value="120"></label>
      <label>Độ dài cạnh<input type="range" data-k="dist" min="20" max="240" step="5" value="80"></label>
      <button id="fit">Căn vừa khung</button>
    </div>
    <div class="legend"><span class="dot" style="background:#a882ff"></span>ảnh
      <span class="dot" style="background:#b3b3b3;margin-left:8px"></span>vùng R (to ∝ đóng góp Δ)<br>
      <span class="ln" style="margin-left:0"></span>quan hệ trong giải thích<span class="ln d"></span>đường đi về ảnh</div>
  </div></div></div>
<div class="pane" id="p-rank"><table><thead><tr><th>Hạng</th><th>Ảnh</th><th>Final = __FORMULA__</th><th>CLIP</th>
  <th>Graph</th><th>Hạng nếu chỉ dùng CLIP</th></tr></thead><tbody id="rows"></tbody></table>
</div>
"""

PAGE_HEIGHT = 640

def results_page(query, results):
    a = ALPHA[MAIN_RUN]
    data = {"items": [_slide(r, k) for k, r in enumerate(results, 1)]}
    body = _PAGE_BODY.replace("__FORMULA__", f"{1 - a:.1f}·CLIP + {a:.1f}·Graph")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_PAGE_CSS}</style>{_force_graph_js()}</head>"
            f"<body>{body}"
            f"<script>const D={json.dumps(data, ensure_ascii=False)};{_PAGE_JS}</script></body></html>")

def results_iframe(query, results):
    page = results_page(query, results)
    return (f'<iframe srcdoc="{_html.escape(page, quote=True)}" style="width:100%;height:{PAGE_HEIGHT}px;border:0;'
            f'border-radius:8px;background:#1e1e1e"></iframe>')

def _log(query, results):
    with open(OUT / "demo_explanations.jsonl", "a", encoding="utf-8") as f:
        for rank, r in enumerate(results, 1):
            f.write(json.dumps({"query": query, "rank": rank, "candidate_id": r["image_id"],
                                "cosine": r["cosine"], "graph": r["graph"], "final": r["final"],
                                "clip_rank": r["clip_rank"], "alpha_graph": ALPHA[MAIN_RUN], "K": K_STAR,
                                "path": [[x["relation"], x["delta_graph"]] for x in r["explanation"]["relations"]
                                         if x["in_explanation"]]}, ensure_ascii=False) + "\n")

def run_query(query_text, q_raw=None, query_sentence=None, query_image_index=None):
    q_raw = encode_free_text(query_text) if q_raw is None else q_raw
    results = search_and_explain(q_raw, query_sentence=query_sentence, query_image_index=query_image_index)
    _log(query_text, results)
    return results_iframe(query_text, results)

def gradio_demo_inference(query_text):
    if not query_text or not query_text.strip():
        return '<p style="color:#888">Vui lòng nhập câu truy vấn tiếng Anh hợp lệ.</p>'
    return run_query(query_text.strip())

# Xem trước trong notebook (không cần tải CLIP): một truy vấn có sẵn trong test, dùng embedding CLIP đã trích.
_qi = int(CASES.query_index.iloc[0])
_qs = query_sentence_index(test["caption_ids"][_qi])
_res = search_and_explain(T_TEXT[_qi], query_sentence=_qs, query_image_index=int(T_TARGET[_qi]))
(OUT / "figures" / "demo_preview.html").write_text(results_page(test["captions"][_qi], _res), encoding="utf-8")
display(HTML(results_iframe(test["captions"][_qi], _res)))
