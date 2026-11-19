import html as _html, urllib.request
from types import SimpleNamespace

_FG_URLS = ["https://cdn.jsdelivr.net/npm/force-graph@1.52.0/dist/force-graph.min.js",
            "https://unpkg.com/force-graph@1.52.0/dist/force-graph.min.js"]
_FG = {}
def force_graph_js():
    if "js" not in _FG:
        local = os.environ.get("XAI_FORCEGRAPH_JS"); _FG["js"] = None
        if local and Path(local).exists():
            _FG["js"] = Path(local).read_text(encoding="utf-8")
        else:
            for u in _FG_URLS:
                try:
                    _FG["js"] = urllib.request.urlopen(u, timeout=30).read().decode("utf-8"); break
                except Exception:
                    continue
    return f"<script>{_FG['js']}</script>" if _FG["js"] else f'<script src="{_FG_URLS[0]}"></script>'

for D in DATA.values():
    lab = pd.Series(D.ent_label); dup = lab.duplicated(keep=False)
    D.ent_choice = [f"{l} [{i}]" if dup[i] else l for i, l in enumerate(D.ent_label)]
    D.choice_to_ent = {c: i for i, c in enumerate(D.ent_choice)}
    rl = pd.Series(D.rel_label); dupr = rl.duplicated(keep=False)
    D.rel_choice = [f"{l} [{i}]" if dupr[i] else l for i, l in enumerate(D.rel_label)]
    D.choice_to_rel = {c: i for i, c in enumerate(D.rel_choice)}

def query_text_for(D, anchor, rel, direction):
    allq = pd.concat([D.test_q, D.val_q])
    hit = allq[(allq.prediction_direction == direction) & (allq.anchor_entity_index == anchor) & (allq.relation_index == rel)]
    if len(hit):
        return hit.query_text.iloc[0]
    same = allq[(allq.prediction_direction == direction) & (allq.relation_index == rel)]
    for _, r in same.head(50).iterrows():
        old = D.ent_label[int(r.anchor_entity_index)]
        if old and old in r.query_text:
            return r.query_text.replace(old, D.ent_label[anchor])
    rl = D.rel_label[rel]; al = D.ent_label[anchor]
    return (f"Find an image of the entity that is related to {al} by {rl}." if direction == "tail"
            else f"Find an image of the entity whose {rl} relation points to {al}.")

def thumb(D, row, size=640):
    img = open_image(D, row)
    if img is None:
        return None
    img.thumbnail((size, size)); b = io.BytesIO(); img.save(b, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

def graph_payload(D, q, ex, K):
    nodes, links = {}, []
    e = ex["entity"]; nodes[e] = dict(id=e, name=D.ent_label[e], kind="cand", val=7, tip=f"{D.ent_label[e]} (thực thể của ảnh)")
    if q.prediction_direction != "direct":
        a = int(q.anchor_entity_index)
        nodes[a] = dict(id=a, name=D.ent_label[a] + (" (= thực thể của ảnh)" if a == e else ""), kind="anchor", val=9,
                        tip=f"{D.ent_label[a]} (anchor của truy vấn" + (", cũng là thực thể của ảnh)" if a == e else ")"))
        src, dst = (a, e) if q.prediction_direction == "tail" else (e, a)
        links.append(dict(source=src, target=dst, rel=D.rel_label[int(q.relation_index)] + " ?", kind="query", delta=0,
                          tip=f"Quan hệ được hỏi: {D.ent_label[src]} –[{D.rel_label[int(q.relation_index)]}]→ {D.ent_label[dst]} · điểm đồ thị {ex['g']:+.3f}"))
    facts = [f for f in ex.get("E3", {}).get("facts", []) if f["delta"] > 0][:K]
    vmax = max([f["delta"] for f in facts] + [1e-9]); seen = set()
    for f in facts:
        for j, (u, v, fid) in enumerate(f["path"]):
            h, t, r = (int(x) for x in D.train[fid])
            for n in (h, t):
                nodes.setdefault(n, dict(id=n, name=D.ent_label[n], kind="other", val=1.5, tip=D.ent_label[n]))
            if (h, t, r) in seen:
                continue
            seen.add((h, t, r)); main = (fid == f["fid"])
            links.append(dict(source=h, target=t, rel=D.rel_label[r], kind="fact" if main else "path", delta=f["delta"] if main else 0,
                              tip=f"{D.ent_label[h]} –[{D.rel_label[r]}]→ {D.ent_label[t]}" + (f" · Δ khi bỏ {f['delta']:+.4f}" if main else " (nối về anchor)")))
        x = f["head"] if f["tail"] == e else f["tail"]
        if nodes[x]["kind"] == "other":
            nodes[x]["val"] = 1.5 + 6 * f["delta"] / vmax; nodes[x]["tip"] = f"{D.ent_label[x]} · Δ {f['delta']:+.4f}"
    return dict(nodes=list(nodes.values()), links=links)

_PAGE_CSS = """
*{box-sizing:border-box}body{margin:0;background:#1e1e1e;color:#dadada;font:13px system-ui,-apple-system,'Segoe UI',sans-serif}
.tabs{display:flex;gap:2px;padding:8px 12px 0;border-bottom:1px solid #333}
.tab{background:none;border:0;color:#8a8a8a;padding:8px 16px;font:inherit;font-weight:600;cursor:pointer;border-bottom:2px solid transparent}
.tab.on{color:#e6e6e6;border-bottom-color:#a882ff}.pane{display:none;padding:12px}.pane.on{display:block}
.gv{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:12px;height:560px}
.viewer{display:flex;flex-direction:column;min-width:0}.vhead{margin-bottom:6px}.vhead b{color:#fff;font-size:14px}
.mut{color:#8a8a8a}.sc{color:#a882ff}
.stage{flex:1;position:relative;background:#171717;border-radius:8px;display:flex;align-items:center;justify-content:center;min-height:0;overflow:hidden}
.stage img{max-width:100%;max-height:360px;display:block}.ph{color:#777;font-size:12px}
.nav{position:absolute;top:50%;transform:translateY(-50%);background:rgba(30,30,30,.75);border:1px solid #444;color:#e6e6e6;width:34px;height:34px;border-radius:50%;cursor:pointer;font-size:18px;line-height:30px;z-index:3}
.nav:hover{border-color:#a882ff}.nav.prev{left:8px}.nav.next{right:8px}
.mods{margin-top:8px;font-size:12px}.bar{display:flex;align-items:center;gap:6px;margin:3px 0}.bar span:first-child{width:92px;color:#8a8a8a}
.bar i{display:block;height:8px;background:#a882ff;border-radius:4px}
.thumbs{display:flex;gap:6px;margin-top:8px}.thumbs div{flex:1;height:50px;border-radius:5px;background:#2a2a2a center/cover no-repeat;cursor:pointer;border:2px solid transparent;opacity:.55;position:relative}
.thumbs div.on{border-color:#a882ff;opacity:1}.thumbs span{position:absolute;left:3px;top:2px;font-size:11px;font-weight:700;color:#fff;text-shadow:0 0 3px #000}
.gwrap{position:relative;border-radius:8px;overflow:hidden;background:#1e1e1e;border:1px solid #2c2c2c;min-width:0}#graph{position:absolute;inset:0}
.empty{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;color:#8a8a8a;padding:20px}
.gear{position:absolute;top:8px;right:8px;z-index:4;background:#2b2b2b;border:1px solid #444;color:#dadada;border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit}
.panel{position:absolute;top:42px;right:8px;width:260px;z-index:4;background:#262626;border:1px solid #3a3a3a;border-radius:8px;padding:10px 12px;display:none;font-size:12px}
.panel.on{display:block}.panel h4{margin:8px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8a8a8a}.panel h4:first-child{margin-top:0}
.panel label{display:flex;justify-content:space-between;align-items:center;margin:5px 0;gap:8px}.panel input[type=range]{width:110px;accent-color:#a882ff}
.panel input[type=checkbox]{accent-color:#a882ff}.panel button{width:100%;margin-top:8px;background:#333;border:1px solid #444;color:#dadada;border-radius:5px;padding:5px;cursor:pointer;font:inherit}
.legend{position:absolute;left:10px;bottom:8px;z-index:3;font-size:11px;color:#8a8a8a;line-height:1.6;background:rgba(30,30,30,.8);padding:4px 8px;border-radius:6px;pointer-events:none}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px -1px 0}
table{width:100%;border-collapse:collapse}th{text-align:left;color:#8a8a8a;font-weight:600;font-size:12px;padding:8px 10px;border-bottom:1px solid #333}
td{padding:8px 10px;border-bottom:1px solid #2a2a2a}tbody tr{cursor:pointer}tbody tr:hover{background:#262626}
td .th{width:84px;height:56px;border-radius:4px;background:#2a2a2a center/cover no-repeat}.up{color:#7bd88f}.down{color:#ff7a7a}
.float-tooltip-kap{background:#2b2b2b!important;color:#e6e6e6!important;border:1px solid #444;font-size:12px!important}
"""

_PAGE_JS = r"""
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const S={labels:true,names:true,arrows:true,paths:true,size:1,width:1,repel:120,dist:80};
let cur=0,hover=null,nb={},fg=null;const gel=$('#graph');
const COL={anchor:'#a882ff',cand:'#f0f0f0',other:'#b3b3b3'};
function tab(t){$$('.tab').forEach(b=>b.classList.toggle('on',b.dataset.t===t));$$('.pane').forEach(p=>p.classList.toggle('on',p.id==='p-'+t));
  if(t==='graph'&&fg)fg.width(gel.clientWidth).height(gel.clientHeight);}
$$('.tab').forEach(b=>b.onclick=()=>tab(b.dataset.t));
function initGraph(){ if(fg||!window.ForceGraph)return;
  fg=ForceGraph()(gel).backgroundColor('#1e1e1e').autoPauseRedraw(false).nodeId('id').nodeVal(n=>n.val).nodeRelSize(4)
   .nodeLabel(n=>n.tip).linkLabel(l=>l.tip).linkVisibility(l=>l.kind!=='path'||S.paths)
   .linkColor(l=>{const a=hover===null||l.source.id===hover||l.target.id===hover;
      if(l.kind==='query')return a?'#888':'#3a3a3a'; if(l.kind==='path')return a?'rgba(168,130,255,0.45)':'rgba(168,130,255,0.1)';
      return a?'rgba(168,130,255,0.9)':'rgba(168,130,255,0.15)';})
   .linkWidth(l=>(l.kind==='fact'?1.2+3*Math.min(1,Math.abs(l.delta)/(D.items[cur].dmax||1)):0.8)*S.width)
   .linkLineDash(l=>l.kind==='query'?[4,3]:(l.kind==='path'?[2,3]:null))
   .linkDirectionalArrowLength(()=>S.arrows?5:0).linkDirectionalArrowRelPos(1).linkDirectionalArrowColor(l=>l.kind==='query'?'#888':'#a882ff')
   .linkCanvasObjectMode(()=>'after').linkCanvasObject((l,ctx,s)=>{ if(!S.labels||s<0.5)return;
      const x=(l.source.x+l.target.x)/2,y=(l.source.y+l.target.y)/2,f=10/s;ctx.font=f+'px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
      const w=ctx.measureText(l.rel).width;ctx.fillStyle='#1e1e1e';ctx.fillRect(x-w/2-2,y-f/2-1,w+4,f+2);
      ctx.fillStyle=l.kind==='query'?'#aaa':'#a882ff';ctx.fillText(l.rel,x,y);})
   .nodeCanvasObject((n,ctx,s)=>{const r=Math.sqrt(n.val)*4*S.size,act=hover===null||(nb[hover]&&nb[hover].has(n.id));
      ctx.beginPath();ctx.arc(n.x,n.y,r,0,2*Math.PI);ctx.globalAlpha=act?1:0.25;ctx.fillStyle=n.id===hover?'#fff':COL[n.kind];ctx.fill();ctx.globalAlpha=1;
      if(!S.names&&n.id!==hover)return;const f=Math.max(11/s,3);ctx.font=(n.kind!=='other'?'bold ':'')+f+'px sans-serif';
      ctx.textAlign='center';ctx.textBaseline='top';ctx.fillStyle=act?'#dadada':'rgba(218,218,218,0.3)';ctx.fillText(n.name,n.x,n.y+r+2);})
   .nodePointerAreaPaint((n,c,ctx)=>{ctx.fillStyle=c;ctx.beginPath();ctx.arc(n.x,n.y,Math.sqrt(n.val)*4*S.size+3,0,2*Math.PI);ctx.fill();})
   .onNodeHover(n=>{hover=n?n.id:null;gel.style.cursor=n?'pointer':null;}).cooldownTicks(150).onEngineStop(()=>fg.zoomToFit(300,50));
  forces();}
function forces(){if(!fg)return;fg.d3Force('charge').strength(-S.repel);fg.d3Force('link').distance(l=>l.kind==='fact'?S.dist:S.dist*1.3);fg.nodeRelSize(4*S.size);fg.d3ReheatSimulation();}
function show(i){cur=(i+D.items.length)%D.items.length;const it=D.items[cur];hover=null;
  $('#vh').innerHTML='<b>#'+it.rank+' · '+esc(it.entity)+'</b>';
  $('#stage-img').innerHTML=it.img?'<img src="'+it.img+'">':'<span class="ph">ảnh không có trên Drive</span>';
  $('#mods').innerHTML=it.mods;$$('.thumbs div').forEach((t,k)=>t.classList.toggle('on',k===cur));
  const has=it.g.links.length>0;$('.empty').style.display=has?'none':'flex';$('.empty').innerHTML=it.note;initGraph();if(!fg)return;
  const g=JSON.parse(JSON.stringify(it.g));nb={};g.nodes.forEach(n=>{nb[n.id]=new Set([n.id]);if(n.kind==='anchor'){n.fx=0;n.fy=0;}});
  g.links.forEach(l=>{nb[l.source].add(l.target);nb[l.target].add(l.source);});fg.width(gel.clientWidth).height(gel.clientHeight).graphData(has?g:{nodes:[],links:[]});}
$('#thumbs').innerHTML=D.items.map((it,k)=>'<div style="background-image:url('+(it.img||'')+')" onclick="show('+k+')"><span>#'+it.rank+'</span></div>').join('');
$('.prev').onclick=()=>show(cur-1);$('.next').onclick=()=>show(cur+1);
document.addEventListener('keydown',e=>{if(!$('#p-graph').classList.contains('on'))return;if(e.key==='ArrowLeft')show(cur-1);if(e.key==='ArrowRight')show(cur+1);});
$('.gear').onclick=()=>$('.panel').classList.toggle('on');
$$('.panel input').forEach(inp=>inp.oninput=()=>{const k=inp.dataset.k;S[k]=inp.type==='checkbox'?inp.checked:+inp.value;if(['repel','dist','size'].includes(k))forces();});
$('#fit').onclick=()=>fg&&fg.zoomToFit(300,50);
$('#rows').innerHTML=D.items.map((it,k)=>{const d=it.clip_rank-it.rank;return '<tr onclick="tab(\'graph\');show('+k+')"><td><b>#'+it.rank+'</b></td><td><div class="th" style="background-image:url('+(it.img||'')+')"></div></td><td>'+esc(it.entity)+'</td><td class="sc">'+it.fused.toFixed(3)+'</td><td>'+it.clip.toFixed(3)+'</td><td>'+(it.graph===null?'—':it.graph.toFixed(3))+'</td><td>#'+it.clip_rank+(d>0?' <span class="up">▲'+d+'</span>':d<0?' <span class="down">▼'+(-d)+'</span>':'')+'</td></tr>';}).join('');
window.addEventListener('resize',()=>fg&&fg.width(gel.clientWidth).height(gel.clientHeight));show(0);
"""

_PAGE_BODY = """
<div class="tabs"><button class="tab on" data-t="graph">Graph</button><button class="tab" data-t="rank">Ranking</button></div>
<div class="pane on" id="p-graph"><div class="gv">
 <div class="viewer"><div class="vhead" id="vh"></div><div class="stage"><button class="nav prev">‹</button><div id="stage-img"></div><button class="nav next">›</button></div>
  <div class="mods" id="mods"></div><div class="thumbs" id="thumbs"></div></div>
 <div class="gwrap"><div id="graph"></div><div class="empty"></div><button class="gear">⚙ Tùy chỉnh</button>
  <div class="panel"><h4>Hiển thị</h4>
   <label>Nhãn quan hệ trên cạnh<input type="checkbox" data-k="labels" checked></label><label>Tên nút<input type="checkbox" data-k="names" checked></label>
   <label>Mũi tên hướng quan hệ<input type="checkbox" data-k="arrows" checked></label><label>Đường đi về anchor<input type="checkbox" data-k="paths" checked></label>
   <label>Kích thước nút<input type="range" data-k="size" min="0.5" max="2" step="0.1" value="1"></label><label>Độ dày cạnh<input type="range" data-k="width" min="0.3" max="3" step="0.1" value="1"></label>
   <h4>Lực</h4><label>Lực đẩy<input type="range" data-k="repel" min="20" max="600" step="10" value="120"></label>
   <label>Độ dài cạnh<input type="range" data-k="dist" min="20" max="240" step="5" value="80"></label><button id="fit">Căn vừa khung</button></div>
  <div class="legend"><span class="dot" style="background:#a882ff"></span>anchor <span class="dot" style="background:#f0f0f0;margin-left:8px"></span>thực thể của ảnh
   <span class="dot" style="background:#b3b3b3;margin-left:8px"></span>lân cận (to ∝ Δ)<br>nét đứt = quan hệ được hỏi · nét liền = bộ ba train bằng chứng</div></div></div></div>
<div class="pane" id="p-rank"><table><thead><tr><th>Hạng</th><th>Ảnh</th><th>Thực thể</th><th>Final = __F__</th><th>CLIP</th><th>Graph</th><th>Hạng nếu chỉ dùng CLIP</th></tr></thead><tbody id="rows"></tbody></table></div>
"""
PAGE_HEIGHT = 680

def mods_html(ex):
    parts = []
    for m in range(3):
        if not ex["present"][m]:
            continue
        d = ex["drop"].get(m)
        parts.append(f'<div class="bar"><span>{MOD[m]}</span><i style="width:{max(2, 120 * ex["weights"][m]):.0f}px"></i>'
                     f'<span class="mut">w {ex["weights"][m]:.2f}' + (f' · che: Δ {d["delta"]:+.3f}, hạng → {d["rank"]}' if d else "") + "</span></div>")
    return "".join(parts)

def search_and_explain(name, q, text_feat, k=5):
    D, sc = DATA[name], SCORERS[(name, MAIN_SEED)]
    pool = D.pool["test"]; feats = F.normalize(torch.tensor(np.asarray(D.img_feat[pool]), dtype=torch.float32), dim=1).to(DEVICE)
    s = (text_feat.to(DEVICE) @ feats.T)[0]; order = torch.argsort(s, descending=True, stable=True)[:min(TOPK, len(pool))]
    rows = torch.as_tensor(pool[order.cpu().numpy()]); clip_top = s[order]
    qp = sc.qproj(text_feat)[0] if sc.kind == "fb" else None
    g, ent = sc.image_scores(q, rows, D, qproj=qp)
    ctx = dict(q=q, rows=rows, clip=clip_top, graph=g, ent=ent, qproj=qp, base=1, tmask=torch.zeros(len(rows), dtype=torch.bool, device=DEVICE),
               alpha=sc.alpha, kind=sc.kind)
    fused = clip_top if (g is None or (sc.kind == "db" and q.prediction_direction == "direct")) else sc.alpha * zscore(clip_top) + (1 - sc.alpha) * zscore(g)
    top = torch.argsort(fused, descending=True, stable=True)[:k]
    items = []
    for rank, p in enumerate(top.tolist(), 1):
        row = int(rows[p]); e = int(ent[p]) if ent is not None else int(D.img_ent[row, 0])
        item = dict(rank=rank, entity=D.ent_label[e], img=thumb(D, row), fused=float(fused[p]), clip=float(clip_top[p]),
                    graph=None if g is None else float(g[p]), clip_rank=p + 1, mods="", note="", g=dict(nodes=[], links=[]))
        if g is None or (sc.kind == "db" and q.prediction_direction == "direct"):
            item["note"] = ("DB15K – truy vấn trực tiếp: mô hình được huấn luyện để xếp hạng loại truy vấn này CHỈ bằng CLIP "
                            "(direct_policy = CLIP_only). AdaMF chấm điểm bộ ba (anchor, quan hệ, ?), mà truy vấn trực tiếp không có anchor "
                            "hay quan hệ nào, nên không cạnh/đường đi nào tham gia xếp hạng. Vẽ đồ thị ở đây sẽ là giải thích hậu kiểm — đề cấm. "
                            f"Ảnh này thuộc thực thể: {D.ent_label[e]}.")
        else:
            ex = explain_image(ctx, D, sc, p); item["mods"] = mods_html(ex)
            item["g"] = graph_payload(D, q, ex, K_STAR[name]); item["dmax"] = max([abs(l["delta"]) for l in item["g"]["links"]] + [1e-9])
            if q.prediction_direction == "direct":
                item["note"] = ("FB15K – truy vấn trực tiếp: điểm đồ thị = cos(chiếu câu truy vấn, embedding gộp của thực thể), "
                                "không đi qua quan hệ nào. Ablation cấp 2 cho thấy dùng đồ thị xáo hoặc ngẫu nhiên vẫn giữ nguyên mức tăng "
                                "của nhóm này, nên không vẽ bộ ba nào làm giải thích. Giải thích trung thực là phần đóng góp theo phương thức "
                                f"(cấu trúc / ảnh / văn bản) ở cột bên trái. Ảnh này thuộc thực thể: {D.ent_label[e]}.")
            elif not item["g"]["links"][1:]:
                item["note"] = "Không có bộ ba train nào làm tăng điểm của thực thể này"
        items.append(item)
    return items

def results_page(name, items):
    a = SCORERS[(name, MAIN_SEED)].alpha
    body = _PAGE_BODY.replace("__F__", f"{a:.1f}·z(CLIP) + {1 - a:.1f}·z(Graph)")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_PAGE_CSS}</style>{force_graph_js()}</head><body>{body}"
            f"<script>const D={json.dumps(dict(items=items), ensure_ascii=False)};{_PAGE_JS}</script></body></html>")

def results_iframe(name, items):
    return (f'<iframe srcdoc="{_html.escape(results_page(name, items), quote=True)}" style="width:100%;height:{PAGE_HEIGHT}px;'
            f'border:0;border-radius:8px;background:#1e1e1e"></iframe>')

def run_demo(name, mode, anchor_choice, rel_choice, direction_choice, free_text):
    D = DATA[name]
    if mode.startswith("Gián tiếp"):
        if anchor_choice not in D.choice_to_ent or rel_choice not in D.choice_to_rel:
            return '<p style="color:#888">Chọn thực thể anchor và quan hệ.</p>'
        a, r = D.choice_to_ent[anchor_choice], D.choice_to_rel[rel_choice]
        d = "tail" if direction_choice.startswith("tail") else "head"
        text = query_text_for(D, a, r, d)
        q = SimpleNamespace(prediction_direction=d, anchor_entity_index=a, relation_index=r, query_text=text, query_id=f"demo|{a}|{r}|{d}")
    else:
        if not free_text or not free_text.strip():
            return '<p style="color:#888">Nhập câu truy vấn tiếng Anh.</p>'
        text = free_text.strip()
        q = SimpleNamespace(prediction_direction="direct", anchor_entity_index=None, relation_index=None, query_text=text, query_id=f"demo|{text}")
    items = search_and_explain(name, q, encode_texts(name, [text]))
    with open(D.out / "demo_explanations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(dataset=name, query=text, top=[dict(rank=i["rank"], entity=i["entity"], fused=i["fused"]) for i in items]), ensure_ascii=False) + "\n")
    return f'<p class="mut" style="color:#8a8a8a;margin:4px 0">Câu truy vấn gửi cho CLIP: “{_html.escape(text)}”</p>' + results_iframe(name, items)

# Xem trước trong notebook: ca thành công đầu tiên của tập đầu tiên
_n0 = DATASET_NAMES[0]; _c0 = CASES[_n0].iloc[0]
_q0 = TEST[_n0].q.iloc[int(_c0.qi)]
_items = search_and_explain(_n0, _q0, TEST[_n0].text[int(_c0.qi):int(_c0.qi) + 1])
(DATA[_n0].out / "figures" / "demo_preview.html").write_text(results_page(_n0, _items), encoding="utf-8")
display(HTML(results_iframe(_n0, _items)))
