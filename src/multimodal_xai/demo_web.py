# Giao diện web demo dùng chung cho ba cấp độ (Visual Genome, DB15K / FB15K-237-IMG, ViHeritage-MMKG).
#
# Mỗi notebook demo chỉ tạo dữ liệu (header + danh sách kết quả); trang HTML, đồ thị lực kiểu Obsidian (force-graph),
# màu sắc và cách đọc giống nhau ở cả ba cấp. Màu theo báo cáo (report/figures/scripts/make_figures.py):
#   xanh ngọc = thực thể neo, cam = ảnh / thực thể của ảnh trả về, xanh dương = phần tử của tập giải thích S,
#   xanh nhạt = đường nối về neo (chỉ để đọc, không đo), xám nét đứt = quan hệ được hỏi, xám = nút khác.
# Màn hình chính chỉ hiện phần cần cho demo; số liệu phụ nằm trong mục "Chi tiết" (mặc định đóng).
# File này được nhúng nguyên văn vào notebook demo cấp 1 và cấp 3; notebook cấp 2 import từ package.
#
# Dữ liệu một kết quả (item):
#   rank, title, correct (True / False / None), img (data URL), graph = {nodes, links} (dùng node(), link()),
#   empty (chữ hiện khi không có đồ thị), why = [(nhãn, nội dung)] (kiểm chứng chính, tối đa 3 dòng),
#   more = [(nhãn, nội dung)] (chi tiết), bars = [{label, w, text}] (chi tiết), note (chi tiết),
#   row = {clip_rank, removed} (bảng xếp hạng),
#   story = một câu trên đồ thị trả lời "Vì sao trả về X?" (bằng chứng mạnh nhất, đường đi về neo),
#   removal = câu hiện khi bấm "Bỏ giải thích S" (hạng trước, sau; so với bỏ ngẫu nhiên).
import base64
import html as _html
import io
import json
import os
import urllib.request
from pathlib import Path

PALETTE = {"anchor": "#1baf7a", "cand": "#eb6834", "other": "#d9d8d4",
           "fact": "#2a78d6", "path": "#9cc0ef", "query": "#8f8e8a"}
PAGE_HEIGHT = 720

_FG_URLS = ["https://cdn.jsdelivr.net/npm/force-graph@1.52.0/dist/force-graph.min.js",
            "https://unpkg.com/force-graph@1.52.0/dist/force-graph.min.js"]
_FG = {}


def force_graph_js():
    """Nhúng thẳng force-graph vào trang (iframe srcdoc không tải được script ngoài ở một số trình duyệt)."""
    if "js" not in _FG:
        _FG["js"] = None
        local = os.environ.get("XAI_FORCEGRAPH_JS")
        if local and Path(local).exists():
            _FG["js"] = Path(local).read_text(encoding="utf-8")
        else:
            for u in _FG_URLS:
                try:
                    _FG["js"] = urllib.request.urlopen(u, timeout=30).read().decode("utf-8")
                    break
                except Exception:
                    continue
    return f"<script>{_FG['js']}</script>" if _FG["js"] else f'<script src="{_FG_URLS[0]}"></script>'


def image_data_url(img, size=640, quality=85):
    """PIL.Image hoặc đường dẫn → data URL JPEG; None nếu không mở được."""
    try:
        from PIL import Image
        if img is None:
            return None
        if not hasattr(img, "convert"):
            img = Image.open(img)
        img = img.convert("RGB")
        img.thumbnail((size, size))
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    except Exception:
        return None


def rank_text(r):
    if r is None:
        return "không có"
    r = float(r)
    return f"{int(r)}" if r == int(r) else f"{r:.1f}"


def _move(r0, x):
    if x is None:
        return "không đo"
    if abs(float(x) - r0) < 0.05:
        return f"vẫn hạng {r0}"
    return f"{'xuống' if x > r0 else 'lên'} hạng {rank_text(x)}"


def removal_sentence(n, unit, r0, after, rand=None):
    """Câu hiện khi bấm "Bỏ giải thích S": hạng sau khi bỏ S, so với bỏ cùng số phần tử ngẫu nhiên (trung bình)."""
    s = f"Bỏ {n} {unit} giải thích: ảnh {_move(r0, after)}"
    if rand is not None:
        s += f" · bỏ {n} {unit} ngẫu nhiên: {_move(r0, rand)}" + (" (trung bình)" if float(rand) != int(rand) else "")
    return s


def node(nid, name, kind, val=2.0, tip=None, result=False):
    """result=True: nút này cũng là thực thể của ảnh trả về (ví dụ mô hình trả về chính thực thể neo), vẽ bằng bức ảnh."""
    return dict(id=nid, name=str(name), kind=kind, val=float(val), tip=str(tip or name), result=bool(result))


def link(source, target, rel, kind, delta=0.0, label=None, tip=None):
    return dict(source=source, target=target, rel=str(rel), kind=kind, delta=float(delta),
                label=str(rel if label is None else label), tip=str(tip or rel))


DEFAULT_LEGEND = {"anchor": "neo", "cand": "ảnh trả về", "other": "nút khác",
                  "fact": "giải thích S (dày theo Δ)", "path": "nối về neo", "query": "quan hệ được hỏi"}

_CSS = r"""
:root{--bg:#1e1e1e;--panel:#252525;--panel2:#2b2b2b;--line:#353535;--text:#e3e3e0;--muted:#9a9a96;--accent:#2a78d6;--good:#1baf7a;--bad:#eb6834}
body.light{--bg:#ffffff;--panel:#f6f6f4;--panel2:#ececea;--line:#dcdbd7;--text:#1f1f1d;--muted:#6b6a66}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;display:flex;flex-direction:column;overflow:hidden}
.hdr{padding:10px 14px 8px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 18px}
.q{font-size:16px;font-weight:600}.meta{font-size:13.5px;color:var(--muted)}.meta b{color:var(--text);font-weight:600}
.tg{font-size:14px;font-weight:600;margin-left:auto}
.tabs{display:flex;gap:2px;padding:2px 10px 0;border-bottom:1px solid var(--line)}
.tab{background:none;border:0;color:var(--muted);padding:7px 14px;font:inherit;font-weight:600;cursor:pointer;border-bottom:2px solid transparent}
.tab.on{color:var(--text);border-bottom-color:var(--accent)}
.pane{display:none;flex:1;min-height:0;padding:10px 12px}.pane.on{display:block}
.gv{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.4fr);gap:12px;height:100%}
.viewer{display:flex;flex-direction:column;min-width:0;min-height:0;overflow:auto;padding-right:4px}
.vhead{margin-bottom:6px;font-size:15px}
.ok{color:var(--good);font-weight:700}.no{color:var(--bad);font-weight:700}
.stage{position:relative;background:var(--panel);border-radius:8px;display:flex;align-items:center;justify-content:center;height:290px;flex:none;overflow:hidden}
.stage img{max-width:100%;max-height:290px;display:block}.ph{color:var(--muted);font-size:13px}
.nav{position:absolute;top:50%;transform:translateY(-50%);background:var(--panel2);border:1px solid var(--line);color:var(--text);width:34px;height:34px;border-radius:50%;cursor:pointer;font-size:18px;line-height:30px;z-index:3;opacity:.9}
.nav:hover{border-color:var(--accent)}.nav.prev{left:8px}.nav.next{right:8px}
.thumbs{display:flex;gap:6px;margin-top:8px;flex:none}
.thumbs div{flex:1;height:48px;border-radius:5px;background:var(--panel2) center/cover no-repeat;cursor:pointer;border:2px solid transparent;opacity:.6;position:relative}
.thumbs div.on{border-color:var(--accent);opacity:1}
.thumbs span{position:absolute;left:3px;top:1px;font-size:11px;font-weight:700;color:#fff;text-shadow:0 0 3px #000}
.box{margin-top:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px;flex:none}
.box h4{margin:0 0 4px;font-size:12px;color:var(--muted);font-weight:600}
.why{font-size:14.5px;margin:3px 0}.why .l{color:var(--muted)}.why .v{font-weight:700}
details.more{margin-top:8px;font-size:12.5px;color:var(--muted);flex:none}details.more summary{cursor:pointer}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin-top:4px}.kv span:nth-child(even){color:var(--text)}
.bar{display:flex;align-items:center;gap:8px;margin:3px 0}.bar .n{width:96px}
.bar i{display:block;height:7px;background:var(--accent);border-radius:4px;flex:none}.bar span:last-child{color:var(--text)}
.gwrap{position:relative;border-radius:8px;overflow:hidden;background:var(--bg);border:1px solid var(--line);min-width:0}
#graph{position:absolute;inset:0}
.empty{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;color:var(--muted);padding:28px;font-size:14px}
.gear{position:absolute;top:8px;right:8px;z-index:4;background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit;font-size:13px}
.panel{position:absolute;top:42px;right:8px;width:260px;z-index:4;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;display:none;font-size:12.5px}
.panel.on{display:block}
.panel label{display:flex;justify-content:space-between;align-items:center;margin:5px 0;gap:8px}
.panel input[type=range]{width:100px;accent-color:var(--accent)}.panel input[type=checkbox]{accent-color:var(--accent)}
.panel button{width:100%;margin-top:8px;background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:5px;padding:5px;cursor:pointer;font:inherit}
.story{position:absolute;left:8px;top:8px;right:150px;z-index:3;background:var(--panel);opacity:.96;border:1px solid var(--line);border-radius:8px;padding:6px 11px;pointer-events:none}
.story .t{font-size:15.5px;font-weight:700}.story .s{font-size:13.5px;margin-top:2px}
.rmS{position:absolute;top:8px;right:48px;z-index:4;background:var(--accent);border:0;color:#fff;border-radius:6px;padding:5px 11px;cursor:pointer;font:inherit;font-size:13px;font-weight:600}
.rmS.on{background:var(--panel2);color:var(--text);border:1px solid var(--line)}
.banner{position:absolute;left:50%;transform:translateX(-50%);top:78px;z-index:3;background:var(--panel);border:2px solid var(--bad);border-radius:8px;padding:7px 14px;font-size:14.5px;font-weight:600;display:none;text-align:center;max-width:90%}
.legend{position:absolute;left:8px;bottom:8px;z-index:3;font-size:12.5px;color:var(--text);background:var(--panel);opacity:.95;padding:4px 10px;border-radius:6px;border:1px solid var(--line);pointer-events:none;display:flex;flex-wrap:wrap;gap:2px 14px;max-width:calc(100% - 16px)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 5px -1px 0}
.ln{display:inline-block;width:20px;height:0;margin:0 5px 3px 0;border-top:3px solid}
table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;color:var(--muted);font-weight:600;font-size:13px;padding:7px 10px;border-bottom:1px solid var(--line)}
td{padding:7px 10px;border-bottom:1px solid var(--line)}tbody tr{cursor:pointer}tbody tr:hover{background:var(--panel)}
td .th{width:84px;height:56px;border-radius:4px;background:var(--panel2) center/cover no-repeat}
.up{color:var(--good)}.down{color:var(--bad)}
#p-rank{overflow:auto}
.float-tooltip-kap{background:var(--panel)!important;color:var(--text)!important;border:1px solid var(--line);font-size:12.5px!important;padding:4px 8px!important;border-radius:5px}
"""

_BODY = r"""
<div class="hdr" id="hdr"></div>
<div class="tabs"><button class="tab on" data-t="graph">Đồ thị giải thích</button><button class="tab" data-t="rank">Bảng xếp hạng</button></div>
<div class="pane on" id="p-graph"><div class="gv">
 <div class="viewer">
  <div class="vhead" id="vh"></div>
  <div class="stage"><button class="nav prev" title="Ảnh trước (←)">‹</button><div id="stage-img"></div><button class="nav next" title="Ảnh sau (→)">›</button></div>
  <div class="thumbs" id="thumbs"></div>
  <div id="info"></div>
 </div>
 <div class="gwrap"><div id="graph"></div><div class="empty"></div><div class="story" id="story"></div>
  <button class="rmS" id="rmS"></button><div class="banner" id="banner"></div><button class="gear">⚙</button>
  <div class="panel">
   <label>Nền sáng<input type="checkbox" data-k="light"></label>
   <label>Nhãn quan hệ<input type="checkbox" data-k="labels" checked></label><label>Tên nút<input type="checkbox" data-k="names" checked></label>
   <label>Mũi tên<input type="checkbox" data-k="arrows" checked></label><label>Đường nối về neo<input type="checkbox" data-k="paths" checked></label>
   <label>Kích thước nút<input type="range" data-k="size" min="0.5" max="2" step="0.1" value="1"></label><label>Độ dày cạnh<input type="range" data-k="width" min="0.3" max="3" step="0.1" value="1"></label>
   <label>Lực đẩy<input type="range" data-k="repel" min="20" max="800" step="10" value="160"></label>
   <label>Độ dài cạnh<input type="range" data-k="dist" min="20" max="260" step="5" value="95"></label><button id="fit">Căn vừa khung</button></div>
  <div class="legend" id="legend"></div></div></div></div>
<div class="pane" id="p-rank"><table><thead><tr id="rank-head"></tr></thead><tbody id="rows"></tbody></table></div>
"""

_JS = r"""
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=t=>String(t??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const P=D.palette;
const THEME={dark:{bg:'#1e1e1e',text:'#e3e3e0',muted:'#9a9a96',label:'#9cc0ef',qlabel:'#b8b7b2'},
             light:{bg:'#ffffff',text:'#1f1f1d',muted:'#6b6a66',label:'#1c5fae',qlabel:'#52514e'}};
const S={light:false,labels:true,names:true,arrows:true,paths:true,size:1,width:1,repel:160,dist:95};
let cur=0,hover=null,nb={},fg=null;const gel=$('#graph');
const T=()=>THEME[S.light?'light':'dark'];
function rgba(hex,a){const n=parseInt(hex.slice(1),16);return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`;}
function tab(t){$$('.tab').forEach(b=>b.classList.toggle('on',b.dataset.t===t));$$('.pane').forEach(p=>p.classList.toggle('on',p.id==='p-'+t));
  if(t==='graph'&&fg)fg.width(gel.clientWidth).height(gel.clientHeight);}
$$('.tab').forEach(b=>b.onclick=()=>tab(b.dataset.t));
// ---------- header: câu truy vấn, neo / quan hệ, hạng ảnh đúng ----------
(function(){const h=D.header;let s='<span class="q">“'+esc(h.query)+'”</span>';
  if(h.chips&&h.chips.length)s+='<span class="meta">'+h.chips.map(c=>esc(c[0])+': <b>'+esc(c[1])+'</b>').join(' · ')+'</span>';
  if(h.target)s+='<span class="tg">'+esc(h.target)+'</span>';$('#hdr').innerHTML=s;})();
// ---------- đồ thị ----------
const rad=n=>Math.sqrt(n.val)*4*S.size;
function mid(l){const s=l.source,t=l.target,c=l.__controlPoints;
  if(c&&c.length===2)return{x:.25*s.x+.5*c[0]+.25*t.x,y:.25*s.y+.5*c[1]+.25*t.y};
  if(c&&c.length===4)return{x:.125*s.x+.375*c[0]+.375*c[2]+.125*t.x,y:.125*s.y+.375*c[1]+.375*c[3]+.125*t.y};
  return{x:(s.x+t.x)/2,y:(s.y+t.y)/2};}
function active(l){return hover===null||l.source.id===hover||l.target.id===hover;}
const gone=l=>S.removed&&l.kind==='fact';
function lcol(l){if(gone(l))return rgba(P.query,.35);const c=P[l.kind]||P.fact,a=active(l);return rgba(c,l.kind==='path'?(a?.85:.15):(a?.95:.15));}
const IMG={};function imgOf(src){if(!src)return null;if(!IMG[src]){const im=new Image();im.src=src;IMG[src]=im;}return IMG[src];}
function collide(){let nodes=[];const f=alpha=>{for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
    const a=nodes[i],b=nodes[j];let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1e-3;const m=rad(a)+rad(b)+34;
    if(d<m){const k=(m-d)/d*.5*Math.min(1,alpha*4);dx*=k;dy*=k;
      if(a.fx===undefined){a.x-=dx;a.y-=dy;}if(b.fx===undefined){b.x+=dx;b.y+=dy;}}}};f.initialize=n=>{nodes=n;};return f;}
function initGraph(){ if(fg||!window.ForceGraph)return;
  fg=ForceGraph()(gel).backgroundColor(T().bg).autoPauseRedraw(false).nodeId('id').nodeVal(n=>n.val).nodeRelSize(4)
   .nodeLabel(n=>esc(n.tip)).linkLabel(l=>esc(l.tip)).linkVisibility(l=>l.kind!=='path'||S.paths)
   .linkColor(lcol).linkCurvature(l=>l.curv||0)
   .linkWidth(l=>(gone(l)?1:l.kind==='fact'?1.4+4*Math.min(1,Math.abs(l.delta)/(D.items[cur].dmax||1)):(l.kind==='query'?1.4:1))*S.width)
   .linkLineDash(l=>gone(l)?[3,5]:l.kind==='query'?[5,4]:(l.kind==='path'?[2,3]:null))
   .linkDirectionalArrowLength(l=>S.arrows&&!gone(l)?(l.kind==='fact'?6:4.5):0).linkDirectionalArrowRelPos(1).linkDirectionalArrowColor(lcol)
   .linkCanvasObjectMode(()=>'after').linkCanvasObject((l,ctx,s)=>{
      if(!S.labels||s<0.35||!l.label||(l.kind==='path'&&!S.paths))return;
      const m=mid(l),f=Math.max(11.5/s,2.5);ctx.font=f+'px system-ui,sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
      const w=ctx.measureText(l.label).width;ctx.globalAlpha=gone(l)?.3:active(l)?1:.25;
      ctx.fillStyle=T().bg;ctx.fillRect(m.x-w/2-3/s,m.y-f/2-1.5/s,w+6/s,f+3/s);
      ctx.fillStyle=l.kind==='query'?T().qlabel:(l.kind==='path'?T().muted:T().label);ctx.fillText(l.label,m.x,m.y);ctx.globalAlpha=1;})
   .nodeCanvasObject((n,ctx,s)=>{const r=rad(n),act=hover===null||(nb[hover]&&nb[hover].has(n.id));
      ctx.globalAlpha=act?1:.25;ctx.beginPath();ctx.arc(n.x,n.y,r+2/s,0,2*Math.PI);ctx.fillStyle=T().bg;ctx.fill();
      ctx.beginPath();ctx.arc(n.x,n.y,r,0,2*Math.PI);ctx.fillStyle=P[n.kind]||P.other;ctx.fill();
      const im=n.img&&imgOf(n.img);
      if(im&&im.complete&&im.naturalWidth){ctx.save();ctx.beginPath();ctx.arc(n.x,n.y,r,0,2*Math.PI);ctx.clip();
        const k=Math.max(2*r/im.naturalWidth,2*r/im.naturalHeight);ctx.drawImage(im,n.x-im.naturalWidth*k/2,n.y-im.naturalHeight*k/2,im.naturalWidth*k,im.naturalHeight*k);
        ctx.restore();ctx.beginPath();ctx.arc(n.x,n.y,r,0,2*Math.PI);ctx.lineWidth=4/s;ctx.strokeStyle=P.cand;ctx.stroke();}
      if(n.id===hover){ctx.lineWidth=2.5/s;ctx.strokeStyle=T().text;ctx.stroke();}
      ctx.globalAlpha=1;if(!S.names&&n.id!==hover)return;
      const f=Math.max(12.5/s,3);ctx.font=(n.kind!=='other'?'600 ':'')+f+'px system-ui,sans-serif';ctx.textAlign='center';ctx.textBaseline='top';
      ctx.lineWidth=3.5/s;ctx.strokeStyle=T().bg;ctx.lineJoin='round';ctx.strokeText(n.name,n.x,n.y+r+3/s);
      ctx.fillStyle=act?T().text:rgba(T().text,.3);ctx.fillText(n.name,n.x,n.y+r+3/s);
      const rk=D.roles.cand+' '+D.items[cur].rank;
      const role=n.role||(n.kind==='anchor'?(n.result?D.roles.anchor+', cũng là '+rk:D.roles.anchor):n.kind==='cand'?rk:'');
      if(role){const f2=Math.max(11/s,2.5);ctx.font=f2+'px system-ui,sans-serif';ctx.strokeText(role,n.x,n.y+r+5/s+f);
        ctx.fillStyle=n.kind==='anchor'?P.anchor:P.cand;ctx.fillText(role,n.x,n.y+r+5/s+f);}})
   .nodePointerAreaPaint((n,c,ctx)=>{ctx.fillStyle=c;ctx.beginPath();ctx.arc(n.x,n.y,rad(n)+4,0,2*Math.PI);ctx.fill();})
   .onNodeHover(n=>{hover=n?n.id:null;gel.style.cursor=n?'pointer':null;})
   .cooldownTicks(160).onEngineStop(()=>fg.zoomToFit(400,90));
  forces();}
function forces(){if(!fg)return;fg.d3Force('charge').strength(-S.repel);fg.d3Force('collide',collide());
  fg.d3Force('link').distance(l=>l.kind==='fact'?S.dist:S.dist*1.25);fg.nodeRelSize(4*S.size);fg.d3ReheatSimulation();}
function seedLayout(g,fix){// bố cục ban đầu cố định (nan quạt quanh nút neo hoặc nút ảnh) để mỗi lần mở đều ra cùng một hình
  const others=g.nodes.filter(n=>n!==fix),cands=others.filter(n=>n.kind==='cand'),rest=others.filter(n=>n.kind!=='cand'),R=S.dist*1.7;
  cands.forEach((n,i)=>{n.x=(i-(cands.length-1)/2)*S.dist;n.y=S.dist*1.2;});
  rest.forEach((n,i)=>{const a=cands.length?(-Math.PI*0.1-Math.PI*0.8*(i+.5)/rest.length):(2*Math.PI*i/rest.length-Math.PI/2);
    n.x=R*Math.cos(a)*(cands.length?1.25:1);n.y=R*Math.sin(a);});}
function curvatures(g){const grp={};
  g.links.forEach(l=>{const a=String(l.source),b=String(l.target),k=a<b?a+'|'+b:b+'|'+a;(grp[k]=grp[k]||[]).push(l);});
  Object.values(grp).forEach(arr=>arr.forEach((l,i)=>{
    if(String(l.source)===String(l.target)){l.curv=1+.4*i;return;}
    if(arr.length===1){l.curv=0;return;}
    let c=(i-(arr.length-1)/2)*.8;if(String(l.source)>String(l.target))c=-c;l.curv=c;}));}
function legend(it){const L=D.legend,kinds=new Set(it.graph.nodes.map(n=>n.kind)),lk=new Set(it.graph.links.map(l=>l.kind));const r=[];
  ['anchor','cand','other'].forEach(k=>{if(kinds.has(k)&&L[k])r.push('<span><span class="dot" style="background:'+P[k]+'"></span>'+esc(L[k])+'</span>');});
  if(lk.has('fact')&&L.fact)r.push('<span><span class="ln" style="border-color:'+P.fact+'"></span>'+esc(L.fact)+'</span>');
  if(lk.has('path')&&L.path)r.push('<span><span class="ln" style="border-color:'+P.path+';border-top-style:dotted"></span>'+esc(L.path)+'</span>');
  if(lk.has('query')&&L.query)r.push('<span><span class="ln" style="border-color:'+P.query+';border-top-style:dashed"></span>'+esc(L.query)+'</span>');
  $('#legend').innerHTML=r.join('');$('#legend').style.display=r.length?'flex':'none';}
// ---------- cột trái: chỉ phần cần cho demo, số liệu phụ trong "Chi tiết" ----------
function badge(it){return it.correct===true?' <span class="ok">✓ ảnh đúng</span>':(it.correct===false?' <span class="no">✗ chưa đúng</span>':'');}
function info(it){let s='';
  if(it.why&&it.why.length)s+='<div class="box"><h4>'+esc(D.check_title)+'</h4>'+it.why.map(x=>'<div class="why"><span class="l">'+esc(x[0])+':</span> <span class="v">'+esc(x[1])+'</span></div>').join('')+'</div>';
  let m='';if(it.more&&it.more.length)m+='<div class="kv">'+it.more.map(x=>'<span>'+esc(x[0])+'</span><span>'+esc(x[1])+'</span>').join('')+'</div>';
  if(it.bars&&it.bars.length)m+='<div style="margin-top:6px">'+esc(D.bar_title)+'</div>'+it.bars.map(b=>'<div class="bar"><span class="n">'+esc(b.label)+'</span><i style="width:'+Math.max(3,Math.round(90*b.w))+'px"></i><span>'+esc(b.text)+'</span></div>').join('');
  if(it.note)m+='<div style="margin-top:6px">'+esc(it.note)+'</div>';
  if(m)s+='<details class="more"><summary>Chi tiết</summary>'+m+'</details>';return s;}
function show(i){cur=(i+D.items.length)%D.items.length;const it=D.items[cur];hover=null;
  $('#vh').innerHTML='<b>#'+it.rank+' · '+esc(it.title)+'</b>'+badge(it);
  $('#stage-img').innerHTML=it.img?'<img src="'+it.img+'" alt="'+esc(it.title)+'">':'<span class="ph">không mở được ảnh</span>';
  $('#info').innerHTML=info(it);$$('.thumbs div').forEach((t,k)=>t.classList.toggle('on',k===cur));
  const has=it.graph.links.length>0;$('.empty').style.display=has?'none':'flex';$('.empty').textContent=it.empty||'Không có phần tử đồ thị nào để vẽ cho ảnh này.';
  $('#story').innerHTML=has?'<div class="t">'+esc(D.story_title.replace('{x}',it.title))+'</div>'+(it.story?'<div class="s">'+esc(it.story)+'</div>':''):'';
  $('#story').style.display=has?'block':'none';
  S.removed=false;$('#banner').style.display='none';const rb=$('#rmS');rb.classList.remove('on');rb.textContent=D.remove_label;
  rb.style.display=has&&it.removal?'block':'none';
  legend(it);initGraph();if(!fg)return;
  const g=JSON.parse(JSON.stringify(it.graph));nb={};const fix=g.nodes.find(n=>n.kind==='anchor')||g.nodes.find(n=>n.kind==='cand');
  g.nodes.forEach(n=>{nb[n.id]=new Set([n.id]);if((n.kind==='cand'||n.result)&&it.img){n.img=it.img;n.val=30;}if(fix&&n.id===fix.id){n.fx=0;n.fy=0;n.x=0;n.y=0;}});seedLayout(g,fix);
  g.links.forEach(l=>{if(nb[l.source])nb[l.source].add(l.target);if(nb[l.target])nb[l.target].add(l.source);});curvatures(g);
  fg.width(gel.clientWidth).height(gel.clientHeight).graphData(has?g:{nodes:[],links:[]});
  const c0=cur;[60,700,1800].forEach(t=>setTimeout(()=>{if(fg&&cur===c0){fg.width(gel.clientWidth).height(gel.clientHeight);fg.zoomToFit(0,90);}},t));}
$('#thumbs').innerHTML=D.items.map((it,k)=>'<div title="'+esc(it.title)+'" style="background-image:url('+(it.img||'')+')" onclick="show('+k+')"><span>#'+it.rank+(it.correct?' ✓':'')+'</span></div>').join('');
$('.prev').onclick=()=>show(cur-1);$('.next').onclick=()=>show(cur+1);
document.addEventListener('keydown',e=>{if(!$('#p-graph').classList.contains('on'))return;if(e.key==='ArrowLeft')show(cur-1);if(e.key==='ArrowRight')show(cur+1);});
$('.gear').onclick=()=>$('.panel').classList.toggle('on');
$('#rmS').onclick=()=>{const it=D.items[cur];S.removed=!S.removed;const rb=$('#rmS');rb.classList.toggle('on',S.removed);
  rb.textContent=S.removed?D.restore_label:D.remove_label;$('#banner').textContent=it.removal||'';$('#banner').style.display=S.removed?'block':'none';};
$$('.panel input').forEach(inp=>inp.oninput=()=>{const k=inp.dataset.k;S[k]=inp.type==='checkbox'?inp.checked:+inp.value;
  if(k==='light'){document.body.classList.toggle('light',S.light);if(fg)fg.backgroundColor(T().bg);}
  if(['repel','dist','size'].includes(k))forces();});
$('#fit').onclick=()=>fg&&fg.zoomToFit(400,90);
// ---------- bảng xếp hạng ----------
$('#rank-head').innerHTML=['Hạng','Ảnh','Kết quả','Ảnh đúng','Hạng nếu chỉ dùng CLIP',D.removed_col].map(x=>'<th>'+esc(x)+'</th>').join('');
$('#rows').innerHTML=D.items.map((it,k)=>{const r=it.row||{},cr=r.clip_rank,d=(cr??it.rank)-it.rank;
  return '<tr onclick="tab(\'graph\');show('+k+')"><td><b>#'+it.rank+'</b></td><td><div class="th" style="background-image:url('+(it.img||'')+')"></div></td><td>'+esc(it.title)+'</td><td>'+(it.correct===true?'<span class="ok">✓</span>':'')+'</td><td>'+(cr==null?'':'#'+esc(cr))+(d>0?' <span class="up">▲'+d+'</span>':d<0?' <span class="down">▼'+(-d)+'</span>':'')+'</td><td>'+esc(r.removed??'')+'</td></tr>';}).join('');
window.addEventListener('resize',()=>fg&&fg.width(gel.clientWidth).height(gel.clientHeight));
(function(){const h=String(location.hash||'');if(h.includes('light')){S.light=true;document.body.classList.add('light');$('.panel input[data-k=light]').checked=true;}
  const m=h.match(/i=(\d+)/);if(m)cur=+m[1];if(h.includes('rank'))tab('rank');if(h.includes('more'))setTimeout(()=>$$('details.more').forEach(d=>d.open=true),0);
  if(h.includes('rm'))setTimeout(()=>$('#rmS').click(),50);})();
show(cur);if(fg)fg.backgroundColor(T().bg);
"""


def _prepare(items):
    for it in items:
        g = it.setdefault("graph", {"nodes": [], "links": []})
        it["dmax"] = max([abs(l["delta"]) for l in g["links"] if l["kind"] == "fact"] + [1e-12])
    return items


def results_page(header, items, legend=None, check_title="Kiểm chứng: bỏ giải thích rồi chấm lại",
                 bar_title="Che từng phương thức rồi chấm lại:", removed_col="Bỏ S → hạng", title="Demo truy vấn có giải thích",
                 story_title="Vì sao trả về {x}?", remove_label="Bỏ giải thích S", restore_label="Khôi phục S",
                 roles=None):
    """header: dict(query, chips=[(nhãn, giá trị)], target). items: danh sách kết quả theo hạng (xem đầu file).
    Mỗi item có thêm: story (một câu nêu bằng chứng mạnh nhất), removal (câu hiện khi bấm "Bỏ giải thích S")."""
    data = dict(header=header, items=_prepare(items), palette=PALETTE, legend={**DEFAULT_LEGEND, **(legend or {})},
                check_title=check_title, bar_title=bar_title, removed_col=removed_col, story_title=story_title,
                remove_label=remove_label, restore_label=restore_label,
                roles={"anchor": "neo của truy vấn", "cand": "ảnh hạng", **(roles or {})})
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (f"<!DOCTYPE html><html lang='vi'><head><meta charset='utf-8'><title>{_html.escape(title)}</title><style>{_CSS}</style>"
            f"{force_graph_js()}</head><body>{_BODY}<script>const D={payload};{_JS}</script></body></html>")


def results_iframe(header, items, height=PAGE_HEIGHT, **kw):
    page = results_page(header, items, **kw)
    return (f'<iframe srcdoc="{_html.escape(page, quote=True)}" style="width:100%;height:{height}px;border:0;'
            f'border-radius:8px;background:#1e1e1e"></iframe>')


def message_html(text):
    return f'<div style="padding:14px;color:#9a9a96;font:14px system-ui,sans-serif">{_html.escape(text)}</div>'


def launch(demo, in_colab, height=PAGE_HEIGHT + 160):
    """Colab: mặc định tạo link công khai (*.gradio.live, tự hết hạn) để mở trang ở tab riêng khi quay video.
    Đặt XAI_DEMO_SHARE=0 để chỉ hiện trong notebook. Tắt notebook sau khi quay để link ngừng hoạt động."""
    share = os.environ.get("XAI_DEMO_SHARE", "1" if in_colab else "0") == "1"
    demo.queue()
    if share:
        return demo.launch(share=True, inline=False, show_error=True)
    return demo.launch(share=False, inline=in_colab, height=height, show_error=True)
