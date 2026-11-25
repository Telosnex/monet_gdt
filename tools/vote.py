# Founder voting on color triads: two arms (e.g. old vs new scorer), blind
# cards and a side-by-side grid, one append-only vote log per session.
#
#   .venv/bin/python tools/vote.py serve <session_dir> [--port 8833]
#   .venv/bin/python tools/vote.py report <session_dir>
#
# Use /grid?blind=1 for color votes (ADR D12): big swatches on /blind are
# judged as colors on their own, apart from the wallpaper.
#
# <session_dir>/session.json (written by the experiment):
#   {"question": str, "arms": ["old", "new"], "seed": int, "repeats": int,
#    "decision": str, optional "choices": {left|right|tie|unsure: button text},
#    "items": [{"id": str, "image": abs path or null, "triads": {arm: [hex, hex, hex]},
#               "meta": {...}}]}
# Triads are [primary, secondary, tertiary], drawn as a large primary (the
# app's main color) and, beside it, secondary above tertiary, labeled P/S/T.
# A two-color entry [p, x] is drawn as a pair (P beside x), for questions
# about two colors.
#
# Pages (http://127.0.0.1:<port>/):
#   /blind        one card at a time, sides random per card; `repeats` cards
#                 return later with sides swapped (consistency check).
#   /grid?blind=1 every item as a row, sides random per row, arms unnamed;
#                 vote per row.
#   /grid         columns named by arm; shows every vote so far; vote per row.
# Grid rows: left palette, wallpaper, right palette, top-aligned.
# Keys on /blind: ← left · → right · ↓ no real difference · ↑ unsure.
# Keys on /grid: the same four vote and move to the next row; j / Tab next
# row, k / Shift+Tab previous row.
# votes.jsonl records the arm on each side, so a vote never depends on
# recomputing the layout. A later vote on the same item and page replaces
# the earlier one in the report; nothing is deleted.
import argparse
import datetime
import html
import http.server
import json
import math
import os
import random
import urllib.parse

KEYS = ("left", "right", "tie", "unsure")
# Blind-page button text; a session may override with "choices": {key: text}.
CHOICES = {
    "left": "Left is better",
    "right": "Right is better",
    "tie": "No real difference",
    "unsure": "Unsure",
}


def load(d):
    S = json.load(open(os.path.join(d, "session.json")))
    assert len(S["arms"]) == 2, "two arms"
    S["by_id"] = {it["id"]: it for it in S["items"]}
    return S


def votes(d):
    p = os.path.join(d, "votes.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def sides(S, salt):
    """{item id: arm on the left}, fixed by seed and salt."""
    rnd = random.Random(f"{S.get('seed', 0)}:{salt}")
    return {it["id"]: S["arms"][rnd.random() < 0.5] for it in S["items"]}


def blind_queue(S):
    """[(entry, item id, left arm, is repeat)] in a fixed order."""
    rnd = random.Random(f"{S.get('seed', 0)}:queue")
    left = sides(S, "blind")
    q = [(i, i, left[i], False) for i in (it["id"] for it in S["items"])]
    rnd.shuffle(q)
    other = {a: b for a, b in [S["arms"], S["arms"][::-1]]}
    for i in rnd.sample([e[1] for e in q], min(S.get("repeats", 0), len(q))):
        e = (i + "#r", i, other[left[i]], True)
        q.insert(rnd.randrange(min(len(q), 10), len(q) + 1), e)
    return q


def pick(v):
    return {"left": v["left"], "right": v["right"], "tie": "tie", "unsure": "unsure"}[
        v["key"]
    ]


SW = """<div class=tri>%s</div>"""


def _ink(h):
    r, g, b = (int(h[i : i + 2], 16) for i in (1, 3, 5))
    return "#000" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "#fff"


def triad_html(t):
    """Primary large; secondary then tertiary below it in reading order, labeled.
    A two-color entry [p, x] is a pair: P beside the candidate."""
    if len(t) == 2:
        sw = lambda h, lab: (
            f'<div class=p style="background:{h};color:{_ink(h)}"><span>{lab}</span></div>'
        )
        return SW % (sw(t[0], "P") + sw(t[1], ""))
    p, s, x = t
    sw = lambda h, cls, lab: (
        f'<div class={cls} style="background:{h};color:{_ink(h)}"><span>{lab}</span></div>'
    )
    return SW % (
        sw(p, "p", "P")
        + "<div class=acc>"
        + sw(s, "s", "S")
        + sw(x, "t", "T")
        + "</div>"
    )


CSS = """<style>
body{margin:0;font:15px system-ui;background:#141414;color:#ddd}
header{padding:10px 18px;border-bottom:1px solid #2a2a2a;line-height:1.5;position:sticky;top:0;background:#141414;z-index:2}
.tri{display:flex;gap:6px;height:150px} .tri .p{flex:3;border-radius:6px}
.tri .acc{flex:2;display:flex;flex-direction:column;gap:6px} .tri .s,.tri .t{flex:1;border-radius:6px}
.tri span{display:block;padding:4px 7px;font:600 12px system-ui;opacity:.75}
button{background:#222;color:#ddd;border:1px solid #444;border-radius:6px;padding:7px 12px;font:inherit;cursor:pointer}
button.on{background:#3d5afe;border-color:#3d5afe;color:#fff} kbd{background:#333;border-radius:4px;padding:1px 6px;margin-right:6px;font:12px ui-monospace,monospace}
a{color:#8ab4f8}
</style>"""

BLIND = (
    """<!doctype html><meta charset=utf-8><title>Blind</title>"""
    + CSS
    + """<style>
body{height:100vh;display:flex;flex-direction:column} #c{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;padding:12px}
#im{max-width:92vw;max-height:52vh;border-radius:6px} #pair{display:flex;gap:40px;width:min(92vw,1100px)} #pair>div{flex:1} .lab{font-weight:600;margin-bottom:6px}
footer{display:flex;gap:10px;justify-content:center;padding:10px 10px 16px}</style>
<header><b>QUESTION</b><br><span id=n></span> · <a href="/grid?blind=1">grid (blind)</a> · <a href="/grid">grid (named)</a></header>
<div id=c><img id=im><div id=pair><div><div class=lab>Left</div><div id=L></div></div><div><div class=lab>Right</div><div id=R></div></div></div></div>
<footer><button data-k=left><kbd>←</kbd>Left is better</button><button data-k=right><kbd>→</kbd>Right is better</button><button data-k=tie><kbd>↓</kbd>No real difference</button><button data-k=unsure><kbd>↑</kbd>Unsure</button></footer>
<script>let Q=[],i=0,busy=false;
function show(){if(i>=Q.length){n.textContent='Done. Thank you.';c.innerHTML='';return}const e=Q[i];
n.textContent=`${i+1}/${Q.length}`;im.style.display=e.image?'':'none';if(e.image)im.src='/image?id='+encodeURIComponent(e.item);L.innerHTML=e.left_html;R.innerHTML=e.right_html}
async function vote(k){if(busy||i>=Q.length)return;busy=true;const e=Q[i];
const r=await fetch('/vote',{method:'POST',body:JSON.stringify({page:'blind',entry:e.entry,item:e.item,left:e.left,right:e.right,repeat:e.repeat,key:k})});
busy=false;if(!r.ok){alert('vote not saved');return}i++;show()}
document.querySelectorAll('footer button').forEach(b=>b.onclick=()=>vote(b.dataset.k));
const K={ArrowLeft:'left',ArrowRight:'right',ArrowDown:'tie',ArrowUp:'unsure'};
document.onkeydown=e=>{if(K[e.key]){e.preventDefault();vote(K[e.key])}};
fetch('/queue').then(r=>r.json()).then(j=>{Q=j;show()});</script>"""
)

GRID_CSS = """<style>
.row{display:grid;grid-template-columns:1fr minmax(260px,380px) 1fr;gap:24px;align-items:start;
padding:14px 18px;border-bottom:1px solid #2a2a2a;border-left:4px solid transparent;outline:none;scroll-margin-top:110px}
.row.cur{border-left-color:#3d5afe;background:#1b1d26}
.btns{display:flex;gap:6px;margin-top:8px;justify-content:center}
</style>"""

GRID_JS = """<script>
const rows=[...document.querySelectorAll('.row')];let cur=0,busy=false;
function focus(i,scroll=true){if(!rows.length)return;cur=Math.max(0,Math.min(rows.length-1,i));
rows.forEach((r,j)=>r.classList.toggle('cur',j===cur));if(scroll)rows[cur].scrollIntoView({block:'start',behavior:'smooth'})}
async function vote(row,k){if(busy)return false;busy=true;
const r=await fetch('/vote',{method:'POST',body:JSON.stringify({page:PAGE,item:row.dataset.item,left:row.dataset.left,right:row.dataset.right,repeat:false,key:k})});
busy=false;if(!r.ok){alert('vote not saved');return false}
row.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.k===k));return true}
rows.forEach((row,i)=>{row.onclick=()=>focus(i,false);
row.querySelectorAll('button').forEach(b=>b.onclick=async e=>{e.stopPropagation();focus(i,false);await vote(row,b.dataset.k)})});
const K={ArrowLeft:'left',ArrowRight:'right',ArrowDown:'tie',ArrowUp:'unsure'};
document.addEventListener('keydown',async e=>{
if(e.metaKey||e.ctrlKey||e.altKey)return;
if(K[e.key]){e.preventDefault();if(await vote(rows[cur],K[e.key]))focus(cur+1);return}
if(e.key==='j'||(e.key==='Tab'&&!e.shiftKey)){e.preventDefault();focus(cur+1);return}
if(e.key==='k'||(e.key==='Tab'&&e.shiftKey)){e.preventDefault();focus(cur-1);return}
});
const first=rows.findIndex(r=>!r.querySelector('button.on'));focus(first<0?0:first,first>0);
</script>"""


def grid_page(d, S, blind):
    V = votes(d)
    page = "grid_blind" if blind else "grid_named"
    left = sides(S, "grid") if blind else {it["id"]: S["arms"][0] for it in S["items"]}
    rnd = random.Random(f"{S.get('seed', 0)}:gridorder")
    items = list(S["items"])
    if blind:
        rnd.shuffle(items)
    last = {}
    for v in V:
        if v.get("page") == page:
            last[v["item"]] = v
    shown = {}  # named grid: every earlier vote per item, by arm
    for v in V:
        shown.setdefault(v["item"], []).append(
            f"{v['page'].replace('grid_', 'grid ')}: {pick(v)}"
            + (" (repeat)" if v.get("repeat") else "")
        )
    a, b = S["arms"]
    head = f"<b>{html.escape(S['question'])}</b><br>{len(items)} items · "
    head += (
        "sides random per row, arms hidden"
        if blind
        else f"left column: <b>{a}</b> · right column: <b>{b}</b>"
    )
    head += ' · <a href="/blind">blind cards</a> · <a href="/grid?blind=1">grid (blind)</a> · <a href="/grid">grid (named)</a>'
    rows = []
    for it in items:
        L = left[it["id"]]
        R = b if L == a else a
        cur = last.get(it["id"])
        cur_side = None
        if cur:
            cur_side = {"tie": "tie", "unsure": "unsure"}.get(cur["key"]) or (
                "left" if pick(cur) == L else "right"
            )
        btn = "".join(
            f"<button data-k={k}{' class=on' if cur_side == k else ''}>{lab}</button>"
            for k, lab in [
                ("left", "◀ left"),
                ("tie", "same"),
                ("unsure", "?"),
                ("right", "right ▶"),
            ]
        )
        note = (
            ""
            if blind
            else '<div style="font-size:12px;color:#999">'
            + html.escape(" · ".join(shown.get(it["id"], [])))
            + "</div>"
        )
        img = (
            f'<img loading=lazy src="/image?id={urllib.parse.quote(it["id"])}" '
            'style="width:100%;max-height:260px;object-fit:contain;border-radius:6px;display:block">'
            if it.get("image")
            else ""
        )
        rows.append(
            f'<div class=row tabindex=-1 data-item="{html.escape(it["id"])}" data-left="{L}" data-right="{R}">'
            f"<div>{triad_html(it['triads'][L])}</div>"
            f"<div>{img}<div class=btns>{btn}</div>{note}</div>"
            f"<div>{triad_html(it['triads'][R])}</div></div>"
        )
    return (
        "<!doctype html><meta charset=utf-8><title>Grid</title>"
        + CSS
        + GRID_CSS
        + f'<header>{head}<br><span style="font-size:13px;color:#aaa">Keys: <kbd>j</kbd><kbd>k</kbd> or '
        "<kbd>Tab</kbd> next/previous row · <kbd>←</kbd> left better · <kbd>→</kbd> right better · "
        "<kbd>↓</kbd> same · <kbd>↑</kbd> unsure (then moves to the next row)</span></header>"
        + "".join(rows)
        + f"<script>const PAGE={json.dumps(page)};</script>"
        + GRID_JS
    )


def serve(d, port):
    S = load(d)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(u.query)
            if u.path in ("/", "/blind"):
                page = BLIND.replace("QUESTION", html.escape(S["question"]))
                for k, default in CHOICES.items():
                    page = page.replace(
                        default, html.escape(S.get("choices", {}).get(k, default))
                    )
                self.send(200, page.encode(), "text/html")
            elif u.path == "/queue":
                done = {v["entry"] for v in votes(d) if v.get("page") == "blind"}
                out = [
                    {
                        "entry": e,
                        "item": i,
                        "left": L,
                        "right": [x for x in S["arms"] if x != L][0],
                        "repeat": r,
                        "image": bool(S["by_id"][i].get("image")),
                        "left_html": triad_html(S["by_id"][i]["triads"][L]),
                        "right_html": triad_html(
                            S["by_id"][i]["triads"][[x for x in S["arms"] if x != L][0]]
                        ),
                    }
                    for e, i, L, r in blind_queue(S)
                    if e not in done
                ]
                self.send(200, json.dumps(out).encode(), "application/json")
            elif u.path == "/grid":
                self.send(
                    200,
                    grid_page(d, S, qs.get("blind", ["0"])[0] == "1").encode(),
                    "text/html",
                )
            elif u.path == "/image":
                it = S["by_id"].get(qs.get("id", [""])[0])
                if not it or not it.get("image"):
                    return self.send(404, b"no such item", "text/plain")
                ext = os.path.splitext(it["image"])[1].lower()
                self.send(
                    200,
                    open(it["image"], "rb").read(),
                    {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg"),
                )
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                v = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert (
                    v["key"] in KEYS
                    and v["item"] in S["by_id"]
                    and {v["left"], v["right"]} == set(S["arms"])
                )
                assert v["page"] in ("blind", "grid_blind", "grid_named")
                v.setdefault("entry", v["item"])
                v["t"] = datetime.datetime.now().isoformat(timespec="seconds")
                with open(os.path.join(d, "votes.jsonl"), "a") as f:
                    f.write(json.dumps(v) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                self.send(200, b"ok", "text/plain")
            except Exception as e:
                self.send(400, str(e).encode(), "text/plain")

    print(
        f"{S['question']}\nhttp://127.0.0.1:{port}/blind  ·  /grid?blind=1  ·  /grid",
        flush=True,
    )
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n) if n else 1.0


def report(d):
    S = load(d)
    V = votes(d)
    a, b = S["arms"]
    out = {"arms": S["arms"], "decision": S.get("decision")}
    for page in ("blind", "grid_blind", "grid_named"):
        first = {}
        for v in V:
            if v["page"] == page and not v.get("repeat"):
                first[v["item"]] = pick(v)
        if not first:
            continue
        c = {k: sum(x == k for x in first.values()) for k in (a, b, "tie", "unsure")}
        r = {"n": len(first), "votes": c, "sign_test_p": round(sign_p(c[a], c[b]), 4)}
        groups = sorted(
            {S["by_id"][i].get("meta", {}).get("group") for i in first} - {None}
        )
        for g in groups:
            sel = [
                x
                for i, x in first.items()
                if S["by_id"][i].get("meta", {}).get("group") == g
            ]
            cg = {k: sum(x == k for x in sel) for k in (a, b, "tie", "unsure")}
            r.setdefault("by_group", {})[g] = {
                "votes": cg,
                "sign_test_p": round(sign_p(cg[a], cg[b]), 4),
            }
        if page == "blind":
            rep = {
                v["item"]: pick(v)
                for v in V
                if v["page"] == "blind" and v.get("repeat")
            }
            r["repeat_same"] = (
                f"{sum(first.get(i) == x for i, x in rep.items())}/{len(rep)}"
            )
        out[page] = r
        out.setdefault("_first", {})[page] = first
    F = out.pop("_first", {})
    if "blind" in F and "grid_blind" in F:
        both = [i for i in F["blind"] if i in F["grid_blind"]]
        out["blind_vs_grid_blind_same"] = (
            f"{sum(F['blind'][i] == F['grid_blind'][i] for i in both)}/{len(both)}"
        )
    json.dump(out, open(os.path.join(d, "report.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve", "report"])
    ap.add_argument("dir")
    ap.add_argument("--port", type=int, default=8833)
    x = ap.parse_args()
    d = os.path.abspath(x.dir)
    serve(d, x.port) if x.cmd == "serve" else report(d)
