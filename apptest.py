# =========================================================
# PEA SPARK ENTERPRISE - UPDATED FEEDER COLORS
# apptest.py
# =========================================================

from flask import Flask, jsonify, render_template, request
import json, os, traceback
import networkx as nx
from scipy.spatial import KDTree

app = Flask(__name__)

G = None
NODE_LIST = []
TREE = None
SWITCH_NODES = {}
SWITCH_STATUS = {}
FEEDER_COLOR = {}  # เก็บสีแยกตาม FEEDERID
FAULT_NODE = None
FAULT_FEEDER = None

# พาเลทสีมาตรฐานสำหรับโครงข่ายไฟฟ้า (20 สีเพื่อให้ครอบคลุม)
DISTINCT_COLORS = [
    "#FF5733", "#33FF57", "#3357FF", "#F333FF", "#33FFF3", 
    "#F3FF33", "#FF8333", "#33FF83", "#8333FF", "#FF3383",
    "#00CCFF", "#FFCC00", "#FF0066", "#00FF99", "#9900FF",
    "#66FF00", "#FF6600", "#0066FF", "#CC00FF", "#00FFCC"
]

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def build():
    global G, NODE_LIST, TREE, FEEDER_COLOR, SWITCH_NODES
    try:
        G = nx.Graph()
        nodes = []
        
        # ค้นหาไฟล์ (รองรับทั้ง .json และ .geojson)
        path = "data/psconductor.geojson" if os.path.exists("data/psconductor.geojson") else "data/psconductor.json"
        data = load_json(path)
        if not data: return

        # 1. รวบรวม FEEDERID ทั้งหมดเพื่อกำหนดสี
        unique_feeders = sorted(list(set(str(f["properties"].get("FEEDERID", "UNK")).strip() for f in data["features"])))
        FEEDER_COLOR = {fid: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, fid in enumerate(unique_feeders)}

        # 2. สร้าง Network Graph
        for f in data["features"]:
            geom = f["geometry"]
            feeder = str(f["properties"].get("FEEDERID", "UNK")).strip()
            
            if geom["type"] == "LineString":
                coords = geom["coordinates"]
                for i in range(len(coords)-1):
                    a, b = tuple(coords[i][:2]), tuple(coords[i+1][:2])
                    G.add_edge(a, b, feeder=feeder)
                    nodes += [a, b]

        NODE_LIST = list(set(nodes))
        TREE = KDTree(NODE_LIST)

        # 3. โหลดอุปกรณ์ Switch/DOF
        d_path = "data/DOF.geojson" if os.path.exists("data/DOF.geojson") else "data/DOF.json"
        d_data = load_json(d_path)
        if d_data:
            for f in d_data["features"]:
                fid = str(f["properties"].get("FACILITYID", f["properties"].get("DEVICEID", "")))
                if not fid: continue
                coord = tuple(f["geometry"]["coordinates"][:2])
                dist, idx = TREE.query(coord)
                SWITCH_NODES[fid] = NODE_LIST[idx]
                if fid not in SWITCH_STATUS: SWITCH_STATUS[fid] = 1 # ปิดวงจรเป็นค่าเริ่มต้น
        
        print(f"Build Success: Found {len(unique_feeders)} Feeders")
    except:
        print(traceback.format_exc())

# รัน build เมื่อเริ่มโปรแกรม
build()

def get_active_nodes():
    active = set()
    # กำหนดจุด Source (เช่นจุดแรกของระบบ หรือกำหนดจาก Breaker)
    sources = [NODE_LIST[0]] if NODE_LIST else [] 
    stack = list(sources)
    while stack:
        n = stack.pop()
        if n in active: continue
        active.add(n)
        for nbr in G.neighbors(n):
            if nbr == FAULT_NODE: continue
            # ตรวจสอบสถานะ Switch
            blocked = False
            for fid, snode in SWITCH_NODES.items():
                if snode == nbr and SWITCH_STATUS.get(fid) == 0:
                    blocked = True; break
            if not blocked: stack.append(nbr)
    return active

@app.route("/")
def index():
    return render_template("indexpro.html")

@app.route("/api/conductor")
def api_conductor():
    active = get_active_nodes()
    path = "data/psconductor.geojson" if os.path.exists("data/psconductor.geojson") else "data/psconductor.json"
    data = load_json(path)
    feats = []
    for f in data["features"]:
        feeder = str(f["properties"].get("FEEDERID", "UNK")).strip()
        # เช็คสถานะการจ่ายไฟ
        coords = f["geometry"]["coordinates"]
        is_on = any(tuple(c[:2]) in active for c in coords)
        
        f["properties"]["status"] = "on" if is_on else "off"
        f["properties"]["color"] = FEEDER_COLOR.get(feeder, "#888888")
        feats.append(f)
    return jsonify({"type": "FeatureCollection", "features": feats})

@app.route("/api/dof")
def api_dof():
    path = "data/DOF.geojson" if os.path.exists("data/DOF.geojson") else "data/DOF.json"
    data = load_json(path)
    feats = []
    for f in data["features"]:
        fid = str(f["properties"].get("FACILITYID", f["properties"].get("DEVICEID", "")))
        if not fid: continue
        pos = SWITCH_STATUS.get(fid, 1)
        f["properties"]["status"] = pos
        f["properties"]["id"] = fid
        feats.append(f)
    return jsonify({"type": "FeatureCollection", "features": feats})

@app.route("/api/scada")
def scada_stats():
    active = get_active_nodes()
    return jsonify({
        "energized_pct": round(len(active)/len(NODE_LIST)*100, 1) if NODE_LIST else 0,
        "open_switches": list(SWITCH_STATUS.values()).count(0),
        "total_switches": len(SWITCH_STATUS),
        "nodes_off": len(NODE_LIST) - len(active),
        "fault_feeder": FAULT_FEEDER or "none"
    })

@app.route("/toggle_switch")
def toggle():
    fid = request.args.get("id")
    if fid in SWITCH_STATUS:
        SWITCH_STATUS[fid] = 1 - SWITCH_STATUS[fid]
    return jsonify({"ok": True, "status": SWITCH_STATUS.get(fid)})

@app.route("/fault")
def set_fault():
    global FAULT_NODE, FAULT_FEEDER
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    dist, idx = TREE.query((lon, lat))
    FAULT_NODE = NODE_LIST[idx]
    for nbr in G.neighbors(FAULT_NODE):
        FAULT_FEEDER = G[FAULT_NODE][nbr].get("feeder", "UNK")
        break
    return jsonify({"ok": True, "feeder": FAULT_FEEDER})

@app.route("/clear_fault")
def clear():
    global FAULT_NODE, FAULT_FEEDER
    FAULT_NODE, FAULT_FEEDER = None, None
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
