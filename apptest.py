from flask import Flask, jsonify, render_template, request
import os, traceback
import networkx as nx
from scipy.spatial import KDTree
import geopandas as gpd

app = Flask(__name__)

# ── global state ──────────────────────────────────────────
G          = None
NODE_LIST  = []
TREE       = None

SWITCH_NODES  = {}
SWITCH_STATUS = {}
SWITCH_TYPE   = {}
FEEDER_COLOR  = {}

FAULT_NODE   = None
FAULT_FEEDER = None

# ── cache อ่านไฟล์ครั้งเดียวตอน build ───────────────────
_CONDUCTOR_FEATURES = []
_DOF_FEATURES       = []

BUILD_ERROR = None   # เก็บ traceback ถ้า build() ล้มเหลว

# ==========================================================
def load_shp(path):
    """โหลด Shapefile แล้วแปลงเป็น GeoJSON-like dict (CRS → EPSG:4326)"""
    abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"ไม่พบไฟล์: {abs_path}")
    gdf = gpd.read_file(abs_path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    import json as _json
    return _json.loads(gdf.to_json())

# ==========================================================
def infer_switch_type(fid, props):
    subtype = str(props.get("SUBTYPE", "")).upper()
    name    = fid.upper()
    if "RECLOSER" in subtype or "REC" in name or name.endswith("R"):
        return "Recloser"
    if "LOADBREAK" in subtype or "LB" in subtype or "LBS" in name:
        return "Load Break"
    if "TVS" in name or "VS" in subtype:
        return "Switch"
    return "Disconnect"

# ==========================================================
def build():
    global G, NODE_LIST, TREE, BUILD_ERROR
    global _CONDUCTOR_FEATURES, _DOF_FEATURES
    global SWITCH_NODES, SWITCH_STATUS, SWITCH_TYPE, FEEDER_COLOR

    BUILD_ERROR = None
    try:
        # ── conductor ─────────────────────────────────────
        conductor_data      = load_shp("data/psconductor.shp")
        _CONDUCTOR_FEATURES = conductor_data["features"]

        G     = nx.Graph()
        nodes = []

        for f in _CONDUCTOR_FEATURES:
            geom   = f["geometry"]
            feeder = str(f["properties"].get("FEEDERID", "UNK"))
            if geom["type"] != "LineString":
                continue
            coords = geom["coordinates"]
            for i in range(len(coords) - 1):
                a = tuple(coords[i])
                b = tuple(coords[i + 1])
                G.add_edge(a, b, feeder=feeder)
                nodes += [a, b]

        if not nodes:
            raise ValueError("psconductor.shp ไม่มี LineString features")

        NODE_LIST = list(set(nodes))
        TREE      = KDTree(NODE_LIST)

        # ── DOF / switches ────────────────────────────────
        dof_data      = load_shp("data/DOF.shp")
        _DOF_FEATURES = dof_data["features"]

        for f in _DOF_FEATURES:
            fid   = str(f["properties"].get("FACILITYID", ""))
            props = f["properties"]
            if "S" not in fid.upper():
                continue
            pos       = int(props.get("PRESENTPOS", 1))
            lon, lat  = f["geometry"]["coordinates"]
            _, idx    = TREE.query([lon, lat])
            SWITCH_NODES[fid]  = NODE_LIST[idx]
            SWITCH_STATUS[fid] = pos
            SWITCH_TYPE[fid]   = infer_switch_type(fid, props)

        # ── feeder colours ────────────────────────────────
        palette = ["#00e5ff","#7c4dff","#ff9100","#00e676",
                   "#ff5252","#ffd600","#e040fb","#40c4ff"]
        for i, fdr in enumerate(sorted(set(nx.get_edge_attributes(G, "feeder").values()))):
            FEEDER_COLOR[fdr] = palette[i % len(palette)]

        print(f"[build] OK  nodes={len(NODE_LIST)}  edges={G.number_of_edges()}  "
              f"switches={len(SWITCH_STATUS)}  feeders={len(FEEDER_COLOR)}")

    except Exception:
        BUILD_ERROR = traceback.format_exc()
        print(f"[build] FAILED\n{BUILD_ERROR}")

# ==========================================================
def require_build():
    if G is None or BUILD_ERROR:
        return jsonify({"error": "build() ยังไม่สำเร็จ",
                        "detail": BUILD_ERROR or "G is None"}), 503
    return None

# ==========================================================
def get_active_nodes():
    G2 = G.copy()
    if FAULT_NODE and FAULT_NODE in G2:
        G2.remove_node(FAULT_NODE)
    for fid, node in SWITCH_NODES.items():
        if SWITCH_STATUS.get(fid, 1) == 0 and node in G2:
            G2.remove_node(node)
    active = set()
    for component in nx.connected_components(G2):
        active |= component
    return active

# ==========================================================
@app.route("/")
def index():
    return render_template("indexpro.html")

# ==========================================================
@app.route("/api/debug")
def debug():
    """เปิด /api/debug เพื่อตรวจสอบสถานะ build"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    return jsonify({
        "build_ok":       G is not None and BUILD_ERROR is None,
        "build_error":    BUILD_ERROR,
        "nodes":          len(NODE_LIST),
        "edges":          G.number_of_edges() if G else 0,
        "switches":       len(SWITCH_STATUS),
        "feeders":        list(FEEDER_COLOR.keys()),
        "conductor_rows": len(_CONDUCTOR_FEATURES),
        "dof_rows":       len(_DOF_FEATURES),
        "data_dir":       data_dir,
        "data_files":     os.listdir(data_dir) if os.path.isdir(data_dir) else [],
        "cwd":            os.getcwd(),
    })

# ==========================================================
@app.route("/api/conductor")
def conductor():
    err = require_build()
    if err: return err
    try:
        active = get_active_nodes()
        feats  = []
        for f in _CONDUCTOR_FEATURES:
            geom   = f["geometry"]
            feeder = str(f["properties"].get("FEEDERID", "UNK"))
            if geom["type"] != "LineString":
                continue
            coords = geom["coordinates"]
            status = "off" if any(tuple(c) not in active for c in coords) else "on"
            feats.append({
                "type": "Feature", "geometry": geom,
                "properties": {"feeder": feeder, "status": status,
                               "color": FEEDER_COLOR.get(feeder, "#888")}
            })
        return jsonify({"type": "FeatureCollection", "features": feats})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

# ==========================================================
@app.route("/api/dof")
def dof():
    err = require_build()
    if err: return err
    try:
        feats = []
        for f in _DOF_FEATURES:
            fid   = str(f["properties"].get("FACILITYID", ""))
            props = f["properties"]
            if "S" not in fid.upper():
                continue
            pos     = SWITCH_STATUS.get(fid, 1)
            sw_type = SWITCH_TYPE.get(fid, "Disconnect")
            lon, lat = f["geometry"]["coordinates"]
            _, idx   = TREE.query([lon, lat])
            near     = NODE_LIST[idx]
            feeder   = "UNK"
            for u, v, d in G.edges(near, data=True):
                feeder = d.get("feeder", "UNK")
                break
            location = str(props.get("LOCATION",
                           props.get("STREETNAME",
                           props.get("SUBSTATION", ""))))
            feats.append({
                "type": "Feature", "geometry": f["geometry"],
                "properties": {"id": fid,
                               "state":    "CLOSE" if pos == 1 else "OPEN",
                               "status":   pos,
                               "type":     sw_type,
                               "feeder":   feeder,
                               "location": location}
            })
        return jsonify({"type": "FeatureCollection", "features": feats})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

# ==========================================================
@app.route("/toggle_switch")
def toggle():
    err = require_build()
    if err: return err
    fid = request.args.get("id")
    if fid in SWITCH_STATUS:
        SWITCH_STATUS[fid] = 1 - SWITCH_STATUS[fid]
    return jsonify({"id": fid, "status": SWITCH_STATUS.get(fid, -1)})

# ==========================================================
@app.route("/fault")
def fault():
    err = require_build()
    if err: return err
    global FAULT_NODE, FAULT_FEEDER
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    _, i       = TREE.query([lon, lat])
    FAULT_NODE = NODE_LIST[i]
    FAULT_FEEDER = "UNK"
    for u, v, d in G.edges(data=True):
        if u == FAULT_NODE or v == FAULT_NODE:
            FAULT_FEEDER = d.get("feeder", "UNK")
            break
    return jsonify({"node": str(FAULT_NODE), "feeder": FAULT_FEEDER})

# ==========================================================
@app.route("/clear_fault")
def clear_fault():
    global FAULT_NODE, FAULT_FEEDER
    FAULT_NODE = FAULT_FEEDER = None
    return jsonify({"status": "cleared"})

# ==========================================================
@app.route("/api/scada")
def scada():
    err = require_build()
    if err: return err
    try:
        active         = get_active_nodes()
        total          = len(NODE_LIST)
        nodes_on       = len(active)
        nodes_off      = total - nodes_on
        energized_pct  = round(nodes_on / total * 100, 1) if total > 0 else 0
        total_switches = len(SWITCH_STATUS)
        open_switches  = sum(1 for v in SWITCH_STATUS.values() if v == 0)

        feeder_nodes = {}
        for u, v, d in G.edges(data=True):
            fdr = d.get("feeder", "UNK")
            feeder_nodes.setdefault(fdr, set()).update([u, v])

        feeder_status = {}
        for fdr, fnodes in feeder_nodes.items():
            on = len(fnodes & active)
            feeder_status[fdr] = {
                "color": FEEDER_COLOR.get(fdr, "#888"),
                "total": len(fnodes),
                "on":    on,
                "pct":   round(on / len(fnodes) * 100, 1) if fnodes else 0
            }

        return jsonify({
            "fault_feeder":   FAULT_FEEDER,
            "fault_node":     str(FAULT_NODE) if FAULT_NODE else None,
            "open_switches":  open_switches,
            "total_switches": total_switches,
            "nodes_on":       nodes_on,
            "nodes_off":      nodes_off,
            "energized_pct":  energized_pct,
            "feeders":        feeder_status
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

# ==========================================================
@app.route("/api/feeders")
def feeders():
    return jsonify({"colors": FEEDER_COLOR,
                    "feeders": sorted(FEEDER_COLOR.keys())})

# ==========================================================
if __name__ == "__main__":
    build()
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)
