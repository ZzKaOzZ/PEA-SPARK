# =========================================================
# PEA SPARK — apptest.py
# =========================================================

from flask import Flask, jsonify, render_template, request
import networkx as nx
from scipy.spatial import KDTree
import traceback, os
import geopandas as gpd

app = Flask(__name__)

# ── paths ─────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")

# ── global state ──────────────────────────────────────────
G             = nx.Graph()
NODE_LIST     = []
TREE          = None
SWITCH_STATUS = {}   # fid -> 1(close) / 0(open)
SWITCH_NODES  = {}   # fid -> nearest graph node
SWITCH_META   = {}   # fid -> {type, feeder, location}
FAULT_NODE    = None
FAULT_FEEDER  = None
FEEDER_COLORS = {}
SOURCE_NODES  = []

BUILD_OK    = False
BUILD_ERROR = None

# ── GeoDataFrames (cache) ──────────────────────────────────
GDF_CONDUCTOR = None
GDF_DOF       = None
GDF_PSCB      = None   # optional breaker layer

COLOR_POOL = [
    "#00e5ff","#ff1744","#00ff90","#ffd600","#ff9100",
    "#7c4dff","#40c4ff","#69f0ae","#ff5252","#e040fb",
    "#18ffff","#ff4081",
]

# ==========================================================
def read_layer(filename):
    """อ่าน shapefile หรือ GeoJSON แล้ว reproject → WGS84"""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=32647)   # default UTM 47N
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

# ==========================================================
def infer_type(fid, props):
    fid_u = str(fid).upper()
    sub   = str(props.get("SUBTYPE","") or "").upper()
    if "RECLOSER" in sub or fid_u.startswith("R"):
        return "Recloser"
    if "LOADBREAK" in sub or "LBS" in fid_u or "LB" in sub:
        return "Load Break"
    if "TVS" in fid_u or "VS" in sub:
        return "Switch"
    return "Disconnect"

# ==========================================================
def build():
    global G, NODE_LIST, TREE, BUILD_OK, BUILD_ERROR
    global SWITCH_STATUS, SWITCH_NODES, SWITCH_META
    global FEEDER_COLORS, SOURCE_NODES
    global GDF_CONDUCTOR, GDF_DOF, GDF_PSCB

    BUILD_OK    = False
    BUILD_ERROR = None

    try:
        # ── conductor ─────────────────────────────────────
        # รองรับทั้ง .shp และ .json
        for fname in ("psconductor.shp", "psconductor.json", "psconductor.geojson"):
            if os.path.exists(os.path.join(DATA_DIR, fname)):
                GDF_CONDUCTOR = read_layer(fname)
                break
        if GDF_CONDUCTOR is None:
            raise FileNotFoundError("ไม่พบ psconductor.shp / .json")

        G = nx.Graph()
        FEEDER_COLORS = {}

        for _, row in GDF_CONDUCTOR.iterrows():
            geom   = row.geometry
            feeder = str(row.get("FEEDERID") or "UNKNOWN")

            if feeder not in FEEDER_COLORS:
                FEEDER_COLORS[feeder] = COLOR_POOL[len(FEEDER_COLORS) % len(COLOR_POOL)]

            if geom is None:
                continue

            lines = [geom] if geom.geom_type == "LineString" else \
                    list(geom.geoms) if geom.geom_type == "MultiLineString" else []

            for line in lines:
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    G.add_edge(tuple(coords[i]), tuple(coords[i+1]), feeder=feeder)

        if G.number_of_nodes() == 0:
            raise ValueError("psconductor ไม่มี LineString nodes")

        NODE_LIST = list(G.nodes())
        TREE      = KDTree(NODE_LIST)

        # ── DOF / switches ────────────────────────────────
        SWITCH_NODES  = {}
        SWITCH_META   = {}
        SOURCE_NODES  = []

        for fname in ("DOF.shp", "DOF.json", "DOF.geojson"):
            if os.path.exists(os.path.join(DATA_DIR, fname)):
                GDF_DOF = read_layer(fname)
                break

        if GDF_DOF is not None:
            for _, row in GDF_DOF.iterrows():
                geom = row.geometry
                if geom is None or geom.geom_type != "Point":
                    continue

                fid = str(row.get("FACILITYID") or row.get("DEVICEID") or
                          row.get("NAME") or "")
                if not fid:
                    continue

                _, idx  = TREE.query([geom.x, geom.y])
                nearest = NODE_LIST[idx]

                feeder   = str(row.get("FEEDERID") or "UNKNOWN")
                location = str(row.get("LOCATION") or row.get("STREETNAME") or
                               row.get("SUBSTATION") or feeder)
                sw_type  = infer_type(fid, dict(row))

                SWITCH_NODES[fid]  = nearest
                SWITCH_META[fid]   = {"type": sw_type, "feeder": feeder, "location": location}
                if fid not in SWITCH_STATUS:
                    SWITCH_STATUS[fid] = int(row.get("PRESENTPOS", 1) or 1)

        # ── optional PSCB (source breakers) ───────────────
        for fname in ("pscb.shp", "pscb.json", "pscb.geojson"):
            p = os.path.join(DATA_DIR, fname)
            if os.path.exists(p):
                GDF_PSCB = read_layer(fname)
                break

        if GDF_PSCB is not None:
            for _, row in GDF_PSCB.iterrows():
                geom = row.geometry
                if geom and geom.geom_type == "Point":
                    _, idx = TREE.query([geom.x, geom.y])
                    SOURCE_NODES.append(NODE_LIST[idx])

        BUILD_OK = True
        print("=" * 55)
        print("BUILD OK")
        print(f"  nodes    = {len(NODE_LIST)}")
        print(f"  edges    = {G.number_of_edges()}")
        print(f"  switches = {len(SWITCH_NODES)}")
        print(f"  feeders  = {len(FEEDER_COLORS)}")
        print(f"  sources  = {len(SOURCE_NODES)}")
        print("=" * 55)

    except Exception:
        BUILD_ERROR = traceback.format_exc()
        print("[BUILD FAILED]\n" + BUILD_ERROR)

# ==========================================================
def get_energized():
    """BFS จาก SOURCE_NODES ถ้ามี มิฉะนั้นใช้ connected-components"""
    G2 = G.copy()

    if FAULT_NODE and FAULT_NODE in G2:
        G2.remove_node(FAULT_NODE)

    for fid, node in SWITCH_NODES.items():
        if SWITCH_STATUS.get(fid, 1) == 0 and node in G2:
            G2.remove_node(node)

    if SOURCE_NODES:
        energized = set()
        stack     = [n for n in SOURCE_NODES if n in G2]
        while stack:
            n = stack.pop()
            if n in energized:
                continue
            energized.add(n)
            stack.extend(nb for nb in G2.neighbors(n) if nb not in energized)
        return energized
    else:
        # ไม่มี source → ถือว่าทุก node ที่ยังเชื่อมอยู่ = energized
        energized = set()
        for comp in nx.connected_components(G2):
            energized |= comp
        return energized

# ==========================================================
@app.route("/")
def home():
    return render_template("indexpro.html")

# ==========================================================
@app.route("/api/debug")
def api_debug():
    files = os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else []
    return jsonify({
        "build_ok":    BUILD_OK,
        "build_error": BUILD_ERROR,
        "nodes":       len(NODE_LIST),
        "edges":       G.number_of_edges(),
        "switches":    len(SWITCH_NODES),
        "feeders":     list(FEEDER_COLORS.keys()),
        "sources":     len(SOURCE_NODES),
        "data_dir":    DATA_DIR,
        "data_files":  files,
    })

# ==========================================================
@app.route("/api/conductor")
def api_conductor():
    if not BUILD_OK:
        return jsonify({"error": BUILD_ERROR or "build ยังไม่สำเร็จ"}), 503
    try:
        energized = get_energized()
        features  = []

        for _, row in GDF_CONDUCTOR.iterrows():
            geom   = row.geometry
            feeder = str(row.get("FEEDERID") or "UNKNOWN")
            color  = FEEDER_COLORS.get(feeder, "#888888")

            if geom is None:
                continue

            lines = [geom] if geom.geom_type == "LineString" else \
                    list(geom.geoms) if geom.geom_type == "MultiLineString" else []

            for line in lines:
                coords = [[x, y] for x, y in line.coords]
                # ตรวจว่าทุก node บนเส้นนี้ energized ไหม
                status = "on" if all(tuple(c) in energized for c in coords) else "off"
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"feeder": feeder, "color": color, "status": status}
                })

        return jsonify({"type": "FeatureCollection", "features": features})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

# ==========================================================
@app.route("/api/dof")
def api_dof():
    if not BUILD_OK:
        return jsonify({"error": BUILD_ERROR or "build ยังไม่สำเร็จ"}), 503
    try:
        features = []

        if GDF_DOF is not None:
            for _, row in GDF_DOF.iterrows():
                geom = row.geometry
                if geom is None or geom.geom_type != "Point":
                    continue
                fid = str(row.get("FACILITYID") or row.get("DEVICEID") or
                          row.get("NAME") or "")
                if not fid:
                    continue
                meta   = SWITCH_META.get(fid, {})
                status = SWITCH_STATUS.get(fid, 1)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
                    "properties": {
                        "id":       fid,
                        "state":    "CLOSE" if status == 1 else "OPEN",
                        "status":   status,
                        "type":     meta.get("type", "Disconnect"),
                        "feeder":   meta.get("feeder", "UNKNOWN"),
                        "location": meta.get("location", ""),
                    }
                })

        # PSCB breakers
        if GDF_PSCB is not None:
            for _, row in GDF_PSCB.iterrows():
                geom = row.geometry
                if geom is None or geom.geom_type != "Point":
                    continue
                fid    = str(row.get("FACILITYID") or "CB")
                feeder = str(row.get("FEEDERID") or "UNKNOWN")
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
                    "properties": {
                        "id": fid, "state": "CLOSE", "status": 1,
                        "type": "Breaker", "feeder": feeder, "location": "SOURCE"
                    }
                })

        return jsonify({"type": "FeatureCollection", "features": features})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

# ==========================================================
@app.route("/api/scada")
def api_scada():
    if not BUILD_OK:
        return jsonify({"error": BUILD_ERROR or "build ยังไม่สำเร็จ"}), 503
    try:
        energized      = get_energized()
        total          = len(NODE_LIST)
        nodes_on       = len(energized)
        nodes_off      = total - nodes_on
        energized_pct  = round(nodes_on / total * 100, 1) if total else 0
        open_switches  = sum(1 for v in SWITCH_STATUS.values() if v == 0)
        total_switches = len(SWITCH_STATUS)

        feeders = {}
        for feeder, color in FEEDER_COLORS.items():
            fnodes = set()
            for a, b, d in G.edges(data=True):
                if d.get("feeder") == feeder:
                    fnodes.update([a, b])
            fon  = len(fnodes & energized)
            fpct = round(fon / len(fnodes) * 100) if fnodes else 0
            feeders[feeder] = {"color": color, "total": len(fnodes),
                               "on": fon, "pct": fpct}

        return jsonify({
            "fault_feeder":   FAULT_FEEDER,
            "fault_node":     str(FAULT_NODE) if FAULT_NODE else None,
            "open_switches":  open_switches,
            "total_switches": total_switches,
            "nodes_on":       nodes_on,
            "nodes_off":      nodes_off,
            "energized_pct":  energized_pct,
            "feeders":        feeders,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

# ==========================================================
@app.route("/toggle_switch")
def toggle_switch():
    fid = request.args.get("id")
    if fid in SWITCH_STATUS:
        SWITCH_STATUS[fid] = 1 - SWITCH_STATUS[fid]
    return jsonify({"ok": True, "status": SWITCH_STATUS.get(fid, -1)})

# ==========================================================
@app.route("/fault")
def fault():
    global FAULT_NODE, FAULT_FEEDER
    if not BUILD_OK:
        return jsonify({"error": "build ยังไม่สำเร็จ"}), 503
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    _, idx     = TREE.query([lon, lat])
    FAULT_NODE = NODE_LIST[idx]
    FAULT_FEEDER = "UNKNOWN"
    for nb in G.neighbors(FAULT_NODE):
        FAULT_FEEDER = G[FAULT_NODE][nb].get("feeder", "UNKNOWN")
        break
    return jsonify({"ok": True, "feeder": FAULT_FEEDER})

# ==========================================================
@app.route("/clear_fault")
def clear_fault():
    global FAULT_NODE, FAULT_FEEDER
    FAULT_NODE = FAULT_FEEDER = None
    return jsonify({"ok": True})

# ==========================================================
if __name__ == "__main__":
    build()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
