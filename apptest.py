# =========================================================
# PEA SPARK — apptest.py
# =========================================================

from flask import Flask, jsonify, render_template, request
import networkx as nx
from scipy.spatial import KDTree
import traceback, os
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString

app = Flask(__name__)

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")

# ── global state ───────────────────────────────────────────
G             = nx.Graph()
NODE_LIST     = []
TREE          = None
SWITCH_STATUS = {}
SWITCH_NODES  = {}
SWITCH_META   = {}
FAULT_NODE    = None
FAULT_FEEDER  = None
FEEDER_COLORS = {}
SOURCE_NODES  = []
BUILD_OK      = False
BUILD_ERROR   = None
BUILD_LOG     = []   # เก็บ log ละเอียดสำหรับ debug

GDF_CONDUCTOR = None
GDF_DOF       = None
GDF_PSCB      = None
GDF_TRANS     = None
GDF_RECLOSER  = None

COLOR_POOL = [
    "#00e5ff","#ff1744","#00ff90","#ffd600","#ff9100",
    "#7c4dff","#40c4ff","#69f0ae","#ff5252","#e040fb",
    "#18ffff","#ff4081",
]

# ==========================================================
def log(msg):
    print(msg)
    BUILD_LOG.append(msg)

# ==========================================================
def read_layer(filename, default_epsg=32647):
    """อ่าน shapefile/GeoJSON → WGS84 พร้อม fallback CRS"""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบ: {path}")
    gdf = gpd.read_file(path)
    log(f"  read {filename}: {len(gdf)} rows, crs={gdf.crs}, geom_types={gdf.geom_type.unique().tolist()}")
    if gdf.crs is None:
        log(f"  CRS ไม่มี → set epsg:{default_epsg}")
        gdf = gdf.set_crs(epsg=default_epsg)
    epsg = gdf.crs.to_epsg()
    if epsg != 4326:
        log(f"  reproject {epsg} → 4326")
        gdf = gdf.to_crs(epsg=4326)
    return gdf

# ==========================================================
def extract_lines(geom):
    """รองรับ LineString, MultiLineString และ *Z variants"""
    if geom is None:
        return []
    t = geom.geom_type
    if t in ("LineString", "LineStringZ"):
        return [geom]
    if t in ("MultiLineString", "MultiLineStringZ"):
        return list(geom.geoms)
    # GeometryCollection อาจมี line ผสมอยู่
    if t == "GeometryCollection":
        result = []
        for g in geom.geoms:
            result += extract_lines(g)
        return result
    return []

# ==========================================================
def infer_type_from_layer(layer_name, fid, props):
    """แยกประเภทจากชื่อ layer ก่อน แล้วค่อย fallback จาก fid/props"""
    ln = layer_name.lower()
    if "recloser" in ln:
        return "Recloser"
    if "trans" in ln:
        return "Transformer"
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
    global G, NODE_LIST, TREE, BUILD_OK, BUILD_ERROR, BUILD_LOG
    global SWITCH_STATUS, SWITCH_NODES, SWITCH_META
    global FEEDER_COLORS, SOURCE_NODES
    global GDF_CONDUCTOR, GDF_DOF, GDF_PSCB, GDF_TRANS, GDF_RECLOSER

    BUILD_OK    = False
    BUILD_ERROR = None
    BUILD_LOG   = []

    try:
        log("=" * 55)
        log("BUILD START")
        log(f"DATA_DIR = {DATA_DIR}")
        log(f"files    = {os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else 'DIR NOT FOUND'}")
        log("=" * 55)

        # ── conductor ──────────────────────────────────────
        GDF_CONDUCTOR = read_layer("psconductor.shp")

        G = nx.Graph()
        FEEDER_COLORS = {}
        skip_geom = 0

        for _, row in GDF_CONDUCTOR.iterrows():
            geom   = row.geometry
            feeder = str(row.get("FEEDERID") or row.get("FEEDER_ID") or "UNKNOWN")

            if feeder not in FEEDER_COLORS:
                FEEDER_COLORS[feeder] = COLOR_POOL[len(FEEDER_COLORS) % len(COLOR_POOL)]

            lines = extract_lines(geom)
            if not lines:
                skip_geom += 1
                continue

            for line in lines:
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    # ตัด Z coordinate ออก (ใช้แค่ X, Y)
                    a = (coords[i][0],   coords[i][1])
                    b = (coords[i+1][0], coords[i+1][1])
                    G.add_edge(a, b, feeder=feeder)

        log(f"  conductor: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, skip={skip_geom}")

        if G.number_of_nodes() == 0:
            # พยายาม inspect geometry จริง
            sample = GDF_CONDUCTOR.geometry.dropna().head(3).tolist()
            log(f"  WARN: nodes=0, sample geom types: {[g.geom_type for g in sample]}")
            raise ValueError(
                f"psconductor มี nodes=0 — geom types: {GDF_CONDUCTOR.geom_type.unique().tolist()}"
            )

        NODE_LIST = list(G.nodes())
        TREE      = KDTree(NODE_LIST)

        # ── DOF / switches ─────────────────────────────────
        SWITCH_NODES  = {}
        SWITCH_META   = {}
        SOURCE_NODES  = []
        GDF_DOF       = None

        # ── DOF (optional) ─────────────────────────────────
        _shp = os.path.join(DATA_DIR, "DOF.shp")
        if os.path.exists(_shp):
            GDF_DOF = read_layer("DOF.shp")
            _load_point_layer(GDF_DOF, "DOF")

        # ── psrecloser (optional) ──────────────────────────
        _shp = os.path.join(DATA_DIR, "psrecloser.shp")
        if os.path.exists(_shp):
            GDF_RECLOSER = read_layer("psrecloser.shp")
            _load_point_layer(GDF_RECLOSER, "psrecloser")

        # ── pstrans (optional) ─────────────────────────────
        _shp = os.path.join(DATA_DIR, "pstrans.shp")
        if os.path.exists(_shp):
            GDF_TRANS = read_layer("pstrans.shp")
            _load_point_layer(GDF_TRANS, "pstrans")

        # ── pscb / source breakers (ถ้ามี) ─────────────────
        # ── pscb (optional) ────────────────────────────────
        GDF_PSCB = None
        _shp = os.path.join(DATA_DIR, "pscb.shp")
        if os.path.exists(_shp):
            GDF_PSCB = read_layer("pscb.shp")
            for _, row in GDF_PSCB.iterrows():
                geom = row.geometry
                if geom and geom.geom_type == "Point":
                    _, idx = TREE.query([geom.x, geom.y])
                    SOURCE_NODES.append(NODE_LIST[idx])
            log(f"  pscb sources: {len(SOURCE_NODES)}")

        BUILD_OK = True
        log("=" * 55)
        log("BUILD OK")
        log(f"  nodes    = {len(NODE_LIST)}")
        log(f"  edges    = {G.number_of_edges()}")
        log(f"  switches = {len(SWITCH_NODES)}")
        log(f"  feeders  = {len(FEEDER_COLORS)}")
        log(f"  sources  = {len(SOURCE_NODES)}")
        log("=" * 55)

    except Exception:
        BUILD_ERROR = traceback.format_exc()
        log("[BUILD FAILED]\n" + BUILD_ERROR)

# ==========================================================
def _load_point_layer(gdf, layer_name):
    """โหลด point layer เข้า SWITCH_NODES/META"""
    count = 0
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type != "Point":
            continue

        fid = str(
            row.get("FACILITYID") or row.get("DEVICEID") or
            row.get("NAME") or row.get("FID") or ""
        )
        if not fid:
            continue

        _, idx  = TREE.query([geom.x, geom.y])
        nearest = NODE_LIST[idx]

        feeder = str(row.get("FEEDERID") or row.get("FEEDER_ID") or "")
        # ถ้า feeder ว่าง → หาจาก edge ที่ใกล้ที่สุดใน graph
        if not feeder or feeder == "None":
            for _u, _v, _d in G.edges(nearest, data=True):
                feeder = _d.get("feeder", "UNKNOWN")
                break
        if not feeder:
            feeder = "UNKNOWN"

        location = str(row.get("LOCATION") or row.get("STREETNAME") or
                       row.get("SUBSTATION") or "")
        sw_type  = infer_type_from_layer(layer_name, fid, dict(row))

        SWITCH_NODES[fid] = nearest
        SWITCH_META[fid]  = {"type": sw_type, "feeder": feeder, "location": location}
        if fid not in SWITCH_STATUS:
            SWITCH_STATUS[fid] = int(row.get("PRESENTPOS", 1) or 1)
        count += 1

    log(f"  {layer_name}: loaded {count} points")

# ==========================================================
def get_energized():
    G2 = G.copy()
    if FAULT_NODE and FAULT_NODE in G2:
        G2.remove_node(FAULT_NODE)
    for fid, node in SWITCH_NODES.items():
        if SWITCH_STATUS.get(fid, 1) == 0 and node in G2:
            G2.remove_node(node)

    if SOURCE_NODES:
        energized = set()
        stack = [n for n in SOURCE_NODES if n in G2]
        while stack:
            n = stack.pop()
            if n in energized:
                continue
            energized.add(n)
            stack.extend(nb for nb in G2.neighbors(n) if nb not in energized)
        return energized
    else:
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
    files = sorted(os.listdir(DATA_DIR)) if os.path.isdir(DATA_DIR) else []
    return jsonify({
        "build_ok":    BUILD_OK,
        "build_error": BUILD_ERROR,
        "build_log":   BUILD_LOG,
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
        return jsonify({"error": BUILD_ERROR or "build ยังไม่สำเร็จ",
                        "log": BUILD_LOG}), 503
    try:
        energized = get_energized()
        features  = []

        for _, row in GDF_CONDUCTOR.iterrows():
            geom   = row.geometry
            feeder = str(row.get("FEEDERID") or row.get("FEEDER_ID") or "UNKNOWN")
            color  = FEEDER_COLORS.get(feeder, "#888888")

            for line in extract_lines(geom):
                coords = [[c[0], c[1]] for c in line.coords]
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
        seen     = set()

        def add_point_layer(gdf, layer_name):
            if gdf is None:
                return
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.geom_type != "Point":
                    continue
                fid = str(row.get("FACILITYID") or row.get("DEVICEID") or
                          row.get("NAME") or row.get("FID") or "")
                if not fid or fid in seen:
                    continue
                seen.add(fid)
                meta   = SWITCH_META.get(fid, {})
                status = SWITCH_STATUS.get(fid, 1)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
                    "properties": {
                        "id":       fid,
                        "state":    "CLOSE" if status == 1 else "OPEN",
                        "status":   status,
                        "type":     meta.get("type", infer_type_from_layer(layer_name, fid, dict(row))),
                        "feeder":   meta.get("feeder", str(row.get("FEEDERID") or "UNKNOWN")),
                        "location": meta.get("location", ""),
                    }
                })

        add_point_layer(GDF_DOF,      "DOF")
        add_point_layer(GDF_RECLOSER, "psrecloser")
        add_point_layer(GDF_TRANS,    "pstrans")

        # PSCB breakers
        if GDF_PSCB is not None:
            for _, row in GDF_PSCB.iterrows():
                geom = row.geometry
                if geom is None or geom.geom_type != "Point":
                    continue
                fid = str(row.get("FACILITYID") or "CB")
                if fid in seen:
                    continue
                seen.add(fid)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
                    "properties": {
                        "id": fid, "state": "CLOSE", "status": 1,
                        "type": "Breaker",
                        "feeder": str(row.get("FEEDERID") or "UNKNOWN"),
                        "location": "SOURCE"
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
            feeders[feeder] = {"color": color, "total": len(fnodes), "on": fon, "pct": fpct}

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
    _, idx       = TREE.query([lon, lat])
    FAULT_NODE   = NODE_LIST[idx]
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
