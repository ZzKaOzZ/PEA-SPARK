# PEA SPARK — apptest.py (แก้: ประเภทจาก FACILITYID ของ DOF, สถานะจาก PRESENTPOS, ฟีดเดอร์จาก psconductor FEEDERID)
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
FEEDER_SEGS   = {}   # นับจำนวน segments ต่อ feeder
SOURCE_NODES  = []
BUILD_OK      = False
BUILD_ERROR   = None
BUILD_LOG     = []

GDF_CONDUCTOR = None
GDF_DOF       = None
GDF_PSCB      = None
GDF_TRANS     = None
GDF_RECLOSER  = None

COLOR_POOL = [
    "#00e5ff","#ff1744","#00ff90","#ffd600","#ff9100",
    "#7c4dff","#40c4ff","#69f0ae","#ff5252","#e040fb",
    "#18ffff","#ff4081","#00bfa5","#ff6d00","#d500f9",
]


def log(msg):
    print(msg)
    BUILD_LOG.append(msg)


def read_layer(filename, default_epsg=32647):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบ: {path}")
    gdf = gpd.read_file(path)
    log(f"read {filename}: {len(gdf)} rows  cols={list(gdf.columns)}")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=default_epsg)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def extract_lines(geom):
    if geom is None:
        return []
    t = geom.geom_type
    if t in ("LineString", "LineStringZ"):
        return [geom]
    if t in ("MultiLineString", "MultiLineStringZ"):
        return list(geom.geoms)
    if t == "GeometryCollection":
        result = []
        for g in geom.geoms:
            result += extract_lines(g)
        return result
    return []


# ── ฟังก์ชันหลัก: อ่านประเภทจาก FACILITYID ──────────────────
# รูปแบบ FACILITYID ของ DOF:
#   PDA04S-14   → Disconnect  (ลงท้าย S-xx ไม่มี TVS/LB/RCL)
#   PDA08S-13   → Load Break  (ดูจาก context หรือ SUBTYPE=Load Break)
#   PDA09TVS-01 → Switch      (มี TVS ใน fid)
#   R... หรือ psrecloser layer → Recloser
#   pstrans layer → Transformer
# ─────────────────────────────────────────────────────────────
def infer_type_from_facilityid(layer_name, fid, props):
    """
    อ้างอิงประเภทสวิทช์จาก FACILITYID ของไฟล์ DOF เป็นหลัก
    พร้อม fallback จาก layer name และ SUBTYPE
    """
    ln    = layer_name.lower()
    fid_u = str(fid).upper().strip()

    # layer-level override
    if "recloser" in ln:
        return "Recloser"
    if "trans" in ln:
        return "Transformer"

    # FACILITYID pattern matching (DOF)
    # TVS = Tie-line Vacuum Switch
    if "TVS" in fid_u:
        return "Switch"

    # RCL / RC_ / starts with R → Recloser
    if "RCL" in fid_u or fid_u.startswith("RC"):
        return "Recloser"

    # LB / LBS → Load Break Switch
    if "LBS" in fid_u or fid_u.endswith("LB"):
        return "Load Break"

    # SUBTYPE column fallback
    sub = str(props.get("SUBTYPE", "") or "").upper()
    if "RECLOSER" in sub:
        return "Recloser"
    if "LOAD" in sub and "BREAK" in sub:
        return "Load Break"
    if "TVS" in sub or "VACUUM" in sub:
        return "Switch"

    # default
    return "Disconnect"


def norm_xy(x, y):
    return (round(x, 6), round(y, 6))


def build():
    global G, NODE_LIST, TREE
    global SWITCH_STATUS, SWITCH_NODES, SWITCH_META
    global FEEDER_COLORS, FEEDER_SEGS, SOURCE_NODES
    global BUILD_OK, BUILD_ERROR
    global GDF_CONDUCTOR, GDF_DOF, GDF_PSCB, GDF_TRANS, GDF_RECLOSER

    BUILD_OK = False
    BUILD_ERROR = None

    try:
        log("BUILD START")

        # ── 1. โหลด psconductor → สร้าง graph + FEEDER_COLORS + FEEDER_SEGS ──
        GDF_CONDUCTOR = read_layer("psconductor.shp")

        G = nx.Graph()
        FEEDER_COLORS = {}
        FEEDER_SEGS   = {}

        for _, row in GDF_CONDUCTOR.iterrows():
            geom   = row.geometry
            # อ้างอิง feeder จาก FEEDERID ของ psconductor
            feeder = str(row.get("FEEDERID") or row.get("FEEDER_ID") or "UNKNOWN").strip()

            if feeder not in FEEDER_COLORS:
                FEEDER_COLORS[feeder] = COLOR_POOL[len(FEEDER_COLORS) % len(COLOR_POOL)]
                FEEDER_SEGS[feeder]   = 0

            for line in extract_lines(geom):
                coords = list(line.coords)
                FEEDER_SEGS[feeder] += len(coords) - 1
                for i in range(len(coords) - 1):
                    a = norm_xy(coords[i][0],   coords[i][1])
                    b = norm_xy(coords[i+1][0], coords[i+1][1])
                    G.add_edge(a, b, feeder=feeder)

        NODE_LIST = list(G.nodes())
        TREE      = KDTree(NODE_LIST)

        SWITCH_NODES = {}
        SWITCH_META  = {}
        SOURCE_NODES = []

        # ── 2. DOF → สวิทช์หลัก (FACILITYID, PRESENTPOS) ──────────
        path = os.path.join(DATA_DIR, "DOF.shp")
        if os.path.exists(path):
            GDF_DOF = read_layer("DOF.shp")
            _load_dof_layer(GDF_DOF)

        # ── 3. psrecloser ──────────────────────────────────────────
        path = os.path.join(DATA_DIR, "psrecloser.shp")
        if os.path.exists(path):
            GDF_RECLOSER = read_layer("psrecloser.shp")
            _load_point_layer(GDF_RECLOSER, "psrecloser")

        # ── 4. pstrans ─────────────────────────────────────────────
        path = os.path.join(DATA_DIR, "pstrans.shp")
        if os.path.exists(path):
            GDF_TRANS = read_layer("pstrans.shp")
            _load_point_layer(GDF_TRANS, "pstrans")

        # ── 5. pscb → source nodes (substation) ───────────────────
        path = os.path.join(DATA_DIR, "pscb.shp")
        if os.path.exists(path):
            GDF_PSCB = read_layer("pscb.shp")
            for _, row in GDF_PSCB.iterrows():
                geom = row.geometry
                if geom and geom.geom_type == "Point":
                    _, idx = TREE.query([geom.x, geom.y])
                    nearest = (round(NODE_LIST[idx][0], 6), round(NODE_LIST[idx][1], 6))
                    SOURCE_NODES.append(nearest)

        BUILD_OK = True
        log(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} switches={len(SWITCH_NODES)}")

    except Exception:
        BUILD_ERROR = traceback.format_exc()
        log(BUILD_ERROR)


def _load_dof_layer(gdf):
    """
    โหลด DOF.shp โดยอ้างอิง:
      - FACILITYID  → ชื่อ/ID ของสวิทช์ + infer type
      - PRESENTPOS  → สถานะ (1=CLOSE, 0=OPEN)
      - FEEDERID ของ psconductor (ผ่าน graph) → feeder
    """
    count = 0
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type != "Point":
            continue

        fid = str(row.get("FACILITYID") or row.get("DEVICEID") or row.get("NAME") or row.get("FID") or "").strip()
        if not fid:
            continue

        _, idx   = TREE.query([geom.x, geom.y])
        nearest  = (round(NODE_LIST[idx][0], 6), round(NODE_LIST[idx][1], 6))

        # feeder จาก graph edge ที่ใกล้ที่สุด (psconductor FEEDERID)
        feeder = ""
        for _u, _v, _d in G.edges(nearest, data=True):
            feeder = _d.get("feeder", "UNKNOWN")
            break
        if not feeder:
            feeder = "UNKNOWN"

        location = str(row.get("LOCATION") or row.get("STREETNAME") or row.get("SUBSTATION") or "").strip()

        # ── ประเภทสวิทช์จาก FACILITYID ──
        sw_type = infer_type_from_facilityid("DOF", fid, dict(row))

        SWITCH_NODES[fid] = nearest
        SWITCH_META[fid]  = {
            "type":     sw_type,
            "feeder":   feeder,
            "location": location,
        }

        # ── สถานะจาก PRESENTPOS ──
        # PRESENTPOS: 1 = CLOSE (ปิด/จ่ายไฟ), 0 = OPEN (เปิด/ตัดวงจร)
        if fid not in SWITCH_STATUS:
            try:
                pos = int(row.get("PRESENTPOS") or 1)
            except (ValueError, TypeError):
                pos = 1
            SWITCH_STATUS[fid] = pos

        count += 1
    log(f"loaded DOF: {count}")


def _load_point_layer(gdf, layer_name):
    """โหลด psrecloser / pstrans (ไม่มี PRESENTPOS — ถือว่า CLOSE เสมอ)"""
    count = 0
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type != "Point":
            continue

        fid = str(row.get("FACILITYID") or row.get("DEVICEID") or row.get("NAME") or row.get("FID") or "").strip()
        if not fid or fid in SWITCH_NODES:
            continue

        _, idx  = TREE.query([geom.x, geom.y])
        nearest = (round(NODE_LIST[idx][0], 6), round(NODE_LIST[idx][1], 6))

        feeder = ""
        for _u, _v, _d in G.edges(nearest, data=True):
            feeder = _d.get("feeder", "UNKNOWN")
            break
        if not feeder:
            feeder = "UNKNOWN"

        location = str(row.get("LOCATION") or row.get("STREETNAME") or row.get("SUBSTATION") or "").strip()
        sw_type  = infer_type_from_facilityid(layer_name, fid, dict(row))

        SWITCH_NODES[fid] = nearest
        SWITCH_META[fid]  = {"type": sw_type, "feeder": feeder, "location": location}

        if fid not in SWITCH_STATUS:
            try:
                pos = int(row.get("PRESENTPOS") or 1)
            except (ValueError, TypeError):
                pos = 1
            SWITCH_STATUS[fid] = pos

        count += 1
    log(f"loaded {layer_name}: {count}")


# ── Energization ────────────────────────────────────────────
def get_energized():
    G2 = G.copy()
    if FAULT_NODE and FAULT_NODE in G2:
        G2.remove_node(FAULT_NODE)
    for fid, node in SWITCH_NODES.items():
        if SWITCH_STATUS.get(fid, 1) == 0 and node in G2:
            G2.remove_node(node)
    energized = set()
    if SOURCE_NODES:
        stack = [n for n in SOURCE_NODES if n in G2]
        while stack:
            n = stack.pop()
            if n in energized:
                continue
            energized.add(n)
            for nb in G2.neighbors(n):
                if nb not in energized:
                    stack.append(nb)
    else:
        for comp in nx.connected_components(G2):
            energized |= comp
    return energized


# ── Routes ──────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("indexpro.html")


@app.route("/api/debug")
def api_debug():
    return jsonify({
        "build_ok":    BUILD_OK,
        "build_error": BUILD_ERROR,
        "nodes":       len(NODE_LIST),
        "edges":       G.number_of_edges(),
        "switches":    len(SWITCH_NODES),
        "sources":     len(SOURCE_NODES),
    })


@app.route("/api/debug_counts")
def debug_counts():
    return jsonify({
        "conductors":   len(GDF_CONDUCTOR) if GDF_CONDUCTOR is not None else 0,
        "dof":          len(GDF_DOF)       if GDF_DOF       is not None else 0,
        "recloser":     len(GDF_RECLOSER)  if GDF_RECLOSER  is not None else 0,
        "transformer":  len(GDF_TRANS)     if GDF_TRANS     is not None else 0,
        "pscb":         len(GDF_PSCB)      if GDF_PSCB      is not None else 0,
        "switch_nodes": len(SWITCH_NODES),
        "graph_nodes":  G.number_of_nodes(),
        "graph_edges":  G.number_of_edges(),
    })


@app.route("/api/conductor")
def api_conductor():
    energized = get_energized()
    features  = []
    for _, row in GDF_CONDUCTOR.iterrows():
        geom   = row.geometry
        feeder = str(row.get("FEEDERID") or row.get("FEEDER_ID") or "UNKNOWN").strip()
        color  = FEEDER_COLORS.get(feeder, "#888")
        for line in extract_lines(geom):
            coords   = [[c[0], c[1]] for c in line.coords]
            edge_on  = False
            for i in range(len(coords) - 1):
                a = norm_xy(coords[i][0],   coords[i][1])
                b = norm_xy(coords[i+1][0], coords[i+1][1])
                if a in energized or b in energized:
                    edge_on = True
                    break
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "feeder": feeder,
                    "color":  color,
                    "status": "on" if edge_on else "off",
                }
            })
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/dof")
def api_dof():
    """
    ใช้ SWITCH_NODES เป็น source of truth ของพิกัด
    (guaranteed EPSG:4326 และ snap กับ graph แล้ว)
    """
    features = []
    for fid, node in SWITCH_NODES.items():
        lon, lat = node  # node = (lon, lat) in 4326
        meta   = SWITCH_META.get(fid, {})
        status = SWITCH_STATUS.get(fid, 1)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "id":       fid,
                "state":    "CLOSE" if status == 1 else "OPEN",
                "status":   status,
                "type":     meta.get("type",     "Disconnect"),
                "feeder":   meta.get("feeder",   "UNKNOWN"),
                "location": meta.get("location", ""),
            }
        })
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/dof_debug")
def api_dof_debug():
    """Debug: ตรวจสอบว่า switch nodes มีข้อมูลถูกต้องไหม"""
    sample = []
    for fid, node in list(SWITCH_NODES.items())[:5]:
        meta = SWITCH_META.get(fid, {})
        sample.append({
            "fid": fid,
            "node": node,
            "type": meta.get("type"),
            "feeder": meta.get("feeder"),
            "status": SWITCH_STATUS.get(fid),
        })
    return jsonify({
        "total_switches": len(SWITCH_NODES),
        "sample": sample,
    })


@app.route("/toggle_switch")
def toggle_switch():
    fid = request.args.get("id")
    if fid in SWITCH_STATUS:
        SWITCH_STATUS[fid] = 1 - SWITCH_STATUS[fid]
    return jsonify({"ok": True})


@app.route("/fault")
def fault():
    global FAULT_NODE, FAULT_FEEDER
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    _, idx       = TREE.query([lon, lat])
    FAULT_NODE   = NODE_LIST[idx]
    FAULT_FEEDER = "UNKNOWN"
    for nb in G.neighbors(FAULT_NODE):
        FAULT_FEEDER = G[FAULT_NODE][nb].get("feeder", "UNKNOWN")
        break
    return jsonify({"ok": True, "feeder": FAULT_FEEDER})


@app.route("/clear_fault")
def clear_fault():
    global FAULT_NODE, FAULT_FEEDER
    FAULT_NODE   = None
    FAULT_FEEDER = None
    return jsonify({"ok": True})


@app.route("/api/scada")
def api_scada():
    energized    = get_energized()
    total_nodes  = len(NODE_LIST)
    nodes_on     = len(energized)
    nodes_off    = total_nodes - nodes_on
    pct          = round((nodes_on / total_nodes) * 100, 1) if total_nodes else 0
    open_switches = sum(1 for v in SWITCH_STATUS.values() if v == 0)

    feeders = {}
    for feeder, color in FEEDER_COLORS.items():
        total = 0
        on    = 0
        for u, v, d in G.edges(data=True):
            if d.get("feeder") != feeder:
                continue
            total += 1
            if u in energized or v in energized:
                on += 1
        fpct = round((on / total) * 100) if total else 0
        feeders[feeder] = {
            "color": color,
            "pct":   fpct,
            "segs":  FEEDER_SEGS.get(feeder, total),   # ← จำนวน segments
        }

    return jsonify({
        "nodes_on":       nodes_on,
        "nodes_off":      nodes_off,
        "energized_pct":  pct,
        "open_switches":  open_switches,
        "total_switches": len(SWITCH_STATUS),
        "fault_feeder":   FAULT_FEEDER,
        "feeders":        feeders,
    })


if __name__ == "__main__":
    build()
    app.run(host="0.0.0.0", port=5000, debug=False)
