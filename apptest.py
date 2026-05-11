# FULL VERSION — apptest.py (แก้แล้ว)
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
BUILD_LOG     = []

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


def log(msg):
    print(msg)
    BUILD_LOG.append(msg)


def read_layer(filename, default_epsg=32647):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบ: {path}")

    gdf = gpd.read_file(path)

    log(f"read {filename}: {len(gdf)} rows")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=default_epsg)

    epsg = gdf.crs.to_epsg()

    if epsg != 4326:
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


def infer_type_from_layer(layer_name, fid, props):
    ln = layer_name.lower()

    if "recloser" in ln:
        return "Recloser"

    if "trans" in ln:
        return "Transformer"

    fid_u = str(fid).upper()
    sub   = str(props.get("SUBTYPE", "") or "").upper()

    if "RECLOSER" in sub or fid_u.startswith("R"):
        return "Recloser"

    if "LOADBREAK" in sub or "LBS" in fid_u or "LB" in sub:
        return "Load Break"

    if "TVS" in fid_u or "VS" in sub:
        return "Switch"

    return "Disconnect"


def norm_xy(x, y):
    return (round(x, 6), round(y, 6))


def build():
    global G, NODE_LIST, TREE
    global SWITCH_STATUS, SWITCH_NODES, SWITCH_META
    global FEEDER_COLORS, SOURCE_NODES
    global BUILD_OK, BUILD_ERROR
    global GDF_CONDUCTOR, GDF_DOF, GDF_PSCB
    global GDF_TRANS, GDF_RECLOSER

    BUILD_OK = False
    BUILD_ERROR = None

    try:
        log("BUILD START")

        GDF_CONDUCTOR = read_layer("psconductor.shp")

        G = nx.Graph()
        FEEDER_COLORS = {}

        for _, row in GDF_CONDUCTOR.iterrows():
            geom   = row.geometry
            feeder = str(row.get("FEEDERID") or row.get("FEEDER_ID") or "UNKNOWN")

            if feeder not in FEEDER_COLORS:
                FEEDER_COLORS[feeder] = COLOR_POOL[
                    len(FEEDER_COLORS) % len(COLOR_POOL)
                ]

            for line in extract_lines(geom):
                coords = list(line.coords)

                for i in range(len(coords)-1):
                    a = norm_xy(coords[i][0], coords[i][1])
                    b = norm_xy(coords[i+1][0], coords[i+1][1])

                    G.add_edge(a, b, feeder=feeder)

        NODE_LIST = list(G.nodes())
        TREE = KDTree(NODE_LIST)

        SWITCH_NODES = {}
        SWITCH_META  = {}
        SOURCE_NODES = []

        # DOF
        path = os.path.join(DATA_DIR, "DOF.shp")
        if os.path.exists(path):
            GDF_DOF = read_layer("DOF.shp")
            _load_point_layer(GDF_DOF, "DOF")

        # RECLOSER
        path = os.path.join(DATA_DIR, "psrecloser.shp")
        if os.path.exists(path):
            GDF_RECLOSER = read_layer("psrecloser.shp")
            _load_point_layer(GDF_RECLOSER, "psrecloser")

        # TRANSFORMER
        path = os.path.join(DATA_DIR, "pstrans.shp")
        if os.path.exists(path):
            GDF_TRANS = read_layer("pstrans.shp")
            _load_point_layer(GDF_TRANS, "pstrans")

        # PSCB
        path = os.path.join(DATA_DIR, "pscb.shp")
        if os.path.exists(path):
            GDF_PSCB = read_layer("pscb.shp")

            for _, row in GDF_PSCB.iterrows():
                geom = row.geometry

                if geom and geom.geom_type == "Point":
                    _, idx = TREE.query([geom.x, geom.y])

                    nearest = (
                        round(NODE_LIST[idx][0], 6),
                        round(NODE_LIST[idx][1], 6)
                    )

                    SOURCE_NODES.append(nearest)

        BUILD_OK = True

        log(f"nodes={G.number_of_nodes()}")
        log(f"edges={G.number_of_edges()}")
        log(f"switches={len(SWITCH_NODES)}")

    except Exception:
        BUILD_ERROR = traceback.format_exc()
        log(BUILD_ERROR)


def _load_point_layer(gdf, layer_name):
    count = 0

    for _, row in gdf.iterrows():
        geom = row.geometry

        if geom is None:
            continue

        if geom.geom_type != "Point":
            continue

        fid = str(
            row.get("FACILITYID") or
            row.get("DEVICEID") or
            row.get("NAME") or
            row.get("FID") or ""
        )

        if not fid:
            continue

        _, idx = TREE.query([geom.x, geom.y])

        nearest = (
            round(NODE_LIST[idx][0], 6),
            round(NODE_LIST[idx][1], 6)
        )

        feeder = str(row.get("FEEDERID") or row.get("FEEDER_ID") or "")

        if not feeder:
            for _u, _v, _d in G.edges(nearest, data=True):
                feeder = _d.get("feeder", "UNKNOWN")
                break

        location = str(
            row.get("LOCATION") or
            row.get("STREETNAME") or
            row.get("SUBSTATION") or ""
        )

        sw_type = infer_type_from_layer(layer_name, fid, dict(row))

        SWITCH_NODES[fid] = nearest

        SWITCH_META[fid] = {
            "type": sw_type,
            "feeder": feeder,
            "location": location
        }

        if fid not in SWITCH_STATUS:
            SWITCH_STATUS[fid] = int(row.get("PRESENTPOS", 1) or 1)

        count += 1

    log(f"loaded {layer_name}: {count}")


def get_energized():
    G2 = G.copy()

    if FAULT_NODE and FAULT_NODE in G2:
        G2.remove_node(FAULT_NODE)

    for fid, node in SWITCH_NODES.items():
        if SWITCH_STATUS.get(fid, 1) == 0:
            if node in G2:
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


@app.route("/")
def home():
    return render_template("indexpro.html")


@app.route("/api/debug")
def api_debug():
    return jsonify({
        "build_ok": BUILD_OK,
        "build_error": BUILD_ERROR,
        "nodes": len(NODE_LIST),
        "edges": G.number_of_edges(),
        "switches": len(SWITCH_NODES),
        "sources": len(SOURCE_NODES),
    })


@app.route('/api/debug_counts')
def debug_counts():
    return jsonify({
        'conductors': len(GDF_CONDUCTOR) if GDF_CONDUCTOR is not None else 0,
        'dof': len(GDF_DOF) if GDF_DOF is not None else 0,
        'recloser': len(GDF_RECLOSER) if GDF_RECLOSER is not None else 0,
        'transformer': len(GDF_TRANS) if GDF_TRANS is not None else 0,
        'pscb': len(GDF_PSCB) if GDF_PSCB is not None else 0,
        'switch_nodes': len(SWITCH_NODES),
        'graph_nodes': G.number_of_nodes(),
        'graph_edges': G.number_of_edges(),
    })


@app.route("/api/conductor")
def api_conductor():
    energized = get_energized()

    features = []

    for _, row in GDF_CONDUCTOR.iterrows():
        geom   = row.geometry
        feeder = str(row.get("FEEDERID") or row.get("FEEDER_ID") or "UNKNOWN")

        color = FEEDER_COLORS.get(feeder, "#888")

        for line in extract_lines(geom):
            coords = [[c[0], c[1]] for c in line.coords]

            edge_on = False

            for i in range(len(coords)-1):
                a = norm_xy(coords[i][0], coords[i][1])
                b = norm_xy(coords[i+1][0], coords[i+1][1])

                if a in energized or b in energized:
                    edge_on = True
                    break

            status = "on" if edge_on else "off"

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "feeder": feeder,
                    "color": color,
                    "status": status
                }
            })

    return jsonify({
        "type": "FeatureCollection",
        "features": features
    })


@app.route("/api/dof")
def api_dof():
    features = []
    seen = set()

    def add_layer(gdf, layer_name):
        if gdf is None:
            return

        for _, row in gdf.iterrows():
            geom = row.geometry

            if geom is None:
                continue

            if geom.geom_type != "Point":
                continue

            fid = str(
                row.get("FACILITYID") or
                row.get("DEVICEID") or
                row.get("NAME") or
                row.get("FID") or ""
            )

            if not fid:
                continue

            if fid in seen:
                continue

            seen.add(fid)

            meta = SWITCH_META.get(fid, {})
            status = SWITCH_STATUS.get(fid, 1)

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [geom.x, geom.y]
                },
                "properties": {
                    "id": fid,
                    "state": "CLOSE" if status == 1 else "OPEN",
                    "status": status,
                    "type": meta.get("type", "Disconnect"),
                    "feeder": meta.get("feeder", "UNKNOWN"),
                    "location": meta.get("location", "")
                }
            })

    add_layer(GDF_DOF, "DOF")
    add_layer(GDF_RECLOSER, "psrecloser")
    add_layer(GDF_TRANS, "pstrans")

    return jsonify({
        "type": "FeatureCollection",
        "features": features
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

    _, idx = TREE.query([lon, lat])

    FAULT_NODE = NODE_LIST[idx]

    FAULT_FEEDER = "UNKNOWN"

    for nb in G.neighbors(FAULT_NODE):
        FAULT_FEEDER = G[FAULT_NODE][nb].get("feeder", "UNKNOWN")
        break

    return jsonify({
        "ok": True,
        "feeder": FAULT_FEEDER
    })


@app.route("/clear_fault")
def clear_fault():
    global FAULT_NODE, FAULT_FEEDER

    FAULT_NODE = None
    FAULT_FEEDER = None

    return jsonify({"ok": True})


if __name__ == "__main__":
    build()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
