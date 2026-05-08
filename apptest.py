from flask import Flask, jsonify, render_template, request
import os
import json
import traceback
import networkx as nx
from scipy.spatial import KDTree

app = Flask(__name__)

# ── global state ──────────────────────────────────────────
G = None
NODE_LIST = []
TREE = None

SWITCH_NODES = {}
SWITCH_STATUS = {}
SWITCH_TYPE = {}
FEEDER_COLOR = {}

FAULT_NODE = None
FAULT_FEEDER = None

# ── cache ─────────────────────────────────────────────────
_CONDUCTOR_FEATURES = []
_DOF_FEATURES = []

BUILD_ERROR = None

# ==========================================================
def load_json(path):

    abs_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        path
    )

    print(f"[load_json] {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"ไม่พบไฟล์: {abs_path}")

    with open(abs_path, encoding="utf-8") as f:
        return json.load(f)

# ==========================================================
def infer_switch_type(fid, props):

    subtype = str(props.get("SUBTYPE", "")).upper()
    name = fid.upper()

    if "RECLOSER" in subtype or "REC" in name:
        return "Recloser"

    if "LOADBREAK" in subtype or "LB" in subtype:
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

        SWITCH_NODES.clear()
        SWITCH_STATUS.clear()
        SWITCH_TYPE.clear()
        FEEDER_COLOR.clear()

        # ==================================================
        # conductor
        # ==================================================

        conductor_data = load_json("data/psconductor.json")
        _CONDUCTOR_FEATURES = conductor_data["features"]

        G = nx.Graph()
        nodes = []

        for f in _CONDUCTOR_FEATURES:

            geom = f.get("geometry", {})
            props = f.get("properties", {})

            feeder = str(props.get("FEEDERID", "UNK"))

            gtype = geom.get("type")
            coords = geom.get("coordinates", [])

            lines = []

            if gtype == "LineString":
                lines = [coords]

            elif gtype == "MultiLineString":
                lines = coords

            else:
                continue

            for line in lines:

                if len(line) < 2:
                    continue

                for i in range(len(line) - 1):

                    a = tuple(line[i])
                    b = tuple(line[i + 1])

                    G.add_edge(a, b, feeder=feeder)

                    nodes += [a, b]

        if not nodes:
            raise ValueError("psconductor.json ไม่มีข้อมูลสาย")

        NODE_LIST = list(set(nodes))
        TREE = KDTree(NODE_LIST)

        # ==================================================
        # switch
        # ==================================================

        dof_data = load_json("data/DOF.json")
        _DOF_FEATURES = dof_data["features"]

        for f in _DOF_FEATURES:

            props = f.get("properties", {})
            geom = f.get("geometry", {})

            fid = str(props.get("FACILITYID", ""))

            if "S" not in fid.upper():
                continue

            if geom.get("type") != "Point":
                continue

            coords = geom.get("coordinates", [])

            if len(coords) < 2:
                continue

            pos = int(props.get("PRESENTPOS", 1))

            lon, lat = coords

            _, idx = TREE.query([lon, lat])

            SWITCH_NODES[fid] = NODE_LIST[idx]
            SWITCH_STATUS[fid] = pos
            SWITCH_TYPE[fid] = infer_switch_type(fid, props)

        # ==================================================
        # feeder colors
        # ==================================================

        palette = [
            "#00e5ff",
            "#7c4dff",
            "#ff9100",
            "#00e676",
            "#ff5252",
            "#ffd600",
            "#e040fb",
            "#40c4ff"
        ]

        feeders = sorted(
            set(nx.get_edge_attributes(G, "feeder").values())
        )

        for i, feeder in enumerate(feeders):
            FEEDER_COLOR[feeder] = palette[i % len(palette)]

        print(
            f"[build] OK "
            f"nodes={len(NODE_LIST)} "
            f"edges={G.number_of_edges()} "
            f"switches={len(SWITCH_STATUS)}"
        )

    except Exception:

        BUILD_ERROR = traceback.format_exc()
        print(BUILD_ERROR)

# ==========================================================
def get_active_nodes():

    G2 = G.copy()

    if FAULT_NODE and FAULT_NODE in G2:
        G2.remove_node(FAULT_NODE)

    for fid, node in SWITCH_NODES.items():

        if SWITCH_STATUS.get(fid, 1) == 0:

            if node in G2:
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
@app.route("/api/conductor")
def conductor():

    try:

        active = get_active_nodes()

        feats = []

        for f in _CONDUCTOR_FEATURES:

            geom = f.get("geometry", {})
            props = f.get("properties", {})

            feeder = str(props.get("FEEDERID", "UNK"))

            gtype = geom.get("type")
            coords = geom.get("coordinates", [])

            lines = []

            if gtype == "LineString":
                lines = [coords]

            elif gtype == "MultiLineString":
                lines = coords

            else:
                continue

            status = "on"

            for line in lines:

                if any(tuple(c) not in active for c in line):
                    status = "off"
                    break

            feats.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "feeder": feeder,
                    "status": status,
                    "color": FEEDER_COLOR.get(feeder, "#888")
                }
            })

        return jsonify({
            "type": "FeatureCollection",
            "features": feats
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()})

# ==========================================================
@app.route("/api/dof")
def dof():

    feats = []

    for f in _DOF_FEATURES:

        props = f.get("properties", {})
        geom = f.get("geometry", {})

        fid = str(props.get("FACILITYID", ""))

        if "S" not in fid.upper():
            continue

        feats.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "id": fid,
                "status": SWITCH_STATUS.get(fid, 1),
                "type": SWITCH_TYPE.get(fid, "Disconnect"),
                "feeder": props.get("FEEDERID", "UNK"),
                "location": props.get("LOCATION", "")
            }
        })

    return jsonify({
        "type": "FeatureCollection",
        "features": feats
    })

# ==========================================================
@app.route("/toggle_switch")
def toggle():

    fid = request.args.get("id")

    if fid in SWITCH_STATUS:
        SWITCH_STATUS[fid] = 1 - SWITCH_STATUS[fid]

    return jsonify({"ok": True})

# ==========================================================
@app.route("/fault")
def fault():

    global FAULT_NODE

    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    _, i = TREE.query([lon, lat])

    FAULT_NODE = NODE_LIST[i]

    return jsonify({"ok": True})

# ==========================================================
@app.route("/clear_fault")
def clear_fault():

    global FAULT_NODE

    FAULT_NODE = None

    return jsonify({"ok": True})

# ==========================================================
@app.route("/api/scada")
def scada():

    active = get_active_nodes()

    total = len(NODE_LIST)

    nodes_on = len(active)

    return jsonify({
        "energized_pct": round(nodes_on / total * 100, 1),
        "nodes_on": nodes_on,
        "nodes_off": total - nodes_on,
        "open_switches": sum(
            1 for v in SWITCH_STATUS.values()
            if v == 0
        ),
        "total_switches": len(SWITCH_STATUS),
        "fault_feeder": FAULT_FEEDER
    })

# ==========================================================
print("=" * 60)
print("PEA SPARK STARTING")
print("=" * 60)

build()

# ==========================================================
@app.route("/api/debug")
def debug():

    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data"
    )

    return jsonify({
        "build_ok": G is not None and BUILD_ERROR is None,
        "build_error": BUILD_ERROR,
        "nodes": len(NODE_LIST),
        "edges": G.number_of_edges() if G else 0,
        "switches": len(SWITCH_STATUS),
        "feeders": list(FEEDER_COLOR.keys()),
        "conductor_rows": len(_CONDUCTOR_FEATURES),
        "dof_rows": len(_DOF_FEATURES),
        "data_dir": data_dir,
        "data_files": os.listdir(data_dir) if os.path.isdir(data_dir) else [],
        "cwd": os.getcwd(),
    })

# ==========================================================
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=3000,
        debug=False
    )

