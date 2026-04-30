from flask import Flask, jsonify, render_template, request
import json, os
import networkx as nx
from scipy.spatial import KDTree

app = Flask(__name__)

G = None
NODE_LIST = []
TREE = None

SWITCH_NODES = {}
SWITCH_STATUS = {}
FEEDER_COLOR = {}

FAULT_NODE = None
FAULT_FEEDER = None

# =========================
def load(path):
    if not os.path.exists(path):
        alt = path.replace(".geojson", ".json")
        if os.path.exists(alt):
            path = alt
        else:
            print("❌ FILE NOT FOUND:", path)
            return {"features": []}

    with open(path, encoding="utf-8") as f:
        return json.load(f)

# =========================
def build():
    global G, NODE_LIST, TREE

    G = nx.Graph()
    nodes = []

    data = load("data/psconductor.geojson")

    for f in data["features"]:
        geom = f["geometry"]
        feeder = str(f["properties"].get("FEEDERID", "UNK"))

        if geom["type"] != "LineString":
            continue

        coords = geom["coordinates"]

        for i in range(len(coords) - 1):
            a = tuple(coords[i])
            b = tuple(coords[i + 1])
            G.add_edge(a, b, feeder=feeder)
            nodes += [a, b]

    NODE_LIST = list(set(nodes))

    if NODE_LIST:
        TREE = KDTree(NODE_LIST)

    # SWITCH
    dof = load("data/DOF.geojson")

    for f in dof["features"]:
        fid = str(f["properties"].get("FACILITYID", ""))

        if "S" not in fid.upper():
            continue

        pos = int(f["properties"].get("PRESENTPOS", 1))
        lon, lat = f["geometry"]["coordinates"]

        if TREE:
            _, i = TREE.query([lon, lat])
            SWITCH_NODES[fid] = NODE_LIST[i]
            SWITCH_STATUS[fid] = pos

    # COLOR
    palette = ["#00e5ff", "#7c4dff", "#ff9100", "#00e676", "#ff5252", "#ffd600"]
    feeders = set(nx.get_edge_attributes(G, 'feeder').values())

    for i, f in enumerate(feeders):
        FEEDER_COLOR[f] = palette[i % len(palette)]

# =========================
def apply_fault():
    try:
        G2 = G.copy()

        if FAULT_NODE and FAULT_NODE in G2:
            G2.remove_node(FAULT_NODE)

        for fid, node in SWITCH_NODES.items():
            if SWITCH_STATUS.get(fid, 1) == 0:
                if node in G2:
                    G2.remove_node(node)

        return G2
    except Exception as e:
        print("❌ apply_fault error:", e)
        return nx.Graph()

# =========================
def get_active_nodes():
    try:
        G2 = apply_fault()

        if G2 is None or len(G2.nodes) == 0:
            return set()

        comps = list(nx.connected_components(G2))
        if not comps:
            return set()

        return set().union(*comps)

    except Exception as e:
        print("❌ get_active_nodes error:", e)
        return set()

# =========================
@app.route("/")
def index():
    return render_template("indexpro.html")

# =========================
@app.route("/fault")
def fault():
    global FAULT_NODE, FAULT_FEEDER

    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))

        if TREE:
            _, i = TREE.query([lon, lat])
            FAULT_NODE = NODE_LIST[i]

            FAULT_FEEDER = "UNK"
            for u, v, d in G.edges(data=True):
                if u == FAULT_NODE or v == FAULT_NODE:
                    FAULT_FEEDER = d.get("feeder", "UNK")
                    break

        return jsonify({"node": str(FAULT_NODE), "feeder": FAULT_FEEDER})

    except Exception as e:
        print("❌ fault error:", e)
        return jsonify({"node": None, "feeder": "-"})

# =========================
@app.route("/api/conductor")
def conductor():
    try:
        active = get_active_nodes()
        data = load("data/psconductor.geojson")

        feats = []
        for f in data["features"]:
            coords = f["geometry"]["coordinates"]
            feeder = str(f["properties"].get("FEEDERID", "UNK"))

            status = "on"
            if any(tuple(c) not in active for c in coords):
                status = "off"

            feats.append({
                "type": "Feature",
                "geometry": f["geometry"],
                "properties": {
                    "feeder": feeder,
                    "status": status,
                    "color": FEEDER_COLOR.get(feeder, "#888")
                }
            })

        return jsonify({"type": "FeatureCollection", "features": feats})

    except Exception as e:
        print("❌ conductor error:", e)
        return jsonify({"type": "FeatureCollection", "features": []})

# =========================
@app.route("/api/dof")
def dof():
    try:
        data = load("data/DOF.geojson")

        feats = []
        for f in data["features"]:
            fid = str(f["properties"].get("FACILITYID", ""))

            if "S" not in fid.upper():
                continue

            pos = SWITCH_STATUS.get(fid, 1)

            feats.append({
                "type": "Feature",
                "geometry": f["geometry"],
                "properties": {
                    "id": fid,
                    "status": pos
                }
            })

        return jsonify({"type": "FeatureCollection", "features": feats})

    except Exception as e:
        print("❌ dof error:", e)
        return jsonify({"type": "FeatureCollection", "features": []})

# =========================
@app.route("/toggle_switch")
def toggle():
    try:
        fid = request.args.get("id")

        if fid in SWITCH_STATUS:
            SWITCH_STATUS[fid] = 1 - SWITCH_STATUS[fid]

        return jsonify({"id": fid, "status": SWITCH_STATUS.get(fid)})

    except Exception as e:
        print("❌ toggle error:", e)
        return jsonify({"id": None, "status": 0})

# =========================
@app.route("/api/scada")
def scada():
    try:
        active = get_active_nodes()
        total = len(NODE_LIST)

        return jsonify({
            "fault_feeder": FAULT_FEEDER or "-",
            "switch_open": sum(1 for v in SWITCH_STATUS.values() if v == 0),
            "nodes_on": len(active),
            "nodes_off": total - len(active)
        })

    except Exception as e:
        print("❌ scada error:", e)
        return jsonify({
            "fault_feeder": "-",
            "switch_open": 0,
            "nodes_on": 0,
            "nodes_off": 0
        })

# =========================
print("🚀 BUILD GRAPH...")
build()
print("✅ BUILD DONE")

# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
