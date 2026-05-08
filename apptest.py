# =========================================================
# PEA SPARK ENTERPRISE - FULL VERSION (FIXED)
# apptest.py
# =========================================================

from flask import Flask, jsonify, render_template, request
import networkx as nx
from scipy.spatial import KDTree
import json
import random
import traceback
import os
import geopandas as gpd

app = Flask(__name__)

# =========================================================
# LOAD GIS DATA
# =========================================================

DATA_DIR = "data"
PSCONDUCTOR = gpd.read_file(os.path.join(DATA_DIR, "psconductor.json"))
DOF = gpd.read_file(os.path.join(DATA_DIR, "DOF.json"))

PSCB = None
if os.path.exists(os.path.join(DATA_DIR, "pscb.json")):
    PSCB = gpd.read_file(os.path.join(DATA_DIR, "pscb.json"))

# UTM Zone 47N to WGS84
PSCONDUCTOR = PSCONDUCTOR.set_crs(epsg=32647, allow_override=True).to_crs(epsg=4326)
DOF = DOF.set_crs(epsg=32647, allow_override=True).to_crs(epsg=4326)

if PSCB is not None:
    PSCB = PSCB.set_crs(epsg=32647, allow_override=True).to_crs(epsg=4326)

# =========================================================
# GLOBAL STATE
# =========================================================

G = nx.Graph()
NODE_LIST = []
TREE = None
SWITCH_STATUS = {}
SWITCH_NODES = {}
FAULT_NODE = None
FAULT_FEEDER = None
BUILD_OK = False
BUILD_ERROR = None
FEEDER_COLORS = {}
SOURCE_NODES = []

COLOR_POOL = ["#00e5ff", "#ff1744", "#00ff90", "#ffd600", "#ff9100", "#7c4dff", "#40c4ff", "#69f0ae", "#ff5252", "#ffff00", "#18ffff", "#ff4081"]

def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================================================
# BUILD NETWORK
# =========================================================

def build():
    global G, NODE_LIST, TREE, BUILD_OK, BUILD_ERROR, FEEDER_COLORS, SWITCH_NODES, SOURCE_NODES
    try:
        BUILD_OK, BUILD_ERROR = False, None
        G = nx.Graph()
        NODE_LIST, SWITCH_NODES, SOURCE_NODES, FEEDER_COLORS = [], {}, [], {}

        # 1. CONDUCTORS
        conductor_path = os.path.join(DATA_DIR, "psconductor.json")
        conductor_json = load_geojson(conductor_path)
        for feat in conductor_json.get("features", []):
            geom = feat.get("geometry", {})
            feeder = str(feat.get("properties", {}).get("FEEDERID") or "UNKNOWN")
            if feeder not in FEEDER_COLORS:
                FEEDER_COLORS[feeder] = COLOR_POOL[len(FEEDER_COLORS) % len(COLOR_POOL)]
            
            coords_batch = [geom.get("coordinates", [])] if geom.get("type") == "LineString" else geom.get("coordinates", [])
            for line in coords_batch:
                for i in range(len(line) - 1):
                    a, b = tuple(line[i][:2]), tuple(line[i+1][:2])
                    G.add_edge(a, b, feeder=feeder)

        # 2. TREE & NODES
        NODE_LIST = list(G.nodes())
        TREE = KDTree(NODE_LIST)

        # 3. DOF SWITCHES
        dof_path = os.path.join(DATA_DIR, "DOF.json")
        dof_json = load_geojson(dof_path)
        for feat in dof_json.get("features", []):
            if feat.get("geometry", {}).get("type") != "Point": continue
            coord = feat["geometry"]["coordinates"][:2]
            dist, idx = TREE.query(coord)
            nearest = NODE_LIST[idx]
            facility = str(feat.get("properties", {}).get("FACILITYID") or f"SW_{idx}")
            SWITCH_NODES[facility] = nearest
            if facility not in SWITCH_STATUS: SWITCH_STATUS[facility] = 1

        # 4. BREAKER SOURCE
        if PSCB is not None:
            for _, row in PSCB.iterrows():
                if row.geometry and row.geometry.geom_type == "Point":
                    dist, idx = TREE.query([row.geometry.x, row.geometry.y])
                    SOURCE_NODES.append(NODE_LIST[idx])

        BUILD_OK = True
    except Exception:
        BUILD_ERROR = traceback.format_exc()
        print(BUILD_ERROR)

def get_energized():
    energized = set()
    if not SOURCE_NODES: return energized
    stack = list(SOURCE_NODES)
    while stack:
        node = stack.pop()
        if node in energized: continue
        energized.add(node)
        for nbr in G.neighbors(node):
            if nbr == FAULT_NODE: continue
            is_blocked = False
            for swid, swnode in SWITCH_NODES.items():
                if SWITCH_STATUS.get(swid, 1) == 0 and swnode == nbr:
                    is_blocked = True; break
            if not is_blocked and nbr not in energized: stack.append(nbr)
    return energized

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home(): return render_template("indexpro.html")

@app.route("/api/conductor")
def api_conductor():
    try:
        energized = get_energized()
        features = []
        for _, row in PSCONDUCTOR.iterrows():
            geom = row.geometry
            if not geom: continue
            feeder = str(row.get("FEEDERID", "UNKNOWN"))
            # Check status
            is_on = False
            if geom.geom_type == "LineString":
                is_on = any(tuple(c[:2]) in energized for c in geom.coords)
            elif geom.geom_type == "MultiLineString":
                is_on = any(tuple(c[:2]) in energized for line in geom.geoms for c in line.coords)
            
            features.append({
                "type": "Feature",
                "geometry": json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry'],
                "properties": {"feeder": feeder, "color": FEEDER_COLORS.get(feeder, "#00ffff"), "status": "on" if is_on else "off"}
            })
        return jsonify({"type": "FeatureCollection", "features": features})
    except: return jsonify({"error": traceback.format_exc()})

@app.route("/api/dof")
def api_dof():
    features = []
    for _, row in DOF.iterrows():
        if not row.geometry: continue
        fid = str(row.get("FACILITYID", "SW"))
        features.append({
            "type": "Feature", "geometry": {"type": "Point", "coordinates": [row.geometry.x, row.geometry.y]},
            "properties": {"id": fid, "feeder": str(row.get("FEEDERID", "UNK")), "status": SWITCH_STATUS.get(fid, 1), "type": "switch"}
        })
    return jsonify({"type": "FeatureCollection", "features": features})

@app.route("/api/scada")
def api_scada():
    energized = get_energized()
    total = len(G.nodes())
    return jsonify({
        "energized_pct": round(len(energized)/total*100, 1) if total > 0 else 0,
        "open_switches": list(SWITCH_STATUS.values()).count(0),
        "total_switches": len(SWITCH_STATUS),
        "fault_feeder": FAULT_FEEDER or "none"
    })

@app.route("/toggle_switch")
def toggle_switch():
    swid = request.args.get("id")
    if swid in SWITCH_STATUS: SWITCH_STATUS[swid] = 1 - SWITCH_STATUS[swid]
    return jsonify({"ok": True})

@app.route("/fault")
def fault():
    global FAULT_NODE, FAULT_FEEDER
    try:
        lat, lon = float(request.args.get("lat")), float(request.args.get("lon"))
        dist, idx = TREE.query((lon, lat))
        FAULT_NODE = NODE_LIST[idx]
        for nbr in G.neighbors(FAULT_NODE):
            FAULT_FEEDER = G[FAULT_NODE][nbr].get("feeder", "UNKNOWN"); break
        return jsonify({"ok": True, "feeder": FAULT_FEEDER})
    except: return jsonify({"ok": False})

@app.route("/clear_fault")
def clear_fault():
    global FAULT_NODE, FAULT_FEEDER
    FAULT_NODE, FAULT_FEEDER = None, None
    return jsonify({"ok": True})

if __name__ == "__main__":
    build()
    app.run(port=5000, debug=True)
