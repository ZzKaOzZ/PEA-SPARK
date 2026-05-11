# =========================================================
# PEA SPARK ENTERPRISE
# FULL VERSION
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

PSCONDUCTOR = gpd.read_file("data/psconductor.json")
DOF = gpd.read_file("data/DOF.json")

# optional
PSCB = None
if os.path.exists("data/pscb.json"):
    PSCB = gpd.read_file("data/pscb.json")

# UTM Zone 47N
PSCONDUCTOR = PSCONDUCTOR.set_crs(epsg=32647, allow_override=True)
DOF = DOF.set_crs(epsg=32647, allow_override=True)

if PSCB is not None:
    PSCB = PSCB.set_crs(epsg=32647, allow_override=True)

# convert latlon
PSCONDUCTOR = PSCONDUCTOR.to_crs(epsg=4326)
DOF = DOF.to_crs(epsg=4326)

if PSCB is not None:
    PSCB = PSCB.to_crs(epsg=4326)

# =========================================================
# GLOBAL
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

DATA_DIR = "data"

# =========================================================
# COLOR POOL
# =========================================================

COLOR_POOL = [
    "#00e5ff",
    "#ff1744",
    "#00ff90",
    "#ffd600",
    "#ff9100",
    "#7c4dff",
    "#40c4ff",
    "#69f0ae",
    "#ff5252",
    "#ffff00",
    "#18ffff",
    "#ff4081",
]

# =========================================================
# LOAD GEOJSON
# =========================================================

def load_geojson(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================================================
# BUILD NETWORK
# =========================================================

def build():

    global G
    global NODE_LIST
    global TREE
    global BUILD_OK
    global BUILD_ERROR
    global FEEDER_COLORS
    global SWITCH_NODES
    global SOURCE_NODES

    try:

        BUILD_OK = False
        BUILD_ERROR = None

        G = nx.Graph()

        NODE_LIST = []
        SWITCH_NODES = {}
        SOURCE_NODES = []
        FEEDER_COLORS = {}

        # =================================================
        # CONDUCTOR
        # =================================================

        conductor_path = os.path.join(DATA_DIR, "psconductor.json")

        conductor_json = load_geojson(conductor_path)

        for feat in conductor_json.get("features", []):

            geom = feat.get("geometry", {})
            props = feat.get("properties", {})

            feeder = str(
                props.get("FEEDERID")
                or "UNKNOWN"
            )

            if feeder not in FEEDER_COLORS:

                FEEDER_COLORS[feeder] = COLOR_POOL[
                    len(FEEDER_COLORS) % len(COLOR_POOL)
                ]

            # -----------------------------------------
            # LINESTRING
            # -----------------------------------------

            if geom.get("type") == "LineString":

                coords = geom.get("coordinates", [])

                for i in range(len(coords) - 1):

                    a = tuple(coords[i])
                    b = tuple(coords[i + 1])

                    G.add_edge(
                        a,
                        b,
                        feeder=feeder
                    )

            # -----------------------------------------
            # MULTILINESTRING
            # -----------------------------------------

            elif geom.get("type") == "MultiLineString":

                for line in geom.get("coordinates", []):

                    for i in range(len(line) - 1):

                        a = tuple(line[i])
                        b = tuple(line[i + 1])

                        G.add_edge(
                            a,
                            b,
                            feeder=feeder
                        )

        # =================================================
        # TREE
        # =================================================

        NODE_LIST = list(G.nodes())

        TREE = KDTree(NODE_LIST)

        # =================================================
        # DOF
        # =================================================

        dof_path = os.path.join(DATA_DIR, "DOF.json")

        dof_json = load_geojson(dof_path)

        for feat in dof_json.get("features", []):

            geom = feat.get("geometry", {})
            props = feat.get("properties", {})

            if geom.get("type") != "Point":
                continue

            coord = geom["coordinates"]

            dist, idx = TREE.query(coord)

            nearest = NODE_LIST[idx]

            facility = str(
                props.get("FACILITYID")
                or props.get("DEVICEID")
                or props.get("NAME")
                or f"SW_{idx}"
            )

            SWITCH_NODES[facility] = nearest

            if facility not in SWITCH_STATUS:

                SWITCH_STATUS[facility] = 1

        # =================================================
        # BREAKER SOURCE
        # =================================================

        if PSCB is not None:

            for _, row in PSCB.iterrows():

                geom = row.geometry

                if geom is None:
                    continue

                if geom.geom_type != "Point":
                    continue

                coord = [geom.x, geom.y]

                dist, idx = TREE.query(coord)

                nearest = NODE_LIST[idx]

                SOURCE_NODES.append(nearest)

        BUILD_OK = True

        print("=" * 60)
        print("BUILD OK")
        print("NODES       =", len(G.nodes()))
        print("EDGES       =", len(G.edges()))
        print("SWITCHES    =", len(SWITCH_NODES))
        print("SOURCES     =", len(SOURCE_NODES))
        print("FEEDERS     =", len(FEEDER_COLORS))
        print("=" * 60)

    except Exception:

        BUILD_OK = False
        BUILD_ERROR = traceback.format_exc()

        print(BUILD_ERROR)

# =========================================================
# ENERGIZED
# =========================================================

def get_energized():

    energized = set()

    if len(SOURCE_NODES) == 0:
        return energized

    stack = list(SOURCE_NODES)

    while stack:

        node = stack.pop()

        if node in energized:
            continue

        energized.add(node)

        for nbr in G.neighbors(node):

            if nbr == FAULT_NODE:
                continue

            blocked = False

            for swid, swnode in SWITCH_NODES.items():

                if SWITCH_STATUS.get(swid, 1) == 0:

                    if swnode == nbr:

                        blocked = True
                        break

            if blocked:
                continue

            if nbr not in energized:
                stack.append(nbr)

    return energized

# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("indexpro.html")

# =========================================================
# DEBUG
# =========================================================

@app.route("/api/debug")
def api_debug():

    return jsonify({
        "build_ok": BUILD_OK,
        "build_error": BUILD_ERROR,
        "nodes": len(G.nodes()),
        "edges": len(G.edges()),
        "switches": len(SWITCH_STATUS),
        "sources": len(SOURCE_NODES),
        "feeders": len(FEEDER_COLORS)
    })

# =========================================================
# API CONDUCTOR
# =========================================================

@app.route("/api/conductor")
def api_conductor():

    try:

        features = []

        for _, row in PSCONDUCTOR.iterrows():

            geom = row.geometry

            if geom is None:
                continue

            feeder = str(row.get("FEEDERID", "UNKNOWN"))

            color = FEEDER_COLORS.get(
                feeder,
                "#00ffff"
            )

            # -----------------------------------------
            # LINESTRING
            # -----------------------------------------

            if geom.geom_type == "LineString":

                coords = [[x, y] for x, y in geom.coords]

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": {
                        "feeder": feeder,
                        "color": color,
                        "status": "on"
                    }
                })

            # -----------------------------------------
            # MULTILINESTRING
            # -----------------------------------------

            elif geom.geom_type == "MultiLineString":

                for line in geom.geoms:

                    coords = [[x, y] for x, y in line.coords]

                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coords
                        },
                        "properties": {
                            "feeder": feeder,
                            "color": color,
                            "status": "on"
                        }
                    })

        return jsonify({
            "type": "FeatureCollection",
            "features": features
        })

    except Exception:

        return jsonify({
            "error": traceback.format_exc()
        })

# =========================================================
# API DOF
# =========================================================

@app.route("/api/dof")
def api_dof():

    try:

        features = []

        # =================================================
        # DOF DEVICES
        # =================================================

        for _, row in DOF.iterrows():

            geom = row.geometry

            if geom is None:
                continue

            if geom.geom_type != "Point":
                continue

            facility = str(
                row.get("FACILITYID", "")
            )

            feeder = str(
                row.get("FEEDERID", "UNKNOWN")
            )

            device_type = "switch"

            if facility.startswith("F"):
                device_type = "dropout"

            elif facility.startswith("S"):
                device_type = "tie switch"

            elif facility.startswith("R"):
                device_type = "recloser"

            status = SWITCH_STATUS.get(
                facility,
                1
            )

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [geom.x, geom.y]
                },
                "properties": {
                    "id": facility,
                    "feeder": feeder,
                    "type": device_type,
                    "status": status,
                    "location": feeder
                }
            })

        # =================================================
        # BREAKERS
        # =================================================

        if PSCB is not None:

            for _, row in PSCB.iterrows():

                geom = row.geometry

                if geom is None:
                    continue

                if geom.geom_type != "Point":
                    continue

                facility = str(
                    row.get("FACILITYID", "CB")
                )

                feeder = str(
                    row.get("FEEDERID", "UNKNOWN")
                )

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [geom.x, geom.y]
                    },
                    "properties": {
                        "id": facility,
                        "feeder": feeder,
                        "type": "breaker",
                        "status": 1,
                        "location": "SOURCE"
                    }
                })

        return jsonify({
            "type": "FeatureCollection",
            "features": features
        })

    except Exception:

        return jsonify({
            "error": traceback.format_exc()
        })

# =========================================================
# API SCADA
# =========================================================

@app.route("/api/scada")
def api_scada():

    try:

        energized = get_energized()

        total_nodes = len(G.nodes())

        nodes_on = len(energized)

        nodes_off = total_nodes - nodes_on

        energized_pct = 0

        if total_nodes > 0:

            energized_pct = round(
                (nodes_on / total_nodes) * 100,
                1
            )

        feeders = {}

        for feeder, color in FEEDER_COLORS.items():

            feeder_nodes = set()

            for a, b, d in G.edges(data=True):

                if d.get("feeder") == feeder:

                    feeder_nodes.add(a)
                    feeder_nodes.add(b)

            if len(feeder_nodes) == 0:

                fpct = 0

            else:

                fon = len(
                    feeder_nodes.intersection(energized)
                )

                fpct = round(
                    (fon / len(feeder_nodes)) * 100
                )

            feeders[feeder] = {
                "pct": fpct,
                "color": color
            }

        return jsonify({

            "nodes_on": nodes_on,
            "nodes_off": nodes_off,

            "energized_pct": energized_pct,

            "fault_feeder": FAULT_FEEDER,

            "open_switches": sum(
                1 for v in SWITCH_STATUS.values()
                if v == 0
            ),

            "total_switches": len(SWITCH_STATUS),

            "feeders": feeders
        })

    except Exception:

        return jsonify({
            "error": traceback.format_exc()
        })

# =========================================================
# TOGGLE SWITCH
# =========================================================

@app.route("/toggle_switch")
def toggle_switch():

    swid = request.args.get("id")

    if swid in SWITCH_STATUS:

        SWITCH_STATUS[swid] = (
            0 if SWITCH_STATUS[swid] == 1 else 1
        )

    return jsonify({
        "ok": True
    })

# =========================================================
# FAULT
# =========================================================

@app.route("/fault")
def fault():

    global FAULT_NODE
    global FAULT_FEEDER

    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    point = (lon, lat)

    dist, idx = TREE.query(point)

    FAULT_NODE = NODE_LIST[idx]

    feeder = "UNKNOWN"

    for nbr in G.neighbors(FAULT_NODE):

        feeder = G[FAULT_NODE][nbr].get(
            "feeder",
            "UNKNOWN"
        )

        break

    FAULT_FEEDER = feeder

    return jsonify({
        "ok": True,
        "feeder": feeder
    })

# =========================================================
# CLEAR FAULT
# =========================================================

@app.route("/clear_fault")
def clear_fault():

    global FAULT_NODE
    global FAULT_FEEDER

    FAULT_NODE = None
    FAULT_FEEDER = None

    return jsonify({
        "ok": True
    })

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    build()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
