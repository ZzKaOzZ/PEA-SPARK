from flask import Flask, jsonify, render_template, request
import json, os
import networkx as nx
from scipy.spatial import KDTree

app = Flask(__name__)

G=None
NODE_LIST=[]
TREE=None

SWITCH_NODES={}
SWITCH_STATUS={}
SWITCH_TYPE={}   # fid -> "Disconnect" | "Load Break" | "Switch" | "Recloser"
FEEDER_COLOR={}

FAULT_NODE=None
FAULT_FEEDER=None

# =========================
def load(path):
    with open(path,encoding="utf-8") as f:
        return json.load(f)

# =========================
def infer_switch_type(fid, props):
    """
    Infer switch type from FACILITYID or properties.
    Priority: explicit SUBTYPE field -> name pattern -> default Disconnect
    """
    subtype = str(props.get("SUBTYPE","")).upper()
    name    = fid.upper()

    if "RECLOSER" in subtype or "REC" in name or name.endswith("R"):
        return "Recloser"
    if "LOADBREAK" in subtype or "LB" in subtype or "LBS" in name:
        return "Load Break"
    if "TVS" in name or "VS" in subtype:
        return "Switch"
    return "Disconnect"

# =========================
def build():

    global G,NODE_LIST,TREE

    G=nx.Graph()
    nodes=[]

    data=load("data/psconductor.geojson")

    for f in data["features"]:
        geom=f["geometry"]
        feeder=str(f["properties"].get("FEEDERID","UNK"))

        if geom["type"]!="LineString":
            continue

        coords=geom["coordinates"]

        for i in range(len(coords)-1):
            a=tuple(coords[i])
            b=tuple(coords[i+1])

            G.add_edge(a,b,feeder=feeder)
            nodes+=[a,b]

    NODE_LIST=list(set(nodes))
    TREE=KDTree(NODE_LIST)

    # SWITCH
    dof=load("data/DOF.geojson")

    for f in dof["features"]:
        fid=str(f["properties"].get("FACILITYID",""))
        props=f["properties"]

        if "S" not in fid.upper():
            continue

        pos=int(props.get("PRESENTPOS",1))
        lon,lat=f["geometry"]["coordinates"]

        _,i=TREE.query([lon,lat])

        SWITCH_NODES[fid]=NODE_LIST[i]
        SWITCH_STATUS[fid]=pos
        SWITCH_TYPE[fid]=infer_switch_type(fid, props)

    # COLOR
    palette=["#00e5ff","#7c4dff","#ff9100","#00e676","#ff5252","#ffd600","#e040fb","#40c4ff"]

    for i,f in enumerate(sorted(set(nx.get_edge_attributes(G,'feeder').values()))):
        FEEDER_COLOR[f]=palette[i%len(palette)]

# =========================
def apply_fault():

    G2=G.copy()

    if FAULT_NODE and FAULT_NODE in G2:
        G2.remove_node(FAULT_NODE)

    for fid,node in SWITCH_NODES.items():
        if SWITCH_STATUS.get(fid,1)==0:
            if node in G2:
                G2.remove_node(node)

    return G2

# =========================
def get_active_nodes():
    G2=apply_fault()
    active=set()

    for n in G2.nodes():
        active |= set(nx.node_connected_component(G2,n))

    return active

# =========================
@app.route("/")
def index():
    return render_template("indexpro.html")

# =========================
@app.route("/fault")
def fault():

    global FAULT_NODE,FAULT_FEEDER

    lat=float(request.args.get("lat"))
    lon=float(request.args.get("lon"))

    _,i=TREE.query([lon,lat])
    FAULT_NODE=NODE_LIST[i]

    FAULT_FEEDER="UNK"
    for u,v,d in G.edges(data=True):
        if u==FAULT_NODE or v==FAULT_NODE:
            FAULT_FEEDER=d.get("feeder","UNK")
            break

    return jsonify({"node":str(FAULT_NODE),"feeder":FAULT_FEEDER})

# =========================
@app.route("/clear_fault")
def clear_fault():
    global FAULT_NODE,FAULT_FEEDER
    FAULT_NODE=None
    FAULT_FEEDER=None
    return jsonify({"status":"cleared"})

# =========================
@app.route("/api/conductor")
def conductor():

    active=get_active_nodes()
    data=load("data/psconductor.geojson")

    feats=[]
    for f in data["features"]:

        coords=f["geometry"]["coordinates"]
        feeder=str(f["properties"].get("FEEDERID","UNK"))

        status="on"
        if any(tuple(c) not in active for c in coords):
            status="off"

        feats.append({
            "type":"Feature",
            "geometry":f["geometry"],
            "properties":{
                "feeder":feeder,
                "status":status,
                "color":FEEDER_COLOR.get(feeder,"#888")
            }
        })

    return jsonify({"type":"FeatureCollection","features":feats})

# =========================
@app.route("/api/dof")
def dof():

    data=load("data/DOF.geojson")
    feats=[]

    for f in data["features"]:
        fid=str(f["properties"].get("FACILITYID",""))
        props=f["properties"]

        if "S" not in fid.upper():
            continue

        pos=SWITCH_STATUS.get(fid,1)
        sw_type=SWITCH_TYPE.get(fid,"Disconnect")

        # derive feeder from nearest edge
        lon,lat=f["geometry"]["coordinates"]
        _,idx=TREE.query([lon,lat])
        near=NODE_LIST[idx]
        feeder="UNK"
        location=str(props.get("LOCATION", props.get("STREETNAME", props.get("SUBSTATION",""))))

        for u,v,d in G.edges(near, data=True):
            feeder=d.get("feeder","UNK")
            break

        feats.append({
            "type":"Feature",
            "geometry":f["geometry"],
            "properties":{
                "id":fid,
                "state":"CLOSE" if pos==1 else "OPEN",
                "status":pos,
                "type": sw_type,
                "feeder": feeder,
                "location": location
            }
        })

    return jsonify({"type":"FeatureCollection","features":feats})

# =========================
@app.route("/toggle_switch")
def toggle():

    fid=request.args.get("id")

    if fid in SWITCH_STATUS:
        SWITCH_STATUS[fid]=1-SWITCH_STATUS[fid]

    return jsonify({"id":fid,"status":SWITCH_STATUS[fid]})

# =========================
@app.route("/api/scada")
def scada():

    active=get_active_nodes()
    total=len(NODE_LIST)
    nodes_on=len(active)
    nodes_off=total-nodes_on
    energized_pct=round(nodes_on/total*100,1) if total>0 else 0

    total_switches=len(SWITCH_STATUS)
    open_switches=sum(1 for v in SWITCH_STATUS.values() if v==0)

    # feeder summary
    feeder_nodes={}
    for u,v,d in G.edges(data=True):
        fdr=d.get("feeder","UNK")
        feeder_nodes.setdefault(fdr,set()).update([u,v])

    feeder_status={}
    for fdr,nodes in feeder_nodes.items():
        on=len(nodes & active)
        feeder_status[fdr]={
            "color": FEEDER_COLOR.get(fdr,"#888"),
            "total": len(nodes),
            "on": on,
            "pct": round(on/len(nodes)*100,1) if nodes else 0
        }

    return jsonify({
        "fault_feeder": FAULT_FEEDER,
        "fault_node": str(FAULT_NODE) if FAULT_NODE else None,
        "open_switches": open_switches,
        "total_switches": total_switches,
        "nodes_on": nodes_on,
        "nodes_off": nodes_off,
        "energized_pct": energized_pct,
        "feeders": feeder_status
    })

# =========================
@app.route("/api/feeders")
def feeders():
    return jsonify({
        "colors": FEEDER_COLOR,
        "feeders": sorted(FEEDER_COLOR.keys())
    })

# =========================
if __name__=="__main__":
    build()
    port=int(os.environ.get("PORT",3000))
    app.run(host="0.0.0.0",port=port)
