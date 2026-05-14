from __future__ import annotations
import json
import math
import os
from collections import deque
from flask import Flask, jsonify, request, abort

try:
    from pyproj import Transformer
    _TRANSFORMER     = Transformer.from_crs("EPSG:24047", "EPSG:4326", always_xy=True)
    _TRANSFORMER_INV = Transformer.from_crs("EPSG:4326", "EPSG:24047", always_xy=True)
    def to_wgs(x: float, y: float) -> tuple[float, float]:
        lon, lat = _TRANSFORMER.transform(x, y)
        return lon, lat
    def to_utm(lon: float, lat: float) -> tuple[float, float]:
        x, y = _TRANSFORMER_INV.transform(lon, lat)
        return x, y
except ImportError:
    raise SystemExit("กรุณาติดตั้ง pyproj: pip install pyproj")

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "templates")

FEEDER_PALETTE = [
    "#00e5ff","#7c4dff","#ff9100","#00e676","#ff5252","#ffd600",
    "#40c4ff","#b388ff","#ff6e40","#69f0ae","#f06292","#ffab40",
]

# ─── Network State ────────────────────────────────────────────────────────────
class NetworkState:
    def __init__(self):
        self.adjacency:    dict[str, set[str]] = {}
        self.node_feeder:  dict[str, str]       = {}
        self.node_xy:      dict[str, tuple[float, float]] = {}
        self.nodes:        list[tuple[str, float, float]] = []

        self.conductor_keys: list[list[str]] = []
        self.conductor_wgs:  list[dict]      = []

        self.switches:     list[dict]        = []
        self.switch_node:  dict[str, str]    = {}
        self.switch_status:dict[str, int]    = {}   # 1=closed 0=open

        self.substations:  list[dict]        = []
        self.cb_node:      dict[str, str]    = {}
        self.cb_feeder:    dict[str, str]    = {}
        self.cb_status:    dict[str, int]    = {}
        self.feeder_cbs:   dict[str, set[str]] = {}

        self.reclosers:    list[dict]        = []
        self.transformers: list[dict]        = []

        self.feeder_color:      dict[str, str] = {}
        self.feeder_edge_count: dict[str, int] = {}

        self.fault_node:   str | None   = None
        self.fault_feeder: str | None   = None
        self.fault_lat:    float | None = None
        self.fault_lon:    float | None = None

_STATE: NetworkState | None = None


def load_json(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def node_key(x: float, y: float) -> str:
    return f"{round(x * 1e4) / 1e4}|{round(y * 1e4) / 1e4}"


def find_nearest(nodes: list[tuple[str, float, float]], x: float, y: float) -> str | None:
    best, best_d = None, math.inf
    for key, nx, ny in nodes:
        d = (nx - x) ** 2 + (ny - y) ** 2
        if d < best_d:
            best_d, best = d, key
    return best


def build_state() -> NetworkState:
    s = NetworkState()
    print("กำลังโหลดข้อมูลเครือข่าย…")

    conductor_fc = load_json("psconductor.json")
    dof_fc       = load_json("DOF.json")
    recloser_fc  = load_json("psrecloser.json")
    trans_fc     = load_json("pstrans.json")
    pscb_fc      = load_json("pscb.json")

    # Conductors
    for feat in conductor_fc.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString": continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2: continue
        props  = feat.get("properties") or {}
        feeder = str(props.get("FEEDERID", "UNK"))
        keys: list[str] = []
        for c in coords:
            x, y = float(c[0]), float(c[1])
            k = node_key(x, y)
            if k not in s.node_xy: s.node_xy[k] = (x, y)
            keys.append(k)
            s.node_feeder.setdefault(k, feeder)
        for i in range(len(keys) - 1):
            a, b = keys[i], keys[i + 1]
            if a == b: continue
            s.adjacency.setdefault(a, set()).add(b)
            s.adjacency.setdefault(b, set()).add(a)
        s.conductor_keys.append(keys)
        s.feeder_edge_count[feeder] = s.feeder_edge_count.get(feeder, 0) + (len(keys) - 1)
        wgs_coords = [list(to_wgs(float(c[0]), float(c[1]))) for c in coords]
        s.conductor_wgs.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": wgs_coords},
            "properties": {"feeder": feeder, "status": "on", "color": "#888"},
        })

    s.nodes = [(k, xy[0], xy[1]) for k, xy in s.node_xy.items()]
    feeders = sorted(s.feeder_edge_count.keys())
    for i, f in enumerate(feeders):
        s.feeder_color[f] = FEEDER_PALETTE[i % len(FEEDER_PALETTE)]
    for cw in s.conductor_wgs:
        cw["properties"]["color"] = s.feeder_color.get(cw["properties"]["feeder"], "#888")

    # Switches
    for feat in dof_fc.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point": continue
        props = feat.get("properties") or {}
        fid   = str(props.get("FACILITYID", ""))
        if not fid or "S" not in fid.upper(): continue
        x, y   = float(geom["coordinates"][0]), float(geom["coordinates"][1])
        status  = 0 if int(props.get("PRESENTPOS", 1)) == 0 else 1
        nearest = find_nearest(s.nodes, x, y)
        if not nearest: continue
        feeder  = str(props.get("FEEDERID", s.node_feeder.get(nearest, "UNK")))
        subtype = int(props.get("SUBTYPECOD", 0))
        kind    = {5: "Load Break", 3: "Disconnect", 2: "Fuse"}.get(subtype, "Switch")
        lon, lat = to_wgs(x, y)
        s.switches.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"id": fid, "feeder": feeder, "location": str(props.get("LOCATION", "")),
                           "state": "CLOSE" if status == 1 else "OPEN", "status": status, "kind": kind},
        })
        s.switch_node[fid]   = nearest
        s.switch_status[fid] = status

    # Substations (source CBs)
    for feat in pscb_fc.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point": continue
        props   = feat.get("properties") or {}
        fid     = str(props.get("FACILITYID", props.get("TAG", "")))
        if not fid: continue
        x, y    = float(geom["coordinates"][0]), float(geom["coordinates"][1])
        status  = 0 if int(props.get("PRESENTPOS", 1)) == 0 else 1
        nearest = find_nearest(s.nodes, x, y)
        if not nearest: continue
        feeder  = str(props.get("FEEDERID", "UNK"))
        lon, lat = to_wgs(x, y)
        s.substations.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"id": fid, "feeder": feeder, "location": str(props.get("LOCATION", "")),
                           "state": "CLOSE" if status == 1 else "OPEN", "status": status,
                           "tag": str(props.get("TAG", "")), "opVolt": str(props.get("OP_VOLT", ""))},
        })
        s.cb_node[fid]    = nearest
        s.cb_feeder[fid]  = feeder
        s.cb_status[fid]  = status
        s.feeder_cbs.setdefault(feeder, set()).add(fid)

    # Reclosers
    for feat in recloser_fc.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point": continue
        props = feat.get("properties") or {}
        x, y  = float(geom["coordinates"][0]), float(geom["coordinates"][1])
        lon, lat = to_wgs(x, y)
        s.reclosers.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"id": str(props.get("FACILITYID", props.get("TAG", "RC"))),
                           "feeder": str(props.get("FEEDERID", "UNK")),
                           "location": str(props.get("LOCATION", ""))}})

    # Transformers
    for feat in trans_fc.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point": continue
        props = feat.get("properties") or {}
        x, y  = float(geom["coordinates"][0]), float(geom["coordinates"][1])
        lon, lat = to_wgs(x, y)
        s.transformers.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"id": str(props.get("FACILITYID", "XF")), "feeder": str(props.get("FEEDERID", "UNK")),
                           "location": str(props.get("LOCATION", "")),
                           "rateKva": float(props.get("RATEKVA", 0) or 0),
                           "owner": str(props.get("OWNER", ""))}})

    print(f"  conductors : {len(s.conductor_keys):,}")
    print(f"  nodes      : {len(s.nodes):,}")
    print(f"  switches   : {len(s.switches):,}")
    print(f"  substations: {len(s.substations):,}")
    return s


def get_state() -> NetworkState:
    global _STATE
    if _STATE is None:
        _STATE = build_state()
    return _STATE


# ─── Energization (source-aware BFS) ─────────────────────────────────────────
def compute_energization_ex(
    adjacency:    dict[str, set[str]],
    node_feeder:  dict[str, str],
    cb_node:      dict[str, str],
    cb_feeder:    dict[str, str],
    cb_status:    dict[str, int],
    feeder_cbs:   dict[str, set[str]],
    switch_node:  dict[str, str],
    switch_status:dict[str, int],
    fault_node:   str | None,
) -> set[str]:
    """Core BFS energization — accepts explicit parameters so it can be used for simulation."""
    removed: set[str] = set()
    if fault_node:
        removed.add(fault_node)
    for fid, st in switch_status.items():
        if st == 0 and fid in switch_node:
            removed.add(switch_node[fid])

    feeder_source_off: set[str] = set()
    for feeder, cb_set in feeder_cbs.items():
        if all(cb_status.get(fid, 1) == 0 for fid in cb_set):
            feeder_source_off.add(feeder)

    source_nodes: set[str] = set()
    for fid, node in cb_node.items():
        feeder = cb_feeder.get(fid, "UNK")
        if cb_status.get(fid, 1) == 1 and node not in removed and feeder not in feeder_source_off:
            source_nodes.add(node)

    energized: set[str] = set()
    queue: deque[str]   = deque()
    for n in source_nodes:
        if n not in removed:
            energized.add(n)
            queue.append(n)
    while queue:
        cur = queue.popleft()
        for nb in adjacency.get(cur, set()):
            if nb not in removed and nb not in energized:
                energized.add(nb)
                queue.append(nb)

    for k in list(energized):
        if node_feeder.get(k, "") in feeder_source_off:
            energized.discard(k)
    return energized


def compute_energization(s: NetworkState) -> set[str]:
    return compute_energization_ex(
        s.adjacency, s.node_feeder,
        s.cb_node, s.cb_feeder, s.cb_status, s.feeder_cbs,
        s.switch_node, s.switch_status, s.fault_node,
    )


def build_live_conductors(s: NetworkState):
    energized = compute_energization(s)
    feeders_affected: set[str] = set()
    feeders_source_open = sorted(
        feeder for feeder, cb_set in s.feeder_cbs.items()
        if all(s.cb_status.get(fid, 1) == 0 for fid in cb_set)
    )
    out = []
    for cw, keys in zip(s.conductor_wgs, s.conductor_keys):
        on = all(k in energized for k in keys)
        if not on:
            feeders_affected.add(cw["properties"]["feeder"])
        out.append({**cw, "properties": {**cw["properties"], "status": "on" if on else "off"}})
    return out, sorted(feeders_affected), feeders_source_open


# ─── FISR: Switching Plan Generator ──────────────────────────────────────────
def bfs_island(
    start: str,
    allowed: set[str],
    adjacency: dict[str, set[str]],
    removed: set[str],
) -> set[str]:
    """BFS through 'allowed' nodes from start."""
    island: set[str] = set()
    if start not in allowed or start in removed:
        return island
    queue = deque([start])
    island.add(start)
    while queue:
        cur = queue.popleft()
        for nb in adjacency.get(cur, set()):
            if nb not in island and nb not in removed and nb in allowed:
                island.add(nb)
                queue.append(nb)
    return island


def generate_switching_plan(s: NetworkState) -> dict:
    """
    Fault Isolation & Service Restoration (FISR) algorithm:

    Phase 1 — Isolation
      หา closed switches ที่คาบเส้นระหว่างโซนมีไฟ ↔ โซน fault
      → เปิดสวิตช์เหล่านี้เพื่อกักฟอลต์ไว้ในโซนเล็กที่สุด

    Phase 2 — Restoration
      หลัง isolation แบ่ง de-energized nodes เป็น islands
      สำหรับแต่ละ island ที่ไม่มี fault:
      → หา open switch ที่คาบระหว่าง island ↔ โซนมีไฟ → ปิดสวิตช์

    ผลลัพธ์: ลิสต์ขั้นตอน switching เรียงลำดับ พร้อม nodes_restored ที่คาดการณ์
    """
    if not s.fault_node:
        return {"error": "ไม่มี fault ที่ active กรุณากดปุ่ม Place fault ก่อน"}

    all_nodes  = set(s.adjacency.keys())
    energized0 = compute_energization(s)
    de_nodes0  = all_nodes - energized0

    if not de_nodes0:
        return {
            "steps": [], "faultFeeder": s.fault_feeder,
            "deenergizedNodes": 0, "totalRestorable": 0, "nodesIrrecoverable": 0,
            "summary": "ทุก node มีไฟอยู่แล้ว ไม่ต้องทำ switching",
        }

    # Removed nodes in current state (open switches + fault)
    removed0: set[str] = set()
    if s.fault_node:
        removed0.add(s.fault_node)
    for fid, st in s.switch_status.items():
        if st == 0 and fid in s.switch_node:
            removed0.add(s.switch_node[fid])

    # ── Phase 1: find fault zone ──────────────────────────────────────────────
    # BFS from fault_node through de-energized territory (find all nodes
    # that lost power directly because of the fault, not open switches)
    fault_zone = bfs_island(s.fault_node, de_nodes0 | {s.fault_node}, s.adjacency, removed0 - {s.fault_node})

    # Isolation candidates: CLOSED switches whose switch-node sits between
    # energized and fault_zone, OR whose switch-node is in fault_zone
    isolation_candidates: list[str] = []
    for fid, status in s.switch_status.items():
        if status != 1:
            continue
        node = s.switch_node.get(fid)
        if not node:
            continue
        neighbors = s.adjacency.get(node, set())
        in_fault   = node in fault_zone
        near_fault = any(nb in fault_zone for nb in neighbors)
        near_energ = any(nb in energized0  for nb in neighbors)
        if (in_fault or near_fault) and near_energ:
            isolation_candidates.append(fid)

    # Limit isolation to the 2 most-separating switches
    # (prefer switches in the faulted feeder)
    isolation_candidates.sort(
        key=lambda fid: (s.node_feeder.get(s.switch_node.get(fid, ""), "") != s.fault_feeder,)
    )
    iso_switches = isolation_candidates[:2]

    steps: list[dict] = []
    for fid in iso_switches:
        sw_props = next((sw["properties"] for sw in s.switches if sw["properties"]["id"] == fid), {})
        steps.append({
            "action":        "OPEN",
            "switchId":      fid,
            "feeder":        sw_props.get("feeder", "?"),
            "location":      sw_props.get("location", ""),
            "reason":        "แยกจุดฟอลต์ออกจากระบบ (Fault Isolation)",
            "nodesRestored": 0,
        })

    # ── Phase 2: restoration ──────────────────────────────────────────────────
    # Simulate the post-isolation state
    sim_sw = dict(s.switch_status)
    for fid in iso_switches:
        sim_sw[fid] = 0

    energized_iso = compute_energization_ex(
        s.adjacency, s.node_feeder,
        s.cb_node, s.cb_feeder, s.cb_status, s.feeder_cbs,
        s.switch_node, sim_sw, s.fault_node,
    )
    de_iso = all_nodes - energized_iso

    # Removed nodes after isolation
    removed_iso: set[str] = set()
    if s.fault_node:
        removed_iso.add(s.fault_node)
    for fid, st in sim_sw.items():
        if st == 0 and fid in s.switch_node:
            removed_iso.add(s.switch_node[fid])

    # Find de-energized islands (exclude fault zone itself)
    visited: set[str] = set(fault_zone) | removed_iso
    restorable: list[dict] = []

    for start in de_iso:
        if start in visited:
            continue
        island = bfs_island(start, de_iso, s.adjacency, removed_iso)
        if not island:
            continue
        visited.update(island)

        # Find an OPEN switch that connects island ↔ energized_iso
        best_sw   = None
        best_sw_n = 0  # prefer switches that restore more nodes
        for fid, st in sim_sw.items():
            if st != 0:
                continue
            node      = s.switch_node.get(fid)
            if not node:
                continue
            neighbors = s.adjacency.get(node, set())
            in_island   = node in island or any(nb in island   for nb in neighbors)
            near_energ  = any(nb in energized_iso for nb in neighbors)
            if in_island and near_energ:
                if best_sw is None:
                    best_sw   = fid
                    best_sw_n = len(island)
        restorable.append({"island": island, "switch": best_sw, "size": len(island)})

    # Add restoration steps (largest island first)
    used_switches: set[str] = set()
    cumulative_sw = dict(sim_sw)  # track state as we add closures
    cumulative_energized = set(energized_iso)

    for item in sorted(restorable, key=lambda x: -x["size"]):
        sw_fid = item["switch"]
        if sw_fid is None or sw_fid in used_switches:
            continue
        used_switches.add(sw_fid)
        cumulative_sw[sw_fid] = 1  # close it

        # Recalculate how many extra nodes this step actually restores
        new_energized = compute_energization_ex(
            s.adjacency, s.node_feeder,
            s.cb_node, s.cb_feeder, s.cb_status, s.feeder_cbs,
            s.switch_node, cumulative_sw, s.fault_node,
        )
        actually_restored = len(new_energized) - len(cumulative_energized)
        cumulative_energized = new_energized

        sw_props = next((sw["properties"] for sw in s.switches if sw["properties"]["id"] == sw_fid), {})
        feeder_alt = sw_props.get("feeder", "?")
        steps.append({
            "action":        "CLOSE",
            "switchId":      sw_fid,
            "feeder":        feeder_alt,
            "location":      sw_props.get("location", ""),
            "reason":        f"คืนไฟให้ {actually_restored:,} nodes (Service Restoration)",
            "nodesRestored": actually_restored,
        })

    # Number steps
    for i, step in enumerate(steps):
        step["step"] = i + 1

    total_restorable   = sum(st["nodesRestored"] for st in steps)
    nodes_irrecoverable = len(fault_zone)
    fault_pct = round(nodes_irrecoverable / max(1, len(all_nodes)) * 100, 2)

    return {
        "steps":             steps,
        "faultFeeder":       s.fault_feeder,
        "faultLat":          s.fault_lat,
        "faultLon":          s.fault_lon,
        "faultZoneNodes":    nodes_irrecoverable,
        "faultZonePct":      fault_pct,
        "deenergizedNodes":  len(de_nodes0),
        "totalRestorable":   total_restorable,
        "nodesIrrecoverable": nodes_irrecoverable,
        "summary": (
            f"ดับ {len(de_nodes0):,} nodes | "
            f"fault zone {nodes_irrecoverable:,} nodes ({fault_pct}%) | "
            f"แผน {len(steps)} ขั้นตอน | "
            f"คืนไฟได้ {total_restorable:,} nodes"
        ),
    }


# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    with open(os.path.join(_HERE, "indexpro.html"), encoding="utf-8") as f:
        return f.read()


# ── Network read endpoints ─────────────────────────────────────────────────────

@app.route("/conductor")
def conductor():
    s = get_state()
    features, _, _ = build_live_conductors(s)
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/switches")
def switches():
    s = get_state()
    out = []
    for sw in s.switches:
        fid    = sw["properties"]["id"]
        status = s.switch_status.get(fid, sw["properties"]["status"])
        out.append({**sw, "properties": {**sw["properties"],
                    "status": status, "state": "CLOSE" if status == 1 else "OPEN"}})
    return jsonify({"type": "FeatureCollection", "features": out})


@app.route("/reclosers")
def reclosers():
    return jsonify({"type": "FeatureCollection", "features": get_state().reclosers})


@app.route("/transformers")
def transformers():
    return jsonify({"type": "FeatureCollection", "features": get_state().transformers})


@app.route("/feeders")
def feeders():
    s = get_state()
    return jsonify({"feeders": [
        {"id": f, "color": c, "edgeCount": s.feeder_edge_count.get(f, 0)}
        for f, c in sorted(s.feeder_color.items())
    ]})


@app.route("/substations")
def substations():
    s = get_state()
    out = []
    for sub in s.substations:
        fid    = sub["properties"]["id"]
        status = s.cb_status.get(fid, sub["properties"]["status"])
        out.append({**sub, "properties": {**sub["properties"],
                    "status": status, "state": "CLOSE" if status == 1 else "OPEN"}})
    return jsonify({"type": "FeatureCollection", "features": out})


@app.route("/scada")
def scada():
    s = get_state()
    energized = compute_energization(s)
    _, feeders_affected, feeders_source_open = build_live_conductors(s)
    return jsonify({
        "faultActive":      bool(s.fault_node),
        "faultFeeder":      s.fault_feeder,
        "switchOpen":       sum(1 for v in s.switch_status.values() if v == 0),
        "switchTotal":      len(s.switch_status),
        "cbOpen":           sum(1 for v in s.cb_status.values() if v == 0),
        "cbTotal":          len(s.cb_status),
        "nodesOn":          len(energized),
        "nodesOff":         len(s.adjacency) - len(energized),
        "feedersAffected":  feeders_affected,
        "feedersSourceOpen": feeders_source_open,
    })


# ── Write endpoints ────────────────────────────────────────────────────────────

@app.route("/switches/<fid>/toggle", methods=["POST"])
def toggle_switch(fid: str):
    s = get_state()
    if fid not in s.switch_node:
        abort(404)
    nxt = 0 if s.switch_status.get(fid, 1) == 1 else 1
    s.switch_status[fid] = nxt
    return jsonify({"id": fid, "status": nxt, "state": "CLOSE" if nxt == 1 else "OPEN"})


@app.route("/substations/<fid>/toggle", methods=["POST"])
def toggle_substation(fid: str):
    s = get_state()
    if fid not in s.cb_node:
        abort(404)
    nxt = 0 if s.cb_status.get(fid, 1) == 1 else 1
    s.cb_status[fid] = nxt
    return jsonify({"id": fid, "status": nxt, "state": "CLOSE" if nxt == 1 else "OPEN"})


@app.route("/fault", methods=["POST"])
def set_fault():
    s    = get_state()
    data = request.get_json(force=True)
    lat, lon = float(data["lat"]), float(data["lon"])
    xu, yu   = to_utm(lon, lat)
    nearest  = find_nearest(s.nodes, xu, yu)
    if nearest:
        s.fault_node   = nearest
        s.fault_feeder = s.node_feeder.get(nearest, "UNK")
        s.fault_lat    = lat
        s.fault_lon    = lon
    return jsonify({"active": bool(s.fault_node), "feeder": s.fault_feeder,
                    "lat": s.fault_lat, "lon": s.fault_lon})


@app.route("/fault", methods=["DELETE"])
def clear_fault():
    s = get_state()
    s.fault_node = s.fault_feeder = s.fault_lat = s.fault_lon = None
    return jsonify({"active": False, "feeder": None, "lat": None, "lon": None})


@app.route("/fault", methods=["GET"])
def get_fault():
    s = get_state()
    return jsonify({"active": bool(s.fault_node), "feeder": s.fault_feeder,
                    "lat": s.fault_lat, "lon": s.fault_lon})


# ── FISR switching plan ────────────────────────────────────────────────────────

@app.route("/switching-plan", methods=["POST"])
def switching_plan():
    """
    วิเคราะห์ switching plan เพื่อลดผู้ได้รับผลกระทบ
    ไม่ต้องส่ง body — ใช้ state ปัจจุบัน (fault + switch positions)
    ตอบกลับ JSON: { steps, summary, faultFeeder, deenergizedNodes, totalRestorable, … }
    """
    plan = generate_switching_plan(get_state())
    return jsonify(plan)


@app.route("/switching-plan/execute/<int:step_idx>", methods=["POST"])
def execute_step(step_idx: int):
    """
    Execute step N of the most-recently-generated plan.
    Client sends the step object (action + switchId) directly in the body.
    """
    data     = request.get_json(force=True)
    action   = data.get("action")
    sw_id    = data.get("switchId")
    s        = get_state()
    if not sw_id:
        abort(400)
    if sw_id in s.switch_node:
        s.switch_status[sw_id] = 1 if action == "CLOSE" else 0
    return jsonify({"ok": True, "switchId": sw_id, "action": action,
                    "newStatus": s.switch_status.get(sw_id)})


if __name__ == "__main__":
    get_state()
    print("\nเซิร์ฟเวอร์พร้อมใช้งาน → http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
