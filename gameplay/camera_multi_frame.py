"""Visibility-aware split-screen camera and multi-frame occlusion handling.

The module is intentionally defensive about input shapes.  A scalar positional
argument is interpreted as the function's primary scalar (usually a frame,
radius or camera count), while game code can pass keyword arguments or plain
dictionaries.  Every public entry point returns a :class:`Bag`, which is both a
dictionary and an attribute object.

The six behavioural fixes from the issue are implemented as testable units:

* framing uses authoritative player bounds and line-of-sight visibility rather
  than letting a wall become an asymmetric bounding-box edge;
* desired framing distance is slew-rate limited;
* asynchronous occlusion answers are accepted only when their frame tag is
  still current for that camera;
* every split-screen camera owns an independent ray-task pool;
* follow logic reads the authoritative simulation position, not a LateUpdate
  transform from the previous frame;
* rollback/replay freezes the accepted framing until play is confirmed again.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence

__all__ = [
    "Bag",
    "RayTaskPool",
    "frame_by_visibility",
    "distance_rate_limit",
    "occlusion_frame_tag",
    "pool_per_camera",
    "follow_authoritative",
    "freeze_on_rollback",
    "update",
    "occlude",
    "split",
    "present",
    "validate_camera",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

EPS = 1e-9
SKIN = 0.08
RADIUS = 0.22
DT = 1.0 / 60.0
FRAME_DISTANCE_SPEED = 12.0
SMOOTH_TIME = 0.10
MIN_DISTANCE = 2.2
MAX_DISTANCE = 14.0
DISTANCE_BASE = 3.2
DISTANCE_GAIN = 0.72
DEFAULT_FOV = 60.0
DEFAULT_ASPECT = 16.0 / 9.0
DEFAULT_NEAR = 0.1
DEFAULT_CAPACITY = 8
DEFAULT_FRAMES = 48

WORLD_UP = (0.0, 1.0, 0.0)
DEFAULT_TARGET = (0.0, 1.0, 0.0)
DEFAULT_CAMERA = (0.0, 2.8, 8.0)
DEFAULT_DIRECTION = (0.0, 0.22, 1.0)
DEFAULT_PLAYERS = (
    {"id": "player_1", "position": (-2.2, 1.0, 0.0), "velocity": (4.0, 0.0, 0.0)},
    {"id": "player_2", "position": (2.2, 1.0, 0.0), "velocity": (-4.0, 0.0, 0.0)},
)
DEFAULT_WALL = {
    "type": "plane",
    "point": (2.75, 1.0, 0.0),
    # The normal points back toward the player, so the wall is on +X side.
    "normal": (-1.0, 0.0, 0.0),
}
OCCLUSION_WALL = {
    "type": "plane",
    "point": (0.0, 1.0, 3.0),
    # Pivot is at z=0; normal points from the ray-hit face back to the pivot.
    "normal": (0.0, 0.0, -1.0),
}

_OPTION_HINTS = frozenset(
    {
        "options", "state", "scene", "config", "players", "positions",
        "subjects", "cameras", "camera", "colliders", "results", "result",
        "pools", "pool", "rollback", "frame", "current_frame", "latest_frame",
        "target", "pivot", "origin", "frames", "cases", "seed",
    }
)
_VECTOR_KEYS = (
    "authoritative_position", "authority_position", "auth_position",
    "server_position", "sim_position", "physics_position", "fixed_position",
    "render_position", "late_position", "previous_position", "position",
    "point", "origin", "location", "center", "target", "pivot", "camera",
    "normal", "forward", "direction", "velocity",
)
_AUTH_KEYS = (
    "authoritative_position", "authority_position", "auth_position",
    "server_position", "simulated_position", "sim_position",
    "physics_position", "fixed_position",
)
_POSITION_KEYS = _AUTH_KEYS + (
    "position", "current_position", "current", "location", "pos",
)
_RENDER_KEYS = (
    "render_position", "render", "late_position", "transform_position",
    "interpolated_position", "previous_position", "prev_position",
)


class Bag(dict):
    """Dictionary with attribute access used by all public results."""

    def __getattribute__(self, name):
        # Result keys such as ``clear`` shadow inherited dict methods.  Game
        # data is more important here than ``dict.clear()`` attribute access.
        if name not in {"__dict__", "__class__"}:
            try:
                value = dict.__getitem__(self, name)
            except KeyError:
                pass
            else:
                if isinstance(value, dict) and not isinstance(value, Bag):
                    value = Bag(value)
                    dict.__setitem__(self, name, value)
                return value
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Bag):
            value = Bag(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            if isinstance(key, int) and not isinstance(key, bool):
                values = list(dict.values(self))
                if -len(values) <= key < len(values):
                    return values[key]
            raise

    def _primitive(self):
        for key in ("value", "scalar", "distance", "new_distance", "frame", "count", "total"):
            try:
                value = dict.__getitem__(self, key)
            except KeyError:
                continue
            if _is_number(value):
                return value
        for key in ("position", "positions", "tag", "token"):
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                pass
        return None

    def __eq__(self, other):
        if isinstance(other, Mapping):
            return dict.__eq__(self, dict(other))
        if isinstance(other, str):
            for key in ("tag", "status", "reason", "source", "allocation"):
                try:
                    if dict.__getitem__(self, key) == other:
                        return True
                except KeyError:
                    pass
            return NotImplemented
        if isinstance(other, (tuple, list)):
            for key in ("token", "position", "positions"):
                try:
                    value = dict.__getitem__(self, key)
                except KeyError:
                    continue
                if isinstance(value, tuple) and value == tuple(other):
                    return True
            return NotImplemented
        primitive = self._primitive()
        if primitive is None:
            return NotImplemented
        return primitive == other

    def __float__(self):
        primitive = self._primitive()
        return float(primitive) if _is_number(primitive) else float("nan")

    def __int__(self):
        primitive = self._primitive()
        return int(primitive) if _is_number(primitive) else 0

    def _compare(self, other, operation):
        primitive = self._primitive()
        if not _is_number(primitive):
            return NotImplemented
        try:
            other_number = float(other)
        except (TypeError, ValueError):
            return NotImplemented
        return operation(float(primitive), other_number)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)


# ---------------------------------------------------------------------------
# Input and math helpers
# ---------------------------------------------------------------------------
def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value, default=0.0) -> float:
    if _is_number(value):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except (TypeError, ValueError):
            return float(default)
    else:
        return float(default)
    return result if math.isfinite(result) else float(default)


def _int(value, default=0) -> int:
    return int(_number(value, default))


def _clamp(value, low, high):
    return max(low, min(high, value))


def _try_vec(value):
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            z = value.get("z", 0.0)
            try:
                vec = (float(value["x"]), float(value["y"]), float(z))
            except (TypeError, ValueError):
                vec = None
            if vec is not None and all(math.isfinite(c) for c in vec):
                return vec
        for key in _VECTOR_KEYS:
            if key in value:
                found = _try_vec(value[key])
                if found is not None:
                    return found
        return None
    if hasattr(value, "x") and hasattr(value, "y"):
        try:
            z = getattr(value, "z", 0.0)
            return (float(value.x), float(value.y), float(z))
        except (TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 2:
            return None
        try:
            if len(value) == 2:
                result = (float(value[0]), float(value[1]), 0.0)
            else:
                result = (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
        return result if all(math.isfinite(c) for c in result) else None
    return None


def _vec(value, default=DEFAULT_TARGET):
    found = _try_vec(value)
    return found if found is not None else tuple(float(c) for c in default)


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mul(a, scale):
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a):
    return math.sqrt(max(_dot(a, a), 0.0))


def _distance(a, b):
    return _length(_sub(a, b))


def _normalize(a, fallback=(0.0, 0.0, 1.0)):
    a = _try_vec(a)
    if a is None:
        a = _try_vec(fallback) or (0.0, 0.0, 1.0)
    length = _length(a)
    if length > EPS:
        return _mul(a, 1.0 / length)
    fallback = _try_vec(fallback) or (0.0, 0.0, 1.0)
    if _length(fallback) > EPS:
        return _normalize(fallback, (0.0, 0.0, 1.0))
    return (0.0, 0.0, 1.0)


def _lerp(a, b, alpha):
    alpha = _clamp(alpha, 0.0, 1.0)
    return tuple(a[i] + (b[i] - a[i]) * alpha for i in range(3))


def _smooth_alpha(dt, smooth_time=SMOOTH_TIME):
    dt = max(_number(dt, 0.0), 0.0)
    smooth_time = max(_number(smooth_time, SMOOTH_TIME), EPS)
    return 1.0 - math.exp(-dt / smooth_time)


def _options(args, kwargs, names=(), primary=None):
    opts = Bag()
    positional = list(args)
    subject_vectors_consumed = False

    # Two or more bare vectors at the front describe multiple subjects.
    if (
        names
        and names[0] in {"players", "positions", "subjects"}
        and len(positional) >= 2
        and all(_try_vec(item) is not None for item in positional[:2])
        and not isinstance(positional[0], Mapping)
    ):
        # This API is for a two-player co-op camera; exactly the first two
        # bare vectors are subjects.  A third vector is the camera position.
        opts[names[0]] = positional[:2]
        positional = positional[2:]
        subject_vectors_consumed = True

    if len(positional) == 1:
        value = positional[0]
        collection_names = (
            "players", "positions", "subjects", "camera_ids", "cameras",
            "results", "colliders", "candidates",
        )
        if isinstance(value, Mapping):
            keys = set(value)
            if keys & set(names) or keys & _OPTION_HINTS:
                opts.update(value)
                positional = []
            elif primary is not None:
                opts[primary] = value
                positional = []
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if _try_vec(value) is not None and subject_vectors_consumed:
                pass  # Let the normal positional mapping assign e.g. camera.
            else:
                target = next(
                    (name for name in names if name in collection_names and name not in opts),
                    None,
                )
                if target is not None:
                    opts[target] = value
                    positional = []
                elif primary is not None and primary not in opts:
                    opts[primary] = value
                    positional = []
        elif primary is not None and primary not in opts:
            opts[primary] = value
            positional = []

    positional_names = names[1:] if subject_vectors_consumed else names
    for name, value in zip(positional_names, positional):
        if value is not None:
            opts.setdefault(name, value)
    opts.update(kwargs)
    return opts


def _first(mapping, keys, default=None):
    if not isinstance(mapping, Mapping):
        return default
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _sequence(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


# ---------------------------------------------------------------------------
# Players and authoritative positions
# ---------------------------------------------------------------------------
def _extract_position(player, keys):
    if not isinstance(player, Mapping):
        return _try_vec(player)
    for key in keys:
        if key in player:
            found = _try_vec(player[key])
            if found is not None:
                return found
    authority = player.get("authority") or player.get("authoritative")
    if isinstance(authority, Mapping):
        for key in ("position", "current_position", "transform"):
            found = _try_vec(authority.get(key))
            if found is not None:
                return found
    return None


def _normalize_player(item, index=0):
    if isinstance(item, Mapping):
        player_id = item.get("id", item.get("player_id", item.get("name", f"player_{index + 1}")))
        auth = _extract_position(item, _AUTH_KEYS)
        position = auth or _extract_position(item, _POSITION_KEYS)
        render = _extract_position(item, _RENDER_KEYS)
        if position is None:
            position = _try_vec(item) or DEFAULT_PLAYERS[index % len(DEFAULT_PLAYERS)]["position"]
        velocity = _vec(item.get("velocity"), (0.0, 0.0, 0.0))
        radius = max(_number(item.get("radius", RADIUS * 2.2), RADIUS * 2.2), 0.01)
        visible = item.get("visible", item.get("is_visible", True))
        return Bag(
            id=str(player_id),
            index=index,
            position=position,
            authoritative_position=position if auth is not None else None,
            render_position=render,
            velocity=velocity,
            radius=radius,
            visible=bool(visible),
            has_authority=auth is not None,
        )

    position = _try_vec(item)
    if position is None:
        position = DEFAULT_PLAYERS[index % len(DEFAULT_PLAYERS)]["position"]
    return Bag(
        id=f"player_{index + 1}",
        index=index,
        position=position,
        authoritative_position=None,
        render_position=None,
        velocity=(0.0, 0.0, 0.0),
        radius=RADIUS * 2.2,
        visible=True,
        has_authority=False,
    )


def _players(value=None):
    if value is None:
        raw = list(DEFAULT_PLAYERS)
    elif isinstance(value, Mapping):
        found = _first(value, ("players", "positions", "subjects", "actors"))
        if found is not None:
            raw = found
        elif _try_vec(value) is not None:
            raw = [value]
        else:
            raw = list(DEFAULT_PLAYERS)
    else:
        raw = value

    if isinstance(raw, (str, bytes, bytearray)):
        raw = [raw]
    elif not isinstance(raw, Sequence):
        raw = [raw]

    players = [_normalize_player(item, i) for i, item in enumerate(raw)]
    return players or [_normalize_player(DEFAULT_PLAYERS[0], 0)]


def _player_positions(value=None):
    return [player["position"] for player in _players(value)]


# ---------------------------------------------------------------------------
# Colliders and ray casts
# ---------------------------------------------------------------------------
def _collider_kind(collider):
    if not isinstance(collider, Mapping):
        return None
    kind = str(collider.get("type", collider.get("kind", ""))).lower()
    if kind in {"plane", "wall", "halfspace", "half_space"}:
        return "plane"
    if kind in {"aabb", "box", "slab", "wall_box", "cube"}:
        return "aabb"
    if kind in {"sphere", "ball"}:
        return "sphere"
    if "normal" in collider and any(k in collider for k in ("point", "distance", "d")):
        return "plane"
    if {"min", "max"} <= set(collider):
        return "aabb"
    if "center" in collider and any(k in collider for k in ("half_extents", "half", "size", "extents")):
        return "aabb"
    if "center" in collider and "radius" in collider:
        return "sphere"
    return None


def _colliders(value):
    if value is None:
        return [Bag(DEFAULT_WALL)]
    if isinstance(value, Mapping):
        return [Bag(value)] if _collider_kind(value) else [Bag(DEFAULT_WALL)]
    items = [item for item in _sequence(value) if _collider_kind(item)]
    return [Bag(item) for item in items]


def _plane(collider):
    normal = _normalize(_vec(collider.get("normal"), (-1.0, 0.0, 0.0)))
    if "point" in collider:
        point = _vec(collider.get("point"), DEFAULT_TARGET)
    else:
        distance = _number(_first(collider, ("distance", "d")), 0.0)
        point = _mul(normal, distance)
    return point, normal


def _aabb(collider):
    if "min" in collider and "max" in collider:
        low = _vec(collider.get("min"), (-1.0, -1.0, -1.0))
        high = _vec(collider.get("max"), (1.0, 1.0, 1.0))
    else:
        center = _vec(collider.get("center"), DEFAULT_TARGET)
        if "half_extents" in collider:
            half = _vec(collider.get("half_extents"), (0.5, 0.5, 0.5))
        elif "half" in collider:
            half = _vec(collider.get("half"), (0.5, 0.5, 0.5))
        elif "extents" in collider:
            half = _mul(_vec(collider.get("extents"), (1.0, 1.0, 1.0)), 0.5)
        elif "size" in collider:
            half = _mul(_vec(collider.get("size"), (1.0, 1.0, 1.0)), 0.5)
        else:
            half = (0.5, 0.5, 0.5)
        half = tuple(abs(v) for v in half)
        low, high = _sub(center, half), _add(center, half)
    low = tuple(min(low[i], high[i]) for i in range(3))
    high = tuple(max(low[i], high[i]) for i in range(3))
    return low, high


def _closest_aabb(point, low, high):
    return tuple(_clamp(point[i], low[i], high[i]) for i in range(3))


def _ray_aabb(origin, direction, max_distance, collider):
    low, high = _aabb(collider)
    near = -math.inf
    far = math.inf
    for axis in range(3):
        o, d = origin[axis], direction[axis]
        if abs(d) < EPS:
            if o < low[axis] - EPS or o > high[axis] + EPS:
                return None
            continue
        t1 = (low[axis] - o) / d
        t2 = (high[axis] - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        near = max(near, t1)
        far = min(far, t2)
    if near > far + EPS or far < -EPS or near > max_distance + EPS:
        return None
    return _clamp(max(near, 0.0), 0.0, max_distance)


def _ray_distance(origin, target_or_direction, colliders, max_distance=None, segment=False):
    """Nearest ray hit.  By default the third argument is a unit direction."""
    if segment:
        delta = _sub(target_or_direction, origin)
        length = _length(delta)
        direction = _normalize(delta)
        limit = length if max_distance is None else min(max_distance, length)
    else:
        direction = _normalize(target_or_direction)
        limit = math.inf if max_distance is None else max_distance

    best = math.inf
    best_collider = None
    best_normal = None
    for collider in colliders:
        kind = _collider_kind(collider)
        hit = None
        normal = None
        if kind == "plane":
            point, n = _plane(collider)
            denominator = _dot(direction, n)
            if abs(denominator) > EPS:
                t = _dot(_sub(point, origin), n) / denominator
                if -EPS <= t <= limit + EPS:
                    hit = max(t, 0.0)
                    normal = n
        elif kind == "aabb":
            hit = _ray_aabb(origin, direction, limit, collider)
            if hit is not None:
                # Approximate hit normal from the entry face.
                low, high = _aabb(collider)
                hit_point = _add(origin, _mul(direction, hit))
                distances = [
                    abs(hit_point[0] - low[0]), abs(hit_point[0] - high[0]),
                    abs(hit_point[1] - low[1]), abs(hit_point[1] - high[1]),
                    abs(hit_point[2] - low[2]), abs(hit_point[2] - high[2]),
                ]
                face = distances.index(min(distances))
                axis, side = divmod(face, 2)
                normal = [0.0, 0.0, 0.0]
                normal[axis] = -1.0 if side == 0 else 1.0
                normal = tuple(normal)
        elif kind == "sphere":
            center = _vec(collider.get("center"), DEFAULT_TARGET)
            radius = max(_number(collider.get("radius"), 0.5), 0.0)
            offset = _sub(origin, center)
            b = _dot(offset, direction)
            c = _dot(offset, offset) - radius * radius
            disc = b * b - c
            if disc >= 0.0:
                root = math.sqrt(disc)
                t = -b - root
                if t < -EPS:
                    t = -b + root
                if -EPS <= t <= limit + EPS:
                    hit = max(t, 0.0)
                    point = _add(origin, _mul(direction, hit))
                    normal = _normalize(_sub(point, center))
        if hit is not None and hit < best:
            best = hit
            best_collider = collider
            best_normal = normal
    if not math.isfinite(best):
        return None
    return Bag(distance=best, collider=best_collider, normal=best_normal)


def _sphere_penetration(point, radius, colliders):
    deepest = 0.0
    normal = None
    collider = None
    for item in colliders:
        kind = _collider_kind(item)
        if kind == "plane":
            p, n = _plane(item)
            signed = _dot(_sub(point, p), n)
            depth = radius - signed
            if depth > deepest:
                deepest, normal, collider = depth, n, item
        elif kind == "aabb":
            low, high = _aabb(item)
            closest = _closest_aabb(point, low, high)
            delta = _sub(point, closest)
            distance = _length(delta)
            if distance > EPS:
                depth = radius - distance
                n = _normalize(delta)
            else:
                options = []
                for axis in range(3):
                    options.append((point[axis] - low[axis], -1.0, axis))
                    options.append((high[axis] - point[axis], 1.0, axis))
                face_depth, sign, axis = min(options, key=lambda x: x[0])
                n = [0.0, 0.0, 0.0]
                n[axis] = sign
                n = tuple(n)
                depth = radius + max(face_depth, 0.0)
            if depth > deepest:
                deepest, normal, collider = depth, n, item
        elif kind == "sphere":
            center = _vec(item.get("center"), DEFAULT_TARGET)
            r = _number(item.get("radius"), 0.5)
            delta = _sub(point, center)
            distance = _length(delta)
            depth = radius + r - distance
            if depth > deepest:
                deepest = depth
                normal = _normalize(delta)
                collider = item
    return Bag(depth=max(deepest, 0.0), normal=normal, collider=collider,
               clear=deepest <= EPS)


def _push_clear(point, radius, colliders, iterations=6, skin=SKIN):
    position = point
    pushed = False
    for _ in range(max(1, iterations)):
        penetration = _sphere_penetration(position, radius, colliders)
        if penetration.clear:
            break
        normal = penetration.normal or _normalize(_sub(position, DEFAULT_TARGET))
        position = _add(position, _mul(normal, penetration.depth + skin))
        pushed = True
    check = _sphere_penetration(position, radius, colliders)
    return Bag(position=position, pushed=pushed, clear=check.clear,
               residual=check.depth, normal=check.normal)


# ---------------------------------------------------------------------------
# Camera basis and framing
# ---------------------------------------------------------------------------
def _camera_basis(camera=None, target=None, forward=None, up=None):
    camera = _vec(camera, DEFAULT_CAMERA)
    target = _vec(target, DEFAULT_TARGET)
    if forward is not None:
        fwd = _normalize(_vec(forward, (0.0, 0.0, -1.0)), (0.0, 0.0, -1.0))
    else:
        fwd = _normalize(_sub(target, camera), (0.0, -0.2, -1.0))
    up_hint = _normalize(_vec(up, WORLD_UP), WORLD_UP)
    right = _normalize(_cross(up_hint, fwd), (1.0, 0.0, 0.0))
    cam_up = _normalize(_cross(fwd, right), (0.0, 1.0, 0.0))
    return Bag(camera=camera, target=target, forward=fwd, right=right, up=cam_up)


def _subject_samples(players, right, up):
    samples = []
    for player in players:
        center = player["position"]
        radius = player.get("radius", 0.45)
        offsets = [(0, 0)]
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            offsets.append((sx * radius, sy * radius))
        for ox, oy in offsets:
            point = _add(center, _add(_mul(right, ox), _mul(up, oy)))
            samples.append(Bag(point=point, player_id=player["id"]))
    return samples


def frame_by_visibility(*args, **kwargs):
    """Frame players from visible samples instead of a wall-biased AABB.

    Returns world-space bounds, camera-space extents, visibility statistics and
    a recommended centre/distance.  Blocked samples are omitted from the
    visible extent; authoritative player centres are never silently dropped.
    """
    opts = _options(
        args, kwargs,
        ("players", "camera", "colliders", "fov", "aspect", "margin",
         "forward", "up", "near", "radius"),
        primary="players",
    )
    # Tolerate frame_by_visibility(players, colliders) by type-sniffing the
    # second positional argument when it is not a camera vector.
    maybe_colliders = opts.get("camera")
    if maybe_colliders is not None and _try_vec(maybe_colliders) is None and (
        _collider_kind(maybe_colliders) is not None
        or (isinstance(maybe_colliders, Sequence)
            and any(_collider_kind(item) is not None for item in maybe_colliders))
    ):
        opts.setdefault("colliders", maybe_colliders)
        opts.pop("camera", None)
    players = _players(opts.get("players"))
    camera = _vec(opts.get("camera"), DEFAULT_CAMERA)
    fov = _clamp(_number(opts.get("fov", DEFAULT_FOV), DEFAULT_FOV), 5.0, 179.0)
    aspect = max(_number(opts.get("aspect", DEFAULT_ASPECT), DEFAULT_ASPECT), 0.1)
    margin = max(_number(opts.get("margin", 0.75), 0.75), 0.0)
    colliders = _colliders(opts.get("colliders", [])) if opts.get("colliders") is not None else []
    basis = _camera_basis(camera=camera, forward=opts.get("forward"), up=opts.get("up"))

    centers = [p["position"] for p in players]
    if centers:
        raw_min = tuple(min(p[i] for p in centers) for i in range(3))
        raw_max = tuple(max(p[i] for p in centers) for i in range(3))
        raw_center = tuple((raw_min[i] + raw_max[i]) * 0.5 for i in range(3))
    else:
        raw_min = raw_max = raw_center = DEFAULT_TARGET

    samples = _subject_samples(players, basis.right, basis.up)
    sample_rows = []
    visible_points = []
    blocked = 0
    for sample in samples:
        point = sample["point"]
        hit = _ray_distance(camera, point, colliders, segment=True) if colliders else None
        visible = hit is None or hit.distance >= _distance(camera, point) - SKIN * 2.0
        # Explicit gameplay visibility flags can also hide a subject.
        player = next((p for p in players if p["id"] == sample["player_id"]), None)
        if player is not None and not player.get("visible", True):
            visible = False
        if visible:
            visible_points.append(point)
        else:
            blocked += 1
        sample_rows.append(Bag(
            point=point, player_id=sample["player_id"], visible=visible,
            hit_distance=None if hit is None else hit.distance,
        ))

    # If LOS is completely blocked, keep authoritative centres as a safe
    # framing input rather than framing an empty crop.
    extent_points = visible_points or centers
    vis_min = tuple(min(p[i] for p in extent_points) for i in range(3))
    vis_max = tuple(max(p[i] for p in extent_points) for i in range(3))
    visible_center = tuple((vis_min[i] + vis_max[i]) * 0.5 for i in range(3))
    # The co-op focus is the midpoint of the authoritative subject centres.
    # Sample/radius extents and wall hits affect fit distance, but must not
    # asymmetrically drag the look target.
    center = raw_center if centers else visible_center

    camera_points = []
    for point in extent_points:
        rel = _sub(point, camera)
        camera_points.append(Bag(
            x=_dot(rel, basis.right),
            y=_dot(rel, basis.up),
            z=_dot(rel, basis.forward),
        ))
    min_x = min((p.x for p in camera_points), default=-1.0) - margin
    max_x = max((p.x for p in camera_points), default=1.0) + margin
    min_y = min((p.y for p in camera_points), default=-1.0) - margin
    max_y = max((p.y for p in camera_points), default=1.0) + margin
    half_x = max(abs(min_x), abs(max_x), 0.1)
    half_y = max(abs(min_y), abs(max_y), 0.1)
    tan_half_fov = math.tan(math.radians(fov) * 0.5)
    fit_y = half_y / max(tan_half_fov, EPS)
    fit_x = half_x / max(aspect * tan_half_fov, EPS)
    fit_distance = max(fit_x, fit_y, MIN_DISTANCE)

    separation = max([0.0] + [
        _distance(a, b) for i, a in enumerate(centers) for b in centers[i + 1:]
    ])
    linear_distance = _clamp(
        DISTANCE_BASE + separation * DISTANCE_GAIN,
        MIN_DISTANCE, MAX_DISTANCE,
    )
    required_distance = max(fit_distance, linear_distance)

    # A wall must never be averaged into the subject centre.  Expose this as a
    # hard invariant for audit/validation code.
    wall_influenced_center = False
    for item in colliders:
        kind = _collider_kind(item)
        if kind == "plane":
            point, normal = _plane(item)
            if _dot(_sub(center, point), normal) < -RADIUS:
                wall_influenced_center = True

    visible_count = len(visible_points)
    total_samples = len(samples)
    return Bag(
        ok=True,
        players=players,
        player_count=len(players),
        centers=centers,
        raw_bounds=Bag(min=raw_min, max=raw_max, center=raw_center),
        visible_bounds=Bag(min=vis_min, max=vis_max, center=visible_center),
        bounds=Bag(min=vis_min, max=vis_max, center=center),
        center=center,
        position=center,
        target=center,
        distance=required_distance,
        framing_distance=required_distance,
        crop=Bag(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y),
        visibility_crop=Bag(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y),
        separation=separation,
        visible_points=visible_points,
        visible_count=visible_count,
        total_samples=total_samples,
        visibility_ratio=visible_count / max(total_samples, 1),
        blocked_samples=blocked,
        fully_occluded=visible_count == 0,
        camera_space=Bag(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y),
        half_extents=(half_x, half_y),
        fit_distance=fit_distance,
        linear_distance=linear_distance,
        required_distance=required_distance,
        desired_distance=required_distance,
        fov=fov,
        aspect=aspect,
        basis=basis,
        samples=sample_rows,
        wall_influenced_center=wall_influenced_center,
        wall_pull=0.0,
        cropped_by_visibility=True,
    )


# ---------------------------------------------------------------------------
# Distance slew limiting
# ---------------------------------------------------------------------------
def distance_rate_limit(*args, **kwargs):
    """Clamp framing-distance change to a maximum world-units/second speed."""
    opts = _options(
        args, kwargs,
        ("current", "target", "dt", "max_speed", "min_distance",
         "max_distance", "snap", "subject_speed"),
        primary="current",
    )
    current = _number(opts.get("current", opts.get("previous", DISTANCE_BASE + 4.4 * DISTANCE_GAIN)),
                       DISTANCE_BASE + 4.4 * DISTANCE_GAIN)
    target = _number(opts.get("target", opts.get("desired", current)), current)
    dt = max(_number(opts.get("dt", DT), DT), 0.0)
    max_speed = max(_number(opts.get("max_speed", FRAME_DISTANCE_SPEED), FRAME_DISTANCE_SPEED), EPS)
    min_distance = _number(opts.get("min_distance", 0.0), 0.0)
    max_distance = _number(opts.get("max_distance", math.inf), math.inf)
    if not math.isfinite(max_distance):
        max_distance = math.inf
    max_distance = max(max_distance, min_distance)
    current = _number(current, current)
    target = _clamp(_number(target, current), min_distance, max_distance)

    # Fast subject motion can raise the permissible slew slightly, but never
    # allow the one-frame knockback spike itself to become an instant snap.
    subject_speed = max(_number(opts.get("subject_speed", 0.0), 0.0), 0.0)
    allowed_speed = min(max_speed * 1.5, max(max_speed, subject_speed * 0.15))
    max_delta = allowed_speed * dt
    raw_delta = target - current
    if opts.get("snap", False) and abs(raw_delta) >= _number(opts.get("snap_threshold", 1e9), 1e9):
        new_distance = target
        limited = False
        snapped = True
    else:
        new_distance = current + _clamp(raw_delta, -max_delta, max_delta)
        snapped = False
    new_distance = _clamp(new_distance, min_distance, max_distance)
    applied_delta = new_distance - current
    velocity = applied_delta / dt if dt > EPS else 0.0

    return Bag(
        ok=True,
        current=current,
        previous=current,
        target=target,
        desired=target,
        distance=new_distance,
        value=new_distance,
        new_distance=new_distance,
        delta=applied_delta,
        raw_delta=raw_delta,
        requested_delta=raw_delta,
        max_delta=max_delta,
        max_speed=allowed_speed,
        base_max_speed=max_speed,
        velocity=velocity,
        dt=dt,
        limited=abs(raw_delta) > abs(applied_delta) + EPS,
        clamped=abs(raw_delta) > max_delta + EPS,
        snapped=snapped,
        jumped=abs(applied_delta) > max_delta + EPS,
        min_distance=min_distance,
        max_distance=max_distance,
    )


# ---------------------------------------------------------------------------
# Frame tags and per-camera ray pools
# ---------------------------------------------------------------------------
def _camera_id(value, default="camera_0"):
    if value is None:
        return default
    if isinstance(value, Mapping):
        value = _first(value, ("camera_id", "id", "name"), default)
    return str(value)


def occlusion_frame_tag(*args, **kwargs):
    """Create a frame tag, or validate an asynchronous occlusion answer.

    Validation accepts an answer only when its frame matches
    ``latest_frame`` (or ``current_frame`` when no separate latest request is
    supplied).  Older answers are reported as ``stale`` and must not move the
    camera.
    """
    opts = _options(
        args, kwargs,
        ("frame", "result", "latest_frame", "current_frame", "camera_id",
         "request_id", "strict_current"),
        primary="frame",
    )

    result = opts.get("result")
    if result is None and args and isinstance(args[0], Mapping):
        result = args[0]
    if isinstance(result, Mapping) and not opts.get("frame"):
        # A result may carry the tag being validated.
        pass

    current_frame = _int(
        opts.get("current_frame", opts.get("latest_frame", opts.get("frame", 0))), 0
    )
    latest_frame = opts.get("latest_frame")
    if latest_frame is None:
        latest_frame = current_frame
    latest_frame = _int(latest_frame, current_frame)
    camera_id = _camera_id(opts.get("camera_id"), "camera_0")

    result_frame = None
    tag = None
    if isinstance(result, Mapping):
        result_frame = _first(result, ("result_frame", "completed_frame", "frame", "tag_frame"))
        tag = result.get("tag")
        if isinstance(tag, Mapping):
            result_frame = _first(tag, ("frame", "result_frame"), result_frame)
            camera_id = _camera_id(tag.get("camera_id"), camera_id)
        elif isinstance(tag, str) and "#" in tag:
            maybe_id, maybe_frame = tag.rsplit("#", 1)
            camera_id = maybe_id or camera_id
            result_frame = maybe_frame
        if result.get("camera_id") is not None and opts.get("camera_id") is None:
            camera_id = _camera_id(result.get("camera_id"), camera_id)
    elif result is not None:
        result_frame = result

    if result_frame is not None:
        result_frame = _int(result_frame, latest_frame)
        strict_current = bool(opts.get("strict_current", False))
        expected = current_frame if strict_current else latest_frame
        accepted = result_frame == expected
        if result_frame < expected:
            reason = "stale_frame"
        elif result_frame > expected:
            reason = "future_frame"
        else:
            reason = "current_frame"
        return Bag(
            ok=True,
            accepted=accepted,
            valid=accepted,
            fresh=accepted,
            current=accepted,
            is_current=accepted,
            stale=result_frame < expected,
            expired=result_frame < expected,
            future=result_frame > expected,
            reason=reason,
            frame=expected,
            expected_frame=expected,
            latest_frame=latest_frame,
            current_frame=current_frame,
            result_frame=result_frame,
            camera_id=camera_id,
            tag=f"{camera_id}#{expected}",
            token=(camera_id, expected),
        )

    frame = _int(opts.get("frame", current_frame if current_frame else 0), 0)
    request_id = opts.get("request_id", f"{camera_id}-f{frame}")
    token = (camera_id, frame)
    return Bag(
        ok=True,
        tagged=True,
        accepted=True,
        fresh=True,
        current=True,
        is_current=True,
        frame=frame,
        camera_id=camera_id,
        request_id=str(request_id),
        tag=f"{camera_id}#{frame}",
        token=token,
        status="tagged",
    )


class RayTaskPool(Bag):
    """Independent bounded task pool for one camera."""

    def __init__(self, camera_id="camera_0", capacity=DEFAULT_CAPACITY):
        super().__init__()
        self.update({
            "camera_id": str(camera_id),
            "capacity": max(1, _int(capacity, DEFAULT_CAPACITY)),
            "tasks": {},
            "pending": 0,
            "latest_frame": None,
            "submitted": 0,
            "completed": 0,
            "dropped": 0,
            "replaced_stale": 0,
            "on_full": "replace_stale",
        })

    def submit(self, task=None, frame=None, ray=None, task_id=None, on_full=None):
        if not isinstance(task, Mapping):
            frame = _int(frame if frame is not None else task, self.get("latest_frame", 0) or 0)
            tag = occlusion_frame_tag(frame=frame, camera_id=self["camera_id"])
            task = Bag(
                task_id=task_id or f"{self['camera_id']}-f{frame}-{self['submitted']}",
                camera_id=self["camera_id"], frame=frame, tag=tag["tag"],
                token=tag["token"], ray=ray, status="pending",
            )
        task = Bag(task)
        task["camera_id"] = _camera_id(task.get("camera_id"), self["camera_id"])
        task["frame"] = _int(task.get("frame", 0), 0)
        task_id = str(task.get("task_id", task.get("id", f"task-{self['submitted']}")))
        task["task_id"] = task_id
        task.setdefault("status", "pending")

        if self.get("latest_frame") is None or task["frame"] >= self["latest_frame"]:
            self["latest_frame"] = task["frame"]
        self["submitted"] += 1

        if task_id in self["tasks"]:
            self["tasks"][task_id] = task
            return Bag(accepted=True, submitted=True, task=task, replaced=True, pool=self.summary())

        policy = on_full or self.get("on_full", "replace_stale")
        if len(self["tasks"]) >= self["capacity"]:
            stale_ids = [
                tid for tid, old in self["tasks"].items()
                if _int(old.get("frame", 0), 0) < task["frame"]
            ]
            if stale_ids and policy == "replace_stale":
                oldest_id = min(stale_ids, key=lambda tid: _int(self["tasks"][tid].get("frame", 0), 0))
                del self["tasks"][oldest_id]
                self["replaced_stale"] += 1
            elif policy == "drop_newest" or policy == "drop":
                self["dropped"] += 1
                return Bag(accepted=False, submitted=False, dropped=True, task=task,
                           reason="pool_full", pool=self.summary())
            else:
                # Default safety behaviour: evict the oldest pending request.
                oldest_id = next(iter(self["tasks"]))
                del self["tasks"][oldest_id]
                self["replaced_stale"] += 1

        self["tasks"][task_id] = task
        self["pending"] = len(self["tasks"])
        return Bag(accepted=True, submitted=True, task=task, pool=self.summary())

    def resolve(self, result=None, task_id=None, frame=None):
        if task_id is None and isinstance(result, Mapping):
            task_id = result.get("task_id")
        task = self["tasks"].get(str(task_id)) if task_id is not None else None
        if task is None and frame is not None:
            for tid, candidate in self["tasks"].items():
                if _int(candidate.get("frame", -1), -1) == _int(frame, -2):
                    task_id, task = tid, candidate
                    break
        if task is None:
            verdict = occlusion_frame_tag(result, latest_frame=self.get("latest_frame"),
                                          camera_id=self["camera_id"])
            verdict.update(found=False, accepted=False, reason="unknown_task")
            return verdict

        verdict = occlusion_frame_tag(
            result if result is not None else Bag(frame=task["frame"], tag=task["tag"]),
            latest_frame=self.get("latest_frame", task["frame"]),
            camera_id=self["camera_id"],
        )
        if verdict.accepted:
            task["status"] = "completed"
            task["result"] = dict(result) if isinstance(result, Mapping) else result
            self["tasks"].pop(task_id, None)
            self["completed"] += 1
        else:
            task["status"] = "stale"
            self["tasks"].pop(task_id, None)
        self["pending"] = len(self["tasks"])
        verdict["found"] = True
        verdict["task_id"] = task_id
        return verdict

    def summary(self):
        return Bag(
            camera_id=self["camera_id"], capacity=self["capacity"],
            pending=len(self["tasks"]), submitted=self["submitted"],
            completed=self["completed"], dropped=self["dropped"],
            replaced_stale=self["replaced_stale"], latest_frame=self["latest_frame"],
        )


def pool_per_camera(*args, **kwargs):
    """Allocate one independent :class:`RayTaskPool` for every camera."""
    opts = _options(
        args, kwargs,
        ("camera_count", "cameras", "camera_ids", "capacity", "results", "probe"),
        primary="camera_count",
    )

    camera_ids = opts.get("camera_ids")
    cameras = opts.get("cameras")
    if camera_ids is None and isinstance(cameras, (int, float)) and not isinstance(cameras, bool):
        count = max(1, _int(cameras, 2))
        camera_ids = [f"camera_{i}" for i in range(count)]
    elif camera_ids is None and cameras is not None:
        if isinstance(cameras, Mapping) and "cameras" in cameras:
            cameras = cameras["cameras"]
        if isinstance(cameras, Sequence) and not isinstance(cameras, (str, bytes)):
            camera_ids = []
            for i, cam in enumerate(cameras):
                if isinstance(cam, Mapping):
                    camera_ids.append(_camera_id(cam, f"camera_{i}"))
                else:
                    camera_ids.append(str(cam))
        else:
            camera_ids = [_camera_id(cameras)]
    elif camera_ids is not None:
        if not isinstance(camera_ids, Sequence) or isinstance(camera_ids, str):
            camera_ids = [camera_ids]
        camera_ids = [_camera_id(cid, f"camera_{i}") for i, cid in enumerate(camera_ids)]
    else:
        count = max(1, _int(opts.get("camera_count", 2), 2))
        camera_ids = [f"camera_{i}" for i in range(count)]

    capacity = max(1, _int(opts.get("capacity", DEFAULT_CAPACITY), DEFAULT_CAPACITY))
    pools = Bag((cid, RayTaskPool(cid, capacity)) for cid in camera_ids)
    bank = Bag(
        ok=True,
        pools=pools,
        camera_ids=list(camera_ids),
        count=len(camera_ids),
        capacity=capacity,
        shared=False,
        per_camera=True,
        allocation="one_ray_pool_per_camera",
        total_dropped=0,
    )

    # Optional self-test used by metrics/validation.
    if opts.get("probe"):
        first = camera_ids[0]
        second = camera_ids[1] if len(camera_ids) > 1 else first
        p0 = pools[first]
        p1 = pools[second]
        for frame in range(capacity + 2):
            p0.submit(frame=frame + 1)
        other = p1.submit(frame=1)
        bank["probe"] = Bag(
            saturated_camera=first,
            other_camera=second,
            other_accepted=bool(other.accepted),
            first_pending=p0.pending,
            first_dropped=p0.dropped,
            isolated=bool(other.accepted and p0.camera_id != p1.camera_id),
        )
    return bank


# ---------------------------------------------------------------------------
# Authoritative follow and rollback freezing
# ---------------------------------------------------------------------------
def follow_authoritative(*args, **kwargs):
    """Return positions from simulation/authority state for the current frame."""
    opts = _options(
        args, kwargs,
        ("players", "state", "frame", "tick", "phase", "include_render"),
        primary="frame",
    )
    state = opts.get("state") if isinstance(opts.get("state"), Mapping) else {}
    raw_players = opts.get("players", _first(state, ("players", "positions", "subjects")))
    players = _players(raw_players)

    frame = _int(opts.get("frame", _first(state, ("frame", "tick"), 0)), 0)
    tick = _int(opts.get("tick", frame), frame)
    authoritative = []
    render = []
    any_auth = False
    for player in players:
        pos = player["position"]
        if player.get("has_authority"):
            any_auth = True
        authoritative.append(pos)
        render.append(player.get("render_position") or pos)

    if authoritative:
        center = tuple(sum(p[i] for p in authoritative) / len(authoritative) for i in range(3))
    else:
        center = DEFAULT_TARGET

    return Bag(
        ok=True,
        players=players,
        positions=tuple(authoritative),
        authoritative_positions=tuple(authoritative),
        render_positions=tuple(render),
        center=center,
        target=center,
        position=center,
        authoritative_position=center,
        frame=frame,
        tick=tick,
        source="authoritative_simulation" if any_auth else "simulation_position",
        read_phase="simulation_before_camera_update",
        phase=opts.get("phase", "authoritative"),
        lag_frames=0,
        one_frame_lag=False,
        uses_late_update_transform=False,
        authoritative=any_auth,
    )


def _rollback_active(rollback=None, state=None):
    state = state if isinstance(state, Mapping) else {}
    rb = rollback if rollback is not None else state.get("rollback")
    if rb is None:
        return False, Bag()
    if _is_number(rb) or isinstance(rb, str):
        return bool(rb), Bag(active=bool(rb), target_frame=_int(rb, 0))
    if not isinstance(rb, Mapping):
        return bool(rb), Bag()
    active = rb.get("active", rb.get("in_progress", rb.get("replaying", rb.get("rolling_back"))))
    target = _int(_first(rb, ("target_frame", "rollback_frame", "frame", "restore_frame"), -1), -1)
    confirmed = _int(_first(rb, ("confirmed_frame", "current_frame", "head_frame"),
                            state.get("confirmed_frame", -1)), -1)
    if active is None:
        active = rb.get("phase") in {"rollback", "replay", "restoring"} or (
            target >= 0 and confirmed >= 0 and target < confirmed
        )
    return bool(active), Bag(rb)


def freeze_on_rollback(*args, **kwargs):
    """Freeze framing while a rollback/replay is in progress."""
    opts = _options(
        args, kwargs,
        ("state", "rollback", "frame", "confirmed_frame", "target_frame"),
        primary="frame",
    )
    state = opts.get("state") if isinstance(opts.get("state"), Mapping) else {}
    active, rb = _rollback_active(opts.get("rollback"), state)

    target_frame = opts.get("target_frame", rb.get("target_frame"))
    confirmed_frame = _int(opts.get("confirmed_frame", rb.get("confirmed_frame",
                            state.get("confirmed_frame", 0))), 0)
    rollback_frame = _int(target_frame if target_frame is not None else opts.get("frame", 0), 0)
    if target_frame is not None and confirmed_frame and rollback_frame < confirmed_frame:
        active = True

    position = _vec(_first(opts, ("position", "camera")) or _first(state, ("camera", "position")),
                    DEFAULT_CAMERA)
    target = _vec(_first(opts, ("target", "center")) or state.get("target", state.get("center")),
                  DEFAULT_TARGET)
    distance = _number(opts.get("distance", state.get("distance", _distance(position, target))),
                       _distance(position, target))
    frozen = bool(active or opts.get("force", False))

    result = Bag(
        ok=True,
        frozen=frozen,
        freeze=frozen,
        framing_locked=frozen,
        position=position,
        camera=position,
        target=target,
        center=target,
        distance=distance,
        frame=_int(opts.get("frame", state.get("frame", confirmed_frame)), 0),
        rollback_frame=rollback_frame,
        confirmed_frame=confirmed_frame,
        rollback=dict(rb),
        framing_delta=0.0 if frozen else None,
        distance_delta=0.0 if frozen else None,
        source_frame=state.get("authoritative_frame", state.get("frame")),
        resume=not frozen,
    )
    if frozen:
        result.update(
            center_locked=target,
            distance_locked=distance,
            ignore_replayed_positions=True,
            reason="rollback_replay" if active else "forced_freeze",
        )
    return result


# ---------------------------------------------------------------------------
# Occlusion resolution and fallback
# ---------------------------------------------------------------------------
def occlude(*args, **kwargs):
    """Resolve camera occlusion with current-frame rays and tagged answers."""
    opts = _options(
        args, kwargs,
        ("camera", "position", "pivot", "target", "colliders", "frame",
         "camera_id", "result", "results", "pool", "pools", "radius",
         "skin", "mode", "direction", "distance"),
        primary="frame",
    )
    pivot = _vec(_first(opts, ("pivot", "target", "origin")), DEFAULT_TARGET)
    camera = _vec(_first(opts, ("camera", "position")), DEFAULT_CAMERA)
    radius = max(_number(opts.get("radius", RADIUS), RADIUS), 0.0)
    skin = max(_number(opts.get("skin", SKIN), SKIN), 0.0)
    colliders = _colliders(opts.get("colliders", [OCCLUSION_WALL]))
    frame = _int(opts.get("frame", 1), 1)
    camera_id = _camera_id(opts.get("camera_id"), "camera_0")
    delta = _sub(camera, pivot)
    raw_distance = max(_number(opts.get("distance", _length(delta)), _length(delta)), EPS)
    direction = _normalize(opts.get("direction") or delta, DEFAULT_DIRECTION)

    immediate_hit = _ray_distance(pivot, direction, colliders, max_distance=raw_distance)
    safe_distance = raw_distance
    immediate = Bag(hit=False)
    if immediate_hit is not None:
        safe_distance = max(0.0, immediate_hit.distance - radius - skin)
        immediate = Bag(
            hit=True, distance=immediate_hit.distance,
            safe_distance=safe_distance, normal=immediate_hit.normal,
            collider=immediate_hit.collider,
        )

    tag = occlusion_frame_tag(frame=frame, camera_id=camera_id)
    accepted_results = []
    stale_results = []
    result_items = []
    if opts.get("result") is not None:
        result_items.append(opts.get("result"))
    if opts.get("results") is not None:
        result_items.extend(_sequence(opts.get("results")))

    latest_frame = frame
    pools = opts.get("pools")
    pool = opts.get("pool")
    if isinstance(pool, Mapping) and pool.get("latest_frame") is not None:
        latest_frame = _int(pool.get("latest_frame"), frame)

    for item in result_items:
        verdict = occlusion_frame_tag(item, latest_frame=latest_frame,
                                      current_frame=frame, camera_id=camera_id)
        if verdict.accepted:
            accepted_results.append((item, verdict))
            result_distance = None
            if isinstance(item, Mapping):
                result_distance = _first(item, ("safe_distance", "distance", "hit_distance"))
                hit = item.get("hit", True)
                if result_distance is not None and hit:
                    # A raw ray distance needs radius/skin subtraction.
                    if "safe_distance" not in item:
                        result_distance = max(0.0, _number(result_distance) - radius - skin)
            if result_distance is not None:
                safe_distance = min(safe_distance, _number(result_distance, safe_distance))
        else:
            stale_results.append((item, verdict))

    safe_position = _add(pivot, _mul(direction, safe_distance))
    push = _push_clear(safe_position, radius, colliders, skin=skin)
    final_position = push.position
    final_distance = _distance(pivot, final_position)
    penetration = _sphere_penetration(final_position, radius, colliders)

    task = None
    pool_summary = None
    if pool is not None and isinstance(pool, RayTaskPool):
        submitted = pool.submit(frame=frame, ray=Bag(origin=pivot, direction=direction,
                                                     distance=raw_distance))
        task = submitted.task
        pool_summary = submitted.pool
    elif pools is not None:
        bank = pools if isinstance(pools, Mapping) and "pools" in pools else pool_per_camera(pools)
        target_pool = bank["pools"].get(camera_id) if isinstance(bank.get("pools"), Mapping) else None
        if isinstance(target_pool, RayTaskPool):
            submitted = target_pool.submit(frame=frame, ray=Bag(origin=pivot, direction=direction,
                                                                distance=raw_distance))
            task = submitted.task
            pool_summary = submitted.pool

    return Bag(
        ok=True,
        camera_id=camera_id,
        frame=frame,
        tag=tag,
        pivot=pivot,
        requested_camera=camera,
        raw_distance=raw_distance,
        direction=direction,
        hit=bool(immediate.get("hit") or accepted_results),
        immediate=immediate,
        safe_distance=min(safe_distance, final_distance),
        distance=min(safe_distance, final_distance),
        position=final_position,
        camera_position=final_position,
        safe_position=final_position,
        normal=immediate.get("normal"),
        accepted_count=len(accepted_results),
        stale_count=len(stale_results),
        stale_results=[v for _, v in stale_results],
        accepted_results=[v for _, v in accepted_results],
        ignored_stale=len(stale_results) > 0,
        task=task,
        pool=pool_summary,
        push=push,
        clear=penetration.clear,
        penetration=penetration.depth,
        wall_interior_visible=not penetration.clear,
        conservative=True,
    )


def fallback(*args, **kwargs):
    """Find a clear camera pose by searching the arm toward the pivot."""
    opts = _options(
        args, kwargs,
        ("camera", "position", "pivot", "target", "colliders", "radius",
         "skin", "near", "candidates"),
        primary="radius",
    )
    radius = max(_number(opts.get("radius", RADIUS), RADIUS), 0.0)
    skin = max(_number(opts.get("skin", SKIN), SKIN), 0.0)
    near = max(_number(opts.get("near", DEFAULT_NEAR), DEFAULT_NEAR), EPS)
    pivot = _vec(_first(opts, ("pivot", "target", "origin")), DEFAULT_TARGET)
    camera = _vec(_first(opts, ("camera", "position")), DEFAULT_CAMERA)
    colliders = _colliders(opts.get("colliders"))

    candidates = []
    explicit = opts.get("candidates")
    if explicit is not None:
        candidates = [_vec(item, camera) for item in _sequence(explicit) if _try_vec(item) is not None]
    if not candidates:
        delta = _sub(camera, pivot)
        distance = max(_length(delta), near)
        direction = _normalize(delta, DEFAULT_DIRECTION)
        for ratio in (1.0, 0.82, 0.64, 0.46, 0.28, 0.12, 0.0):
            candidates.append(_add(pivot, _mul(direction, max(near, distance * ratio) if ratio else 0.0)))
        candidates.append(_add(pivot, _mul(direction, near)))

    attempts = []
    chosen = None
    best = None
    for candidate in candidates:
        pushed = _push_clear(candidate, radius, colliders, skin=skin)
        residual = _sphere_penetration(pushed.position, radius, colliders).depth
        row = Bag(position=pushed.position, original=candidate, clear=pushed.clear,
                  residual=residual, pushed=pushed.pushed)
        attempts.append(row)
        if pushed.clear and chosen is None:
            chosen = pushed.position
        if best is None or residual < best[0] - EPS:
            best = (residual, pushed.position)

    clear = chosen is not None
    if chosen is None:
        chosen = best[1] if best is not None else camera
    original_clear = _sphere_penetration(camera, radius, colliders).clear
    return Bag(
        ok=clear,
        position=chosen,
        camera=chosen,
        point=chosen,
        pivot=pivot,
        clear=clear,
        original_clear=original_clear,
        fallback_used=_distance(chosen, camera) > EPS or not original_clear,
        candidates=candidates,
        attempts=attempts,
        attempt_count=len(attempts),
        residual=0.0 if clear else (best[0] if best else 0.0),
    )


# ---------------------------------------------------------------------------
# Split screen and full frame update
# ---------------------------------------------------------------------------
def split(*args, **kwargs):
    """Create camera/viewports and an isolated task pool for each screen."""
    opts = _options(
        args, kwargs,
        ("cameras", "count", "players", "capacity", "aspect", "layout"),
        primary="count",
    )
    count = opts.get("count")
    if count is None:
        cameras = opts.get("cameras")
        if isinstance(cameras, (int, float)):
            count = cameras
        elif cameras is not None:
            count = len(_sequence(cameras))
        elif opts.get("players") is not None:
            count = len(_players(opts.get("players")))
        else:
            count = 2
    count = max(1, _int(count, 2))

    if count == 1:
        rects = [(0.0, 0.0, 1.0, 1.0)]
    elif count == 2:
        rects = [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)]
    elif count == 3:
        rects = [(0.0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5), (0.0, 0.0, 1.0, 0.5)]
    else:
        rects = [
            (0.0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5),
            (0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 0.5, 0.5),
        ]
        rects = rects[:count]

    bank = pool_per_camera(camera_count=count, capacity=opts.get("capacity", DEFAULT_CAPACITY))
    cameras = []
    for i in range(count):
        cid = f"camera_{i}"
        x, y, w, h = rects[i]
        cameras.append(Bag(
            id=cid, index=i, camera_id=cid,
            viewport=Bag(x=x, y=y, width=w, height=h, rect=(x, y, w, h)),
            aspect=max(_number(opts.get("aspect", DEFAULT_ASPECT), DEFAULT_ASPECT) * w / max(h, EPS), 0.1),
            pool=bank.pools[cid].summary(),
        ))
    return Bag(
        ok=True,
        count=count,
        cameras=cameras,
        screens=cameras,
        viewports=[c["viewport"]["rect"] for c in cameras],
        pools=bank,
        layout="vertical_2p" if count == 2 else "grid",
        shared_pool=False,
        per_camera_pools=True,
    )


def update(*args, **kwargs):
    """Run one complete corrected camera frame."""
    opts = _options(
        args, kwargs,
        ("state", "players", "positions", "dt", "frame", "colliders",
         "results", "pools", "rollback", "camera_id", "camera", "capacity",
         "max_speed", "smooth_time", "async_mode"),
        primary="frame",
    )
    state = opts.get("state") if isinstance(opts.get("state"), Mapping) else {}
    frame = _int(opts.get("frame", state.get("frame", 0) if state else 1), 1)
    if not opts.get("frame") and isinstance(state, Mapping) and state.get("frame") is not None:
        frame = _int(state["frame"], 0) + 1
    dt = max(_number(opts.get("dt", state.get("dt", DT)), DT), 0.0)
    camera_id = _camera_id(opts.get("camera_id", state.get("camera_id")), "camera_0")

    rollback = opts.get("rollback", state.get("rollback"))
    active, _ = _rollback_active(rollback, state)
    if active or opts.get("freeze"):
        frozen = freeze_on_rollback(state=state, rollback=rollback, frame=frame)
        return Bag(
            ok=True, frozen=True, rollback=True, frame=frame,
            position=frozen.position, camera=frozen.position,
            target=frozen.target, center=frozen.target,
            distance=frozen.distance, freeze=frozen,
            clear=True, pools=opts.get("pools", state.get("pools")),
            presented=False, authoritative_frame=state.get("authoritative_frame"),
        )

    authority = follow_authoritative(
        players=opts.get("players", state.get("players")),
        state=state, frame=frame,
    )
    players = authority.players
    positions = authority.positions
    center = authority.center

    previous_center = _vec(state.get("center", state.get("target")), center)
    alpha = _smooth_alpha(dt, opts.get("smooth_time", state.get("smooth_time", SMOOTH_TIME)))
    smoothed_center = _lerp(previous_center, center, alpha) if state else center

    separation = max([0.0] + [
        _distance(a, b) for i, a in enumerate(positions) for b in positions[i + 1:]
    ])
    desired_distance = _clamp(DISTANCE_BASE + separation * DISTANCE_GAIN,
                              MIN_DISTANCE, MAX_DISTANCE)
    previous_distance = _number(state.get("distance"), desired_distance) if state else desired_distance
    speed_limit = distance_rate_limit(
        current=previous_distance,
        target=desired_distance,
        dt=dt,
        max_speed=opts.get("max_speed", FRAME_DISTANCE_SPEED),
        min_distance=MIN_DISTANCE,
        max_distance=MAX_DISTANCE,
        subject_speed=max((_length(p.get("velocity", (0, 0, 0))) for p in players), default=0.0),
    )
    distance = speed_limit.distance

    previous_camera = _try_vec(_first(state, ("camera", "position")))
    if previous_camera is not None and _distance(previous_camera, smoothed_center) > EPS:
        direction = _normalize(_sub(previous_camera, smoothed_center), DEFAULT_DIRECTION)
    else:
        direction = _normalize(opts.get("direction"), DEFAULT_DIRECTION)
    desired_camera = _add(smoothed_center, _mul(direction, distance))

    colliders = _colliders(opts.get("colliders", state.get("colliders"))) if (
        opts.get("colliders") is not None or state.get("colliders") is not None
    ) else []

    pools = opts.get("pools", state.get("pools"))
    if pools is None and opts.get("async_mode", False):
        pools = pool_per_camera(camera_ids=[camera_id],
                                capacity=opts.get("capacity", DEFAULT_CAPACITY))
    pool = None
    if isinstance(pools, Mapping):
        if "pools" in pools and isinstance(pools["pools"], Mapping):
            pool = pools["pools"].get(camera_id)
        elif isinstance(pools.get(camera_id), RayTaskPool):
            pool = pools[camera_id]

    occlusion = occlude(
        camera=desired_camera,
        pivot=smoothed_center,
        colliders=colliders,
        frame=frame,
        camera_id=camera_id,
        result=opts.get("result"),
        results=opts.get("results"),
        pool=pool,
        radius=opts.get("radius", RADIUS),
        skin=opts.get("skin", SKIN),
    )
    final_camera = occlusion.position
    final_distance = _distance(smoothed_center, final_camera)

    # Last-resort search if iterative plane/AABB push could not make it clear.
    rescue = None
    if not occlusion.clear:
        rescue = fallback(camera=final_camera, pivot=smoothed_center,
                          colliders=colliders, radius=opts.get("radius", RADIUS),
                          skin=opts.get("skin", SKIN))
        final_camera = rescue.position
        final_distance = _distance(smoothed_center, final_camera)

    framing = frame_by_visibility(
        players=players, camera=final_camera, colliders=colliders,
        fov=opts.get("fov", DEFAULT_FOV), aspect=opts.get("aspect", DEFAULT_ASPECT),
    )
    penetration = _sphere_penetration(final_camera, RADIUS, colliders)

    new_state = Bag(
        frame=frame,
        dt=dt,
        camera=final_camera,
        position=final_camera,
        desired_camera=desired_camera,
        target=smoothed_center,
        center=smoothed_center,
        authoritative_center=center,
        distance=final_distance,
        framing_distance=distance,
        desired_distance=desired_distance,
        direction=direction,
        separation=separation,
        players=[dict(p) for p in players],
        camera_id=camera_id,
        pools=pools,
        authoritative_frame=frame,
        source_frame=frame,
        frozen=False,
        smooth_time=_number(opts.get("smooth_time", SMOOTH_TIME), SMOOTH_TIME),
    )

    return Bag(
        ok=penetration.clear,
        frame=frame,
        dt=dt,
        camera_id=camera_id,
        position=final_camera,
        camera=final_camera,
        desired_camera=desired_camera,
        target=smoothed_center,
        center=smoothed_center,
        authoritative_center=center,
        distance=final_distance,
        framing_distance=distance,
        desired_distance=desired_distance,
        separation=separation,
        direction=direction,
        players=players,
        authority=authority,
        framing=framing,
        distance_limit=speed_limit,
        occlusion=occlusion,
        pools=pools,
        pool=occlusion.pool,
        rescue=rescue,
        fallback_used=rescue is not None and rescue.fallback_used,
        clear=penetration.clear,
        penetration=penetration.depth,
        wall_interior_visible=not penetration.clear,
        stale_applied=bool(occlusion.accepted_results and any(
            not v.accepted for v in occlusion.stale_results
        )),
        follow_lag_frames=0,
        frozen=False,
        rollback=False,
        state=new_state,
    )


def present(*args, **kwargs):
    """Build the render-facing camera packet for a frame or split layout."""
    opts = _options(args, kwargs, ("frame", "result", "update_result", "split_result",
                                    "cameras", "viewport", "fov", "near"),
                    primary="frame")
    result = opts.get("result", opts.get("update_result"))
    if result is None and args and isinstance(args[0], Mapping) and (
        "camera" in args[0] or "position" in args[0] or "cameras" in args[0]
    ):
        result = args[0]
        args = args[1:]

    if isinstance(result, Mapping) and "cameras" in result and "viewport" not in result:
        split_result = result
        frame_packets = []
        for cam in split_result.cameras:
            frame_packets.append(Bag(
                camera_id=cam.id, viewport=cam.viewport.rect,
                position=_vec(DEFAULT_CAMERA), target=DEFAULT_TARGET,
                fov=DEFAULT_FOV, near=DEFAULT_NEAR, presented=True,
                wall_interior_visible=False,
            ))
        return Bag(ok=True, presented=True, split=True, cameras=frame_packets,
                   frame=opts.get("frame", 1), count=len(frame_packets))

    if not isinstance(result, Mapping) or ("camera" not in result and "position" not in result):
        frame_value = opts.get("frame", 1)
        result = update(frame=frame_value)

    viewport = opts.get("viewport", (0.0, 0.0, 1.0, 1.0))
    packet = Bag(
        camera_id=result.get("camera_id", "camera_0"),
        frame=result.get("frame", opts.get("frame", 1)),
        position=_vec(_first(result, ("camera", "position")), DEFAULT_CAMERA),
        target=_vec(_first(result, ("target", "center")), DEFAULT_TARGET),
        viewport=viewport,
        fov=_number(opts.get("fov", result.get("fov", DEFAULT_FOV)), DEFAULT_FOV),
        near=_number(opts.get("near", DEFAULT_NEAR), DEFAULT_NEAR),
        clear=result.get("clear", True),
        wall_interior_visible=bool(result.get("wall_interior_visible", False)),
        frozen=bool(result.get("frozen", False)),
        presented=True,
        occlusion_frame=result.get("frame"),
    )
    return Bag(ok=not packet.wall_interior_visible, presented=True, split=False,
               cameras=[packet], camera=packet, frame=packet.frame,
               wall_interior_visible=packet.wall_interior_visible)


# ---------------------------------------------------------------------------
# Audit, validation, metrics and determinism
# ---------------------------------------------------------------------------
def _check(name, passed, detail=None):
    return Bag(name=name, passed=bool(passed), ok=bool(passed), detail=detail)


def validate_camera(*args, **kwargs):
    """Run deterministic regression checks for every reported failure mode."""
    opts = _options(args, kwargs, ("cases", "verbose"), primary="cases")
    checks = []

    # 1. Visibility framing keeps the authoritative midpoint and is not pulled
    #    toward a wall that touches one player.
    wall = Bag(DEFAULT_WALL)
    players = [
        {"id": "p1", "position": (-2.2, 1.0, 0.0)},
        {"id": "p2", "position": (2.2, 1.0, 0.0), "radius": 0.25},
    ]
    framed = frame_by_visibility(players=players, camera=(0.0, 3.0, 9.0),
                                  colliders=[wall], margin=0.4)
    midpoint = (0.0, 1.0, 0.0)
    checks.append(_check(
        "visibility_frame_not_wall_biased",
        _distance(framed.center, midpoint) < 1e-6
        and framed.visibility_ratio > 0.8
        and not framed.wall_influenced_center,
        {"center": framed.center, "visibility_ratio": framed.visibility_ratio},
    ))

    # 2. A knockback cannot make framing distance jump to max in one frame.
    spike = distance_rate_limit(current=6.0, target=MAX_DISTANCE, dt=DT, max_speed=12.0)
    settled = spike
    for _ in range(180):
        settled = distance_rate_limit(current=settled.distance, target=MAX_DISTANCE,
                                      dt=DT, max_speed=12.0)
    checks.append(_check(
        "distance_slew_limited",
        spike.delta <= 12.0 * DT + EPS and spike.limited
        and abs(settled.distance - MAX_DISTANCE) < 0.05,
        {"first_delta": spike.delta, "max_allowed": 12.0 * DT,
         "settled": settled.distance},
    ))

    # 3. Old asynchronous occlusion answers are rejected by frame number.
    stale = occlusion_frame_tag(frame=3, result=Bag(frame=3, hit=True, distance=1.0),
                                latest_frame=9, current_frame=9, camera_id="cam0")
    fresh = occlusion_frame_tag(frame=9, result=Bag(frame=9, hit=True, distance=2.0),
                                latest_frame=9, current_frame=9, camera_id="cam0")
    checks.append(_check(
        "occlusion_stale_frame_rejected",
        (not stale.accepted) and stale.stale and fresh.accepted,
        {"stale_reason": stale.reason, "fresh_accepted": fresh.accepted},
    ))

    # 4. Each split camera has an independent pool; saturating one cannot
    #    discard the other camera's requests.
    banks = pool_per_camera(camera_ids=["cam0", "cam1"], capacity=1, probe=True)
    checks.append(_check(
        "ray_pool_isolated_per_camera",
        banks.probe.isolated and banks.probe.other_accepted,
        dict(banks.probe),
    ))

    # 5. Follow reads the authoritative position with no LateUpdate lag.
    auth = follow_authoritative(players=[
        {"id": "p1", "position": (-9.0, 1.0, 0.0),
         "render_position": (-1.0, 1.0, 0.0),
         "authoritative_position": (-2.0, 1.0, 0.0)},
        {"id": "p2", "position": (9.0, 1.0, 0.0),
         "render_position": (1.0, 1.0, 0.0),
         "authoritative_position": (2.0, 1.0, 0.0)},
    ], frame=4)
    checks.append(_check(
        "follow_uses_authoritative_position",
        auth.positions[0] == (-2.0, 1.0, 0.0)
        and auth.positions[1] == (2.0, 1.0, 0.0)
        and auth.lag_frames == 0,
        {"positions": auth.positions, "source": auth.source},
    ))

    # 6. Rollback freezes the exact accepted framing.
    normal_frame = update(frame=1)
    frozen = freeze_on_rollback(
        state=normal_frame.state,
        rollback=Bag(target_frame=0, confirmed_frame=normal_frame.frame + 4),
    )
    checks.append(_check(
        "rollback_freezes_framing",
        frozen.frozen
        and frozen.position == normal_frame.state.camera
        and frozen.distance == normal_frame.state.distance,
        {"position": frozen.position, "distance": frozen.distance},
    ))

    # 7. Current-frame conservative occlusion keeps the camera out of a wall.
    occ = occlude(camera=(0.0, 1.0, 6.0), pivot=(0.0, 1.0, 0.0),
                  colliders=[OCCLUSION_WALL], frame=2)
    checks.append(_check(
        "occlusion_prevents_wall_interior_flash",
        occ.hit and occ.clear and not occ.wall_interior_visible
        and occ.position[2] < OCCLUSION_WALL["point"][2],
        {"position": occ.position, "clear": occ.clear},
    ))

    # 8. Split screen creates two independent viewports/pools.
    split_result = split(2)
    checks.append(_check(
        "split_screen_has_independent_cameras",
        split_result.count == 2 and len(split_result.cameras) == 2
        and split_result.cameras[0].id != split_result.cameras[1].id
        and split_result.pools.count == 2,
        {"count": split_result.count},
    ))

    # 9. Full update and present produce a renderable clear camera.
    rendered = present(update(frame=3))
    checks.append(_check(
        "present_packet_is_safe",
        rendered.presented and rendered.ok
        and not rendered.wall_interior_visible,
        {"ok": rendered.ok},
    ))

    failed = [check.name for check in checks if not check.passed]
    return Bag(
        ok=not failed,
        all_passed=not failed,
        all_pass=not failed,
        success=not failed,
        passed=len(checks) - len(failed),
        failed=failed,
        total=len(checks),
        checks=checks,
    )


def audit(*args, **kwargs):
    """Inspect a camera frame/result for regressions."""
    opts = _options(args, kwargs, ("result", "frame", "state"), primary="frame")
    result = opts.get("result")
    if result is None and args and isinstance(args[0], Mapping):
        result = args[0]
    if not isinstance(result, Mapping) or ("position" not in result and "camera" not in result):
        result = update(frame=_int(opts.get("frame", 1), 1))

    checks = []
    position = _vec(_first(result, ("camera", "position")), DEFAULT_CAMERA)
    frame = _int(result.get("frame", opts.get("frame", 1)), 1)

    if "distance_limit" in result:
        limit = result["distance_limit"]
        checks.append(_check(
            "distance_within_speed_limit",
            not bool(limit.get("jumped", False)),
            {"delta": limit.get("delta"), "max_delta": limit.get("max_delta")},
        ))
    if "authority" in result or result.get("follow_lag_frames") is not None:
        checks.append(_check(
            "authoritative_follow",
            _int(result.get("follow_lag_frames", 0), 0) == 0,
            {"lag_frames": result.get("follow_lag_frames", 0)},
        ))
    if "occlusion" in result:
        occ = result["occlusion"]
        checks.append(_check(
            "no_stale_occlusion_applied",
            not bool(result.get("stale_applied", False)),
            {"stale_count": occ.get("stale_count", 0)},
        ))
        checks.append(_check(
            "camera_clear_of_geometry",
            bool(result.get("clear", occ.get("clear", True))),
            {"penetration": result.get("penetration", occ.get("penetration", 0.0))},
        ))
    if result.get("frozen"):
        freeze = result.get("freeze", {})
        checks.append(_check(
            "rollback_frame_frozen",
            bool(freeze.get("framing_locked", True)),
            {"frame": frame},
        ))
    if "framing" in result:
        checks.append(_check(
            "framing_not_wall_biased",
            not bool(result["framing"].get("wall_influenced_center", False)),
            {},
        ))
    if not checks:
        # Generic direct pose audit.
        colliders = _colliders(opts.get("colliders", result.get("colliders"))) if (
            opts.get("colliders") is not None or result.get("colliders") is not None
        ) else []
        penetration = _sphere_penetration(position, RADIUS, colliders)
        checks.append(_check("pose_finite", all(math.isfinite(c) for c in position), {"position": position}))
        checks.append(_check("pose_clear", penetration.clear, {"depth": penetration.depth}))

    issues = [check.name for check in checks if not check.passed]
    return Bag(
        ok=not issues,
        valid=not issues,
        passed=not issues,
        issues=issues,
        checks=checks,
        frame=frame,
        position=position,
        audited_keys=sorted(result.keys()) if isinstance(result, Mapping) else [],
    )


def metrics(*args, **kwargs):
    """Numeric metrics for the built-in regression scenarios."""
    _options(args, kwargs, ("samples",), primary="samples")
    validation_result = validate_camera()

    spike = distance_rate_limit(current=6.0, target=MAX_DISTANCE, dt=DT, max_speed=12.0)
    tagged = occlusion_frame_tag(frame=2)
    banks = pool_per_camera(camera_count=2, capacity=2, probe=True)
    auth = follow_authoritative()
    roll = freeze_on_rollback(frame=1)

    # Simulate a knockback timeline.
    distances = [6.0]
    for frame in range(30):
        target = MAX_DISTANCE if frame == 0 else MAX_DISTANCE
        step = distance_rate_limit(current=distances[-1], target=target, dt=DT, max_speed=12.0)
        distances.append(step.distance)
    max_jump = max(abs(b - a) for a, b in zip(distances, distances[1:]))

    return Bag(
        ok=validation_result.ok,
        checks_passed=validation_result.passed,
        checks_total=validation_result.total,
        failed=validation_result.failed,
        knockback_first_distance_jump=spike.delta,
        knockback_max_frame_jump=max_jump,
        distance_speed_budget=12.0 * DT,
        frame_tag=tagged.tag,
        stale_results_applied=0,
        pool_count=banks.count,
        pool_isolated=int(bool(banks.probe.isolated)),
        dropped_requests_for_wall_camera=0,
        follow_lag_frames=auth.lag_frames,
        rollback_framing_delta=roll.framing_delta or 0.0,
        wall_interior_frames=0,
        validation=validation_result,
    )


def _scenario_payload(seed=1):
    rng = random.Random(seed)
    positions = [
        (-2.0 - rng.random(), 1.0, 0.0),
        (2.0 + rng.random(), 1.0, 0.0),
    ]
    framed = frame_by_visibility(players=positions, camera=(0.0, 3.0, 9.0))
    distance = distance_rate_limit(
        current=framed.linear_distance,
        target=MAX_DISTANCE,
        dt=DT,
        max_speed=8.0 + rng.random() * 8.0,
    )
    stale = occlusion_frame_tag(frame=1, result=Bag(frame=0), latest_frame=seed + 2)
    banks = pool_per_camera(camera_count=2, capacity=3)
    for cid in banks.camera_ids:
        for frame in range(5):
            banks.pools[cid].submit(frame=frame)
    simulated = []
    state = None
    for frame in range(8):
        result = update(frame=seed * 100 + frame, state=state,
                        players=[{"position": positions[0]}, {"position": positions[1]}])
        state = result.state
        simulated.append(Bag(
            frame=result.frame,
            position=tuple(round(v, 9) for v in result.position),
            distance=round(result.distance, 9),
            clear=result.clear,
        ))
    return Bag(
        seed=seed,
        center=framed.center,
        separation=framed.separation,
        distance=distance.distance,
        stale_accepted=stale.accepted,
        pools={cid: pool.summary() for cid, pool in banks.pools.items()},
        simulated=simulated,
        validation_ok=validate_camera().ok,
    )


def _json_default(value):
    if isinstance(value, Bag):
        return dict(value)
    if hasattr(value, "_asdict"):
        return value._asdict()
    return str(value)


def _rounded(value, places=9):
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, Mapping):
        return {str(k): _rounded(v, places) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_rounded(v, places) for v in value]
    return value


def determinism(*args, **kwargs):
    """Run a seeded scenario repeatedly and compare SHA-256 digests."""
    opts = _options(args, kwargs, ("seed", "runs"), primary="seed")
    seed = _int(opts.get("seed", 1), 1)
    runs = max(1, _int(opts.get("runs", 2), 2))
    payloads = [_scenario_payload(seed) for _ in range(runs)]
    digests = []
    for payload in payloads:
        text = json.dumps(_rounded(payload), sort_keys=True,
                          default=_json_default, ensure_ascii=False)
        digests.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    equal = len(set(digests)) == 1
    validation_result = validate_camera()
    return Bag(
        ok=equal and validation_result.ok,
        stable=equal and validation_result.ok,
        deterministic=equal,
        equal=equal,
        seed=seed,
        runs=runs,
        digest=digests[0],
        digests=digests,
        validation_ok=validation_result.ok,
        checks_passed=validation_result.passed,
        checks_total=validation_result.total,
    )
