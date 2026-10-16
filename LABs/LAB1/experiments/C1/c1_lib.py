# -*- coding: utf-8 -*-
"""C1 共享纯函数：prompt 拼接、回答分层解析、大圆距离、周期经度回绕。

协议来源：指南3 §4.1（同学参考报告，非官方）——原问题句追加固定指令：
    {CSV sentence}
    Output only JSON with numeric latitude and longitude in decimal degrees.
    Answer:

解析分层口径（C1 方案 §2.3/§5.1）：
    tier ∈ {json, embedded, regex, out_of_range, failed}
    - json/embedded/regex：解析成功且两坐标有限、在 [-90,90]/[-180,180] 内（按首次命中的层级命名）；
    - out_of_range：解析成功但越界或非有限（保留 raw 值）；
    - failed：三层全部失败（不猜裸数字，notes 记录文本中前两个数字）。
不做符号纠正、不交换 lat/lon、不自动回绕；周期口径只在评估层计算。
"""
import json
import math
import re

import numpy as np

PROMPT_TEMPLATE = (
    "{sentence}\n"
    "Output only JSON with numeric latitude and longitude in decimal degrees.\n"
    "Answer:"
)

_LAT_KEYS = {"lat", "latitude"}
_LON_KEYS = {"lon", "lng", "long", "longitude"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
# 键值兜底：只允许标点/空白作为键与数字之间的间隔（不允许字母），避免把邻近键的数值误配
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_LAT_RE = re.compile(r"(?i)(?<![a-z0-9])(?:latitude|lat)(?![a-z0-9])[^a-z0-9+\-]*?(" + _NUM + r")")
_LON_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:longitude|long|lng|lon)(?![a-z0-9])[^a-z0-9+\-]*?(" + _NUM + r")"
)
_ANY_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

HAVERSINE_R_KM = 6371.0


def build_prompt(sentence: str) -> str:
    return PROMPT_TEMPLATE.format(sentence=sentence)


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _extract_latlon(obj):
    if not isinstance(obj, dict):
        return None
    lat = lon = None
    for key, value in obj.items():
        if not isinstance(key, str):
            continue
        key_lower = key.strip().lower()
        number = _as_number(value)
        if number is None:
            continue
        if key_lower in _LAT_KEYS and lat is None:
            lat = number
        elif key_lower in _LON_KEYS and lon is None:
            lon = number
    if lat is None or lon is None:
        return None
    return lat, lon


def _loads_dict(text):
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _iter_json_objects(text):
    """花括号平衡扫描，逐个产出可解析的 JSON 对象（自首个 '{' 起）。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    obj = _loads_dict(text[start : index + 1])
                    if obj is not None:
                        yield obj
                    break
        start = text.find("{", start + 1)


def parse_output(raw: str) -> dict:
    """分层解析一条原始回答，返回 tier/lat_raw/lon_raw/in_range/pred_lat/pred_lon/notes。"""
    text = (raw or "").strip()
    lat = lon = None
    tier = None

    candidate = text
    fence = _FENCE_RE.search(text)
    if fence and fence.group(1).strip().startswith("{"):
        candidate = fence.group(1).strip()

    pair = _extract_latlon(_loads_dict(candidate))
    if pair is not None:
        tier = "json"
        lat, lon = pair
    else:
        for obj in _iter_json_objects(candidate):
            pair = _extract_latlon(obj)
            if pair is not None:
                tier = "embedded"
                lat, lon = pair
                break

    if tier is None:
        lat_match = _LAT_RE.search(text)
        lon_match = _LON_RE.search(text)
        if lat_match and lon_match:
            tier = "regex"
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))

    if tier is None:
        first_numbers = _ANY_NUM_RE.findall(text)[:2]
        return {
            "tier": "failed",
            "lat_raw": None,
            "lon_raw": None,
            "in_range": False,
            "pred_lat": None,
            "pred_lon": None,
            "notes": f"first_numbers={first_numbers}",
        }

    finite = math.isfinite(lat) and math.isfinite(lon)
    in_range = finite and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
    if not in_range:
        return {
            "tier": "out_of_range",
            "lat_raw": lat,
            "lon_raw": lon,
            "in_range": False,
            "pred_lat": None,
            "pred_lon": None,
            "notes": "",
        }
    return {
        "tier": tier,
        "lat_raw": lat,
        "lon_raw": lon,
        "in_range": True,
        "pred_lat": lat,
        "pred_lon": lon,
        "notes": "",
    }


def wrap_lon_deg(delta):
    """把经度差回绕到 [-180, 180)。"""
    return (np.asarray(delta, dtype=np.float64) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    """大圆距离（km，R=6371.0）。输入为十进制度，北纬/东经为正。"""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return HAVERSINE_R_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def coord_sq_error(pred_lat, pred_lon, true_lat, true_lon, periodic_lon=False):
    """课程口径单条平方误差：两坐标平方误差平均；periodic_lon=True 时经度差先回绕。"""
    dlat = np.asarray(pred_lat, dtype=np.float64) - np.asarray(true_lat, dtype=np.float64)
    dlon = np.asarray(pred_lon, dtype=np.float64) - np.asarray(true_lon, dtype=np.float64)
    if periodic_lon:
        dlon = wrap_lon_deg(dlon)
    return (dlat**2 + dlon**2) / 2.0
