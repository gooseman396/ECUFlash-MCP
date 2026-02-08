from __future__ import annotations

from dataclasses import dataclass
import ast
import csv
from datetime import datetime, timezone
import fnmatch
from functools import lru_cache
import json
import os
import shutil
import struct
from typing import Iterable
import re
import xml.etree.ElementTree as ET

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EvoX_Calibration_Bridge")

ROOT_DIR = os.path.dirname(__file__)
DEFAULT_ROM_PATH = os.path.join(ROOT_DIR, "Roms")
DEFAULT_OUTPUT_ROM_PATH = os.path.join(DEFAULT_ROM_PATH, "modified")
DEFAULT_LOG_PATH = os.path.join(ROOT_DIR, "logs")
DEFAULT_XML_PATH = r"C:\Program Files (x86)\OpenECU\EcuFlash\rommetadata\mitsubishi\evo"

ROM_PATH = os.getenv("EVO_ROM_PATH", DEFAULT_ROM_PATH)
OUTPUT_ROM_PATH = os.getenv(
    "EVO_OUTPUT_ROM_PATH",
    DEFAULT_OUTPUT_ROM_PATH,
)
LOG_PATH = os.getenv(
    "EVO_LOG_PATH",
    DEFAULT_LOG_PATH,
)
XML_PATHS = os.getenv(
    "EVO_XML_PATHS",
    DEFAULT_XML_PATH,
).split(";")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "mcp_config.json")
CONFIG_PATH = os.getenv("EVO_CONFIG_PATH", DEFAULT_CONFIG_PATH)
CONFIG_STRICT = os.getenv("EVO_CONFIG_STRICT", "false").strip().lower() in ("true", "1", "yes")
CONFIG_WARNINGS: list[str] = []
CONFIG_ERRORS: list[str] = []
TABLE_ALLOWLIST_BASE = [
    name.strip().lower()
    for name in os.getenv("EVO_TABLE_ALLOWLIST", "").split(",")
    if name.strip()
]
ACTIVE_ALLOWLIST_PROFILE = os.getenv("EVO_ALLOWLIST_PROFILE", "").strip().lower() or None
REQUIRE_BACKUP = os.getenv("EVO_REQUIRE_BACKUP", "true").strip().lower() in ("true", "1", "yes")
DEFAULT_USE_MAX_DELTA = os.getenv("EVO_DEFAULT_USE_MAX_DELTA", "false").strip().lower() in (
    "true",
    "1",
    "yes",
)

ALLOWLIST_PROFILES: dict[str, list[str]] = {
    "launch": ["*launch*"],
    "boost": ["*boost*", "*wgdc*", "*wgd*", "*wastegate*"],
    "fuel": ["*fuel*", "*afr*"],
    "timing": ["*timing*"],
    "mivec": ["*mivec*"],
    "rev": ["*rev*", "*rpm*"],
}

COMMON_TABLE_QUERIES: dict[str, list[str]] = {
    "launch": [
        "Launch High Octane Timing Map",
        "Launch High Octane Fuel Map",
        "Stationary Rev Limiter",
        "Maximum Speed permitted for Launch Maps",
        "Minimum TPS required for NLTS or Launch Maps",
    ],
    "boost": [
        "Boost Target",
        "WGDC",
        "Wastegate",
        "Boost Cut",
        "RAX Direct Boost Baro Comp",
    ],
    "fuel": [
        "High Octane Fuel Map",
        "Low Octane Fuel Map",
        "Injector Scaling",
        "Fuel Pressure",
    ],
    "timing": [
        "High Octane Timing Map",
        "Low Octane Timing Map",
        "Timing",
    ],
    "mivec": ["MIVEC"],
    "rev": ["Rev Limiter", "Stationary Rev Limiter"],
}

GUARDRAIL_PROFILES: dict[str, dict[str, object]] = {
    "street": {
        "max_delta": 2.0,
        "low_rpm_rpm_threshold": 3500.0,
        "low_rpm_boost_cap": 12.0,
        "boost_target_caps_by_gear": {"1": 14.0, "2": 16.0, "3": 18.0},
    },
    "track": {
        "max_delta": 3.0,
        "low_rpm_rpm_threshold": 3500.0,
        "low_rpm_boost_cap": 16.0,
        "boost_target_caps_by_gear": {"1": 16.0, "2": 18.0, "3": 20.0},
    },
    "drag": {
        "max_delta": 4.0,
        "low_rpm_rpm_threshold": 3500.0,
        "low_rpm_boost_cap": 18.0,
        "boost_target_caps_by_gear": {"1": 18.0, "2": 20.0, "3": 22.0},
    },
}

LOG_SIGNAL_SETS: dict[str, list[str]] = {
    "launch": ["rpm", "tps", "speed", "boost", "afr", "timing", "knock", "load", "time"],
    "boost": ["rpm", "tps", "boost", "target_boost", "wgdc", "gear", "iat", "load"],
    "fuel": ["rpm", "tps", "afr", "stft", "ltft", "ipw", "idc", "fuel_pressure", "load"],
    "timing": ["rpm", "tps", "timing", "knock", "iat", "load", "boost"],
}

LAUNCH_HEALTH_SETTINGS: dict[str, float] = {
    "timing_floor": -5.0,
    "boost_oscillation_threshold": 5.0,
    "knock_threshold": 0.0,
}

TORQUE_RISK_SETTINGS: dict[str, float] = {
    "rpm_threshold": 3500.0,
    "boost_threshold": 18.0,
    "timing_floor": 0.0,
    "knock_threshold": 0.0,
}

TUNING_USECASES: dict[str, dict[str, object]] = {
    "boost_targeting": {
        "category": "boost",
        "description": "Set boost targets and verify load/RPM axes alignment.",
        "queries": ["Boost Target", "Boost Target Engine Load", "Boost Target (High)", "Boost Target (Low)"],
        "required_log_signals": ["rpm", "boost", "target_boost", "wgdc", "tps", "gear", "iat", "load"],
        "notes": ["Check for oscillation in low gears before raising targets."],
    },
    "boost_control": {
        "category": "boost",
        "description": "Validate WGDC control maps and limits against targets.",
        "queries": ["WGDC", "Wastegate", "Boost Error", "Boost Cut", "RAX Direct Boost"],
        "required_log_signals": ["rpm", "boost", "target_boost", "wgdc", "gear", "iat", "load"],
        "notes": ["Confirm direct-boost tables align with target tables if enabled."],
    },
    "boost_safety": {
        "category": "boost",
        "description": "Confirm boost limits, cut tables, and compensation maps.",
        "queries": ["Boost Cut", "Baro Comp", "IAT Comp", "Boost Limit"],
        "required_log_signals": ["rpm", "boost", "iat", "baro", "gear", "load"],
        "notes": ["Ensure compensations do not exceed turbo efficiency."],
    },
    "fuel_wot": {
        "category": "fuel",
        "description": "Shape WOT fuel tables for target AFR under boost.",
        "queries": ["High Octane Fuel Map", "Low Octane Fuel Map", "Target AFR", "Open Loop Fuel"],
        "required_log_signals": ["rpm", "afr", "boost", "tps", "ipw", "idc", "fuel_pressure", "load"],
        "notes": ["Verify IDC headroom and pressure stability."],
    },
    "fuel_transient": {
        "category": "fuel",
        "description": "Check transient fueling and tip-in enrichment.",
        "queries": ["Transient Fuel", "Tip-in", "Accel Enrich"],
        "required_log_signals": ["rpm", "tps", "afr", "stft", "ltft", "ipw", "load"],
        "notes": ["Look for lean spikes on throttle transitions."],
    },
    "injector_setup": {
        "category": "fuel",
        "description": "Confirm injector scaling and latency values.",
        "queries": ["Injector Scaling", "Injector Latency", "Fuel Pressure"],
        "required_log_signals": ["rpm", "stft", "ltft", "ipw", "idc", "fuel_pressure"],
        "notes": ["Scaling errors show up in trims across load ranges."],
    },
    "timing_wot": {
        "category": "timing",
        "description": "Shape WOT timing for power while managing knock.",
        "queries": ["High Octane Timing Map", "Low Octane Timing Map", "Timing", "Knock"],
        "required_log_signals": ["rpm", "timing", "knock", "boost", "iat", "load"],
        "notes": ["Prioritize knock-free timing in peak torque regions."],
    },
    "timing_spool": {
        "category": "timing",
        "description": "Use timing to help spool without excessive EGTs.",
        "queries": ["Timing", "Launch High Octane Timing Map"],
        "required_log_signals": ["rpm", "timing", "boost", "iat", "egt", "load"],
        "notes": ["Avoid large timing drops that create launch bog."],
    },
    "timing_knock_response": {
        "category": "timing",
        "description": "Review knock response, correction, and limits.",
        "queries": ["Knock", "Knock Sum", "Knock Threshold", "Knock Retard"],
        "required_log_signals": ["rpm", "knock", "timing", "iat", "load", "boost"],
        "notes": ["Confirm consistent recovery after knock events."],
    },
}

TABLE_SIZE_OVERRIDES: dict[str, tuple[int, int]] = {
    "high octane timing map": (23, 22),
    "low octane timing map": (23, 22),
    "launch high octane timing map": (23, 22),
    "high octane fuel map": (16, 21),
    "low octane fuel map": (16, 21),
    "launch high octane fuel map": (16, 21),
}

TABLE_SIZE_PATTERNS: list[tuple[re.Pattern[str], tuple[int, int]]] = [
    (re.compile(r"alternate #\d+ high octane timing map", re.IGNORECASE), (23, 22)),
    (re.compile(r"alternate #\d+ high octane fuel map", re.IGNORECASE), (16, 21)),
    (re.compile(r"boost target engine load", re.IGNORECASE), (9, 18)),
    (re.compile(r"boost target", re.IGNORECASE), (9, 18)),
    (re.compile(r"wgdc map", re.IGNORECASE), (9, 18)),
    (re.compile(r"timing map", re.IGNORECASE), (23, 22)),
    (re.compile(r"fuel map", re.IGNORECASE), (16, 21)),
]

AXIS_ADDRESS_REGISTRY: dict[int, dict[str, object]] = {
    0x6286A: {"elements": 9, "scaling": "Throttle_Main - Stored Minimum Throttle %"},
    0x627E8: {"elements": 18, "scaling": "RPM"},
    0x6112C: {"elements": 21, "scaling": "Load"},
    0x61106: {"elements": 16, "scaling": "RPM"},
    0x611A6: {"elements": 22, "scaling": "Load"},
    0x61172: {"elements": 23, "scaling": "RPM"},
}

BLOB_DECODER_REGISTRY: dict[str, dict[str, object]] = {
    "MapSwitchingMaster": {
        "length": 1,
        "data_type": "uint8",
        "enum": {0: "OFF", 1: "Alt Map", 2: "MethSafe", 3: "Flex"},
    },
    "MethSafeMapMaster": {
        "length": 1,
        "data_type": "uint8",
        "enum": {0: "OFF", 1: "5V ADC", 2: "Resistor ADC"},
    },
    "AlternateMapMaxCount": {"length": 1, "data_type": "uint8"},
    "FlexFuelInput": {"length": 1, "data_type": "uint8"},
    "Crossover_RPMSpeed_Ratio_5MT": {"length": 2, "data_type": "uint16"},
    "blobbits": {"length": 2, "data_type": "bits"},
}


def _load_config(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except (json.JSONDecodeError, OSError):
        return {}


def _validate_config(config: dict) -> tuple[dict, list[str], list[str]]:
    if not isinstance(config, dict):
        return {}, [], ["Config root must be a JSON object."]

    allowed = {
        "axis_address_registry",
        "table_size_overrides",
        "table_size_patterns",
        "common_table_queries",
        "allowlist_profiles",
        "guardrail_profiles",
        "log_signal_sets",
        "launch_health",
        "torque_risk",
        "tuning_usecases",
    }

    warnings: list[str] = []
    errors: list[str] = []
    validated: dict[str, object] = {}

    for key in config.keys():
        if key not in allowed:
            warnings.append(f"Unknown config key: {key}")

    axis_registry = config.get("axis_address_registry")
    if axis_registry is not None:
        if not isinstance(axis_registry, dict):
            errors.append("axis_address_registry must be an object.")
        else:
            validated["axis_address_registry"] = axis_registry

    size_overrides = config.get("table_size_overrides")
    if size_overrides is not None:
        if not isinstance(size_overrides, dict):
            errors.append("table_size_overrides must be an object.")
        else:
            validated["table_size_overrides"] = size_overrides

    size_patterns = config.get("table_size_patterns")
    if size_patterns is not None:
        if not isinstance(size_patterns, list):
            errors.append("table_size_patterns must be a list.")
        else:
            validated["table_size_patterns"] = size_patterns

    common_queries = config.get("common_table_queries")
    if common_queries is not None:
        if not isinstance(common_queries, dict):
            errors.append("common_table_queries must be an object.")
        else:
            validated["common_table_queries"] = common_queries

    allowlist_profiles = config.get("allowlist_profiles")
    if allowlist_profiles is not None:
        if not isinstance(allowlist_profiles, dict):
            errors.append("allowlist_profiles must be an object.")
        else:
            validated["allowlist_profiles"] = allowlist_profiles

    guardrail_profiles = config.get("guardrail_profiles")
    if guardrail_profiles is not None:
        if not isinstance(guardrail_profiles, dict):
            errors.append("guardrail_profiles must be an object.")
        else:
            validated["guardrail_profiles"] = guardrail_profiles

    log_signal_sets = config.get("log_signal_sets")
    if log_signal_sets is not None:
        if not isinstance(log_signal_sets, dict):
            errors.append("log_signal_sets must be an object.")
        else:
            validated["log_signal_sets"] = log_signal_sets

    launch_health = config.get("launch_health")
    if launch_health is not None:
        if not isinstance(launch_health, dict):
            errors.append("launch_health must be an object.")
        else:
            validated["launch_health"] = launch_health

    torque_risk = config.get("torque_risk")
    if torque_risk is not None:
        if not isinstance(torque_risk, dict):
            errors.append("torque_risk must be an object.")
        else:
            validated["torque_risk"] = torque_risk

    tuning_usecases = config.get("tuning_usecases")
    if tuning_usecases is not None:
        if not isinstance(tuning_usecases, dict):
            errors.append("tuning_usecases must be an object.")
        else:
            validated["tuning_usecases"] = tuning_usecases

    return validated, warnings, errors


def _apply_config(config: dict) -> None:
    global ALLOWLIST_PROFILES
    global COMMON_TABLE_QUERIES
    global TABLE_SIZE_OVERRIDES
    global TABLE_SIZE_PATTERNS
    global AXIS_ADDRESS_REGISTRY
    global GUARDRAIL_PROFILES
    global LOG_SIGNAL_SETS
    global LAUNCH_HEALTH_SETTINGS
    global TORQUE_RISK_SETTINGS
    global TUNING_USECASES

    allowlist_profiles = config.get("allowlist_profiles")
    if isinstance(allowlist_profiles, dict):
        ALLOWLIST_PROFILES = {str(k): list(v) for k, v in allowlist_profiles.items()}

    common_queries = config.get("common_table_queries")
    if isinstance(common_queries, dict):
        COMMON_TABLE_QUERIES = {str(k): list(v) for k, v in common_queries.items()}

    size_overrides = config.get("table_size_overrides")
    if isinstance(size_overrides, dict):
        TABLE_SIZE_OVERRIDES = {
            str(k).lower(): (int(v[0]), int(v[1]))
            for k, v in size_overrides.items()
            if isinstance(v, (list, tuple)) and len(v) == 2
        }

    size_patterns = config.get("table_size_patterns")
    if isinstance(size_patterns, list):
        compiled: list[tuple[re.Pattern[str], tuple[int, int]]] = []
        for entry in size_patterns:
            if not isinstance(entry, dict):
                continue
            pattern = entry.get("pattern")
            size = entry.get("size")
            if not pattern or not isinstance(size, (list, tuple)) or len(size) != 2:
                continue
            compiled.append((re.compile(pattern, re.IGNORECASE), (int(size[0]), int(size[1]))))
        if compiled:
            TABLE_SIZE_PATTERNS = compiled

    axis_registry = config.get("axis_address_registry")
    if isinstance(axis_registry, dict):
        parsed: dict[int, dict[str, object]] = {}
        for key, value in axis_registry.items():
            try:
                address = int(str(key), 16) if str(key).lower().startswith("0x") else int(key)
            except ValueError:
                continue
            if not isinstance(value, dict):
                continue
            elements = value.get("elements")
            scaling = value.get("scaling")
            if elements is None:
                continue
            parsed[address] = {"elements": int(elements), "scaling": scaling}
        if parsed:
            AXIS_ADDRESS_REGISTRY = parsed

    guardrail_profiles = config.get("guardrail_profiles")
    if isinstance(guardrail_profiles, dict):
        GUARDRAIL_PROFILES = {str(k): dict(v) for k, v in guardrail_profiles.items()}

    log_signal_sets = config.get("log_signal_sets")
    if isinstance(log_signal_sets, dict):
        LOG_SIGNAL_SETS = {str(k): list(v) for k, v in log_signal_sets.items()}

    launch_health = config.get("launch_health")
    if isinstance(launch_health, dict):
        LAUNCH_HEALTH_SETTINGS = {str(k): float(v) for k, v in launch_health.items()}

    torque_risk = config.get("torque_risk")
    if isinstance(torque_risk, dict):
        TORQUE_RISK_SETTINGS = {str(k): float(v) for k, v in torque_risk.items()}

    tuning_usecases = config.get("tuning_usecases")
    if isinstance(tuning_usecases, dict):
        TUNING_USECASES = {str(k): dict(v) for k, v in tuning_usecases.items()}


_raw_config = _load_config(CONFIG_PATH)
_validated_config, _config_warnings, _config_errors = _validate_config(_raw_config)
CONFIG_WARNINGS = _config_warnings
CONFIG_ERRORS = _config_errors
if CONFIG_ERRORS and CONFIG_STRICT:
    raise ValueError(f"Config validation failed: {CONFIG_ERRORS}")
_apply_config(_validated_config)


@dataclass
class TableDef:
    name: str
    address: int
    rows: int | None
    cols: int | None
    data_type: str | None
    scaling: str | None
    swapxy: bool
    source_xml: str


def _iter_xml_files(paths: Iterable[str]) -> Iterable[str]:
    for base in paths:
        base = base.strip()
        if not base:
            continue
        if os.path.isfile(base) and base.lower().endswith(".xml"):
            yield base
            continue
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for filename in files:
                if filename.lower().endswith(".xml"):
                    yield os.path.join(root, filename)


def _xml_paths_key() -> tuple[str, ...]:
    return tuple(path.strip() for path in XML_PATHS if path.strip())


@lru_cache(maxsize=128)
def _load_xml_root(xml_path: str) -> ET.Element | None:
    try:
        return ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None


def _get_xmlid(xml_path: str) -> str | None:
    root = _load_xml_root(xml_path)
    if root is None:
        return None
    xmlid = root.findtext(".//xmlid")
    if not xmlid:
        return None
    return xmlid.strip()


@lru_cache(maxsize=8)
def _build_xmlid_map_cached(paths_key: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for xml_path in _iter_xml_files(paths_key):
        xmlid = _get_xmlid(xml_path)
        if xmlid:
            mapping[xmlid] = xml_path
    return mapping


def _build_xmlid_map() -> dict[str, str]:
    return _build_xmlid_map_cached(_xml_paths_key())


def _resolve_xml_chain(xml_path: str) -> list[str]:
    xml_map = _build_xmlid_map()
    chain: list[str] = []
    seen: set[str] = set()

    def walk(path: str | None) -> None:
        if not path or path in seen:
            return
        seen.add(path)
        chain.append(path)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            return
        for include in root.findall("include"):
            include_id = (include.text or "").strip()
            if include_id:
                walk(xml_map.get(include_id))

    walk(xml_path)
    return chain


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        if value.lower().startswith("0x"):
            return int(value, 16)
        try:
            return int(value)
        except ValueError:
            return int(value, 16)
    except ValueError:
        return None


def _find_attr_int(elem: ET.Element, names: Iterable[str]) -> int | None:
    for name in names:
        if name in elem.attrib:
            value = _to_int(elem.attrib.get(name))
            if value is not None:
                return value
    return None


def _find_attr_str(elem: ET.Element, names: Iterable[str]) -> str | None:
    for name in names:
        value = elem.attrib.get(name)
        if value:
            return value
    return None


def _extract_scaling(elem: ET.Element) -> str | None:
    scaling = _find_attr_str(elem, ["scaling", "formula", "math"])
    if scaling:
        return scaling
    for child in elem.iter():
        if child.tag.lower() in ("scaling", "math") and child.text:
            return child.text.strip()
    return None


def _parse_table_defs_for_xml(xml_path: str) -> list[TableDef]:
    table_defs: list[TableDef] = []
    root = _load_xml_root(xml_path)
    if root is None:
        return table_defs
    for elem in root.iter():
        name = _find_attr_str(elem, ["name", "id", "description"])
        address = _find_attr_int(elem, ["address", "storageaddress", "offset", "start"])
        if not name or address is None:
            continue
        swapxy_value = (elem.attrib.get("swapxy") or "").strip().lower()
        swapxy = swapxy_value in ("true", "1", "yes")
        rows = _find_attr_int(elem, ["rows", "row", "ysize", "y"])
        cols = _find_attr_int(elem, ["columns", "cols", "xsize", "x"])
        if rows is None or cols is None:
            for child in elem:
                tag = child.tag.lower()
                axis_type = (child.attrib.get("type") or "").lower()
                if tag in ("xaxis", "yaxis", "axis", "table") or axis_type in ("x axis", "y axis"):
                    elements = _find_attr_int(child, ["elements", "size", "count", "cols", "rows"])
                    if elements is None:
                        axis_addr = _find_attr_int(child, ["address", "storageaddress", "offset", "start"])
                        elements, _ = _axis_elements_from_address(axis_addr)
                    if elements is None:
                        continue
                    if cols is None and (tag == "xaxis" or axis_type == "x axis"):
                        cols = elements
                    elif rows is None and (tag == "yaxis" or axis_type == "y axis"):
                        rows = elements
        if rows is None and cols is None:
            axis_elements, axis_scaling = _axis_elements_from_address(address)
            if axis_elements is not None:
                cols = axis_elements
        data_type = _find_attr_str(elem, ["type", "datatype", "storage", "storagetype"])
        scaling = _extract_scaling(elem)
        if scaling is None and rows is None and cols is None:
            _, axis_scaling = _axis_elements_from_address(address)
            if axis_scaling:
                scaling = axis_scaling
        table_defs.append(
            TableDef(
                name=name.strip(),
                address=address,
                rows=rows,
                cols=cols,
                data_type=data_type,
                scaling=scaling,
                swapxy=swapxy,
                source_xml=xml_path,
            )
        )
    return table_defs


def _tabledef_from_elem(elem: ET.Element, xml_path: str, rows: int | None = None, cols: int | None = None) -> TableDef | None:
    name = _find_attr_str(elem, ["name", "id", "description"])
    address = _find_attr_int(elem, ["address", "storageaddress", "offset", "start"])
    if not name or address is None:
        return None
    swapxy_value = (elem.attrib.get("swapxy") or "").strip().lower()
    swapxy = swapxy_value in ("true", "1", "yes")
    data_type = _find_attr_str(elem, ["type", "datatype", "storage", "storagetype"])
    scaling = _extract_scaling(elem)
    return TableDef(
        name=name.strip(),
        address=address,
        rows=rows,
        cols=cols,
        data_type=data_type,
        scaling=scaling,
        swapxy=swapxy,
        source_xml=xml_path,
    )


def _iter_table_elems_for_xml(xml_path: str) -> Iterable[tuple[ET.Element, str]]:
    root = _load_xml_root(xml_path)
    if root is None:
        return []
    return [(elem, xml_path) for elem in root.iter()]


def _find_table_elem_in_chain(chain: list[str], table_name: str) -> tuple[ET.Element, str] | None:
    name_lower = table_name.lower()
    exact_match: tuple[ET.Element, str] | None = None
    for xml_path in chain:
        root = _load_xml_root(xml_path)
        if root is None:
            continue
        for elem in root.iter():
            name = _find_attr_str(elem, ["name", "id", "description"])
            if not name:
                continue
            if name.lower() == name_lower:
                return elem, xml_path
            if exact_match is None and name_lower in name.lower():
                exact_match = (elem, xml_path)
    return exact_match


def _parse_table_defs() -> list[TableDef]:
    table_defs: list[TableDef] = []
    for xml_path in _iter_xml_files(XML_PATHS):
        table_defs.extend(_parse_table_defs_for_xml(xml_path))
    return table_defs


def _match_tables(table_name: str) -> list[TableDef]:
    name_lower = table_name.lower()
    matches = []
    for table_def in _parse_table_defs():
        if name_lower in table_def.name.lower():
            matches.append(table_def)
    return matches


def _match_tables_in_defs(table_name: str, table_defs: list[TableDef]) -> list[TableDef]:
    name_lower = table_name.lower()
    exact = [td for td in table_defs if td.name.lower() == name_lower]
    if exact:
        return exact
    return [td for td in table_defs if name_lower in td.name.lower()]


def _dtype_format(data_type: str | None, endian: str) -> tuple[str, int]:
    endian_prefix = ">" if endian.lower() == "big" else "<"
    normalized = (data_type or "uint16").lower()
    mapping = {
        "uint8": ("B", 1),
        "u8": ("B", 1),
        "int8": ("b", 1),
        "s8": ("b", 1),
        "uint16": ("H", 2),
        "u16": ("H", 2),
        "int16": ("h", 2),
        "s16": ("h", 2),
        "uint32": ("I", 4),
        "u32": ("I", 4),
        "int32": ("i", 4),
        "s32": ("i", 4),
        "float": ("f", 4),
        "float32": ("f", 4),
    }
    fmt, size = mapping.get(normalized, ("H", 2))
    return f"{endian_prefix}{fmt}", size


def _safe_eval_scaling(expr: str, x: float) -> float:
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.Constant,
        ast.Load,
        ast.Name,
        ast.USub,
        ast.UAdd,
    }
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            raise ValueError("Unsupported scaling expression")
        if isinstance(node, ast.Name) and node.id != "x":
            raise ValueError("Only 'x' is allowed in scaling")
    return float(eval(compile(tree, "<scaling>", "eval"), {"__builtins__": {}}, {"x": x}))


def _safe_eval_inverse(expr: str, x: float) -> float:
    return _safe_eval_scaling(expr, x)


def _iter_rom_strings(rom_path: str, min_len: int = 4, chunk_size: int = 65536) -> Iterable[tuple[int, str]]:
    pattern = re.compile(rb"[ -~]{%d,}" % min_len)
    overlap = min_len - 1
    offset = 0
    tail = b""
    with open(rom_path, "rb") as rom_file:
        while True:
            chunk = rom_file.read(chunk_size)
            if not chunk:
                break
            data = tail + chunk
            for match in pattern.finditer(data):
                start = match.start()
                text = match.group().decode("ascii", errors="ignore")
                yield offset + start - len(tail), text
            tail = data[-overlap:] if len(data) >= overlap else data
            offset += len(chunk)


def _xml_rom_id_map() -> dict[str, str]:
    rom_map: dict[str, str] = {}
    for xml_path in _iter_xml_files(XML_PATHS):
        base = os.path.basename(xml_path)
        if not base.lower().endswith(".xml"):
            continue
        name = base[:-4]
        parts = name.split(" ")
        if parts and parts[0].isdigit():
            rom_map[parts[0]] = name
    return rom_map


def _parse_scaling_defs() -> dict[str, dict[str, str]]:
    return _parse_scaling_defs_cached(_xml_paths_key())


@lru_cache(maxsize=8)
def _parse_scaling_defs_cached(paths_key: tuple[str, ...]) -> dict[str, dict[str, str]]:
    scalings: dict[str, dict[str, str]] = {}
    for xml_path in _iter_xml_files(paths_key):
        root = _load_xml_root(xml_path)
        if root is None:
            continue
        for elem in root.iter():
            if elem.tag.lower() != "scaling":
                continue
            name = elem.attrib.get("name")
            if not name:
                continue
            scaling = dict(elem.attrib)
            if elem.text and elem.text.strip():
                scaling["formula"] = elem.text.strip()
            scalings[name] = scaling
    return scalings


def _get_scaling_meta(scaling_name: str | None) -> dict[str, str | None]:
    if not scaling_name:
        return {
            "name": None,
            "units": None,
            "toexpr": None,
            "frexpr": None,
            "min": None,
            "max": None,
        }
    scaling_defs = _parse_scaling_defs()
    scaling = scaling_defs.get(scaling_name, {})
    return {
        "name": scaling_name,
        "units": _normalize_units(scaling.get("units")),
        "toexpr": scaling.get("toexpr"),
        "frexpr": scaling.get("frexpr"),
        "min": scaling.get("min"),
        "max": scaling.get("max"),
    }


def _safe_log_path(filename: str) -> str | None:
    if not filename or any(ch in filename for ch in ("..", "/", "\\")):
        return None
    return os.path.join(LOG_PATH, filename)


def _normalize_units(units: str | None) -> str | None:
    if not units:
        return None
    normalized = units.strip().lower()
    mapping = {
        "rpm": "RPM",
        "r/min": "RPM",
        "load": "Load",
        "%": "%",
        "deg": "deg",
        "degrees": "deg",
        "afr": "AFR",
        "kpa": "kPa",
        "psi": "psi",
        "km/h": "km/h",
        "ms": "ms",
        "volts": "V",
        "v": "V",
    }
    return mapping.get(normalized, units)


def _detect_log_columns(fields: list[str] | None) -> dict[str, str | None]:
    if not fields:
        return {}
    def find_exact(name: str) -> str | None:
        return name if name in fields else None

    def find_contains(parts: Iterable[str]) -> str | None:
        for field in fields:
            if not field:
                continue
            field_lower = field.lower()
            if any(part in field_lower for part in parts):
                return field
        return None

    return {
        "time": find_exact("LogEntrySeconds") or find_contains(["time", "seconds"]),
        "rpm": find_exact("RPM") or find_contains(["rpm"]),
        "tps": find_exact("TPS") or find_contains(["tps", "throttle"]),
        "speed": find_exact("Speed") or find_contains(["speed", "vehicle"]),
        "boost": find_exact("Boost") or find_contains(["boost", "map"]),
        "target_boost": find_contains(["target boost", "boost target", "boosttar"]),
        "wgdc": find_contains(["wgdc", "wastegate", "wg duty", "duty"]),
        "gear": find_contains(["gear"]),
        "iat": find_contains(["iat", "intake temp", "iatf", "mat"]),
        "afr": find_exact("AFR") or find_contains(["afr", "wideband"]),
        "stft": find_contains(["stft", "short term"]),
        "ltft": find_contains(["ltft", "long term"]),
        "fuel_pressure": find_contains(["fuel pressure", "fp"]),
        "timing": find_exact("TimingAdv") or find_contains(["timing", "spark"]),
        "knock": find_exact("KnockSum") or find_contains(["knock"]),
        "baro": find_contains(["baro", "barometric"]),
        "egt": find_contains(["egt", "exhaust gas"]),
        "ipw": find_exact("IPW") or find_contains(["ipw", "pulse"]),
        "idc": find_exact("IDC") or find_contains(["idc"]),
        "load": find_exact("Load") or find_contains(["load"]),
    }


def _missing_log_columns(columns: dict[str, str | None], required: Iterable[str]) -> list[str]:
    return [key for key in required if not columns.get(key)]


def _infer_table_size(table_name: str | None) -> tuple[int | None, int | None]:
    if not table_name:
        return None, None
    key = table_name.strip().lower()
    if key in TABLE_SIZE_OVERRIDES:
        return TABLE_SIZE_OVERRIDES[key]
    for pattern, size in TABLE_SIZE_PATTERNS:
        if pattern.search(table_name):
            return size
    return None, None


def _axis_elements_from_address(address: int | None) -> tuple[int | None, str | None]:
    if address is None:
        return None, None
    entry = AXIS_ADDRESS_REGISTRY.get(address)
    if not entry:
        return None, None
    return entry.get("elements"), entry.get("scaling")


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_table_name(table_name: str) -> str | None:
    if not table_name or not table_name.strip():
        return "table_name is required."
    return None


def _validate_log_filters(
    min_tps: float,
    max_speed: float,
    rpm_min: float,
    rpm_max: float,
    max_rows: int,
    top_n: int | None = None,
) -> str | None:
    if min_tps < 0 or min_tps > 100:
        return "min_tps must be between 0 and 100."
    if max_speed < 0:
        return "max_speed must be >= 0."
    if rpm_min < 0 or rpm_max < 0 or rpm_min > rpm_max:
        return "rpm_min must be <= rpm_max and both >= 0."
    if max_rows <= 0:
        return "max_rows must be > 0."
    if top_n is not None and top_n <= 0:
        return "top_n must be > 0."
    return None


def _resolve_table_meta(table_def: TableDef, rows: int | None, cols: int | None, data_type: str | None, endian: str) -> tuple[int, int, str, str, str | None]:
    rows = rows or table_def.rows
    cols = cols or table_def.cols
    if rows is None or cols is None:
        inferred_rows, inferred_cols = _infer_table_size(table_def.name)
        if rows is None:
            rows = inferred_rows
        if cols is None:
            cols = inferred_cols
    if rows is None or cols is None:
        table_type = (table_def.data_type or "").lower()
        if table_type in ("1d", "2d") and rows is None and cols is None:
            rows, cols = 1, 1
        else:
            raise ValueError(
                f"Row/column size not found in XML for '{table_def.name}'. Provide rows/cols manually."
            )

    data_type = data_type or table_def.data_type
    if data_type and data_type.lower() in ("1d", "2d", "3d", "table"):
        data_type = None

    scaling_defs = _parse_scaling_defs()
    resolved_scaling = None
    if table_def.scaling:
        resolved_scaling = scaling_defs.get(table_def.scaling, {}).get("toexpr")
        if resolved_scaling is None and "x" in table_def.scaling:
            resolved_scaling = table_def.scaling
        scaling_storage = scaling_defs.get(table_def.scaling, {}).get("storagetype")
        if scaling_storage:
            data_type = scaling_storage
        if endian == "big":
            endian = scaling_defs.get(table_def.scaling, {}).get("endian", endian)

    return rows, cols, data_type or "uint16", endian, resolved_scaling


def _flatten_table(table: list[list[float]], swapxy: bool) -> list[float]:
    if not table:
        return []
    if swapxy:
        transposed = [list(row) for row in zip(*table)]
        return [value for column in transposed for value in column]
    return [value for row in table for value in row]


def _apply_swapxy(values: list[float], rows: int, cols: int, swapxy: bool) -> list[list[float]]:
    if not swapxy:
        return [values[i * cols : (i + 1) * cols] for i in range(rows)]
    table_raw = [values[i * rows : (i + 1) * rows] for i in range(cols)]
    return [list(row) for row in zip(*table_raw)]


def _parse_rom_id_defs() -> list[tuple[str, int, str]]:
    rom_defs: list[tuple[str, int, str]] = []
    for xml_path in _iter_xml_files(XML_PATHS):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        internal_id_hex = None
        internal_id_address = None
        for elem in root.iter():
            tag = elem.tag.lower()
            if tag == "internalidhex" and elem.text:
                internal_id_hex = elem.text.strip()
            elif tag == "internalidaddress" and elem.text:
                internal_id_address = _to_int(elem.text.strip())
        if internal_id_hex and internal_id_address is not None:
            rom_defs.append((internal_id_hex, internal_id_address, xml_path))
    return rom_defs


def _format_table_rows(table_defs: list[TableDef]) -> list[str]:
    rows: list[str] = []
    for table_def in table_defs:
        addr = f"0x{table_def.address:X}" if table_def.address is not None else "-"
        source = os.path.basename(table_def.source_xml)
        rows.append(f"{table_def.name} | {addr} | {source}")
    return rows


def _resolve_rom_chain(rom_filename: str) -> list[str]:
    info = identify_rom(rom_filename)
    matches = info.get("matches") if isinstance(info, dict) else None
    if not matches:
        return []
    xml_path = matches[0].get("xml")
    if not xml_path:
        return []
    return _resolve_xml_chain(xml_path)


def _get_allowlist() -> list[str]:
    allowlist = list(TABLE_ALLOWLIST_BASE)
    if ACTIVE_ALLOWLIST_PROFILE and ACTIVE_ALLOWLIST_PROFILE in ALLOWLIST_PROFILES:
        allowlist.extend(ALLOWLIST_PROFILES[ACTIVE_ALLOWLIST_PROFILE])
    return allowlist


def _allowlist_match(name: str, allowlist: list[str]) -> bool:
    if not allowlist:
        return True
    name_lower = name.lower()
    for entry in allowlist:
        entry = entry.lower()
        if "*" in entry:
            if fnmatch.fnmatchcase(name_lower, entry):
                return True
        elif entry == name_lower:
            return True
    return False


def _default_max_delta(table_name: str) -> float | None:
    name = table_name.lower()
    if "timing" in name:
        return 2.0
    if "afr" in name or "fuel" in name:
        return 0.5
    if "boost" in name:
        return 2.0
    if "wgdc" in name or "wgd" in name or "wastegate" in name:
        return 5.0
    if "mivec" in name:
        return 5.0
    if "rev" in name or "rpm" in name:
        return 200.0
    return None


def _read_table_from_def(
    rom_filename: str,
    table_def: TableDef,
    rows: int | None,
    cols: int | None,
    data_type: str | None,
    endian: str,
    apply_scaling: bool,
) -> dict:
    assumed_scalar = False
    try:
        rows, cols, data_type, endian, resolved_scaling = _resolve_table_meta(
            table_def, rows, cols, data_type, endian
        )
    except ValueError as exc:
        return {
            "error": str(exc),
            "matched_table": table_def.name,
            "source_xml": table_def.source_xml,
            "rows": rows or table_def.rows,
            "cols": cols or table_def.cols,
        }

    if (table_def.data_type or "").lower() in ("1d", "2d") and table_def.rows is None and table_def.cols is None:
        if rows == 1 and cols == 1:
            assumed_scalar = True

    fmt, size = _dtype_format(data_type, endian)
    count = rows * cols
    byte_count = count * size

    rom_path = os.path.join(ROM_PATH, rom_filename)
    if not os.path.isfile(rom_path):
        return {"error": f"ROM file not found: {rom_path}"}

    with open(rom_path, "rb") as rom_file:
        rom_file.seek(table_def.address)
        raw = rom_file.read(byte_count)

    if len(raw) != byte_count:
        return {"error": "ROM read size mismatch", "expected": byte_count, "got": len(raw)}

    values = []
    for idx in range(count):
        start = idx * size
        chunk = raw[start : start + size]
        values.append(struct.unpack(fmt, chunk)[0])

    scaling = resolved_scaling if apply_scaling else None
    if scaling:
        try:
            values = [_safe_eval_scaling(scaling, float(v)) for v in values]
        except ValueError as exc:
            return {
                "error": f"Scaling failed: {exc}",
                "matched_table": table_def.name,
                "scaling": scaling,
            }

    if table_def.swapxy:
        table_raw = [values[i * rows : (i + 1) * rows] for i in range(cols)]
        table = [list(row) for row in zip(*table_raw)]
    else:
        table = _apply_swapxy(values, rows, cols, table_def.swapxy)

    scaling_meta = _get_scaling_meta(table_def.scaling)

    result = {
        "table_name": table_def.name,
        "address": f"0x{table_def.address:X}",
        "rows": rows,
        "cols": cols,
        "data_type": data_type or "uint16",
        "endian": endian,
        "scaling": scaling,
        "scaling_name": scaling_meta["name"],
        "units": scaling_meta["units"],
        "scaling_toexpr": scaling_meta["toexpr"],
        "scaling_frexpr": scaling_meta["frexpr"],
        "scaling_min": scaling_meta["min"],
        "scaling_max": scaling_meta["max"],
        "swapxy": table_def.swapxy,
        "source_xml": table_def.source_xml,
        "data": table,
    }
    if assumed_scalar:
        result["warning"] = "Assumed scalar size 1x1 due to missing dimensions."
    return result


def _normalize_table_result(result: dict) -> dict:
    result.setdefault("axis_x", None)
    result.setdefault("axis_y", None)
    result.setdefault("rows", None)
    result.setdefault("cols", None)
    result.setdefault("units", None)
    result.setdefault("scaling_name", None)
    result.setdefault("source_xml", None)
    return result


def _read_scalar_from_def(
    rom_filename: str,
    table_def: TableDef,
    endian: str,
) -> dict:
    scaled = _read_table_from_def(rom_filename, table_def, 1, 1, None, endian, True)
    raw = _read_table_from_def(rom_filename, table_def, 1, 1, None, endian, False)
    if "data" not in scaled or "data" not in raw:
        return {"error": "Unable to read scalar", "scaled": scaled, "raw": raw}

    scaling_meta = _get_scaling_meta(table_def.scaling)

    return {
        "table_name": table_def.name,
        "address": f"0x{table_def.address:X}",
        "source_xml": table_def.source_xml,
        "scaling": table_def.scaling,
        "scaling_name": scaling_meta["name"],
        "units": scaling_meta["units"],
        "scaling_toexpr": scaling_meta["toexpr"],
        "scaling_frexpr": scaling_meta["frexpr"],
        "scaling_min": scaling_meta["min"],
        "scaling_max": scaling_meta["max"],
        "raw_value": raw["data"][0][0],
        "value": scaled["data"][0][0],
    }


def _read_raw_bytes(rom_filename: str, address: int, length: int) -> dict:
    if length <= 0:
        return {"error": "length must be > 0."}
    rom_path = os.path.join(ROM_PATH, rom_filename)
    if not os.path.isfile(rom_path):
        return {"error": f"ROM file not found: {rom_path}"}
    with open(rom_path, "rb") as rom_file:
        rom_file.seek(address)
        data = rom_file.read(length)
    if len(data) != length:
        return {"error": "ROM read size mismatch", "expected": length, "got": len(data)}
    return {"data": data}


def _write_table_from_def(
    rom_filename: str,
    table_def: TableDef,
    data: list[list[float]],
    output_filename: str | None,
    overwrite_output: bool,
    max_delta: float | None,
    use_default_max_delta: bool,
    write_metadata: bool,
) -> dict:
    if not _validate_allowlist(table_def.name):
        return {"error": f"Table '{table_def.name}' is not in allowlist."}

    rom_path = os.path.join(ROM_PATH, rom_filename)
    if not os.path.isfile(rom_path):
        return {"error": f"ROM file not found: {rom_path}"}

    try:
        rows, cols, data_type, endian, resolved_scaling = _resolve_table_meta(
            table_def, None, None, None, "big"
        )
    except ValueError as exc:
        return {"error": str(exc), "matched_table": table_def.name}

    if len(data) != rows or any(len(row) != cols for row in data):
        return {"error": "Data shape does not match table dimensions.", "rows": rows, "cols": cols}

    flat_values = _flatten_table(data, table_def.swapxy)
    inverse_values, inverse_expr = _inverse_scale(flat_values, table_def)

    if max_delta is None and use_default_max_delta:
        max_delta = _default_max_delta(table_def.name)

    if max_delta is not None:
        current = _read_table_from_def(rom_filename, table_def, None, None, None, "big", True)
        if "data" not in current:
            return {"error": "Unable to read current table for delta check."}
        current_flat = _flatten_table(current["data"], table_def.swapxy)
        for idx, (old, new) in enumerate(zip(current_flat, flat_values)):
            if abs(float(new) - float(old)) > max_delta:
                return {"error": "Max delta exceeded", "index": idx, "old": old, "new": new}

    os.makedirs(OUTPUT_ROM_PATH, exist_ok=True)
    output_filename = output_filename or rom_filename.replace(".bin", "_modified.bin")
    output_path = os.path.join(OUTPUT_ROM_PATH, output_filename)

    if os.path.isfile(output_path) and not overwrite_output:
        return {"error": f"Output file exists: {output_path}"}

    shutil.copyfile(rom_path, output_path)
    if os.path.isfile(output_path) and REQUIRE_BACKUP:
        backup_path = output_path + ".bak"
        if not os.path.isfile(backup_path):
            shutil.copyfile(output_path, backup_path)

    encoded = _encode_values(inverse_values, data_type, endian)
    with open(output_path, "r+b") as rom_file:
        rom_file.seek(table_def.address)
        rom_file.write(encoded)

    metadata_path = None
    if write_metadata:
        rom_info = identify_rom(rom_filename)
        chain = _resolve_rom_chain(rom_filename)
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rom_filename": rom_filename,
            "rom_id": rom_info.get("matches", [{}])[0].get("rom_id") if isinstance(rom_info, dict) else None,
            "definition_chain": chain,
            "output_path": output_path,
            "table_name": table_def.name,
            "rows": rows,
            "cols": cols,
            "applied_max_delta": max_delta,
        }
        metadata_path = output_path + ".meta.json"
        with open(metadata_path, "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2)

    return {
        "rom_filename": rom_filename,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "table_name": table_def.name,
        "rows": rows,
        "cols": cols,
        "data_type": data_type,
        "endian": endian,
        "scaling": resolved_scaling,
        "inverse_scaling": inverse_expr,
        "swapxy": table_def.swapxy,
        "applied_max_delta": max_delta,
    }


def _preview_write_table_from_def(
    rom_filename: str,
    table_def: TableDef,
    data: list[list[float]],
    max_delta: float | None,
    use_default_max_delta: bool,
) -> dict:
    if not _validate_allowlist(table_def.name):
        return {"error": f"Table '{table_def.name}' is not in allowlist."}

    rom_path = os.path.join(ROM_PATH, rom_filename)
    if not os.path.isfile(rom_path):
        return {"error": f"ROM file not found: {rom_path}"}

    try:
        rows, cols, data_type, endian, resolved_scaling = _resolve_table_meta(
            table_def, None, None, None, "big"
        )
    except ValueError as exc:
        return {"error": str(exc), "matched_table": table_def.name}

    if len(data) != rows or any(len(row) != cols for row in data):
        return {"error": "Data shape does not match table dimensions.", "rows": rows, "cols": cols}

    flat_values = _flatten_table(data, table_def.swapxy)

    if max_delta is None and use_default_max_delta:
        max_delta = _default_max_delta(table_def.name)

    current = _read_table_from_def(rom_filename, table_def, None, None, None, "big", True)
    if "data" not in current:
        return {"error": "Unable to read current table for delta check.", "current": current}

    current_flat = _flatten_table(current["data"], table_def.swapxy)
    deltas = [float(new) - float(old) for old, new in zip(current_flat, flat_values)]
    changed_indices = [idx for idx, delta in enumerate(deltas) if delta != 0]
    max_abs_delta = max((abs(delta) for delta in deltas), default=0.0)
    blocked = False
    if max_delta is not None:
        blocked = any(abs(delta) > max_delta for delta in deltas)

    sample_deltas = []
    for idx in changed_indices[:10]:
        sample_deltas.append(
            {
                "index": idx,
                "old": current_flat[idx],
                "new": flat_values[idx],
                "delta": deltas[idx],
            }
        )

    return {
        "rom_filename": rom_filename,
        "table_name": table_def.name,
        "rows": rows,
        "cols": cols,
        "data_type": data_type,
        "endian": endian,
        "scaling": resolved_scaling,
        "swapxy": table_def.swapxy,
        "applied_max_delta": max_delta,
        "diff_cells": len(changed_indices),
        "max_abs_delta": max_abs_delta,
        "blocked_by_max_delta": blocked,
        "sample_deltas": sample_deltas,
    }


@mcp.tool()
def list_tables(contains: str = "", limit: int = 50) -> list[str]:
    """Lists table names that match a substring (case-insensitive)."""
    contains = contains.lower().strip()
    table_defs = _parse_table_defs()
    results = []
    for table_def in table_defs:
        if not contains or contains in table_def.name.lower():
            results.append(f"{table_def.name} | 0x{table_def.address:X} | {table_def.source_xml}")
        if len(results) >= limit:
            break
    return results


@mcp.tool()
def get_allowlist_profiles() -> dict:
    """Returns available allowlist profiles and the active profile."""
    return {
        "active_profile": ACTIVE_ALLOWLIST_PROFILE,
        "profiles": list(ALLOWLIST_PROFILES.keys()),
        "allowlist": _get_allowlist(),
    }


@mcp.tool()
def set_allowlist_profile(profile: str | None) -> dict:
    """Sets the active allowlist profile."""
    global ACTIVE_ALLOWLIST_PROFILE
    if not profile:
        ACTIVE_ALLOWLIST_PROFILE = None
    else:
        profile = profile.strip().lower()
        if profile not in ALLOWLIST_PROFILES:
            return {"error": f"Unknown profile: {profile}", "profiles": list(ALLOWLIST_PROFILES.keys())}
        ACTIVE_ALLOWLIST_PROFILE = profile
    return {
        "active_profile": ACTIVE_ALLOWLIST_PROFILE,
        "allowlist": _get_allowlist(),
    }


@mcp.tool()
def get_current_context() -> dict:
    """Returns current server configuration and allowlist context."""
    return {
        "rom_path": ROM_PATH,
        "output_rom_path": OUTPUT_ROM_PATH,
        "log_path": LOG_PATH,
        "xml_paths": XML_PATHS,
        "config_path": CONFIG_PATH,
        "config_strict": CONFIG_STRICT,
        "config_warnings": CONFIG_WARNINGS,
        "config_errors": CONFIG_ERRORS,
        "allowlist_profile": ACTIVE_ALLOWLIST_PROFILE,
        "allowlist": _get_allowlist(),
        "require_backup": REQUIRE_BACKUP,
    }


@mcp.tool()
def list_tables_for_rom(rom_filename: str, contains: str = "", limit: int = 50, pretty: bool = True) -> list:
    """Lists table names for a ROM, following the XML include chain."""
    contains = contains.lower().strip()
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches = [td for td in table_defs if not contains or contains in td.name.lower()]
    matches = matches[:limit]
    if pretty:
        return _format_table_rows(matches)
    return [
        {
            "name": td.name,
            "address": f"0x{td.address:X}",
            "rows": td.rows,
            "cols": td.cols,
            "data_type": td.data_type,
            "scaling": td.scaling,
            "swapxy": td.swapxy,
            "source_xml": td.source_xml,
        }
        for td in matches
    ]


@mcp.tool()
def list_common_tables_for_rom(rom_filename: str, profile: str = "launch", limit: int = 50, pretty: bool = True) -> list:
    """Lists common tables for a ROM based on a profile (launch/boost/fuel/timing/mivec/rev)."""
    profile_key = profile.strip().lower()
    queries = COMMON_TABLE_QUERIES.get(profile_key)
    if not queries:
        return {"error": f"Unknown profile: {profile}", "profiles": list(COMMON_TABLE_QUERIES.keys())}

    table_defs: list[TableDef] = []
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches: list[TableDef] = []
    for query in queries:
        matches.extend([td for td in table_defs if query.lower() in td.name.lower()])

    unique: dict[str, TableDef] = {}
    for td in matches:
        key = td.name.lower()
        if key not in unique:
            unique[key] = td

    results = list(unique.values())[:limit]
    if pretty:
        return _format_table_rows(results)
    return [
        {
            "name": td.name,
            "address": f"0x{td.address:X}",
            "rows": td.rows,
            "cols": td.cols,
            "data_type": td.data_type,
            "scaling": td.scaling,
            "swapxy": td.swapxy,
            "source_xml": td.source_xml,
        }
        for td in results
    ]


@mcp.tool()
def list_tuning_usecases(category: str | None = None) -> list[dict]:
    """Lists tuning usecases and their categories."""
    cat = category.strip().lower() if category else None
    results = []
    for name, info in TUNING_USECASES.items():
        usecase_cat = str(info.get("category", "")).lower()
        if cat and usecase_cat != cat:
            continue
        results.append(
            {
                "name": name,
                "category": info.get("category"),
                "description": info.get("description"),
            }
        )
    return results


def _resolve_usecase_key(usecase: str) -> str | None:
    key = usecase.strip().lower()
    for existing in TUNING_USECASES.keys():
        if existing.lower() == key:
            return existing
    return None


@mcp.tool()
def get_tuning_usecase(usecase: str) -> dict:
    """Returns metadata for a tuning usecase."""
    resolved = _resolve_usecase_key(usecase)
    info = TUNING_USECASES.get(resolved) if resolved else None
    if not info:
        return {"error": f"Unknown usecase: {usecase}", "usecases": list(TUNING_USECASES.keys())}
    return {"name": resolved, **info}


@mcp.tool()
def list_usecase_tables_for_rom(
    rom_filename: str,
    usecase: str,
    limit: int = 50,
    pretty: bool = True,
) -> list:
    """Lists tables for a ROM using tuning usecase queries."""
    resolved = _resolve_usecase_key(usecase)
    info = TUNING_USECASES.get(resolved) if resolved else None
    if not info:
        return {"error": f"Unknown usecase: {usecase}", "usecases": list(TUNING_USECASES.keys())}
    queries = info.get("queries") or []
    if not queries:
        return {"error": f"Usecase has no queries: {usecase}"}

    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches: list[TableDef] = []
    for query in queries:
        matches.extend([td for td in table_defs if str(query).lower() in td.name.lower()])

    unique: dict[str, TableDef] = {}
    for td in matches:
        key = td.name.lower()
        if key not in unique:
            unique[key] = td

    results = list(unique.values())[:limit]
    if pretty:
        return _format_table_rows(results)
    return [
        {
            "name": td.name,
            "address": f"0x{td.address:X}",
            "rows": td.rows,
            "cols": td.cols,
            "data_type": td.data_type,
            "scaling": td.scaling,
            "swapxy": td.swapxy,
            "source_xml": td.source_xml,
        }
        for td in results
    ]


@mcp.tool()
def get_guardrail_profiles() -> dict:
    """Returns guardrail profiles and their thresholds."""
    return {"profiles": GUARDRAIL_PROFILES}


@mcp.tool()
def search_tables_for_rom(rom_filename: str, query: str, limit: int = 50, pretty: bool = True) -> list:
    """Searches table definitions for a ROM using the XML include chain."""
    query = query.lower().strip()
    if not query:
        return []

    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    matches = []
    for xml_path in chain:
        root = _load_xml_root(xml_path)
        if root is None:
            continue
        for elem in root.iter():
            name = _find_attr_str(elem, ["name", "id", "description"])
            if not name:
                continue
            scaling = _extract_scaling(elem) or ""
            category = elem.attrib.get("category") or ""
            if query in name.lower() or query in scaling.lower() or query in category.lower():
                addr = _find_attr_int(elem, ["address", "storageaddress", "offset", "start"])
                if addr is None:
                    continue
                matches.append(
                    {
                        "name": name.strip(),
                        "address": f"0x{addr:X}",
                        "scaling": scaling or None,
                        "category": category or None,
                        "source_xml": xml_path,
                    }
                )
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break

    if pretty:
        return [
            f"{m['name']} | {m['address']} | {os.path.basename(m['source_xml'])}"
            for m in matches
        ]
    return matches


@mcp.tool()
def get_definition_chain(rom_filename: str) -> list[dict]:
    """Returns the XML include chain used for a ROM."""
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return [{"error": "Unable to resolve ROM definition chain."}]
    return [{"xmlid": _get_xmlid(path), "path": path} for path in chain]


@mcp.tool()
def refresh_xml_cache() -> dict:
    """Clears cached XML/scaling data so changes are picked up without restart."""
    _load_xml_root.cache_clear()
    _build_xmlid_map_cached.cache_clear()
    _parse_scaling_defs_cached.cache_clear()
    return {"status": "ok"}


@mcp.tool()
def find_table(rom_filename: str, table_name: str) -> dict:
    """Finds a table definition with axis metadata using the ROM's XML chain."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_elem_entry = _find_table_elem_in_chain(chain, table_name)
    if table_elem_entry is None:
        table_defs: list[TableDef] = []
        for xml_path in chain:
            table_defs.extend(_parse_table_defs_for_xml(xml_path))
        candidates = _match_tables_in_defs(table_name, table_defs)
        return {
            "error": f"No tables matched '{table_name}'.",
            "candidates": [td.name for td in candidates],
        }

    table_elem, xml_path = table_elem_entry
    table_def = _tabledef_from_elem(table_elem, xml_path)
    if table_def is None:
        return {"error": f"No tables matched '{table_name}'."}

    axis_defs = []
    for child in table_elem:
        axis_type = (child.attrib.get("type") or "").lower()
        tag = child.tag.lower()
        if tag not in ("table", "xaxis", "yaxis", "axis"):
            continue
        axis_defs.append(
            {
                "name": child.attrib.get("name"),
                "type": axis_type or tag,
                "address": child.attrib.get("address"),
                "elements": _find_attr_int(child, ["elements", "size", "count", "cols", "rows"]),
                "scaling": child.attrib.get("scaling"),
            }
        )

    inferred_rows, inferred_cols = _infer_table_size(table_def.name)
    return {
        "table": {
            "name": table_def.name,
            "address": f"0x{table_def.address:X}",
            "rows": table_def.rows or inferred_rows,
            "cols": table_def.cols or inferred_cols,
            "data_type": table_def.data_type,
            "scaling": table_def.scaling,
            "swapxy": table_def.swapxy,
            "source_xml": table_def.source_xml,
        },
        "axes": axis_defs,
    }


@mcp.tool()
def read_table(
    rom_filename: str,
    table_name: str,
    rows: int | None = None,
    cols: int | None = None,
    data_type: str | None = None,
    endian: str = "big",
    apply_scaling: bool = True,
) -> dict:
    """Reads a table by name from a ROM using addresses from XML definitions."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    matches = _match_tables(table_name)
    if not matches:
        return {"error": f"No tables matched '{table_name}'."}
    result = _read_table_from_def(rom_filename, matches[0], rows, cols, data_type, endian, apply_scaling)
    return _normalize_table_result(result)


@mcp.tool()
def read_table_for_rom(
    rom_filename: str,
    table_name: str,
    rows: int | None = None,
    cols: int | None = None,
    data_type: str | None = None,
    endian: str = "big",
    apply_scaling: bool = True,
) -> dict:
    """Reads a table by name using the ROM's XML include chain for matching."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches = _match_tables_in_defs(table_name, table_defs)
    if not matches:
        return {"error": f"No tables matched '{table_name}'."}

    result = _read_table_from_def(rom_filename, matches[0], rows, cols, data_type, endian, apply_scaling)
    return _normalize_table_result(result)


def _read_axis_values(rom_filename: str, axis_def: TableDef, axis_type: str, apply_scaling: bool) -> dict:
    if axis_type == "x":
        table = _read_table_from_def(rom_filename, axis_def, 1, axis_def.cols, None, "big", apply_scaling)
        values = table.get("data", [[]])[0] if "data" in table else []
    else:
        table = _read_table_from_def(rom_filename, axis_def, axis_def.rows, 1, None, "big", apply_scaling)
        values = [row[0] for row in table.get("data", [])] if "data" in table else []
    scaling_meta = _get_scaling_meta(axis_def.scaling)
    return {
        "name": axis_def.name,
        "values": values,
        "units": scaling_meta["units"],
        "scaling_name": scaling_meta["name"],
    }


@mcp.tool()
def read_table_with_axes_for_rom(
    rom_filename: str,
    table_name: str,
    apply_scaling: bool = True,
) -> dict:
    """Reads a table and returns axis values using the ROM's XML include chain."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_elem_entry = _find_table_elem_in_chain(chain, table_name)
    if table_elem_entry is None:
        return {"error": f"No tables matched '{table_name}'."}
    table_elem, xml_path = table_elem_entry

    table_def = _tabledef_from_elem(table_elem, xml_path)
    if table_def is None:
        return {"error": f"No tables matched '{table_name}'."}

    inferred_rows, inferred_cols = _infer_table_size(table_def.name)
    if table_def.rows is None:
        table_def = TableDef(
            name=table_def.name,
            address=table_def.address,
            rows=inferred_rows,
            cols=table_def.cols,
            data_type=table_def.data_type,
            scaling=table_def.scaling,
            swapxy=table_def.swapxy,
            source_xml=table_def.source_xml,
        )
    if table_def.cols is None:
        table_def = TableDef(
            name=table_def.name,
            address=table_def.address,
            rows=table_def.rows,
            cols=inferred_cols,
            data_type=table_def.data_type,
            scaling=table_def.scaling,
            swapxy=table_def.swapxy,
            source_xml=table_def.source_xml,
        )

    axis_x: dict | None = None
    axis_y: dict | None = None
    for child in table_elem:
        axis_type = (child.attrib.get("type") or "").lower()
        tag = child.tag.lower()
        if tag not in ("table", "xaxis", "yaxis", "axis"):
            continue
        elements = _find_attr_int(child, ["elements", "size", "count", "cols", "rows"])
        axis_addr = _find_attr_int(child, ["address", "storageaddress", "offset", "start"])
        if elements is None:
            elements, axis_scaling = _axis_elements_from_address(axis_addr)
            if axis_scaling and not child.attrib.get("scaling"):
                child.attrib["scaling"] = axis_scaling
        if elements is None:
            continue
        if "x axis" in axis_type or tag == "xaxis":
            axis_def = _tabledef_from_elem(child, xml_path, rows=1, cols=elements)
            if axis_def:
                axis_x = _read_axis_values(rom_filename, axis_def, "x", apply_scaling)
        elif "y axis" in axis_type or tag == "yaxis":
            axis_def = _tabledef_from_elem(child, xml_path, rows=elements, cols=1)
            if axis_def:
                axis_y = _read_axis_values(rom_filename, axis_def, "y", apply_scaling)

    if table_def.swapxy and axis_x and axis_y:
        axis_x, axis_y = axis_y, axis_x

    table = _read_table_from_def(rom_filename, table_def, None, None, None, "big", apply_scaling)
    table = _normalize_table_result(table)
    return {
        "table": table,
        "x_axis": axis_x,
        "y_axis": axis_y,
    }


@mcp.tool()
def read_scalar_for_rom(
    rom_filename: str,
    table_name: str,
    endian: str = "big",
) -> dict:
    """Reads a 1D scalar by name using the ROM's XML include chain."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches = _match_tables_in_defs(table_name, table_defs)
    if not matches:
        return {"error": f"No tables matched '{table_name}'."}

    return _read_scalar_from_def(rom_filename, matches[0], endian)


@mcp.tool()
def read_blob_for_rom(
    rom_filename: str,
    table_name: str,
    length: int,
    data_type: str | None = "uint16",
    endian: str = "big",
) -> dict:
    """Reads raw blob data for tables like RAX Direct Boost comp entries."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    if length <= 0:
        return {"error": "length must be > 0."}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches = _match_tables_in_defs(table_name, table_defs)
    if not matches:
        return {"error": f"No tables matched '{table_name}'."}

    table_def = matches[0]
    raw = _read_raw_bytes(rom_filename, table_def.address, length)
    if "data" not in raw:
        return raw
    data = raw["data"]

    values = None
    if data_type:
        fmt, size = _dtype_format(data_type, endian)
        if len(data) % size != 0:
            return {
                "error": "length is not aligned to data_type size",
                "length": len(data),
                "data_type": data_type,
                "size": size,
            }
        count = len(data) // size
        values = [struct.unpack(fmt, data[i * size : (i + 1) * size])[0] for i in range(count)]

    return {
        "table_name": table_def.name,
        "address": f"0x{table_def.address:X}",
        "source_xml": table_def.source_xml,
        "length": len(data),
        "hex": data.hex(),
        "data_type": data_type,
        "endian": endian,
        "values": values,
    }


@mcp.tool()
def read_blob_bits_for_rom(
    rom_filename: str,
    table_name: str,
    length: int,
) -> dict:
    """Reads raw bytes and returns set bit positions for blob-bit tables."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    if length <= 0:
        return {"error": "length must be > 0."}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches = _match_tables_in_defs(table_name, table_defs)
    if not matches:
        return {"error": f"No tables matched '{table_name}'."}

    table_def = matches[0]
    raw = _read_raw_bytes(rom_filename, table_def.address, length)
    if "data" not in raw:
        return raw
    data = raw["data"]
    set_bits = []
    for byte_index, value in enumerate(data):
        for bit in range(8):
            if value & (1 << bit):
                set_bits.append(byte_index * 8 + bit)

    return {
        "table_name": table_def.name,
        "address": f"0x{table_def.address:X}",
        "source_xml": table_def.source_xml,
        "length": len(data),
        "hex": data.hex(),
        "set_bits": set_bits,
    }


@mcp.tool()
def decode_blob_table_for_rom(
    rom_filename: str,
    table_name: str,
    length: int | None = None,
) -> dict:
    """Decodes known blob tables into enums/bitfields when possible."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches = _match_tables_in_defs(table_name, table_defs)
    if not matches:
        return {"error": f"No tables matched '{table_name}'."}

    table_def = matches[0]
    decoder = BLOB_DECODER_REGISTRY.get(table_def.scaling or "")
    if decoder is None:
        return {
            "error": "No decoder registered for this blob scaling.",
            "table_name": table_def.name,
            "scaling": table_def.scaling,
        }

    resolved_length = length or int(decoder.get("length", 0))
    if resolved_length <= 0:
        return {
            "error": "Decoder length not provided.",
            "table_name": table_def.name,
            "scaling": table_def.scaling,
        }

    if decoder.get("data_type") == "bits":
        return read_blob_bits_for_rom(rom_filename, table_def.name, resolved_length)

    data_type = decoder.get("data_type")
    endian = decoder.get("endian", "big")
    raw = read_blob_for_rom(
        rom_filename,
        table_def.name,
        length=resolved_length,
        data_type=data_type,
        endian=endian,
    )
    if "values" not in raw:
        return raw

    decoded = None
    enum_map = decoder.get("enum")
    if enum_map and raw.get("values"):
        decoded = enum_map.get(raw["values"][0])

    return {
        "table_name": table_def.name,
        "address": raw.get("address"),
        "source_xml": table_def.source_xml,
        "length": raw.get("length"),
        "data_type": raw.get("data_type"),
        "endian": raw.get("endian"),
        "values": raw.get("values"),
        "decoded": decoded,
    }


@mcp.tool()
def diagnose_rom_definitions(rom_filename: str, sample_limit: int = 10) -> dict:
    """Reports definition gaps for the ROM XML chain."""
    if sample_limit <= 0:
        return {"error": "sample_limit must be > 0."}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    missing_dims = []
    missing_axes = []
    blob_tables = []
    scaling_gaps = []
    scalings = _parse_scaling_defs()

    for xml_path in chain:
        root = _load_xml_root(xml_path)
        if root is None:
            continue
        for elem in root.iter():
            name = _find_attr_str(elem, ["name", "id", "description"])
            address = _find_attr_int(elem, ["address", "storageaddress", "offset", "start"])
            if not name or address is None:
                continue
            data_type = _find_attr_str(elem, ["type", "datatype", "storage", "storagetype"]) or ""
            scaling = _extract_scaling(elem)
            rows = _find_attr_int(elem, ["rows", "row", "ysize", "y"])
            cols = _find_attr_int(elem, ["columns", "cols", "xsize", "x"])

            if rows is None or cols is None:
                inferred = _infer_table_size(name)
                if inferred == (None, None):
                    axis_has_elements = False
                    for child in elem:
                        tag = child.tag.lower()
                        axis_type = (child.attrib.get("type") or "").lower()
                        if tag in ("xaxis", "yaxis", "axis", "table") or axis_type in ("x axis", "y axis"):
                            elements = _find_attr_int(child, ["elements", "size", "count", "cols", "rows"])
                            if elements is None:
                                axis_addr = _find_attr_int(child, ["address", "storageaddress", "offset", "start"])
                                elements, _ = _axis_elements_from_address(axis_addr)
                            if elements is not None:
                                axis_has_elements = True
                                break
                    if not axis_has_elements:
                        missing_dims.append((name, xml_path, data_type, scaling))
                    else:
                        missing_axes.append((name, xml_path, data_type, scaling))

            if scaling in scalings and scalings[scaling].get("storagetype") == "bloblist":
                blob_tables.append((name, xml_path, data_type, scaling))

            if scaling:
                sdef = scalings.get(scaling, {})
                toexpr = sdef.get("toexpr") or sdef.get("formula")
                if toexpr is None and "x" not in scaling:
                    scaling_gaps.append((name, xml_path, scaling))

    return {
        "rom_filename": rom_filename,
        "chain": chain,
        "missing_dims": {
            "count": len(missing_dims),
            "samples": missing_dims[:sample_limit],
        },
        "missing_axes": {
            "count": len(missing_axes),
            "samples": missing_axes[:sample_limit],
        },
        "blob_tables": {
            "count": len(blob_tables),
            "samples": blob_tables[:sample_limit],
        },
        "scaling_gaps": {
            "count": len(scaling_gaps),
            "samples": scaling_gaps[:sample_limit],
        },
    }


@mcp.tool()
def get_header(rom_filename: str, size: int = 1024) -> str:
    """Reads the first bytes of a ROM for header analysis."""
    rom_path = os.path.join(ROM_PATH, rom_filename)
    if not os.path.isfile(rom_path):
        return f"ROM file not found: {rom_path}"
    with open(rom_path, "rb") as rom_file:
        data = rom_file.read(size)
    return data.hex()


@mcp.tool()
def get_rom_info(rom_filename: str, max_results: int = 20) -> dict:
    """Scans the ROM for ASCII ID strings and matches against XML filenames."""
    rom_path = os.path.join(ROM_PATH, rom_filename)
    if not os.path.isfile(rom_path):
        return {"error": f"ROM file not found: {rom_path}"}

    rom_map = _xml_rom_id_map()
    id_pattern = re.compile(r"\b\d{8}\b")
    candidates: list[dict] = []

    for offset, text in _iter_rom_strings(rom_path, min_len=4):
        for match in id_pattern.finditer(text):
            rom_id = match.group(0)
            candidates.append(
                {
                    "rom_id": rom_id,
                    "offset": f"0x{offset + match.start():X}",
                    "xml_match": rom_map.get(rom_id),
                    "context": text.strip()[:120],
                }
            )
        if len(candidates) >= max_results:
            break

    return {
        "rom_filename": rom_filename,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


@mcp.tool()
def identify_rom(rom_filename: str, max_results: int = 10) -> dict:
    """Identifies ROM by matching internal ID bytes against XML definitions."""
    rom_path = os.path.join(ROM_PATH, rom_filename)
    if not os.path.isfile(rom_path):
        return {"error": f"ROM file not found: {rom_path}"}

    matches: list[dict] = []
    with open(rom_path, "rb") as rom_file:
        for internal_id_hex, address, xml_path in _parse_rom_id_defs():
            try:
                expected = bytes.fromhex(internal_id_hex)
            except ValueError:
                continue
            rom_file.seek(address)
            data = rom_file.read(len(expected))
            if data == expected:
                matches.append(
                    {
                        "rom_id": internal_id_hex,
                        "address": f"0x{address:X}",
                        "xml": xml_path,
                    }
                )
            if len(matches) >= max_results:
                break

    return {
        "rom_filename": rom_filename,
        "matches": matches,
        "match_count": len(matches),
    }


def _tabledef_map_for_chain(chain: list[str]) -> dict[str, TableDef]:
    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))
    mapping: dict[str, TableDef] = {}
    for table_def in table_defs:
        key = table_def.name.lower()
        if key not in mapping:
            mapping[key] = table_def
    return mapping


@mcp.tool()
def compare_roms(
    rom_filename_a: str,
    rom_filename_b: str,
    contains: str = "",
    apply_scaling: bool = False,
    tolerance: float = 0.0,
    limit: int = 100,
    pretty: bool = True,
) -> dict:
    """Compares two ROMs and lists changed tables using the ROM definition chain."""
    if limit <= 0:
        return {"error": "limit must be > 0."}
    if tolerance < 0:
        return {"error": "tolerance must be >= 0."}
    chain_a = _resolve_rom_chain(rom_filename_a)
    chain_b = _resolve_rom_chain(rom_filename_b)
    if not chain_a or not chain_b:
        return {"error": "Unable to resolve ROM definition chain for one or both ROMs."}

    map_a = _tabledef_map_for_chain(chain_a)
    map_b = _tabledef_map_for_chain(chain_b)
    contains = contains.lower().strip()

    shared_names = [
        name for name in map_a.keys() if name in map_b and (not contains or contains in name)
    ]

    changes = []
    errors = []
    for name in shared_names:
        table_a = map_a[name]
        table_b = map_b[name]
        read_a = _read_table_from_def(rom_filename_a, table_a, None, None, None, "big", apply_scaling)
        read_b = _read_table_from_def(rom_filename_b, table_b, None, None, None, "big", apply_scaling)
        if "data" not in read_a or "data" not in read_b:
            errors.append({"table": table_a.name, "a": read_a, "b": read_b})
            continue

        data_a = read_a["data"]
        data_b = read_b["data"]
        if len(data_a) != len(data_b) or any(len(ra) != len(rb) for ra, rb in zip(data_a, data_b)):
            changes.append(
                {
                    "table": table_a.name,
                    "rows_a": len(data_a),
                    "cols_a": len(data_a[0]) if data_a else 0,
                    "rows_b": len(data_b),
                    "cols_b": len(data_b[0]) if data_b else 0,
                    "diff_cells": None,
                }
            )
            continue

        diff_cells = 0
        for row_a, row_b in zip(data_a, data_b):
            for val_a, val_b in zip(row_a, row_b):
                if tolerance == 0.0:
                    changed = val_a != val_b
                else:
                    try:
                        changed = abs(float(val_a) - float(val_b)) > tolerance
                    except (TypeError, ValueError):
                        changed = val_a != val_b
                if changed:
                    diff_cells += 1

        if diff_cells:
            changes.append(
                {
                    "table": table_a.name,
                    "rows": len(data_a),
                    "cols": len(data_a[0]) if data_a else 0,
                    "diff_cells": diff_cells,
                }
            )

        if len(changes) >= limit:
            break

    pretty_changes = [
        f"{c['table']} | {c.get('rows', c.get('rows_a'))}x{c.get('cols', c.get('cols_a'))} | diff_cells={c.get('diff_cells')}"
        for c in changes
    ]

    return {
        "rom_a": rom_filename_a,
        "rom_b": rom_filename_b,
        "filters": {"contains": contains, "apply_scaling": apply_scaling, "tolerance": tolerance},
        "compared_tables": len(shared_names),
        "changed_count": len(changes),
        "errors": errors,
        "changes": pretty_changes if pretty else changes,
    }


def _table_size_for_def(table_def: TableDef) -> tuple[int | None, int | None]:
    rows = table_def.rows
    cols = table_def.cols
    if rows is None or cols is None:
        inferred_rows, inferred_cols = _infer_table_size(table_def.name)
        rows = rows or inferred_rows
        cols = cols or inferred_cols
    return rows, cols


@mcp.tool()
def check_boost_control_consistency(rom_filename: str) -> dict:
    """Checks boost target/WGDC/direct-boost tables for mismatches or missing dependencies."""
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    target_tables = [td for td in table_defs if "boost target" in td.name.lower()]
    wgdc_tables = [
        td
        for td in table_defs
        if "wgdc" in td.name.lower() or "wastegate" in td.name.lower()
    ]
    direct_tables = [
        td
        for td in table_defs
        if "direct boost" in td.name.lower() or "rax" in td.name.lower()
    ]

    warnings = []
    if direct_tables and not target_tables:
        warnings.append("Direct boost tables found without boost target tables.")
    if direct_tables and not wgdc_tables:
        warnings.append("Direct boost tables found without WGDC tables.")

    size_mismatches = []
    if target_tables and wgdc_tables:
        target_rows, target_cols = _table_size_for_def(target_tables[0])
        wgdc_rows, wgdc_cols = _table_size_for_def(wgdc_tables[0])
        if target_rows != wgdc_rows or target_cols != wgdc_cols:
            size_mismatches.append(
                {
                    "target": target_tables[0].name,
                    "wgdc": wgdc_tables[0].name,
                    "target_size": (target_rows, target_cols),
                    "wgdc_size": (wgdc_rows, wgdc_cols),
                }
            )

    def summarize(tables: list[TableDef]) -> list[dict]:
        summary = []
        for td in tables:
            rows, cols = _table_size_for_def(td)
            summary.append(
                {
                    "name": td.name,
                    "rows": rows,
                    "cols": cols,
                    "scaling": td.scaling,
                    "source_xml": td.source_xml,
                }
            )
        return summary

    return {
        "rom_filename": rom_filename,
        "boost_targets": summarize(target_tables),
        "wgdc_tables": summarize(wgdc_tables),
        "direct_boost_tables": summarize(direct_tables),
        "warnings": warnings,
        "size_mismatches": size_mismatches,
    }


@mcp.tool()
def compare_table_between_roms(
    rom_filename_a: str,
    rom_filename_b: str,
    table_name: str,
    apply_scaling: bool = True,
    tolerance: float = 0.0,
) -> dict:
    """Summarizes table deltas between two ROMs without writing changes."""
    if tolerance < 0:
        return {"error": "tolerance must be >= 0."}
    read_a = read_table_for_rom(rom_filename_a, table_name, apply_scaling=apply_scaling)
    if "data" not in read_a:
        return {"error": "Unable to read table from rom_filename_a.", "result": read_a}
    read_b = read_table_for_rom(rom_filename_b, table_name, apply_scaling=apply_scaling)
    if "data" not in read_b:
        return {"error": "Unable to read table from rom_filename_b.", "result": read_b}

    data_a = read_a["data"]
    data_b = read_b["data"]
    if len(data_a) != len(data_b) or any(len(ra) != len(rb) for ra, rb in zip(data_a, data_b)):
        return {
            "rom_a": rom_filename_a,
            "rom_b": rom_filename_b,
            "table": table_name,
            "error": "Table dimensions differ.",
            "rows_a": len(data_a),
            "cols_a": len(data_a[0]) if data_a else 0,
            "rows_b": len(data_b),
            "cols_b": len(data_b[0]) if data_b else 0,
        }

    flat_a = [float(value) for row in data_a for value in row]
    flat_b = [float(value) for row in data_b for value in row]
    deltas = [b - a for a, b in zip(flat_a, flat_b)]

    changed = [delta for delta in deltas if abs(delta) > tolerance]
    max_abs_delta = max((abs(delta) for delta in deltas), default=0.0)
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

    sample_deltas = []
    for idx, delta in enumerate(deltas[:10]):
        if abs(delta) > tolerance:
            sample_deltas.append({"index": idx, "delta": delta})

    return {
        "rom_a": rom_filename_a,
        "rom_b": rom_filename_b,
        "table": table_name,
        "rows": len(data_a),
        "cols": len(data_a[0]) if data_a else 0,
        "diff_cells": len(changed),
        "max_abs_delta": max_abs_delta,
        "avg_delta": avg_delta,
        "sample_deltas": sample_deltas,
    }


@mcp.tool()
def list_logs(contains: str = "", limit: int = 50) -> list[str]:
    """Lists log files in the log directory, optionally filtered by substring."""
    if not os.path.isdir(LOG_PATH):
        return []
    contains = contains.lower().strip()
    results = []
    for filename in sorted(os.listdir(LOG_PATH)):
        if not filename.lower().endswith((".csv", ".log", ".txt")):
            continue
        if contains and contains not in filename.lower():
            continue
        results.append(filename)
        if len(results) >= limit:
            break
    return results


@mcp.tool()
def read_log(filename: str, max_lines: int = 200) -> dict:
    """Reads a log file from the log directory and returns up to max_lines."""
    log_path = _safe_log_path(filename)
    if log_path is None:
        return {"error": "Invalid filename."}
    if not os.path.isfile(log_path):
        return {"error": f"Log file not found: {log_path}"}
    lines: list[str] = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        for idx, line in enumerate(log_file):
            lines.append(line.rstrip("\n"))
            if idx + 1 >= max_lines:
                break
    return {"filename": filename, "lines": lines, "line_count": len(lines)}


@mcp.tool()
def check_log_signals(filename: str, signal_set: str = "launch") -> dict:
    """Checks a log file for required signals using a named signal set."""
    log_path = _safe_log_path(filename)
    if log_path is None:
        return {"error": "Invalid filename."}
    if not os.path.isfile(log_path):
        return {"error": f"Log file not found: {log_path}"}

    required = LOG_SIGNAL_SETS.get(signal_set)
    if required is None:
        return {"error": f"Unknown signal_set: {signal_set}", "signal_sets": list(LOG_SIGNAL_SETS.keys())}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        reader = csv.DictReader(log_file)
        fields = reader.fieldnames or []
        columns = _detect_log_columns(fields)
        missing = _missing_log_columns(columns, required)

    return {
        "filename": filename,
        "signal_set": signal_set,
        "required_signals": required,
        "detected_columns": columns,
        "missing_columns": missing,
    }


@mcp.tool()
def analyze_launch_log(
    filename: str,
    min_tps: float = 70.0,
    max_speed: float = 5.0,
    rpm_min: float = 2500.0,
    rpm_max: float = 5000.0,
    max_samples: int = 10,
    max_rows: int = 200000,
) -> dict:
    """Summarizes a launch window from a log file using common TPS/speed/RPM filters."""
    error = _validate_log_filters(min_tps, max_speed, rpm_min, rpm_max, max_rows)
    if error:
        return {"error": error}
    log_path = _safe_log_path(filename)
    if log_path is None:
        return {"error": "Invalid filename."}
    if not os.path.isfile(log_path):
        return {"error": f"Log file not found: {log_path}"}

    samples: list[dict] = []
    rpms: list[float] = []
    speeds: list[float] = []
    boosts: list[float] = []
    afrs: list[float] = []
    timings: list[float] = []
    knocks: list[float] = []
    times: list[float] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        reader = csv.DictReader(log_file)
        fields = reader.fieldnames or []
        columns = _detect_log_columns(fields)
        missing = _missing_log_columns(columns, ["rpm", "tps", "speed", "load", "time"])
        missing = _missing_log_columns(columns, ["rpm", "tps", "speed", "boost", "afr", "timing", "knock", "load", "time"])

        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            rpm = _safe_float(row.get(columns.get("rpm"))) if columns.get("rpm") else None
            tps = _safe_float(row.get(columns.get("tps"))) if columns.get("tps") else None
            speed = _safe_float(row.get(columns.get("speed"))) if columns.get("speed") else None
            if rpm is None or tps is None or speed is None:
                continue
            if not (speed <= max_speed and tps >= min_tps and rpm_min <= rpm <= rpm_max):
                continue

            boost = _safe_float(row.get(columns.get("boost"))) if columns.get("boost") else None
            afr = _safe_float(row.get(columns.get("afr"))) if columns.get("afr") else None
            timing = _safe_float(row.get(columns.get("timing"))) if columns.get("timing") else None
            knock = _safe_float(row.get(columns.get("knock"))) if columns.get("knock") else None
            ipw = _safe_float(row.get(columns.get("ipw"))) if columns.get("ipw") else None
            idc = _safe_float(row.get(columns.get("idc"))) if columns.get("idc") else None
            load = _safe_float(row.get(columns.get("load"))) if columns.get("load") else None
            time_val = _safe_float(row.get(columns.get("time"))) if columns.get("time") else None

            rpms.append(rpm)
            speeds.append(speed)
            if boost is not None:
                boosts.append(boost)
            if afr is not None:
                afrs.append(afr)
            if timing is not None:
                timings.append(timing)
            if knock is not None:
                knocks.append(knock)
            if time_val is not None:
                times.append(time_val)

            if len(samples) < max_samples:
                samples.append(
                    {
                        "rpm": rpm,
                        "tps": tps,
                        "speed": speed,
                        "boost": boost,
                        "afr": afr,
                        "timing": timing,
                        "knock": knock,
                        "ipw": ipw,
                        "idc": idc,
                        "load": load,
                        "time": time_val,
                    }
                )

    if not rpms:
        return {
            "filename": filename,
            "error": "No launch samples found with current filters.",
            "filters": {
                "min_tps": min_tps,
                "max_speed": max_speed,
                "rpm_min": rpm_min,
                "rpm_max": rpm_max,
            },
            "detected_columns": columns,
            "missing_columns": missing,
        }

    summary = {
        "rpm_min": min(rpms),
        "rpm_max": max(rpms),
        "speed_min": min(speeds) if speeds else None,
        "speed_max": max(speeds) if speeds else None,
        "boost_min": min(boosts) if boosts else None,
        "boost_max": max(boosts) if boosts else None,
        "afr_min": min(afrs) if afrs else None,
        "afr_max": max(afrs) if afrs else None,
        "timing_min": min(timings) if timings else None,
        "timing_max": max(timings) if timings else None,
        "knock_min": min(knocks) if knocks else None,
        "knock_max": max(knocks) if knocks else None,
        "time_start": min(times) if times else None,
        "time_end": max(times) if times else None,
        "sample_count": len(rpms),
    }

    pretty_samples = [
        " | ".join(
            f"{key}={value}" for key, value in sample.items() if value is not None
        )
        for sample in samples
    ]

    return {
        "filename": filename,
        "filters": {
            "min_tps": min_tps,
            "max_speed": max_speed,
            "rpm_min": rpm_min,
            "rpm_max": rpm_max,
        },
        "detected_columns": columns,
        "missing_columns": missing,
        "summary": summary,
        "samples": pretty_samples,
    }


def _scalar_value_or_error(rom_filename: str, table_name: str) -> dict:
    result = read_scalar_for_rom(rom_filename, table_name)
    if "value" not in result:
        return {"error": result}
    return {"value": result["value"], "units": result.get("units"), "table": result.get("table_name")}


def _table_min_max(table: list[list[float]]) -> dict:
    flat = [float(value) for row in table for value in row]
    if not flat:
        return {"min": None, "max": None}
    return {"min": min(flat), "max": max(flat)}


@mcp.tool()
def analyze_launch_health(
    rom_filename: str,
    log_filename: str,
    min_tps: float = 70.0,
    max_speed: float = 5.0,
    rpm_min: float = 2500.0,
    rpm_max: float = 5000.0,
) -> dict:
    """Combines ROM tables and log data into a launch health report."""
    launch = analyze_launch_log(
        log_filename,
        min_tps=min_tps,
        max_speed=max_speed,
        rpm_min=rpm_min,
        rpm_max=rpm_max,
    )
    if "summary" not in launch:
        return {"error": "Launch log summary unavailable.", "launch": launch}

    signal_check = check_log_signals(log_filename, "launch")

    rom_scalars = {
        "launch_speed_limit": _scalar_value_or_error(
            rom_filename, "Maximum Speed permitted for Launch Maps"
        ),
        "launch_tps_min": _scalar_value_or_error(
            rom_filename, "Minimum TPS required for NLTS or Launch Maps"
        ),
        "stationary_rev_limit": _scalar_value_or_error(
            rom_filename, "Stationary Rev Limiter"
        ),
    }

    launch_timing = read_table_for_rom(rom_filename, "Launch High Octane Timing Map")
    launch_fuel = read_table_for_rom(rom_filename, "Launch High Octane Fuel Map")

    timing_stats = None
    if "data" in launch_timing:
        timing_stats = _table_min_max(launch_timing["data"])

    fuel_stats = None
    if "data" in launch_fuel:
        fuel_stats = _table_min_max(launch_fuel["data"])

    summary = launch["summary"]
    flags = []
    timing_floor = LAUNCH_HEALTH_SETTINGS.get("timing_floor", -5.0)
    boost_oscillation = LAUNCH_HEALTH_SETTINGS.get("boost_oscillation_threshold", 5.0)
    knock_threshold = LAUNCH_HEALTH_SETTINGS.get("knock_threshold", 0.0)

    if summary.get("timing_min") is not None and summary["timing_min"] < timing_floor:
        flags.append({"issue": "timing_below_floor", "timing_min": summary["timing_min"]})

    boost_min = summary.get("boost_min")
    boost_max = summary.get("boost_max")
    if boost_min is not None and boost_max is not None:
        if boost_max - boost_min >= boost_oscillation:
            flags.append({"issue": "boost_oscillation", "boost_delta": boost_max - boost_min})

    if summary.get("knock_max") is not None and summary["knock_max"] > knock_threshold:
        flags.append({"issue": "knock_detected", "knock_max": summary["knock_max"]})

    speed_max = summary.get("speed_max")
    launch_speed_limit = rom_scalars.get("launch_speed_limit", {}).get("value")
    if speed_max is not None and launch_speed_limit is not None:
        if speed_max > launch_speed_limit:
            flags.append(
                {
                    "issue": "launch_speed_exceeds_limit",
                    "speed_max": speed_max,
                    "limit": launch_speed_limit,
                }
            )

    return {
        "rom_filename": rom_filename,
        "log_filename": log_filename,
        "launch_summary": summary,
        "signal_check": signal_check,
        "rom_scalars": rom_scalars,
        "launch_timing_stats": timing_stats,
        "launch_fuel_stats": fuel_stats,
        "flags": flags,
    }


@mcp.tool()
def estimate_low_rpm_torque_risk(
    filename: str,
    rpm_threshold: float | None = None,
    boost_threshold: float | None = None,
    timing_floor: float | None = None,
    knock_threshold: float | None = None,
    min_tps: float | None = None,
    max_speed: float | None = None,
    max_rows: int = 200000,
) -> dict:
    """Estimates low-RPM torque risk from log samples using configurable thresholds."""
    log_path = _safe_log_path(filename)
    if log_path is None:
        return {"error": "Invalid filename."}
    if not os.path.isfile(log_path):
        return {"error": f"Log file not found: {log_path}"}
    if max_rows <= 0:
        return {"error": "max_rows must be > 0."}

    rpm_threshold = rpm_threshold if rpm_threshold is not None else TORQUE_RISK_SETTINGS.get("rpm_threshold", 3500.0)
    boost_threshold = boost_threshold if boost_threshold is not None else TORQUE_RISK_SETTINGS.get("boost_threshold", 18.0)
    timing_floor = timing_floor if timing_floor is not None else TORQUE_RISK_SETTINGS.get("timing_floor", 0.0)
    knock_threshold = knock_threshold if knock_threshold is not None else TORQUE_RISK_SETTINGS.get("knock_threshold", 0.0)

    total_samples = 0
    low_rpm_samples = 0
    risk_samples = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        reader = csv.DictReader(log_file)
        fields = reader.fieldnames or []
        columns = _detect_log_columns(fields)
        missing = _missing_log_columns(columns, ["rpm", "boost", "timing", "knock"])

        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            rpm = _safe_float(row.get(columns.get("rpm"))) if columns.get("rpm") else None
            if rpm is None:
                continue
            if min_tps is not None:
                tps = _safe_float(row.get(columns.get("tps"))) if columns.get("tps") else None
                if tps is None or tps < min_tps:
                    continue
            if max_speed is not None:
                speed = _safe_float(row.get(columns.get("speed"))) if columns.get("speed") else None
                if speed is None or speed > max_speed:
                    continue

            total_samples += 1
            if rpm <= rpm_threshold:
                low_rpm_samples += 1
                boost = _safe_float(row.get(columns.get("boost"))) if columns.get("boost") else None
                timing = _safe_float(row.get(columns.get("timing"))) if columns.get("timing") else None
                knock = _safe_float(row.get(columns.get("knock"))) if columns.get("knock") else None

                boost_flag = boost is not None and boost >= boost_threshold
                timing_flag = timing is not None and timing <= timing_floor
                knock_flag = knock is not None and knock > knock_threshold

                if boost_flag and (timing_flag or knock_flag):
                    risk_samples += 1

    risk_ratio = (risk_samples / low_rpm_samples) if low_rpm_samples else 0.0

    return {
        "filename": filename,
        "thresholds": {
            "rpm_threshold": rpm_threshold,
            "boost_threshold": boost_threshold,
            "timing_floor": timing_floor,
            "knock_threshold": knock_threshold,
        },
        "filters": {
            "min_tps": min_tps,
            "max_speed": max_speed,
        },
        "total_samples": total_samples,
        "low_rpm_samples": low_rpm_samples,
        "risk_samples": risk_samples,
        "risk_ratio": risk_ratio,
        "missing_columns": missing,
    }


@mcp.tool()
def compare_launch_logs(
    filename_a: str,
    filename_b: str,
    min_tps: float = 70.0,
    max_speed: float = 5.0,
    rpm_min: float = 2500.0,
    rpm_max: float = 5000.0,
) -> dict:
    """Compares launch summaries between two logs using the same filters."""
    error = _validate_log_filters(min_tps, max_speed, rpm_min, rpm_max, 1)
    if error:
        return {"error": error}
    result_a = analyze_launch_log(
        filename_a,
        min_tps=min_tps,
        max_speed=max_speed,
        rpm_min=rpm_min,
        rpm_max=rpm_max,
    )
    result_b = analyze_launch_log(
        filename_b,
        min_tps=min_tps,
        max_speed=max_speed,
        rpm_min=rpm_min,
        rpm_max=rpm_max,
    )

    if "summary" not in result_a or "summary" not in result_b:
        return {
            "filename_a": filename_a,
            "filename_b": filename_b,
            "error": "One or both logs did not yield launch samples.",
            "result_a": result_a,
            "result_b": result_b,
        }

    summary_a = result_a["summary"]
    summary_b = result_b["summary"]
    keys = [
        "rpm_min",
        "rpm_max",
        "boost_min",
        "boost_max",
        "afr_min",
        "afr_max",
        "timing_min",
        "timing_max",
        "knock_min",
        "knock_max",
    ]

    def delta(key: str) -> float | None:
        a_val = summary_a.get(key)
        b_val = summary_b.get(key)
        if a_val is None or b_val is None:
            return None
        return b_val - a_val

    diff = {key: delta(key) for key in keys}

    pretty = [
        f"{key}: {summary_a.get(key)} -> {summary_b.get(key)} (delta {diff.get(key)})"
        for key in keys
    ]

    return {
        "filename_a": filename_a,
        "filename_b": filename_b,
        "filters": {
            "min_tps": min_tps,
            "max_speed": max_speed,
            "rpm_min": rpm_min,
            "rpm_max": rpm_max,
        },
        "summary_a": summary_a,
        "summary_b": summary_b,
        "diff": diff,
        "pretty": pretty,
    }


@mcp.tool()
def extract_launch_window(
    filename: str,
    output_filename: str | None = None,
    min_tps: float = 70.0,
    max_speed: float = 5.0,
    rpm_min: float = 2500.0,
    rpm_max: float = 5000.0,
    max_rows: int = 200000,
    return_series: bool = True,
) -> dict:
    """Extracts launch-window rows and returns structured series data."""
    error = _validate_log_filters(min_tps, max_speed, rpm_min, rpm_max, max_rows)
    if error:
        return {"error": error}
    log_path = _safe_log_path(filename)
    if log_path is None:
        return {"error": "Invalid filename."}
    if not os.path.isfile(log_path):
        return {"error": f"Log file not found: {log_path}"}

    if output_filename:
        output_path = _safe_log_path(output_filename)
        if output_path is None:
            return {"error": "Invalid output filename."}
    else:
        base, _ = os.path.splitext(filename)
        output_path = os.path.join(LOG_PATH, f"{base}_launch.csv")

    rows_written = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        reader = csv.DictReader(log_file)
        fields = reader.fieldnames or []
        columns = _detect_log_columns(fields)
        missing = _missing_log_columns(columns, ["rpm", "tps", "speed", "boost", "afr", "timing", "knock", "load", "time"])
        if not fields:
            return {"error": "No headers found in log file."}

        series: dict[str, list[float | None]] = {
            "time": [],
            "rpm": [],
            "tps": [],
            "speed": [],
            "boost": [],
            "afr": [],
            "timing": [],
            "knock": [],
            "ipw": [],
            "idc": [],
            "load": [],
        }

        with open(output_path, "w", newline="", encoding="utf-8") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fields)
            writer.writeheader()

            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                rpm = _safe_float(row.get(columns.get("rpm"))) if columns.get("rpm") else None
                tps = _safe_float(row.get(columns.get("tps"))) if columns.get("tps") else None
                speed = _safe_float(row.get(columns.get("speed"))) if columns.get("speed") else None
                if rpm is None or tps is None or speed is None:
                    continue
                if not (speed <= max_speed and tps >= min_tps and rpm_min <= rpm <= rpm_max):
                    continue
                writer.writerow(row)
                rows_written += 1
                if return_series:
                    series["rpm"].append(rpm)
                    series["tps"].append(tps)
                    series["speed"].append(speed)
                    series["boost"].append(_safe_float(row.get(columns.get("boost"))) if columns.get("boost") else None)
                    series["afr"].append(_safe_float(row.get(columns.get("afr"))) if columns.get("afr") else None)
                    series["timing"].append(_safe_float(row.get(columns.get("timing"))) if columns.get("timing") else None)
                    series["knock"].append(_safe_float(row.get(columns.get("knock"))) if columns.get("knock") else None)
                    series["ipw"].append(_safe_float(row.get(columns.get("ipw"))) if columns.get("ipw") else None)
                    series["idc"].append(_safe_float(row.get(columns.get("idc"))) if columns.get("idc") else None)
                    series["load"].append(_safe_float(row.get(columns.get("load"))) if columns.get("load") else None)
                    series["time"].append(_safe_float(row.get(columns.get("time"))) if columns.get("time") else None)

    return {
        "filename": filename,
        "output_filename": os.path.basename(output_path),
        "rows_written": rows_written,
        "filters": {
            "min_tps": min_tps,
            "max_speed": max_speed,
            "rpm_min": rpm_min,
            "rpm_max": rpm_max,
        },
        "detected_columns": columns,
        "missing_columns": missing,
        "series": series if return_series else None,
    }


def _nearest_index(values: list[float], target: float) -> int | None:
    if not values:
        return None
    best_idx = 0
    best_diff = abs(values[0] - target)
    for idx, value in enumerate(values[1:], start=1):
        diff = abs(value - target)
        if diff < best_diff:
            best_idx = idx
            best_diff = diff
    return best_idx


def _pick_log_value(source: str, row: dict, columns: dict[str, str | None]) -> float | None:
    field = columns.get(source)
    if not field:
        return None
    return _safe_float(row.get(field))


@mcp.tool()
def map_log_to_table(
    rom_filename: str,
    table_name: str,
    log_filename: str,
    x_source: str | None = None,
    y_source: str | None = None,
    min_tps: float = 70.0,
    max_speed: float = 5.0,
    rpm_min: float = 2500.0,
    rpm_max: float = 5000.0,
    max_rows: int = 200000,
    top_n: int = 10,
) -> dict:
    """Maps log samples to table cell indices using axis values."""
    error = _validate_log_filters(min_tps, max_speed, rpm_min, rpm_max, max_rows, top_n=top_n)
    if error:
        return {"error": error}
    table_bundle = read_table_with_axes_for_rom(rom_filename, table_name)
    if "table" not in table_bundle:
        return table_bundle

    x_axis = table_bundle.get("x_axis")
    y_axis = table_bundle.get("y_axis")
    if not x_axis or not y_axis:
        return {"error": "Axis data not found for table."}

    log_path = _safe_log_path(log_filename)
    if log_path is None:
        return {"error": "Invalid filename."}
    if not os.path.isfile(log_path):
        return {"error": f"Log file not found: {log_path}"}

    missing: list[str] = []
    counts: dict[tuple[int, int], int] = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        reader = csv.DictReader(log_file)
        fields = reader.fieldnames or []
        columns = _detect_log_columns(fields)

        if x_source is None:
            if "rpm" in (x_axis.get("name") or "").lower():
                x_source = "rpm"
            elif "load" in (x_axis.get("name") or "").lower():
                x_source = "load"
            elif "throttle" in (x_axis.get("name") or "").lower():
                x_source = "tps"
        if y_source is None:
            if "rpm" in (y_axis.get("name") or "").lower():
                y_source = "rpm"
            elif "load" in (y_axis.get("name") or "").lower():
                y_source = "load"
            elif "throttle" in (y_axis.get("name") or "").lower():
                y_source = "tps"

        if x_source is None or y_source is None:
            return {"error": "Unable to infer axis sources; provide x_source/y_source."}

        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            rpm = _pick_log_value("rpm", row, columns)
            tps = _pick_log_value("tps", row, columns)
            speed = _pick_log_value("speed", row, columns)
            if rpm is None or tps is None or speed is None:
                continue
            if not (speed <= max_speed and tps >= min_tps and rpm_min <= rpm <= rpm_max):
                continue

            x_val = _pick_log_value(x_source, row, columns)
            y_val = _pick_log_value(y_source, row, columns)
            if x_val is None or y_val is None:
                continue

            x_idx = _nearest_index([float(v) for v in x_axis.get("values", [])], x_val)
            y_idx = _nearest_index([float(v) for v in y_axis.get("values", [])], y_val)
            if x_idx is None or y_idx is None:
                continue
            key = (y_idx, x_idx)
            counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    pretty = [
        f"row={row} col={col} count={count} x={x_axis['values'][col]} y={y_axis['values'][row]}"
        for (row, col), count in ranked
    ]

    return {
        "rom": rom_filename,
        "table": table_name,
        "log": log_filename,
        "x_source": x_source,
        "y_source": y_source,
        "filters": {
            "min_tps": min_tps,
            "max_speed": max_speed,
            "rpm_min": rpm_min,
            "rpm_max": rpm_max,
        },
        "detected_columns": columns,
        "missing_columns": missing,
        "top_cells": pretty,
        "total_cells": len(counts),
    }


def _validate_allowlist(table_name: str) -> bool:
    return _allowlist_match(table_name, _get_allowlist())


def _encode_values(values: list[float], data_type: str, endian: str) -> bytes:
    fmt, size = _dtype_format(data_type, endian)
    encoded = bytearray()
    for value in values:
        if fmt.endswith("B"):
            value = max(0, min(255, int(round(value))))
        elif fmt.endswith("b"):
            value = max(-128, min(127, int(round(value))))
        elif fmt.endswith("H"):
            value = max(0, min(65535, int(round(value))))
        elif fmt.endswith("h"):
            value = max(-32768, min(32767, int(round(value))))
        encoded.extend(struct.pack(fmt, value))
    return bytes(encoded)


def _inverse_scale(values: list[float], table_def: TableDef) -> tuple[list[float], str | None]:
    scaling_defs = _parse_scaling_defs()
    if not table_def.scaling:
        return values, None
    scaling_def = scaling_defs.get(table_def.scaling, {})
    expr = scaling_def.get("frexpr") or scaling_def.get("fromexpr")
    if not expr:
        return values, None
    return [_safe_eval_inverse(expr, float(v)) for v in values], expr


@mcp.tool()
def write_table(
    rom_filename: str,
    table_name: str,
    data: list[list[float]],
    output_filename: str | None = None,
    overwrite_output: bool = False,
    max_delta: float | None = None,
    use_default_max_delta: bool = DEFAULT_USE_MAX_DELTA,
    write_metadata: bool = True,
) -> dict:
    """Writes an entire table to a new ROM file (with safeguards)."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    matches = _match_tables(table_name)
    if not matches:
        return {"error": f"No tables matched '{table_name}'."}
    return _write_table_from_def(
        rom_filename,
        matches[0],
        data,
        output_filename,
        overwrite_output,
        max_delta,
        use_default_max_delta,
        write_metadata,
    )


@mcp.tool()
def write_cell(
    rom_filename: str,
    table_name: str,
    row: int,
    col: int,
    value: float,
    output_filename: str | None = None,
    overwrite_output: bool = False,
    max_delta: float | None = None,
    use_default_max_delta: bool = DEFAULT_USE_MAX_DELTA,
    write_metadata: bool = True,
) -> dict:
    """Writes a single cell to a new ROM file (with safeguards)."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    current = read_table(rom_filename, table_name)
    if "data" not in current:
        return current
    data = current["data"]
    if row < 0 or col < 0 or row >= len(data) or col >= len(data[0]):
        return {"error": "Cell index out of bounds."}
    if max_delta is not None:
        if abs(float(value) - float(data[row][col])) > max_delta:
            return {"error": "Max delta exceeded", "old": data[row][col], "new": value}
    data[row][col] = value
    return write_table(
        rom_filename,
        table_name,
        data,
        output_filename=output_filename,
        overwrite_output=overwrite_output,
        max_delta=max_delta,
        use_default_max_delta=use_default_max_delta,
        write_metadata=write_metadata,
    )


@mcp.tool()
def write_table_for_rom(
    rom_filename: str,
    table_name: str,
    data: list[list[float]],
    output_filename: str | None = None,
    overwrite_output: bool = False,
    max_delta: float | None = None,
    use_default_max_delta: bool = DEFAULT_USE_MAX_DELTA,
    write_metadata: bool = True,
) -> dict:
    """Writes a table using the ROM's XML include chain for matching."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches = _match_tables_in_defs(table_name, table_defs)
    if not matches:
        return {"error": f"No tables matched '{table_name}'."}

    return _write_table_from_def(
        rom_filename,
        matches[0],
        data,
        output_filename,
        overwrite_output,
        max_delta,
        use_default_max_delta,
        write_metadata,
    )


@mcp.tool()
def preview_write_table_for_rom(
    rom_filename: str,
    table_name: str,
    data: list[list[float]],
    max_delta: float | None = None,
    use_default_max_delta: bool = DEFAULT_USE_MAX_DELTA,
) -> dict:
    """Previews a table write and reports deltas without writing."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    chain = _resolve_rom_chain(rom_filename)
    if not chain:
        return {"error": "Unable to resolve ROM definition chain."}

    table_defs: list[TableDef] = []
    for xml_path in chain:
        table_defs.extend(_parse_table_defs_for_xml(xml_path))

    matches = _match_tables_in_defs(table_name, table_defs)
    if not matches:
        return {"error": f"No tables matched '{table_name}'."}

    return _preview_write_table_from_def(
        rom_filename,
        matches[0],
        data,
        max_delta,
        use_default_max_delta,
    )


@mcp.tool()
def write_cell_for_rom(
    rom_filename: str,
    table_name: str,
    row: int,
    col: int,
    value: float,
    output_filename: str | None = None,
    overwrite_output: bool = False,
    max_delta: float | None = None,
    use_default_max_delta: bool = DEFAULT_USE_MAX_DELTA,
    write_metadata: bool = True,
) -> dict:
    """Writes a single cell using the ROM's XML include chain for matching."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    current = read_table_for_rom(rom_filename, table_name)
    if "data" not in current:
        return current
    data = current["data"]
    if row < 0 or col < 0 or row >= len(data) or col >= len(data[0]):
        return {"error": "Cell index out of bounds."}
    if max_delta is not None:
        if abs(float(value) - float(data[row][col])) > max_delta:
            return {"error": "Max delta exceeded", "old": data[row][col], "new": value}
    data[row][col] = value
    return write_table_for_rom(
        rom_filename,
        table_name,
        data,
        output_filename=output_filename,
        overwrite_output=overwrite_output,
        max_delta=max_delta,
        use_default_max_delta=use_default_max_delta,
        write_metadata=write_metadata,
    )


@mcp.tool()
def preview_write_cell_for_rom(
    rom_filename: str,
    table_name: str,
    row: int,
    col: int,
    value: float,
    max_delta: float | None = None,
    use_default_max_delta: bool = DEFAULT_USE_MAX_DELTA,
) -> dict:
    """Previews a single-cell write without writing."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    current = read_table_for_rom(rom_filename, table_name)
    if "data" not in current:
        return current
    data = current["data"]
    if row < 0 or col < 0 or row >= len(data) or col >= len(data[0]):
        return {"error": "Cell index out of bounds."}
    data[row][col] = value
    return preview_write_table_for_rom(
        rom_filename,
        table_name,
        data,
        max_delta=max_delta,
        use_default_max_delta=use_default_max_delta,
    )


@mcp.tool()
def write_scalar_for_rom(
    rom_filename: str,
    table_name: str,
    value: float,
    output_filename: str | None = None,
    overwrite_output: bool = False,
    max_delta: float | None = None,
    use_default_max_delta: bool = DEFAULT_USE_MAX_DELTA,
    write_metadata: bool = True,
) -> dict:
    """Writes a 1D scalar by name using the ROM's XML include chain."""
    error = _validate_table_name(table_name)
    if error:
        return {"error": error}
    return write_table_for_rom(
        rom_filename,
        table_name,
        [[value]],
        output_filename=output_filename,
        overwrite_output=overwrite_output,
        max_delta=max_delta,
        use_default_max_delta=use_default_max_delta,
        write_metadata=write_metadata,
    )


@mcp.resource("tuning://current_tune")
def current_tune_resource() -> str:
    """Provides the current active tune details."""
    return "ROM ID: 55570006 | Turbo: GTX3576 | Injectors: ID1050x"


if __name__ == "__main__":
    mcp.run(transport="stdio")
