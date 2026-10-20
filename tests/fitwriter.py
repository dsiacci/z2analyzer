"""A tiny FIT encoder for the tests: file_id, session and record messages, with valid CRCs."""

from __future__ import annotations

import struct
from datetime import datetime, timezone

FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)
_CRC = [0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
        0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400]

# (field number, struct code, FIT base type)
FILE_ID = [(0, "B", 0x00), (1, "H", 0x84), (4, "I", 0x86)]
SESSION = [(253, "I", 0x86), (5, "B", 0x00), (6, "B", 0x00)]
RECORD = [(253, "I", 0x86), (0, "i", 0x85), (1, "i", 0x85), (3, "B", 0x02), (4, "B", 0x02),
          (5, "I", 0x86), (7, "H", 0x84), (13, "b", 0x01), (73, "I", 0x86), (78, "I", 0x86)]


def crc16(data: bytes, crc: int = 0) -> int:
    for byte in data:
        tmp = _CRC[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC[byte & 0xF]
        tmp = _CRC[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC[(byte >> 4) & 0xF]
    return crc


def _definition(local: int, global_num: int, fields) -> bytes:
    out = struct.pack("<BBBHB", 0x40 | local, 0, 0, global_num, len(fields))
    for num, code, base in fields:
        out += struct.pack("<BBB", num, struct.calcsize("<" + code), base)
    return out


def _data(local: int, fields, values) -> bytes:
    return struct.pack("<B", local) + struct.pack("<" + "".join(c for _, c, _ in fields), *values)


def fit_ts(dt: datetime) -> int:
    return int((dt - FIT_EPOCH).total_seconds())


def write_fit(path, start: datetime, samples: list[dict], manufacturer: int = 1, sub_sport: int = 7) -> None:
    """samples: dicts with lat, lon (degrees), hr, cadence, dist (m), power, temp, speed (m/s), alt (m), t (s)."""
    body = _definition(0, 0, FILE_ID) + _data(0, FILE_ID, [4, manufacturer, fit_ts(start)])
    body += _definition(1, 20, RECORD)
    for s in samples:
        body += _data(1, RECORD, [
            fit_ts(start) + int(s["t"]),
            int(round(s["lat"] * 2**31 / 180.0)), int(round(s["lon"] * 2**31 / 180.0)),
            int(s["hr"]), int(s["cadence"]), int(round(s["dist"] * 100)), int(s["power"]), int(s["temp"]),
            int(round(s["speed"] * 1000)), int(round((s["alt"] + 500) * 5)),
        ])
    body += _definition(2, 18, SESSION) + _data(2, SESSION, [fit_ts(start) + len(samples), 2, sub_sport])
    header = struct.pack("<BBHI4s", 14, 0x20, 2132, len(body), b".FIT")
    header += struct.pack("<H", crc16(header))
    data = header + body
    with open(path, "wb") as f:
        f.write(data + struct.pack("<H", crc16(data)))
