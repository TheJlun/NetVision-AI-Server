import os
import json
import time
import uuid
import struct
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException

app = FastAPI(
    title="NetVision AI Server",
    version="0.2.0"
)

API_KEY = os.getenv("NETVISION_API_KEY", "NetVision")
DATA_DIR = Path(os.getenv("DATA_DIR", "dataset"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "samples.jsonl"

FIELDS = [
    "deltaYaw",
    "deltaPitch",
    "accelYaw",
    "accelPitch",
    "jerkPitch",
    "jerkYaw",
    "gcdErrorYaw",
    "gcdErrorPitch"
]


def table_pos(buf, pos):
    offset = int.from_bytes(
        buf[pos:pos + 4],
        "little",
        signed=True
    )
    return pos - offset


def field_pos(buf, table, field_index):
    vtable = table_pos(buf, table)

    vlen = int.from_bytes(
        buf[vtable:vtable + 2],
        "little"
    )

    entry = vtable + 4 + field_index * 2

    if entry + 2 > vtable + vlen:
        return 0

    offset = int.from_bytes(
        buf[entry:entry + 2],
        "little"
    )

    return table + offset if offset else 0


def indirect(buf, pos):
    return pos + int.from_bytes(
        buf[pos:pos + 4],
        "little"
    )


def decode_flatbuffer(buf):
    if len(buf) < 8:
        raise ValueError("Payload too small")

    root = int.from_bytes(buf[:4], "little")

    vec_field = field_pos(buf, root, 0)

    if not vec_field:
        return []

    vec = indirect(buf, vec_field)

    count = int.from_bytes(
        buf[vec:vec + 4],
        "little"
    )

    if count > 10000:
        raise ValueError("Too many ticks")

    start = vec + 4
    ticks = []

    for i in range(count):
        element = start + i * 4
        tick_table = indirect(buf, element)

        row = {}

        for j, name in enumerate(FIELDS):
            pos = field_pos(buf, tick_table, j)

            if pos:
                value = struct.unpack_from(
                    "<f", buf, pos
                )[0]
            else:
                value = 0.0

            row[name] = value

        ticks.append(row)

    return ticks


def save_samples(ticks):
    timestamp = int(time.time())
    session = uuid.uuid4().hex[:12]

    with DATA_FILE.open("a", encoding="utf-8") as file:
        for index, tick in enumerate(ticks):
            sample = {
                "timestamp": timestamp,
                "session_id": session,
                "index": index,
                **tick
            }

            file.write(
                json.dumps(sample, ensure_ascii=False)
                + "\n"
            )


@app.get("/")
def root():
    return {
        "name": "NetVision AI Server",
        "status": "compatible-receiver",
        "model_ready": False,
        "protocol": "FlatBuffers TickDataSequence",
        "note": "Receives and records data; no trained model."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": False
    }


@app.post("/")
async def netvision_request(request: Request):
    supplied = request.headers.get("X-API-Key", "")

    if API_KEY and supplied != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    body = await request.body()

    try:
        ticks = decode_flatbuffer(body)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid FlatBuffers payload: {exc}"
        )

    save_samples(ticks)

    return {
        "probability": 0.0
    }


@app.get("/count")
def count():
    if not DATA_FILE.exists():
        return {"samples": 0}

    with DATA_FILE.open("r", encoding="utf-8") as file:
        total = sum(1 for _ in file)

    return {"samples": total}


@app.get("/dataset-info")
def dataset_info():
    return {
        "file": str(DATA_FILE),
        "exists": DATA_FILE.exists(),
        "size_bytes": (
            DATA_FILE.stat().st_size
            if DATA_FILE.exists()
            else 0
        ),
        "samples": count()["samples"],
        "model_ready": False
    }
