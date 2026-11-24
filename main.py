import os, json, math, time, uuid
from pathlib import Path
from typing import Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import csv
import io

app = FastAPI(title="NetVision AI Server", version="0.2.0")
API_KEY = os.getenv("NETVISION_API_KEY", "NetVision")
DATA_DIR = Path(os.getenv("DATA_DIR", "dataset"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = DATA_DIR / "samples.jsonl"
FIELDS = ["deltaYaw", "deltaPitch", "accelYaw", "accelPitch", "jerkPitch", "jerkYaw", "gcdErrorYaw", "gcdErrorPitch"]

# FlatBuffers table helpers: sequence.ticks is field 0; each tick has 8 float fields.
def _table_pos(buf: bytes, pos: int) -> int:
    return pos - int.from_bytes(buf[pos:pos+4], "little", signed=True)
def _field_pos(buf: bytes, table: int, field_index: int):
    vtable = _table_pos(buf, table)
    vlen = int.from_bytes(buf[vtable:vtable+2], "little")
    entry = vtable + 4 + field_index * 2
    if entry + 2 > vtable + vlen:
        return 0
    off = int.from_bytes(buf[entry:entry+2], "little")
    return table + off if off else 0
def _indirect(buf: bytes, pos: int) -> int:
    return pos + int.from_bytes(buf[pos:pos+4], "little")
def _decode_flatbuffer(buf: bytes):
    if len(buf) < 8:
        raise ValueError("Payload too small for FlatBuffers")
    root = int.from_bytes(buf[:4], "little")
    root_table = root
    vec_field = _field_pos(buf, root_table, 0)
    if not vec_field:
        return []
    vec = _indirect(buf, vec_field)
    count = int.from_bytes(buf[vec:vec+4], "little")
    if count > 10000:
        raise ValueError("Tick vector too large")
    start = vec + 4
    ticks = []
    for i in range(count):
        element = start + i * 4
        tick_table = _indirect(buf, element)
        row = {}
        for j, name in enumerate(FIELDS):
            p = _field_pos(buf, tick_table, j)
            # FlatBuffers default for omitted float fields is 0.
            row[name] = struct.unpack_from("<f", buf, p)[0] if p else 0.0
        ticks.append(row)
    return ticks

import struct

def save_samples(ticks):
    stamp = int(time.time())
    session = uuid.uuid4().hex[:12]
    with DATA_FILE.open("a", encoding="utf-8") as f:
        for index, tick in enumerate(ticks):
            f.write(json.dumps({"timestamp": stamp, "session_id": session, "index": index, "label": "unlabeled", **tick}, ensure_ascii=False) + "\n")

@app.get("/")
def root():
    return {"name": "NetVision AI Server", "status": "compatible-receiver", "model_ready": False,
            "protocol": "FlatBuffers TickDataSequence", "note": "Receives and records data; no trained model."}

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": False}

@app.post("/")
async def netvision_request(request: Request):
    supplied = request.headers.get("X-API-Key", "")
    if API_KEY and supplied != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    body = await request.body()
    try:
        ticks = _decode_flatbuffer(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FlatBuffers payload: {exc}")
    save_samples(ticks)
    # Valid response shape for NetVision. This is intentionally neutral until a model is trained.
    return {"probability": 0.0}

@app.get("/count")
def count():
    if not DATA_FILE.exists():
        return {"samples": 0}
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return {"samples": sum(1 for _ in f)}

@app.get("/dataset-info")
def dataset_info():
    return {"file": str(DATA_FILE), "exists": DATA_FILE.exists(), "size_bytes": DATA_FILE.stat().st_size if DATA_FILE.exists() else 0,
            "samples": count()["samples"], "model_ready": False}


@app.get("/export")
def export_jsonl():
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="Dataset is empty")
    return FileResponse(DATA_FILE, media_type="application/x-ndjson", filename="netvision-dataset.jsonl")

@app.get("/export.csv")
def export_csv():
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="Dataset is empty")
    output = io.StringIO()
    columns = ["timestamp", "session_id", "index", "label", *FIELDS]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    with DATA_FILE.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                writer.writerow(json.loads(line))
            except json.JSONDecodeError:
                continue
    data = output.getvalue().encode("utf-8-sig")
    return StreamingResponse(iter([data]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=netvision-dataset.csv"})
