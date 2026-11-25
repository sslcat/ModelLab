import json
import os
import sqlite3
import time
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path

from .datasets import canon, digest, dispatch_rows, validate

ROOT = Path(__file__).resolve().parents[1]


def home():
    return Path(os.environ.get("MODELLAB_STATE", ROOT / "state")).resolve()


@contextmanager
def db(write=False):
    home().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home() / "lab.sqlite3", timeout=20)
    conn.row_factory = sqlite3.Row
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with db(True) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, meta TEXT NOT NULL, payload BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, doc TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT NOT NULL);
        """)
        con.execute(
            "INSERT OR IGNORE INTO kv VALUES('namespace',?)", (uuid.uuid4().hex[:12],)
        )
    from research.receipts import locked_bundle, metadata

    controlled = metadata(locked_bundle())
    with db(True) as con:
        con.execute("INSERT OR IGNORE INTO datasets VALUES(?,?,?)",
                    (controlled["id"], canon(controlled), b""))
    save_dataset(
        "Stockroom receipts", "Original · MIT", dispatch_rows(), 42, 0.15, builtin=True
    )


def save_dataset(name, license, rows, seed, fraction, builtin=False):
    meta, splits = validate(rows, seed, fraction)
    key = digest([meta["split_id"], name, license, builtin])
    meta.update(id=key, name=name, license=license, builtin=builtin)
    if builtin:
        splits["test"] = dispatch_rows(True)
    meta["test"] = len(splits.get("test", []))
    meta["bundle_hash"] = digest(splits)
    with db(True) as con:
        payload = b"" if builtin else zlib.compress(canon(splits).encode(), 9)
        con.execute(
            "INSERT INTO datasets VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,meta=excluded.meta",
            (key, canon(meta), payload),
        )
    return meta


def datasets():
    with db() as con:
        return [
            dict(json.loads(r[0]), staged=bool(r[1]) or json.loads(r[0])["builtin"])
            for r in con.execute(
                "SELECT meta,length(payload) FROM datasets ORDER BY rowid DESC"
            )
        ]


def dataset_meta(key):
    with db() as con:
        row = con.execute("SELECT meta FROM datasets WHERE id=?", (key,)).fetchone()
    if row is None:
        raise ValueError("Unknown dataset.")
    return json.loads(row[0])


def dataset(key):
    with db() as con:
        row = con.execute(
            "SELECT meta,payload FROM datasets WHERE id=?", (key,)
        ).fetchone()
    if row is None:
        raise ValueError("Unknown dataset.")
    meta = json.loads(row[0])
    if meta.get("protocol") == "size-v1":
        from research.size_corpus import locked_bundle
        return meta, locked_bundle()
    if meta.get("protocol") == "receipt-v2.1":
        from research.receipts import locked_bundle
        return meta, locked_bundle()
    if meta["builtin"]:
        _, splits = validate(dispatch_rows(), meta["seed"], meta["fraction"])
        return meta, dict(splits, test=dispatch_rows(True))
    if not row[1]:
        raise ValueError(
            "Re-import the original dataset with the same name, license and split settings to stage another run."
        )
    return meta, json.loads(zlib.decompress(row[1]))


def clear_staging(dataset_id):
    with db(True) as con:
        con.execute("PRAGMA secure_delete=ON")
        con.execute("UPDATE datasets SET payload=? WHERE id=?", (b"", dataset_id))


def runs(con=None):
    if con is None:
        with db() as c:
            return runs(c)
    return [
        json.loads(r[0])
        for r in con.execute("SELECT doc FROM runs ORDER BY rowid DESC")
    ]


def run(key, con=None):
    if con is None:
        with db() as c:
            return run(key, c)
    row = con.execute("SELECT doc FROM runs WHERE id=?", (key,)).fetchone()
    if row is None:
        raise ValueError("Unknown experiment.")
    return json.loads(row[0])


def put(con, doc):
    con.execute(
        "INSERT INTO runs VALUES(?,?) ON CONFLICT(id) DO UPDATE SET doc=excluded.doc",
        (doc["id"], canon(doc)),
    )


def update(key, **fields):
    with db(True) as con:
        doc = run(key, con)
        doc.update(fields)
        put(con, doc)
    return doc


def new_draft(config, con):
    row = con.execute("SELECT meta FROM datasets WHERE id=?", (config["dataset"],)).fetchone()
    if row is None:
        raise ValueError("Unknown dataset.")
    meta = json.loads(row[0])
    namespace = con.execute("SELECT v FROM kv WHERE k='namespace'").fetchone()[0]
    key = uuid.uuid4().hex
    doc = {
        "id": key,
        "resource_name": f"modellab-{namespace}-{key[:12]}",
        "config": config,
        "dataset_name": meta["name"],
        "bundle_hash": meta["bundle_hash"],
        "status": "draft",
        "created": time.time(),
        "metrics": [],
        "pod": None,
        "quote": None,
        "reserved": 0,
        "actual": None,
        "detail": "",
        "archived": False,
        "cancel": False,
    }
    return doc


def draft(config):
    if dataset_meta(config["dataset"]).get("protocol") == "size-v1":
        raise ValueError("Use the frozen model-size sessions under Experiments for this dataset.")
    dataset(config["dataset"])  # Confirm the source can actually be staged.
    with db(True) as con:
        doc = new_draft(config, con)
        put(con, doc)
        return doc
