"""Materialize the four prespecified drafts. This module only accesses local SQLite."""

from remote.controlled import protocol
from studio import store, lifecycle
from studio.config import Config
from research.receipts import locked_bundle, metadata
from studio.datasets import digest


def configurations():
    data = metadata(locked_bundle())
    return [
        Config(
            name=f"Controlled receipts · rank {rank}",
            dataset=data["id"],
            rank=rank,
            alpha=2 * rank,
            minutes=60,
        ).model_dump()
        for rank in (4, 8, 16, 32)
    ]


def prepare():
    store.init()
    ids = []
    for cfg in configurations():
        with store.db(True) as con:
            signature = digest([cfg, protocol()])
            existing = next(
                (d for d in store.runs(con) if d.get("research_design") == signature),
                None,
            )
            if existing:
                if existing["status"] == "draft" and not existing.get(
                    "research_worker_sha256"
                ):
                    existing["research_worker_sha256"] = lifecycle.worker_sources()
                    store.put(con, existing)
                lifecycle.check_research_lock(existing)
                ids.append(existing["id"])
                continue
            doc = store.new_draft(cfg, con)
            doc["research_design"] = signature
            doc["research_worker_sha256"] = lifecycle.worker_sources()
            store.put(con, doc)
            ids.append(doc["id"])
    return ids


if __name__ == "__main__":
    for key in prepare():
        print(key, store.run(key)["config"]["name"], store.run(key)["status"])
