"""Monotonic aging settlement across out-of-order batch reference instants.

A scheduling batch accepts an injected ``now``, so committed batches may run
with a reference instant EARLIER than a previous batch. While a tenant's
queue stays continuously non-empty its aging watermark
(``tenants.aging_settled_at``) must advance monotonically: an earlier batch
must never write the watermark backwards, or a later batch re-measures an
already-compensated interval and grants the same AGING minutes twice
(settle at 00:10, batch at 00:05, batch again at 00:10 used to pay the
00:05..00:10 wait a second time).

Covered here:

* a capacity-blocked head observed at 00:10, then 00:05, then again 00:10
  earns the 00:00..00:10 wait exactly once; the watermark stays at 00:10
  and the ledger/balance never receive the extra five minutes;
* the watermark never moves backwards for a ready tenant, and a later
  instant continues accruing from the stored point;
* a dependency-blocked head behaves the same, and once the blockage lifts
  the actual readiness instant remains the lower bound of the wait even
  when earlier batches interleave;
* after the queue drains the watermark resets and a freshly submitted job
  starts a brand-new wait window from its own arrival;
* preview and committed batches share the exact decision: a preview at an
  earlier instant pays nothing and persists nothing, and preview+commit at
  the same instant stay ledger-identical.
"""
from __future__ import annotations

import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def submit(tenant, gpus=1, minutes=10, weight=1.0, when=None, depends_on=None):
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
    return scheduler._get_job_or_404(db.get_conn(), job["id"])


def pin_completed(job_id, when):
    """Mark a job COMPLETED at a deterministic instant (test time control)."""
    db.get_conn().execute(
        "UPDATE jobs SET status = 'COMPLETED', completed_at = ? WHERE id = ?",
        (when, job_id),
    )


def uncomplete(job_id):
    db.get_conn().execute(
        "UPDATE jobs SET status = 'QUEUED', completed_at = NULL WHERE id = ?",
        (job_id,),
    )


def batch(now_minutes):
    return scheduler.run_batch(now=at(now_minutes))


def preview(now_minutes):
    return scheduler.preview_batch(now=at(now_minutes))


def aging_for(result, tenant_id):
    return [
        e for e in result["events"]
        if e["kind"] == "AGING" and e["tenant_id"] == tenant_id
    ]


def settled(tenant_id):
    return scheduler.get_tenant(tenant_id)["aging_settled_at"]


def fill_cluster(when=0, minutes=10):
    """Dispatch an 8-GPU job for tenant 'a' so every other head is blocked.

    The RUNNING job is intentionally left RUNNING across the test batches;
    a later ``complete_job`` frees the cluster.
    """
    heavy = submit("a", gpus=8, minutes=minutes, weight=4.0, when=iso(when))
    r = scheduler.run_batch(now=at(when))
    assert r["selection_order"] == [heavy["id"]]
    return heavy


# ---------------------------------------------------------------------------
# Out-of-order settlement must never re-credit a wait already paid
# ---------------------------------------------------------------------------

def test_earlier_batch_does_not_repay_already_settled_aging(conn):
    # The bug report, exactly: the same capacity-blocked head is first
    # settled at 00:10, then a batch runs at 00:05, then again at 00:10.
    fill_cluster()
    head = submit("b", gpus=1, minutes=5, when=iso(0))

    r1 = batch(10)  # settle 00:00..00:10 -> 10 AGING minutes
    assert aging_for(r1, "b") and aging_for(r1, "b")[0]["amount"] == 10.0
    assert settled("b") == iso(10)

    r2 = batch(5)  # earlier instant: head still undispatched
    assert aging_for(r2, "b") == []
    # The watermark must NOT have been wound back to 00:05.
    assert settled("b") == iso(10)

    r3 = batch(10)  # back to 00:10: the 00:05..00:10 wait was already paid
    assert r3["selection_order"] == []
    assert aging_for(r3, "b") == []
    assert settled("b") == iso(10)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(10.0)

    # Ledger-wide: only one AGING row for b, of exactly 10 minutes.
    ledger = scheduler.list_ledger(tenant_id="b", limit=10000)
    aging_rows = [e for e in ledger if e["kind"] == "AGING"]
    assert [e["amount"] for e in aging_rows] == [10.0]
    assert head["id"] not in r3["selection_order"]


def test_watermark_advances_only_forward_and_later_wait_accrues(conn):
    fill_cluster()
    submit("b", gpus=1, minutes=5, when=iso(0))

    batch(10)
    assert settled("b") == iso(10)
    batch(2)   # far earlier: nothing, watermark intact
    batch(10)  # same instant again: nothing
    assert settled("b") == iso(10)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(10.0)

    # A genuinely later instant accrues only the new 00:10..00:15 interval.
    r = batch(15)
    assert aging_for(r, "b")[0]["amount"] == pytest.approx(5.0)
    assert settled("b") == iso(15)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(15.0)


def test_earlier_blocked_batch_does_not_watermark_future_release(conn):
    # Head blocked on a prerequisite at 00:50 (checkpoint to 00:50, no
    # credit). An earlier batch at 00:40 must not move the checkpoint
    # backwards; after the prerequisite completes at 00:59 the wait is
    # bounded by that readiness instant.
    fill_cluster()
    pre = submit("p", gpus=1, minutes=5, when=iso(0))
    submit("b", gpus=1, minutes=5, when=iso(0), depends_on=[pre["id"]])

    batch(50)  # blocked: checkpoint to t=50, no credit
    assert aging_for(batch(40), "b") == []  # earlier blocked batch
    assert settled("b") == iso(50)

    pin_completed(pre["id"], iso(59))
    r = batch(70)  # ready since t=59: 70 - 59 = 11 minutes
    assert aging_for(r, "b")[0]["amount"] == pytest.approx(11.0)

    # Earlier batch while ready pays nothing and keeps the t=70 watermark.
    assert aging_for(batch(55), "b") == []
    assert settled("b") == iso(70)
    # The next later batch accrues only 70..80; readiness is still the floor.
    r2 = batch(80)
    assert aging_for(r2, "b")[0]["amount"] == pytest.approx(10.0)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(21.0)


def test_blocked_checkpoint_first_observation_still_advances(conn):
    # NULL watermark + blocked head: the first observation still adopts
    # ``now`` (no back-off), only out-of-order values are clamped.
    fill_cluster()
    pre = submit("p", gpus=1, minutes=5, when=iso(0))
    submit("b", gpus=1, minutes=5, when=iso(0), depends_on=[pre["id"]])

    batch(30)
    assert settled("b") == iso(30)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == 0.0
    # An earlier blocked batch cannot lower it.
    batch(10)
    assert settled("b") == iso(30)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == 0.0


# ---------------------------------------------------------------------------
# Queue drain still closes the aging period; new job starts a fresh window
# ---------------------------------------------------------------------------

def test_drained_queue_then_new_job_starts_fresh_window_after_out_of_order(conn):
    heavy = fill_cluster()
    head = submit("b", gpus=1, minutes=5, when=iso(0))
    batch(10)  # b's head blocked, watermark at t=10

    # Free the cluster and dispatch b's head at an EARLIER instant: the
    # queue drains there and the aging clock resets to NULL regardless of
    # the earlier reference instant.
    scheduler.complete_job(heavy["id"])
    r = batch(8)
    assert r["selection_order"] == [head["id"]]
    assert settled("b") is None
    scheduler.complete_job(head["id"])  # release b's GPU for the next phase

    # A brand new job at t=100 starts a clean wait window from its arrival,
    # never from the pre-drain t=10 settlement point.
    new = submit("b", gpus=1, minutes=1000, when=iso(100))
    fill_cluster(when=100, minutes=1)  # cheap filler dispatches, blocks b
    r2 = batch(110)
    assert r2["selection_order"] == []
    assert aging_for(r2, "b")[0]["amount"] == pytest.approx(10.0)
    assert new["id"] not in r2["selection_order"]


# ---------------------------------------------------------------------------
# Preview shares the monotonic decision and writes nothing
# ---------------------------------------------------------------------------

def test_preview_at_earlier_instant_pays_nothing_and_persists_nothing(conn):
    fill_cluster()
    submit("b", gpus=1, minutes=5, when=iso(0))
    batch(10)  # watermark t=10, balance 10

    conn_obj = db.get_conn()
    ledger_before = conn_obj.execute(
        "SELECT COUNT(*) AS n FROM ledger"
    ).fetchone()["n"]
    p = preview(5)
    assert aging_for(p, "b") == []
    # Zero writes: balance and watermark are untouched by the preview.
    assert settled("b") == iso(10)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(10.0)
    assert conn_obj.execute(
        "SELECT COUNT(*) AS n FROM ledger"
    ).fetchone()["n"] == ledger_before


def test_preview_and_commit_identical_at_earlier_instant(conn):
    fill_cluster()
    submit("b", gpus=1, minutes=5, when=iso(0))
    batch(10)  # settle t=10
    # Same state, same earlier instant: preview and commit must agree
    # byte-for-byte on events (no AGING either way) and balances.
    p = preview(5)
    c = batch(5)
    assert [
        (e["kind"], e["tenant_id"], e["amount"], e["balance_after"])
        for e in p["events"]
    ] == [
        (e["kind"], e["tenant_id"], e["amount"], e["balance_after"])
        for e in c["events"]
    ]
    assert aging_for(c, "b") == []
    assert settled("b") == iso(10)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def test_http_out_of_order_batches_do_not_double_credit_aging(client):
    from app import db as db_mod

    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 8, "duration_min": 10, "weight": 4})
    head = client.post("/jobs", json={
        "tenant_id": "b", "gpus": 1, "duration_min": 5}).json()["id"]
    db_mod.get_conn().execute(
        "UPDATE jobs SET queued_at = ? WHERE id = ?", (iso(0), head)
    )
    client.post("/scheduler/batch", json={"now": iso(0)})

    def aging_amounts(body):
        return [e["amount"] for e in body["events"]
                if e["kind"] == "AGING" and e["tenant_id"] == "b"]

    r10 = client.post("/scheduler/batch", json={"now": iso(10)})
    assert aging_amounts(r10.json()) == [10.0]
    r05 = client.post("/scheduler/batch", json={"now": iso(5)})
    assert aging_amounts(r05.json()) == []
    r10_again = client.post("/scheduler/batch", json={"now": iso(10)})
    assert aging_amounts(r10_again.json()) == []

    tenant = client.get("/tenants/b").json()
    assert tenant["balance_gpu_min"] == pytest.approx(10.0)
    assert tenant["aging_settled_at"] == iso(10)

    # A preview interleaved at the earlier instant changes nothing either.
    p = client.post("/scheduler/preview", json={"now": iso(3)})
    assert aging_amounts(p.json()) == []
    assert client.get("/tenants/b").json()["balance_gpu_min"] == \
        pytest.approx(10.0)
