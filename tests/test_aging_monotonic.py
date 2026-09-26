"""Out-of-order batch reference instants must never re-credit settled aging.

Regression tests for the aging-watermark monotonicity fix: a committed batch
at a *later* ``now`` (say 00:10) advances the tenant's aging settlement point.
A subsequent batch invoked with an *earlier* ``now`` (00:05) must not move
that watermark backwards — otherwise a third batch back at 00:10 would pay
the 00:05..00:10 aging interval a second time. During one continuous queued
period the watermark only moves forward; the readiness instant after a
dependency blockage lifts stays the wait floor; only a drained queue resets
the clock so a freshly submitted job starts a new wait window.
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
    return scheduler.get_job(job["id"])


def pin_completed(job_id, when):
    db.get_conn().execute(
        "UPDATE jobs SET status = 'COMPLETED', completed_at = ? WHERE id = ?",
        (when, job_id),
    )


def aging_events(result, tenant_id):
    return [
        e for e in result["events"]
        if e["kind"] == "AGING" and e["tenant_id"] == tenant_id
    ]


def test_earlier_batch_does_not_rewind_watermark_or_double_credit(conn):
    # a's 8-GPU job saturates the cluster first; b's small ready job, queued
    # behind it, stays the undispatched head and only aging compensates it.
    submit("a", gpus=8, minutes=5000, weight=4.0, when=iso(0))
    first = scheduler.run_batch(now=at(10))
    assert first["selection_order"] == ["J000001"]

    blocked = submit("b", gpus=1, minutes=5000, weight=1.0, when=iso(0))

    # Settle at 00:10: b earns 10 minutes of AGING and the watermark is 00:10.
    r10 = scheduler.run_batch(now=at(10))
    assert r10["selection_order"] == []
    aging = aging_events(r10, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(10.0)
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(10)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(10.0)
    assert blocked["id"] == "J000002"

    # An earlier batch at 00:05 must grant nothing and keep the watermark at
    # 00:10 (the bug rewrote it to 00:05).
    r05 = scheduler.run_batch(now=at(5))
    assert aging_events(r05, "b") == []
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(10)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(10.0)

    # Back at 00:10 the interval 00:05..00:10 must NOT be paid a second time.
    r10_again = scheduler.run_batch(now=at(10))
    assert aging_events(r10_again, "b") == []
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(10)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(10.0)

    # Forward progress still works once real time passes: only the
    # not-yet-settled 00:10..00:15 interval is compensated.
    r15 = scheduler.run_batch(now=at(15))
    aging = aging_events(r15, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(5.0)
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(15)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(15.0)


def test_earlier_batch_after_blocked_period_keeps_readiness_floor(conn):
    # A dependency-blocked head earns nothing and its blocked batches merely
    # carry the checkpoint forward. Once the blockage lifts, the true
    # readiness instant is the wait floor (even past a checkpoint advanced
    # during the blocked wait); after that, an earlier batch can no longer
    # rewind the watermark. A cluster-filling job keeps the released head
    # capacity-blocked and unaffordable, so only aging ever moves.
    pre = submit("a", gpus=8, minutes=5000, weight=4.0, when=iso(0))
    dep = submit("b", gpus=1, minutes=5000, when=iso(0),
                 depends_on=[pre["id"]])

    r30 = scheduler.run_batch(now=at(30))  # a dispatches; b blocked
    assert r30["selection_order"] == [pre["id"]]
    assert aging_events(r30, "b") == []
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(30)

    # Another blocked batch only carries the checkpoint forward, no credit.
    r34 = scheduler.run_batch(now=at(34))
    assert aging_events(r34, "b") == []
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(34)

    # Release at 00:35 and immediately keep the cluster full with another
    # filler, before any ready-state aging batch observes b. High weight makes
    # it affordable in one round so it takes the freed capacity ahead of b.
    pin_completed(pre["id"], iso(35))
    submit("a", gpus=8, minutes=5000, weight=1000.0, when=iso(35))

    # At 00:40 the filler dispatches; b is ready since 00:35 and earns exactly
    # the 5 minutes since release — the blocked checkpoint at 00:34 does not
    # reach past the readiness floor.
    r40 = scheduler.run_batch(now=at(40))
    assert r40["selection_order"] == ["J000003"]
    aging = aging_events(r40, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(5.0)  # 40 - 35, not 40 - 34
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(40)

    # Forward batches measure only the not-yet-settled interval.
    r60 = scheduler.run_batch(now=at(60))
    aging = aging_events(r60, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(20.0)  # 60 - 40
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(60)
    assert dep["id"] == "J000002"


def test_preview_earlier_now_shares_the_monotone_judgement(conn):
    submit("a", gpus=8, minutes=5000, weight=4.0, when=iso(0))
    first = scheduler.run_batch(now=at(10))
    assert first["selection_order"] == ["J000001"]
    submit("b", gpus=1, minutes=5000, weight=1.0, when=iso(0))

    committed = scheduler.run_batch(now=at(10))
    assert aging_events(committed, "b")[0]["amount"] == pytest.approx(10.0)

    # A preview at the earlier instant shows the same zero extra credit and
    # writes nothing.
    preview = scheduler.preview_batch(now=at(5))
    assert aging_events(preview, "b") == []
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(10)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(10.0)

    # The committed batch at that same earlier instant matches the preview:
    # no AGING event, ledger unchanged, watermark still at 00:10.
    replayed = scheduler.run_batch(now=at(5))
    assert [
        (e["kind"], e["tenant_id"], e["amount"])
        for e in replayed["events"]
    ] == [
        (e["kind"], e["tenant_id"], e["amount"])
        for e in preview["events"]
    ]
    assert aging_events(replayed, "b") == []
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(10)
    assert scheduler.get_tenant("b")["balance_gpu_min"] == pytest.approx(10.0)


def test_drain_resets_watermark_before_any_new_wait_period(conn):
    # A later-then-earlier pair on one continuous queue never rewinds; only
    # draining the queue resets the clock, after which a new job's wait starts
    # from that job's own arrival. A long cluster-filling job keeps the new
    # job capacity-blocked so it stays the head.
    submit("a", gpus=1, minutes=10, weight=1.0, when=iso(0))
    r = scheduler.run_batch(now=at(10))
    assert r["selection_order"] == ["J000001"]
    scheduler.complete_job("J000001")
    assert scheduler.get_tenant("a")["aging_settled_at"] is None

    filler = submit("f", gpus=8, minutes=5000, weight=1.0, when=iso(99))
    filler_batch = scheduler.run_batch(now=at(99))
    assert filler_batch["selection_order"] == [filler["id"]]

    submit("a", gpus=8, minutes=5000, weight=1.0, when=iso(100))
    scheduler.run_batch(now=at(110))
    assert scheduler.get_tenant("a")["aging_settled_at"] == iso(110)
    # An earlier retroactive batch on the NEW queued period still cannot
    # rewind the new watermark.
    earlier = scheduler.run_batch(now=at(105))
    assert aging_events(earlier, "a") == []
    assert scheduler.get_tenant("a")["aging_settled_at"] == iso(110)
