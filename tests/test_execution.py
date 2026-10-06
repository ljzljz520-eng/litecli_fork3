# coding=UTF-8

"""Tests for per-statement execution contexts, budgets, the cancellation
state machine, transaction taint rules and multi-statement batch policies."""

import os
import threading

import pytest

from litecli import execution as execmod
from litecli.execution import (
    LOCK_WAIT,
    SEVERE_ERROR,
    TEMP_DISK,
    USER_CANCEL,
    VM_STEPS,
    Budget,
    ExecutionContext,
    ExecutionLimit,
    TransactionGuard,
    TransactionTaintedError,
    classify_statement,
    format_toolbar,
    measure_dir,
    parse_policy,
)
from litecli.sqlexecute import payload_bytes

from .utils import dbtest

# An intentionally unbounded query: recursive CTE that only terminates when
# aborted through the progress handler.
ENDLESS_CTE = "WITH RECURSIVE cnt(x) AS (" "SELECT 1 UNION ALL SELECT x+1 FROM cnt" ") SELECT max(x) FROM cnt"


# ---------------------------------------------------------------------------
# Budget parsing
# ---------------------------------------------------------------------------


def test_budget_defaults_to_unlimited():
    budget = Budget.from_config_section({})
    assert budget.wall_seconds is None
    assert budget.vm_steps is None
    assert budget.output_rows is None
    assert budget.output_bytes is None
    assert budget.temp_bytes is None


def test_budget_parses_configured_values():
    section = {
        "execution_budget_wall_seconds": "2.5",
        "execution_budget_vm_steps": "100000",
        "execution_budget_output_rows": "50",
        "execution_budget_output_bytes": "1024",
        "execution_budget_temp_bytes": "4096",
    }
    budget = Budget.from_config_section(section)
    assert budget.wall_seconds == 2.5
    assert budget.vm_steps == 100000
    assert budget.output_rows == 50
    assert budget.output_bytes == 1024
    assert budget.temp_bytes == 4096


def test_parse_policy():
    assert parse_policy("stop") == "stop"
    assert parse_policy(" CONTINUE ") == "continue"
    assert parse_policy("rollback") == "rollback"
    assert parse_policy(None) == "stop"
    with pytest.raises(ValueError):
        parse_policy("bogus")


# ---------------------------------------------------------------------------
# Statement classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, klass",
    [
        ("SELECT 1", "read"),
        ("select * from t", "read"),
        ("EXPLAIN QUERY PLAN SELECT 1", "read"),
        ("PRAGMA table_info(t)", "read"),
        ("BEGIN", "txn_begin"),
        ("BEGIN IMMEDIATE", "txn_begin"),
        ("COMMIT", "txn_commit"),
        ("END", "txn_commit"),
        ("ROLLBACK", "txn_rollback"),
        ("SAVEPOINT x", "savepoint"),
        ("INSERT INTO t VALUES (1)", "dml"),
        ("UPDATE t SET a = 2", "dml"),
        ("DELETE FROM t", "dml"),
        ("CREATE TABLE t(a)", "ddl"),
        ("DROP TABLE t", "ddl"),
        ("ALTER TABLE t ADD COLUMN b", "ddl"),
        ("WITH x AS (SELECT 1) SELECT * FROM x", "read"),
        ("WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x", "dml"),
        ("-- comment\nSELECT 2", "read"),
    ],
)
def test_classify_statement(sql, klass):
    assert classify_statement(sql) == klass


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_context_starts_running():
    ctx = ExecutionContext("select 1")
    assert ctx.state == "running"


def test_request_cancel_transitions_to_cancelling():
    ctx = ExecutionContext("select 1")
    ctx.request_cancel()
    assert ctx.state == "cancelling"
    assert ctx.cancel_requested()
    # Idempotent.
    ctx.request_cancel()
    assert ctx.state == "cancelling"


def test_finish_records_terminal_state_once():
    ctx = ExecutionContext("select 1")
    ctx.finish("completed")
    assert ctx.state == "completed"
    assert ctx.finished_at is not None
    ctx.finish("failed")
    assert ctx.state == "completed"


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


def test_progress_vm_step_budget():
    ctx = ExecutionContext("select 1", budget=Budget(vm_steps=10))
    result = 0
    for _ in range(10):
        result = ctx.on_progress()
    assert result == 1
    assert ctx.abort_reason == "vm_steps"
    assert ctx.steps == 10


def test_progress_cancel_flag_aborts():
    ctx = ExecutionContext("select 1", budget=Budget(vm_steps=1_000_000))
    ctx.request_cancel()
    assert ctx.on_progress() == 1
    assert ctx.abort_reason == "user_cancel"


def test_progress_wall_time_budget():
    ctx = ExecutionContext("select 1", budget=Budget(wall_seconds=0.0, vm_steps=None))
    result = 0
    # Clock reads are throttled to every 64 ticks.
    for _ in range(64):
        result = ctx.on_progress()
    assert result == 1
    assert ctx.abort_reason == "wall_time"


def test_progress_temp_disk_budget(tmp_path):
    payload_file = tmp_path / "big.tmp"
    payload_file.write_bytes(b"x" * 5000)
    ctx = ExecutionContext("select 1", budget=Budget(temp_bytes=1024), temp_dir=str(tmp_path))
    result = 0
    for _ in range(4096):
        result = ctx.on_progress()
    assert result == 1
    assert ctx.abort_reason == TEMP_DISK


def test_progress_temp_peak_tracking(tmp_path):
    payload_file = tmp_path / "small.tmp"
    payload_file.write_bytes(b"x" * 10)
    ctx = ExecutionContext("select 1", budget=Budget(temp_bytes=10_000), temp_dir=str(tmp_path))
    for _ in range(4096):
        assert ctx.on_progress() == 0
    assert ctx.temp_bytes_peak == 10


def test_measure_dir_ignores_dirs_and_missing(tmp_path):
    (tmp_path / "a").write_bytes(b"1234")
    os.mkdir(str(tmp_path / "sub"))
    assert measure_dir(str(tmp_path)) == 4


# ---------------------------------------------------------------------------
# Output accounting
# ---------------------------------------------------------------------------


def test_account_rows_budget():
    ctx = ExecutionContext("select 1", budget=Budget(output_rows=5))
    ctx.account_rows(5, 10)
    with pytest.raises(ExecutionLimit) as info:
        ctx.account_rows(1, 2)
    assert info.value.reason == "output_rows"


def test_account_bytes_budget():
    ctx = ExecutionContext("select 1", budget=Budget(output_bytes=10))
    ctx.account_rows(2, 10)
    with pytest.raises(ExecutionLimit) as info:
        ctx.account_rows(1, 1)
    assert info.value.reason == "output_bytes"


def test_payload_bytes_handles_null_and_text():
    assert payload_bytes([("abc", None, "é")]) == len(b"abc") + len("é".encode())


# ---------------------------------------------------------------------------
# Busy handler (context-level)
# ---------------------------------------------------------------------------


def test_busy_handler_keeps_waiting_then_observes_cancel():
    ctx = ExecutionContext("update t set a=1", budget=Budget(wall_seconds=100))
    assert ctx.on_busy(1) == 1
    assert ctx.lock_invocations == 1
    ctx.request_cancel()
    assert ctx.on_busy(2) == 0
    assert ctx.abort_reason == USER_CANCEL


def test_busy_handler_lock_timeout():
    ctx = ExecutionContext("update t set a=1", lock_wait_timeout=0.0)
    # Zero timeout: aborts on the first invocation.
    assert ctx.on_busy(1) == 0
    assert ctx.abort_reason == LOCK_WAIT


# ---------------------------------------------------------------------------
# Transaction guard rules
# ---------------------------------------------------------------------------


@pytest.fixture
def guard():
    return TransactionGuard()


def test_guard_read_abort_not_tainted(guard):
    guard.note_abort("read", USER_CANCEL, in_transaction=True)
    assert guard.tainted is False


def test_guard_dml_abort_defensively_tainted(guard):
    # SQLite guarantees interrupted INSERT/UPDATE/DELETE rolls back the whole
    # transaction, so an aborted DML with in_transaction still True is an
    # abnormal state and treated as tainted.
    guard.note_abort("dml", VM_STEPS, in_transaction=True)
    assert guard.tainted is True


def test_guard_ddl_abort_not_tainted(guard):
    # DDL is fully transactional in SQLite.
    guard.note_abort("ddl", USER_CANCEL, in_transaction=True)
    assert guard.tainted is False


def test_guard_commit_abort_taints_when_tx_open(guard):
    guard.note_abort("txn_commit", USER_CANCEL, in_transaction=True)
    assert guard.tainted is True


def test_guard_commit_abort_after_commit_not_tainted(guard):
    # Observed in_transaction=False: commit completed before the abort landed.
    guard.note_abort("txn_commit", USER_CANCEL, in_transaction=False)
    assert guard.tainted is False


def test_guard_lock_wait_not_tainted(guard):
    guard.note_abort("dml", LOCK_WAIT, in_transaction=True)
    assert guard.tainted is False


def test_guard_severe_error_taints(guard):
    guard.note_abort("dml", SEVERE_ERROR, in_transaction=True)
    assert guard.tainted is True


def test_guard_no_taint_without_transaction(guard):
    guard.note_abort("dml", SEVERE_ERROR, in_transaction=False)
    assert guard.tainted is False


def test_guard_blocks_non_rollback_when_tainted(guard):
    guard.tainted = True
    for klass in ("read", "dml", "ddl", "txn_begin", "txn_commit", "savepoint", "other"):
        with pytest.raises(TransactionTaintedError):
            guard.check_permitted(klass)
    # The one and only permitted statement.
    guard.check_permitted("txn_rollback")


def test_guard_success_resets(guard):
    guard.tainted = True
    guard.note_success("txn_rollback")
    assert guard.tainted is False


# ---------------------------------------------------------------------------
# Batch controller
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy, expected",
    [("stop", "stop"), ("continue", "continue"), ("rollback", "rollback")],
)
def test_batch_policy_decisions(policy, expected):
    batch = execmod.BatchController(policy, size=2)
    ctx = ExecutionContext("select 1")
    assert batch.decide(ctx, USER_CANCEL) == expected


def test_batch_escalation_overrides_continue():
    batch = execmod.BatchController("continue", size=2)
    batch.request_abort()
    assert batch.abort_requested()
    ctx = ExecutionContext("select 1")
    assert batch.decide(ctx, USER_CANCEL) == "stop"


# ---------------------------------------------------------------------------
# Toolbar text
# ---------------------------------------------------------------------------


def test_toolbar_unlimited_idle():
    text = format_toolbar(None, Budget.unlimited(), tainted=False, in_transaction=False)
    assert "[idle]" in text
    assert "∞" in text


def test_toolbar_tainted_chip():
    text = format_toolbar(None, Budget.unlimited(), tainted=True, in_transaction=True)
    assert "TXN-TAINTED" in text


def test_toolbar_after_completed_with_limits():
    ctx = ExecutionContext("select 1", budget=Budget(wall_seconds=10, vm_steps=100_000))
    ctx.finish("completed")
    text = format_toolbar(ctx, ctx.budget, tainted=False, in_transaction=True)
    assert "[completed]" in text
    assert "txn" in text
    assert "t=" in text
    assert "vm=" in text


# ---------------------------------------------------------------------------
# Integration: real SQLite connection via the `executor` fixture
# ---------------------------------------------------------------------------


def _unlimited(executor):
    executor.configure(budget=Budget.unlimited())


def _vm_budget(executor, steps):
    executor.configure(budget=Budget(vm_steps=steps))


@dbtest
def test_integration_vm_budget_aborts_cte(executor):
    _vm_budget(executor, 3000)
    with pytest.raises(ExecutionLimit) as info:
        list(executor.run(ENDLESS_CTE))
    assert info.value.reason == "vm_steps"
    assert executor.last_ctx.state == "interrupted"
    assert executor.guard.tainted is False


@dbtest
def test_integration_wall_time_budget_aborts_cte(executor):
    executor.configure(budget=Budget(wall_seconds=0.2))
    with pytest.raises(ExecutionLimit) as info:
        list(executor.run(ENDLESS_CTE))
    assert info.value.reason == "wall_time"


@dbtest
def test_integration_output_row_budget(executor):
    list(executor.run("create table t(a)"))
    list(executor.run("insert into t values (1)"))
    list(executor.run("insert into t values (2)"))
    executor.configure(budget=Budget(output_rows=1))
    with pytest.raises(ExecutionLimit) as info:
        list(executor.run("select * from t"))
    assert info.value.reason == "output_rows"


@dbtest
def test_integration_output_byte_budget(executor):
    list(executor.run("create table t(a)"))
    list(executor.run("insert into t values ('abcdef')"))
    executor.configure(budget=Budget(output_bytes=3))
    with pytest.raises(ExecutionLimit):
        list(executor.run("select * from t"))


@dbtest
def test_integration_transaction_survives_read_abort(executor):
    list(executor.run("create table t(a)"))
    list(executor.run("insert into t values (1)"))
    list(executor.run("BEGIN"))
    list(executor.run("insert into t values (2)"))

    _vm_budget(executor, 3000)
    with pytest.raises(ExecutionLimit):
        list(executor.run(ENDLESS_CTE))
    assert executor.guard.tainted is False
    assert executor.conn.in_transaction is True

    _unlimited(executor)
    list(executor.run("insert into t values (3)"))
    list(executor.run("COMMIT"))

    result = list(executor.run("select * from t order by a"))
    rows = result[0][1]
    assert rows == [(1,), (2,), (3,)]


@dbtest
def test_integration_dml_abort_rolls_back_whole_transaction(executor):
    list(executor.run("create table t(a)"))
    list(executor.run("insert into t values (1)"))
    list(executor.run("BEGIN"))
    list(executor.run("insert into t values (2)"))
    # An UPDATE that can never finish under a tiny step budget.
    heavy_update = "WITH RECURSIVE cnt(x) AS (" "SELECT 1 UNION ALL SELECT x+1 FROM cnt" ") UPDATE t SET a = (SELECT max(x) FROM cnt)"
    executor.configure(budget=Budget(vm_steps=5000))
    with pytest.raises(ExecutionLimit):
        list(executor.run(heavy_update))
    # SQLite rule: interrupted INSERT/UPDATE/DELETE inside an explicit
    # transaction rolls the whole transaction back automatically.
    assert executor.conn.in_transaction is False
    result = list(executor.run("select count(*) from t"))
    assert result[0][1] == [(1,)]


# ---------------------------------------------------------------------------
# Integration: multi-statement batch policies
# ---------------------------------------------------------------------------


@dbtest
def test_batch_stop_policy(executor):
    _vm_budget(executor, 3000)
    executor.configure(budget=Budget(vm_steps=3000), batch_policy="stop")
    batch_sql = ENDLESS_CTE + "; SELECT 'after'"
    with pytest.raises(ExecutionLimit):
        list(executor.run(batch_sql))


@dbtest
def test_batch_continue_policy(executor):
    executor.configure(budget=Budget(vm_steps=3000), batch_policy="continue")
    batch_sql = ENDLESS_CTE + "; SELECT 'after'"
    results = list(executor.run(batch_sql))
    # Skip notice for statement 1, then the actual result of statement 2.
    statuses = [item[3] for item in results]
    assert any("skipped" in str(status) for status in statuses)
    final = results[-1]
    assert final[1] == [("after",)]


@dbtest
def test_batch_rollback_policy(executor):
    list(executor.run("create table t(a)"))
    list(executor.run("BEGIN"))
    list(executor.run("insert into t values (1)"))
    notices: list[str] = []
    executor.notice = lambda message, **_: notices.append(message)

    executor.configure(budget=Budget(vm_steps=3000), batch_policy="rollback")
    with pytest.raises(ExecutionLimit):
        list(executor.run(ENDLESS_CTE + "; SELECT 'after'"))

    assert executor.conn.in_transaction is False
    assert any("rolled back" in message for message in notices)
    result = list(executor.run("select count(*) from t"))
    assert result[0][1] == [(0,)]


@dbtest
def test_batch_double_ctrlc_escalation_during_run(executor):
    # Policy says continue, but two Ctrl-C within the escalation window force
    # the whole batch to stop: first cancels statement 1 (would be skipped),
    # second escalates while statement 2 runs.
    executor.configure(budget=Budget.unlimited(), batch_policy="continue")

    def worker() -> None:
        threading.Event().wait(0.2)
        executor.request_cancel()
        threading.Event().wait(0.2)
        executor.request_cancel()

    thread = threading.Thread(target=worker)
    thread.start()
    with pytest.raises(ExecutionLimit) as info:
        list(executor.run(ENDLESS_CTE + "; " + ENDLESS_CTE))
    thread.join()
    assert info.value.reason == USER_CANCEL


# ---------------------------------------------------------------------------
# Integration: GeneratorExit (consumer closes output early)
# ---------------------------------------------------------------------------


@dbtest
def test_generator_exit_without_abort_is_safe(executor):
    res = executor.run("SELECT 1; SELECT 2")
    assert next(res)[1] == [(1,)]
    res.close()


@dbtest
def test_generator_exit_with_escalation_honors_rollback_policy(executor):
    list(executor.run("create table t(a)"))
    list(executor.run("BEGIN"))
    list(executor.run("insert into t values (1)"))
    notices: list[str] = []
    executor.notice = lambda message, **_: notices.append(message)

    executor.configure(budget=Budget.unlimited(), batch_policy="rollback")
    res = executor.run("SELECT 1")
    assert next(res)[1] == [(1,)]
    # Ctrl-C during the output phase: escalate, then the consumer closes the
    # generator; the rollback policy must run inside the GeneratorExit path.
    executor.request_cancel()
    res.close()

    assert executor.conn.in_transaction is False
    assert any("rolled back" in message for message in notices)


# ---------------------------------------------------------------------------
# Integration: cross-thread cancellation (real thread, real interrupt)
# ---------------------------------------------------------------------------


@dbtest
def test_cancel_from_other_thread(executor):
    def worker():
        threading.Event().wait(0.2)
        executor.request_cancel()

    thread = threading.Thread(target=worker)
    thread.start()
    with pytest.raises(ExecutionLimit) as info:
        list(executor.run(ENDLESS_CTE))
    thread.join()
    assert info.value.reason == USER_CANCEL
    assert executor.guard.tainted is False
