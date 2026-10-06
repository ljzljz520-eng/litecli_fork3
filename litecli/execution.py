"""Per-statement execution contexts, budgets and cancellation state machine.

Every SQL statement executed through :class:`~litecli.sqlexecute.SQLExecute`
runs inside an :class:`ExecutionContext` that carries five independent
budgets:

* wall-clock time (seconds),
* SQLite virtual-machine steps (bytecode instructions),
* emitted output rows,
* emitted output payload bytes,
* temporary disk space.

Cancellation is a cooperative protocol.  The *only* SQLite entry points that
are allowed to be invoked asynchronously (e.g. from a signal handler or
another thread) are ``sqlite3_interrupt()`` (exposed as
``connection.interrupt()``) and the lightweight cancel flag on the context;
all other decisions are taken at well-defined SQLite VM boundaries by the
progress handler.  This module therefore encodes four SQLite transaction
rules explicitly:

1. A *read* statement (SELECT/EXPLAIN/PRAGMA) aborted via the progress
   handler (return value ``1``) or via ``sqlite3_interrupt()`` is guaranteed
   a statement rollback; an explicit surrounding transaction stays open and
   usable.
2. An *INSERT, UPDATE or DELETE* aborted inside an explicit transaction is
   treated more harshly by SQLite: the **entire transaction is rolled back
   automatically** (see the sqlite3_interrupt documentation).  The observed
   ``connection.in_transaction`` is therefore False and there is nothing
   left to continue or taint.
3. DDL (CREATE/DROP/ALTER) is fully transactional but is not subject to the
   INSERT/UPDATE/DELETE rule above: an aborted DDL statement is rolled back
   at statement level while the explicit transaction survives.
4. COMMIT is atomic.  Once SQLite enters the commit critical section it
   deliberately stops honouring interrupts and progress callbacks, so an
   interrupted COMMIT either *completely committed* or *did not commit at
   all*.  The post-state is therefore always observed via
   ``connection.in_transaction`` and is authoritative: if the connection is
   still in a transaction the commit aborted before the point of no return
   and the transaction is marked :data:`TAINTED` (the user asked to cancel
   the commit, so only ROLLBACK is permitted); otherwise the commit
   completed and the transaction is gone.

Taint is also raised for irrecoverable engine errors (SQLITE_FULL,
SQLITE_IOERR, SQLITE_CORRUPT, ...) inside a transaction.  A tainted
transaction rejects every statement except ROLLBACK.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable
from time import monotonic
from typing import Any, cast

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

IDLE = "idle"
RUNNING = "running"
CANCELLING = "cancelling"
INTERRUPTED = "interrupted"
COMPLETED = "completed"
FAILED = "failed"

TERMINAL_STATES = frozenset({INTERRUPTED, COMPLETED, FAILED})

# ---------------------------------------------------------------------------
# Abort reasons
# ---------------------------------------------------------------------------

USER_CANCEL = "user_cancel"
WALL_TIME = "wall_time"
VM_STEPS = "vm_steps"
OUTPUT_ROWS = "output_rows"
OUTPUT_BYTES = "output_bytes"
TEMP_DISK = "temp_disk"
LOCK_WAIT = "lock_wait"
SEVERE_ERROR = "severe_error"

# Reasons produced by aborting the VM (progress handler / interrupt).
VM_ABORT_REASONS = frozenset({USER_CANCEL, WALL_TIME, VM_STEPS, TEMP_DISK})

_REASON_TEXT = {
    USER_CANCEL: "cancelled by user",
    WALL_TIME: "statement exceeded wall-time budget",
    VM_STEPS: "statement exceeded SQLite VM step budget",
    OUTPUT_ROWS: "result set exceeded row budget",
    OUTPUT_BYTES: "result set exceeded output byte budget",
    TEMP_DISK: "statement exceeded temporary disk budget",
    LOCK_WAIT: "aborted while waiting on a database lock",
    SEVERE_ERROR: "irrecoverable database error",
}

# Progress callback throttling.  The fast path (cancel flag + step counter)
# runs on every tick; clock reads are amortised 1/64 and directory scans
# 1/4096 so the callback stays allocation-free in the common case.
_TIME_CHECK_MASK = 0x3F
_TEMP_CHECK_MASK = 0xFFF

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExecutionLimit(Exception):
    """Raised when a statement is aborted because a budget was exceeded."""

    def __init__(self, reason: str, limit: float | None = None, used: float | None = None):
        self.reason = reason
        self.limit = limit
        self.used = used
        super().__init__(format_limit_message(reason, limit, used))


class TransactionTaintedError(Exception):
    """Raised when a non-ROLLBACK statement hits a tainted transaction."""


def format_limit_message(reason: str, limit: float | None = None, used: float | None = None) -> str:
    message = _REASON_TEXT.get(reason, reason)
    if limit is not None:
        if reason == WALL_TIME:
            message += f" ({float(used or 0.0):.3f}s >= {float(limit):.3f}s)"
        elif reason in (OUTPUT_BYTES, TEMP_DISK):
            message += f" ({_short_int(int(used or 0))} >= {_short_int(int(limit))} bytes)"
        else:
            message += f" ({_short_int(int(used or 0))} >= {_short_int(int(limit))})"
    return message


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


class Budget:
    """The five per-statement budgets.

    ``None`` means "unlimited"; zero configured values map to ``None`` so
    that default installations keep the historical, unbounded behaviour.
    """

    __slots__ = ("output_bytes", "output_rows", "temp_bytes", "vm_steps", "wall_seconds")

    def __init__(
        self,
        wall_seconds: float | None = None,
        vm_steps: int | None = None,
        output_rows: int | None = None,
        output_bytes: int | None = None,
        temp_bytes: int | None = None,
    ) -> None:
        self.wall_seconds = wall_seconds
        self.vm_steps = vm_steps
        self.output_rows = output_rows
        self.output_bytes = output_bytes
        self.temp_bytes = temp_bytes

    @classmethod
    def unlimited(cls) -> Budget:
        return cls()

    @classmethod
    def from_config_section(cls, section: Any) -> Budget:
        """Read budgets from a ConfigObj ``[main]`` section."""

        def number(key: str, cast_func: Callable[[str], int | float]) -> int | float | None:
            raw = section.get(key, "0")
            value = cast_func(str(raw if raw is not None else "0"))
            return value if value > 0 else None

        wall = number("execution_budget_wall_seconds", float)
        return cls(
            wall_seconds=wall,
            vm_steps=cast("int|None", number("execution_budget_vm_steps", int)),
            output_rows=cast("int|None", number("execution_budget_output_rows", int)),
            output_bytes=cast("int|None", number("execution_budget_output_bytes", int)),
            temp_bytes=cast("int|None", number("execution_budget_temp_bytes", int)),
        )


# ---------------------------------------------------------------------------
# Statement classification
# ---------------------------------------------------------------------------

READ = "read"
DML = "dml"
DDL = "ddl"
TXN_BEGIN = "txn_begin"
TXN_COMMIT = "txn_commit"
TXN_ROLLBACK = "txn_rollback"
SAVEPOINT = "savepoint"
OTHER = "other"

_WRITE_CLASSES = frozenset({DML, DDL, OTHER})

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_FIRST_KEYWORD_RE = re.compile(r"[A-Za-z]+")

_DML_KEYWORDS = frozenset({"insert", "update", "delete", "replace"})
_DDL_KEYWORDS = frozenset({"create", "drop", "alter", "reindex", "analyze", "vacuum", "truncate"})


def classify_statement(sql: str) -> str:
    """Classify a statement for transaction-safety bookkeeping."""
    text = _COMMENT_RE.sub(" ", sql.lstrip())
    match = _FIRST_KEYWORD_RE.search(text)
    if not match:
        return OTHER
    keyword = match.group(0).lower()

    if keyword == "begin":
        return TXN_BEGIN
    if keyword in ("commit", "end"):
        return TXN_COMMIT
    if keyword == "rollback":
        return TXN_ROLLBACK
    if keyword in ("savepoint", "release"):
        return SAVEPOINT
    if keyword in ("select", "explain", "pragma"):
        return READ
    if keyword == "with":
        # Data-modifying common table expressions (WITH ... INSERT/UPDATE/...)
        # write; everything else is a read.
        tail = text[match.end() :].lower()
        return DML if re.search(r"\b(insert|update|delete|replace)\b", tail) else READ
    if keyword in _DML_KEYWORDS:
        return DML
    if keyword in _DDL_KEYWORDS:
        return DDL
    return OTHER


def is_write(klass: str) -> bool:
    return klass in _WRITE_CLASSES


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


class ExecutionContext:
    """Budgets, counters and lifecycle state for one statement."""

    def __init__(
        self,
        sql: str,
        sequence: int = 0,
        budget: Budget | None = None,
        temp_dir: str | None = None,
        lock_wait_timeout: float | None = None,
    ) -> None:
        self.sql = sql
        self.sequence = sequence
        self.budget = budget or Budget.unlimited()
        self.temp_dir = temp_dir
        self.lock_wait_timeout = lock_wait_timeout

        self.state: str = RUNNING
        self.abort_reason: str | None = None
        # The Python exception the abort pipeline decided to surface for this
        # context, assigned by the executor; None while still unresolved.
        self.mapped_error: BaseException | None = None

        self.started: float = monotonic()
        self.finished_at: float | None = None
        self.deadline: float | None = self.started + self.budget.wall_seconds if self.budget.wall_seconds is not None else None

        # Counters. steps/rows/bytes are only written by the stepping thread;
        # reads from other threads are benign under the GIL (torn reads of an
        # int are impossible in CPython).
        self.steps = 0
        self.rows = 0
        self.bytes = 0
        self.temp_bytes_peak = 0

        # Lock-wait bookkeeping (monotonic seconds accumulated between busy
        # handler invocations).
        self.lock_invocations = 0
        self.lock_wait_seconds = 0.0
        self._lock_wait_started: float | None = None

        self._cancel = threading.Event()

    # -- cancellation -----------------------------------------------------

    def request_cancel(self) -> None:
        """Request cancellation.  Idempotent and safe from any thread."""
        self._cancel.set()
        if self.state == RUNNING:
            self.state = CANCELLING

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    # -- VM callbacks -----------------------------------------------------

    def on_progress(self) -> int:
        """SQLite progress handler: return 1 to abort, 0 to continue.

        Deliberately tiny: integer arithmetic and one event check on the
        fast path; ``monotonic()`` is read at most every 64 ticks and the
        temp directory is scanned at most every 4096 ticks.
        """
        steps = self.steps + 1
        self.steps = steps

        if self._cancel.is_set():
            self.abort_reason = USER_CANCEL
            return 1

        budget = self.budget
        step_limit = budget.vm_steps
        if step_limit is not None and steps >= step_limit:
            self.abort_reason = VM_STEPS
            return 1

        if (steps & _TIME_CHECK_MASK) == 0:
            if self.deadline is not None and monotonic() >= self.deadline:
                self.abort_reason = WALL_TIME
                return 1

        if self.temp_dir is not None and budget.temp_bytes is not None and (steps & _TEMP_CHECK_MASK) == 0:
            used = measure_dir(self.temp_dir)
            self.temp_bytes_peak = max(self.temp_bytes_peak, used)
            if used >= budget.temp_bytes:
                self.abort_reason = TEMP_DISK
                return 1

        return 0

    def on_busy(self, invocations: int) -> int:
        """SQLite busy handler: return 1 to keep waiting, 0 to abort.

        Only invoked while the statement is blocked on a lock, which is
        disjoint from normal VM stepping and from the progress handler, so
        wall-time and cancel checks are repeated here explicitly.
        """
        now = monotonic()
        wait_started = self._lock_wait_started
        if wait_started is not None:
            self.lock_wait_seconds += now - wait_started
        self._lock_wait_started = now
        self.lock_invocations = invocations

        if self._cancel.is_set():
            self.abort_reason = USER_CANCEL
            return 0
        if self.deadline is not None and now >= self.deadline:
            self.abort_reason = WALL_TIME
            return 0
        if self.lock_wait_timeout is not None and self.lock_wait_seconds >= self.lock_wait_timeout:
            self.abort_reason = LOCK_WAIT
            return 0
        return 1

    def busy_done(self) -> None:
        """Flush the final lock-wait interval once the statement returns."""
        wait_started = self._lock_wait_started
        if wait_started is not None:
            self.lock_wait_seconds += monotonic() - wait_started
            self._lock_wait_started = None

    # -- output accounting ------------------------------------------------

    def account_rows(self, row_count: int, payload_bytes: int) -> None:
        """Account a fetched chunk of rows against the output budgets.

        With a limit of N the (N+1)-th row is the first one that is not
        emitted, so the comparison is strictly greater-than.
        """
        self.rows += row_count
        self.bytes += payload_bytes
        budget = self.budget
        if budget.output_rows is not None and self.rows > budget.output_rows:
            raise ExecutionLimit(OUTPUT_ROWS, budget.output_rows, self.rows)
        if budget.output_bytes is not None and self.bytes > budget.output_bytes:
            raise ExecutionLimit(OUTPUT_BYTES, budget.output_bytes, self.bytes)

    # -- lifecycle --------------------------------------------------------

    def finish(self, state: str) -> None:
        if self.state in TERMINAL_STATES:
            return
        self.state = state
        self.finished_at = monotonic()

    @property
    def elapsed(self) -> float:
        if self.finished_at is not None:
            return self.finished_at - self.started
        return monotonic() - self.started

    # -- remaining budgets (toolbar) -------------------------------------

    def remaining_wall(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - monotonic())

    def remaining(self, limit: float | None, used: float) -> int | float | None:
        if limit is None:
            return None
        return max(0, limit - used)


# ---------------------------------------------------------------------------
# Transaction guard
# ---------------------------------------------------------------------------


class TransactionGuard:
    """Tracks whether the current transaction may still be used.

    The guard is a model of observable SQLite state, not an independent lock
    manager: taint decisions take the *observed* ``in_transaction`` value
    after the aborted statement returns.
    """

    def __init__(self) -> None:
        self.tainted = False

    def reset(self) -> None:
        self.tainted = False

    def check_permitted(self, klass: str) -> None:
        """Raise unless *klass* may execute against the current txn state."""
        if not self.tainted:
            return
        if klass != TXN_ROLLBACK:
            raise TransactionTaintedError("Transaction is tainted after an aborted statement; " "only ROLLBACK is permitted.")

    def note_success(self, klass: str) -> None:
        if klass in (TXN_COMMIT, TXN_ROLLBACK):
            self.reset()

    def note_abort(self, klass: str, reason: str, in_transaction: bool) -> bool:
        """Apply the taint rules and return the resulting taint flag.

        Rules (see module docstring for the SQLite guarantees behind them):

        * no open transaction                  -> nothing can be tainted;
        * VM abort of a read or DDL statement  -> statement rollback, txn
                                                   survives, not tainted;
        * VM abort of DML while a transaction
          is somehow still open                -> impossible per SQLite
                                                   guarantees, defensively
                                                   tainted;
        * VM abort of COMMIT with txn open     -> tainted;
        * output row/byte budget                -> statement rollback, not
                                                   tainted;
        * lock wait (SQLITE_BUSY)              -> statement rollback, not
                                                   tainted;
        * irrecoverable engine error           -> tainted.
        """
        if not in_transaction:
            self.tainted = False
            return False

        if reason in VM_ABORT_REASONS:
            if klass == TXN_COMMIT:
                self.tainted = True
            elif klass == DML:
                # Should be unreachable: interrupted INSERT/UPDATE/DELETE
                # rolls back the whole transaction, so an open txn here means
                # the engine state contradicts its documented guarantee.
                self.tainted = True
            else:
                self.tainted = False
        elif reason in (OUTPUT_ROWS, OUTPUT_BYTES):
            # Python-side fetch limits: cursor finalization performs a
            # statement rollback; explicit transaction survives.
            self.tainted = False
        elif reason == LOCK_WAIT:
            self.tainted = False
        else:
            # SEVERE_ERROR or anything unclassified: conservatively tainted.
            self.tainted = True
        return self.tainted


# ---------------------------------------------------------------------------
# Multi-statement batches
# ---------------------------------------------------------------------------

STOP = "stop"
CONTINUE = "continue"
ROLLBACK = "rollback"

_VALID_POLICIES = frozenset({STOP, CONTINUE, ROLLBACK})


def parse_policy(value: str | None) -> str:
    policy = (value or STOP).strip().lower()
    if policy not in _VALID_POLICIES:
        raise ValueError(f"batch_cancel_policy must be one of: stop, continue, rollback (got {value!r})")
    return policy


class BatchController:
    """Coordinates cancellation across one multi-statement batch."""

    def __init__(self, policy: str = STOP, size: int = 1) -> None:
        self.policy = policy
        self.size = size
        self._aborted = threading.Event()

    def request_abort(self) -> None:
        """Force the batch to stop after the current statement (escalation)."""
        self._aborted.set()

    def abort_requested(self) -> bool:
        return self._aborted.is_set()

    def decide(self, ctx: ExecutionContext, reason: str) -> str:
        """Decide what happens to the remaining statements.

        Escalation (a second Ctrl-C, or Ctrl-C during output) always means
        stop, overriding the configured policy.
        """
        if self._aborted.is_set():
            return STOP
        return self.policy


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def measure_dir(path: str) -> int:
    """Return the sum of regular-file sizes under *path* (non-recursive).

    SQLite temp files are plain files in a single directory; symlinks are not
    followed and missing entries are ignored so transient create/unlink races
    cannot surface.
    """
    total = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


# ---------------------------------------------------------------------------
# Error text classification (Python 3.9 has no sqlite_errorcode)
# ---------------------------------------------------------------------------

_BUSY_MARKERS = ("database is locked", "database table is locked", "database schema is locked")
_INTERRUPT_MARKERS = ("interrupted",)
_SEVERE_MARKERS = (
    "disk i/o error",
    "database disk image is malformed",
    "file is not a database",
    "out of memory",
    "database or disk is full",
    "no space left on device",
)


def is_busy_message(message: str) -> bool:
    return any(marker in message for marker in _BUSY_MARKERS)


def is_interrupted_message(message: str) -> bool:
    return any(marker in message for marker in _INTERRUPT_MARKERS)


def is_severe_message(message: str) -> bool:
    return any(marker in message for marker in _SEVERE_MARKERS)


# ---------------------------------------------------------------------------
# Toolbar formatting
# ---------------------------------------------------------------------------


def _short_int(value: int) -> str:
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        return _scale(value, 1000, "k")
    if value < 1_000_000_000:
        return _scale(value, 1_000_000, "M")
    return _scale(value, 1_000_000_000, "G")


def _scale(value: int, base: int, suffix: str) -> str:
    scaled = value / base
    text = f"{scaled:.1f}".rstrip("0").rstrip(".")
    return text + suffix


def _bytes(value: int | None) -> str:
    if value is None:
        return "∞"
    if value < 1024:
        return f"{value}B"
    if value < 1024 * 1024:
        return _scale(value, 1024, "KB")
    if value < 1024 * 1024 * 1024:
        return _scale(value, 1024 * 1024, "MB")
    return _scale(value, 1024 * 1024 * 1024, "GB")


def _seconds(value: float | None) -> str:
    if value is None:
        return "∞"
    return f"{value:.1f}s"


def _int_value(value: int | None) -> str:
    return "∞" if value is None else _short_int(value)


def format_toolbar(
    ctx: ExecutionContext | None,
    budget: Budget,
    tainted: bool,
    in_transaction: bool,
) -> str:
    """Return the one-line budget/state text for the bottom toolbar."""
    state = ctx.state if ctx is not None else IDLE
    parts = [f"[{state}]"]
    if tainted:
        parts.append("TXN-TAINTED")
    elif in_transaction:
        parts.append("txn")

    if ctx is not None:
        used_wall = ctx.elapsed
        remaining_wall = ctx.remaining(budget.wall_seconds, used_wall)
        remaining_steps = ctx.remaining(budget.vm_steps, ctx.steps)
        remaining_rows = ctx.remaining(budget.output_rows, ctx.rows)
        remaining_bytes = ctx.remaining(budget.output_bytes, ctx.bytes)
        remaining_temp = ctx.remaining(budget.temp_bytes, ctx.temp_bytes_peak)
    else:
        remaining_wall = budget.wall_seconds
        remaining_steps = budget.vm_steps
        remaining_rows = budget.output_rows
        remaining_bytes = budget.output_bytes
        remaining_temp = budget.temp_bytes

    parts.append("t=" + _seconds(remaining_wall))
    parts.append("vm=" + _int_value(cast("int|None", remaining_steps)))
    parts.append("rows=" + _int_value(cast("int|None", remaining_rows)))
    parts.append("b=" + _bytes(cast("int|None", remaining_bytes)))
    parts.append("tmp=" + _bytes(cast("int|None", remaining_temp)))
    return " ".join(parts)
