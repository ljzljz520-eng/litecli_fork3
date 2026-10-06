from __future__ import annotations

import logging
import os
import os.path
import shutil
import tempfile
import time
from collections.abc import Callable, Generator, Iterable
from contextlib import closing
from typing import Any, cast
from urllib.parse import urlparse

import sqlparse

try:
    import sqlean as _sqlite3

    _sqlite3.extensions.enable_all()
except ImportError:
    import sqlite3 as _sqlite3

from litecli import execution as exec_ctx

# Batch action names used by the per-statement abort pipeline.
from litecli.execution import CONTINUE as ACTION_CONTINUE
from litecli.execution import (
    LOCK_WAIT,
    SEVERE_ERROR,
    USER_CANCEL,
    BatchController,
    Budget,
    ExecutionContext,
    ExecutionLimit,
    TransactionGuard,
    classify_statement,
    is_busy_message,
    is_interrupted_message,
    is_severe_message,
)
from litecli.execution import ROLLBACK as ACTION_ROLLBACK
from litecli.execution import STOP as ACTION_STOP
from litecli.packages import special
from litecli.packages.special.utils import check_if_sqlitedotcommand

sqlite3 = cast(Any, _sqlite3)
OperationalError = sqlite3.OperationalError

_logger = logging.getLogger(__name__)

# FIELD_TYPES = decoders.copy()
# FIELD_TYPES.update({
#     FIELD_TYPE.NULL: type(None)
# })


class SQLExecute:
    databases_query = """
        PRAGMA database_list
    """

    tables_query = """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'sqlean_%'
        ORDER BY 1
    """

    table_columns_query = """
        SELECT m.name as tableName, p.name as columnName
        FROM sqlite_master m
        JOIN pragma_table_info((m.name)) p
        WHERE m.type IN ('table','view') AND m.name NOT LIKE 'sqlite_%' AND m.name NOT LIKE 'sqlean_%'
        ORDER BY tableName, columnName
    """

    indexes_query = """
        SELECT name, sql
        FROM sqlite_master
        WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
        ORDER BY 1
    """

    functions_query = '''SELECT ROUTINE_NAME FROM INFORMATION_SCHEMA.ROUTINES
    WHERE ROUTINE_TYPE="FUNCTION" AND ROUTINE_SCHEMA = "%s"'''

    def __init__(self, database: str | None):
        self.dbname: str | None = database
        self._server_type: tuple[str, str] | None = None
        # Connection can be sqlite3.Connection or sqlean.sqlite3 connection.
        self.conn: Any | None = None

        # Execution governance: budgets, state machine and batch policy.
        self.budget: Budget = Budget.unlimited()
        self.batch_policy: str = exec_ctx.STOP
        self.progress_interval: int = 1000
        self.lock_wait_timeout: float | None = None
        self.guard = TransactionGuard()
        self._batch: BatchController | None = None
        self._current_ctx: ExecutionContext | None = None
        # True only while the generator is suspended at a yield handing a
        # statement result to the consumer. Distinguishes "VDBE executing"
        # from "output phase" so Ctrl-C during output aborts the batch.
        self._stmt_delivering = False
        self.last_ctx: ExecutionContext | None = None
        self._temp_dir: str | None = None
        self._native_busy = False
        self._last_cancel_ts: float | None = None
        # Notice sink; replaced by LiteCli with an echo() bound method.
        self.notice: Callable[..., None] = lambda message, **_: _logger.debug(message)

        if not database:
            _logger.debug("Database is not specified. Skip connection.")
            return
        self.connect()

    def configure(
        self,
        budget: Budget | None = None,
        batch_policy: str | None = None,
        progress_interval: int | None = None,
        lock_wait_timeout: float | None = None,
    ) -> None:
        """Apply execution settings; re-installs handlers if connected."""
        if budget is not None:
            self.budget = budget
        if batch_policy is not None:
            self.batch_policy = batch_policy
        if progress_interval is not None:
            self.progress_interval = progress_interval
        self.lock_wait_timeout = lock_wait_timeout
        if self.conn is not None:
            self._install_handlers(self.conn)

    def connect(self, database: str | None = None) -> None:
        db = database or self.dbname
        _logger.debug("Connection DB Params: \n\tdatabase: %r", db)
        if db is None:
            # Nothing to connect to.
            return

        location = urlparse(db)
        if location.scheme and location.scheme == "file":
            uri = True
            db_name = db
            db_filename = location.path
        else:
            uri = False
            db_filename = db_name = os.path.expanduser(db)
            db_dir_name = os.path.dirname(os.path.abspath(db_filename))
            if not os.path.exists(db_dir_name):
                raise Exception(f"Path does not exist: {db_dir_name}")

        conn = sqlite3.connect(database=db_name, isolation_level=None, uri=uri)
        conn.text_factory = lambda x: x.decode("utf-8", "backslashreplace")
        if self.conn:
            self.conn.close()
        self._teardown_temp_dir()

        self.conn = conn
        # A fresh connection has no transaction and no taint history.
        self.guard.reset()
        self._install_handlers(conn)
        # Update them after the connection is made to ensure that it was a
        # successful connection.
        self.dbname = db_filename

    # -- handler installation ---------------------------------------------

    def _install_handlers(self, conn: Any) -> None:
        conn.set_progress_handler(self._on_progress, self.progress_interval)
        # Native busy handlers exist on Python 3.13+; when unavailable we
        # emulate busy-wait handling around execute() (see _execute_sql).
        if hasattr(conn, "set_busy_handler"):
            conn.set_busy_handler(self._on_busy)
            self._native_busy = True
        else:
            self._native_busy = False
            # busy_timeout stays 0: SQLITE_BUSY surfaces immediately and our
            # emulation decides how long to keep retrying.
        self._setup_temp_dir(conn)

    def _on_progress(self) -> int:
        ctx = self._current_ctx
        if ctx is None:
            return 0
        return ctx.on_progress()

    def _on_busy(self, invocations: int) -> int:
        ctx = self._current_ctx
        if ctx is None:
            return 1
        return ctx.on_busy(invocations)

    # -- temp directory for the disk budget -------------------------------

    def _setup_temp_dir(self, conn: Any) -> None:
        self._teardown_temp_dir()
        if self.budget.temp_bytes is None:
            return
        directory = tempfile.mkdtemp(prefix="litecli-tmp-")
        try:
            # sqlite3_os_init picks the temp directory from this (deprecated
            # but still functional) per-connection pragma; quote as a string
            # literal. Any temp files (temp DB, materializations, sorts) then
            # land in this directory and can be measured cheaply.
            conn.execute('PRAGMA temp_store_directory = "{}"'.format(directory.replace('"', "")))
        except OperationalError as e:
            _logger.debug("Unable to set temp_store_directory: %r", e)
            shutil.rmtree(directory, ignore_errors=True)
            return
        self._temp_dir = directory

    def _teardown_temp_dir(self) -> None:
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    # -- cancellation entry point -----------------------------------------

    def request_cancel(self) -> None:
        """Signal cancellation from a Ctrl-C handler.

        Ctrl-C while the VM executes a statement cancels that statement; a
        second Ctrl-C within the escalation window escalates to the batch.
        Ctrl-C during the output phase, between statements, or with no
        active context aborts the whole batch directly.  The SQLite
        interrupt is always issued as an out-of-band nudge; the persistent
        context flags cover its no-op-when-idle race.
        """
        now = time.monotonic()
        ctx = self._current_ctx
        output_phase = self._stmt_delivering or (ctx is not None and ctx.state in exec_ctx.TERMINAL_STATES)
        if ctx is None or output_phase:
            if self._batch is not None:
                self._batch.request_abort()
        else:
            ctx.request_cancel()
            if self._last_cancel_ts is not None and now - self._last_cancel_ts < 0.5:
                if self._batch is not None:
                    self._batch.request_abort()
        self._last_cancel_ts = now
        self._interrupt_conn()

    def _interrupt_conn(self) -> None:
        conn = self.conn
        if conn is not None:
            try:
                conn.interrupt()
            except Exception as e:  # defensive: sqlite3/sqlean differences
                _logger.debug("interrupt() failed: %r", e)

    def run(self, statement: str) -> Iterable[tuple]:
        """Execute the sql in the database and return the results. The results
        are a list of tuples. Each tuple has 4 values
        (title, rows, headers, status).
        """
        # Remove spaces and EOL
        statement = statement.strip()
        if not statement:  # Empty string
            yield (None, None, None, None)
            return

        # Split the sql into separate queries and run each one.
        # Unless it's saving a favorite query, in which case we
        # want to save them all together.
        if statement.startswith("\\fs"):
            components = [statement]
        else:
            components = sqlparse.split(statement)

        batch = BatchController(self.batch_policy, len(components))
        self._batch = batch

        try:
            for index, component in enumerate(components):
                # Remove spaces, eol and semi-colons.
                sql = component.rstrip(";")

                # \G is treated specially since we have to set the expanded output.
                if sql.endswith("\\G"):
                    special.set_expanded_output(True)
                    sql = sql[:-2].strip()

                if not self.conn and not (
                    sql.startswith(".open")
                    or sql.lower().startswith("use")
                    or sql.startswith("\\u")
                    or sql.startswith("\\?")
                    or sql.startswith("\\q")
                    or sql.startswith("help")
                    or sql.startswith("exit")
                    or sql.startswith("quit")
                ):
                    _logger.debug("Not connected to database. Will not run statement: %s.", sql)
                    raise OperationalError("Not connected to database.")

                klass = classify_statement(sql)
                # Tainted transactions reject everything except ROLLBACK.
                self.guard.check_permitted(klass)

                cur = self.conn.cursor() if self.conn else None
                ctx = ExecutionContext(sql, index, self.budget, self._temp_dir, self.lock_wait_timeout)
                self._current_ctx = ctx
                try:
                    if batch.abort_requested():
                        raise ExecutionLimit(USER_CANCEL)

                    try:  # Special command
                        _logger.debug("Trying a dbspecial command. sql: %r", sql)
                        special_results = special.execute(cur, sql)
                    except special.CommandNotFound:  # Regular SQL
                        if check_if_sqlitedotcommand(sql):
                            self._stmt_delivering = True
                            yield ("dot command not implemented", None, None, None)
                            self._stmt_delivering = False
                        else:
                            _logger.debug("Regular sql statement. sql: %r", sql)
                            assert cur is not None
                            self._execute_sql(cur, sql, ctx)
                            self._stmt_delivering = True
                            yield self.get_result(cur, ctx)
                            self._stmt_delivering = False
                    else:
                        for result in special_results:
                            self._stmt_delivering = True
                            yield self._account_special(ctx, result)
                            self._stmt_delivering = False

                    # Ctrl-C during the output phase marks the batch aborted;
                    # raise through the same pipeline rather than completing
                    # the context.
                    if batch.abort_requested():
                        raise ExecutionLimit(USER_CANCEL)

                    ctx.busy_done()
                    ctx.finish(exec_ctx.COMPLETED)
                    self.guard.note_success(klass)
                except BaseException as exc:
                    action = self._abort_pipeline(ctx, klass, exc)
                    if isinstance(exc, GeneratorExit):
                        # Consumer went away; safety pipeline already ran.
                        raise
                    if action == ACTION_CONTINUE:
                        yield (None, None, None, "statement {} skipped ({}).".format(index + 1, ctx.abort_reason or "cancelled"))
                        continue
                    if action == ACTION_ROLLBACK:
                        self._policy_rollback()
                    mapped = ctx.mapped_error
                    if mapped is None or mapped is exc:
                        raise
                    raise mapped from exc
                finally:
                    if self._current_ctx is ctx:
                        self._current_ctx = None
                    self.last_ctx = ctx
        finally:
            self._batch = None

    # -- statement execution ---------------------------------------------

    def _execute_sql(self, cur: Any, sql: str, ctx: ExecutionContext) -> None:
        """Execute regular SQL, emulating a busy handler when the runtime
        (Python < 3.13) does not expose a native one."""
        if self._native_busy:
            cur.execute(sql)
            return

        attempts = 0
        while True:
            try:
                cur.execute(sql)
                return
            except OperationalError as e:
                if not is_busy_message(str(e)):
                    raise
                # Retrying is only safe before the statement executed any
                # bytecode (no progress tick). After that point SQLite has
                # already performed a statement rollback and we just report
                # the lock abort.
                if ctx.steps > 0:
                    ctx.abort_reason = LOCK_WAIT
                    raise
                attempts += 1
                if not ctx.on_busy(attempts):
                    raise ExecutionLimit(ctx.abort_reason or LOCK_WAIT)
                # Backoff between lock probes; a Ctrl-C here is mapped to a
                # user cancel by the abort pipeline.
                time.sleep(min(0.002 * attempts, 0.05))

    def _account_special(self, ctx: ExecutionContext, result: tuple) -> tuple:
        """Account a special-command result tuple against output budgets."""
        title, rows, headers, status = result
        if rows is None:
            return result
        if hasattr(rows, "fetchmany"):
            materialized: list = []
            while True:
                chunk = rows.fetchmany(1000)
                if not chunk:
                    break
                ctx.account_rows(len(chunk), payload_bytes(chunk))
                materialized.extend(chunk)
            rows = materialized
        else:
            rows = list(rows)
            ctx.account_rows(len(rows), payload_bytes(rows))
        return (title, rows, headers, status)

    def _abort_pipeline(self, ctx: ExecutionContext, klass: str, exc: BaseException) -> str:
        """Classify an exception, update guard/state and choose batch action."""
        if isinstance(exc, GeneratorExit):
            # The consumer closed us. If a cancel intent exists -- batch
            # escalation (Ctrl-C during output) or the context itself was
            # cancelling -- honor the configured batch policy, including a
            # policy rollback.
            batch = self._batch
            batch_aborted = batch is not None and batch.abort_requested()
            ctx_cancelled = ctx is not None and ctx.cancel_requested()
            if (
                (batch_aborted or ctx_cancelled)
                and batch is not None
                and batch.policy == exec_ctx.ROLLBACK
                and self.conn is not None
                and self.conn.in_transaction
            ):
                self._policy_rollback()
            return ACTION_STOP

        reason = self._classify_abort(ctx, exc)
        in_transaction = bool(self.conn.in_transaction) if self.conn is not None else False
        if reason is None:
            # Ordinary SQL error (syntax, missing table, ...): same behavior
            # as historically -- raise immediately, no batch policy.
            ctx.finish(exec_ctx.FAILED)
            ctx.mapped_error = exc
            return ACTION_STOP

        ctx.busy_done()
        self.guard.note_abort(klass, reason, in_transaction)
        if reason == SEVERE_ERROR:
            ctx.finish(exec_ctx.FAILED)
            # Preserve the engine message for display; taint already applied.
            ctx.mapped_error = exc
        else:
            ctx.finish(exec_ctx.INTERRUPTED)
            ctx.mapped_error = ExecutionLimit(reason, limit_for(reason, ctx), used_for(reason, ctx))

        batch = self._batch
        if batch is None:
            return ACTION_STOP
        action = batch.decide(ctx, reason)
        if action == ACTION_ROLLBACK and not in_transaction:
            action = ACTION_STOP
        return action

    def _classify_abort(self, ctx: ExecutionContext, exc: BaseException) -> str | None:
        if isinstance(exc, ExecutionLimit):
            return exc.reason
        if isinstance(exc, KeyboardInterrupt):
            # SIGINT landed inside the generator: convert to a user cancel.
            ctx.request_cancel()
            self._interrupt_conn()
            return USER_CANCEL
        if isinstance(exc, OperationalError):
            message = str(exc)
            if is_interrupted_message(message):
                # Abort came from the progress handler/interrupt; use the
                # reason the handler recorded before returning non-zero.
                return ctx.abort_reason or USER_CANCEL
            if is_busy_message(message):
                return LOCK_WAIT
            if is_severe_message(message):
                return SEVERE_ERROR
        return None

    def _policy_rollback(self) -> None:
        if self._force_rollback():
            self.notice("Query cancelled; transaction rolled back.", err=True, fg="yellow")
        else:
            self.notice("ROLLBACK could not complete; transaction remains tainted.", err=True, fg="red")

    def _force_rollback(self) -> bool:
        """Roll back cancel-proof: retries BUSY/INTERRUPT until done."""
        conn = self.conn
        if conn is None:
            return False
        if not conn.in_transaction:
            self.guard.reset()
            return True
        # Inert installed callbacks while rolling back.
        self._current_ctx = None
        for attempt in range(100):
            try:
                conn.execute("ROLLBACK")
                self.guard.reset()
                return True
            except OperationalError as e:
                message = str(e)
                if is_busy_message(message):
                    time.sleep(min(0.01 * (attempt + 1), 0.2))
                    continue
                if is_interrupted_message(message):
                    # An interrupted ROLLBACK leaves the transaction active.
                    continue
                return False
        return False

    def get_result(self, cursor: Any, ctx: ExecutionContext | None = None) -> tuple[str | None, list | None, list | None, str]:
        """Get the current result's data from the cursor."""
        title = headers = None

        # cursor.description is not None for queries that return result sets,
        # e.g. SELECT.
        if cursor.description is not None:
            headers = [x[0] for x in cursor.description]
            status = "{count} row{s} in set"
            if ctx is None:
                cursor = list(cursor)
                rowcount = len(cursor)
            else:
                # Fetch in bounded chunks so the output row/byte budgets are
                # enforced while stepping, not after materializing the set.
                rows: list = []
                while True:
                    chunk = cursor.fetchmany(1000)
                    if not chunk:
                        break
                    ctx.account_rows(len(chunk), payload_bytes(chunk))
                    rows.extend(chunk)
                cursor = rows
                rowcount = len(rows)
        else:
            _logger.debug("No rows in result.")
            if cursor.rowcount == -1:
                status = "Query OK"
            else:
                status = "Query OK, {count} row{s} affected"
            rowcount = cursor.rowcount
            cursor = None

        status = status.format(count=rowcount, s="" if rowcount == 1 else "s")

        return (title, cursor, headers, status)

    def tables(self) -> Generator[tuple[str], None, None]:
        """Yields table names"""
        if not self.conn:
            return
        with closing(self.conn.cursor()) as cur:
            _logger.debug("Tables Query. sql: %r", self.tables_query)
            cur.execute(self.tables_query)
            for row in cur:
                yield row

    def table_columns(self) -> Generator[tuple[str, str], None, None]:
        """Yields column names"""
        if not self.conn:
            return
        with closing(self.conn.cursor()) as cur:
            _logger.debug("Columns Query. sql: %r", self.table_columns_query)
            cur.execute(self.table_columns_query)
            for row in cur:
                yield row

    def databases(self) -> Generator[str, None, None]:
        if not self.conn:
            return

        with closing(self.conn.cursor()) as cur:
            _logger.debug("Databases Query. sql: %r", self.databases_query)
            for row in cur.execute(self.databases_query):
                yield row[1]

    def functions(self) -> Iterable[tuple]:
        """Yields tuples of (schema_name, function_name)"""
        if not self.conn:
            return
        with closing(self.conn.cursor()) as cur:
            _logger.debug("Functions Query. sql: %r", self.functions_query)
            cur.execute(self.functions_query % self.dbname)
            for row in cur:
                yield row

    def server_type(self) -> tuple[str, str]:
        self._server_type = ("sqlite3", "3")
        return self._server_type


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def payload_bytes(rows: Iterable[tuple]) -> int:
    """Sum the UTF-8 payload size of all values in *rows*.

    SQL NULL contributes zero bytes (``str(None)`` would misrepresent the
    database value).  This is the deterministic definition used for the
    output-byte budget: pre-formatting payload bytes.
    """
    total = 0
    for row in rows:
        for value in row:
            if value is None:
                continue
            total += len(str(value).encode("utf-8", "backslashreplace"))
    return total


def limit_for(reason: str, ctx: ExecutionContext) -> int | float | None:
    budget = ctx.budget
    return {
        "wall_time": budget.wall_seconds,
        "vm_steps": budget.vm_steps,
        "output_rows": budget.output_rows,
        "output_bytes": budget.output_bytes,
        "temp_disk": budget.temp_bytes,
    }.get(reason)


def used_for(reason: str, ctx: ExecutionContext) -> int | float:
    return {
        "wall_time": ctx.elapsed,
        "vm_steps": ctx.steps,
        "output_rows": ctx.rows,
        "output_bytes": ctx.bytes,
        "temp_disk": ctx.temp_bytes_peak,
    }.get(reason, 0)
