from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prompt_toolkit.application import get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode

from litecli.execution import format_toolbar


def create_toolbar_tokens_func(cli: Any, show_fish_help: Callable[[], bool]) -> Callable[[], list[tuple[str, str]]]:
    """Return a function that generates the toolbar tokens."""

    def get_toolbar_tokens() -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        result.append(("class:bottom-toolbar", " "))

        if cli.multi_line:
            result.append(("class:bottom-toolbar", " (Semi-colon [;] will end the line) "))

        if cli.multi_line:
            result.append(("class:bottom-toolbar.on", "[F3] Multiline: ON  "))
        else:
            result.append(("class:bottom-toolbar.off", "[F3] Multiline: OFF  "))
        if cli.prompt_app.editing_mode == EditingMode.VI:
            result.append(("class:botton-toolbar.on", f"Vi-mode ({_get_vi_mode()})"))

        if show_fish_help():
            result.append(("class:bottom-toolbar", "  Right-arrow to complete suggestion"))

        if cli.completion_refresher.is_refreshing():
            result.append(("class:bottom-toolbar", "     Refreshing completions..."))

        sqlexecute = getattr(cli, "sqlexecute", None)
        if sqlexecute is not None and getattr(sqlexecute, "conn", None) is not None:
            tainted = bool(sqlexecute.guard.tainted)
            in_transaction = bool(sqlexecute.conn.in_transaction)
            budget_text = format_toolbar(sqlexecute.last_ctx, sqlexecute.budget, tainted, in_transaction)
            if tainted:
                style = "class:bottom-toolbar.transaction.failed"
            elif in_transaction:
                style = "class:bottom-toolbar.transaction.valid"
            else:
                style = "class:bottom-toolbar"
            result.append((style, "  " + budget_text + " "))

        return result

    return get_toolbar_tokens


def _get_vi_mode() -> str:
    """Get the current vi mode for display."""
    return {
        InputMode.INSERT: "I",
        InputMode.NAVIGATION: "N",
        InputMode.REPLACE: "R",
        InputMode.INSERT_MULTIPLE: "M",
        InputMode.REPLACE_SINGLE: "R",
    }[get_app().vi_state.input_mode]
