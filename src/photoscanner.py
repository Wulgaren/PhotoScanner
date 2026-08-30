#!/usr/bin/env python3
"""Interactive launcher: pick a job, fill params, print the command, run it."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.prompt import Confirm, Prompt

from photo_scanner.paths import (
    LAUNCHER_STATE,
    MODEL_PATH,
    OUTPUT_DIR,
    REPO_ROOT,
    SRC_DIR,
    ensure_data_dirs,
)

console = Console()
QUIT = -1
DECLINED = -2

COMMON_HELP = {
    "train": "Learn taste from favorited photos before a cutoff date.",
    "scan": "Score newer photos and write deletion suggestions.",
    "review": "Open the browser UI to keep or cull suggestions.",
    "learn": "Turn album rescues and leftover deletes into training examples.",
    "move": "Put low-scoring photos in the Photos “To Delete” album.",
    "captions": "Fill empty captions in a smart album from learned usernames.",
    "twitter": "Curate Twitter/X images from Discord (TweetShift) with the same model.",
    "summarize": "Write announcements_summary.txt via Apple Foundation Models.",
}

MENU_ORDER = ("train", "scan", "review", "learn", "move", "captions", "twitter", "summarize")
MENU_LABELS = {
    "train": "Train",
    "scan": "Scan",
    "review": "Review (browser)",
    "learn": "Learn from feedback",
    "move": "Move to “To Delete” album",
    "captions": "Add captions",
    "twitter": "Twitter curator",
    "summarize": "Summarize announcements",
}
CYCLE = ("train", "scan", "review", "learn")
SCRIPTS = {
    "train": "train_model.py",
    "scan": "scan_photos.py",
    "review": "review_gui.py",
    "learn": "learn_from_feedback.py",
    "move": "move_to_album.py",
    "captions": "add_captions.py",
    "twitter": "twitter_curator.py",
    "summarize": "summarize_announcements.py",
}


def load_state() -> dict[str, dict[str, Any]]:
    if not LAUNCHER_STATE.exists():
        return {}
    try:
        data = json.loads(LAUNCHER_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: dict[str, dict[str, Any]]) -> None:
    ensure_data_dirs()
    LAUNCHER_STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def last(state: dict, tool: str, key: str, default: Any) -> Any:
    block = state.get(tool) or {}
    if key not in block or block[key] is None:
        return default
    return block[key]


def put(state: dict, tool: str, **kwargs: Any) -> None:
    block = dict(state.get(tool) or {})
    block.update(kwargs)
    state[tool] = block
    save_state(state)


def ask_str(label: str, default: str) -> str:
    return Prompt.ask(label, default=str(default)).strip()


def ask_float(label: str, default: float) -> float:
    while True:
        raw = Prompt.ask(label, default=str(default)).strip()
        try:
            return float(raw)
        except ValueError:
            console.print("[red]Need a number.[/red]")


def ask_int(label: str, default: int) -> int:
    while True:
        raw = Prompt.ask(label, default=str(default)).strip()
        try:
            return int(raw)
        except ValueError:
            console.print("[red]Need a whole number.[/red]")


def ask_optional_int(label: str, default: int | None) -> int | None:
    shown = "" if default is None else str(default)
    raw = Prompt.ask(f"{label} [dim](empty = omit)[/dim]", default=shown).strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        console.print("[red]Need a whole number or empty.[/red]")
        return ask_optional_int(label, default)


def ask_optional_path(label: str, default: str | None) -> str | None:
    shown = default or ""
    raw = Prompt.ask(f"{label} [dim](empty = omit)[/dim]", default=shown).strip()
    return raw or None


def more_options() -> bool:
    return Confirm.ask("More options?", default=False)


def python_display() -> str:
    py = Path(sys.executable)
    if not py.is_absolute():
        py = (Path.cwd() / py).absolute()
    try:
        return str(py.relative_to(REPO_ROOT))
    except ValueError:
        return str(py)


def format_cmd(argv: list[str]) -> str:
    shown = [python_display()]
    script = Path(argv[1])
    try:
        shown.append(str(script.absolute().relative_to(REPO_ROOT)))
    except ValueError:
        shown.append(argv[1])
    shown.extend(argv[2:])
    return shlex.join(shown)


def run_argv(argv: list[str]) -> int:
    console.print()
    console.print("[bold]Will run[/bold]")
    console.print(f"  [gold1]{format_cmd(argv)}[/gold1]")
    console.print()
    if not Confirm.ask("Run it?", default=True):
        return DECLINED
    console.print()
    return subprocess.run(argv, cwd=REPO_ROOT).returncode


def has_model() -> bool:
    return MODEL_PATH.is_file()


def latest_scan() -> Path | None:
    files = sorted(OUTPUT_DIR.glob("scan_results_*.json"), reverse=True)
    return files[0] if files else None


def header(title: str, blurb: str) -> None:
    console.print()
    console.print(f"[bold gold1]── {title} ──[/bold gold1]")
    console.print(f"[dim]{blurb}[/dim]")
    console.print()


def cycle_gate(title: str, blurb: str) -> str:
    header(title, blurb)
    return Prompt.ask(
        "[bold]c[/bold] continue  [bold]s[/bold] skip  [bold]q[/bold] quit",
        choices=["c", "s", "q"],
        default="c",
        show_choices=False,
    )


def collect_train(state: dict) -> list[str]:
    cutoff = ask_str("Cutoff date (YYYY-MM-DD)", last(state, "train", "cutoff_date", "2023-11-18"))
    args: dict[str, Any] = {"cutoff_date": cutoff}
    extra: list[str] = ["--cutoff-date", cutoff]
    if more_options():
        sample = ask_optional_int("Sample size", last(state, "train", "sample_size", None))
        batch = ask_int("Batch size", last(state, "train", "batch_size", 32))
        args["sample_size"] = sample
        args["batch_size"] = batch
        if sample is not None:
            extra += ["--sample-size", str(sample)]
        if batch != 32:
            extra += ["--batch-size", str(batch)]
    put(state, "train", **args)
    return extra


def collect_scan(state: dict) -> list[str]:
    after = ask_str("Scan photos after (YYYY-MM-DD)", last(state, "scan", "after", "2023-11-18"))
    threshold = ask_float("Score threshold (0–1)", last(state, "scan", "threshold", 0.3))
    args: dict[str, Any] = {"after": after, "threshold": threshold}
    extra = ["--after", after, "--threshold", str(threshold)]
    if more_options():
        batch = ask_int("Batch size", last(state, "scan", "batch_size", 32))
        limit = ask_optional_int("Limit (photo count)", last(state, "scan", "limit", None))
        args["batch_size"] = batch
        args["limit"] = limit
        if batch != 32:
            extra += ["--batch-size", str(batch)]
        if limit is not None:
            extra += ["--limit", str(limit)]
    put(state, "scan", **args)
    return extra


def collect_review(state: dict) -> list[str]:
    mode = Prompt.ask(
        "Mode",
        choices=["flat", "grouped"],
        default=last(state, "review", "mode", "flat"),
    )
    threshold = ask_float("Starting threshold (0–1)", last(state, "review", "threshold", 0.5))
    args: dict[str, Any] = {"mode": mode, "threshold": threshold}
    extra = ["--mode", mode, "--threshold", str(threshold)]
    if more_options():
        results = ask_optional_path("Scan results JSON", last(state, "review", "results", None))
        page_size = ask_int("Page size (flat)", last(state, "review", "page_size", 3))
        port = ask_int("Port", last(state, "review", "port", 8765))
        no_open = Confirm.ask("Skip opening the browser?", default=bool(last(state, "review", "no_open", False)))
        args.update(results=results, page_size=page_size, port=port, no_open=no_open)
        if results:
            extra += ["--results", results]
        if page_size != 3:
            extra += ["--page-size", str(page_size)]
        if port != 8765:
            extra += ["--port", str(port)]
        if no_open:
            extra.append("--no-open")
    put(state, "review", **args)
    return extra


def collect_learn(state: dict) -> list[str]:
    extra: list[str] = []
    if more_options():
        album = ask_str("Album name", last(state, "learn", "album", "To Delete"))
        put(state, "learn", album=album)
        if album != "To Delete":
            extra += ["--album", album]
    return extra


def collect_move(state: dict) -> list[str]:
    threshold = ask_float("Score threshold (0–1)", last(state, "move", "threshold", 0.8))
    dry_run = Confirm.ask("Dry run (don’t touch Photos)?", default=bool(last(state, "move", "dry_run", False)))
    args: dict[str, Any] = {"threshold": threshold, "dry_run": dry_run}
    extra = ["--threshold", str(threshold)]
    if dry_run:
        extra.append("--dry-run")
    if more_options():
        album = ask_str("Album name", last(state, "move", "album", "To Delete"))
        results = ask_optional_path("Scan results JSON", last(state, "move", "results", None))
        args.update(album=album, results=results)
        if album != "To Delete":
            extra += ["--album", album]
        if results:
            extra += ["--results", results]
    put(state, "move", **args)
    return extra


def collect_captions(state: dict) -> list[str]:
    extra: list[str] = []
    if more_options():
        dry_run = Confirm.ask(
            "Dry run (don’t write captions or the map)?",
            default=bool(last(state, "captions", "dry_run", False)),
        )
        album = ask_str("Album name", last(state, "captions", "album", "Non-added photos"))
        put(state, "captions", dry_run=dry_run, album=album)
        if dry_run:
            extra.append("--dry-run")
        if album != "Non-added photos":
            extra += ["--album", album]
    return extra


def collect_twitter(state: dict) -> list[str]:
    hours = ask_int("Backfill hours (0 = live only)", last(state, "twitter", "hours", 0))
    no_listen = Confirm.ask("Exit after backfill (no live listen)?", default=bool(last(state, "twitter", "no_listen", False)))
    extra: list[str] = []
    args: dict[str, Any] = {"hours": hours, "no_listen": no_listen}
    if hours:
        extra += ["--hours", str(hours)]
    if no_listen:
        extra.append("--no-listen")
    if more_options():
        raw = Prompt.ask(
            "Score threshold [dim](empty = config.json)[/dim]",
            default="" if last(state, "twitter", "threshold", None) is None else str(last(state, "twitter", "threshold", None)),
        ).strip()
        threshold: float | None
        if raw == "":
            threshold = None
        else:
            try:
                threshold = float(raw)
            except ValueError:
                console.print("[red]Need a number or empty.[/red]")
                threshold = None
        args["threshold"] = threshold
        if threshold is not None:
            extra += ["--threshold", str(threshold)]
    put(state, "twitter", **args)
    return extra


def collect_summarize(state: dict) -> list[str]:
    dry_run = Confirm.ask("Print only (don’t write the file)?", default=bool(last(state, "summarize", "dry_run", False)))
    extra: list[str] = []
    args: dict[str, Any] = {"dry_run": dry_run}
    if dry_run:
        extra.append("--dry-run")
    if more_options():
        path = ask_optional_path("announcements.txt path", last(state, "summarize", "path", None))
        binary = ask_optional_path("summarizer binary", last(state, "summarize", "binary", None))
        args.update(path=path, binary=binary)
        if path:
            extra += ["--path", path]
        if binary:
            extra += ["--binary", binary]
    put(state, "summarize", **args)
    return extra


COLLECT: dict[str, Callable[[dict], list[str]]] = {
    "train": collect_train,
    "scan": collect_scan,
    "review": collect_review,
    "learn": collect_learn,
    "move": collect_move,
    "captions": collect_captions,
    "twitter": collect_twitter,
    "summarize": collect_summarize,
}


def argv_for(tool: str, extra: list[str]) -> list[str]:
    return [sys.executable, str(SRC_DIR / SCRIPTS[tool]), *extra]


def run_tool(state: dict, tool: str, *, gated: bool) -> int:
    title = MENU_LABELS[tool]
    blurb = COMMON_HELP[tool]
    if gated:
        action = cycle_gate(title, blurb)
        if action == "q":
            return QUIT
        if action == "s":
            console.print("[dim]Skipped.[/dim]")
            return 0
    else:
        header(title, blurb)

    if not prepare(state, tool):
        if gated:
            console.print("[dim]Skipped.[/dim]")
            return 0
        return QUIT

    extra = COLLECT[tool](state)
    code = run_argv(argv_for(tool, extra))
    if code == DECLINED:
        if gated:
            console.print("[dim]Skipped.[/dim]")
            return 0
        return QUIT
    return code


def prepare(state: dict, tool: str) -> bool:
    """Offer missing prior steps. Returns False if the user declines or a prior step fails."""
    if tool in ("scan", "twitter") and not has_model():
        if not Confirm.ask("No trained model. Train first?", default=True):
            console.print("[yellow]Need a model first.[/yellow]")
            return False
        code = run_tool(state, "train", gated=False)
        if code in (QUIT, DECLINED):
            return False
        if code != 0:
            console.print("[red]Train did not finish cleanly.[/red]")
            return False
        if not has_model():
            console.print("[red]Still no model after train.[/red]")
            return False

    if tool in ("review", "move") and latest_scan() is None:
        if not Confirm.ask("No scan results. Scan first?", default=True):
            console.print("[yellow]Need a scan first.[/yellow]")
            return False
        code = run_tool(state, "scan", gated=False)
        if code in (QUIT, DECLINED):
            return False
        if code != 0:
            console.print("[red]Scan did not finish cleanly.[/red]")
            return False
        if latest_scan() is None:
            console.print("[red]Still no scan results.[/red]")
            return False

    return True


def guided_cycle(state: dict) -> None:
    console.print()
    console.print("[bold]Guided cycle[/bold]  Train → Scan → Review → Learn")
    console.print("[dim]Skip any step. After the last one (or quit), you’re back at the shell.[/dim]")
    for tool in CYCLE:
        code = run_tool(state, tool, gated=True)
        if code == QUIT:
            console.print("[dim]Quit cycle.[/dim]")
            return
        if code != 0:
            console.print("[red]Stopped the cycle — that step exited with an error.[/red]")
            return


def pick_tool(state: dict) -> None:
    console.print()
    for i, tool in enumerate(MENU_ORDER, start=1):
        console.print(f"  [bold gold1]{i}[/bold gold1]  {MENU_LABELS[tool]}")
        console.print(f"     [dim]{COMMON_HELP[tool]}[/dim]")
    console.print("  [bold]q[/bold]  Quit")
    console.print()
    choice = Prompt.ask("Tool").strip().lower()
    if choice in {"q", "quit", ""}:
        return
    if choice.isdigit() and 1 <= int(choice) <= len(MENU_ORDER):
        tool = MENU_ORDER[int(choice) - 1]
    else:
        console.print("[red]Unknown choice.[/red]")
        return
    code = run_tool(state, tool, gated=False)
    if code not in (0, QUIT, DECLINED):
        sys.exit(code)


def main() -> None:
    ensure_data_dirs()
    state = load_state()

    console.print()
    console.print("[bold]PhotoScanner[/bold]  [dim]local taste · local files[/dim]")
    console.print()
    console.print("  [bold gold1]1[/bold gold1]  Guided cycle")
    console.print("     [dim]Train → Scan → Review → Learn[/dim]")
    console.print("  [bold gold1]2[/bold gold1]  Pick a tool")
    console.print("     [dim]One job, then done[/dim]")
    console.print("  [bold]q[/bold]  Quit")
    console.print()

    choice = Prompt.ask("Choose", choices=["1", "2", "q"], default="1", show_choices=False)
    if choice == "q":
        return
    if choice == "1":
        guided_cycle(state)
        return
    pick_tool(state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)
