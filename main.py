import argparse
import logging
import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
from rich.theme import Theme
from rich import box

from simple_term_menu import TerminalMenu

# Core runner logic
from scanner.runner import run_scan, settings
from scanner.notifier import TelegramNotifier

# ─── Logging Setup ──────────────────────────────────────────
DEV_MODE = True

logger = logging.getLogger(__name__)

def setup_logging(dev: bool = False):
    global DEV_MODE
    DEV_MODE = dev
    
    # ── Root Logger ──
    level = logging.DEBUG if dev else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Root takes all
    
    # Clear existing handlers
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    # 1. Console Handler (Respects DEV_MODE)
    c_level = logging.DEBUG if dev else logging.WARNING
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(c_level)
    c_handler.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(name)-15.15s │ %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(c_handler)

    # 2. File Handler (Always active, level follows 'dev')
    f_handler = logging.FileHandler("stockpy.log", mode="a", encoding="utf-8")
    f_handler.setLevel(level)
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(f_handler)

    # Specific logger settings
    logging.getLogger("scanner").setLevel(level)
    logging.getLogger(__name__).setLevel(level)
    
    # Suppress noisy third-party loggers
    for noisy in ("yfinance", "peewee", "urllib3", "curl_cffi"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(f"Logging initialized. Mode: {'DEV' if dev else 'NORMAL'}. File: stockpy.log")

# ─── Theme & Console ────────────────────────────────────────
custom_theme = Theme({
    "info":      "cyan",
    "success":   "bold green",
    "warning":   "bold yellow",
    "danger":    "bold red",
    "highlight": "bold magenta",
    "muted":     "dim white",
    "accent":    "bold cyan",
})

console = Console(theme=custom_theme)

# ─── UI Constants ───────────────────────────────────────────
BANNER = r"""[bold cyan]
 ███████╗████████╗ ██████╗  ██████╗██╗  ██╗██████╗ ██╗   ██╗
 ██╔════╝╚══██╔══╝██╔═══██╗██╔════╝██║ ██╔╝██╔══██╗╚██╗ ██╔╝
 ███████╗   ██║   ██║   ██║██║     █████╔╝ ██████╔╝ ╚████╔╝
 ╚════██║   ██║   ██║   ██║██║     ██╔═██╗ ██╔═══╝   ╚██╔╝
 ███████║   ██║   ╚██████╔╝╚██████╗██║  ██╗██║        ██║
 ╚══════╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝        ╚═╝
[/bold cyan]"""

SUBTITLE = "[dim]─── Indian Stock Market Scanner & Research Assistant ───[/dim]"

NSE_POPULAR = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "BAJFINANCE.NS",
]

# ─── UI Helpers ─────────────────────────────────────────────

def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def show_banner():
    clear_screen()
    console.print(BANNER, justify="center")
    console.print(SUBTITLE, justify="center")
    if DEV_MODE:
        console.print("[bold yellow]  ⚡ DEV MODE — verbose logging enabled[/bold yellow]", justify="center")
    console.print()


def arrow_menu(title: str, options: list[str], *, show_banner_above: bool = True) -> int:
    """Displays an arrow-key navigable menu."""
    if show_banner_above:
        show_banner()
        console.print(f"  [accent]{title}[/accent]\n")

    menu = TerminalMenu(
        options,
        menu_cursor="  ❯ ",
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("fg_cyan", "bold", "underline"),
        cycle_cursor=True,
        clear_screen=False,
    )
    idx = menu.show()
    return idx if idx is not None else -1

# ─── Report Rendering ──────────────────────────────────────

def _fmt_number(val, prefix="", suffix=""):
    """Format a number with optional prefix/suffix, or return '—'."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{prefix}{val:,.2f}{suffix}"
    return f"{prefix}{val:,}{suffix}"


def render_report(results: list):
    """Renders the final styled report."""
    console.print()
    console.print(Rule("[bold cyan]📊  Analysis Report[/bold cyan]", style="cyan"))
    console.print()

    # Overview table
    summary = Table(
        box=box.SIMPLE_HEAVY,
        border_style="dim",
        padding=(0, 1),
        title="[bold]Overview[/bold]",
        title_style="cyan",
    )
    summary.add_column("Ticker", style="bold white", min_width=14)
    summary.add_column("Signal", justify="center", min_width=10)
    summary.add_column("Close", justify="right", style="white", min_width=10)
    summary.add_column("SMA50", justify="right", style="cyan", min_width=10)
    summary.add_column("SMA200", justify="right", style="cyan", min_width=10)
    summary.add_column("RSI", justify="center", min_width=7)
    summary.add_column("Vol Ratio", justify="center", min_width=10)

    for r in results:
        ticker = r["ticker"]
        if r["status"] == "failed":
            summary.add_row(ticker, "[red]ERROR[/red]", "—", "—", "—", "—", "—")
            continue
        ev = r["evaluation"]
        tech = ev.get("technicals", {})
        p_count = len(ev.get("pros", []))
        c_count = len(ev.get("cons", []))

        if p_count > c_count:
            signal = "[green]▲ Bullish[/green]"
        elif c_count > p_count:
            signal = "[red]▼ Bearish[/red]"
        else:
            signal = "[yellow]◆ Neutral[/yellow]"

        rsi = tech.get("rsi")
        if rsi is not None:
            rsi_str = f"[red]{rsi:.1f}[/red]" if rsi > 70 else f"[green]{rsi:.1f}[/green]" if rsi < 30 else f"{rsi:.1f}"
        else:
            rsi_str = "—"

        vol_ratio = tech.get("volume_ratio")
        vr_str = f"[green]{vol_ratio:.2f}x[/green]" if vol_ratio and vol_ratio > 1.5 else f"{vol_ratio:.2f}x" if vol_ratio else "—"

        summary.add_row(
            ticker, signal,
            _fmt_number(tech.get("close"), prefix="₹"),
            _fmt_number(tech.get("sma50"), prefix="₹"),
            _fmt_number(tech.get("sma200"), prefix="₹"),
            rsi_str, vr_str,
        )

    console.print(summary)
    console.print()

    # Detailed cards
    for result in results:
        ticker = result["ticker"]
        if result["status"] == "failed":
            console.print(Panel(f"[danger]✖ Error:[/danger] {result.get('error', 'Unknown')}", title=f"[bold red]{ticker}[/bold red]", border_style="red", box=box.HEAVY))
            continue

        ev = result["evaluation"]
        pros, cons, tech = ev.get("pros", []), ev.get("cons", []), ev.get("technicals", {})

        content = Text()
        content.append("  TECHNICAL DATA\n", style="bold cyan underline")
        
        data_lines = [
            ("Close Price", _fmt_number(tech.get("close"), prefix="₹"), "white"),
            ("SMA 50", _fmt_number(tech.get("sma50"), prefix="₹"), "white"),
            ("SMA 200", _fmt_number(tech.get("sma200"), prefix="₹"), "white"),
            ("RSI (14)", _fmt_number(tech.get("rsi")), "white"),
            ("Volume", _fmt_number(tech.get("volume")), "white"),
            ("Avg Vol (20d)", _fmt_number(tech.get("volume_avg_20")), "white"),
            ("Vol Ratio", _fmt_number(tech.get("volume_ratio"), suffix="x"), "white"),
        ]
        
        news_data = ev.get("news", {})
        if news_data:
            sentiment = news_data.get("sentiment", "Neutral")
            color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "yellow"
            data_lines.append(("News Sentiment", sentiment, color))

        for label, value, color in data_lines:
            content.append(f"   {label:<16}", style="dim")
            content.append(f"{value}\n", style=color)

        content.append("\n  PROS\n", style="bold green underline")
        if pros:
            for p in pros: content.append(f"   ✔  {p}\n", style="green")
        else: content.append("   —  None identified\n", style="dim")

        content.append("\n  CONS\n", style="bold red underline")
        if cons:
            for c in cons: content.append(f"   ✖  {c}\n", style="red")
        else: content.append("   —  None identified\n", style="dim")

        border, badge = ("green", "🟢") if len(pros) > len(cons) else ("red", "🔴") if len(cons) > len(pros) else ("yellow", "🟡")
        console.print(Panel(content, title=f"[bold]{badge}  {ticker}[/bold]", subtitle=f"[dim]{settings['period']} data[/dim]", border_style=border, box=box.ROUNDED, padding=(1, 3)))

# ─── Menu Handlers ──────────────────────────────────────────

def handle_scan():
    show_banner()
    console.print("  [accent]🔍  Scan Stocks[/accent]\n")
    raw = Prompt.ask("  [bold yellow]Tickers[/bold yellow] (space separated)")
    tickers = raw.strip().split()
    if tickers:
        results = run_scan(tickers, console, render_report)
        
        # Telegram Notification
        notifier = TelegramNotifier(settings.get("telegram_token"), settings.get("telegram_chat_id"))
        if notifier.is_configured():
            with console.status("[bold cyan]Sending to Telegram...[/]"):
                report_text = notifier.format_analysis_report(results)
                notifier.send_message(report_text)


def handle_quick_scan():
    show_banner()
    console.print("  [accent]📋  Quick Scan — Select Stocks[/accent]\n")
    menu = TerminalMenu(NSE_POPULAR, multi_select=True, show_multi_select_hint=True, menu_cursor="  ❯ ", menu_cursor_style=("fg_cyan", "bold"), title="  Use SPACE to select, ENTER to confirm:\n")
    selected = menu.show()
    if selected:
        tickers = [NSE_POPULAR[i] for i in selected]
        results = run_scan(tickers, console, render_report)
        
        # Telegram Notification
        notifier = TelegramNotifier(settings.get("telegram_token"), settings.get("telegram_chat_id"))
        if notifier.is_configured():
            with console.status("[bold cyan]Sending to Telegram...[/]"):
                report_text = notifier.format_analysis_report(results)
                notifier.send_message(report_text)


def handle_settings():
    while True:
        options = [
            f"Workers          ──  {settings['workers']}", 
            f"Data Period      ──  {settings['period']}", 
            f"Telegram Token   ──  {'SET' if settings['telegram_token'] else 'NOT SET'}",
            f"Telegram Chat ID ──  {settings['telegram_chat_id'] or 'NOT SET'}",
            f"Dev Mode         ──  {'ON' if DEV_MODE else 'OFF'}", 
            "← Back"
        ]
        idx = arrow_menu("⚙️  Settings", options)

        if idx == 0:
            w_idx = arrow_menu("Select workers", [f"  {i} worker{'s' if i > 1 else ''}" for i in range(1, 11)])
            if w_idx >= 0: settings["workers"] = w_idx + 1
        elif idx == 1:
            opts = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
            p_idx = arrow_menu("Select data period", [f"  {p}" for p in opts])
            if p_idx >= 0: settings["period"] = opts[p_idx]
        elif idx == 2:
            token = Prompt.ask("  Enter Telegram Bot Token", default=settings.get("telegram_token") or "")
            if token: settings["telegram_token"] = token
        elif idx == 3:
            chat_id = Prompt.ask("  Enter Telegram Chat ID", default=settings.get("telegram_chat_id") or "")
            if chat_id: settings["telegram_chat_id"] = chat_id
        elif idx == 4:
            toggle_dev_mode()
        else:
            break


def toggle_dev_mode():
    global DEV_MODE
    DEV_MODE = not DEV_MODE
    setup_logging(DEV_MODE)
    console.print(f"  [success]Dev Mode → {'ON' if DEV_MODE else 'OFF'}[/success]")


def handle_help():
    show_banner()
    help_text = (
        "[bold cyan]Stockpy[/bold cyan] scans Indian equities and provides pros/cons based on technicals and news sentiment.\n\n"
        "[bold]Pipeline:[/bold]\n  1. Fetch data  2. Indicators  3. News Sentiment  4. Report\n\n"
        "[bold]Keyboard:[/bold]\n  ↑ ↓  Navigate    Enter  Select    Space  Multi-select    Esc/q  Back"
    )
    console.print(Panel(help_text, title="[bold cyan]ℹ️  Help[/bold cyan]", border_style="cyan", padding=(1, 4)))
    input("\n  Press Enter to return...")


def interactive_mode():
    show_banner()
    items = ["🔍  Scan Stocks", "📋  Quick Scan", "⚙️   Settings", "ℹ️   Help", "🚪  Exit"]
    while True:
        idx = arrow_menu("What would you like to do?", items)
        if idx == 0: handle_scan()
        elif idx == 1: handle_quick_scan()
        elif idx == 2: handle_settings()
        elif idx == 3: handle_help()
        elif idx == 4 or idx == -1:
            console.print("\n  [muted]Goodbye! 👋[/muted]\n")
            sys.exit(0)
        console.print()
        input("  Press Enter to return to menu...")


def main():
    parser = argparse.ArgumentParser(description="Stockpy — Indian stock market scanner")
    parser.add_argument('tickers', nargs='*', help="Stock tickers")
    parser.add_argument('--workers', type=int, default=settings["workers"], help="Threads")
    parser.add_argument('--interactive', '-i', action='store_true', help="Interactive menu")
    parser.add_argument('--dev', action='store_true', help="Verbose logging")

    args = parser.parse_args()
    setup_logging(dev=args.dev)

    if args.interactive or not args.tickers:
        interactive_mode()
    else:
        settings["workers"] = args.workers
        show_banner()
        results = run_scan(args.tickers, console, render_report)
        
        # Telegram Notification
        notifier = TelegramNotifier(settings.get("telegram_token"), settings.get("telegram_chat_id"))
        if notifier.is_configured():
            with console.status("[bold cyan]Sending to Telegram...[/]"):
                report_text = notifier.format_analysis_report(results)
                notifier.send_message(report_text)


if __name__ == "__main__":
    main()
