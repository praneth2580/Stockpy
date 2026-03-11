import time
import logging
import concurrent.futures
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

from scanner.data_fetcher import fetch_stock_data
from scanner.indicators import calculate_indicators
from scanner.news import fetch_news_sentiment
from scanner.evaluator import evaluate_stock
from scanner.screener import pick_top_candidates

logger = logging.getLogger(__name__)

# ─── Scanning State ─────────────────────────────────────────
settings = {
    "workers": 3,
    "period": "1y",
    "telegram_token": "5710041825:AAEulSFLC4TBEidHKYcmsmBht-u7_AJUbj4",
    "telegram_chat_id": "1175853690",
}

def process_ticker(ticker: str, period: str) -> dict:
    """Full analysis pipeline for a single ticker."""
    logger.debug(f"[pipeline] Start → {ticker}")
    try:
        logger.debug(f"[fetch]     Downloading {period} of data for {ticker}")
        df = fetch_stock_data(ticker, period=period)

        logger.debug(f"[indicators] Computing SMA / RSI / Volume for {ticker}")
        df = calculate_indicators(df)

        logger.debug(f"[news]      Fetching sentiment for {ticker}")
        news = fetch_news_sentiment(ticker)

        logger.debug(f"[evaluate]  Generating pros/cons for {ticker}")
        evaluation = evaluate_stock(ticker, df, news)

        logger.debug(f"[pipeline] Done ✔ {ticker}")
        return {"ticker": ticker, "evaluation": evaluation, "status": "success"}
    except Exception as e:
        logger.error(f"[pipeline] Failed ✖ {ticker}: {e}")
        return {"ticker": ticker, "status": "failed", "error": str(e)}

def run_scan(tickers: list, console, render_report_callback):
    """Runs the parallel scan and renders results."""
    workers = settings["workers"]
    period = settings["period"]

    console.print()
    console.print(Panel(
        f"[info]Tickers:[/info]  {', '.join(tickers)}\n"
        f"[info]Workers:[/info]  {workers}    [info]Period:[/info]  {period}",
        title="[bold cyan]⏳  Scanning[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 2),
    ))
    console.print()

    results = []
    start = time.time()

    with Progress(
        SpinnerColumn("dots", style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style="dim", complete_style="cyan", finished_style="green"),
        TextColumn("[bold]{task.completed}/{task.total}[/bold]"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(tickers))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(process_ticker, t, period): t for t in tickers}
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                results.append(result)
                status_icon = "[green]✔[/green]" if result["status"] == "success" else "[red]✖[/red]"
                progress.update(
                    task, advance=1,
                    description=f"{status_icon} {result['ticker']}"
                )

    elapsed = time.time() - start
    console.print(f"[success]✔ Done[/success] in [bold]{elapsed:.2f}s[/bold]")

    render_report_callback(results)
    return results


def scan_and_rank(tickers: list, console, top_n: int = 10):
    """
    Runs the standard scan pipeline and then ranks the results,
    returning the top N candidates along with the full result set.
    """

    def _noop_render(_results):
        # We don't want to render the full report here; caller decides.
        pass

    results = run_scan(tickers, console, _noop_render)
    top = pick_top_candidates(results, top_n=top_n)
    return top, results
