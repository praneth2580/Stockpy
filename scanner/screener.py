from typing import Dict, List, Tuple


def score_stock(result: Dict) -> float:
    """
    Converts a single analysis result into a numeric score.
    Higher score = more attractive candidate based on simple,
    interpretable heuristics (trend, momentum, volume, news).
    """
    if result.get("status") != "success":
        return float("-inf")

    ev = result.get("evaluation", {}) or {}
    pros: List[str] = ev.get("pros", []) or []
    cons: List[str] = ev.get("cons", []) or []
    tech: Dict = ev.get("technicals", {}) or {}
    news: Dict = ev.get("news", {}) or {}

    score = 0.0

    # 1) Base on pros / cons count
    score += len(pros) * 1.0
    score -= len(cons) * 1.0

    # 2) Trend / RSI / volume cues via text signals
    for p in pros:
        if "long-term uptrend" in p:
            score += 2.0
        if "potentially oversold" in p:
            score += 1.5
        if "RSI is neutral" in p:
            score += 0.5
        if "Recent trading volume is significantly above average" in p:
            score += 1.0
        if "Positive news sentiment" in p:
            score += 1.5

    for c in cons:
        if "long-term downtrend" in c:
            score -= 2.0
        if "overbought" in c:
            score -= 1.5
        if "weak momentum" in c:
            score -= 1.0
        if "Negative news sentiment" in c:
            score -= 1.5

    # 3) Raw technical numbers for light biasing
    rsi = tech.get("rsi")
    if rsi is not None:
        if 45 <= rsi <= 65:
            score += 1.0
        elif rsi > 70:
            score -= 1.0

    vol_ratio = tech.get("volume_ratio")
    if vol_ratio is not None:
        if vol_ratio >= 1.5:
            score += 1.0
        elif vol_ratio <= 0.7:
            score -= 0.5

    sentiment = news.get("sentiment", "Unknown")
    if sentiment == "Positive":
        score += 1.0
    elif sentiment == "Negative":
        score -= 1.0

    return score


def pick_top_candidates(results: List[Dict], top_n: int = 10) -> List[Tuple[float, Dict]]:
    """
    Scores all successful results and returns (score, result) tuples
    for the top N candidates.
    """
    scored: List[Tuple[float, Dict]] = []

    for r in results:
        s = score_stock(r)
        if s == float("-inf"):
            continue
        scored.append((s, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]

