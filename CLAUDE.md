# CLAUDE.md

This repository contains a Python-based research tool that scans Indian stock markets (NSE/BSE) and surfaces potentially interesting stocks for manual investment decisions.

The system does not attempt to fully automate trading or guarantee predictions. Its purpose is to filter large numbers of stocks and present useful signals, pros/cons, and supporting data so a human can make the final decision.

## Core Objective

The tool should:

1. Fetch historical stock data for Indian equities.
2. Compute technical indicators and momentum signals.
3. Identify stocks showing potentially strong trends or breakout conditions.
4. Gather recent news related to selected stocks.
5. Analyze sentiment and summarize signals.
6. Output clear **pros and cons** for each candidate stock.

The system acts as a **market scanner and research assistant**, not a trading bot.

## Data Sources

Primary data sources may include:

- Yahoo Finance (via `yfinance`)
- NSE/BSE public endpoints if needed
- News APIs for company-related headlines

Historical price data is used for indicator calculations and trend detection.

## Analysis Philosophy

The analysis pipeline should prioritize interpretable signals over complex black-box models.

Important signals include:

- Moving average trends (MA50 / MA200)
- RSI and momentum indicators
- Volume spikes
- Breakout patterns
- Volatility behavior

Signals should be combined into simple heuristic evaluations rather than opaque predictions.

Outputs should clearly explain **why a stock was selected**.

## News and Sentiment

For stocks identified by the technical scanner:

- fetch recent news headlines
- run lightweight sentiment analysis
- summarize whether sentiment appears positive, negative, or mixed

The system should surface useful context rather than raw articles.

## Output Format

The program should present results in a structured format such as:

Stock Name

Pros
- strong upward trend
- increased trading volume
- positive news sentiment

Cons
- overbought RSI
- high volatility
- weak recent earnings

The goal is to give the user enough context to continue manual research.

## Development Guidelines

When modifying or adding analysis:

- Prefer deterministic rules over complex ML models.
- Keep computations explainable.
- Avoid introducing heavy dependencies unless clearly beneficial.
- Maintain readable signal explanations.

The tool should remain lightweight and easy to run locally.

## Scope

The project focuses on:

- scanning many stocks quickly
- identifying interesting candidates
- summarizing signals

It does **not** attempt to:

- automate trades
- guarantee investment outcomes
- produce definitive price predictions

Human judgment remains the final step.