# AGENTS.md

## Critical Information

- **Port 8000**: Running a Docker container unrelated to this project. Ignore it.
- **Config file**: `config/config.ini` (not config.ini.example)
- **Dashboard port**: 8100
- **Dashboard URL**: `http://localhost:8100/`
- **Python venv**: `.venv` in project root - ALWAYS activate with `source .venv/bin/activate` before running Python commands

## Project overview
Autonomous asyncio-first crypto trading bot that uses LLMs + RAG to make BUY/SELL/HOLD decisions. Integrates market data, news, and chart analysis via ChromaDB vector memory and multi-provider AI support (Google Gemini, OpenRouter, BlockRun.AI, LM Studio).

## Setup & commands
- Install dependencies: `.venv/bin/pip install -r requirements.txt`
- Install dev dependencies: `.venv/bin/pip install -r requirements-dev.txt`
- Run: `.venv/bin/python start.py`
- Test: `.venv/bin/pytest tests/`
- Lint (ruff): `.venv/bin/ruff check .`
- Lint (pylint): `.venv/bin/pylint src/`

## Code style
- **IMPORTANT**: All code and comments MUST be in English, even if conversations with the agent are in French
- Max line length: 120 characters
- Indent: 4 spaces
- Pylint fail-under: 8.0
- Pylint disables: too-few-public-methods, broad-exception-caught
- Use type hints where possible
- docstrings for modules, classes, and functions

## Testing instructions
- Tests use pytest with pytest-asyncio and pytest-timeout
- Mock config via `tests/conftest.py` (patches `src.config.loader`)
- Run all tests: `.venv/bin/pytest tests/`
- Run specific test file: `.venv/bin/pytest tests/test_vector_memory.py`

## Dashboard Testing
- Use Playwright for browser testing of the dashboard
- **IMPORTANT**: Always activate the project venv first: `source .venv/bin/activate`
- Install playwright: `pip install playwright && playwright install chromium`
- Run dashboard on port 8100 before testing

## Dashboard Tabs to Test
1. Overview - Live Dashboard with KPIs
2. Brain Activity - Neural Network State + Visual Cortex
3. Last Prompt - Trading Context Sent to LLM
4. Last Response - LLM Analysis Output
5. Statistics - Trading Statistics
6. Latest News - Crypto News
7. Market Data - Active Position
8. Memory Bank - Vector Experience Database

## Architecture notes
- **Entry point**: `start.py` - Composition Root with Dependency Injection pattern
- **Core bot**: `src/app.py` - `CryptoTradingBot` class manages polling and lifecycle
- **Key subsystems**:
  - `src/analyzer/` - Technical analysis, pattern recognition, prompt building
  - `src/rag/` - RAG engine for news and market data context
  - `src/trading/` - Vector memory, positions, brain service, statistics
  - `src/dashboard/` - FastAPI + WebSocket real-time dashboard
  - `src/managers/` - Risk, persistence, model routing
  - `src/platforms/` - Exchange (CCXT), API clients (CoinGecko, CryptoCompare, DefiLlama, Alternative.me)
  - `src/factories/` - DI dependency construction
  - `src/indicators/` - NumPy/Numba technical indicators

## Key conventions
- All external API calls are async (aiohttp)
- ChromaDB vector store per symbol/timeframe in `data/trading/brain_{symbol}_{timeframe}/`
- Config loaded via `src.config.loader` singleton
- LiteLLM handles multi-provider abstraction (googleai, openrouter, blockrun, local)
- Discord notifier with file handler persistence for tracked messages
- Keyboard hotkeys at runtime: 'a' = force analysis, 'd' = toggle dashboard, 'h' = help, 'q' = quit
