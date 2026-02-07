# Kura-Next — Voice Agent Framework

## Project Overview

Enterprise voice agent framework using Google Gemini 2.5 Flash Native Audio. Python 3.11+, async/event-driven, hexagonal architecture (Ports & Adapters).

## Common Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Lint
ruff check .

# Auto-fix lint issues
ruff check . --fix

# Format
ruff format .

# Type check
mypy .

# Run server
python -m src.main

# Run CLI mode (dev/testing)
python -m src.main --cli
```

## Code Style

- Python 3.11+ — use modern syntax (match/case, `type` aliases, `X | Y` unions)
- 100 character line length (configured in ruff)
- 100% type-hinted — `mypy --strict` must pass
- Use `async`/`await` throughout; never block the event loop
- Imports sorted by ruff (isort rules enabled: `I`)
- Ruff lint rules: `E`, `F`, `I`, `N`, `W`, `UP`

## Architecture

```
src/
├── framework/core/     # Domain: Orchestrator, Agent, Context, Signals
├── infrastructure/     # Adapters: Gemini LLM, Database (SQLAlchemy)
├── client/agents/      # App agents: Router, Identity, TaskManager
├── server/             # FastAPI app, Twilio handler, config
└── main.py             # Entry point
```

- **Hexagonal architecture** — domain core has zero external dependencies
- **Orchestrator** is the central event loop coordinating agents and signals
- **Signals/Responses** for inter-component communication (TextSignal, AudioSignal, ToolCall)
- **Context hierarchy**: GlobalContext -> SessionContext -> UserContext
- Agents extend `AbstractAgent` in `framework/core/agent.py`
- Database uses Repository pattern (`infrastructure/database/repository.py`)

## Testing

- Framework: `pytest` with `pytest-asyncio` (asyncio_mode = "auto")
- Tests live in `tests/` — fixtures in `conftest.py`
- Use in-memory SQLite for DB tests (see `conftest.py`)
- Always run `pytest tests/ -v` before committing

## Environment

- Copy `.env.example` to `.env` for local config
- Required: `GOOGLE_API_KEY`
- Database: `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_PORT`, `DB_NAME`
- Optional Twilio: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`
- Never commit `.env` files or secrets

## Git Workflow

- Branch naming: `feature/<name>`, `bugfix/<name>`, `docs/<name>`
- Run `pytest`, `ruff check .`, and `mypy .` before committing
- Write clear commit messages describing the "why"

## Agent Teams

This project enables Claude Code experimental agent teams for parallel development workflows. See `.claude/settings.json` for team configuration.

To enable agent teams, set the environment variable:
```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

Or use the flag in `.claude/settings.json` under `env`.
