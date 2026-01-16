# Kura-Next: Enterprise Voice Agent Framework

A production-grade, object-oriented framework for orchestrating stateful, multi-modal AI agents using Google Gemini 2.5 Flash Native Audio. This system is designed to demonstrate advanced software engineering patterns including hexagonal architecture, asynchronous concurrency, and strict type safety.

## Technical Overview

Kura-Next provides a robust engine for building voice-first applications. It decouples the core agent logic from the input/output transport, allowing the same agent definitions to operate over CLI, WebSockets, or Telephony (Twilio) interfaces.

### Key Features

*   **Native Audio Integration**: Utilizes Gemini 2.5 Flash's native audio modalities for sub-500ms latency interactions.
*   **Event-Driven Architecture**: Built on Python's `asyncio` event loop for high-performance non-blocking I/O.
*   **Hexagonal Design**: Strict separation of concerns using Ports and Adapters patterns.
*   **Stateful Context Management**: Hierarchical state tracking (Global, Session, User) with persistence.
*   **VUI Optimization**: Implements latency masking (filler audio), warm agent handoffs, and background intervention detection.
*   **Type Safety**: 100% type-hinted codebase verified with `mypy`.

## System Architecture

The system is composed of the following layers:

1.  **Orchestrator**: The central kernel that manages the event loop and signal routing.
2.  **Agents**: Autonomous units encapsulating business logic and tools.
3.  **Infrastructure**: Adapters for external services (LLM, Database, Twilio).
4.  **Interfaces**: Abstractions for input/output streams.

## Prerequisites

*   Python 3.11 or higher
*   Google Cloud Project with Gemini API access
*   Twilio Account (for telephony integration)
*   PostgreSQL (Production) or SQLite (Development)

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-repo/agent-framework.git
    cd AgentFramework
    ```

2.  **Initialize the virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Configuration is managed via environment variables. Create a `.env` file in the project root:

```ini
# AI Configuration
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-2.0-flash-exp

# Database Configuration
# For SQLite (Development):
DATABASE_URL=sqlite+aiosqlite:///./data/kura.db
# For PostgreSQL (Production):
# DB_HOST=aws-0-us-east-1.pooler.supabase.com
# DB_USER=postgres
# DB_PASSWORD=your_password
# DB_PORT=5432
# DB_NAME=postgres

# Telephony Configuration (Optional)
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=+15550000000

# Server Settings
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## Usage

### CLI Mode (Testing)
For rapid prototyping and logic verification without telephony overhead:

```bash
python -m src.main --cli
```

### Server Mode (Production)
Starts the FastAPI server to handle WebSocket connections and Twilio webhooks:

```bash
python -m src.main
```

The server exposes the following endpoints:
- `POST /twilio/voice`: Webhook for incoming Twilio calls.
- `WS /ws/audio`: General-purpose WebSocket for web clients.

## Project Structure

```text
src/
├── framework/          # Core engine (Orchestrator, Context, Signals)
│   ├── core/
│   └── interfaces/
├── infrastructure/     # Adapters (Gemini, SQLAlchemy)
├── client/             # Reference Implementation (TaskMaster)
│   ├── agents/         # Domain-specific agents
│   └── tools/          # Tool definitions
├── server/             # FastAPI application
└── main.py             # Entry point
```

## Development Standards

*   **Code Style**: Adheres to PEP 8. formatted via `ruff`.
*   **Type Checking**: Strict typing enforced via `mypy`.
*   **Testing**: Unit and integration tests using `pytest`.

To run the test suite:
```bash
pytest tests/ -v
```
