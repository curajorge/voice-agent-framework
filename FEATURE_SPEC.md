# Feature Specification: AI Voice Agents That Reason, Plan, and Execute

> Derived from [DataGOL's "Building AI Voice Agents: To Reason, Plan and Execute"](https://dev.to/jyotish_bora_0ce3be5a374b/how-datagol-gave-its-ai-agents-a-voice-building-with-pipecat-and-a-custom-langgraph-frame-m6a)
> and supporting research on agentic AI architectures.
>
> **Date**: 2026-03-07
> **Status**: Proposed
> **Codebase**: Kura-Next Enterprise Voice Agent Framework

---

## Executive Summary

This specification defines 5 feature pillars to evolve the Kura-Next framework from
a capable voice agent platform into a fully agentic system that can **reason** through
problems, **plan** multi-step solutions, and **execute** them with production-grade
voice UX. Each pillar was investigated against the current codebase to identify
concrete gaps and actionable recommendations.

---

## Pillar 1: Reasoning Engine (ReAct Loop)

### Current State
- Tool execution in `orchestrator.py` (`_execute_tools`, line ~438) is **single-pass**: tools execute, results go to a callback, and the turn ends.
- `GeminiLiveSession.send_tool_response` exists (`gemini_audio.py`, line 252) but is **never called** — tool results are invisible to the LLM.
- No explicit chain-of-thought capture. LLM reasoning is opaque with no structured `thought` field on `Response` or `ToolCall`.
- Tool failures produce a hardcoded apology string. The LLM never reasons about recovery.
- The LLM cannot examine a tool result and decide to call another tool — multi-step operations require separate user turns.

### Features to Build

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| R1 | **ReAct Execution Loop** | P0 | Replace `_execute_tools` with an iterative Thought→Action→Observation loop. Execute tools, feed results back to the LLM, check if it wants more tools or a final response, repeat up to `max_iterations`. |
| R2 | **Tool Result Feedback** | P0 | Add `process_tool_results()` to `AbstractAgent` that re-invokes the LLM with tool outputs appended to conversation history. Wire up `GeminiLiveSession.send_tool_response` in `BaseClientAgent`. |
| R3 | **Chain-of-Thought Capture** | P1 | Add a `thought` field to `Response` and `ToolCall` in `signals.py` to capture and log reasoning at each step. |
| R4 | **Tool-Role in History** | P1 | Add a `"tool"` role to `ConversationTurn` in `context.py` so tool results persist in conversation history for future context. |
| R5 | **Error-as-Observation** | P1 | Route tool errors back into the ReAct loop as observations instead of emitting hardcoded apologies, letting the LLM reason about retry or graceful degradation. |
| R6 | **Progressive Filler for Multi-Iteration** | P2 | Add escalating filler phrases for multi-iteration loops (e.g., "Let me check...", "Still working on that...") to mask compounding latency. |
| R7 | **Loop Safety Guard** | P1 | Add configurable `max_react_iterations` (default ~5) to `Orchestrator.__init__` to prevent infinite cycling. |

### Key Files to Modify
- `src/framework/core/orchestrator.py` — ReAct loop, tool execution
- `src/framework/core/agent.py` — `process_tool_results()` abstract method
- `src/framework/core/signals.py` — `thought` field on Response/ToolCall
- `src/framework/core/context.py` — tool role in ConversationTurn
- `src/client/agents/base.py` — wire `send_tool_response`
- `src/infrastructure/llm/gemini_audio.py` — activate existing tool response path

---

## Pillar 2: Planning Engine (Plan-and-Execute)

### Current State
- Requests go straight from LLM to execution with **no decomposition step**. Complex requests like "create three tasks and mark the old ones done" rely entirely on a single LLM pass.
- No `Plan`, `Step`, or `PlanStatus` data structures exist anywhere in the codebase.
- Tool execution in `_execute_tools()` (line ~457) runs tools in a flat for-loop — no parallel execution, no dependency awareness.
- Tool failures produce an apology. No mechanism to revise remaining steps or retry alternatives.
- The router maps each request to exactly one agent — no concept of a request needing multi-agent coordination.

### Features to Build

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| P1 | **Plan & Step Data Models** | P0 | Add `PlanStep` and `ExecutionPlan` models in `src/framework/core/` with ordered steps, dependencies, status tracking, and results. |
| P2 | **Planning Agent** | P0 | Build a new agent (or extend Router) that decomposes complex requests into an `ExecutionPlan` before execution begins. |
| P3 | **Complexity Detection** | P1 | Add complexity scoring to the Router so simple requests skip planning (preserving latency) while complex ones trigger decomposition. |
| P4 | **Plan Executor** | P0 | Add `_execute_plan()` to the Orchestrator that iterates plan steps, switches agents per step, and feeds each step's output into the next. |
| P5 | **Parallel Step Execution** | P1 | Use `asyncio.gather()` for independent plan steps, replacing the sequential for-loop. |
| P6 | **Replanning on Failure** | P1 | On step failure, call back to the Planning Agent with partial results and the error to produce a revised plan. |
| P7 | **Plan Persistence** | P2 | Store the active plan in `SessionContext` so it persists across turns and users can modify or extend plans mid-conversation. |
| P8 | **Planning System Prompt** | P0 | Create `resources/prompts/planner/v1_system.txt` instructing the LLM to output structured, ordered plans with dependency annotations. |
| P9 | **Speculative Tool Calling** | P2 | Pre-fetch likely-needed data optimistically before the LLM explicitly requests it, reducing perceived latency. |

### Key Files to Modify/Create
- `src/framework/core/plan.py` — new: Plan/Step models
- `src/framework/core/orchestrator.py` — plan executor
- `src/client/agents/planner.py` — new: PlanningAgent
- `src/client/agents/router.py` — complexity detection
- `src/framework/core/context.py` — plan storage in SessionContext
- `resources/prompts/planner/v1_system.txt` — new: planning prompt

---

## Pillar 3: Memory Architecture

### Current State
- `SessionContext.conversation_history` is an **in-memory list** that dies when the session ends. Returning callers start from zero context.
- No database tables store conversation turns or session records. Only `User` and `Task` models exist.
- No episodic, semantic, or procedural memory.
- No vector/embedding infrastructure — no similarity search capability.
- `twilio_handler.py` looks up user and task count at session start but loads **zero prior conversational context**.
- The Scratchpad is useful for intra-session slot filling but is **not persisted** across sessions.
- Conversation history grows **unbounded** during long sessions — no summarization or sliding window.

### Features to Build

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| M1 | **Conversation Persistence Models** | P0 | Add `ConversationSession` and `ConversationMessage` database tables in `models.py` with corresponding repositories. |
| M2 | **Per-Round Memory Writes** | P0 | In the orchestrator's processing loop, asynchronously persist noteworthy turns without adding latency to the voice path. |
| M3 | **Session-Start Pre-Loading** | P0 | After user lookup in `twilio_handler.py`, fetch recent memories and inject them into the system prompt alongside `HandoffData`. |
| M4 | **Memory Service** | P1 | Build a central `MemoryService` handling per-turn writes, end-of-session summarization, importance scoring, and memory consolidation/decay. |
| M5 | **Vector Search Infrastructure** | P1 | Integrate pgvector (PostgreSQL) or sqlite-vec (dev) for embedding storage; wrap an embedding API for vector generation. |
| M6 | **Episodic Memory** | P1 | Store session summaries with timestamps/topics; retrieve by recency and vector similarity for case-based reasoning. |
| M7 | **Semantic Memory** | P2 | Structured factual knowledge store with retrieval — user facts, preferences, domain knowledge. |
| M8 | **Procedural Memory** | P2 | Accumulate learned user preferences (default priority, verbosity) and inject them into agent prompts dynamically. |
| M9 | **Short-Term Memory Management** | P1 | Add rolling summarization or a sliding window to `conversation_history` to prevent unbounded growth in long sessions. |

### Key Files to Modify/Create
- `src/infrastructure/database/models.py` — new tables
- `src/infrastructure/database/repository.py` — new repositories
- `src/infrastructure/memory/` — new: MemoryService, vector store adapter
- `src/framework/core/orchestrator.py` — per-round memory writes
- `src/framework/core/context.py` — memory integration
- `src/server/twilio_handler.py` — session-start pre-loading

---

## Pillar 4: Voice Pipeline & Latency Masking

### Current State
- Filler audio on the Twilio path is **text-only** — `_send_filler_impl()` only logs; no actual audio plays during tool calls, causing **dead air on phone calls**.
- `_send_to_twilio()` calls `_execute_tool()` directly without any latency masking — the orchestrator's filler logic is bypassed entirely.
- Agent switching has a bare `asyncio.sleep(0.3)` plus Gemini session creation (~1-2s total) with **zero bridge audio**.
- No local VAD — turn boundary detection is fully delegated to Gemini, adding round-trip latency.
- No barge-in detection — users cannot interrupt agent speech.
- TTFA metric resets on every Twilio media chunk, not on actual end-of-utterance.
- `record_filler_played()` is defined in `metrics.py` but **never called**.
- All transport is WebSocket-based — no WebRTC support.
- Uses deprecated `audioop` module (breaks on Python 3.13+).

### Features to Build

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| V1 | **Pre-Rendered Filler Audio** | P0 | Pre-render filler phrases as mulaw audio at startup; stream them immediately when `send_filler()` is called on the Twilio path. |
| V2 | **Filler During Tool Execution** | P0 | Play filler concurrently with `_execute_tool()` in `_send_to_twilio()`, cancel on result. |
| V3 | **Bridge Audio for Agent Switching** | P0 | Play bridge audio during agent switching instead of silent `asyncio.sleep(0.3)`. |
| V4 | **Local VAD** | P1 | Add local VAD (silero-vad or webrtcvad) in `_receive_from_twilio()` for accurate endpointing and correct TTFA measurement. |
| V5 | **Barge-In Detection** | P1 | Use VAD to detect user speech during playback, send Twilio "clear" event, and forward new utterance. |
| V6 | **Metrics Instrumentation** | P1 | Wire up `record_filler_played()` and other metrics; export via OpenTelemetry or Prometheus. |
| V7 | **Replace `audioop`** | P1 | Replace deprecated `audioop` with `soundfile`, `pydub`, or numpy-based resampling for Python 3.13+ compatibility. |
| V8 | **Parallel Tool Execution** | P1 | Use `asyncio.gather()` in `_execute_tools()` for independent tool calls. |
| V9 | **WebRTC Transport** | P2 | Add WebRTC transport option for browser-based clients targeting sub-500ms latency. |
| V10 | **Audio Jitter Buffer** | P2 | Add smoothing/queuing for Gemini audio chunks before forwarding to Twilio. |

### Key Files to Modify
- `src/server/twilio_handler.py` — filler audio, bridge audio, VAD, barge-in
- `src/framework/core/orchestrator.py` — parallel tool execution
- `src/framework/core/metrics.py` — instrumentation
- `src/infrastructure/llm/gemini_audio.py` — audio processing
- `src/framework/core/io_handler.py` — WebRTC transport

---

## Pillar 5: Multi-Agent Collaboration & Orchestration

### Current State
- Only one agent is active at a time — tool execution is sequential.
- No supervisor/manager pattern — no agent can delegate sub-tasks and aggregate results.
- No agent-to-agent communication beyond one-way warm handoffs that fully deactivate the previous agent.
- Tools are siloed per-agent with no cross-agent discovery or invocation.
- **Zero MCP support** — the tool system is entirely Gemini-specific.
- `HandoffData` is fire-and-forget — no mechanism for an agent to return results to the caller.
- Routing is flat with only 2 hardcoded targets (`identity` and `task_manager`) using keyword matching.
- No dynamic agent registration after orchestrator startup.

### Features to Build

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| A1 | **Centralized Tool Registry** | P0 | Build a `ToolRegistry` so any agent can discover and invoke tools from other agents, eliminating silos. |
| A2 | **Parallel Agent Execution** | P1 | Use `asyncio.gather()` for independent agent tasks (e.g., run task + calendar agents in parallel, then summarize). |
| A3 | **Supervisor Agent Pattern** | P1 | Build a `SupervisorAgent` that spawns sub-tasks to specialized agents, monitors progress, and combines outputs. |
| A4 | **Bidirectional Handoffs** | P1 | Extend `HandoffData` with `return_to` and `result` fields so agents can delegate and receive results back. |
| A5 | **Workflow Engine** | P2 | Create a workflow abstraction supporting sequential, parallel, and conditional agent execution patterns. |
| A6 | **MCP Adapter Layer** | P1 | Expose existing `Tool` definitions as MCP-compliant tools and consume external MCP tool servers. |
| A7 | **Fallback Chains** | P1 | Route agent failures to alternate agents instead of producing generic error messages. |
| A8 | **Capability-Based Routing** | P1 | Replace hardcoded keyword lists with capability/confidence-scored routing for extensible agent selection. |
| A9 | **Dynamic Agent Registration** | P2 | Allow agents to be registered/deregistered at runtime without restarting the orchestrator. |

### Key Files to Modify/Create
- `src/framework/core/tool_registry.py` — new: centralized tool registry
- `src/framework/core/orchestrator.py` — parallel execution, fallback chains
- `src/framework/core/context.py` — bidirectional HandoffData
- `src/client/agents/supervisor.py` — new: SupervisorAgent
- `src/client/agents/router.py` — capability-based routing
- `src/infrastructure/mcp/` — new: MCP adapter layer

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)
> Get the core reasoning and voice fixes in place.

- **R1** ReAct Execution Loop
- **R2** Tool Result Feedback
- **V1** Pre-Rendered Filler Audio
- **V2** Filler During Tool Execution
- **V3** Bridge Audio for Agent Switching
- **M1** Conversation Persistence Models
- **M2** Per-Round Memory Writes
- **A1** Centralized Tool Registry

### Phase 2: Intelligence (Weeks 4-6)
> Add planning, memory retrieval, and smarter routing.

- **P1** Plan & Step Data Models
- **P2** Planning Agent
- **P4** Plan Executor
- **P8** Planning System Prompt
- **M3** Session-Start Pre-Loading
- **M4** Memory Service
- **M5** Vector Search Infrastructure
- **R3** Chain-of-Thought Capture
- **R5** Error-as-Observation
- **A8** Capability-Based Routing

### Phase 3: Advanced (Weeks 7-10)
> Multi-agent workflows, advanced memory, production hardening.

- **P3** Complexity Detection
- **P5** Parallel Step Execution
- **P6** Replanning on Failure
- **M6** Episodic Memory
- **M9** Short-Term Memory Management
- **V4** Local VAD
- **V5** Barge-In Detection
- **V7** Replace audioop
- **A3** Supervisor Agent Pattern
- **A4** Bidirectional Handoffs
- **A6** MCP Adapter Layer
- **A7** Fallback Chains

### Phase 4: Polish (Weeks 11-12)
> Optimization, advanced features, and production readiness.

- **P7** Plan Persistence
- **P9** Speculative Tool Calling
- **M7** Semantic Memory
- **M8** Procedural Memory
- **V6** Metrics Instrumentation
- **V8** Parallel Tool Execution
- **V9** WebRTC Transport
- **V10** Audio Jitter Buffer
- **A2** Parallel Agent Execution
- **A5** Workflow Engine
- **A9** Dynamic Agent Registration
- **R6** Progressive Filler for Multi-Iteration
- **R7** Loop Safety Guard

---

## Feature Count Summary

| Pillar | Features | P0 | P1 | P2 |
|--------|----------|----|----|-----|
| Reasoning (ReAct) | 7 | 2 | 4 | 1 |
| Planning (Plan-and-Execute) | 9 | 3 | 3 | 3 |
| Memory Architecture | 9 | 3 | 4 | 2 |
| Voice Pipeline & Latency | 10 | 3 | 4 | 3 |
| Multi-Agent Orchestration | 9 | 1 | 5 | 3 |
| **Total** | **44** | **12** | **20** | **12** |

---

## References

- [DataGOL: How We Gave Our AI Agents a Voice](https://dev.to/jyotish_bora_0ce3be5a374b/how-datagol-gave-its-ai-agents-a-voice-building-with-pipecat-and-a-custom-langgraph-frame-m6a)
- [AI Agent Architectures: Reasoning, Planning, and Tool Calling (arXiv)](https://arxiv.org/html/2404.11584v1)
- [ReAct Agent Architecture (IBM)](https://www.ibm.com/think/topics/react-agent)
- [Memory for Voice Agents (Mem0)](https://mem0.ai/blog/ai-memory-for-voice-agents)
- [Speculative Tool Calling for Voice (GetStream)](https://getstream.io/blog/speculative-tool-calling-voice/)
- [Google Cloud: Agentic AI Design Patterns](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)
- [Microsoft CORPGEN: Hierarchical Planning](https://www.marktechpost.com/2026/02/26/microsoft-research-introduces-corpgen-to-manage-multi-horizon-tasks-for-autonomous-ai-agents-using-hierarchical-planning-and-memory/)
- [AI Agent Systems: Architectures, Applications, and Evaluation (arXiv)](https://arxiv.org/html/2601.01743v1)
