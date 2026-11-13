# Lapua

Lapua is a Go SDK for building tool-using agents. It provides a streaming execution loop, host-controlled tool permissions, and conversation state behind `agentcore.Session`. It supports the Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses APIs.

Lapua is a fork of [Norma](https://github.com/Autumn-27/Norma), originally created by [Autumn-27](https://github.com/Autumn-27). Original copyright © 2026 Autumn-27. See [LICENSE](LICENSE) for the Apache 2.0 license and copyright notice.

Build with Go 1.26.3 or later. Import SDK packages under `github.com/infinmalum/Lapua`, for example `github.com/infinmalum/Lapua/agentcore`.

## Packages

| Package | Responsibility |
|---|---|
| `llm` | Vendor-neutral message model (`Message`/`ContentBlock`/`StreamEvent`) + `Provider` interface. `NewProvider(Config{Format})` selects Anthropic Messages, OpenAI Chat Completions, or OpenAI Responses. Each adapter normalizes streaming and tool calls into the same event type. |
| `tool` | `CoreTool` contract, fail-closed `Build` factory, `Registry`, a lightweight JSON-Schema validator, and the core tools: **Read, Write, Edit, MultiEdit, LS, Glob, Grep, Bash** (default set) plus opt-in **WebFetch** / **WebSearch** (network, permission-gated), **TodoWrite** (stateful planning), **AskUserQuestion** (structured mid-task questions), background-task tools (**TaskOutput/TaskStop/TaskList/Monitor**), and **deferred tools** — schemas withheld from the model to save context, discovered on demand via `SearchExtraTools` and invoked through `ExecuteExtraTool`. |
| `permission` | Evaluates allow and deny rules, tool policy, permission mode, and host approval. Deny rules take priority, including for deferred targets and nested agent calls. |
| `harness` | The loop: `Query` streams `Event`s (`iter.Seq2`), schedules tools that declare concurrency safety in parallel and other tools exclusively, pairs every `tool_use` with a `tool_result` (synthetic on deny/abort), enforces termination reasons, and threads cancellation. `QueryDeps` injects side effects for testing. |
| `compaction` | Context-window management: Snip / MicroCompact / Context Collapse / AutoCompact, a circuit breaker, reactive recovery from prompt-too-long, and non-destructive `compact_boundary` summaries. Invoked skills are re-injected verbatim after each boundary so their instructions survive summarization. |
| `noa` / `noaadapter` | Optional model-driven compression. The adapter supplies a projected request view, the Compress tool, and durable archives. |
| `hook` | Eight lifecycle hooks (PreToolUse, PostToolUse, Stop, …) as Go callbacks that can block, rewrite input, inject context, or gate the Stop transition. |
| `subagent` | An `Agent` tool that delegates to isolated child agents reusing the same loop, with a recursion-depth guard and the enclosing tool permission policy. |
| `mcp` | Model Context Protocol integration: a stdio JSON-RPC client and in-process servers, exposing tools as `mcp__<server>__<tool>`. |
| `plan` | Plan mode: read-only exploration → `ExitPlanMode` presents a plan for approval → the permission mode flips mid-loop so the agent can implement. |
| `memory` | Persistent long-term memory: markdown-with-frontmatter files + a `MEMORY.md` index, relevance selection, staleness warnings, and `RecallMemory`/`RecordMemory` tools. |
| `skill` | Named, reusable instruction packages loaded from `SKILL.md` files (YAML frontmatter — `name`/`description`/`license`/`compatibility`/`mcps` — plus a Markdown body). A `Skill` tool injects a chosen skill's procedure on demand as standing guidance that survives compaction, and can unlock a skill's declared MCP tools on invocation. |
| `coordinator` | Coordinator mode: a top-level agent that decomposes work and delegates to isolated `worker` subagents — LLM-driven (Agent tool + system segment) or programmatic (`RunParallel` fan-out). |
| `agentcore` | The public entry point: a stateful `Session` whose `Prompt` streams the loop, plus a one-shot `Run`. Aggregates everything above, and optionally persists an append-only JSONL transcript so a session can be resumed (`Session.Resume`). |

## Quick start

```go
prov, _ := llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, Model: "your-model-name"})
// or:  llm.Config{Format: llm.FormatOpenAI, BaseURL: "https://api.deepseek.com/v1", Model: "deepseek-chat"}

sess := agentcore.NewSession(agentcore.Options{
    Provider:       prov,
    SystemPrompt:   []string{"You are a helpful coding agent."}, // the SDK injects NO preset prompt
    PermissionMode: permission.ModeDefault,
    CanUseTool:     myApprovalCallback, // gates mutating tools
})

for ev, err := range sess.Prompt(ctx, "Add a /health endpoint and run the tests.") {
    if err != nil { /* ... */ }
    if ev.Kind == harness.KindText { fmt.Print(ev.Text) }
}
```

## CLI demo

```bash
go build -o lapua ./cmd/agentcore

ANTHROPIC_API_KEY=... ./lapua -provider anthropic -model your-model-name
OPENAI_API_KEY=...    ./lapua -provider openai    -model gpt-4o
OPENAI_API_KEY=...    ./lapua -provider openai-responses -model your-model-name
./lapua -provider openai -model gpt-4o -yes -p "summarize the go files here"
```

Mutating tools (Write/Edit/Bash) prompt for approval unless `-yes` is set. Supply your own system prompt with `-system FILE`.

## Model API formats

| CLI `-provider` | SDK `llm.Config.Format` | Path appended to the base URL | Environment defaults |
|---|---|---|---|
| `anthropic` | `llm.FormatAnthropic` | `/v1/messages` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`) |
| `openai` | `llm.FormatOpenAI` | `/chat/completions` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` (default `https://api.openai.com/v1`) |
| `openai-responses` | `llm.FormatOpenAIResponses` | `/responses` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` (default `https://api.openai.com/v1`) |

Set `-base-url` in the CLI or `llm.Config.BaseURL` in the SDK for a compatible gateway. Include any API prefix the gateway requires, such as `/v1` for the default OpenAI base URL, and leave off the path in the table. Supply the model identifier accepted by that endpoint. All three SDK adapters support streaming and non-streaming requests. The CLI uses streaming. Implement `llm.Provider` to connect a different protocol.

## Tool permissions

`AllowedTools` and `DisallowedTools` accept exact tool names or `*`. Argument selectors such as `Bash(git:*)` are unsupported and deny calls to the named tool. For argument-specific approval, evaluate the arguments in the host's `CanUseTool` callback and leave the tool subject to approval. The permission policy gates tool calls. Process and filesystem isolation are the host's responsibility.

Deferred tools omit their schemas from the model's tool list. The model can request their schemas through `SearchExtraTools` and invoke unlocked targets through `ExecuteExtraTool`. Both the wrapper and the target are checked. Child agents inherit the enclosing query's allow and deny rules and default permission mode. An explicitly configured child mode can further restrict calls.

## Context and session state

Use `agentcore.Options.Compaction` for the built-in automatic summarizer, or call `noaadapter.Enable` before creating the session to install model-driven compression. NOA requires a durable `ArchiveBaseDir`. Set `ModelContextLimit` to match the deployment. The adapter supplies its own request view and takes precedence over built-in compaction when both are configured.

`Session.Usage()` accumulates token counts across completed prompts, including counts restored by `Session.Resume`. The usage reported in a prompt's events describes that prompt. When a transcript write fails, the session continues in memory and exposes the error through `Session.TranscriptError()` and the optional `OnWarn` callback. Resume reads complete JSONL records. On the first append after recovery from an incomplete last record, the store saves the original log as a sibling `.recovery-*` file before trimming that record. A malformed complete record causes resume to return an error.

## Examples

- `examples/custom-llm` — three provider configs + a custom non-coding prompt + a custom tool.
- `examples/coding-agent` — full coding agent with interactive permissions, a PreToolUse hook, and compaction.
- `examples/security-audit` — read-only auditor that delegates to a subagent.
- `examples/planner-memory` — plan mode (explore → approve → implement) plus persistent memory.

## Tests

```bash
go test ./...
```

Tests cover provider adapters with local HTTP fixtures, tool execution and permission checks, transcript recovery, and simulated long sessions with NOA. The GitHub Actions workflow runs the suite on pushes and pull requests.
