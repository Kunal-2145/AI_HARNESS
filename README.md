# AI Harness

An interactive AI harness that classifies requests, plans work, and chooses among workspace tools and configured MCP servers. General chat does not open MCP connections; issue prompts can use GitHub MCP when configured.

## Setup

The evaluator provides `AI_API_KEY` through the process environment. For local development, supply it through your shell or secret manager; `.env.example` intentionally contains an empty value. Never commit a local `.env` file or put a key in source code.

The default provider is OpenRouter at `https://openrouter.ai/api/v1`, using `deepseek/deepseek-v4.1-flash:free`. The API key must be supplied as `AI_API_KEY`. To use another OpenAI-compatible endpoint, such as a compatible AWS endpoint, set `LLM_BASE_URL` and `LLM_MODEL` for that provider.

Optional settings:

```dotenv
AI_HARNESS_WORKSPACE=.
GITHUB_MCP_URL=https://api.githubcopilot.com/mcp/?toolsets=issues,repos
MCP_SERVERS_JSON='[{"name":"docs","url":"http://localhost:8080/mcp","description":"Internal documentation search","token_env":"DOCS_MCP_TOKEN"}]'
```

GitHub is registered when `GITHUB_PERSONAL_ACCESS_TOKEN` is set in the environment. Add any other MCP servers in `MCP_SERVERS_JSON` as an array of `{name, url, description, token_env}` objects. `token_env` refers to the name of an environment variable containing that server's bearer token. Never put token values directly in `MCP_SERVERS_JSON` or commit `.env`.

## Evaluator Flow

The expected evaluator sequence is:

```bash
git clone <repository-url>
cd <repository-directory>
make setup
make run
```

`make run` remains active and displays a `You>` prompt. Enter a task or GitHub issue URL there, for example `Fix the failing tests for https://github.com/owner/repo/issues/123`. Type `exit` or `quit` to stop. The harness plans from the configured capability descriptions; if no capability covers the request, it should explain the limitation rather than claim it can perform the work. GitHub issue/repository tools are registered when `GITHUB_PERSONAL_ACCESS_TOKEN` is set.

`make test` runs the test suite. `make clean` removes generated caches and build artifacts without deleting the virtual environment.

The agent loop has a 12-step application-level cap; provider context, output-token, and rate limits still apply.

## Session Context

Conversation turns are stored in SQLite at `~/.ai-harness/history.sqlite3` and associated with the configured workspace. The next launch resumes that workspace's most recently used session. Enter `/new` to start a separate session. Up to 12 recent messages and 16,000 characters are included in model context; older messages remain stored but are not sent with every request. Set `AI_HARNESS_DB_PATH` to choose another database location. The database contains prompts, assistant responses, and selected execution plans, so keep it private and do not commit it.

In unattended environments, tools execute without `y/N` prompts. Local file listing, reading, and writing are confined to `AI_HARNESS_WORKSPACE` and hide `.env`, `.git`, and virtual-environment paths; no special Desktop access is configured. Terminal commands use argument arrays without a shell, an executable allowlist, a workspace working directory, secret-filtered environment variables, and time/output limits. Note that a working-directory restriction is not an operating-system sandbox: a permitted interpreter can itself access other paths if its code does so. Git push/commit and other high-impact Git commands are blocked by default. The agent loop has a 12-step application-level cap; provider context, output-token, and rate limits still apply.

## Package Layout

- `orchestration/`: model-driven task planning and routing decisions.
- `mcp_tools/`: generic configured MCP discovery, plus local filesystem, content search, and terminal tools.
- `agents/`: tool-using execution agent and review/test helpers.
- `tools/`: shared tool schema and invocation registry.
- `state/`: request, plan, selected-server, and answer state.
- `config/`: environment-backed model, workspace, and MCP server settings.
- `tests/`: offline tests for routing, workspace boundaries, unattended execution, and MCP dispatch.

For a local checkout, `make setup` installs locked project and test dependencies with `uv`. The `make run` target is interactive so the evaluator can feed requests after startup; requests are read from standard input.

