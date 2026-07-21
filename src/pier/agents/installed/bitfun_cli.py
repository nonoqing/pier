"""BitFun CLI adapter for Pier's isolated-agent execution model.

This adapter intentionally remains opt-in through ``--agent-import-path``.
It is maintained in the BitFun fork rather than registered as an upstream Pier
agent, while still using Pier's native network policy and artifact lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from pier.agents.base import BaseAgent
from pier.agents.installed.base import NonZeroAgentExitCodeError
from pier.agents.network import allowlist_from_urls, collect_url_values
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.network import NetworkAllowlist
from pier.models.trial.paths import EnvironmentPaths

_NETWORK_POLICY_PREAMBLE = """Network policy for this evaluation:
The task workspace has no general internet access. Do not use web search,
package downloads, or remote source retrieval. The configured model endpoint
is the only permitted outbound connection, and only for agent-model turns."""

_FAILURE_LOG_MAX_BYTES = 512 * 1024
_FAILURE_LOG_HEAD_BYTES = 8 * 1024
_FAILURE_LOG_TAIL_BYTES = 32 * 1024
_FAILURE_LOG_TRUNC_MARKER = "\n...[truncated for host log]...\n"
_BITFUN_DIR = "bitfun"
_SENSITIVE_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "id_token",
        "auth_token",
        "bearer_token",
        "authorization",
        "password",
        "passphrase",
        "secret",
        "client_secret",
        "private_key",
        "credential",
        "credentials",
    }
)
_SENSITIVE_CONFIG_SUFFIXES = ("_secret", "_password", "_private_key")
_PIER_RUNTIME_CONFIG_HOME = "/tmp/pier-bitfun-config"
_TELEMETRY_MODE = "full"


def _format_failure_log_text(text: str) -> str:
    if len(text) <= _FAILURE_LOG_MAX_BYTES:
        return text
    return (
        text[:_FAILURE_LOG_HEAD_BYTES]
        + _FAILURE_LOG_TRUNC_MARKER
        + text[-_FAILURE_LOG_TAIL_BYTES:]
    )


def build_commit_final_changes_script(log_dir: str) -> str:
    """Commit task changes and retain auditable before/after Git evidence."""
    return f"""set -eu
LOG_DIR={shlex.quote(log_dir)}
mkdir -p "$LOG_DIR"
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "not-a-git-repository" > "$LOG_DIR/git-commit.error.txt"
  exit 1
fi
export GIT_AUTHOR_NAME="Pier BitFun"
export GIT_AUTHOR_EMAIL="bitfun-cli@pier.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
git rev-parse HEAD > "$LOG_DIR/git-head.before-commit.txt"
if git diff --quiet && git diff --cached --quiet && \\
   [ -z "$(git ls-files --others --exclude-standard)" ]; then
  echo "no-changes" > "$LOG_DIR/git-commit.result.txt"
  git rev-parse HEAD > "$LOG_DIR/git-head.after-commit.txt"
  exit 0
fi
git add -A
git commit --no-verify -m "Pier BitFun evaluation result"
git rev-parse HEAD > "$LOG_DIR/git-head.after-commit.txt"
echo "committed" > "$LOG_DIR/git-commit.result.txt"
"""


def build_repo_state_capture_script(log_dir: str, phase: str) -> str:
    """Capture Git evidence without producing a second submission artifact."""
    if phase not in {"before", "after"}:
        raise ValueError("phase must be 'before' or 'after'")
    return f"""set +e
LOG_DIR={shlex.quote(log_dir)}
mkdir -p "$LOG_DIR"
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "not-a-git-repository" > "$LOG_DIR/git-{phase}.error.txt"
  cat "$LOG_DIR/git-{phase}.error.txt"
  exit 0
fi
git rev-parse HEAD 2>&1 | tee "$LOG_DIR/git-head.{phase}.txt" || true
git status --porcelain=v1 2>&1 | tee "$LOG_DIR/git-status.{phase}.txt" || true
git log --oneline --decorate -n 20 2>&1 | tee "$LOG_DIR/git-log.{phase}.txt" || true
exit 0
"""


def _build_cp_back_command(log_dir: str) -> str:
    """Copy BitFun diagnostics into Pier's agent log directory in all outcomes."""
    return f"""set +e
LOG_DIR={shlex.quote(log_dir)}
BITFUN_DIR="$LOG_DIR/{_BITFUN_DIR}"
mkdir -p "$BITFUN_DIR/sessions"
PROJECT_PATH=""
for d in "$HOME/.bitfun/projects/testbed" "$HOME/.bitfun/projects/-testbed"; do
  {{ [ -d "$d/sessions" ] || [ -d "$d/request-traces" ]; }} && PROJECT_PATH="$d" && break
done
if [ -z "$PROJECT_PATH" ]; then
  LATEST_SESSIONS=$(ls -dt "$HOME"/.bitfun/projects/*/sessions/ 2>/dev/null | head -1)
  [ -n "$LATEST_SESSIONS" ] && PROJECT_PATH=$(dirname "${{LATEST_SESSIONS%/}}")
fi
if [ -z "$PROJECT_PATH" ]; then
  LATEST_TRACES=$(ls -dt "$HOME"/.bitfun/projects/*/request-traces/ 2>/dev/null | head -1)
  [ -n "$LATEST_TRACES" ] && PROJECT_PATH=$(dirname "${{LATEST_TRACES%/}}")
fi
if [ -n "$PROJECT_PATH" ] && [ -d "$PROJECT_PATH/sessions" ]; then
  cp -R "$PROJECT_PATH/sessions"/. "$BITFUN_DIR/sessions/" 2>/dev/null || true
fi
if [ -n "$PROJECT_PATH" ] && [ -d "$PROJECT_PATH/request-traces" ]; then
  mkdir -p "$BITFUN_DIR/request-traces"
  cp -R "$PROJECT_PATH/request-traces"/. "$BITFUN_DIR/request-traces/" 2>/dev/null || true
fi
BITFUN_CONFIG_HOME="${{XDG_CONFIG_HOME:-$HOME/.config}}"
BITFUN_CONFIG_DIR="$BITFUN_CONFIG_HOME/bitfun"
if [ -d "$BITFUN_CONFIG_DIR/data/token_usage" ]; then
  cp -R "$BITFUN_CONFIG_DIR/data/token_usage" "$BITFUN_DIR/" 2>/dev/null || true
fi
if [ -d "$BITFUN_CONFIG_DIR/cli-logs" ]; then
  cp -R "$BITFUN_CONFIG_DIR/cli-logs" "$BITFUN_DIR/" 2>/dev/null || true
fi
if [ -f "$BITFUN_CONFIG_DIR/logs/bitfun-cli.log" ]; then
  cp "$BITFUN_CONFIG_DIR/logs/bitfun-cli.log" "$BITFUN_DIR/cli.log" 2>/dev/null || true
fi
if [ -f "$BITFUN_CONFIG_DIR/logs/ai-request-audit.jsonl" ]; then
  cp "$BITFUN_CONFIG_DIR/logs/ai-request-audit.jsonl" "$BITFUN_DIR/ai-request-audit.jsonl" 2>/dev/null || true
fi
if command -v tar >/dev/null 2>&1 && [ -d "$BITFUN_DIR/request-traces" ]; then
  tar -C "$BITFUN_DIR" -czf "$BITFUN_DIR/request-traces.tar.gz" request-traces 2>/dev/null || true
fi
printf '{{"sessions":%s,"request_traces":%s,"token_usage":%s,"cli_logs":%s,"cli_log":%s,"ai_request_audit":%s}}\n' \\
  "$([ -d "$BITFUN_DIR/sessions" ] && printf true || printf false)" \\
  "$([ -d "$BITFUN_DIR/request-traces" ] && printf true || printf false)" \\
  "$([ -d "$BITFUN_DIR/token_usage" ] && printf true || printf false)" \\
  "$([ -d "$BITFUN_DIR/cli-logs" ] && printf true || printf false)" \\
  "$([ -f "$BITFUN_DIR/cli.log" ] && printf true || printf false)" \\
  "$([ -f "$BITFUN_DIR/ai-request-audit.jsonl" ] && printf true || printf false)" \\
  > "$BITFUN_DIR/cp-back-manifest.json" 2>/dev/null || true
cat "$BITFUN_DIR/cp-back-manifest.json" 2>/dev/null || true
exit 0
"""


class BitfunCli(BaseAgent):
    """Run a pre-mounted BitFun CLI and leave its final work in ``HEAD``.

    The task's ``pre_artifacts.sh`` remains the only component that creates
    ``/logs/artifacts/model.patch``. The optional BitFun output patch is not
    used by this adapter, avoiding a second, competing artifact path.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        binary_path: str = "/usr/local/bin/bitfun-cli",
        exec_agent: str = "agentic",
        model_endpoint_urls: Iterable[str] | None = None,
        bitfun_config: dict[str, Any] | None = None,
        commit_final_changes: bool = True,
        network_policy_prompt: bool = True,
        extra_env: dict[str, str] | None = None,
        version: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(binary_path, str) or not binary_path:
            raise ValueError("binary_path must be a non-empty string")
        if not isinstance(exec_agent, str) or not exec_agent:
            raise ValueError("exec_agent must be a non-empty string")
        if not isinstance(commit_final_changes, bool):
            raise ValueError("commit_final_changes must be a bool")
        if not isinstance(network_policy_prompt, bool):
            raise ValueError("network_policy_prompt must be a bool")
        if bitfun_config is not None and not isinstance(bitfun_config, dict):
            raise ValueError("bitfun_config must be a dict when provided")

        self._binary_path = binary_path
        self._exec_agent = exec_agent
        self._model_endpoint_urls = (
            [model_endpoint_urls]
            if isinstance(model_endpoint_urls, str)
            else list(model_endpoint_urls or [])
        )
        if not all(isinstance(url, str) for url in self._model_endpoint_urls):
            raise ValueError("model_endpoint_urls must contain only strings")
        self._bitfun_config = bitfun_config
        self._commit_final_changes = commit_final_changes
        self._network_policy_prompt = network_policy_prompt
        self._extra_env = dict(extra_env or {})
        self._version = version
        self._binary_sha256: str | None = None
        self._runtime_config_metadata: dict[str, Any] | None = None
        self._effective_xdg_config_home: str | None = None
        self._pier_egress_proxy_configured = False
        self._telemetry_metadata: dict[str, Any] | None = None
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    @staticmethod
    def name() -> str:
        return "bitfun-cli"

    def version(self) -> str | None:
        return self._version

    def _endpoint_values(self) -> list[str]:
        values = list(self._model_endpoint_urls)
        values.extend(collect_url_values(self._bitfun_config or {}))
        values.extend(collect_url_values(self._extra_env))
        return values

    def network_allowlist(self) -> NetworkAllowlist:
        return allowlist_from_urls(self._endpoint_values())

    def _runtime_env(self, environment: BaseEnvironment) -> dict[str, str]:
        env = dict(environment.agent_process_env(self._extra_env) or {})
        if self._effective_xdg_config_home is not None:
            env["XDG_CONFIG_HOME"] = self._effective_xdg_config_home
        return env

    def _instruction_for(self, environment: BaseEnvironment, instruction: str) -> str:
        if (
            self._network_policy_prompt
            and not environment.task_env_config.allow_internet
        ):
            return f"{_NETWORK_POLICY_PREAMBLE}\n\n{instruction}"
        return instruction

    @property
    def _remote_bitfun_dir(self) -> str:
        return (EnvironmentPaths.agent_dir / _BITFUN_DIR).as_posix()

    @property
    def _remote_agent_log(self) -> str:
        return (EnvironmentPaths.agent_dir / "bitfun.txt").as_posix()

    async def _exec_checked(
        self, environment: BaseEnvironment, *, command: str, label: str
    ) -> Any:
        result = await environment.exec(
            command=command, env=self._runtime_env(environment)
        )
        if result.return_code != 0:
            self._persist_failure_output(result.stdout, result.stderr)
            raise NonZeroAgentExitCodeError(
                f"BitFun {label} exited with {result.return_code}"
            )
        return result

    async def setup(self, environment: BaseEnvironment) -> None:
        binary = shlex.quote(self._binary_path)
        result = await self._exec_checked(
            environment,
            label="CLI setup",
            command=(
                "set -eu; "
                f"test -e {binary}; chmod a+x {binary} 2>/dev/null || true; "
                f"{binary} --version"
            ),
        )
        if result.stdout and not self._version:
            self._version = result.stdout.strip().splitlines()[0]
        checksum_result = await environment.exec(
            command=(
                f"if command -v sha256sum >/dev/null 2>&1; then sha256sum {binary}; "
                f"else shasum -a 256 {binary}; fi"
            ),
            env=self._runtime_env(environment),
        )
        if checksum_result.return_code == 0 and checksum_result.stdout:
            self._binary_sha256 = checksum_result.stdout.split()[0]

    def _provenance_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "binary_path": self._binary_path,
            "binary_sha256": self._binary_sha256,
            "binary_version": self._version,
            "model_name": self.model_name,
            "model_endpoint_domains": self.network_allowlist().domains,
            "exec_agent": self._exec_agent,
            "commit_final_changes": self._commit_final_changes,
            "network_policy_prompt": self._network_policy_prompt,
            "stdout_path": "agent/bitfun.txt",
            "diagnostics_path": "agent/bitfun",
            "git_evidence_path": "agent/bitfun/git",
            "pier_egress_proxy_configured": self._pier_egress_proxy_configured,
        }
        if self._runtime_config_metadata is not None:
            metadata["runtime_config"] = self._runtime_config_metadata
        if self._telemetry_metadata is not None:
            metadata["telemetry"] = self._telemetry_metadata
        return metadata

    def _update_context_metadata(self, context: AgentContext) -> None:
        metadata = dict(context.metadata or {})
        metadata["bitfun_cli"] = self._provenance_metadata()
        context.metadata = metadata

    @staticmethod
    def _is_sensitive_config_key(key: str) -> bool:
        normalized = key.lower().replace("-", "_").replace(" ", "_")
        return normalized in _SENSITIVE_CONFIG_KEYS or normalized.endswith(
            _SENSITIVE_CONFIG_SUFFIXES
        )

    @classmethod
    def _redact_config_secrets(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[REDACTED]"
                if isinstance(key, str) and cls._is_sensitive_config_key(key)
                else cls._redact_config_secrets(child)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [cls._redact_config_secrets(item) for item in value]
        return value

    @staticmethod
    def _model_config_summary(config: dict[str, Any]) -> dict[str, Any]:
        ai = config.get("ai") if isinstance(config.get("ai"), dict) else {}
        defaults = ai.get("default_models")
        if not isinstance(defaults, dict):
            defaults = {}
        selected_ids = {
            key: value
            for key, value in defaults.items()
            if key in {"primary", "fast"} and isinstance(value, str)
        }
        models = ai.get("models")
        if not isinstance(models, list):
            models = (
                config.get("models") if isinstance(config.get("models"), list) else []
            )
        by_id = {
            model.get("id"): model
            for model in models
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        }
        selected_models: dict[str, dict[str, Any]] = {}
        for role, model_id in selected_ids.items():
            model = by_id.get(model_id)
            if isinstance(model, dict):
                selected_models[role] = {
                    key: model[key]
                    for key in (
                        "id",
                        "name",
                        "provider",
                        "model_name",
                        "reasoning_effort",
                        "reasoning_mode",
                        "enabled",
                    )
                    if key in model
                }
        return {
            "redacted_config_path": "agent/bitfun/config/app.redacted.json",
            "redacted_config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "default_models": selected_ids,
            "selected_models": selected_models,
        }

    async def _configure_pier_egress_proxy(self, environment: BaseEnvironment) -> None:
        """Project Pier's authenticated proxy into an isolated BitFun config.

        BitFun deliberately disables environment-derived proxies unless
        ``ai.proxy`` is enabled. Pier's isolated environment injects the proxy
        URL through ``HTTPS_PROXY``; copying the mounted runtime config to
        ``/tmp`` lets this adapter enable that proxy without mutating the
        worker's persistent BitFun configuration or retaining proxy credentials.
        """
        base_env = self._runtime_env(environment)
        proxy_raw = base_env.get("HTTPS_PROXY") or base_env.get("HTTP_PROXY")
        if not proxy_raw:
            return
        parsed = urlsplit(proxy_raw)
        if not parsed.scheme or not parsed.hostname or not parsed.username:
            raise ValueError("Pier egress proxy URL is missing a host or username")
        try:
            port_suffix = f":{parsed.port}" if parsed.port is not None else ""
        except ValueError as exc:
            raise ValueError("Pier egress proxy URL has an invalid port") from exc
        source_home = self._extra_env.get("XDG_CONFIG_HOME", "/root/.config")
        source_path = f"{source_home.rstrip('/')}/bitfun/config/app.json"
        target_path = f"{_PIER_RUNTIME_CONFIG_HOME}/bitfun/config/app.json"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".bitfun-proxy-config-",
            suffix=".json",
            dir=self.logs_dir,
            delete=False,
        ) as raw_file:
            raw_path = Path(raw_file.name)
        try:
            await environment.download_file(source_path, raw_path)
            config = json.loads(raw_path.read_text())
            if not isinstance(config, dict):
                raise ValueError("BitFun runtime app config is not a JSON object")
            ai = config.setdefault("ai", {})
            if not isinstance(ai, dict):
                raise ValueError("BitFun runtime app config has an invalid ai section")
            ai["proxy"] = {
                "enabled": True,
                "url": f"{parsed.scheme}://{parsed.hostname}{port_suffix}",
                "username": unquote(parsed.username),
                "password": unquote(parsed.password or ""),
            }
            raw_path.write_text(json.dumps(config, indent=2) + "\n")
            mkdir_result = await environment.exec(
                command=f"mkdir -p {shlex.quote(str(Path(target_path).parent))}",
                env=base_env,
            )
            if mkdir_result.return_code != 0:
                raise RuntimeError("could not create isolated BitFun config directory")
            await environment.upload_file(raw_path, target_path)
            self._effective_xdg_config_home = _PIER_RUNTIME_CONFIG_HOME
            self._pier_egress_proxy_configured = True
        finally:
            raw_path.unlink(missing_ok=True)

    async def _configure_telemetry(self, environment: BaseEnvironment) -> None:
        """Enable full request tracing in the isolated evaluation config only."""
        base_env = self._runtime_env(environment)
        source_home = self._effective_xdg_config_home or self._extra_env.get(
            "XDG_CONFIG_HOME", "/root/.config"
        )
        source_path = f"{source_home.rstrip('/')}/bitfun/config/app.json"
        target_path = f"{_PIER_RUNTIME_CONFIG_HOME}/bitfun/config/app.json"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".bitfun-telemetry-config-",
            suffix=".json",
            dir=self.logs_dir,
            delete=False,
        ) as raw_file:
            raw_path = Path(raw_file.name)
        try:
            await environment.download_file(source_path, raw_path)
            config = json.loads(raw_path.read_text())
            if not isinstance(config, dict):
                raise ValueError("BitFun runtime app config is not a JSON object")
            app = config.setdefault("app", {})
            if not isinstance(app, dict):
                raise ValueError("BitFun runtime app config has an invalid app section")
            logging = app.setdefault("logging", {})
            if not isinstance(logging, dict):
                raise ValueError("BitFun runtime app config has an invalid logging section")
            logging["model_exchange_tracing"] = {"mode": _TELEMETRY_MODE}
            raw_path.write_text(json.dumps(config, indent=2) + "\n")
            mkdir_result = await environment.exec(
                command=f"mkdir -p {shlex.quote(str(Path(target_path).parent))}",
                env=base_env,
            )
            if mkdir_result.return_code != 0:
                raise RuntimeError("could not create isolated BitFun telemetry config")
            await environment.upload_file(raw_path, target_path)
            self._effective_xdg_config_home = _PIER_RUNTIME_CONFIG_HOME
        finally:
            raw_path.unlink(missing_ok=True)

    async def _capture_final_config(self, environment: BaseEnvironment) -> None:
        """Persist only a redacted runtime config and expose a safe summary."""
        probe = await environment.exec(
            command=(
                'CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"; '
                'CONFIG_PATH="$CONFIG_HOME/bitfun/config/app.json"; '
                'printf "path=%s\\nexists=%s\\n" "$CONFIG_PATH" '
                '"$([ -f "$CONFIG_PATH" ] && printf true || printf false)"'
            ),
            env=self._runtime_env(environment),
        )
        parsed = dict(
            line.split("=", 1)
            for line in (probe.stdout or "").splitlines()
            if "=" in line
        )
        source = parsed.get("path")
        if probe.return_code != 0 or parsed.get("exists") != "true" or not source:
            self._runtime_config_metadata = {
                "redacted_config_path": None,
                "capture_error": "runtime app config unavailable",
            }
            return

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".bitfun-app-config-",
            suffix=".json",
            dir=self.logs_dir,
            delete=False,
        ) as raw_file:
            raw_path = Path(raw_file.name)
        try:
            await environment.download_file(source, raw_path)
            raw_config = json.loads(raw_path.read_text())
            if not isinstance(raw_config, dict):
                raise ValueError("runtime app config is not a JSON object")
            redacted = self._redact_config_secrets(raw_config)
            target_local = self.logs_dir / _BITFUN_DIR / "config" / "app.redacted.json"
            target_local.parent.mkdir(parents=True, exist_ok=True)
            target_local.write_text(json.dumps(redacted, indent=2) + "\n")
            target_remote = f"{self._remote_bitfun_dir}/config/app.redacted.json"
            mkdir_result = await environment.exec(
                command=f"mkdir -p {shlex.quote(str(Path(target_remote).parent))}",
                env=self._runtime_env(environment),
            )
            if mkdir_result.return_code != 0:
                raise RuntimeError("could not create remote redacted config directory")
            await environment.upload_file(target_local, target_remote)
            self._runtime_config_metadata = self._model_config_summary(redacted)
        except Exception as exc:
            self._runtime_config_metadata = {
                "redacted_config_path": None,
                "capture_error": str(exc),
            }
        finally:
            raw_path.unlink(missing_ok=True)

    def _persist_failure_output(self, stdout: str | None, stderr: str | None) -> None:
        self._persist_host_diagnostic("bitfun.txt", stdout, stderr, skip_empty=True)

    def _persist_success_trace(self, stdout: str | None, stderr: str | None) -> None:
        """Persist full stream-json events; do not apply failure-log truncation."""
        payload = "".join(part for part in (stdout, stderr) if part)
        if payload:
            path = self.logs_dir / _BITFUN_DIR / "exec-events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, errors="replace")
        self._telemetry_metadata = {
            "mode": _TELEMETRY_MODE,
            "events_path": "agent/bitfun/exec-events.jsonl",
            "request_traces_path": "agent/bitfun/request-traces",
        }

    @staticmethod
    def _telemetry_int(usage: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    def _finalize_telemetry(self, context: AgentContext) -> None:
        """Map persisted stream events and request traces to Pier result fields."""
        root = self.logs_dir / _BITFUN_DIR
        event_count = 0
        tool_calls = 0
        events = root / "exec-events.jsonl"
        if events.is_file():
            for line in events.read_text(errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    event_count += 1
                    if event.get("type") in {"tool_start", "subagent_tool_start"}:
                        tool_calls += 1

        requests = 0
        rounds: set[str] = set()
        input_tokens = output_tokens = cache_tokens = 0
        usage_records = 0
        for trace in root.glob("request-traces/**/*.json"):
            try:
                record = json.loads(trace.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            requests += 1
            operation_id = record.get("operation_id")
            if isinstance(operation_id, str) and operation_id:
                rounds.add(operation_id)
            response = record.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
            if not isinstance(usage, dict):
                continue
            usage_records += 1
            input_tokens += self._telemetry_int(
                usage, "prompt_tokens", "input_tokens", "prompt_token_count"
            ) or 0
            output_tokens += self._telemetry_int(
                usage, "completion_tokens", "output_tokens", "candidates_token_count"
            ) or 0
            details = usage.get("prompt_tokens_details")
            cache_tokens += self._telemetry_int(
                details if isinstance(details, dict) else {},
                "cached_tokens", "cache_read_input_tokens", "cached_content_token_count",
            ) or self._telemetry_int(
                usage, "cache_tokens", "cached_tokens", "cache_read_input_tokens"
            ) or 0

        archive = root / "request-traces.tar.gz"
        if not requests and archive.is_file():
            try:
                with tarfile.open(archive, "r:gz") as bundle:
                    for member in bundle.getmembers():
                        if not member.isfile() or "request-traces/" not in member.name:
                            continue
                        handle = bundle.extractfile(member)
                        if handle is None:
                            continue
                        record = json.loads(handle.read())
                        if not isinstance(record, dict):
                            continue
                        requests += 1
                        operation_id = record.get("operation_id")
                        if isinstance(operation_id, str) and operation_id:
                            rounds.add(operation_id)
                        response = record.get("response")
                        usage = response.get("usage") if isinstance(response, dict) else None
                        if not isinstance(usage, dict):
                            continue
                        usage_records += 1
                        input_tokens += self._telemetry_int(
                            usage, "prompt_tokens", "input_tokens", "prompt_token_count"
                        ) or 0
                        output_tokens += self._telemetry_int(
                            usage, "completion_tokens", "output_tokens", "candidates_token_count"
                        ) or 0
                        details = usage.get("prompt_tokens_details")
                        cache_tokens += self._telemetry_int(
                            details if isinstance(details, dict) else {},
                            "cached_tokens", "cache_read_input_tokens", "cached_content_token_count",
                        ) or self._telemetry_int(
                            usage, "cache_tokens", "cached_tokens", "cache_read_input_tokens"
                        ) or 0
            except (OSError, tarfile.TarError, json.JSONDecodeError):
                pass

        self._telemetry_metadata = {
            "mode": _TELEMETRY_MODE,
            "events_path": "agent/bitfun/exec-events.jsonl",
            "request_traces_path": "agent/bitfun/request-traces",
            "stream_event_count": event_count,
            "tool_calls": tool_calls,
            "model_requests": requests,
            "model_rounds": len(rounds),
            "usage_records": usage_records,
        }
        if rounds:
            context.n_agent_steps = len(rounds)
        if usage_records:
            context.n_input_tokens = input_tokens
            context.n_output_tokens = output_tokens
            context.n_cache_tokens = cache_tokens

    def _persist_host_diagnostic(
        self,
        relative_path: str,
        stdout: str | None,
        stderr: str | None,
        *,
        skip_empty: bool = False,
    ) -> None:
        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            if parts:
                parts.append("\n--- stderr ---\n")
            parts.append(stderr)
        if skip_empty and not parts:
            return
        path = self.logs_dir / _BITFUN_DIR / relative_path
        if relative_path == "bitfun.txt":
            path = self.logs_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_format_failure_log_text("".join(parts)), errors="replace")

    async def _capture_repo_state(
        self, environment: BaseEnvironment, phase: str
    ) -> None:
        """Capture Git state both in the container and on Pier's host logs.

        The host-side copy makes the evidence durable even when an environment
        implementation delays or filters bind-mount propagation during teardown.
        """
        result = await environment.exec(
            command=build_repo_state_capture_script(
                f"{self._remote_bitfun_dir}/git", phase
            ),
            env=self._runtime_env(environment),
        )
        self._persist_host_diagnostic(
            f"git/git-state.{phase}.host.txt", result.stdout, result.stderr
        )

    async def _copy_back_diagnostics(self, environment: BaseEnvironment) -> None:
        """Run the in-container cp-back and retain its manifest on the host."""
        result = await environment.exec(
            command=_build_cp_back_command(EnvironmentPaths.agent_dir.as_posix()),
            env=self._runtime_env(environment),
        )
        self._persist_host_diagnostic(
            "cp-back-manifest.host.json", result.stdout, result.stderr
        )
        try:
            target = self.logs_dir / _BITFUN_DIR / "request-traces.tar.gz"
            target.parent.mkdir(parents=True, exist_ok=True)
            await environment.download_file(
                f"{self._remote_bitfun_dir}/request-traces.tar.gz", target
            )
        except Exception as exc:
            self.logger.debug("BitFun request trace archive download failed: %s", exc)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if (
            not environment.task_env_config.allow_internet
            and not self.network_allowlist().domains
        ):
            raise ValueError(
                "model_endpoint_urls or a BitFun provider base URL is required "
                "for an air-gapped BitFun Pier run"
            )
        git_dir = f"{self._remote_bitfun_dir}/git"
        try:
            await self._configure_pier_egress_proxy(environment)
            await self._configure_telemetry(environment)
            self._update_context_metadata(context)
            await self._capture_repo_state(environment, "before")
            command = (
                "set -o pipefail\n"
                f"mkdir -p {shlex.quote(EnvironmentPaths.agent_dir.as_posix())}\n"
                "if command -v stdbuf >/dev/null 2>&1; then\n"
                f"  bitfun_tee() {{ stdbuf -oL tee {shlex.quote(self._remote_agent_log)}; }}\n"
                "else\n"
                f"  bitfun_tee() {{ tee {shlex.quote(self._remote_agent_log)}; }}\n"
                "fi\n"
                f"{shlex.quote(self._binary_path)} exec --output-format stream-json --agent {shlex.quote(self._exec_agent)} -- "
                f"{shlex.quote(self._instruction_for(environment, instruction))} "
                "2>&1 | bitfun_tee\n"
                "rc=${PIPESTATUS[0]}\n"
                "exit $rc"
            )
            result = await self._exec_checked(environment, label="CLI", command=command)
            self._persist_success_trace(result.stdout, result.stderr)
            if self._commit_final_changes:
                await self._exec_checked(
                    environment,
                    label="final Git commit",
                    command=build_commit_final_changes_script(git_dir),
                )
        finally:
            try:
                await self._capture_repo_state(environment, "after")
            except Exception as exc:
                self.logger.debug("BitFun final Git evidence capture failed: %s", exc)
            try:
                await self._copy_back_diagnostics(environment)
            except Exception as exc:
                self.logger.debug("BitFun diagnostics cp-back failed: %s", exc)
            self._finalize_telemetry(context)
            try:
                await self._capture_final_config(environment)
            except Exception as exc:
                self._runtime_config_metadata = {
                    "redacted_config_path": None,
                    "capture_error": str(exc),
                }
            self._update_context_metadata(context)
