import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pier.agents.installed.base import NonZeroAgentExitCodeError
from pier.agents.installed.bitfun_cli import (
    BitfunCli,
    _NETWORK_POLICY_PREAMBLE,
    _format_failure_log_text,
    build_commit_final_changes_script,
    build_repo_state_capture_script,
)
from pier.environments.base import ExecResult
from pier.models.agent.context import AgentContext


class FakeEnvironment:
    def __init__(
        self,
        *,
        allow_internet: bool = False,
        fail_cli: bool = False,
        proxy_url: str | None = None,
    ):
        self.task_env_config = SimpleNamespace(allow_internet=allow_internet)
        self.fail_cli = fail_cli
        self.proxy_url = proxy_url
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.uploads: dict[str, str] = {}

    def agent_process_env(self, env: dict[str, str]) -> dict[str, str]:
        proxy_env = (
            {"HTTP_PROXY": self.proxy_url, "HTTPS_PROXY": self.proxy_url}
            if self.proxy_url
            else {}
        )
        return {"PIER_NETWORK_PROXY": "enabled", **proxy_env, **env}

    async def exec(self, *, command: str, env: dict[str, str]):
        self.calls.append((command, env))
        if "git-head.before.txt" in command:
            return ExecResult(stdout="before-head\n", return_code=0)
        if "git-head.after.txt" in command:
            return ExecResult(stdout="after-head\n", return_code=0)
        if "cp-back-manifest.json" in command:
            return ExecResult(stdout='{"sessions":false}\n', return_code=0)
        if "CONFIG_PATH" in command:
            return ExecResult(
                stdout="path=/testbed/.config/bitfun/config/app.json\nexists=true\n",
                return_code=0,
            )
        if "bitfun-cli exec" in command and self.fail_cli:
            return ExecResult(
                stdout="provider request failed: diagnostic marker", return_code=7
            )
        if "bitfun-cli --version" in command:
            return ExecResult(stdout="bitfun-cli 1.2.3\n", return_code=0)
        if "sha256sum" in command:
            return ExecResult(
                stdout="checksum  /usr/local/bin/bitfun-cli\n", return_code=0
            )
        return ExecResult(stdout="", return_code=0)

    async def download_file(self, source_path: str, target_path: Path):
        if source_path in self.uploads:
            Path(target_path).write_text(self.uploads[source_path])
            return
        Path(target_path).write_text(
            json.dumps(
                {
                    "ai": {
                        "default_models": {
                            "primary": "deepseek-v4-pro",
                            "fast": "deepseek-v4-pro",
                        },
                        "models": [
                            {
                                "id": "deepseek-v4-pro",
                                "provider": "deepseek",
                                "model_name": "DeepSeek-V4-Pro",
                                "reasoning_effort": "max",
                                "api_key": "must-not-leak",
                            }
                        ],
                    }
                }
            )
        )

    async def upload_file(self, source_path: Path, target_path: str):
        self.uploads[target_path] = Path(source_path).read_text()


def test_network_allowlist_uses_explicit_and_configured_endpoints(tmp_path: Path):
    agent = BitfunCli(
        logs_dir=tmp_path,
        model_endpoint_urls=["https://gateway.example.com/v1"],
        bitfun_config={"providers": [{"base_url": "https://models.example.net"}]},
    )

    assert set(agent.network_allowlist().domains) == {
        "gateway.example.com",
        "models.example.net",
    }


def test_network_allowlist_accepts_the_single_url_cli_kwarg_form(tmp_path: Path):
    agent = BitfunCli(
        logs_dir=tmp_path,
        model_endpoint_urls="https://gateway.example.com/v1",
    )

    assert agent.network_allowlist().domains == ["gateway.example.com"]


def test_commit_script_commits_all_changes_without_fabricating_a_patch():
    script = build_commit_final_changes_script("/logs/agent/bitfun/git")

    assert "git add -A" in script
    assert "git commit --no-verify" in script
    assert "git-head.before-commit.txt" in script
    assert "/logs/artifacts/model.patch" not in script


def test_commit_script_creates_a_commit_and_records_heads(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("base\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("changed\n")

    log_dir = tmp_path / "logs"
    subprocess.run(
        ["bash", "-c", build_commit_final_changes_script(log_dir.as_posix())],
        cwd=repo,
        check=True,
    )

    assert (log_dir / "git-commit.result.txt").read_text().strip() == "committed"
    assert (log_dir / "git-head.before-commit.txt").read_text() != (
        log_dir / "git-head.after-commit.txt"
    ).read_text()


def test_repo_state_capture_is_evidence_only():
    script = build_repo_state_capture_script("/logs/agent/bitfun/git", "before")

    assert "git-head.before.txt" in script
    assert "git status --porcelain=v1" in script
    assert "git diff" not in script


def test_failure_log_is_bounded_and_preserves_tail_marker():
    payload = "a" * (512 * 1024 + 1) + "TAIL_MARKER"
    text = _format_failure_log_text(payload)

    assert "[truncated for host log]" in text
    assert text.endswith("TAIL_MARKER")
    assert len(text) < len(payload)


def test_run_preserves_diagnostics_and_runtime_config(tmp_path: Path):
    agent = BitfunCli(
        logs_dir=tmp_path,
        model_name="deepseek-v4-pro",
        model_endpoint_urls=["https://gateway.example.com/v1"],
        extra_env={"XDG_CONFIG_HOME": "/testbed/.config"},
    )
    environment = FakeEnvironment()
    context = AgentContext()

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Fix the failing test.", environment, context))

    commands = [command for command, _ in environment.calls]
    run_command = next(command for command in commands if "bitfun-cli exec" in command)
    assert _NETWORK_POLICY_PREAMBLE in run_command
    assert "stdbuf -oL tee" in run_command
    assert "--output-format stream-json" in run_command
    assert any("git-head.before.txt" in command for command in commands)
    assert any("git-head.after.txt" in command for command in commands)
    assert any("cp-back-manifest.json" in command for command in commands)
    assert (
        tmp_path / "bitfun/git/git-state.before.host.txt"
    ).read_text() == "before-head\n"
    assert (
        tmp_path / "bitfun/git/git-state.after.host.txt"
    ).read_text() == "after-head\n"
    assert (
        tmp_path / "bitfun/cp-back-manifest.host.json"
    ).read_text() == '{"sessions":false}\n'
    metadata = context.metadata["bitfun_cli"]
    assert metadata["model_endpoint_domains"] == ["gateway.example.com"]
    assert metadata["runtime_config"]["default_models"] == {
        "primary": "deepseek-v4-pro",
        "fast": "deepseek-v4-pro",
    }
    assert metadata["telemetry"]["mode"] == "full"
    uploaded = environment.uploads["/logs/agent/bitfun/config/app.redacted.json"]
    assert "must-not-leak" not in uploaded
    assert "[REDACTED]" in uploaded


def test_cli_failure_persists_output_runs_finally_and_raises_pier_error(tmp_path: Path):
    agent = BitfunCli(
        logs_dir=tmp_path,
        model_endpoint_urls=["https://gateway.example.com/v1"],
    )
    environment = FakeEnvironment(fail_cli=True)

    with pytest.raises(NonZeroAgentExitCodeError, match="CLI exited with 7"):
        asyncio.run(agent.run("Fix the failing test.", environment, AgentContext()))

    assert "diagnostic marker" in (tmp_path / "bitfun.txt").read_text()
    commands = [command for command, _ in environment.calls]
    assert any("git-head.after.txt" in command for command in commands)
    assert any("cp-back-manifest.json" in command for command in commands)
    assert (
        tmp_path / "bitfun/git/git-state.after.host.txt"
    ).read_text() == "after-head\n"


def test_pier_proxy_is_projected_into_an_isolated_redacted_runtime_config(
    tmp_path: Path,
):
    agent = BitfunCli(
        logs_dir=tmp_path,
        model_endpoint_urls=["https://gateway.example.com/v1"],
        extra_env={"XDG_CONFIG_HOME": "/testbed/.config"},
    )
    environment = FakeEnvironment(
        proxy_url="http://agent:proxy-secret@pier-egress-proxy:8080"
    )
    context = AgentContext()

    asyncio.run(agent.run("Fix the failing test.", environment, context))

    projected = json.loads(
        environment.uploads["/tmp/pier-bitfun-config/bitfun/config/app.json"]
    )
    assert projected["ai"]["proxy"]["enabled"] is True
    assert projected["ai"]["proxy"]["url"] == "http://pier-egress-proxy:8080"
    assert context.metadata["bitfun_cli"]["pier_egress_proxy_configured"] is True
    assert projected["app"]["logging"]["model_exchange_tracing"] == {"mode": "full"}
    redacted = environment.uploads["/logs/agent/bitfun/config/app.redacted.json"]
    assert "proxy-secret" not in redacted
    assert "[REDACTED]" in redacted


def test_air_gapped_run_requires_a_model_endpoint(tmp_path: Path):
    agent = BitfunCli(logs_dir=tmp_path)

    with pytest.raises(ValueError, match="model_endpoint_urls"):
        asyncio.run(
            agent.run("Fix the failing test.", FakeEnvironment(), AgentContext())
        )


def test_finalize_telemetry_maps_events_and_usage_to_agent_context(tmp_path: Path):
    telemetry_dir = tmp_path / "bitfun"
    telemetry_dir.mkdir()
    (telemetry_dir / "exec-events.jsonl").write_text(
        '{"type":"tool_start"}\n{"type":"subagent_tool_start"}\n'
    )
    traces = telemetry_dir / "request-traces" / "session"
    traces.mkdir(parents=True)
    (traces / "000001.json").write_text(
        json.dumps(
            {
                "operation_id": "round-1",
                "response": {
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    }
                },
            }
        )
    )
    context = AgentContext()

    agent = BitfunCli(logs_dir=tmp_path)
    agent._finalize_telemetry(context)

    assert context.n_agent_steps == 1
    assert context.n_input_tokens == 12
    assert context.n_output_tokens == 3
    assert context.n_cache_tokens == 2
    assert context.metadata is None
    assert agent._telemetry_metadata["tool_calls"] == 2
