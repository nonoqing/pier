import asyncio
from pathlib import Path
from types import SimpleNamespace

from pier.agents.installed.bitfun_cli import (
    BitfunCli,
    _NETWORK_POLICY_PREAMBLE,
    build_commit_final_changes_script,
)
from pier.environments.base import ExecResult
from pier.models.agent.context import AgentContext


class FakeEnvironment:
    def __init__(self, allow_internet: bool = False):
        self.task_env_config = SimpleNamespace(allow_internet=allow_internet)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def agent_process_env(self, env: dict[str, str]) -> dict[str, str]:
        return {"PIER_NETWORK_PROXY": "enabled", **env}

    async def exec(self, *, command: str, env: dict[str, str]):
        self.calls.append((command, env))
        return ExecResult(stdout="bitfun-cli 1.2.3\n", return_code=0)


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
    script = build_commit_final_changes_script()

    assert "git add -A" in script
    assert "git commit --no-verify" in script
    assert "/logs/artifacts/model.patch" not in script


def test_run_uses_pier_network_environment_and_commits_after_success(tmp_path: Path):
    agent = BitfunCli(
        logs_dir=tmp_path,
        model_endpoint_urls=["https://gateway.example.com/v1"],
        extra_env={"OPENAI_API_KEY": "test-key"},
    )
    environment = FakeEnvironment()
    context = AgentContext()

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Fix the failing test.", environment, context))

    run_command, run_env = environment.calls[2]
    assert _NETWORK_POLICY_PREAMBLE in run_command
    assert "bitfun-cli exec --agent agentic" in run_command
    assert run_env["PIER_NETWORK_PROXY"] == "enabled"
    assert run_env["OPENAI_API_KEY"] == "test-key"
    assert environment.calls[3][0] == build_commit_final_changes_script()
    assert context.metadata == {
        "bitfun_cli": {
            "binary_path": "/usr/local/bin/bitfun-cli",
            "binary_sha256": "bitfun-cli",
            "binary_version": "bitfun-cli 1.2.3",
            "model_name": None,
            "model_endpoint_domains": ["gateway.example.com"],
            "exec_agent": "agentic",
            "commit_final_changes": True,
            "network_policy_prompt": True,
        }
    }


def test_air_gapped_run_requires_a_model_endpoint(tmp_path: Path):
    agent = BitfunCli(logs_dir=tmp_path)

    try:
        asyncio.run(agent.run("Fix the failing test.", FakeEnvironment(), object()))
    except ValueError as exc:
        assert "model_endpoint_urls" in str(exc)
    else:
        raise AssertionError("expected an air-gapped endpoint validation error")
