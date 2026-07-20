"""BitFun CLI adapter for Pier's isolated-agent execution model.

This adapter intentionally remains opt-in through ``--agent-import-path``.
It is maintained in the BitFun fork rather than registered as an upstream Pier
agent, while still using Pier's native network policy and artifact lifecycle.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pier.agents.base import BaseAgent
from pier.agents.network import allowlist_from_urls, collect_url_values
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.network import NetworkAllowlist
from pier.models.trial.paths import EnvironmentPaths

_NETWORK_POLICY_PREAMBLE = """Network policy for this evaluation:
The task workspace has no general internet access. Do not use web search,
package downloads, or remote source retrieval. The configured model endpoint
is the only permitted outbound connection, and only for agent-model turns."""


def build_commit_final_changes_script() -> str:
    """Commit the task worktree so the task's pre_artifacts hook can diff HEAD."""
    return """set -eu
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "BitFun agent workspace is not a Git repository" >&2
  exit 1
fi
if git diff --quiet && git diff --cached --quiet && \\
   [ -z "$(git ls-files --others --exclude-standard)" ]; then
  echo "no-changes"
  exit 0
fi
export GIT_AUTHOR_NAME="Pier BitFun"
export GIT_AUTHOR_EMAIL="bitfun-cli@pier.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
git add -A
git commit --no-verify -m "Pier BitFun evaluation result"
git rev-parse HEAD
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
        return environment.agent_process_env(self._extra_env)

    def _instruction_for(self, environment: BaseEnvironment, instruction: str) -> str:
        if self._network_policy_prompt and not environment.task_env_config.allow_internet:
            return f"{_NETWORK_POLICY_PREAMBLE}\n\n{instruction}"
        return instruction

    async def setup(self, environment: BaseEnvironment) -> None:
        binary = shlex.quote(self._binary_path)
        result = await environment.exec(
            command=(
                "set -eu; "
                f"test -e {binary}; chmod a+x {binary} 2>/dev/null || true; "
                f"{binary} --version"
            ),
            env=self._runtime_env(environment),
        )
        if result.return_code != 0:
            raise RuntimeError(f"BitFun CLI setup failed with exit {result.return_code}")
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
        return {
            "binary_path": self._binary_path,
            "binary_sha256": self._binary_sha256,
            "binary_version": self._version,
            "model_name": self.model_name,
            "model_endpoint_domains": self.network_allowlist().domains,
            "exec_agent": self._exec_agent,
            "commit_final_changes": self._commit_final_changes,
            "network_policy_prompt": self._network_policy_prompt,
        }

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not environment.task_env_config.allow_internet and not self.network_allowlist().domains:
            raise ValueError(
                "model_endpoint_urls or a BitFun provider base URL is required "
                "for an air-gapped BitFun Pier run"
            )
        context.metadata = {"bitfun_cli": self._provenance_metadata()}

        agent_log = EnvironmentPaths.agent_dir / "bitfun.txt"
        command = (
            "set -o pipefail; "
            f"mkdir -p {shlex.quote(EnvironmentPaths.agent_dir.as_posix())}; "
            f"{shlex.quote(self._binary_path)} exec "
            f"--agent {shlex.quote(self._exec_agent)} -- "
            f"{shlex.quote(self._instruction_for(environment, instruction))} "
            f"2>&1 | tee {shlex.quote(agent_log.as_posix())}"
        )
        result = await environment.exec(command=command, env=self._runtime_env(environment))
        if result.return_code != 0:
            raise RuntimeError(f"BitFun CLI exited with {result.return_code}")

        if self._commit_final_changes:
            commit_result = await environment.exec(
                command=build_commit_final_changes_script(),
                env=self._runtime_env(environment),
            )
            if commit_result.return_code != 0:
                raise RuntimeError(
                    "BitFun completed but its workspace could not be committed "
                    f"(exit {commit_result.return_code})"
                )
