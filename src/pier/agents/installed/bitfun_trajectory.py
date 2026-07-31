"""Convert BitFun ``stream-json`` output into Harbor-compatible ATIF."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pier.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_int(value: Any) -> int | None:
    return value if _is_int(value) else None


def _without_none(values: dict[str, Any]) -> dict[str, Any] | None:
    filtered = {key: value for key, value in values.items() if value is not None}
    return filtered or None


def _event_timestamp(record: dict[str, Any]) -> str | None:
    raw = record.get("timestamp")
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        return None
    seconds = raw.get("secs_since_epoch")
    nanos = raw.get("nanos_since_epoch", 0)
    if not _is_int(seconds) or not _is_int(nanos):
        return None
    try:
        timestamp = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
            microsecond=nanos // 1_000
        )
    except (OverflowError, OSError, ValueError):
        return None
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass
class _ToolState:
    tool_id: str
    tool_name: str = "unknown"
    timestamp: str | None = None
    params: dict[str, Any] | None = None
    params_chunks: list[str] = field(default_factory=list)
    terminal_event: dict[str, Any] | None = None
    attempt_id: str | None = None
    attempt_index: int | None = None

    def arguments(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if self.params is not None:
            return self.params, None
        raw = "".join(self.params_chunks)
        if not raw:
            return {}, None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}, {"raw_arguments": raw, "arguments_parse_error": True}
        if isinstance(parsed, dict):
            return parsed, None
        return {"input": parsed}, None


@dataclass
class _RoundState:
    round_id: str
    turn_id: str
    timestamp: str | None = None
    round_index: int | None = None
    model_config_id: str | None = None
    model_name: str | None = None
    thinking_chunks: list[str] = field(default_factory=list)
    text_chunks: list[str] = field(default_factory=list)
    tools: dict[str, _ToolState] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None


@dataclass
class _TurnState:
    turn_id: str
    timestamp: str | None = None
    turn_index: int | None = None
    user_input: str | None = None
    user_message_metadata: dict[str, Any] | None = None
    round_ids: list[str] = field(default_factory=list)
    completion: dict[str, Any] | None = None


def _tool_observation(tool: _ToolState) -> Observation | None:
    event = tool.terminal_event
    if event is None:
        return None
    event_type = event.get("event_type")
    raw_result = event.get("result")
    result_for_assistant = event.get("result_for_assistant")
    error = event.get("error")
    if isinstance(result_for_assistant, str) and result_for_assistant:
        content: str | None = result_for_assistant
    elif isinstance(error, str) and error:
        content = error
    elif raw_result is not None:
        try:
            content = json.dumps(raw_result, ensure_ascii=False)
        except (TypeError, ValueError):
            content = str(raw_result)
    else:
        content = None

    extra = _without_none(
        {
            "raw_result": raw_result,
            "success": event_type == "Completed",
            "error": error,
            "tool_duration_ms": _optional_int(event.get("duration_ms")),
        }
    )
    return Observation(
        results=[
            ObservationResult(
                source_call_id=tool.tool_id,
                content=content,
                extra=extra,
            )
        ]
    )


def _round_metrics(round_state: _RoundState) -> Metrics | None:
    usage = round_state.usage
    if usage is None:
        return None
    return Metrics(
        prompt_tokens=_optional_int(usage.get("input_tokens")),
        completion_tokens=_optional_int(usage.get("output_tokens")),
        cached_tokens=_optional_int(usage.get("cached_tokens")),
        extra=_without_none(
            {
                "total_tokens": _optional_int(usage.get("total_tokens")),
                "max_context_tokens": _optional_int(usage.get("max_context_tokens")),
            }
        ),
    )


def _round_extra(round_state: _RoundState) -> dict[str, Any] | None:
    completion = round_state.completion or {}
    return _without_none(
        {
            "turn_id": round_state.turn_id,
            "round_id": round_state.round_id,
            "round_index": round_state.round_index,
            "model_config_id": round_state.model_config_id,
            "round_status": "completed" if round_state.completion else None,
            "has_tool_calls": completion.get("has_tool_calls"),
            "duration_ms": _optional_int(completion.get("duration_ms")),
            "attempt_count": _optional_int(completion.get("attempt_count")),
        }
    )


def _tool_step(
    *,
    step_id: int,
    tool: _ToolState,
    round_state: _RoundState,
    reasoning_content: str | None,
    metrics: Metrics | None,
    llm_call_count: int,
) -> Step:
    arguments, argument_extra = tool.arguments()
    terminal = tool.terminal_event or {}
    tool_extra = _without_none(
        {
            **(argument_extra or {}),
            "attempt_id": tool.attempt_id,
            "attempt_index": tool.attempt_index,
            "queue_wait_ms": _optional_int(terminal.get("queue_wait_ms")),
            "preflight_ms": _optional_int(terminal.get("preflight_ms")),
            "confirmation_wait_ms": _optional_int(terminal.get("confirmation_wait_ms")),
            "execution_ms": _optional_int(terminal.get("execution_ms")),
        }
    )
    status = terminal.get("event_type")
    return Step(
        step_id=step_id,
        timestamp=tool.timestamp or round_state.timestamp,
        source="agent",
        model_name=round_state.model_name,
        message=(
            f"Executed {tool.tool_name}"
            if status in {"Completed", "Failed"}
            else f"Called {tool.tool_name}"
        ),
        reasoning_content=reasoning_content,
        tool_calls=[
            ToolCall(
                tool_call_id=tool.tool_id,
                function_name=tool.tool_name,
                arguments=arguments,
                extra=tool_extra,
            )
        ],
        observation=_tool_observation(tool),
        metrics=metrics,
        llm_call_count=llm_call_count,
        extra=_without_none(
            {
                "turn_id": round_state.turn_id,
                "round_id": round_state.round_id,
                "tool_status": status.lower() if isinstance(status, str) else None,
            }
        ),
    )


def convert_bitfun_stream_to_trajectory(
    events_path: Path,
    *,
    agent_version: str | None,
    default_model_name: str | None,
    exec_agent: str,
) -> Trajectory | None:
    """Convert BitFun stream envelopes into a validated ATIF-v1.7 trajectory."""
    turns: dict[str, _TurnState] = {}
    turn_order: list[str] = []
    rounds: dict[str, _RoundState] = {}
    current_round_by_turn: dict[str, str] = {}
    session_id: str | None = None
    stream_event_count = 0

    def get_turn(turn_id: str) -> _TurnState:
        if turn_id not in turns:
            turns[turn_id] = _TurnState(turn_id=turn_id)
            turn_order.append(turn_id)
        return turns[turn_id]

    def get_round(event: dict[str, Any], timestamp: str | None) -> _RoundState | None:
        raw_round_id = event.get("round_id")
        raw_turn_id = event.get("turn_id")
        if not isinstance(raw_round_id, str) or not raw_round_id:
            return None
        turn_id = (
            raw_turn_id if isinstance(raw_turn_id, str) and raw_turn_id else "unknown"
        )
        round_state = rounds.get(raw_round_id)
        if round_state is None:
            round_state = _RoundState(
                round_id=raw_round_id,
                turn_id=turn_id,
                timestamp=timestamp,
            )
            rounds[raw_round_id] = round_state
            turn = get_turn(turn_id)
            turn.round_ids.append(raw_round_id)
        return round_state

    try:
        handle = events_path.open(errors="replace")
    except OSError:
        return None
    with handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            nested = record.get("event")
            event = nested if isinstance(nested, dict) else record
            if not isinstance(event, dict):
                continue
            stream_event_count += 1
            timestamp = _event_timestamp(record)
            raw_session_id = event.get("session_id")
            if session_id is None and isinstance(raw_session_id, str):
                session_id = raw_session_id
            event_type = event.get("type")
            raw_turn_id = event.get("turn_id")
            turn_id = (
                raw_turn_id
                if isinstance(raw_turn_id, str) and raw_turn_id
                else "unknown"
            )

            if event_type == "DialogTurnStarted":
                turn = get_turn(turn_id)
                turn.timestamp = timestamp
                turn.turn_index = _optional_int(event.get("turn_index"))
                original = event.get("original_user_input")
                user_input = event.get("user_input")
                turn.user_input = (
                    original
                    if isinstance(original, str) and original
                    else user_input
                    if isinstance(user_input, str)
                    else ""
                )
                metadata = event.get("user_message_metadata")
                turn.user_message_metadata = (
                    metadata if isinstance(metadata, dict) else None
                )
                continue

            if event_type == "DialogTurnCompleted":
                get_turn(turn_id).completion = event
                continue

            round_state = get_round(event, timestamp)
            if event_type == "ModelRoundStarted" and round_state is not None:
                round_state.timestamp = timestamp or round_state.timestamp
                round_state.round_index = _optional_int(event.get("round_index"))
                model_config_id = event.get("model_config_id")
                model_name = event.get("effective_model_name")
                round_state.model_config_id = (
                    model_config_id if isinstance(model_config_id, str) else None
                )
                round_state.model_name = (
                    model_name if isinstance(model_name, str) else default_model_name
                )
                current_round_by_turn[turn_id] = round_state.round_id
                continue

            if event_type == "TokenUsageUpdated":
                current_round_id = current_round_by_turn.get(turn_id)
                if current_round_id is not None:
                    rounds[current_round_id].usage = event
                    model_name = event.get("effective_model_name")
                    if not rounds[current_round_id].model_name and isinstance(
                        model_name, str
                    ):
                        rounds[current_round_id].model_name = model_name
                continue

            if round_state is None:
                continue
            if event_type == "ThinkingChunk":
                content = event.get("content")
                if isinstance(content, str):
                    round_state.thinking_chunks.append(content)
                continue
            if event_type == "TextChunk":
                text = event.get("text")
                if isinstance(text, str):
                    round_state.text_chunks.append(text)
                continue
            if event_type == "ModelRoundCompleted":
                round_state.completion = event
                model_name = event.get("effective_model_name")
                if isinstance(model_name, str):
                    round_state.model_name = model_name
                continue
            if event_type != "ToolEvent":
                continue
            tool_event = event.get("tool_event")
            if not isinstance(tool_event, dict):
                continue
            raw_tool_id = tool_event.get("tool_id")
            if not isinstance(raw_tool_id, str) or not raw_tool_id:
                continue
            tool = round_state.tools.get(raw_tool_id)
            if tool is None:
                tool = _ToolState(tool_id=raw_tool_id, timestamp=timestamp)
                round_state.tools[raw_tool_id] = tool
            raw_tool_name = tool_event.get("tool_name")
            if isinstance(raw_tool_name, str) and raw_tool_name:
                tool.tool_name = raw_tool_name
            raw_attempt_id = event.get("attempt_id")
            if isinstance(raw_attempt_id, str):
                tool.attempt_id = raw_attempt_id
            tool.attempt_index = _optional_int(event.get("attempt_index"))
            tool_event_type = tool_event.get("event_type")
            if tool_event_type == "ParamsPartial":
                params_chunk = tool_event.get("params")
                if isinstance(params_chunk, str):
                    tool.params_chunks.append(params_chunk)
            elif tool_event_type == "Started":
                params = tool_event.get("params")
                if isinstance(params, dict):
                    tool.params = params
                tool.timestamp = timestamp or tool.timestamp
            elif tool_event_type in {"Completed", "Failed"}:
                tool.terminal_event = tool_event

    steps: list[Step] = []
    usage_records = 0
    prompt_tokens = completion_tokens = cached_tokens = 0
    peak_context_tokens: int | None = None
    tool_call_count = 0
    observed_model_name = default_model_name

    for turn_id in turn_order:
        turn = turns[turn_id]
        if turn.user_input is not None:
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=turn.timestamp,
                    source="user",
                    message=turn.user_input,
                    extra=_without_none(
                        {
                            "turn_id": turn.turn_id,
                            "turn_index": turn.turn_index,
                            "turn_kind": "user_dialog",
                            "user_message_metadata": turn.user_message_metadata,
                        }
                    ),
                )
            )

        for round_id in turn.round_ids:
            round_state = rounds[round_id]
            observed_model_name = round_state.model_name or observed_model_name
            usage = round_state.usage
            if usage is not None:
                usage_records += 1
                round_prompt = _optional_int(usage.get("input_tokens")) or 0
                prompt_tokens += round_prompt
                completion_tokens += _optional_int(usage.get("output_tokens")) or 0
                cached_tokens += _optional_int(usage.get("cached_tokens")) or 0
                peak_context_tokens = (
                    round_prompt
                    if peak_context_tokens is None
                    else max(peak_context_tokens, round_prompt)
                )

            message = "".join(round_state.text_chunks)
            reasoning = "".join(round_state.thinking_chunks) or None
            metrics = _round_metrics(round_state)
            llm_call_recorded = False
            if message or not round_state.tools:
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=round_state.timestamp,
                        source="agent",
                        model_name=round_state.model_name,
                        message=message,
                        reasoning_content=reasoning,
                        metrics=metrics,
                        llm_call_count=1,
                        extra=_round_extra(round_state),
                    )
                )
                llm_call_recorded = True

            for tool in round_state.tools.values():
                first_llm_step = not llm_call_recorded
                steps.append(
                    _tool_step(
                        step_id=len(steps) + 1,
                        tool=tool,
                        round_state=round_state,
                        reasoning_content=reasoning if first_llm_step else None,
                        metrics=metrics if first_llm_step else None,
                        llm_call_count=1 if first_llm_step else 0,
                    )
                )
                llm_call_recorded = True
                tool_call_count += 1

    if not steps:
        return None

    turn_completions = [
        turns[turn_id].completion
        for turn_id in turn_order
        if turns[turn_id].completion is not None
    ]
    final_metrics = FinalMetrics(
        total_prompt_tokens=prompt_tokens if usage_records else None,
        total_completion_tokens=completion_tokens if usage_records else None,
        total_cached_tokens=cached_tokens if usage_records else None,
        total_steps=len(steps),
        extra=_without_none(
            {
                "model_rounds": len(rounds),
                "llm_call_count": len(rounds),
                "tool_calls": tool_call_count,
                "usage_records": usage_records,
                "stream_event_count": stream_event_count,
                "peak_context_tokens": peak_context_tokens,
                "dialog_turns": len(turn_order),
                "dialog_success": (
                    all(event.get("success") is True for event in turn_completions)
                    if turn_completions
                    else None
                ),
            }
        ),
    )
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(
            name="bitfun-cli",
            version=agent_version or "unknown",
            model_name=observed_model_name,
            extra={
                "exec_agent": exec_agent,
                "source_format": "bitfun-stream-json",
            },
        ),
        steps=steps,
        final_metrics=final_metrics,
        extra={"source_path": "agent/bitfun/exec-events.jsonl"},
    )
