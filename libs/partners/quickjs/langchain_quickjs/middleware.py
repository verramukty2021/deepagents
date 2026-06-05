"""`CodeInterpreterMiddleware`: exposes a sandboxed JavaScript REPL tool."""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core._api import beta
from langchain_core._api.deprecation import warn_deprecated
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.skills import SkillMetadata
    from langgraph.runtime import Runtime

from langchain_quickjs._format import format_outcome
from langchain_quickjs._prompt import (
    render_eval_tool_code_doc,
    render_eval_tool_description,
    render_repl_system_prompt,
)
from langchain_quickjs._ptc import (
    PTCOption,
    filter_tools_for_ptc,
    render_ptc_prompt,
)
from langchain_quickjs._repl import _Registry

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_MAX_PTC_CALLS = 256
_DEFAULT_MAX_RESULT_CHARS = 4_000
_DEFAULT_TOOL_NAME = "eval"


class REPLState(AgentState):
    """State schema for `CodeInterpreterMiddleware`."""

    _quickjs_slot_id: NotRequired[Annotated[str | None, PrivateStateAttr]]
    _quickjs_snapshot_payload: NotRequired[Annotated[bytes | None, PrivateStateAttr]]


class EvalSchema(BaseModel):
    """Input schema for the `eval` tool."""

    code: str = Field(
        description=(
            "JavaScript expression or statement(s) to evaluate. "
            "No fs/network/real-clock access."
        ),
    )


def _resolve_persistence_flags(
    *,
    mode: Literal["thread", "turn", "call"] | None,
    snapshot_between_turns: bool | None,
) -> tuple[Literal["thread", "turn", "call"], bool, bool]:
    """Normalize persistence configuration and enforce invariant constraints."""
    if snapshot_between_turns is not None:
        warn_deprecated(
            since="0.1.2",
            removal="0.2.0",
            message=(
                "Passing `snapshot_between_turns` to "
                "`CodeInterpreterMiddleware` is deprecated and will be "
                "removed in langchain-quickjs==0.2.0. Use `mode='thread'` "
                "or `mode='turn'` instead."
            ),
            package="langchain-quickjs",
        )
    if mode is None:
        if snapshot_between_turns is None or snapshot_between_turns:
            return "thread", True, False
        return "turn", False, False

    if mode == "thread":
        if snapshot_between_turns is False:
            msg = "`snapshot_between_turns=False` is incompatible with `mode='thread'`."
            raise ValueError(msg)
        return "thread", True, False

    if mode == "turn":
        if snapshot_between_turns is True:
            msg = "`snapshot_between_turns=True` is incompatible with `mode='turn'`."
            raise ValueError(msg)
        return "turn", False, False

    if snapshot_between_turns is True:
        msg = "`snapshot_between_turns=True` is incompatible with `mode='call'`."
        raise ValueError(msg)
    return "call", False, True


def _new_slot_id() -> str:
    """Create a private interpreter slot id."""
    return f"qjs_{uuid.uuid4().hex}"


@beta()
class CodeInterpreterMiddleware(AgentMiddleware[REPLState, ContextT, ResponseT]):
    """Middleware exposing a JS REPL to the agent.

    Each LangGraph thread gets its own QuickJS slot (worker + runtime +
    context), so globals from one conversation cannot leak into another.

    Args:
        memory_limit: Bytes the QuickJS heap may use. Shared across all
            contexts under the same Runtime. Default 64 MiB.
        timeout: Per-call wall-clock timeout in seconds. Applied to every
            `eval` on every context. Default 5.
        max_ptc_calls: Maximum number of `tools.*` bridge calls allowed
            during one `eval` execution. Exceeding this budget throws
            from the host-function bridge before invoking the tool.
            Uncaught overflows surface as `PTCCallBudgetExceeded`.
            `None` disables the budget (unsafe for untrusted prompts;
            enables PTC-call DoS patterns). Default 256.

            !!! warning

                Setting `max_ptc_calls=None` disables the call budget and can allow
                unbounded PTC host-call loops (DoS risk). Only disable in trusted
                environments.

        tool_name: Name of the tool exposed to the model. Default `eval`.
        max_result_chars: Result and stdout blocks are independently
            truncated to this many characters before being sent back to
            the model. Console buffering is also bounded to this value
            during collection. Default 4000.
        capture_console: If `True`, install a `console` object that
            buffers `console.log/warn/error` calls and emits them in
            `<stdout>` blocks alongside the result. Default `True`.
        skills_backend: Optional `BackendProtocol` the REPL reads skill
            source files from. When set and a paired
            `SkillsMiddleware` populates `skills_metadata` in state,
            skills with a `module` frontmatter key become dynamic-
            importable from the REPL as `await import("@/skills/<name>")`.
            When `None`, skill modules are not installed
            (`import(...)` fails at the resolver). This must be the
            same backend `SkillsMiddleware` uses.
        ptc: Programmatic tool calling — expose agent tools inside the
            REPL as `tools.<camelCase>(input) => Promise<string>`. One
            `eval` call can then orchestrate many tool calls (loops,
            `Promise.all`, conditional branching). Accepts:

            - `None` (default) — disabled.
            - `list[str | BaseTool]` — allowlist entries may be:
              - `str` tool names, matched against the agent's toolset.
              - `BaseTool` instances, exposed directly even if not on
                the agent's tool list.

            Mixed lists are supported. Explicit `BaseTool` entries are
            considered first; then name-matched agent tools are added.
            Duplicate names are deduplicated.

            !!! warning
                PTC calls currently execute through the REPL bridge and
                do **not** go through the normal `ToolNode` path. As a
                result, `interrupt_on` / HITL approval workflows are not
                enforced per PTC-invoked tool call.

            The REPL's own tool is always excluded; a model asking for
            `tools.eval("...")` would recurse pointlessly.
        mode: REPL state persistence mode.
            - `"thread"`: state persists across calls and across turns.
            - `"turn"`: state persists across calls within a turn only.
            - `"call"`: each eval call runs in a fresh REPL.
            If omitted, defaults to `"thread"`
        snapshot_between_turns: Compatibility knob for turn-vs-thread
            behavior. When `mode` is omitted, `True` resolves to
            `"thread"` and `False` resolves to `"turn"`. When `mode` is
            provided, incompatible combinations raise `ValueError`.

            !!! deprecated

                Passing `snapshot_between_turns` is deprecated. Use
                `mode="thread"` or `mode="turn"` instead.
        max_snapshot_bytes: Maximum serialized snapshot payload size allowed
            in middleware state. If a snapshot exceeds this size, it is
            dropped (`_quickjs_snapshot_payload=None`). Defaults to
            `memory_limit`.

    Example:
        ```python
        from deepagents import create_deep_agent
        from langchain_quickjs import CodeInterpreterMiddleware

        agent = create_deep_agent(
            model="claude-sonnet-4-6",
            middleware=[CodeInterpreterMiddleware()],
        )
        ```
    """

    state_schema = REPLState

    def __init__(
        self,
        *,
        memory_limit: int = _DEFAULT_MEMORY_LIMIT,
        timeout: float = _DEFAULT_TIMEOUT,
        max_ptc_calls: int | None = _DEFAULT_MAX_PTC_CALLS,
        tool_name: str = _DEFAULT_TOOL_NAME,
        max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
        capture_console: bool = True,
        ptc: PTCOption | None = None,
        skills_backend: "BackendProtocol | None" = None,
        mode: Literal["thread", "turn", "call"] | None = None,
        snapshot_between_turns: bool | None = None,
        max_snapshot_bytes: int | None = None,
    ) -> None:
        """Initialize REPL middleware state and build the exposed eval tool."""
        super().__init__()
        if max_ptc_calls is not None and max_ptc_calls < 1:
            msg = "`max_ptc_calls` must be >= 1 or None"
            raise ValueError(msg)
        if max_snapshot_bytes is not None and max_snapshot_bytes < 1:
            msg = "`max_snapshot_bytes` must be >= 1 or None"
            raise ValueError(msg)
        self._memory_limit = memory_limit
        self._timeout = timeout
        self._max_ptc_calls = max_ptc_calls
        self._tool_name = tool_name
        self._max_result_chars = max_result_chars
        self._capture_console = capture_console
        self._ptc = ptc
        self._skills_backend = skills_backend
        (
            self._mode,
            self._snapshot_between_turns,
            self._reset_between_calls,
        ) = _resolve_persistence_flags(
            mode=mode,
            snapshot_between_turns=snapshot_between_turns,
        )
        self._max_snapshot_bytes = (
            memory_limit if max_snapshot_bytes is None else max_snapshot_bytes
        )
        self._registry = _Registry(
            memory_limit=memory_limit,
            timeout=timeout,
            capture_console=capture_console,
            max_stdout_chars=max_result_chars,
            max_ptc_calls=max_ptc_calls,
        )
        self._base_system_prompt = render_repl_system_prompt(
            tool_name=tool_name,
            timeout=timeout,
            memory_limit_mb=memory_limit // (1024 * 1024),
            mode=self._mode,
        )
        self._ptc_prompt_cache: tuple[frozenset[str], str] | None = None
        self._ptc_tools_by_slot: dict[str, tuple[BaseTool, ...]] = {}
        self.tools: list[BaseTool] = [self._build_tool()]

    def _build_tool(self) -> BaseTool:
        tool_name = self._tool_name
        max_chars = self._max_result_chars
        middleware = self
        code_doc = render_eval_tool_code_doc(mode=self._mode)
        tool_description = render_eval_tool_description(mode=self._mode)

        def _make_tool_message(
            outcome: Any,
            tool_call_id: str | None,
        ) -> ToolMessage:
            return ToolMessage(
                content=format_outcome(outcome, max_result_chars=max_chars),
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        def sync_eval(
            runtime: ToolRuntime[None, Any],
            code: Annotated[str, code_doc],
        ) -> ToolMessage:
            slot_id = middleware._slot_id_for_state(runtime.state)
            repl = middleware._repl_for_eval(slot_id)
            skills = middleware._skills_for_eval(runtime)
            try:
                outcome = repl.eval_sync(
                    code,
                    skills=skills,
                    skills_backend=middleware._skills_backend,
                    outer_runtime=runtime,
                )
            finally:
                if middleware._reset_between_calls:
                    middleware._registry.reset_repl(slot_id)
            return _make_tool_message(outcome, runtime.tool_call_id)

        async def async_eval(
            runtime: ToolRuntime[None, Any],
            code: Annotated[str, code_doc],
        ) -> ToolMessage:
            slot_id = middleware._slot_id_for_state(runtime.state)
            repl = middleware._repl_for_eval(slot_id)
            skills = middleware._skills_for_eval(runtime)
            try:
                outcome = await repl.eval_async(
                    code,
                    skills=skills,
                    skills_backend=middleware._skills_backend,
                    outer_runtime=runtime,
                    outer_loop=asyncio.get_running_loop(),
                )
            finally:
                if middleware._reset_between_calls:
                    middleware._registry.reset_repl(slot_id)
            return _make_tool_message(outcome, runtime.tool_call_id)

        return StructuredTool.from_function(
            name=tool_name,
            description=tool_description,
            func=sync_eval,
            coroutine=async_eval,
            infer_schema=False,
            args_schema=EvalSchema,
            metadata={"ls_code_input_language": "javascript"},
        )

    def _ptc_tool_names(self) -> set[str]:
        """Collect tool names from the PTC configuration."""
        names: set[str] = set()
        for entry in self._ptc or []:
            if isinstance(entry, str):
                names.add(entry)
            elif isinstance(entry, BaseTool):
                names.add(entry.name)
        return names

    def _skills_for_eval(
        self,
        runtime: ToolRuntime[None, Any],
    ) -> dict[str, "SkillMetadata"] | None:
        """Return per-eval skill metadata map."""
        if self._skills_backend is None:
            return None
        metadata_list = (
            runtime.state.get("skills_metadata", []) if runtime.state else []
        )
        ptc_names = self._ptc_tool_names()
        result: dict[str, SkillMetadata] = {}
        for m in metadata_list:
            raw = m.get("metadata", {}).get("required-ptc-tools", "")
            required = str(raw).split() if raw else []
            missing = [t for t in required if t not in ptc_names]
            if missing:
                logger.warning(
                    "Skill '%s' requires PTC tools not in ptc config: %s",
                    m["name"],
                    ", ".join(missing),
                )
                continue
            result[m["name"]] = m
        return result

    def _validate_required_ptc_tools(self, state: REPLState) -> None:
        """Raise if any skill requires PTC tools not in the config."""
        if self._skills_backend is None:
            return
        metadata_list: list[SkillMetadata] = state.get("skills_metadata", [])  # type: ignore[assignment]
        ptc_names = self._ptc_tool_names()
        for skill in metadata_list:
            raw = skill.get("metadata", {}).get("required-ptc-tools", "")
            required = str(raw).split() if raw else []
            missing = [t for t in required if t not in ptc_names]
            if missing:
                msg = (
                    f"Skill '{skill['name']}' requires PTC tools"
                    " that are not configured: "
                    f"{', '.join(missing)}. "
                    f"Add them to CodeInterpreterMiddleware(ptc=[...])."
                )
                raise ValueError(msg)

    def _repl_for_eval(self, slot_id: str) -> Any:
        """Return the REPL slot for one eval invocation."""
        repl = self._registry.get(slot_id)
        if self._reset_between_calls and self._ptc is not None:
            repl.install_tools(list(self._ptc_tools_by_slot.get(slot_id, ())))
        return repl

    def _slot_id_for_state(
        self,
        state: dict[str, Any] | None,
    ) -> str:
        """Return the interpreter slot id for this state, minting one if needed."""
        if state is None:
            return _new_slot_id()
        slot_id = state.get("_quickjs_slot_id")
        if isinstance(slot_id, str) and slot_id:
            return slot_id
        slot_id = _new_slot_id()
        state["_quickjs_slot_id"] = slot_id
        return slot_id

    def _slot_update_for_runtime(self) -> dict[str, str]:
        """Build a private state update with a fresh slot id when needed."""
        return {"_quickjs_slot_id": _new_slot_id()}

    def before_agent(
        self,
        state: REPLState,
        runtime: "Runtime[ContextT]",  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Ensure a private REPL slot exists and restore snapshot bytes if present."""
        self._validate_required_ptc_tools(state)
        slot_id = state.get("_quickjs_slot_id")
        update: dict[str, Any] | None = None
        if not isinstance(slot_id, str) or not slot_id:
            update = self._slot_update_for_runtime()
            slot_id = update["_quickjs_slot_id"]
        if self._reset_between_calls or not self._snapshot_between_turns:
            return update
        payload = state.get("_quickjs_snapshot_payload")
        if payload is None:
            return update
        repl = self._registry.get(slot_id)
        try:
            repl.restore_snapshot(payload, inject_globals=True)
        except Exception:  # noqa: BLE001  # best-effort restore path
            logger.warning(
                "Failed to restore QuickJS snapshot for slot_id=%s",
                slot_id,
                exc_info=True,
            )
            if update is None:
                update = {}
            update["_quickjs_snapshot_payload"] = None
        return update

    async def abefore_agent(
        self,
        state: REPLState,
        runtime: "Runtime[ContextT]",  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Ensure a private REPL slot exists and restore snapshot bytes if present."""
        self._validate_required_ptc_tools(state)
        slot_id = state.get("_quickjs_slot_id")
        update: dict[str, Any] | None = None
        if not isinstance(slot_id, str) or not slot_id:
            update = self._slot_update_for_runtime()
            slot_id = update["_quickjs_slot_id"]
        if self._reset_between_calls or not self._snapshot_between_turns:
            return update
        payload = state.get("_quickjs_snapshot_payload")
        if payload is None:
            return update
        repl = self._registry.get(slot_id)
        try:
            await repl.arestore_snapshot(payload, inject_globals=True)
        except Exception:  # noqa: BLE001  # best-effort restore path
            logger.warning(
                "Failed to restore QuickJS snapshot for slot_id=%s",
                slot_id,
                exc_info=True,
            )
            if update is None:
                update = {}
            update["_quickjs_snapshot_payload"] = None
        return update

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject the REPL's system-prompt snippet on every model call."""
        prompt = self._prepare_for_call(request)
        return handler(
            request.override(
                system_message=self._extend(request.system_message, prompt)
            ),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """(async) Inject the REPL's system-prompt snippet on every model call."""
        prompt = self._prepare_for_call(request)
        return await handler(
            request.override(
                system_message=self._extend(request.system_message, prompt)
            ),
        )

    def _prepare_for_call(self, request: ModelRequest[ContextT]) -> str:
        """Install PTC bindings for this turn and return the system-prompt addendum.

        Called from both sync and async model-call wrappers. Reads the
        live tool list off the request (middlewares upstream may have
        filtered it), decides what PTC exposes this turn, registers any
        missing host-function bridges on the current thread's REPL, and
        rebuilds `globalThis.tools` if the exposed name set changed.
        """
        if self._ptc is None:
            return self._base_system_prompt
        request_tools: list[BaseTool] = list(getattr(request, "tools", []) or [])
        exposed = filter_tools_for_ptc(
            request_tools,
            self._ptc,
            self_tool_name=self._tool_name,
        )
        # Install on the current slot's REPL. If the slot hasn't
        # evaluated anything yet, this creates the context lazily — which
        # is fine: PTC bindings must be in place *before* the first eval
        # that references them, and the next eval on this slot is the
        # earliest that could matter.
        slot_id = self._slot_id_for_state(getattr(request, "state", None))
        repl = self._registry.get(slot_id)
        repl.install_tools(exposed)
        self._ptc_tools_by_slot[slot_id] = tuple(exposed)
        # Rendering the TS-ish signature block is cheap but not free;
        # cache by the set of exposed names. The set doesn't encode tool
        # *identity* — if a tool keeps its name but its schema changes
        # between turns, the cached prompt staleness is on the caller.
        # Same tradeoff the TS package accepts; see the module docstring.
        exposed_names = frozenset(t.name for t in exposed)
        if self._ptc_prompt_cache is None or self._ptc_prompt_cache[0] != exposed_names:
            self._ptc_prompt_cache = (
                exposed_names,
                render_ptc_prompt(exposed, tool_name=self._tool_name),
            )
        return self._base_system_prompt + self._ptc_prompt_cache[1]

    def _extend(
        self, system_message: SystemMessage | None, prompt: str
    ) -> SystemMessage:
        return append_to_system_message(system_message, prompt)

    def _snapshot_update(
        self, *, payload: bytes, slot_id: str
    ) -> dict[str, bytes | None]:
        """Build state update for a serialized snapshot payload."""
        size = len(payload)
        if size > self._max_snapshot_bytes:
            logger.warning(
                (
                    "Dropping QuickJS snapshot for slot_id=%s "
                    "(size=%d bytes exceeds max_snapshot_bytes=%d)"
                ),
                slot_id,
                size,
                self._max_snapshot_bytes,
            )
            return {"_quickjs_snapshot_payload": None}
        return {"_quickjs_snapshot_payload": payload}

    def after_agent(
        self,
        state: REPLState,
        runtime: "Runtime[ContextT]",  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Snapshot REPL state (optional) and evict this turn's REPL slot."""
        slot_id = state.get("_quickjs_slot_id")
        if not isinstance(slot_id, str) or not slot_id:
            return None
        self._ptc_tools_by_slot.pop(slot_id, None)
        if self._reset_between_calls or not self._snapshot_between_turns:
            self._registry.evict(slot_id)
            return None

        repl = self._registry.get_if_exists(slot_id)
        if repl is None:
            return None
        update: dict[str, Any]
        try:
            update = self._snapshot_update(
                payload=repl.create_snapshot(),
                slot_id=slot_id,
            )
        except Exception:  # noqa: BLE001  # best-effort snapshot path
            logger.warning(
                "Failed to create QuickJS snapshot for slot_id=%s",
                slot_id,
                exc_info=True,
            )
            update = {"_quickjs_snapshot_payload": None}
        finally:
            self._registry.evict(slot_id)
        return update

    async def aafter_agent(
        self,
        state: REPLState,
        runtime: "Runtime[ContextT]",  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of `after_agent` snapshot+evict behavior."""
        slot_id = state.get("_quickjs_slot_id")
        if not isinstance(slot_id, str) or not slot_id:
            return None
        self._ptc_tools_by_slot.pop(slot_id, None)
        if self._reset_between_calls or not self._snapshot_between_turns:
            await self._registry.aevict(slot_id)
            return None

        repl = self._registry.get_if_exists(slot_id)
        if repl is None:
            return None
        update: dict[str, Any]
        try:
            update = self._snapshot_update(
                payload=await repl.acreate_snapshot(),
                slot_id=slot_id,
            )
        except Exception:  # noqa: BLE001  # best-effort snapshot path
            logger.warning(
                "Failed to create QuickJS snapshot for slot_id=%s",
                slot_id,
                exc_info=True,
            )
            update = {"_quickjs_snapshot_payload": None}
        finally:
            await self._registry.aevict(slot_id)
        return update

    def __del__(self) -> None:
        """Best-effort Runtime cleanup on GC; never raises at shutdown."""
        # Wrapped in `contextlib.suppress` because __del__ must not raise
        # during interpreter shutdown, when dependencies may already be
        # half-unloaded.
        with contextlib.suppress(Exception):
            self._registry.close()
