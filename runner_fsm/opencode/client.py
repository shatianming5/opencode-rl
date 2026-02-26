from __future__ import annotations

import base64
import json
import os
import signal
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..dtypes import AgentClient, AgentResult, TurnEvent
from .tool_parser import parse_tool_calls, format_tool_results
from .tool_executor import ToolPolicy, execute_tool_calls
from ..utils.subprocess import tail

def select_bash_mode(*, purpose: str, default_bash_mode: str, scaffold_bash_mode: str) -> str:
    p = str(purpose or "").strip().lower()
    default = str(default_bash_mode or "restricted").strip().lower() or "restricted"
    scaffold = str(scaffold_bash_mode or default).strip().lower() or default
    if p in ("scaffold_contract", "repair_contract"):
        return scaffold
    return default

@dataclass(frozen=True)
class OpenCodeServerConfig:

    base_url: str
    username: str
    password: str

class OpenCodeRequestError(RuntimeError):

    def __init__(self, *, method: str, url: str, status: int | None, detail: str):
        super().__init__(f"OpenCode request failed: {method} {url} ({status}) {detail}")
        self.method = method
        self.url = url
        self.status = status
        self.detail = detail

class OpenCodeClient(AgentClient):

    def __init__(
        self,
        *,
        repo: Path,
        plan_rel: str,
        pipeline_rel: str | None,
        model: str,
        base_url: str | None,
        timeout_seconds: int,
        request_retry_attempts: int = 2,
        request_retry_backoff_seconds: float = 2.0,
        session_recover_attempts: int | None = None,
        session_recover_backoff_seconds: float | None = None,
        context_length: int | None = None,
        max_prompt_chars: int | None = None,
        bash_mode: str,
        scaffold_bash_mode: str = "full",
        unattended: str,
        server_log_path: Path | None = None,
        username: str | None = None,
        password: str | None = None,
        session_title: str | None = None,
    ) -> None:
        self._repo = repo
        self._plan_rel = str(plan_rel or "PLAN.md").strip() or "PLAN.md"
        self._pipeline_rel = str(pipeline_rel).strip() if pipeline_rel else None
        self._timeout_seconds = int(timeout_seconds) if timeout_seconds else 300
        self._request_retry_attempts = max(0, int(request_retry_attempts or 0))
        try:
            _backoff = float(request_retry_backoff_seconds)
        except Exception:
            _backoff = 2.0
        self._request_retry_backoff_seconds = max(0.0, _backoff)
        _recover_attempts_raw = (
            session_recover_attempts
            if session_recover_attempts is not None
            else os.environ.get("AIDER_OPENCODE_SESSION_RECOVER_ATTEMPTS", "2")
        )
        try:
            self._session_recover_attempts = max(0, int(_recover_attempts_raw or 0))
        except Exception:
            self._session_recover_attempts = 2
        _recover_backoff_raw = (
            session_recover_backoff_seconds
            if session_recover_backoff_seconds is not None
            else os.environ.get("AIDER_OPENCODE_SESSION_RECOVER_BACKOFF_SECONDS", "2.0")
        )
        try:
            self._session_recover_backoff_seconds = max(0.0, float(_recover_backoff_raw or 0.0))
        except Exception:
            self._session_recover_backoff_seconds = 2.0
        try:
            _context_length = int(context_length or 0)
        except Exception:
            _context_length = 0
        self._context_length: int | None = _context_length if _context_length > 0 else None
        try:
            _max_prompt_chars = int(max_prompt_chars or 0)
        except Exception:
            _max_prompt_chars = 0
        self._max_prompt_chars: int | None = _max_prompt_chars if _max_prompt_chars > 0 else None
        self._bash_mode = (bash_mode or "restricted").strip().lower()
        if self._bash_mode not in ("restricted", "full"):
            raise ValueError("invalid_bash_mode")
        self._scaffold_bash_mode = (scaffold_bash_mode or "full").strip().lower()
        if self._scaffold_bash_mode not in ("restricted", "full"):
            raise ValueError("invalid_scaffold_bash_mode")
        self._unattended = str(unattended or "strict").strip().lower() or "strict"
        self._session_title = str(session_title or f"runner:{repo.name}")
        self._server_log_path = server_log_path.resolve() if server_log_path is not None else None

        model_str = str(model or "").strip()
        if not model_str:
            provider_id, model_id = "openai", "gpt-4o-mini"
        elif "/" in model_str:
            provider_id, model_id = model_str.split("/", 1)
            provider_id = provider_id.strip() or "openai"
            model_id = model_id.strip() or "gpt-4o-mini"
        else:
            provider_id, model_id = "openai", model_str
        self._model_obj: dict[str, str] = {"providerID": provider_id, "modelID": model_id}
        self._model_str: str = f"{provider_id}/{model_id}"

        self._proc: subprocess.Popen[str] | None = None
        self._proxy_proc: subprocess.Popen[str] | None = None
        self._proxy_log_file = None
        self._token_log_path: Path | None = None
        self._temp_config_home: Path | None = None
        self._server_log_file = None
        self._owns_local_server = not bool(base_url)

        if base_url:
            base_url_s = str(base_url).strip()
            if not base_url_s:
                raise ValueError("empty_url")
            self._server = OpenCodeServerConfig(
                base_url=base_url_s.rstrip("/"),
                username=(username or "opencode").strip() or "opencode",
                password=(password or "").strip(),
            )
        else:
            self._server = self._start_local_server(
                repo=repo, server_log_path=self._server_log_path, username=username
            )

        try:
            deadline = time.time() + 60
            last_err = ""
            _health_start = time.time()
            _health_dots = 0
            while time.time() < deadline:
                try:
                    self._request_json("GET", "/global/health", body=None, require_auth=bool(self._server.password))
                    elapsed_h = time.time() - _health_start
                    print(f"    server ready ({elapsed_h:.1f}s)", flush=True)
                    break
                except Exception as e:
                    last_err = str(e)
                    _health_dots += 1
                    if _health_dots % 25 == 0:  # 每 5 秒打印一次（0.2s * 25）
                        elapsed_h = time.time() - _health_start
                        print(f"    waiting for server... ({elapsed_h:.0f}s)", flush=True)
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"OpenCode server failed health check: {tail(last_err, 2000)}")

            data = self._request_json(
                "POST",
                "/session",
                body={"title": self._session_title},
                require_auth=bool(self._server.password),
            )
            if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].strip():
                self._session_id = data["id"]
            elif isinstance(data, dict) and isinstance(data.get("sessionID"), str) and data["sessionID"].strip():
                self._session_id = data["sessionID"]
            else:
                raise RuntimeError(
                    f"unexpected /session response: {tail(json.dumps(data, ensure_ascii=False), 2000)}"
                )
        except Exception:
            # If init fails after starting a local server, ensure we don't leak the process.
            try:
                self.close()
            except Exception:
                pass
            raise

    def close(self) -> None:
        self._stop_local_server()
        if self._server_log_file is not None:
            try:
                self._server_log_file.close()
            except Exception:
                pass
            self._server_log_file = None

    def _stop_local_server(self) -> None:
        if self._proc is not None:
            try:
                # Do not block shutdown on a potentially wedged server.
                self._request_json("POST", "/instance/dispose", body=None, require_auth=True, timeout_seconds=5)
            except Exception:
                pass
            try:
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                    except Exception:
                        self._proc.terminate()
                else:  # pragma: no cover
                    self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    if os.name == "posix":
                        try:
                            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                        except Exception:
                            self._proc.kill()
                    else:  # pragma: no cover
                        self._proc.kill()
            except Exception:
                pass
            finally:
                self._proc = None
        if self._server_log_file is not None:
            try:
                self._server_log_file.close()
            except Exception:
                pass
            self._server_log_file = None
        # Stop LLM proxy
        if self._proxy_proc is not None:
            try:
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(self._proxy_proc.pid), signal.SIGTERM)
                    except Exception:
                        self._proxy_proc.terminate()
                else:
                    self._proxy_proc.terminate()
                self._proxy_proc.wait(timeout=3)
            except Exception:
                try:
                    self._proxy_proc.kill()
                except Exception:
                    pass
            finally:
                self._proxy_proc = None
        if self._proxy_log_file is not None:
            try:
                self._proxy_log_file.close()
            except Exception:
                pass
            self._proxy_log_file = None
        # Clean up temp config dir
        if self._temp_config_home is not None:
            try:
                shutil.rmtree(self._temp_config_home, ignore_errors=True)
            except Exception:
                pass
            self._temp_config_home = None

    class _LLMWaitMonitor:
        """Tail token log (from LLM proxy) + heartbeat while waiting for LLM response.

        Token log format (written by llm_proxy.py):
            [HH:MM:SS] >>> stream start model=glm-4.7
            <raw token content, no newlines between tokens>
            [HH:MM:SS] <<< stream end 523 tokens 4.2s

        Output format:
            | >>> stream start model=glm-4.7
            | 让我先看看数据格式，用 terminal 执行以下命令：...
            | <<< stream end 523 tokens 4.2s
        """

        def __init__(self, token_log: Path | None, turn: int, heartbeat_interval: float = 5.0):
            self._token_log = token_log
            self._turn = turn
            self._interval = heartbeat_interval
            self._stop = threading.Event()
            self._thread: threading.Thread | None = None
            self._start = 0.0
            self._log_offset = 0
            self._line_buf = ""  # accumulate tokens into lines

        def start(self):
            self._start = time.time()
            if self._token_log:
                try:
                    self._log_offset = self._token_log.stat().st_size
                except Exception:
                    self._log_offset = 0
                print(f"    ... waiting for LLM response (turn {self._turn})", flush=True)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def stop(self):
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=1)
            # Flush any remaining buffer
            if self._line_buf.strip():
                text = self._line_buf.strip()
                if len(text) > 200:
                    text = text[:197] + "..."
                print(f"    | {text}", flush=True)
                self._line_buf = ""

        def _run(self):
            last_activity = time.time()
            while not self._stop.wait(0.3):
                printed = self._tail_token_log()
                now = time.time()
                if printed:
                    last_activity = now
                elif (now - last_activity) >= self._interval:
                    elapsed = now - self._start
                    print(f"    ... LLM thinking ({elapsed:.0f}s)", flush=True)
                    last_activity = now

        def _tail_token_log(self) -> bool:
            if not self._token_log:
                return False
            try:
                size = self._token_log.stat().st_size
                if size <= self._log_offset:
                    return False
                with open(self._token_log, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(self._log_offset)
                    new = f.read(size - self._log_offset)
                self._log_offset = size
                if not new:
                    return False
            except Exception:
                return False

            printed = False
            for ch in new:
                if ch == "\n":
                    line = self._line_buf.strip()
                    if line:
                        # Metadata lines (>>> / <<<) print as-is; token lines truncate
                        if len(line) > 200:
                            line = line[:197] + "..."
                        print(f"    | {line}", flush=True)
                        printed = True
                    self._line_buf = ""
                else:
                    self._line_buf += ch
                    # Flush partial tokens every 80 chars (show streaming progress)
                    if len(self._line_buf) >= 80 and "\n" not in self._line_buf:
                        text = self._line_buf.strip()
                        if text:
                            print(f"    | {text}", flush=True)
                            printed = True
                        self._line_buf = ""
            return printed

    def run(self, text: str, *, fsm_state: str, iter_idx: int, purpose: str, on_turn=None) -> AgentResult:
        policy = ToolPolicy(
            repo=self._repo.resolve(),
            plan_path=(self._repo / self._plan_rel).resolve(),
            pipeline_path=((self._repo / self._pipeline_rel).resolve() if self._pipeline_rel else None),
            purpose=purpose,
            bash_mode=select_bash_mode(
                purpose=purpose,
                default_bash_mode=self._bash_mode,
                scaffold_bash_mode=self._scaffold_bash_mode,
            ),
            unattended=self._unattended,
        )

        prompt = text
        trace: list[dict[str, Any]] = []
        for turn_idx in range(20):
            monitor = self._LLMWaitMonitor(self._token_log_path, turn_idx + 1) if on_turn else None
            try:
                if monitor:
                    monitor.start()
                try:
                    msg = self._post_message_with_retry(model=self._model_obj, text=prompt)
                except OpenCodeRequestError as e:
                    # Compatibility fallback: some builds may accept model as a string.
                    if e.status in (400, 422):
                        msg = self._post_message_with_retry(model=self._model_str, text=prompt)
                    else:
                        raise
            finally:
                if monitor:
                    monitor.stop()
            opencode_err = None
            if isinstance(msg, dict):
                info = msg.get("info")
                if isinstance(info, dict):
                    err_obj = info.get("error")
                    if isinstance(err_obj, dict):
                        name = str(err_obj.get("name") or "").strip() or "Error"
                        data = err_obj.get("data")
                        detail = ""
                        if isinstance(data, dict):
                            detail = str(data.get("message") or "").strip()
                        if not detail:
                            detail = str(data).strip() if data is not None else ""
                        opencode_err = f"{name}: {detail}" if detail else name
            if opencode_err:
                raise RuntimeError(f"OpenCode agent error: {opencode_err}")

            if not isinstance(msg, dict):
                assistant_text = str(msg)
            else:
                parts = msg.get("parts")
                if not isinstance(parts, list):
                    assistant_text = str(msg)
                else:
                    texts: list[str] = []
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text" and isinstance(part.get("text"), str):
                            t = part["text"]
                            if t.strip():
                                texts.append(t)
                    assistant_text = "\n".join(texts) or str(msg)
            calls = parse_tool_calls(assistant_text)
            if not calls:
                trace.append(
                    {
                        "turn": int(turn_idx + 1),
                        "assistant_text_tail": tail(assistant_text or "", 4000),
                        "calls": [],
                        "results": [],
                    }
                )
                if on_turn:
                    on_turn(TurnEvent(turn=turn_idx + 1, assistant_text=assistant_text, finished=True))
                return AgentResult(assistant_text=assistant_text, raw=msg, tool_trace=trace)

            results = execute_tool_calls(calls, repo=self._repo, policy=policy)
            compact_results: list[dict[str, Any]] = []
            for r in results:
                detail = dict(r.detail or {})
                if isinstance(detail.get("content"), str):
                    detail["content"] = tail(detail["content"], 4000)
                if isinstance(detail.get("stdout"), str):
                    detail["stdout"] = tail(detail["stdout"], 4000)
                if isinstance(detail.get("stderr"), str):
                    detail["stderr"] = tail(detail["stderr"], 4000)
                compact_results.append(detail | {"tool": r.kind, "ok": bool(r.ok)})
            trace.append(
                {
                    "turn": int(turn_idx + 1),
                    "assistant_text_tail": tail(assistant_text or "", 4000),
                    "calls": [
                        {
                            "kind": str(c.kind),
                            "payload": c.payload if isinstance(c.payload, (dict, list, str, int, float, bool)) else str(c.payload),
                        }
                        for c in calls
                    ],
                    "results": compact_results,
                }
            )

            if on_turn:
                on_turn(TurnEvent(
                    turn=turn_idx + 1,
                    assistant_text=assistant_text,
                    calls=list(calls),
                    results=list(results),
                ))

            # For scaffold runs, we don't need the agent to "finish talking" if the contract
            # is already valid. Some models keep emitting extra tool calls indefinitely.
            if str(purpose or "").strip().lower() == "scaffold_contract" and self._pipeline_rel:
                try:
                    from ..core.pipeline_spec import load_pipeline_spec
                    from ..contract.validation import validate_scaffold_contract

                    pipeline_path = (self._repo / self._pipeline_rel).resolve()
                    if pipeline_path.exists():
                        parsed = load_pipeline_spec(pipeline_path)
                        report = validate_scaffold_contract(self._repo, pipeline=parsed, require_metrics=True)
                        if not report.errors:
                            return AgentResult(assistant_text=assistant_text, raw=msg, tool_trace=trace)
                except Exception:
                    pass

            prompt = format_tool_results(results)

        raise RuntimeError("OpenCode tool loop exceeded 20 turns without a final response.")

    def _start_local_server(
        self,
        *,
        repo: Path,
        server_log_path: Path | None,
        username: str | None,
        append_log: bool = False,
    ) -> OpenCodeServerConfig:
        if not shutil.which("opencode"):
            raise RuntimeError("`opencode` not found in PATH. Install it from https://opencode.ai/")

        host = "127.0.0.1"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            _host, port = s.getsockname()

        user = (username or "opencode").strip() or "opencode"
        pwd = secrets.token_urlsafe(24)
        env = dict(os.environ)
        # OpenCode's OpenAI-compatible provider reads `OPENAI_BASE_URL`.
        # Keep compatibility with `.env` files using `OPENAI_API_BASE`.
        if not str(env.get("OPENAI_BASE_URL") or "").strip():
            api_base = str(env.get("OPENAI_API_BASE") or "").strip().rstrip("/")
            if api_base:
                env["OPENAI_BASE_URL"] = api_base if api_base.endswith("/v1") else (api_base + "/v1")
        env["OPENCODE_SERVER_USERNAME"] = user
        env["OPENCODE_SERVER_PASSWORD"] = pwd

        # --- Start LLM streaming proxy ---
        if server_log_path is not None:
            self._start_llm_proxy(env, server_log_path.parent)

        cmd = ["opencode", "serve", "--hostname", host, "--port", str(port)]

        stdout = subprocess.DEVNULL
        if server_log_path is not None:
            server_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass
            self._server_log_file = server_log_path.open(
                "a" if append_log else "w",
                encoding="utf-8",
            )
            stdout = self._server_log_file

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stdout,
            start_new_session=True,
        )
        return OpenCodeServerConfig(base_url=f"http://{host}:{port}", username=user, password=pwd)

    def _start_llm_proxy(self, env: dict, log_dir: Path) -> None:
        """Start LLM streaming proxy and redirect OpenCode to use it.

        OpenCode reads its LLM API URL from ``~/.config/opencode/opencode.json``
        (via the provider's ``options.baseURL``), **not** from the
        ``OPENAI_BASE_URL`` environment variable.  So we must:

        1. Read the real ``baseURL`` from the OpenCode config.
        2. Start a local proxy pointing at that upstream.
        3. Create a temporary copy of the config with ``baseURL`` pointing
           at the proxy.
        4. Set ``XDG_CONFIG_HOME`` so the OpenCode server process picks up
           the temporary config.
        """
        proxy_script = Path(__file__).with_name("llm_proxy.py")
        if not proxy_script.exists():
            return

        # ---- 1. Find the real upstream URL ----
        upstream_url = ""
        config_data = None

        # Try OpenCode config (respects XDG_CONFIG_HOME)
        xdg = str(env.get("XDG_CONFIG_HOME") or "").strip()
        config_home = Path(xdg) if xdg else Path.home() / ".config"
        config_path = config_home / "opencode" / "opencode.json"
        provider_id = self._model_obj.get("providerID", "openai")

        if config_path.exists():
            try:
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                providers = config_data.get("provider", {})
                if provider_id in providers:
                    opts = providers[provider_id].get("options", {})
                    upstream_url = str(opts.get("baseURL", "")).strip().rstrip("/")
            except Exception:
                pass

        # Fallback: use OPENAI_BASE_URL env var
        if not upstream_url:
            upstream_url = str(env.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
            # Strip /v1 — the proxy forwards the full path including /v1
            if upstream_url.endswith("/v1"):
                upstream_url = upstream_url[:-3]

        if not upstream_url:
            return  # No URL to proxy

        # ---- 2. Start proxy process ----
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, proxy_port = s.getsockname()

        log_dir.mkdir(parents=True, exist_ok=True)
        self._token_log_path = log_dir / "llm_tokens.log"
        self._token_log_path.write_text("", encoding="utf-8")

        proxy_log = log_dir / "llm_proxy.log"
        self._proxy_log_file = proxy_log.open("w", encoding="utf-8")

        self._proxy_proc = subprocess.Popen(
            [
                sys.executable, str(proxy_script),
                "--port", str(proxy_port),
                "--upstream", upstream_url,
                "--log", str(self._token_log_path),
                "--debug",
            ],
            stdin=subprocess.DEVNULL,
            stdout=self._proxy_log_file,
            stderr=self._proxy_log_file,
            text=True,
            start_new_session=True,
        )

        proxy_base = f"http://127.0.0.1:{proxy_port}"

        # ---- 3. Redirect OpenCode to the proxy ----
        # a) If we found an OpenCode config with the provider, create a temp
        #    copy with baseURL pointing at the proxy.
        if config_data and provider_id in config_data.get("provider", {}):
            import copy
            patched = copy.deepcopy(config_data)
            patched["provider"][provider_id]["options"]["baseURL"] = proxy_base
            self._temp_config_home = Path(tempfile.mkdtemp(prefix="opencode_proxy_"))
            temp_cfg_dir = self._temp_config_home / "opencode"
            temp_cfg_dir.mkdir(parents=True)
            (temp_cfg_dir / "opencode.json").write_text(
                json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            env["XDG_CONFIG_HOME"] = str(self._temp_config_home)

        # b) Also set OPENAI_BASE_URL for providers that read it from env.
        env["OPENAI_BASE_URL"] = f"{proxy_base}/v1"

        # Wait briefly for proxy to start
        time.sleep(0.3)

    def _post_message_with_retry(self, *, model: Any, text: str) -> Any:
        attempts = 1 + int(self._request_retry_attempts or 0)
        include_context = True
        last_err: OpenCodeRequestError | None = None
        recover_budget = int(self._session_recover_attempts or 0) if self._owns_local_server else 0
        recover_tries = 0

        for attempt in range(1, attempts + 1):
            try:
                return self._post_message(model=model, text=text, include_context=include_context)
            except OpenCodeRequestError as e:
                last_err = e
                # Some OpenCode builds may reject unknown fields; degrade gracefully.
                if include_context and self._context_length is not None and e.status in (400, 422):
                    include_context = False
                    try:
                        return self._post_message(model=model, text=text, include_context=False)
                    except OpenCodeRequestError as e2:
                        last_err = e2
                        e = e2

                transport_unavailable = False
                if e.status is None:
                    d = str(e.detail or "").strip().lower()
                    if d:
                        needles = (
                            "connection refused",
                            "failed to establish a new connection",
                            "connection reset",
                            "connection aborted",
                            "connection closed",
                            "remote end closed",
                            "network is unreachable",
                            "name or service not known",
                            "temporary failure in name resolution",
                            "timed out",
                            "timeout",
                        )
                        transport_unavailable = any(n in d for n in needles)

                if transport_unavailable and recover_tries < recover_budget:
                    recover_tries += 1
                    try:
                        recover_fn = getattr(self, "_recover_local_server_session", None)
                        if callable(recover_fn):
                            recover_fn(reason=e.detail)
                        else:
                            if not self._owns_local_server:
                                raise RuntimeError("session_recover_not_local_server")
                            username = (
                                str(getattr(self, "_server", None).username).strip()
                                if getattr(self, "_server", None) is not None
                                else "opencode"
                            ) or "opencode"
                            self._stop_local_server()
                            self._server = self._start_local_server(
                                repo=self._repo,
                                server_log_path=self._server_log_path,
                                username=username,
                                append_log=True,
                            )

                            deadline = time.time() + 60
                            last_health_err = ""
                            _rh_start = time.time()
                            _rh_dots = 0
                            print(f"    recovering server...", flush=True)
                            while time.time() < deadline:
                                try:
                                    self._request_json(
                                        "GET",
                                        "/global/health",
                                        body=None,
                                        require_auth=bool(self._server.password),
                                    )
                                    print(f"    server recovered ({time.time() - _rh_start:.1f}s)", flush=True)
                                    break
                                except Exception as health_exc:
                                    last_health_err = str(health_exc)
                                    _rh_dots += 1
                                    if _rh_dots % 25 == 0:
                                        print(f"    waiting for server... ({time.time() - _rh_start:.0f}s)", flush=True)
                                    time.sleep(0.2)
                            else:
                                raise RuntimeError(
                                    f"OpenCode server failed health check: {tail(last_health_err, 2000)}"
                                )

                            data = self._request_json(
                                "POST",
                                "/session",
                                body={"title": self._session_title},
                                require_auth=bool(self._server.password),
                            )
                            if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].strip():
                                self._session_id = data["id"]
                            elif (
                                isinstance(data, dict)
                                and isinstance(data.get("sessionID"), str)
                                and data["sessionID"].strip()
                            ):
                                self._session_id = data["sessionID"]
                            else:
                                raise RuntimeError(
                                    f"unexpected /session response: {tail(json.dumps(data, ensure_ascii=False), 2000)}"
                                )
                    except Exception as recover_exc:
                        last_err = OpenCodeRequestError(
                            method=e.method,
                            url=e.url,
                            status=e.status,
                            detail=f"{e.detail}; recover_failed: {tail(str(recover_exc), 1200)}",
                        )
                    else:
                        sleep_fn = getattr(self, "_sleep_session_recover_backoff", None)
                        if callable(sleep_fn):
                            sleep_fn(recover_idx=recover_tries)
                        else:
                            base = float(self._session_recover_backoff_seconds or 0.0)
                            if base > 0:
                                delay = min(30.0, base * (2 ** max(0, int(recover_tries) - 1)))
                                if delay > 0:
                                    time.sleep(delay)
                        continue

                should_retry_fn = getattr(self, "_should_retry_request_error", None)
                if callable(should_retry_fn):
                    should_retry = bool(should_retry_fn(e))
                elif e.status is None:
                    should_retry = True
                else:
                    try:
                        code = int(e.status)
                    except Exception:
                        should_retry = True
                    else:
                        should_retry = code in (408, 409, 425, 429) or code >= 500

                if attempt >= attempts or not should_retry:
                    raise

                sleep_fn = getattr(self, "_sleep_retry_backoff", None)
                if callable(sleep_fn):
                    sleep_fn(attempt_idx=attempt)
                else:
                    base = float(self._request_retry_backoff_seconds or 0.0)
                    if base > 0:
                        delay = min(30.0, base * (2 ** max(0, int(attempt) - 1)))
                        if delay > 0:
                            time.sleep(delay)

        if last_err is not None:
            raise last_err
        raise RuntimeError("opencode_retry_failed_without_error")

    def _post_message(self, *, model: Any, text: str, include_context: bool = True) -> Any:
        s = str(text or "")
        cap = self._max_prompt_chars
        if cap is None or cap <= 0 or len(s) <= cap:
            clipped_text = s
        elif cap < 128:
            clipped_text = s[-cap:]
        else:
            marker = "\n...[TRUNCATED_FOR_OPENCODE_CONTEXT]...\n"
            head = max(32, cap // 2)
            tail_keep = max(32, cap - head - len(marker))
            clipped_text = s[:head] + marker + s[-tail_keep:]
        body = {
            "agent": "build",
            "model": model,
            "parts": [{"type": "text", "text": clipped_text}],
        }
        if include_context and self._context_length is not None:
            # Best-effort: different OpenCode versions may ignore this field.
            body["contextLength"] = int(self._context_length)
        data = self._request_json(
            "POST",
            f"/session/{self._session_id}/message",
            body=body,
            require_auth=bool(self._server.password),
        )
        # Some OpenCode builds/transports may respond with 200 + empty body; treat it as a transient transport failure
        # so the caller can retry or recover the local session instead of silently returning "None".
        if data is None:
            url = f"{self._server.base_url}/session/{self._session_id}/message"
            raise OpenCodeRequestError(method="POST", url=url, status=None, detail="connection closed: empty_response_body")
        return data

    def _request_json(self, method: str, path: str, *, body: Any, require_auth: bool, timeout_seconds: float | None = None) -> Any:
        url = f"{self._server.base_url}{path}"
        headers = {"Accept": "application/json"}
        if require_auth and self._server.password:
            token = base64.b64encode(
                f"{self._server.username}:{self._server.password}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"

        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = Request(url, method=method, data=data, headers=headers)
        timeout = self._timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8", errors="replace"))
        except HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise OpenCodeRequestError(method=method, url=url, status=int(getattr(e, "code", 0) or 0), detail=tail(detail, 2000))
        except URLError as e:
            raise OpenCodeRequestError(method=method, url=url, status=None, detail=str(e))
        except (TimeoutError, socket.timeout) as e:
            raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"timeout: {e}")
        except OSError as e:
            raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"os_error: {e}")
