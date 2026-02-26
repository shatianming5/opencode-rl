"""Transparent LLM API proxy that logs streaming tokens to a file.

Architecture:
    OpenCode Server  ---->  This Proxy  ---->  Real LLM API
                              |
                              v
                         token log file  ---->  _LLMWaitMonitor tails this

Usage:
    python llm_proxy.py --port 8201 --upstream https://real-api/v1 --log /tmp/tokens.log

The proxy:
1. Forwards all requests to the upstream LLM API unchanged
2. For streaming responses (SSE), intercepts chunks and writes decoded tokens to the log
3. For non-streaming responses, forwards as-is
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError


_upstream: str = ""
_log_path: str = ""
_debug: bool = False


def _write_log(msg: str):
    try:
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(msg)
            f.flush()
    except Exception:
        pass


def _extract_token(chunk: dict) -> str:
    """Extract token content from an SSE chunk, trying multiple formats."""
    # Standard OpenAI format: choices[].delta.content
    for c in chunk.get("choices") or []:
        delta = c.get("delta") or {}
        content = delta.get("content")
        if content:
            return content
        # Some models put text in "text" instead of "content"
        text = delta.get("text")
        if text:
            return text
        # Reasoning/thinking content
        reasoning = delta.get("reasoning_content")
        if reasoning:
            return reasoning

    # Non-standard: top-level content
    if chunk.get("content"):
        return chunk["content"]

    # Anthropic-style: content[].text
    for block in chunk.get("content") or []:
        if isinstance(block, dict) and block.get("text"):
            return block["text"]

    return ""


def _parse_sse_line(text: str) -> str | None:
    """Parse an SSE data line, return the payload or None if not a data line."""
    # Standard: "data: {...}" or "data: [DONE]"
    if text.startswith("data: "):
        return text[6:]
    # No-space variant: "data:{...}"
    if text.startswith("data:"):
        return text[5:]
    return None


def _strip_encrypted(obj: dict | list, _depth: int = 0) -> bool:
    """Recursively strip ``encrypted_content`` fields from a request payload.

    OpenAI Codex models return encrypted reasoning tokens tied to a specific
    organization.  When the conversation history is sent back through a
    different org / Azure deployment, the server rejects them with
    ``invalid_encrypted_content``.  Removing these fields allows multi-turn
    conversations to work across different endpoints.

    Returns True if any field was removed.
    """
    if _depth > 20:
        return False
    changed = False
    if isinstance(obj, dict):
        if "encrypted_content" in obj:
            del obj["encrypted_content"]
            changed = True
        # Also strip reasoning items that only carry encrypted content
        # from Responses API "input" arrays.
        if obj.get("type") == "reasoning" and "encrypted_content" not in obj:
            # Already stripped above; mark for removal from parent list
            pass
        for v in obj.values():
            if isinstance(v, (dict, list)):
                changed |= _strip_encrypted(v, _depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                changed |= _strip_encrypted(item, _depth + 1)
        # Remove reasoning items that are now empty after stripping
        before = len(obj)
        obj[:] = [
            item for item in obj
            if not (isinstance(item, dict) and item.get("type") == "reasoning"
                    and "encrypted_content" not in item
                    and not item.get("summary"))
        ]
        if len(obj) != before:
            changed = True
    return changed


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward requests to upstream, intercept streaming responses."""

    def log_message(self, format, *args):
        # Suppress default access log to stderr
        pass

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len else b""

        # Build upstream request
        url = f"{_upstream}{self.path}"
        headers = {}
        for key in ("Content-Type", "Authorization", "Accept"):
            val = self.headers.get(key)
            if val:
                headers[key] = val
        headers["Content-Type"] = self.headers.get("Content-Type", "application/json")

        # Check if request asks for streaming
        is_stream = False
        try:
            req_json = json.loads(body)
            is_stream = req_json.get("stream", False)
            model = req_json.get("model", "?")
            # Strip encrypted_content from requests to avoid cross-org
            # decryption failures in multi-turn conversations.
            if _strip_encrypted(req_json):
                body = json.dumps(req_json).encode("utf-8")
                headers["Content-Type"] = "application/json"
        except Exception:
            model = "?"

        req = Request(url, data=body, headers=headers, method="POST")

        try:
            resp = urlopen(req, timeout=600)
        except HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())
            return
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy error: {e}".encode())
            return

        # Forward response headers
        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in ("transfer-encoding", "connection"):
                self.send_header(k, v)
        self.end_headers()

        if not is_stream:
            data = resp.read()
            self.wfile.write(data)
            return

        # Streaming: read SSE chunks, log tokens, forward to client
        _write_log(f"\n[{time.strftime('%H:%M:%S')}] >>> stream start model={model}\n")
        token_count = 0
        raw_lines = 0
        t0 = time.time()

        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                # Forward raw bytes to OpenCode server
                self.wfile.write(line)
                self.wfile.flush()

                # Parse SSE data lines for token content
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                payload = _parse_sse_line(text)
                if payload is None:
                    # Not an SSE data line — log for debugging on first few
                    if _debug and raw_lines < 3:
                        _write_log(f"[DBG non-data] {text[:200]}\n")
                    continue

                raw_lines += 1
                if payload.strip() == "[DONE]":
                    continue

                try:
                    chunk = json.loads(payload)
                    content = _extract_token(chunk)
                    if content:
                        token_count += 1
                        _write_log(content)
                    elif _debug and raw_lines <= 3:
                        # Log first few unparseable chunks for debugging
                        _write_log(f"[DBG no-content] {payload[:300]}\n")
                except json.JSONDecodeError:
                    if _debug and raw_lines <= 3:
                        _write_log(f"[DBG bad-json] {payload[:300]}\n")
        except Exception as e:
            _write_log(f"\n[ERR] {e}\n")

        elapsed = time.time() - t0
        _write_log(f"\n[{time.strftime('%H:%M:%S')}] <<< stream end {token_count} tokens {elapsed:.1f}s (data_lines={raw_lines})\n")

    def do_GET(self):
        # Forward GET requests (e.g. /v1/models)
        url = f"{_upstream}{self.path}"
        headers = {}
        for key in ("Authorization", "Accept"):
            val = self.headers.get(key)
            if val:
                headers[key] = val
        req = Request(url, headers=headers, method="GET")
        try:
            resp = urlopen(req, timeout=30)
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.read())
        except HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy error: {e}".encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", type=str, required=True)
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--debug", action="store_true", help="Log raw SSE data for debugging")
    args = parser.parse_args()

    global _upstream, _log_path, _debug
    _upstream = args.upstream.rstrip("/")
    _log_path = args.log
    _debug = args.debug

    server = HTTPServer(("127.0.0.1", args.port), ProxyHandler)
    print(f"LLM proxy listening on 127.0.0.1:{args.port} -> {_upstream}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
