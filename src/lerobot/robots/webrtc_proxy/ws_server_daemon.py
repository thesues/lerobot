# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebSocket DATA relay: the middle-man for the ``ws`` transport backend.

Unlike ``signaling_server.py`` (which forwards only SDP for the *aiortc* backend and
never touches media), this daemon is the full **data plane**: it relays BOTH the
control/state/action messages (TEXT) and the video frames (BINARY) between the Mac
daemon and the cloud controller. Neither peer ever talks WebRTC — each just dials OUT
to this one public endpoint and pushes/pulls everything over its WebSocket.

Why this exists: when BOTH peers sit behind symmetric NAT (the common cloud-Pod ↔
home-Mac case), aiortc P2P hole-punching is impossible and there is no public peer to
anchor on. Running this daemon on the *controller* side (public via an L7 ingress /
ALB) turns the link into two ordinary outbound client connections — the one topology
that always works (see ``NAT_TRAVERSAL_NOTES.md``).

    Mac daemon (role=robot) ──ws push state+frames / recv actions──▶  THIS daemon  ◀──ws──  controller
                                                                    (public, ALB)         (role=controller, local)

A session has two roles — ``robot`` and ``controller``. Each connects to
``GET /ws?session=<id>&role=<role>`` and the daemon forwards every message to the
*other* role. It is a near-dumb forwarder:

- TEXT (hello / channel messages / bye): forwarded; buffered (bounded) for a peer that
  has not joined yet, so the ``hello`` handshake and any early control survive.
- BINARY (one video frame = 8-byte big-endian seq + JPEG): forwarded; only the LATEST
  is held for an absent peer (stale frames are dropped, never a backlog).
- When one side drops, the survivor gets ``{"kind":"bye"}`` so the daemon can recycle.

Run standalone (on the controller side / in the cloud Pod, exposed publicly):
    python -m lerobot.robots.webrtc_proxy.ws_server_daemon --port 8765 \
        --auth-token "$SIGNALING_AUTH_TOKEN"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import socket
from collections import defaultdict, deque

from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)

_OTHER = {"robot": "controller", "controller": "robot"}

# Bound the per-peer TEXT backlog for a not-yet-joined peer. The publisher only streams
# after the hello handshake (so this normally holds just the hello + a little control),
# but cap it so a misbehaving peer can never grow it without bound.
_MAX_TEXT_BACKLOG = 256


def _split_env(name: str) -> list[str]:
    return [u.strip() for u in os.environ.get(name, "").split(",") if u.strip()]


class WsDataRelay:
    """Forwards TEXT + BINARY between the two roles of each session, buffering for late joiners.

    A public daemon must gate the door: with ``auth_token`` set, every WebSocket must
    present a matching token (``Authorization: Bearer <token>`` header, or ``?token=``
    query as a fallback) or it is rejected before pairing. NOTE: a single shared token
    only stops random scanners — it does NOT isolate sessions (anyone holding it can join
    any session_id). Multi-tenant deployments need per-session signed tokens.
    """

    def __init__(self, auth_token: str | None = None) -> None:
        self._auth_token = auth_token
        self._peers: dict[tuple[str, str], web.WebSocketResponse] = {}
        # Buffered-for-late-joiner state, keyed by (session, role-of-the-recipient).
        self._text_backlog: dict[tuple[str, str], deque[str]] = defaultdict(
            lambda: deque(maxlen=_MAX_TEXT_BACKLOG)
        )
        self._last_frame: dict[tuple[str, str], bytes] = {}

    def _authorized(self, request: web.Request) -> bool:
        if not self._auth_token:
            return True
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else request.query.get("token", "")
        return secrets.compare_digest(token, self._auth_token)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return web.Response(status=401, text="unauthorized")
        session = request.query.get("session", "default")
        role = request.query.get("role", "")
        if role not in _OTHER:
            return web.Response(status=400, text="role must be 'robot' or 'controller'")
        other = _OTHER[role]

        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)  # 0 = no cap (video frames)
        await ws.prepare(request)
        self._peers[(session, role)] = ws
        logger.info("ws-relay: %s joined session %s", role, session)

        # Flush anything buffered for me before I joined: TEXT backlog first (hello +
        # early control, in order), then the single freshest frame if one is waiting.
        for text in self._text_backlog.pop((session, role), ()):
            await ws.send_str(text)
        frame = self._last_frame.pop((session, role), None)
        if frame is not None:
            await ws.send_bytes(frame)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._forward_text(session, other, msg.data)
                elif msg.type == WSMsgType.BINARY:
                    self._forward_frame(session, other, msg.data)
                    await self._flush_frame(session, other)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            self._peers.pop((session, role), None)
            peer = self._peers.get((session, other))
            if peer is not None and not peer.closed:
                await peer.send_json({"kind": "bye"})
            # Drop any buffered media for this dead pairing; stale once the peer is gone.
            self._last_frame.pop((session, other), None)
            logger.info("ws-relay: %s left session %s", role, session)
        return ws

    async def _forward_text(self, session: str, other: str, data: str) -> None:
        peer = self._peers.get((session, other))
        if peer is not None and not peer.closed:
            await peer.send_str(data)
        else:
            self._text_backlog[(session, other)].append(data)

    def _forward_frame(self, session: str, other: str, data: bytes) -> None:
        # Always overwrite: only the freshest frame is ever worth delivering (a slow or
        # absent peer must never accumulate a video backlog).
        self._last_frame[(session, other)] = data

    async def _flush_frame(self, session: str, other: str) -> None:
        peer = self._peers.get((session, other))
        if peer is not None and not peer.closed:
            frame = self._last_frame.pop((session, other), None)
            if frame is not None:
                await peer.send_bytes(frame)


async def _health(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


def make_app(auth_token: str | None = None) -> web.Application:
    relay = WsDataRelay(auth_token=auth_token)
    app = web.Application()
    app.router.add_get("/ws", relay.handle)
    app.router.add_get("/", _health)  # 200 for an HTTP health check (ALB / k8s probe)
    return app


async def start_relay(
    host: str = "127.0.0.1", port: int = 0, auth_token: str | None = None
) -> tuple[web.AppRunner, int]:
    """Start the data relay on ``host:port`` (port 0 => ephemeral). Returns (runner, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    actual_port = sock.getsockname()[1]
    runner = web.AppRunner(make_app(auth_token=auth_token))
    await runner.setup()
    await web.SockSite(runner, sock).start()
    logger.info("ws data relay listening on ws://%s:%d/ws (auth=%s)", host, actual_port, bool(auth_token))
    return runner, actual_port


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRTCProxyRobot ws-transport DATA relay (media + control)")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 (server binds all ifaces by design)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("SIGNALING_AUTH_TOKEN"),
        help="shared token every peer must present (Authorization: Bearer ...). Public relays should set it.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    async def _run() -> None:
        runner, _ = await start_relay(args.host, args.port, auth_token=args.auth_token)
        try:
            await asyncio.Event().wait()  # run forever
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
