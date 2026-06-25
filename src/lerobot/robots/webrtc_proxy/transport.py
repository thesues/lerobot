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

"""Pluggable transport layer.

The proxy's logic (capture loop, alignment, watchdog, control-plane RPC, the Robot
API) is transport-agnostic; it only needs:

- named **data channels** (configurable reliability) to send/receive small JSON
  (state, action, control), and
- a one-way **video stream** carrying frames tagged with a capture ``seq``.

``Transport`` is that contract. ``AiortcTransport`` implements it with aiortc
(WebRTC P2P + DataChannels, the default, self-contained backend). A different
backend — e.g. a LiveKit SFU for cross-public-net / scale — can implement the same
interface without touching ``CaptureAgent`` / ``_ProxyEndpoint`` / ``WebRTCProxyRobot``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from fractions import Fraction

import numpy as np

from .protocol import VIDEO_CLOCK_RATE, VIDEO_PTS_PER_SEQ, channel_kwargs
from .signaling import Signaling

logger = logging.getLogger(__name__)


def make_transport(
    backend: str,
    *,
    role: str,
    channels: dict[str, bool],
    ice_servers: list[str | dict] | None = None,
    livekit_url: str | None = None,
    livekit_token: str | None = None,
    ws_url: str | None = None,
    ws_session: str | None = None,
    ws_token: str | None = None,
    ws_codec: str = "jpeg",
    ws_jpeg_quality: int = 80,
    ws_bitrate: int = 2_000_000,
    ws_hwaccel: bool = False,
) -> "Transport":
    """Build a transport for ``backend`` ("aiortc" | "livekit" | "ws").

    Both ends of a session MUST use the same backend (an aiortc P2P peer, a LiveKit room
    and a ws-relay peer don't interoperate). "aiortc" is the default, self-contained P2P
    backend. "livekit" (EXPERIMENTAL scaffold, see ``transport_livekit.py``) routes via a
    LiveKit SFU and needs ``livekit_url`` + ``livekit_token``. "ws" relays BOTH media and
    control through a ``ws_server_daemon`` (``ws_url`` + ``ws_session``, optional
    ``ws_token``) — the direct-style backend that works when both peers are behind
    symmetric NAT, since each side merely dials OUT to the public daemon (see
    ``NAT_TRAVERSAL_NOTES.md``). The ws video codec is ``ws_codec`` ("jpeg" intra-frame,
    default; or "h264" inter-frame via PyAV — both ends must match).
    """
    if backend == "aiortc":
        return AiortcTransport(role=role, channels=channels, ice_servers=ice_servers)
    if backend == "livekit":
        if not livekit_url or not livekit_token:
            raise ValueError("transport_backend='livekit' requires livekit_url and livekit_token")
        from .transport_livekit import LiveKitTransport

        return LiveKitTransport(role=role, channels=channels, url=livekit_url, token=livekit_token)
    if backend == "ws":
        if not ws_url or not ws_session:
            raise ValueError("transport_backend='ws' requires ws_url and ws_session")
        return WsRelayTransport(
            role=role,
            channels=channels,
            url=ws_url,
            session=ws_session,
            token=ws_token,
            codec=ws_codec,
            jpeg_quality=ws_jpeg_quality,
            bitrate=ws_bitrate,
            hwaccel=ws_hwaccel,
        )
    raise ValueError(f"unknown transport backend {backend!r} (expected 'aiortc', 'livekit' or 'ws')")


class Channel(ABC):
    """A named message pipe. ``send`` is best-effort (drops if not open)."""

    @abstractmethod
    def send(self, data: str) -> None: ...

    @abstractmethod
    def on_message(self, callback: Callable[[str], None]) -> None: ...

    @property
    @abstractmethod
    def is_open(self) -> bool: ...


class Transport(ABC):
    """Bidirectional transport: named data channels + one video stream.

    Roles: the ``"publisher"`` side offers + sends video; the ``"subscriber"`` side
    answers + receives video. Data channels are bidirectional regardless of role.
    """

    def __init__(self) -> None:
        self.connected = asyncio.Event()
        self.closed = asyncio.Event()

    @abstractmethod
    async def open(self, signaling: Signaling) -> None:
        """Establish the connection (exchange SDP via ``signaling``) and wire channels."""

    @abstractmethod
    def channel(self, label: str) -> Channel:
        """Return the channel handle for ``label`` (created/expected at construction)."""

    @abstractmethod
    def send_frame(self, seq: int, img: np.ndarray) -> None:
        """Publish one video frame tagged with its capture ``seq`` (publisher only)."""

    @abstractmethod
    def set_frame_handler(self, callback: Callable[[int, np.ndarray], None]) -> None:
        """Register ``callback(seq, rgb_ndarray)`` for received frames (subscriber only)."""

    async def wait_closed(self) -> None:
        await self.closed.wait()

    @abstractmethod
    async def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# aiortc backend
# --------------------------------------------------------------------------- #
class _AiortcChannel(Channel):
    """Wraps an aiortc RTCDataChannel, tolerating "registered before it exists"."""

    def __init__(self) -> None:
        self._ch = None
        self._pending_cb: Callable[[str], None] | None = None

    def bind(self, ch) -> None:  # noqa: ANN001 (RTCDataChannel)
        self._ch = ch
        if self._pending_cb is not None:
            ch.on("message", self._pending_cb)

    def send(self, data: str) -> None:
        if self._ch is not None and self._ch.readyState == "open":
            self._ch.send(data)

    def on_message(self, callback: Callable[[str], None]) -> None:
        if self._ch is not None:
            self._ch.on("message", callback)
        else:
            self._pending_cb = callback

    @property
    def is_open(self) -> bool:
        return self._ch is not None and self._ch.readyState == "open"


class _PublisherTrack:
    """An aiortc MediaStreamTrack fed by ``send_frame``; seq rides the frame pts."""

    kind = "video"

    def __init__(self) -> None:
        from aiortc import MediaStreamTrack

        # Build the actual track lazily to avoid importing aiortc at module import.
        class _Track(MediaStreamTrack):
            kind = "video"

            def __init__(self) -> None:
                super().__init__()
                self._q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(maxsize=2)

            def push(self, seq: int, img: np.ndarray) -> None:
                if self._q.full():
                    _ = self._q.get_nowait()  # drop oldest; don't block the capture clock
                self._q.put_nowait((seq, img))

            async def recv(self):
                from av import VideoFrame

                seq, img = await self._q.get()
                frame = VideoFrame.from_ndarray(img, format="rgb24")
                frame.pts = seq * VIDEO_PTS_PER_SEQ
                frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
                return frame

        self.track = _Track()

    def push(self, seq: int, img: np.ndarray) -> None:
        self.track.push(seq, img)


class AiortcTransport(Transport):
    """Default backend: WebRTC P2P (media track + DataChannels) over aiortc."""

    def __init__(
        self,
        *,
        role: str,  # "publisher" (offers + sends video) | "subscriber" (answers + recvs video)
        channels: dict[str, bool],  # label -> reliable (reliability honoured by the publisher/offerer)
        # Each entry is a STUN url string or a dict {"urls", ...}. aiortc is direct-UDP:
        # STUN gives a server-reflexive candidate for cross-NAT direct P2P. Static config;
        # the signaling relay also hands out STUN at open(). (The dict form accepts
        # username/credential as a generic escape hatch, but the media-relay path is the
        # LiveKit backend, not a TURN server under aiortc — see DESIGN §11.1.)
        ice_servers: list[str | dict] | None = None,
    ) -> None:
        super().__init__()
        if role not in ("publisher", "subscriber"):
            raise ValueError(f"role must be 'publisher' or 'subscriber', got {role!r}")
        self.role = role
        self._channel_specs = dict(channels)
        self._ice_cfg: list[str | dict] = list(ice_servers or [])
        # The RTCPeerConnection is built in open(), once the signaling relay has had a
        # chance to hand us STUN servers. aiortc fixes iceServers at construction, so we
        # can't build it earlier.
        self.pc = None
        self._channels: dict[str, _AiortcChannel] = {label: _AiortcChannel() for label in channels}
        self._pub = _PublisherTrack() if role == "publisher" else None
        self._frame_cb: Callable[[int, np.ndarray], None] | None = None

    @staticmethod
    def _to_ice_server(cfg: str | dict):  # noqa: ANN205 (RTCIceServer)
        """Coerce a config entry (url string or {urls,username,credential}) to RTCIceServer."""
        from aiortc import RTCIceServer

        if isinstance(cfg, str):
            return RTCIceServer(urls=cfg)
        return RTCIceServer(
            urls=cfg["urls"], username=cfg.get("username"), credential=cfg.get("credential")
        )

    def _register(self) -> None:
        @self.pc.on("connectionstatechange")
        async def _on_state() -> None:
            state = self.pc.connectionState
            logger.info("transport connectionState=%s", state)
            if state == "connected":
                self.connected.set()
            elif state in ("failed", "closed", "disconnected"):
                self.closed.set()

        if self.role == "subscriber":

            @self.pc.on("datachannel")
            def _on_channel(ch):  # noqa: ANN001
                if ch.label in self._channels:
                    self._channels[ch.label].bind(ch)

            @self.pc.on("track")
            def _on_track(track):  # noqa: ANN001
                asyncio.ensure_future(self._consume(track))

    async def _consume(self, track) -> None:  # noqa: ANN001
        while True:
            try:
                frame = await track.recv()
            except Exception:
                logger.info("transport: video track ended")
                return
            seconds = float(frame.pts) * float(frame.time_base)
            seq = round(seconds * VIDEO_CLOCK_RATE / VIDEO_PTS_PER_SEQ)
            if self._frame_cb is not None:
                self._frame_cb(seq, frame.to_ndarray(format="rgb24"))

    async def open(self, signaling: Signaling) -> None:
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

        await signaling.open()
        # Merge static (config) ICE servers with any the relay handed us on connect
        # (e.g. TURN with freshly-minted ephemeral credentials). Built now because aiortc
        # fixes iceServers at RTCPeerConnection construction.
        ice_cfg = self._ice_cfg + list(getattr(signaling, "ice_servers", None) or [])
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[self._to_ice_server(c) for c in ice_cfg])
        )
        self._register()
        if self.role == "publisher":
            for label, reliable in self._channel_specs.items():
                self._channels[label].bind(self.pc.createDataChannel(label, **channel_kwargs(reliable)))
            if self._pub is not None:
                self.pc.addTrack(self._pub.track)
            await self.pc.setLocalDescription(await self.pc.createOffer())
            await signaling.send(self.pc.localDescription)
            answer = await signaling.recv()
            if not isinstance(answer, RTCSessionDescription):
                raise RuntimeError(f"publisher expected an SDP answer, got {type(answer)!r}")
            await self.pc.setRemoteDescription(answer)
        else:
            offer = await signaling.recv()
            if not isinstance(offer, RTCSessionDescription):
                raise RuntimeError(f"subscriber expected an SDP offer, got {type(offer)!r}")
            await self.pc.setRemoteDescription(offer)
            await self.pc.setLocalDescription(await self.pc.createAnswer())
            await signaling.send(self.pc.localDescription)

    def channel(self, label: str) -> Channel:
        return self._channels[label]

    def send_frame(self, seq: int, img: np.ndarray) -> None:
        if self._pub is not None:
            self._pub.push(seq, img)

    def set_frame_handler(self, callback: Callable[[int, np.ndarray], None]) -> None:
        self._frame_cb = callback

    async def close(self) -> None:
        self.closed.set()
        await self.pc.close()


# --------------------------------------------------------------------------- #
# ws backend (media + control relayed through ws_server_daemon)
# --------------------------------------------------------------------------- #
# Frame wire format on the BINARY channel: 8-byte big-endian capture seq, then the
# encoded payload — a whole JPEG ("jpeg" codec, every frame independently decodable) or
# one H.264 access-unit packet ("h264" codec, inter-frame). JPEG/H.264 both round-trip
# the RGB array regardless of channel-order labelling, so no BGR/RGB swap is needed.
_FRAME_SEQ = struct.Struct(">Q")

# When the H.264 send backlog blows past this (a stalled socket), drop the WHOLE backlog
# and force the next frame to be a keyframe — a partial inter-frame stream is undecodable,
# so we resync at the next IDR rather than dribble corrupt packets.
_H264_MAX_BACKLOG = 90


class _WsChannel(Channel):
    """A named pipe over the shared relay WebSocket. Messages ride a ``{"kind":"ch"}``
    envelope so the one socket multiplexes state/action/control. Best-effort: ``send``
    drops until the peer handshake completes (``connected``)."""

    def __init__(self, label: str, transport: "WsRelayTransport") -> None:
        self._label = label
        self._t = transport
        self._cb: Callable[[str], None] | None = None

    def send(self, data: str) -> None:
        if self._t.connected.is_set():
            self._t._enqueue_text({"kind": "ch", "label": self._label, "data": data})

    def on_message(self, callback: Callable[[str], None]) -> None:
        self._cb = callback

    def _dispatch(self, data: str) -> None:
        if self._cb is not None:
            self._cb(data)

    @property
    def is_open(self) -> bool:
        return self._t.connected.is_set()


class WsRelayTransport(Transport):
    """Relay backend: media + control multiplexed over ONE WebSocket to ``ws_server_daemon``.

    Both peers dial OUT to the same public daemon (``role`` -> robot/controller) and the
    daemon forwards everything between them. Frames go on the BINARY channel (JPEG, drop
    -oldest = freshest-wins), small JSON on TEXT. Presence is a one-shot ``hello``
    handshake: receiving the peer's hello sets ``connected`` (the daemon buffers it for a
    late joiner). When the daemon reports the peer left (``bye``) or the socket drops,
    ``closed`` fires so the Mac daemon recycles for the next session.

    Video codec (``codec``, both ends must match):
      - "jpeg" (default): intra-frame, every frame independently decodable. The writer
        holds only the freshest unsent frame (drop-stale) — simple, but ~10-20 Mbps.
      - "h264": inter-frame via PyAV (libav). ~3-10x smaller, at the cost of a strict
        ordered packet stream (can't drop a single frame), so packets queue FIFO and a
        blown backlog resyncs at the next forced keyframe. The publisher only starts
        encoding once the peer is present, so the stream always opens on an IDR — a fresh
        subscriber can decode from the first packet with no extra keyframe signalling.

    Trade-off vs aiortc/livekit: one ordered TCP socket means video and control share
    head-of-line. Acceptable for the symmetric-NAT case where direct P2P is impossible.
    """

    def __init__(
        self,
        *,
        role: str,  # "publisher" (robot, sends video) | "subscriber" (controller, recvs video)
        channels: dict[str, bool],  # label -> reliable (ignored: TCP is always reliable+ordered)
        url: str,  # ws://host:port/ws of the ws_server_daemon
        session: str,
        token: str | None = None,
        codec: str = "jpeg",  # "jpeg" (intra) | "h264" (inter, PyAV)
        jpeg_quality: int = 80,
        bitrate: int = 2_000_000,  # h264 target bitrate (bits/s)
        hwaccel: bool = False,  # h264 publisher: use the platform hw encoder (e.g. videotoolbox)
    ) -> None:
        super().__init__()
        if role not in ("publisher", "subscriber"):
            raise ValueError(f"role must be 'publisher' or 'subscriber', got {role!r}")
        if codec not in ("jpeg", "h264"):
            raise ValueError(f"ws codec must be 'jpeg' or 'h264', got {codec!r}")
        self.role = role
        self._relay_role = "robot" if role == "publisher" else "controller"
        sep = "&" if "?" in url else "?"
        self._url = f"{url}{sep}session={session}&role={self._relay_role}"
        self._token = token
        self._codec = codec
        self._jpeg_quality = int(jpeg_quality)
        self._bitrate = int(bitrate)
        self._hwaccel = bool(hwaccel)
        self._channels: dict[str, _WsChannel] = {label: _WsChannel(label, self) for label in channels}
        self._frame_cb: Callable[[int, np.ndarray], None] | None = None
        self._client = None
        self._ws = None
        self._out_text: list[dict] = []  # pending TEXT envelopes (FIFO)
        # jpeg: hold only the freshest unsent frame. h264: an ordered FIFO of packets that
        # must NOT be reordered or partially dropped (inter-frame dependency).
        self._out_frame: bytes | None = None
        self._out_packets: deque[bytes] = deque()
        self._enc = None  # lazy PyAV encoder (publisher, h264)
        self._enc_wh: tuple[int, int] | None = None
        self._dec = None  # lazy PyAV decoder (subscriber, h264)
        self._force_idr = False
        self._wake = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def open(self, signaling: Signaling | None = None) -> None:
        # The ws backend does its own signaling over the relay socket; the SDP `signaling`
        # arg (used by aiortc) is ignored.
        import aiohttp

        self._client = aiohttp.ClientSession()
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        self._ws = await self._client.ws_connect(self._url, headers=headers, max_msg_size=0)
        # Announce presence so the peer's `connected` fires; the daemon buffers this until
        # the peer joins. Sent directly (not via a channel) so it is NOT gated on connected.
        await self._ws.send_str(json.dumps({"kind": "hello", "role": self._relay_role}))
        self._tasks = [
            asyncio.ensure_future(self._reader()),
            asyncio.ensure_future(self._writer()),
        ]

    def _enqueue_text(self, envelope: dict) -> None:
        self._out_text.append(envelope)
        self._wake.set()

    async def _reader(self) -> None:
        import aiohttp

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._on_text(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self._on_frame(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ws transport: reader error")
        finally:
            self.closed.set()
            self._wake.set()  # let the writer notice closure and exit

    def _on_text(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            logger.exception("ws transport: bad TEXT message")
            return
        kind = data.get("kind")
        if kind == "hello":
            self.connected.set()
        elif kind == "ch":
            ch = self._channels.get(data.get("label"))
            if ch is not None:
                ch._dispatch(data.get("data", ""))
        elif kind == "bye":
            self.closed.set()
            self._wake.set()
        # ignore anything else

    def _on_frame(self, payload: bytes) -> None:
        if self._frame_cb is None or len(payload) < _FRAME_SEQ.size:
            return
        seq = _FRAME_SEQ.unpack(payload[: _FRAME_SEQ.size])[0]
        body = payload[_FRAME_SEQ.size :]
        try:
            if self._codec == "h264":
                for img in self._decode_h264(body):
                    self._frame_cb(seq, img)
            else:
                import cv2

                img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self._frame_cb(seq, img)
        except Exception:
            logger.exception("ws transport: frame decode error")

    async def _writer(self) -> None:
        try:
            while not self.closed.is_set():
                await self._wake.wait()
                self._wake.clear()
                # Drain control/state/action first (small, latency-sensitive), then video.
                while self._out_text and not self.closed.is_set():
                    await self._ws.send_str(json.dumps(self._out_text.pop(0), separators=(",", ":")))
                if self._codec == "h264":
                    # Ordered packet stream — send all in FIFO order, never reorder/drop one.
                    while self._out_packets and not self.closed.is_set():
                        await self._ws.send_bytes(self._out_packets.popleft())
                else:
                    frame = self._out_frame  # freshest unsent JPEG (older ones overwritten)
                    if frame is not None and not self.closed.is_set():
                        self._out_frame = None
                        await self._ws.send_bytes(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ws transport: writer error")
            self.closed.set()

    def channel(self, label: str) -> Channel:
        return self._channels[label]

    def send_frame(self, seq: int, img: np.ndarray) -> None:
        if not self.connected.is_set():
            return  # don't stream until the peer is present (avoids relay-side buffering)
        try:
            if self._codec == "h264":
                self._encode_h264(seq, img)
            else:
                import cv2

                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
                if ok:
                    self._out_frame = _FRAME_SEQ.pack(seq) + buf.tobytes()  # freshest wins
                    self._wake.set()
        except Exception:
            logger.exception("ws transport: frame encode error")

    # ----- h264 (PyAV / libav) codec ---------------------------------------
    def _make_encoder(self, width: int, height: int):  # noqa: ANN202 (av.CodecContext)
        import av

        name = "h264_videotoolbox" if self._hwaccel else "libx264"
        cc = av.CodecContext.create(name, "w")
        cc.width = width
        cc.height = height
        cc.pix_fmt = "yuv420p"
        cc.bit_rate = self._bitrate
        cc.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        if name == "libx264":
            # Low-latency teleop: emit each frame immediately, no B-frame reordering.
            cc.options = {"tune": "zerolatency", "preset": "ultrafast"}
        return cc

    def _encode_h264(self, seq: int, img: np.ndarray) -> None:
        import av

        wh = (int(img.shape[1]), int(img.shape[0]))
        # (Re)create the encoder on first frame, a resolution change, or a forced resync —
        # a brand-new encoder's first packet is always an IDR keyframe.
        if self._enc is None or self._enc_wh != wh or self._force_idr:
            self._enc = self._make_encoder(*wh)
            self._enc_wh = wh
            self._force_idr = False
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24").reformat(
            format="yuv420p"
        )
        frame.pts = seq * VIDEO_PTS_PER_SEQ
        frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        for pkt in self._enc.encode(frame):
            self._out_packets.append(_FRAME_SEQ.pack(seq) + bytes(pkt))
        if len(self._out_packets) > _H264_MAX_BACKLOG:
            # Socket can't keep up: a partial inter-frame stream is undecodable, so drop the
            # whole backlog and resync the decoder at the next IDR (forced on next encode).
            logger.warning("ws transport: h264 backlog overflow -> drop + resync (IDR)")
            self._out_packets.clear()
            self._force_idr = True
        self._wake.set()

    def _decode_h264(self, body: bytes) -> list[np.ndarray]:
        import av

        if self._dec is None:
            self._dec = av.CodecContext.create("h264", "r")
        # Packets before the first IDR decode to nothing; our stream opens on an IDR.
        frames = self._dec.decode(av.packet.Packet(body))
        return [f.to_ndarray(format="rgb24") for f in frames]

    def set_frame_handler(self, callback: Callable[[int, np.ndarray], None]) -> None:
        self._frame_cb = callback

    async def close(self) -> None:
        self.closed.set()
        self._wake.set()
        for t in self._tasks:
            t.cancel()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.close()
