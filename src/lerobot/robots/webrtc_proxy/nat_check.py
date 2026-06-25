#!/usr/bin/env python3
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

"""Detect whether this host is behind a SYMMETRIC NAT (which breaks WebRTC hole punching).

Method: from ONE UDP socket, query several different STUN servers and compare the
reflexive (mapped) PORT each one reports.
  - all ports identical  -> endpoint-independent mapping -> CONE NAT  (hole punch OK)
  - ports differ          -> endpoint-dependent  mapping -> SYMMETRIC NAT (need TURN/SFU)

Only needs a STUN Binding response (no RFC3489 CHANGE-REQUEST), so any STUN server works.
Pure stdlib — runnable anywhere (Mac, a pod, an ECS) with no dependencies.

    python3 nat_check.py

See NAT_TRAVERSAL_NOTES.md for the full write-up and a real-world result.
"""

import os
import socket
import struct

# Domestic-first (Google STUN may be unreachable from mainland China).
STUN_SERVERS = [
    ("stun.qq.com", 3478),
    ("stun.miwifi.com", 3478),
    ("stun.chat.bilibili.com", 3478),
    ("stun.l.google.com", 19302),
]
MAGIC_COOKIE = 0x2112A442


def stun_mapped(host: str, port: int, sock: socket.socket) -> str | None:
    """Send a STUN Binding Request and return the reflexive 'ip:port', or None."""
    sock.sendto(struct.pack(">HHI", 1, 0, MAGIC_COOKIE) + os.urandom(12), (host, port))
    sock.settimeout(3)
    data, _ = sock.recvfrom(2048)
    _, mlen, _ = struct.unpack(">HHI", data[:8])
    i = 20
    while i < 20 + mlen:
        atype, alen = struct.unpack(">HH", data[i : i + 4])
        val = data[i + 4 : i + 4 + alen]
        i += 4 + alen + ((4 - alen % 4) % 4)  # attrs are 4-byte aligned
        if atype in (0x0001, 0x0020):  # MAPPED-ADDRESS / XOR-MAPPED-ADDRESS
            p = struct.unpack(">H", val[2:4])[0]
            ip = bytearray(val[4:8])
            if atype == 0x0020:  # XOR-MAPPED: un-xor with the magic cookie
                p ^= MAGIC_COOKIE >> 16
                for k, c in enumerate(struct.pack(">I", MAGIC_COOKIE)):
                    ip[k] ^= c
            return ".".join(map(str, ip)) + ":" + str(p)
    return None


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    print("localport", sock.getsockname()[1])
    ports = set()
    for host, port in STUN_SERVERS:
        try:
            mapped = stun_mapped(host, port, sock)
            print(host, "->", mapped)
            if mapped:
                ports.add(mapped.split(":")[1])
        except Exception as e:  # noqa: BLE001 - best-effort probe
            print(host, "ERR", type(e).__name__)
    print("MAPPED_PORTS", sorted(ports))
    if len(ports) > 1:
        verdict = "SYMMETRIC (hole punch FAILS -> need TURN/LiveKit, or a public-IP peer)"
    elif ports:
        verdict = "CONE (hole punch OK)"
    else:
        verdict = "NO-STUN-REACHABLE"
    print("VERDICT", verdict)


if __name__ == "__main__":
    main()
