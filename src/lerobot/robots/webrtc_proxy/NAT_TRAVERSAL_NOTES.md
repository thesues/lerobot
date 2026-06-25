# WebRTC 媒体直连与 NAT 穿透实测笔记

> 场景:`WebRTCProxyRobot`(云端 controller)通过 WebRTC 连接家里 Mac 上的真实机械臂(`mac_daemon`)。
> 信令走 relay(`signaling_server.py`),**媒体(视频帧 + 关节动作)走两端直连 P2P**。
> 本文记录:为什么「云端 Pod ↔ 家里 Mac」这种拓扑下 aiortc 直连**穿不过 NAT**,以及什么拓扑才能通。

---

## 1. 结论先行(TL;DR)

| 拓扑 | 媒体能否直连(aiortc) | 原因 |
|---|---|---|
| **同一局域网**(LAN) | ✅ 通 | 用 host candidate 直接互通,不经 NAT |
| **一端公网 + 一端 NAT**(如 ECS+EIP ↔ 家里 Mac) | ✅ 通 | NAT 端**主动拨出**连固定公网端点,出方向 NAT 永远放行 |
| **两端都在 NAT 后,且其中含对称 NAT** | ❌ 不通 | 打洞(hole punching)失败,只能 TURN/SFU 中继 |

**实测**:家里 Mac 与云 VKE Pod **两端都是对称(Symmetric)NAT** → aiortc 直连物理上不可能 → 必须 **TURN 或 LiveKit(SFU)** 中继,或把 controller 放到**有公网 IP 的 ECS** 上。

---

## 2. 背景:WebRTC 怎么穿 NAT(ICE 三类候选)

WebRTC 用 ICE 收集三类「候选地址」,按优先级尝试直连:

| 候选类型 | 怎么连 | 能穿什么 |
|---|---|---|
| **host** | 直接用本地 IP | 仅同一局域网 |
| **srflx(STUN)** | STUN 探到公网映射 `IP:port`,两端互相打洞 | **锥形(Cone)NAT** 能穿;**对称 NAT 穿不了** |
| **relay(TURN)** | 媒体经 TURN 公网中继转发 | 任意 NAT(兜底,永远成功) |

> 关键:**STUN 只负责「发现」公网映射,媒体仍是 P2P;真正的中继是 TURN。**
> 本项目的 relay(`signaling_server.py`)**只发 STUN、不发 TURN**(`IceConfig` 只接受 `--stun-url`),
> 所以 aiortc 后端只有上表前两层,缺第三层兜底 —— 一旦打洞失败就直接连不上。

---

## 3. 为什么「对称 NAT」打不通

打洞靠 srflx:每端往 STUN 服务器发包,拿到自己的公网映射 `IP:X`,交换后互相往对方的 `IP:X` 发包。
能不能成,取决于 **NAT 的「映射是否与目的地无关」**:

- **锥形 NAT(Cone)**:不管发给谁,都复用同一个外部端口 `X`。
  → 你跟 STUN 拿到的 `X` 对端能用 → **打洞成功**。
  
- **对称 NAT(Symmetric)**:**对每个不同目的地分配不同的外部端口**。
  → 你跟 STUN 服务器(目的地 A)拿到端口 `X`,但对端是目的地 B → NAT 给的是端口 `Y ≠ X`
  → 对端按 `X` 发根本进不来 → **打洞失败**。
  
  ![image-20260624212923373](/Users/dongmao.zhang/Library/Application Support/typora-user-images/image-20260624212923373.png)

![image-20260624212940131](/Users/dongmao.zhang/Library/Application Support/typora-user-images/image-20260624212940131.png)



打洞结果矩阵:

| 组合 | 结果 |
|---|---|
| 锥形 ↔ 锥形 | ✅ 通 |
| 锥形 ↔ 对称 | ⚠️ 时好时坏(需端口预测,不可靠) |
| **对称 ↔ 对称** | ❌ 基本不通 |

**只要有一端是对称 NAT,打洞就基本废。** 这是 NAT 的物理限制,STUN 救不了,只有 TURN/SFU 中继能救。

### 为什么云厂商节点几乎都是对称 NAT
云厂商的 **NAT 网关 / EIP 出网**要把成百上千实例复用到少数公网 IP 上,只能**按「每条流」分配外部端口** → 天然是对称型。
所以**云 Pod 出网那一侧基本注定是对称 NAT**。

### 为什么「一端公网」就能通
对称 NAT 只破坏「**双方互相打洞**」;它**不影响「主动拨出连一个已知的固定公网端点**」。
所以家里 Mac(对称)→ ECS 公网 IP:port,是普通的「客户端连服务器」,Mac 发起、NAT 建映射、ECS 回到这个映射 → **稳通**。
ICE 里这表现为:Mac 朝 ECS 的 host/公网候选发连通性检查 → 成功;反方向(ECS 朝 Mac 的 srflx)失败也无所谓,有一条通的就行。

---

## 4. 测试方法:用 STUN 判断是不是对称 NAT

原理:**从同一个 UDP socket,依次问多个不同的 STUN 服务器,比较它们各自看到的「映射端口」。**
- 端口**都一样** → 映射与目的地无关 → **锥形**(可打洞)。
- 端口**不一样** → 映射随目的地变 → **对称**(穿不了)。

> 这个方法只需要 STUN 的 binding 响应,**不依赖** RFC3489 的 CHANGE-REQUEST(很多现代 STUN 服务器不支持那个),所以任意 STUN 服务器都能用。

纯标准库脚本(无需任何依赖),`nat_check.py`:

```python
import socket, struct, os

def stun_mapped(host, port, sock):
    # STUN Binding Request: type=0x0001, len=0, magic cookie, 12-byte txid
    sock.sendto(struct.pack('>HHI', 1, 0, 0x2112A442) + os.urandom(12), (host, port))
    sock.settimeout(3)
    data, _ = sock.recvfrom(2048)
    _, mlen, _ = struct.unpack('>HHI', data[:8])
    i = 20
    while i < 20 + mlen:
        atype, alen = struct.unpack('>HH', data[i:i+4])
        val = data[i+4:i+4+alen]
        i += 4 + alen + ((4 - alen % 4) % 4)
        if atype in (0x0001, 0x0020):            # MAPPED-ADDRESS / XOR-MAPPED-ADDRESS
            p = struct.unpack('>H', val[2:4])[0]
            ip = bytearray(val[4:8])
            if atype == 0x0020:                  # XOR-MAPPED: 反异或 magic cookie
                p ^= 0x2112
                for k, c in enumerate(b'\x21\x12\xa4\x42'):
                    ip[k] ^= c
            return '.'.join(map(str, ip)) + ':' + str(p)

# 多个 STUN 服务器(域内优先;Google 的国内可能不通)
servers = [('stun.qq.com', 3478), ('stun.miwifi.com', 3478),
           ('stun.chat.bilibili.com', 3478), ('stun.l.google.com', 19302)]

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('', 0))
print('localport', s.getsockname()[1])
ports = set()
for h, p in servers:
    try:
        m = stun_mapped(h, p, s)
        print(h, '->', m)
        if m:
            ports.add(m.split(':')[1])
    except Exception as e:
        print(h, 'ERR', type(e).__name__)
print('MAPPED_PORTS', sorted(ports))
print('VERDICT',
      'SYMMETRIC (穿不了, 需 TURN/LiveKit)' if len(ports) > 1
      else ('CONE (可打洞)' if ports else 'NO-STUN-REACHABLE'))
```

运行:
```bash
# 本机(Mac)
python3 nat_check.py

# 容器内(无需装包,纯标准库)。可用 base64 避免多行粘贴问题:
#   base64 -i nat_check.py | tr -d '\n'  # 在本机生成
#   echo <BASE64> | base64 -d | python3   # 在 Pod 里执行
```

---

## 5. 实测结果(2026-06-25,北京)

**家里 Mac(住宅宽带):**

```
stun.miwifi.com        -> 73.59.114.97:50234
stun.chat.bilibili.com -> 73.59.114.97:61212
stun.l.google.com      -> 71.18.96.11:54345
MAPPED_PORTS ['50234', '54345', '61212']     # 端口全不同
VERDICT SYMMETRIC
```

**云端 VKE Pod(火山 VKE,出网经 NAT 网关 / EIP):**
```
stun.miwifi.com        -> 115.191.16.13:36885
stun.chat.bilibili.com -> 115.191.16.13:60749
stun.l.google.com      -> 115.191.16.13:28280
MAPPED_PORTS ['28280', '36885', '60749']     # 同一公网 IP,端口全不同
VERDICT SYMMETRIC
```

**判定:两端都是对称 NAT → aiortc 直连(srflx 打洞)物理上不可能。** 与第 3 节的矩阵一致。

---

## 6. 可行方案

按「是否需要中继 / 是否依赖外部服务」排序:

1. **同局域网**:直接通(host candidate),无需任何配置。仅适用于两端在同一内网。

2. **controller 放公网 ECS(EIP)+ aiortc**:
   - 一台**普通 ECS(带 EIP,非 VKE 节点)** 上同时跑 relay + controller;
   - 家里 Mac 拨出连它 → 信令 + 媒体都通(只有一端 NAT,见第 3 节)。
   - 优点:一台机器、不依赖 LiveKit、不依赖 VKE;媒体直连(低延迟)。
   - ⚠️ 不要用 **VKE 节点**绑 EIP:节点本身走 NAT 网关出网,绑 EIP 会和默认路由冲突(入向走 EIP、回包走 NAT 网关,源 IP 对不上 → 连接被重置),实测不稳。

3. **LiveKit(SFU)/ TURN 中继**:
   - 两端都**主动拨出**连公网的 LiveKit/TURN → 媒体经它中转 → 任意 NAT(含对称)都穿。
   - 镜像换 `lerobot[webrtc-livekit]`,两端 `--transport livekit`;或自建 coturn 并让 relay 下发 TURN 地址。
   - 优点:VKE 这种「两端都 NAT」的拓扑唯一可行的直连替代;媒体经中继(略增延迟)。

---

## 7. 一句话总结

- **媒体能不能直连,看「有没有一端不在对称 NAT 后」**:
  有公网端 / 同局域网 → 能直连;两端都对称 NAT → 不能,必须中继。
- 云厂商节点(NAT 网关 / EIP)**几乎都是对称 NAT**,家用宽带**也可能是对称**(本次实测就是)。
- 所以「云 Pod ↔ 家里 Mac」**别指望 aiortc 直连**;要么把 controller 放公网 ECS(单端公网),要么上 LiveKit/TURN。



8. ​	
