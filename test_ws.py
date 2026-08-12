import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        await ws.send(json.dumps({"mode": "rule_bot"}))
        for _ in range(5):
            msg = await asyncio.wait_for(ws.recv(), timeout=2)
            data = json.loads(msg)
            t = data.get("t", 0)
            p = data.get("p", [])
            e = data.get("e", [])
            b = len(data.get("b", []))
            print(f"Tick {t}: player={p}, enemy={e}, bullets={b}")
        print("Game WebSocket OK!")

asyncio.run(test())
