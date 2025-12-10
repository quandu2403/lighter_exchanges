"""
Dual-exchange monitoring bot for Lighter and Aster Futures.

The script keeps a running view of balances, open orders, and positions on
both exchanges. It notifies (via the console by default) when:
- an order is opened or closed on either exchange
- a position exists without any attached TP/SL style order
- a position's unrealized PnL changes
- a new position starts or an existing position closes

Environment variables expected for private endpoints:
- LIGHTER_BASE_URL: Lighter API base URL (defaults to https://mainnet.zklighter.elliot.ai)
- LIGHTER_ACCOUNT_INDEX: Account index to monitor
- LIGHTER_AUTH_HEADER: Optional auth token for the account endpoints
- ASTER_USER: EVM address that owns the Aster API key
- ASTER_SIGNER: Signer address registered with the Aster API key
- ASTER_PRIVATE_KEY: Private key for signing requests (0x-prefixed)
- ASTER_HOST: Optional override for the Aster futures API host

Because the code polls authenticated endpoints it is best run inside an
isolated environment and with small polling intervals until you are happy
with the output. The default notifier simply prints to stdout, but you can
swap it for a custom class (for example, a Slack or email sender) by
implementing a ``notify`` method.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import aiohttp
from eth_abi import encode
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

import lighter

TP_SL_ORDER_TYPES = {
    "TAKE_PROFIT",
    "TAKE_PROFIT_MARKET",
    "STOP",
    "STOP_MARKET",
    "STOP_LOSS",
    "STOP_LOSS_LIMIT",
    "TAKE_PROFIT_LIMIT",
    "TAKE_PROFIT_MARKET",
    "take-profit",
    "take-profit-limit",
    "stop-loss",
    "stop-loss-limit",
}


@dataclass
class ExchangeSnapshot:
    balances: Dict[str, float] = field(default_factory=dict)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)

    def position_keys(self) -> Sequence[str]:
        return [f"{p.get('symbol')}:{p.get('positionSide', p.get('sign'))}" for p in self.positions]

    def order_ids(self) -> Sequence[str]:
        def _extract(order: Dict[str, Any]) -> str:
            for key in ("order_id", "orderId", "clientOrderId", "client_order_id"):
                if key in order:
                    return str(order[key])
            return json.dumps(order, sort_keys=True)

        return [_extract(order) for order in self.orders]


class ConsoleNotifier:
    def notify(self, message: str) -> None:
        logging.info(message)


class AsterFuturesClient:
    def __init__(
        self,
        user: str,
        signer: str,
        private_key: str,
        host: str = "https://fapi.asterdex.com",
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.user = Web3.to_checksum_address(user)
        self.signer = Web3.to_checksum_address(signer)
        self.private_key = private_key
        self.host = host.rstrip("/")
        self.session_provided = session is not None
        self.session = session or aiohttp.ClientSession()

    async def close(self) -> None:
        if not self.session_provided:
            await self.session.close()

    async def balance(self) -> ExchangeSnapshot:
        data = await self._request("/fapi/v3/balance")
        balances = {row.get("asset", "USDT"): float(row.get("balance", 0)) for row in data}
        positions = await self._request("/fapi/v3/positionRisk")
        orders = await self._request("/fapi/v3/openOrders")
        return ExchangeSnapshot(balances=balances, positions=positions, orders=orders)

    async def _request(self, path: str, method: str = "GET", params: Optional[Dict[str, Any]] = None) -> Any:
        payload = self._signed_params(params or {})
        url = f"{self.host}{path}"
        async with self.session.request(method, url, params=payload if method == "GET" else None, data=None if method == "GET" else payload) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _signed_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        base_params: Dict[str, Any] = {key: value for key, value in params.items() if value is not None}
        base_params.setdefault("recvWindow", 50000)
        base_params.setdefault("timestamp", int(round(time.time() * 1000)))
        nonce = math.trunc(time.time() * 1_000_000)

        prepared = self._trim_dict(dict(base_params))
        json_str = json.dumps(prepared, sort_keys=True).replace(" ", "").replace("'", "\"")
        encoded = encode(["string", "address", "address", "uint256"], [json_str, self.user, self.signer, nonce])
        keccak_hex = Web3.keccak(encoded).hex()

        signable_msg = encode_defunct(hexstr=keccak_hex)
        signed_message = Account.sign_message(signable_message=signable_msg, private_key=self.private_key)

        base_params.update(
            {
                "nonce": nonce,
                "user": self.user,
                "signer": self.signer,
                "signature": "0x" + signed_message.signature.hex(),
            }
        )
        return base_params

    def _trim_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        for key in list(payload.keys()):
            value = payload[key]
            if isinstance(value, list):
                payload[key] = json.dumps([self._trim_dict(item) if isinstance(item, dict) else str(item) for item in value])
                continue
            if isinstance(value, dict):
                payload[key] = json.dumps(self._trim_dict(value))
                continue
            payload[key] = str(value)
        return payload


class LighterAccountClient:
    def __init__(
        self,
        account_index: int,
        host: str,
        auth_header: Optional[str] = None,
    ) -> None:
        self.account_index = account_index
        self.auth_header = auth_header
        self.api_client = lighter.ApiClient(configuration=lighter.Configuration(host=host))
        self.account_api = lighter.AccountApi(self.api_client)
        self.order_api = lighter.OrderApi(self.api_client)

    async def close(self) -> None:
        await self.api_client.close()

    async def snapshot(self) -> ExchangeSnapshot:
        account = await self.account_api.account(by="index", value=str(self.account_index))
        active_positions = [p for p in account.positions if float(p.position) != 0]
        orders: List[Dict[str, Any]] = []
        for position in active_positions:
            active = await self.order_api.account_active_orders(
                account_index=self.account_index,
                market_id=position.market_id,
                auth=self.auth_header,
            )
            orders.extend([order.to_dict() for order in active.orders])

        balances = {asset.symbol: float(asset.balance) for asset in account.assets}
        return ExchangeSnapshot(
            balances=balances,
            positions=[p.to_dict() for p in active_positions],
            orders=orders,
        )


class DualExchangeTracker:
    def __init__(
        self,
        lighter_client: LighterAccountClient,
        aster_client: AsterFuturesClient,
        notifier: ConsoleNotifier,
        poll_interval: float = 10.0,
    ) -> None:
        self.lighter_client = lighter_client
        self.aster_client = aster_client
        self.notifier = notifier
        self.poll_interval = poll_interval
        self.lighter_state = ExchangeSnapshot()
        self.aster_state = ExchangeSnapshot()

    async def run(self) -> None:
        while True:
            await asyncio.gather(self._tick_lighter(), self._tick_aster())
            await asyncio.sleep(self.poll_interval)

    async def _tick_lighter(self) -> None:
        new_state = await self.lighter_client.snapshot()
        self._report_changes("Lighter", self.lighter_state, new_state)
        self.lighter_state = new_state

    async def _tick_aster(self) -> None:
        new_state = await self.aster_client.balance()
        self._report_changes("Aster", self.aster_state, new_state)
        self.aster_state = new_state

    def _report_changes(self, name: str, previous: ExchangeSnapshot, current: ExchangeSnapshot) -> None:
        self._report_balances(name, previous.balances, current.balances)
        self._report_orders(name, previous.orders, current.orders)
        self._report_positions(name, previous.positions, current.positions, current.orders)

    def _report_balances(self, name: str, old: Dict[str, float], new: Dict[str, float]) -> None:
        for asset, balance in new.items():
            if asset not in old:
                self.notifier.notify(f"[{name}] New asset balance detected: {asset} {balance}")
            elif abs(old[asset] - balance) > 1e-8:
                delta = balance - old[asset]
                self.notifier.notify(f"[{name}] Balance change for {asset}: {balance} (Δ {delta:+f})")

    def _report_orders(self, name: str, old: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> None:
        old_ids = set(self._order_id(order) for order in old)
        new_ids = set(self._order_id(order) for order in new)

        for order in new:
            order_id = self._order_id(order)
            if order_id not in old_ids:
                self.notifier.notify(f"[{name}] New order {order_id} on {order.get('symbol')}: type={order.get('type')} side={order.get('side', order.get('is_ask'))}")

        for order in old:
            order_id = self._order_id(order)
            if order_id not in new_ids:
                self.notifier.notify(f"[{name}] Order closed or canceled: {order_id} on {order.get('symbol')}")

    def _report_positions(
        self,
        name: str,
        old_positions: List[Dict[str, Any]],
        new_positions: List[Dict[str, Any]],
        current_orders: List[Dict[str, Any]],
    ) -> None:
        old_keys = {self._position_key(pos) for pos in old_positions if abs(float(pos.get("position", 0))) > 0}
        for pos in new_positions:
            key = self._position_key(pos)
            qty = float(pos.get("position", pos.get("positionAmt", 0)))
            unrealized = self._unrealized_pnl(pos)
            if key not in old_keys:
                self.notifier.notify(f"[{name}] New position {key} size={qty} unrealizedPnL={unrealized}")
            else:
                old_match = next((p for p in old_positions if self._position_key(p) == key), None)
                if old_match is not None and abs(self._unrealized_pnl(pos) - self._unrealized_pnl(old_match)) > 1e-6:
                    self.notifier.notify(
                        f"[{name}] Updated position {key} unrealizedPnL {self._unrealized_pnl(old_match)} -> {unrealized}"
                    )

            if qty != 0 and not self._has_tp_sl(pos, current_orders):
                self.notifier.notify(f"[{name}] Position {key} has no TP/SL order attached")

        new_keys = {self._position_key(pos) for pos in new_positions if abs(float(pos.get("position", 0))) > 0}
        for pos in old_positions:
            key = self._position_key(pos)
            if key not in new_keys:
                self.notifier.notify(f"[{name}] Position closed: {key}")

    def _has_tp_sl(self, position: Dict[str, Any], orders: Iterable[Dict[str, Any]]) -> bool:
        symbol = position.get("symbol")
        for order in orders:
            if order.get("symbol") != symbol:
                continue
            order_type = str(order.get("type", "")).upper()
            if order_type in TP_SL_ORDER_TYPES:
                return True
        return False

    def _order_id(self, order: Dict[str, Any]) -> str:
        for key in ("order_id", "orderId", "clientOrderId", "client_order_id"):
            if key in order:
                return str(order[key])
        return json.dumps(order, sort_keys=True)

    def _position_key(self, position: Dict[str, Any]) -> str:
        side = position.get("positionSide", position.get("sign", ""))
        return f"{position.get('symbol', position.get('market_id'))}:{side}"

    def _unrealized_pnl(self, position: Dict[str, Any]) -> float:
        for key in ("unrealizedPnl", "unrealized_pnl", "unRealizedProfit"):
            if key in position:
                try:
                    return float(position.get(key, 0))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0


async def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Lighter and Aster accounts simultaneously.")
    parser.add_argument("--lighter-base-url", default=os.environ.get("LIGHTER_BASE_URL", "https://mainnet.zklighter.elliot.ai"))
    parser.add_argument("--lighter-account", type=int, default=int(os.environ.get("LIGHTER_ACCOUNT_INDEX", 0)))
    parser.add_argument("--lighter-auth", default=os.environ.get("LIGHTER_AUTH_HEADER"))
    parser.add_argument("--aster-user", default=os.environ.get("ASTER_USER"))
    parser.add_argument("--aster-signer", default=os.environ.get("ASTER_SIGNER"))
    parser.add_argument("--aster-key", default=os.environ.get("ASTER_PRIVATE_KEY"))
    parser.add_argument("--aster-host", default=os.environ.get("ASTER_HOST", "https://fapi.asterdex.com"))
    parser.add_argument("--interval", type=float, default=10.0)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    lighter_client = LighterAccountClient(args.lighter_account, args.lighter_base_url, args.lighter_auth)
    aster_client = AsterFuturesClient(args.aster_user, args.aster_signer, args.aster_key, host=args.aster_host)
    tracker = DualExchangeTracker(lighter_client, aster_client, ConsoleNotifier(), poll_interval=args.interval)

    try:
        await tracker.run()
    finally:
        await asyncio.gather(lighter_client.close(), aster_client.close())


if __name__ == "__main__":
    asyncio.run(main())
