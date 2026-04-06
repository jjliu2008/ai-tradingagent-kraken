from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()


BASE_SEPOLIA_CHAIN_ID = 84532
DEFAULT_IDENTITY_REGISTRY = "0x8004A818BFB912233c491871b3d84c89A494BD9e"
DEFAULT_REPUTATION_REGISTRY = "0x8004B663056A597Dffe9eCcC1965A193B7388713"
DEFAULT_REGISTRATION_FILE = "agent_registration.json"
DEFAULT_EVIDENCE_DIR = "runtime/erc8004"

IDENTITY_REGISTRY_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"indexed": False, "internalType": "string", "name": "agentURI", "type": "string"},
            {"indexed": True, "internalType": "address", "name": "owner", "type": "address"},
        ],
        "name": "Registered",
        "type": "event",
    },
    {
        "inputs": [{"internalType": "string", "name": "agentURI", "type": "string"}],
        "name": "register",
        "outputs": [{"internalType": "uint256", "name": "agentId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "register",
        "outputs": [{"internalType": "uint256", "name": "agentId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"internalType": "string", "name": "newURI", "type": "string"},
        ],
        "name": "setAgentURI",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "agentId", "type": "uint256"}],
        "name": "getAgentWallet",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "name": "ownerOf",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "name": "tokenURI",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
]

REPUTATION_REGISTRY_ABI = [
    {
        "inputs": [],
        "name": "getIdentityRegistry",
        "outputs": [{"internalType": "address", "name": "identityRegistry", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"internalType": "int128", "name": "value", "type": "int128"},
            {"internalType": "uint8", "name": "valueDecimals", "type": "uint8"},
            {"internalType": "string", "name": "tag1", "type": "string"},
            {"internalType": "string", "name": "tag2", "type": "string"},
            {"internalType": "string", "name": "endpoint", "type": "string"},
            {"internalType": "string", "name": "feedbackURI", "type": "string"},
            {"internalType": "bytes32", "name": "feedbackHash", "type": "bytes32"},
        ],
        "name": "giveFeedback",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


@dataclass(frozen=True)
class ERC8004Config:
    enabled: bool
    rpc_url: str
    chain_id: int
    identity_registry: str
    reputation_registry: str
    private_key: str
    feedback_private_key: str
    agent_id: int | None
    dashboard_url: str
    registration_file: Path
    evidence_dir: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _maybe_import_web3() -> tuple[Any, Any, Any, Any]:
    try:
        from eth_account import Account
        from eth_account.messages import encode_typed_data
        from web3 import HTTPProvider, Web3
    except Exception:
        return None, None, None, None
    return Web3, HTTPProvider, Account, encode_typed_data


def load_config() -> ERC8004Config:
    rpc_url = os.environ.get("ERC8004_RPC_URL", "").strip()
    private_key = os.environ.get("ERC8004_PRIVATE_KEY", "").strip()
    feedback_private_key = os.environ.get("ERC8004_FEEDBACK_PRIVATE_KEY", "").strip()
    agent_id_raw = os.environ.get("ERC8004_AGENT_ID", "").strip()
    try:
        agent_id = int(agent_id_raw) if agent_id_raw else None
    except ValueError:
        agent_id = None
    return ERC8004Config(
        enabled=bool(rpc_url and private_key),
        rpc_url=rpc_url,
        chain_id=int(os.environ.get("ERC8004_CHAIN_ID", BASE_SEPOLIA_CHAIN_ID)),
        identity_registry=os.environ.get("ERC8004_IDENTITY_REGISTRY", DEFAULT_IDENTITY_REGISTRY),
        reputation_registry=os.environ.get("ERC8004_REPUTATION_REGISTRY", DEFAULT_REPUTATION_REGISTRY),
        private_key=private_key,
        feedback_private_key=feedback_private_key,
        agent_id=agent_id,
        dashboard_url=os.environ.get("ERC8004_DASHBOARD_URL", "").strip(),
        registration_file=Path(os.environ.get("ERC8004_REGISTRATION_FILE", DEFAULT_REGISTRATION_FILE)),
        evidence_dir=Path(os.environ.get("ERC8004_EVIDENCE_DIR", DEFAULT_EVIDENCE_DIR)),
    )


def _build_web3(config: ERC8004Config) -> Any:
    Web3, HTTPProvider, _, _ = _maybe_import_web3()
    if Web3 is None:
        raise RuntimeError("web3 is not installed. Install dependencies with `pip install -r requirements.txt`.")
    if not config.rpc_url:
        raise RuntimeError("ERC8004_RPC_URL is not configured.")
    return Web3(HTTPProvider(config.rpc_url, request_kwargs={"timeout": 8}))


def _account_from_key(private_key: str) -> tuple[str, Any]:
    _, _, Account, _ = _maybe_import_web3()
    if Account is None:
        raise RuntimeError("web3/eth-account is not installed.")
    account = Account.from_key(private_key)
    return account.address, account


def _contract(w3: Any, address: str, abi: list[dict[str, Any]]) -> Any:
    return w3.eth.contract(address=w3.to_checksum_address(address), abi=abi)


def _build_transaction(w3: Any, account: Any, call: Any, value: int = 0) -> dict[str, Any]:
    base = {
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "chainId": w3.eth.chain_id,
        "value": value,
    }
    try:
        base["gasPrice"] = w3.eth.gas_price
    except Exception:
        pass
    gas_estimate = call.estimate_gas(base)
    base["gas"] = int(gas_estimate * 1.2)
    return call.build_transaction(base)


def _send_transaction(w3: Any, account: Any, tx: dict[str, Any]) -> dict[str, Any]:
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return {
        "tx_hash": tx_hash.hex(),
        "block_number": receipt.blockNumber,
        "status": receipt.status,
        "gas_used": receipt.gasUsed,
        "receipt": receipt,
    }


def build_registration_payload(
    name: str = "Kraken Momentum Agent",
    description: str = "Autonomous 15m momentum-confluence trading agent with AI-powered risk guardrails",
    image: str = "",
    dashboard_url: str = "",
    wallet_address: str = "",
    strategy_name: str = "tc15_tighter_volume_cap",
    exchange: str = "kraken",
    agent_id: int | None = None,
    identity_registry: str = DEFAULT_IDENTITY_REGISTRY,
    chain_id: int = BASE_SEPOLIA_CHAIN_ID,
) -> dict[str, Any]:
    services: list[dict[str, Any]] = []
    if dashboard_url:
        services.append({"name": "web", "endpoint": dashboard_url})
    registrations: list[dict[str, Any]] = []
    if agent_id is not None:
        registrations.append(
            {
                "agentId": agent_id,
                "agentRegistry": f"eip155:{chain_id}:{identity_registry}",
            }
        )
    return {
        "type": "https://eips.ethereum.org/EIPS/eip-8004#registration-v1",
        "name": name,
        "description": description,
        "image": image,
        "services": services,
        "x402Support": False,
        "active": True,
        "registrations": registrations,
        "supportedTrust": ["reputation", "crypto-economic"],
        "capabilities": ["spot_trading", "risk_management", "market_analysis"],
        "walletAddress": wallet_address,
        "strategy": strategy_name,
        "exchange": exchange,
    }


def load_registration_payload(config: ERC8004Config) -> dict[str, Any]:
    if config.registration_file.exists():
        return json.loads(config.registration_file.read_text(encoding="utf-8"))
    address = ""
    if config.private_key:
        address, _ = _account_from_key(config.private_key)
    return build_registration_payload(
        dashboard_url=config.dashboard_url,
        wallet_address=address,
        agent_id=config.agent_id,
        identity_registry=config.identity_registry,
        chain_id=config.chain_id,
    )


def registration_data_uri(payload: dict[str, Any]) -> str:
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    encoded = base64.b64encode(compact.encode("utf-8")).decode("ascii")
    return f"data:application/json;base64,{encoded}"


def write_registration_file(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def get_identity_status(config: ERC8004Config | None = None) -> dict[str, Any]:
    config = config or load_config()
    status: dict[str, Any] = {
        "enabled": config.enabled,
        "chain_id": config.chain_id,
        "identity_registry": config.identity_registry,
        "reputation_registry": config.reputation_registry,
        "agent_id": config.agent_id,
        "dashboard_url": config.dashboard_url,
        "registration_file": str(config.registration_file),
        "evidence_dir": str(config.evidence_dir),
        "configured_at": _utc_now_iso(),
    }
    if not config.private_key:
        status["ready"] = False
        status["reason"] = "missing_private_key"
        return status
    try:
        address, _ = _account_from_key(config.private_key)
    except Exception as exc:
        status["ready"] = False
        status["reason"] = f"wallet_error:{exc}"
        return status

    status["wallet_address"] = address
    status["ready"] = bool(config.rpc_url and config.agent_id is not None)
    if not config.rpc_url or config.agent_id is None:
        status["reason"] = "missing_rpc_or_agent_id"
        return status

    try:
        w3 = _build_web3(config)
        identity = _contract(w3, config.identity_registry, IDENTITY_REGISTRY_ABI)
        owner = identity.functions.ownerOf(config.agent_id).call()
        wallet = identity.functions.getAgentWallet(config.agent_id).call()
        token_uri = identity.functions.tokenURI(config.agent_id).call()
        status.update(
            {
                "owner": owner,
                "agent_wallet": wallet,
                "token_uri": token_uri,
                "wallet_verified": wallet.lower() == address.lower() if wallet else False,
                "rpc_connected": True,
            }
        )
    except Exception as exc:
        status["rpc_connected"] = False
        status["reason"] = f"chain_query_failed:{exc}"
    return status


def register_agent(agent_uri: str | None = None, config: ERC8004Config | None = None) -> dict[str, Any]:
    config = config or load_config()
    if not config.enabled:
        raise RuntimeError("ERC-8004 registration requires ERC8004_RPC_URL and ERC8004_PRIVATE_KEY.")
    if agent_uri is None:
        payload = load_registration_payload(config)
        agent_uri = registration_data_uri(payload)

    w3 = _build_web3(config)
    _, account = _account_from_key(config.private_key)
    identity = _contract(w3, config.identity_registry, IDENTITY_REGISTRY_ABI)
    tx = _build_transaction(w3, account, identity.functions.register(agent_uri))
    sent = _send_transaction(w3, account, tx)
    try:
        logs = identity.events.Registered().process_receipt(sent["receipt"])
        agent_id = int(logs[0]["args"]["agentId"]) if logs else None
    except Exception:
        agent_id = None
    return {
        "action": "register_agent",
        "agent_id": agent_id,
        "agent_uri": agent_uri,
        "wallet_address": account.address,
        "tx_hash": sent["tx_hash"],
        "status": sent["status"],
        "block_number": sent["block_number"],
    }


def set_agent_uri(agent_id: int, agent_uri: str, config: ERC8004Config | None = None) -> dict[str, Any]:
    config = config or load_config()
    w3 = _build_web3(config)
    _, account = _account_from_key(config.private_key)
    identity = _contract(w3, config.identity_registry, IDENTITY_REGISTRY_ABI)
    tx = _build_transaction(w3, account, identity.functions.setAgentURI(int(agent_id), agent_uri))
    sent = _send_transaction(w3, account, tx)
    return {
        "action": "set_agent_uri",
        "agent_id": int(agent_id),
        "agent_uri": agent_uri,
        "tx_hash": sent["tx_hash"],
        "status": sent["status"],
        "block_number": sent["block_number"],
    }


def sign_trade_intent(trade_details: dict[str, Any], config: ERC8004Config | None = None) -> dict[str, Any]:
    config = config or load_config()
    if not config.private_key or config.agent_id is None:
        return {
            "enabled": False,
            "skipped": True,
            "reason": "missing_private_key_or_agent_id",
        }

    _, _, Account, encode_typed_data = _maybe_import_web3()
    if Account is None or encode_typed_data is None:
        return {
            "enabled": False,
            "skipped": True,
            "reason": "web3_not_installed",
        }

    address, account = _account_from_key(config.private_key)
    confidence_bps = int(round(float(trade_details.get("ai_confidence", 0.0)) * 10000))
    typed_data = {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "TradeIntent": [
                {"name": "agentId", "type": "uint256"},
                {"name": "pair", "type": "string"},
                {"name": "side", "type": "string"},
                {"name": "size", "type": "string"},
                {"name": "price", "type": "string"},
                {"name": "timestamp", "type": "uint256"},
                {"name": "guardrailsPassed", "type": "bool"},
                {"name": "aiConfidenceBps", "type": "uint256"},
                {"name": "riskSummary", "type": "string"},
                {"name": "strategy", "type": "string"},
            ],
        },
        "primaryType": "TradeIntent",
        "domain": {
            "name": "KrakenMomentumAgent",
            "version": "1",
            "chainId": config.chain_id,
            "verifyingContract": config.identity_registry,
        },
        "message": {
            "agentId": int(config.agent_id),
            "pair": str(trade_details.get("pair", "")),
            "side": str(trade_details.get("side", "")),
            "size": str(trade_details.get("size", "")),
            "price": str(trade_details.get("price", "")),
            "timestamp": int(trade_details.get("timestamp", 0)),
            "guardrailsPassed": bool(trade_details.get("guardrails_passed", False)),
            "aiConfidenceBps": confidence_bps,
            "riskSummary": str(trade_details.get("risk_summary", "")),
            "strategy": str(trade_details.get("strategy", "")),
        },
    }
    signable = encode_typed_data(full_message=typed_data)
    signed = Account.sign_message(signable, private_key=config.private_key)
    return {
        "enabled": True,
        "skipped": False,
        "agent_id": config.agent_id,
        "signer": address,
        "signature": signed.signature.hex(),
        "message_hash": signed.message_hash.hex(),
        "typed_data": typed_data,
        "intent": dict(typed_data["message"]),
    }


def _compact_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _keccak_hex(data: bytes) -> str:
    Web3, _, _, _ = _maybe_import_web3()
    if Web3 is None:
        raise RuntimeError("web3 is not installed.")
    return Web3.keccak(data).hex()


def _feedback_submitter_ready(config: ERC8004Config) -> tuple[bool, str, str]:
    if not config.feedback_private_key:
        return False, "", "missing_feedback_private_key"
    owner_address, _ = _account_from_key(config.private_key)
    feedback_address, _ = _account_from_key(config.feedback_private_key)
    if owner_address.lower() == feedback_address.lower():
        return False, feedback_address, "feedback_submitter_must_differ_from_agent_owner"
    return True, feedback_address, ""


def post_trade_feedback(trade_result: dict[str, Any], config: ERC8004Config | None = None) -> dict[str, Any]:
    config = config or load_config()
    if not (config.rpc_url and config.agent_id and config.private_key):
        return {
            "enabled": False,
            "posted": False,
            "reason": "missing_core_erc8004_config",
        }

    ready, feedback_address, reason = _feedback_submitter_ready(config)
    if not ready:
        return {
            "enabled": False,
            "posted": False,
            "reason": reason,
            "feedback_address": feedback_address,
        }

    pnl_pct = float(trade_result.get("pnl_pct", 0.0))
    evidence = {
        "agentRegistry": f"eip155:{config.chain_id}:{config.identity_registry}",
        "agentId": int(config.agent_id),
        "clientAddress": f"eip155:{config.chain_id}:{feedback_address}",
        "createdAt": _utc_now_iso(),
        "value": int(round(pnl_pct * 10000)),
        "valueDecimals": 2,
        "tag1": "tradingYield",
        "tag2": str(trade_result.get("reason", "")),
        "endpoint": f"kraken:{trade_result.get('pair', '')}",
        "trade": trade_result,
    }
    config.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = config.evidence_dir / f"feedback_{trade_result.get('pair','pair')}_{int(trade_result.get('exit_ts', 0) or 0)}.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    evidence_bytes = _compact_json_bytes(evidence)
    feedback_hash = _keccak_hex(evidence_bytes)
    feedback_uri = registration_data_uri(evidence)

    w3 = _build_web3(config)
    _, feedback_account = _account_from_key(config.feedback_private_key)
    reputation = _contract(w3, config.reputation_registry, REPUTATION_REGISTRY_ABI)
    tx = _build_transaction(
        w3,
        feedback_account,
        reputation.functions.giveFeedback(
            int(config.agent_id),
            int(evidence["value"]),
            int(evidence["valueDecimals"]),
            str(evidence["tag1"]),
            str(evidence["tag2"]),
            str(evidence["endpoint"]),
            feedback_uri,
            bytes.fromhex(feedback_hash.removeprefix("0x")),
        ),
    )
    sent = _send_transaction(w3, feedback_account, tx)
    return {
        "enabled": True,
        "posted": bool(sent["status"] == 1),
        "agent_id": config.agent_id,
        "feedback_address": feedback_account.address,
        "evidence_path": str(evidence_path),
        "feedback_uri": feedback_uri,
        "feedback_hash": feedback_hash,
        "tx_hash": sent["tx_hash"],
        "status": sent["status"],
        "block_number": sent["block_number"],
        "tag1": evidence["tag1"],
        "tag2": evidence["tag2"],
        "value": evidence["value"],
        "value_decimals": evidence["valueDecimals"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERC-8004 integration helper for the Kraken agent")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    sub.add_parser("wallet")
    sub.add_parser("register-agent")

    set_uri = sub.add_parser("set-agent-uri")
    set_uri.add_argument("--agent-id", type=int, required=True)
    set_uri.add_argument("--uri", required=True)

    render = sub.add_parser("render-registration")
    render.add_argument("--out", default=DEFAULT_REGISTRATION_FILE)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    if args.command == "status":
        print(json.dumps(get_identity_status(config), indent=2))
        return

    if args.command == "wallet":
        if not config.private_key:
            raise SystemExit("ERC8004_PRIVATE_KEY is not configured.")
        address, _ = _account_from_key(config.private_key)
        print(json.dumps({"wallet_address": address}, indent=2))
        return

    if args.command == "render-registration":
        payload = load_registration_payload(config)
        path = write_registration_file(Path(args.out), payload)
        print(json.dumps({"written": str(path)}, indent=2))
        return

    if args.command == "register-agent":
        result = register_agent(config=config)
        print(json.dumps(result, indent=2))
        return

    if args.command == "set-agent-uri":
        result = set_agent_uri(args.agent_id, args.uri, config=config)
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
