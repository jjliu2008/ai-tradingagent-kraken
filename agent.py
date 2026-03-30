"""
Kraken AI Trading Agent
Entry point for the LabLab AI Trading Agents Hackathon.
"""
import argparse
import json
import os

import anthropic
from dotenv import load_dotenv

import kraken_client as kraken

load_dotenv()

SYSTEM_PROMPT = """You are an AI trading agent for the Kraken exchange.
You analyze market data and make trading decisions using paper trading (no real money).

Available tools let you check prices, portfolio status, and place paper trades.
Always reason step by step: check the market, assess the portfolio, then decide.

Current mode: {mode}
Trading pair: {pair}
"""


def build_tools() -> list[dict]:
    return [
        {
            "name": "get_ticker",
            "description": "Get current price and 24h stats for a trading pair.",
            "input_schema": {
                "type": "object",
                "properties": {"pair": {"type": "string", "description": "e.g. XBTUSD"}},
                "required": ["pair"],
            },
        },
        {
            "name": "get_portfolio_status",
            "description": "Get current paper portfolio value, P&L, and trade count.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "paper_buy",
            "description": "Buy an asset in paper trading mode.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pair": {"type": "string"},
                    "volume": {"type": "number", "description": "Amount to buy"},
                },
                "required": ["pair", "volume"],
            },
        },
        {
            "name": "paper_sell",
            "description": "Sell an asset in paper trading mode.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pair": {"type": "string"},
                    "volume": {"type": "number", "description": "Amount to sell"},
                },
                "required": ["pair", "volume"],
            },
        },
    ]


def handle_tool(name: str, inputs: dict) -> str:
    try:
        if name == "get_ticker":
            result = kraken.ticker(inputs["pair"])
        elif name == "get_portfolio_status":
            result = kraken.paper_status()
        elif name == "paper_buy":
            result = kraken.paper_buy(inputs["pair"], inputs["volume"])
        elif name == "paper_sell":
            result = kraken.paper_sell(inputs["pair"], inputs["volume"])
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_agent(pair: str = "XBTUSD", mode: str = "paper"):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"Starting trading agent | pair={pair} mode={mode}")
    print("-" * 50)

    messages = [
        {
            "role": "user",
            "content": (
                f"Analyze the current market for {pair} and decide whether to buy, sell, "
                f"or hold. Check the current price and our portfolio status first, "
                f"then make a decision with your reasoning."
            ),
        }
    ]

    tools = build_tools()
    system = SYSTEM_PROMPT.format(mode=mode, pair=pair)

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Collect assistant message
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAgent decision:\n{block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool] {block.name}({block.input})")
                    result = handle_tool(block.name, block.input)
                    print(f"  [result] {result[:200]}")
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result}
                    )
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kraken AI Trading Agent")
    parser.add_argument("--pair", default="XBTUSD", help="Trading pair (default: XBTUSD)")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    args = parser.parse_args()

    if args.mode == "live":
        confirm = input("WARNING: Live mode uses real money. Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            exit(0)

    run_agent(pair=args.pair, mode=args.mode)
