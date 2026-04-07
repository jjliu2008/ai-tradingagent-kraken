from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pair_diagnostics as diag


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the current research queue from pair diagnostics.")
    parser.add_argument("--queue-csv", default="results/research_queue.csv")
    args = parser.parse_args()

    payload = diag.build_diagnostics()
    pairs = payload["pairs"]

    queue = []
    for item in pairs:
        queue.append(
            {
                "pair": item["pair"],
                "priority": item["priority"],
                "verdict": item["verdict"],
                "current_strategy": item["current_strategy"] or "",
                "diagnosis": item["diagnosis"],
                "next_hypotheses": ",".join(item["next_hypotheses"]),
            }
        )

    queue = sorted(queue, key=lambda row: (-int(row["priority"]), row["pair"]))
    out = Path(args.queue_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(queue[0].keys()))
        writer.writeheader()
        writer.writerows(queue)

    print("Research priority queue:")
    for row in queue:
        print(
            f"  {row['pair']:<8} {row['verdict']:<9} p={row['priority']:<3} "
            f"{row['current_strategy'] or 'none':<24} {row['diagnosis']}"
        )
    print(f"\nQueue -> {out}")


if __name__ == "__main__":
    main()
