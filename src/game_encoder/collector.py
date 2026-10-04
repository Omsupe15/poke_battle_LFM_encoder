import argparse
import asyncio
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
)
from game_encoder.recording_player import RecordingPlayer
from game_encoder.server_manager import is_server_running, start_local_showdown


def run_worker_battles(
    worker_id: int,
    total_battles: int,
    battle_format: str,
    output_shard_path: str,
    max_concurrent_battles: int = 10,
    heuristics_ratio: float = 0.7,
) -> dict:
    """
    Subprocess worker that executes battles against an opponent mix
    (70% SimpleHeuristicsPlayer, 30% MaxBasePowerPlayer) and flushes
    recorded state-action transitions to a shard file.
    """
    async def _async_worker():
        worker_suffix = f"{worker_id}{uuid.uuid4().hex[:6]}"
        rec_acc = AccountConfiguration(f"Rec{worker_suffix}", None)
        opp_acc_sh = AccountConfiguration(f"OppSH{worker_suffix}", None)
        opp_acc_mb = AccountConfiguration(f"OppMB{worker_suffix}", None)

        recorder = RecordingPlayer(
            account_configuration=rec_acc,
            server_configuration=LocalhostServerConfiguration,
            battle_format=battle_format,
            max_concurrent_battles=max_concurrent_battles,
        )

        n_sh = int(total_battles * heuristics_ratio)
        n_mb = total_battles - n_sh

        start_time = time.time()

        # Phase 1: 70% against SimpleHeuristicsPlayer
        if n_sh > 0:
            opp_sh = SimpleHeuristicsPlayer(
                account_configuration=opp_acc_sh,
                server_configuration=LocalhostServerConfiguration,
                battle_format=battle_format,
                max_concurrent_battles=max_concurrent_battles,
            )
            await recorder.battle_against(opp_sh, n_battles=n_sh)

        # Phase 2: 30% against MaxBasePowerPlayer
        if n_mb > 0:
            opp_mb = MaxBasePowerPlayer(
                account_configuration=opp_acc_mb,
                server_configuration=LocalhostServerConfiguration,
                battle_format=battle_format,
                max_concurrent_battles=max_concurrent_battles,
            )
            await recorder.battle_against(opp_mb, n_battles=n_mb)

        elapsed = time.time() - start_time
        turns_recorded = recorder.save_trajectories(output_shard_path)

        return {
            "worker_id": worker_id,
            "battles": total_battles,
            "won_battles": recorder.n_won_battles,
            "finished_battles": recorder.n_finished_battles,
            "turns_recorded": turns_recorded,
            "elapsed_seconds": elapsed,
        }

    return asyncio.run(_async_worker())


def collect_dataset(
    total_battles: int = 100,
    num_workers: int = 4,
    battle_format: str = "gen9randombattle",
    output_file: str = "data/lfm_poke_train_50k.jsonl",
    max_concurrent_battles: int = 10,
    heuristics_ratio: float = 0.7,
) -> None:
    """
    Coordinates multi-process data collection across workers and merges output shards.
    """
    output_dir = os.path.dirname(os.path.abspath(output_file))
    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    # Ensure local server is online
    if not is_server_running(port=8000):
        print("[INFO] Pokémon Showdown server not detected on port 8000. Launching...")
        start_local_showdown(port=8000)

    battles_per_worker = total_battles // num_workers
    remainder = total_battles % num_workers

    worker_plan = []
    for i in range(num_workers):
        worker_battles = battles_per_worker + (1 if i < remainder else 0)
        if worker_battles > 0:
            shard_path = os.path.join(shards_dir, f"worker_{i}.jsonl")
            worker_plan.append((i, worker_battles, shard_path))

    print(
        f"[START] Commencing collection of {total_battles} battles across "
        f"{len(worker_plan)} worker(s) in format '{battle_format}'."
    )
    print(
        f"        Opponent Archetype Mix: {int(heuristics_ratio*100)}% SimpleHeuristics, "
        f"{int((1-heuristics_ratio)*100)}% MaxBasePower"
    )

    overall_start = time.time()
    results = []

    if num_workers == 1:
        # Single-worker direct run
        i, b_count, shard_path = worker_plan[0]
        res = run_worker_battles(
            worker_id=i,
            total_battles=b_count,
            battle_format=battle_format,
            output_shard_path=shard_path,
            max_concurrent_battles=max_concurrent_battles,
            heuristics_ratio=heuristics_ratio,
        )
        results.append(res)
    else:
        # Multi-worker process pool
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_worker_battles,
                    worker_id,
                    b_count,
                    battle_format,
                    shard_path,
                    max_concurrent_battles,
                    heuristics_ratio,
                )
                for worker_id, b_count, shard_path in worker_plan
            ]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                print(
                    f"  -> Worker {res['worker_id']} completed {res['finished_battles']} battles "
                    f"({res['turns_recorded']} turns in {res['elapsed_seconds']:.1f}s)"
                )

    total_time = time.time() - overall_start
    total_turns = sum(r["turns_recorded"] for r in results)
    total_finished = sum(r["finished_battles"] for r in results)
    total_won = sum(r["won_battles"] for r in results)

    # Merge shards into final output file
    print(f"[MERGE] Combining worker shards into '{output_file}'...")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for _, _, shard_path in worker_plan:
            if os.path.exists(shard_path):
                with open(shard_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)

    print("=" * 60)
    print(f"[COMPLETE] Data Collection Finished in {total_time:.2f} seconds.")
    print(f"  Total Battles Finished : {total_finished}/{total_battles}")
    print(f"  Win Rate of Recorder   : {total_won / max(1, total_finished) * 100:.1f}%")
    print(f"  Total Turns Recorded   : {total_turns}")
    print(f"  Throughput             : {total_turns / max(0.1, total_time):.1f} turns/sec")
    print(f"  Output Saved To        : {output_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="High-Throughput poke-env Data Collector")
    parser.add_argument("--battles", type=int, default=50, help="Total number of battles to record")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker processes")
    parser.add_argument("--format", type=str, default="gen9randombattle", help="Battle format (default: gen9randombattle)")
    parser.add_argument("--output", type=str, default="data/poke_trajectories.jsonl", help="Output JSONL path")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent battles per player instance")
    parser.add_argument("--heuristics-ratio", type=float, default=0.7, help="Ratio of SimpleHeuristics opponents vs MaxBasePower")

    args = parser.parse_args()
    collect_dataset(
        total_battles=args.battles,
        num_workers=args.workers,
        battle_format=args.format,
        output_file=args.output,
        max_concurrent_battles=args.concurrency,
        heuristics_ratio=args.heuristics_ratio,
    )


if __name__ == "__main__":
    main()
