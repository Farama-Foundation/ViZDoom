import sys
from argparse import ArgumentParser
from pathlib import Path

from vizdoom.pettingzoo_wrapper.human_policy_duel import play_against_policy


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "ippo_multi_duel_cnn__79265ad4_26_08_30-09_48_53"
    / "checkpoints"
    / "checkpoint_1024000.pt"
)


def main() -> None:
    parser = ArgumentParser(
        description="Play multi_duel against a trained IPPO policy at normal FPS."
    )
    parser.add_argument("-e", "--episodes", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=5029)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("human_vs_policy_results.json"),
        help="JSON file for per-episode and aggregate metrics",
    )
    args = parser.parse_args()
    play_against_policy(
        args.checkpoint,
        episodes=args.episodes,
        port=args.port,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
