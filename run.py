import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from parkai.trainer import Trainer, TrainConfig
from parkai.world import World, default_world

DEFAULT_LOT = os.path.join(ROOT, "lots", "default.json")
DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "autosave.pkl")


def load_world(path):
    if path and os.path.exists(path):
        return World.load(path)
    w = default_world()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        w.save(path)
    return w


def build_trainer(args):
    world = load_world(args.lot) if args.lot or not args.load else None
    if args.load:
        return Trainer.load(args.load, world)
    return Trainer(world, TrainConfig(agent=args.agent, num_envs=args.envs, seed=args.seed))


def main():
    p = argparse.ArgumentParser(description="ParkingAI: watch a reinforcement-learning agent learn to park.")
    p.add_argument("--lot", default=None, help="parking lot JSON (default: lots/default.json)")
    p.add_argument("--load", default=None, help="resume from a checkpoint .pkl (default: checkpoints/autosave.pkl if it exists)")
    p.add_argument("--fresh", action="store_true", help="start a new untrained agent even if an autosave exists")
    p.add_argument("--agent", default=None, help="agent type (ppo or random)")
    p.add_argument("--envs", type=int, default=16, help="parallel training cars")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", help="train without a window")
    p.add_argument("--minutes", type=float, default=10.0, help="headless training time")
    p.add_argument("--save", default=None, help="headless checkpoint path (default: checkpoints/autosave.pkl)")
    args = p.parse_args()
    if args.agent is None:
        args.agent = TrainConfig().agent
    if args.load is None and not args.fresh and os.path.exists(DEFAULT_CKPT):
        args.load = DEFAULT_CKPT
        print("Resuming from %s (use --fresh to start over)" % DEFAULT_CKPT)
    if args.lot is None and not args.load:
        args.lot = DEFAULT_LOT

    trainer = build_trainer(args)
    if args.headless:
        from parkai.headless import train_headless
        train_headless(trainer, args.minutes, args.save or DEFAULT_CKPT)
        return
    from parkai.ui.app import run_app
    run_app(trainer, lot_path=args.lot or DEFAULT_LOT, checkpoint_path=args.save or DEFAULT_CKPT)


if __name__ == "__main__":
    main()
