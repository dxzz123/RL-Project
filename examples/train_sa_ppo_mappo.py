"""Convenience entry point; the same CLI as python -m charging_env.rl.train."""
from charging_env.rl.train import main

if __name__ == '__main__':
    raise SystemExit(main())
