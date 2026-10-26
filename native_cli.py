"""Native circuit-sizing interface."""
from _paths import activate
activate()
from runtime.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
