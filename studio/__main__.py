"""Start the loopback-only public demo, or explicitly select a live local workspace."""

import argparse
import uvicorn
from .store import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use your local SQLite workspace and cloud credentials",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not (ROOT / "out" / "index.html").is_file():
        parser.error("Build the studio first with npm ci and npm run build.")
    uvicorn.run(
        "studio.api:app" if args.live else "studio.demo:app",
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
