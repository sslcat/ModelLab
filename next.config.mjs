import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

export default function config(phase) {
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    return {
      devIndicators: false,
      async rewrites() {
        return [
          {
            source: "/api/:path*",
            destination: "http://127.0.0.1:8765/api/:path*",
          },
        ];
      },
    };
  }
  return { output: "export", devIndicators: false, poweredByHeader: false };
}
