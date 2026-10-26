# Environment Configuration

## Runtime Environment
- **Sandbox**: This application runs in an E2B sandbox environment
- **Preview Access**: The application preview is accessible to users via an E2B-exposed URL embedded within an iframe.

## Development Server
- **Port**: The development server runs on port 4000
- **Default State**: The dev server is already running
- **Important**: Do NOT start the dev server unless explicitly requested by the user
- **If Starting**: When starting a dev server:
  1. First kill any existing process on port 4000
  2. Then start the new server instance using `bun run dev --port 4000`

## Memory Safety
- **Sandbox memory watchdog**: The sandbox may terminate processes automatically when available memory becomes critically low. This is infrastructure-controlled behavior and is unrelated to the code being modified.
- **Process termination**: A development server or other high-memory process may disappear unexpectedly if the memory watchdog intervenes.
- **Diagnostics**: The agent does not have access to the infrastructure-level watchdog's privileged state or logs. Do not attempt to access, modify, restart, disable, or otherwise interact with the watchdog.
- **No auto-restart**: If a development server is no longer running, do NOT restart it yourself.
- **If the dev server stops unexpectedly**: Stop all live checks and ask the user to restart it. Do not attempt to restart it yourself.


## Package Management
- **Package Manager**: Use `bun` for all package management operations
- **Installation**: `bun install`
- **Running Scripts**: `bun run <script-name>`

## Quality Checks
- **Type Checking**: After completing tasks, run `bun run typecheck` to verify type safety
- **Build Command**: Do NOT run the build command unless the user explicitly requests it
- **Default Verification**: Use typecheck as the standard post-completion verification step

<!-- BEGIN:nextjs-agent-rules -->
 
# This is NOT the Next.js you know
 
This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.
 
This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.
 
<!-- END:nextjs-agent-rules -->
