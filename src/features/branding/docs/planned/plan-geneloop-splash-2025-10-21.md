# GeneLoop CLI — Startup Splash & Rebrand Plan (2025-10-21)

## 1) Mission & Context
- Goal: Fork OpenAI Codex CLI and rebrand as GeneLoop CLI with a new launch screen that matches the visual style shown (large pixelated ASCII logo, gradient colorway, helpful tips, prompt area), while retaining all core Codex CLI functionality (auth, config, agents).
- Constraints: Keep core behavior intact; banner must degrade gracefully on limited terminals; cross-platform support (macOS/Linux/Windows); minimal new runtime deps; ≤300-line modules; TypeScript + strict types; follow vertical-slice architecture.
- Consumers: Internal engineers, early adopters, and users familiar with Codex CLI who expect stability and clarity.
- Success: Visually distinctive banner using Gene/DNA gradient (#00FF00→#0000FF), frictionless startup on common shells/terminals, no regressions in auth/config/agent flows, docs updated, and a clean release under the new bin `geneloop`.

## 2) Deliverables
- New banner module with ASCII art for "GeneLoop" + DNA gradient.
- Tips section and optional theme selector on first run (opt-in, stored in config).
- Configurable color on/off with robust color detection.
- Updated README and branding strings throughout the CLI.

## 3) Tech Choices (doc-backed)
- Colors & styles: `chalk` (v5, ESM) — truecolor/hex, detection via `supportsColor` (exported by chalk). Docs confirm hex/rgb APIs and support levels.
- ASCII art: `figlet.js` with curated font (e.g., `ANSI Shadow`, `Big`) or pre-baked ASCII string for speed. Docs confirm `textSync`, custom font loading, and font listing.
- Gradient: Implement a tiny, local gradient utility (no external dep) to lerp from `#00FF00`→`#0000FF` across characters. Keeps footprint minimal and avoids searching for gradient libs.

References used (Context7): chalk basic/hex/rgb usage, color levels; figlet.js `textSync`, font listing and custom fonts.

## 4) Architecture Placement
- Feature slice: `src/features/branding/`
  - `ui/banner.ts` — main banner composition (≤200 lines)
  - `ui/tips.ts` — tips rendering + layout helpers (≤120 lines)
  - `theme/colors.ts` — palette + gradient helpers (≤150 lines)
  - `index.ts` — exported `showStartup()` for wiring in app entry
  - `docs/` — plan/summary docs per AGENTS.md

## 5) Step-by-Step Plan
1. Fork & setup
   - Fork Codex CLI → `geneloop-cli`.
   - Node ≥18 LTS; enable ESM if upstream is ESM. Ensure `typescript --strict`.
   - `pnpm`/`npm` install; run existing tests/build.

2. Locate startup entry and splash
   - Search for startup hooks (e.g., `bin/codex`, `src/cli/index.ts`, `src/ui/banner.ts`, usage of `chalk`).
   - Identify the function that prints the initial splash and the first prompt render.

3. Introduce branding slice
   - Create `src/features/branding/` with modules listed in section 4.
   - Wire `showStartup()` into the app’s main entry before the first prompt.

4. Theme + gradient utilities
   - Implement `hexToRgb`, `lerp`, `gradientLine(text, fromHex, toHex)` and `supportsTruecolor` guards.
   - Use your Light Mode gradient from the demo CSS as the default. We will mirror the exact CSS gradient stops (provide CSS variable names or values) and map them to per-character coloring.
   - Keep DNA green→blue as an alternate preset.

5. ASCII art for "GeneLoop"
   - Pre-generate offline (figlet or artii) and bake as a constant file (no runtime figlet dependency).
   - Provide a tiny dev script to regenerate the ASCII on demand; do not ship it in production bundle.

6. Banner composition
   - Gradient-apply per-character across each ASCII line.
   - Add a thin border/padding and title line. Avoid extra deps like `boxen` by composing with Unicode box-drawing chars.

7. Tips & prompt area
   - Print 3–5 quick tips (auth, config, help, examples); respect terminal width.
   - Default theme = Light Mode (from demo CSS). Provide `--theme light|dna|mono|none`; persist in config.

8. Cross-platform checks
   - Test on macOS iTerm/Terminal, Ubuntu (xterm-256color), Windows Terminal/PowerShell.
   - Fallback: disable gradients when `chalk.level < 3` (use basic colors or monochrome).

9. Rebrand sweep
   - Rename visible strings from Codex to GeneLoop (banner, help, config headings, telemetry events’ app field).
   - Keep APIs and flags stable; add aliases only where safe.

10. Docs & release
   - Update README, screenshots, and `package.json` (`name: 'geneloop-cli'`, `bin: 'geneloop'`).
   - No publishing yet per scope; prepare docs and screenshots only.

## 6) Code Sketches (TypeScript)

### theme/colors.ts
```ts
// src/features/branding/theme/colors.ts
import chalk, {supportsColor} from 'chalk';

// Light Mode gradient (demo blues) — stops come from demo CSS
// Source CSS (light): src/features/biocode-demo/docs/active/geneloop-showcase.html
// :root[data-theme="light"]
//   --secondary: oklch(0.4820 0.0825 206.0615)
//   --primary:   oklch(0.5855 0.1006 208.0245)
//   --accent:    oklch(0.6697 0.1154 208.9445)
// Implementation note: we will precompute sRGB hex for these three stops and paste below.
export const LIGHT_BLUE_STOPS = [
  /* secondary */ '#TODO_SEC_HEX',
  /* primary   */ '#TODO_PRI_HEX',
  /* accent    */ '#TODO_ACC_HEX',
] as const;

// Alternate preset
export const DNA_GRADIENT = ['#00FF00', '#0000FF'] as const;

export const supportsTruecolor = () => Boolean(supportsColor && supportsColor.has256 && supportsColor.has16m);

const hexToRgb = (hex: string) => {
  const clean = hex.replace('#', '');
  const num = parseInt(clean.length === 3
    ? clean.split('').map((c) => c + c).join('')
    : clean, 16);
  return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
};

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export function gradientLine(text: string, fromHex = LIGHT_BLUE_STOPS[0], toHex = LIGHT_BLUE_STOPS[LIGHT_BLUE_STOPS.length-1]): string {
  if (!supportsTruecolor()) return chalk.green(text); // graceful fallback
  const from = hexToRgb(fromHex);
  const to = hexToRgb(toHex);
  const len = Math.max(1, text.length);
  let out = '';
  for (let i = 0; i < text.length; i++) {
    const t = i / (len - 1);
    const r = Math.round(lerp(from.r, to.r, t));
    const g = Math.round(lerp(from.g, to.g, t));
    const b = Math.round(lerp(from.b, to.b, t));
    out += chalk.rgb(r, g, b)(text[i]);
  }
  return out;
}
```

### ui/banner.ts
```ts
// src/features/branding/ui/banner.ts
import chalk from 'chalk';
import {gradientLine} from '../theme/colors';

const BORDER = { tl: '╭', tr: '╮', bl: '╰', br: '╯', h: '─', v: '│' } as const;

// Option A: Generate at runtime with figlet (configure font elsewhere)
export function renderAsciiLogo(lines: string[]): string[] { return lines; }

// GeneLoop ASCII (baked, provided by user file)
// Source file: /Users/sethmorton/Downloads/ascii-text-art.txt (16 lines)
// We will paste the final content into src/features/branding/ui/logo.geneloop.ts
// and import it here. Placeholder for now:
const GENELOOP_ASCII = [
  '<<PASTE_FINAL_ASCII_LINES_HERE>>'
];

export function buildBanner(): string {
  const logo = GENELOOP_ASCII.map((line) => gradientLine(line)).join('\n');
  const content = `${logo}\n\n` +
    chalk.dim('Tips:') + '\n' +
    `  • Run ${chalk.bold('geneloop auth login')} to authenticate\n` +
    `  • Explore commands with ${chalk.bold('geneloop help')}\n` +
    `  • Configure defaults via ${chalk.bold('geneloop config')}\n` +
    `  • Start an agent session with ${chalk.bold('geneloop')}\n`;

  const lines = content.split('\n');
  const width = Math.max(...lines.map((l) => l.length));
  const pad = (s: string) => s + ' '.repeat(width - s.length);

  const top = `${BORDER.tl}${BORDER.h.repeat(width + 2)}${BORDER.tr}`;
  const body = lines.map((l) => `${BORDER.v} ${pad(l)} ${BORDER.v}`).join('\n');
  const bottom = `${BORDER.bl}${BORDER.h.repeat(width + 2)}${BORDER.br}`;

  return [top, body, bottom].join('\n');
}

export function showBanner(log = console.log): void {
  log(buildBanner());
}
```

### Offline ASCII generation (dev-only, not shipped)
```ts
// tools/make-ascii.ts (dev script)
// Usage: ts-node tools/make-ascii.ts "GeneLoop" "ANSI Shadow"
import figlet from 'figlet';
const text = process.argv[2] ?? 'GeneLoop';
const font = process.argv[3] ?? 'ANSI Shadow';
const raw = figlet.textSync(text, { font });
console.log(raw);
// Paste the output into src/features/branding/ui/logo.geneloop.ts
```

### Wiring into CLI entry
```ts
// src/cli/index.ts (or equivalent main)
import {showBanner} from '../features/branding';

export async function main() {
  showBanner();
  // existing init: load config → auth check → enter prompt
}
```

## 7) Cross-Platform & Fallbacks
- Use `chalk.level` to detect color support. If `<3`, skip truecolor gradient and use a single accent color.
- Avoid large Unicode blocks that render inconsistently; prefer ASCII/box-drawing characters.
- Respect terminal width (`process.stdout.columns`) and center or clamp banner width when necessary.

## 8) Rebrand Tasks Checklist
- Rename `package.json` → `name: geneloop-cli`; `bin` → `geneloop`.
- Replace user-facing strings: app name, help headers, telemetry app id (if present).
- Keep env vars and config keys backward-compatible or provide aliases.
- Update README badges, examples, and screenshots.

## 9) Test Plan
- Unit: gradient utilities (hex parsing, lerp), banner width calculation, figlet integration (where used).
- Snapshot: banner output under truecolor vs basic color modes.
- Manual: macOS Terminal/iTerm2, Ubuntu xterm-256color, Windows Terminal. Verify monochrome mode with `FORCE_COLOR=0`.

## 10) Open Questions
1) Please share the exact Light Mode gradient from the demo (CSS var names and resolved hex/rgb stops, including angle). We’ll mirror it precisely.
2) Confirm the figlet font you want for the baked logo (e.g., ANSI Shadow, Big). If you already have an ASCII you like, share it and we’ll paste it into constants.
3) Any tagline or subheader you want under “GeneLoop”? E.g., “Terminal-native AI workflows”.
4) Preferred tips list (3–5 bullets) to show in the box? Otherwise I’ll keep the proposed defaults.
