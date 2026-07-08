---
name: hermes-desktop-plugins
description: Write desktop app plugins that add UI panes and commands.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [desktop, plugins, ui, extension]
    category: productivity
    related_skills: []
---

# Hermes Desktop Plugins Skill

Write plugins for the Hermes desktop app: statusbar items, layout panes,
command-palette commands, keybinds, routes, and themes. A plugin is a single
plain-JavaScript ESM file the app loads at runtime — no build step, no repo
changes. This skill does not cover backend plugins (`~/.hermes/plugins/`);
those are Python and documented separately.

## When to Use

- The user asks for a new desktop UI element (a pane, a statusbar widget, a
  dashboard, a command) without modifying the app itself.
- You want to surface data you compute (via gateway RPC) inside the app.

## Prerequisites

- The Hermes desktop app (it loads plugins; the CLI/gateway alone does not).
- Write access to `$HERMES_HOME/desktop-plugins/` (usually
  `~/.hermes/desktop-plugins/`).

## How to Run

1. Create `~/.hermes/desktop-plugins/<name>/plugin.js` from
   `templates/plugin.js` (relative to this skill directory). Keep `<name>`
   equal to the plugin `id`.
2. Tell the user to press ⌘K in the desktop app and run
   **Reload desktop plugins** (or relaunch the app). The plugin loads and
   its contributions appear.
3. To update, rewrite the file and reload again — same id replaces the old
   registration cleanly.

## Quick Reference

The ONLY import surface is `@hermes/plugin-sdk` (plus `react` /
`react/jsx-runtime`, which resolve to the app's own React — write UI with
`jsx()` calls, not JSX syntax; the file is not compiled).

- `host.state.*` — readonly reactive atoms: `activeSessionId`, `cwd`,
  `gateway`, `model`, `profile`, `viewport`. Read with `.get()` in handlers,
  `useValue(atom)` in components.
- `host.request(method, params)` — gateway JSON-RPC (sessions, config,
  skills, cron — everything the app uses).
- `host.onEvent(type, fn)` — live gateway events (`'*'` for all). Returns a
  disposer.
- `host.notify({ kind, message })`, `host.navigate(path)`, `host.logs(...)`,
  `host.status()`, `haptic('tap')`.
- `ctx.register({ id, area, order?, render?, data? })` — contribute UI.
  Key areas: `'statusBar.right'`/`'statusBar.left'` (chips),
  `'panes'` (layout zones — set `title` and
  `data: { placement: 'left'|'right'|'bottom'|'main', width?, height? }`;
  the pane auto-joins a matching zone), `PALETTE_AREA` (⌘K commands),
  `KEYBINDS_AREA` (rebindable actions).
- `ctx.storage.get/set/remove` — persistence namespaced to your plugin.
- UI: `Tip`, `Button`, `Codicon`, `Input`, `StatusDot`, `LogView`, `cn`,
  `icons.*` — use these so the plugin looks native.

## Procedure

1. Pick a short kebab-case `id`; the folder name must match.
2. Start from `templates/plugin.js`; keep the default export shape
   (`{ id, name, register(ctx) }`).
3. For a pane, register `area: 'panes'` with a `placement` hint and a
   `render` returning your component — the app places it into a sensible
   zone automatically; the user can drag it anywhere afterwards.
4. Fetch data with `host.request` and/or subscribe with `host.onEvent`;
   never poll faster than a few seconds.
5. Write the file with your file tools, then ask the user to run
   **Reload desktop plugins** from ⌘K.

## Pitfalls

- JSX syntax will not parse — the file loads uncompiled. Use
  `jsx('div', { children: ... })` from `react/jsx-runtime`.
- Do not import anything except `@hermes/plugin-sdk`, `react`, and
  `react/jsx-runtime`; other specifiers fail to resolve.
- Handlers must read state imperatively (`$atom.get()`), never from render
  closures — rapid events will otherwise see stale values.
- Keep components small; subscribe (`useValue`) only in the leaf that
  renders the value.

## Verification

- The plugin's UI appears after **Reload desktop plugins**.
- No error toast ("Plugin <name> failed to load") appears; if it does, the
  message names the failure — fix and reload.
- For panes: the new zone is visible and draggable like any core pane.
