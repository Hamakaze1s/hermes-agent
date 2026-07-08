/**
 * Example plugin — the authoring + publishing reference. A folder under
 * `src/plugins/` with a `plugin.tsx` that default-exports a `HermesPlugin` is
 * all it takes; `discoverBundledPlugins()` finds and registers it (no import,
 * no registry edit). Delete this folder and the statusbar item is gone.
 *
 * The ONLY import surface is `@hermes/plugin-sdk` (lint-enforced) — the
 * vscode-module model. This one chip dogfoods the whole authoring kit:
 *  - `render()` contribution — full stateful React in a slot;
 *  - `ctx.storage` — the count survives reloads (namespaced persistence);
 *  - `host.onEvent('*')` — live gateway stream, counted in the tooltip;
 *  - plugin-local `atom` + `useValue` — module state, leaf subscription;
 *  - `haptic` / `host.notify` / `Tip` / `cn` — the design language.
 */

import { atom, cn, haptic, type HermesPlugin, host, Tip, useValue } from '@hermes/plugin-sdk'

const $clicks = atom(0)
const $events = atom(0)

function ClickCounter() {
  const count = useValue($clicks)
  const events = useValue($events)
  const gateway = useValue(host.state.gateway)

  return (
    <Tip label={`Example plugin — gateway ${gateway}, ${events} events heard`}>
      <button
        className={cn(
          'inline-flex h-full items-center gap-1 rounded-none px-1.5 text-[0.6875rem] tabular-nums transition-colors',
          'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground',
          count > 0 && 'text-foreground'
        )}
        onClick={() => {
          haptic('tap')
          // Imperative read in the handler ($atom.get()), reactive read in the
          // render (useValue) — never a stale closure.
          const next = $clicks.get() + 1
          $clicks.set(next)

          if (next % 10 === 0) {
            host.notify({ kind: 'success', message: `Example plugin: ${next} clicks!` })
          }
        }}
        type="button"
      >
        <span aria-hidden>◉</span>
        <span>{count === 0 ? 'click me' : `clicked ${count}×`}</span>
      </button>
    </Tip>
  )
}

const plugin: HermesPlugin = {
  id: 'example',
  name: 'Example Plugin',
  register(ctx) {
    // Persisted count: hydrate once, write through on every change.
    $clicks.set(ctx.storage.get('clicks', 0))
    $clicks.listen(clicks => ctx.storage.set('clicks', clicks))

    // Hear the live gateway stream (deltas, lifecycle, tools — everything).
    host.onEvent('*', () => $events.set($events.get() + 1))

    // Provenance (source: 'plugin:example') and the namespaced registry id
    // (example:counter) are stamped by ctx — authors write plain contributions.
    ctx.register({
      id: 'counter',
      area: 'statusBar.right',
      order: 100,
      render: () => <ClickCounter />
    })
  }
}

export default plugin
