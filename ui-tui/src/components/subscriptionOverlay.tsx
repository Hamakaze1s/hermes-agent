import { Box, Text, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { SubscriptionOverlayState } from '../app/interfaces.js'
import type { SubscriptionStateResponse } from '../gatewayTypes.js'
import type { Theme } from '../theme.js'

import { ActionRow, barCells, footer, MenuRow } from './overlayPrimitives.js'

interface SubscriptionOverlayProps {
  /** Replace the overlay slot (screen transitions + pending data). */
  onPatch: (next: Partial<SubscriptionOverlayState>) => void
  /** Close the overlay entirely. */
  onClose: () => void
  overlay: SubscriptionOverlayState
  t: Theme
}

/**
 * The /subscription modal — deep-link only, NEVER charges in-terminal.
 * Mirrors billingOverlay.tsx's structure: a pure-render state machine
 * (overview → confirm → handoff, plus stepup for Phase 4) where all RPCs
 * live in subscription.ts and are reached through `overlay.ctx`.
 */
export function SubscriptionOverlay({ onClose, onPatch, overlay, t }: SubscriptionOverlayProps) {
  const { ctx, screen, state: s } = overlay

  return (
    <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" paddingX={1}>
      {screen === 'overview' && <OverviewScreen ctx={ctx} onClose={onClose} onPatch={onPatch} s={s} t={t} />}
      {screen === 'confirm' && (
        <ConfirmScreen
          ctx={ctx}
          onBack={() => onPatch({ screen: 'overview' })}
          onClose={onClose}
          onPatch={onPatch}
          overlay={overlay}
          s={s}
          t={t}
        />
      )}
      {screen === 'handoff' && <HandoffScreen onClose={onClose} t={t} />}
      {/* stepup screen is built in Phase 4 (U9) */}
    </Box>
  )
}

// ── Screen: Overview (covers states a–e + dunning) ───────────────────

interface ScreenProps {
  ctx: SubscriptionOverlayState['ctx']
  onClose: () => void
  onPatch: (next: Partial<SubscriptionOverlayState>) => void
  s: SubscriptionStateResponse
  t: Theme
}

/** Usage bar from subscription allowance (monthly_credits vs credits_remaining). */
function usageBar(s: SubscriptionStateResponse): null | string {
  const c = s.current

  if (!c || !c.monthly_credits || !c.credits_remaining) {
    return null
  }

  const monthly = Number(c.monthly_credits)
  const remaining = Number(c.credits_remaining)

  if (!(monthly > 0) || Number.isNaN(remaining)) {
    return null
  }

  const spent = Math.max(0, monthly - remaining)
  const { bar, pct } = barCells(spent / monthly)

  return `${remaining} of ${c.monthly_credits} remaining   ${bar} ${100 - pct}% left`
}

function OverviewScreen({ ctx, onClose, onPatch, s, t }: ScreenProps) {
  // (d) not-admin: read-only + note + Manage/portal only.
  const canChange = s.can_change_plan && s.is_admin
  const c = s.current
  const isFree = !c?.tier_id
  const hasPendingDowngrade = !!c?.pending_downgrade_tier_name
  const isPastDue = !!c?.is_past_due

  // Dunning: past due — don't re-offer fresh subscribe; route to manage/portal.
  const dunningNote = isPastDue && c?.cycle_ends_at
    ? `Payment past due — your plan is still active until ${c.cycle_ends_at}.`
    : null

  const downgradeNote = hasPendingDowngrade
    ? `Scheduled to switch to ${c?.pending_downgrade_tier_name} on ${c?.pending_downgrade_at}.`
    : null

  const notAdminNote = !canChange ? 'Plan changes need an org admin/owner.' : null

  // Build the tier list (only enabled tiers for the menu; current marked).
  const enabledTiers = s.tiers.filter(tier => tier.is_enabled)
  const currentTierOrder = c?.tier_id ? s.tiers.find(tier => tier.tier_id === c.tier_id)?.tier_order : undefined
  const isTopTier = currentTierOrder != null && enabledTiers.every(tier => tier.tier_order <= currentTierOrder)
  const topTierNote = isTopTier ? "You're on the top plan." : null

  // Menu items: tiers (selectable) + Manage on portal + Close.
  // For not-admin: just Manage on portal + Close.
  const tierItems: { label: string; tierId?: string }[] = canChange
    ? [
        ...enabledTiers.map(tier => ({
          label: `${tier.is_current ? '✓ ' : ''}${tier.name} — ${tier.dollars_per_month_display}/mo (${tier.monthly_credits} credits)`,
          tierId: tier.tier_id
        })),
        { label: 'Manage on portal' },
        { label: 'Close' }
      ]
    : [{ label: 'Manage on portal' }, { label: 'Close' }]

  const [sel, setSel] = useState(0)

  const choose = (i: number) => {
    const item = tierItems[i]

    if (!item) {
      return
    }

    if (item.label === 'Close') {
      return onClose()
    }

    if (item.label === 'Manage on portal') {
      if (s.portal_url) {
        ctx.sys('Opening portal in your browser…')
        void ctx.openManageLink()
      }

      return onClose()
    }

    // A tier was selected — go to confirm (deep-link, no in-terminal charge).
    if (item.tierId && item.tierId !== c?.tier_id) {
      onPatch({ screen: 'confirm', pendingTargetTierId: item.tierId })
    }
  }

  useInput((ch, key) => {
    if (key.escape) {
      return onClose()
    }

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)
    }

    if (key.downArrow && sel < tierItems.length - 1) {
      setSel(v => v + 1)
    }

    if (key.return) {
      return choose(sel)
    }

    const n = parseInt(ch, 10)

    if (n >= 1 && n <= tierItems.length) {
      return choose(n - 1)
    }
  })

  const header = isFree ? 'Subscribe to a plan' : c?.tier_name ? `Your plan: ${c.tier_name}` : 'Subscription'
  const bar = usageBar(s)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {header}
      </Text>
      {bar && <Text color={t.color.text}>{bar}</Text>}
      {c?.credits_remaining && !bar && (
        <Text color={t.color.text}>Credits remaining: {c.credits_remaining}</Text>
      )}
      {s.org_name && (
        <Text color={t.color.muted}>
          Org: {s.org_name}
          {s.role ? ` · ${s.role}` : ''}
        </Text>
      )}
      {dunningNote && (
        <Box marginTop={1}>
          <Text color={t.color.warn}>{dunningNote}</Text>
        </Box>
      )}
      {downgradeNote && (
        <Box marginTop={1}>
          <Text color={t.color.warn}>{downgradeNote}</Text>
        </Box>
      )}
      {notAdminNote && (
        <Box marginTop={1}>
          <Text color={t.color.warn}>{notAdminNote}</Text>
        </Box>
      )}
      {topTierNote && (
        <Box marginTop={1}>
          <Text color={t.color.muted}>{topTierNote}</Text>
        </Box>
      )}

      <Text />
      {tierItems.map((item, i) => (
        <MenuRow active={sel === i} index={i + 1} key={item.label} label={item.label} t={t} />
      ))}

      <Text />
      {footer(`↑/↓ select · 1-${tierItems.length} quick pick · Enter confirm · Esc close`, t)}
    </Box>
  )
}

// ── Screen: Confirm (y/n deep-link, NO in-terminal charge) ───────────

interface ConfirmScreenProps extends ScreenProps {
  onBack: () => void
  onPatch: (next: Partial<SubscriptionOverlayState>) => void
  overlay: SubscriptionOverlayState
}

function ConfirmScreen({ ctx, onBack, onClose, onPatch, overlay, s, t }: ConfirmScreenProps) {
  const targetTierId = overlay.pendingTargetTierId ?? undefined
  const targetTier = s.tiers.find(tier => tier.tier_id === targetTierId)
  const isUpgrade = !s.current?.tier_id

  const [sel, setSel] = useState(0)
  const items = ['Continue to your subscription page', 'Cancel']
  const [transitioned, setTransitioned] = useState(false)

  const confirm = () => {
    if (transitioned) {
      return
    }

    setTransitioned(true)
    onPatch({ screen: 'handoff' })

    void ctx.openManageLink(targetTierId).then(ok => {
      if (!ok) {
        // If openManageLink surfaces insufficient_scope, the ctx closure in
        // subscription.ts handles the stepup transition (Phase 4 wiring).
        // For now, return to overview on any failure.
        onPatch({ screen: 'overview' })
      }
    })
  }

  const choose = (i: number) => {
    if (i === 0) {
      return confirm()
    }

    return onBack()
  }

  useInput((ch, key) => {
    if (key.escape) {
      return onBack()
    }

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)
    }

    if (key.downArrow && sel < items.length - 1) {
      setSel(v => v + 1)
    }

    if (key.return) {
      return choose(sel)
    }

    if (ch === 'y') {
      return choose(0)
    }

    if (ch === 'n') {
      return choose(1)
    }
  })

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {isUpgrade ? 'Confirm subscription' : 'Confirm plan change'}
      </Text>
      {targetTier && (
        <Text color={t.color.text}>
          {targetTier.name} — {targetTier.dollars_per_month_display}/mo ({targetTier.monthly_credits} credits)
        </Text>
      )}
      <Text color={t.color.muted}>You'll finish this change securely on your subscription page in your browser.</Text>

      <Text />
      {items.map((label, i) => (
        <ActionRow active={sel === i} color={i === 0 ? t.color.ok : undefined} key={label} label={label} t={t} />
      ))}

      <Text />
      {footer('y/Enter confirm · n/Esc cancel', t)}
    </Box>
  )
}

// ── Screen: Handoff (transient) ──────────────────────────────────────

function HandoffScreen({ onClose, t }: { onClose: () => void; t: Theme }) {
  useInput((_ch, key) => {
    if (key.escape) {
      return onClose()
    }
  })

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        Opening your subscription page…
      </Text>
      <Text color={t.color.muted}>Opening your subscription page in the browser…</Text>
      <Text color={t.color.muted}>Finish on the page that just opened. Re-run /subscription to see the change.</Text>
      <Text />
      {footer('Esc close', t)}
    </Box>
  )
}
