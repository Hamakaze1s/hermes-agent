import type {
  BillingMutationResponse,
  SubscriptionManageLinkResponse,
  SubscriptionStateResponse
} from '../../../gatewayTypes.js'
import { openExternalUrl } from '../../../lib/openExternalUrl.js'
import type { SubscriptionOverlayCtx } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

type Sys = (text: string) => void

/**
 * Build the ctx the overlay uses to talk to the gateway + emit transcript
 * lines.  Mirrors topup.ts's buildOverlayCtx — all RPC + error-mapping logic
 * lives here (single source of truth); the overlay only renders + routes keys.
 */
const buildSubscriptionCtx = (ctx: SlashRunCtx, sys: Sys): SubscriptionOverlayCtx => ({
  openManageLink: (targetTierId?: string) =>
    ctx.gateway
      .rpc<SubscriptionManageLinkResponse>('subscription.manage_link', {
        ...(targetTierId ? { target_tier_id: targetTierId } : {})
      })
      .then(r => {
        if (r && r.ok && r.url) {
          openExternalUrl(r.url)
          sys('Opening your subscription page in the browser — finish the change there; re-run /subscription to confirm.')

          return true
        }

        // insufficient_scope → Phase 4 step-up (handled by the overlay).
        // For now, other errors just log + resolve false.
        if (r && r.error) {
          sys(r.message ?? r.error)
        }

        return false
      })
      .catch(e => {
        ctx.guardedErr(e)

        return false
      }),
  refreshState: () =>
    ctx.gateway
      .rpc<SubscriptionStateResponse>('subscription.state', {})
      .then(r => r ?? null)
      .catch(() => null),
  requestRemoteSpending: () =>
    ctx.gateway
      .rpc<BillingMutationResponse>('billing.step_up', { session_id: ctx.sid ?? undefined })
      .then(r => !!(r && r.ok && r.granted))
      .catch(() => false),
  sys
})

export const subscriptionCommands: SlashCommand[] = [
  {
    help: 'View or change your Nous subscription plan',
    name: 'subscription',
    aliases: ['upgrade'],
    // ZERO sub-commands: bare `/subscription` fetches state and opens the
    // overlay (deep-link only — NEVER charges in-terminal).
    run: (_arg, ctx) => {
      const sys: Sys = ctx.transcript.sys

      ctx.gateway
        .rpc<SubscriptionStateResponse>('subscription.state', {})
        .then(
          ctx.guarded<SubscriptionStateResponse>(s => {
            if (!s.logged_in) {
              sys('Not logged into Nous Portal — run /portal to log in, then /subscription.')

              return
            }

            patchOverlayState({
              subscription: {
                ctx: buildSubscriptionCtx(ctx, sys),
                pendingTargetTierId: null,
                resumeScreen: null,
                screen: 'overview',
                state: s
              }
            })
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
