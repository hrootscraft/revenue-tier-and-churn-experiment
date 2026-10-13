# Simplifications relative to a production exchange

The simulator is deliberately simpler than a real institutional exchange. Each row names the
simplification, what production would look like, and which way it biases the conclusions.

| # | Simplification | Production reality | Direction / impact |
|---|---|---|---|
| 1 | **SUTVA: no interference between clients.** A client's outcome depends only on their own cell. | On a shared order book, treated clients' extra taker flow hits control clients' resting orders. | Institutional flow is a small share of total volume, so spillover is likely small; a real analysis would add a cluster- or market-level sensitivity check. |
| 2 | **Quarterly tier window.** Tier is set by the previous quarter's volume. | Many venues use a rolling 30-day window and re-tier daily. | Rolling windows make crossings faster and noisier; the quarterly window is conservative for detecting them. |
| 3 | **Three cells, no concurrent campaigns.** | Several campaigns run at once; exclusion lists keep them from overlapping. | Not modelled; a real holdout would be a global exclusion layer. |
| 4 | **Engagement lapse is a single telemetry event with a client-constant hazard.** | Engagement is a sequence of interactions; clients can re-engage. | Re-engagement would raise the long-run effect; our decay estimate is conservative. |
| 5 | **Volume drift is log-normal with one observed driver (AUM flows).** | Drift has many drivers: mandates won/lost, strategy changes, regulation, macro. | The share of drift that CRM covariates explain sets how much AIPW beats CUPED; we assume a strong CRM signal (`drift_flow_coef` vs `drift_log_sd` in the config) and document it in DECISIONS.md. With a weaker signal AIPW's advantage shrinks toward CUPED's. |
| 6 | **Three planted market confounders** (volatility spike, competitor promo, seasonal rally). | Many more: listings, outages, ETF news, regional regulation. | More confounders widen the week-by-week CIs but cannot bias a treated-vs-holdout contrast. |
| 7 | **Top-tier clients get zero volume effect.** | They might respond to bundled UX or relationship outreach. | Conservative; excludes ~1% of clients. |
| 8 | **Client-level noise is independent across clients within a week.** | Market makers' volumes co-move with microstructure regimes. | Understates within-segment variance; segment-level SEs are slightly optimistic. |
| 9 | **The incentive's rate cut is mechanical** (half-step during the campaign, full next-tier rate if the threshold is reached). | Real incentives carry caps, claw-backs and negotiated terms. | The decomposition identity holds for any rate rule; only the size of the rate effect would change. |
| 10 | **No churn / lifetime-value channel.** | Clients who cross a tier may churn less ("golden handcuff"). | We dropped it to keep one outcome (revenue) and one durability mechanism (engagement decay). It is the natural next extension. |
| 11 | **Simulated data with planted effects.** | Real data: the truth is unknown. | This is the point of the project: every method is graded against a known answer. The README says so up front. |
