# Simulation Simplifications vs. Production Coinbase Systems

This document lists the deliberate simplifications made in the data generator
relative to how a fee-tier acceleration experiment would run on the actual
Coinbase institutional platform. Each simplification is noted with its direction
of bias and its effect on the analysis conclusions.

| # | Simplification | Production Reality | Direction of Bias / Analysis Impact |
|---|---------------|-------------------|--------------------------------------|
| 1 | **SUTVA assumed — no spillover between clients** | On a shared order book, treatment clients sending additional taker volume interact with control clients' resting limit orders, creating indirect effects | Likely small: institutional flow is a tiny share of Coinbase's total volume; real analysis would run a SUTVA sensitivity check |
| 2 | **Quarterly fixed window for tier assignment** | Real Coinbase uses a rolling 30-day trailing volume window; tiers can update every day | Understates speed of tier changes; smooths weekly noise — conservative for detecting tier crossings |
| 3 | **Single treatment arm (progress bar + email only)** | Coinbase may run multiple concurrent institutional product experiments; exclusion lists prevent overlap | Concurrent exposure not modeled; in practice, a holdout/exclusion layer would isolate the experiment |
| 4 | **Flat pre-period baseline (IID noise around a static mean)** | Real institutional volumes track crypto market trends, quarterly seasonality, and client-level growth trajectories | Understates CUPED variance reduction (less autocorrelation to absorb); power estimates are conservative |
| 5 | **Segment-to-tier mapping based on stylized priors** | Real mapping would be calibrated from Coinbase's institutional CRM data | Segment response modifiers can dominate tier modifiers in the HTE heatmap — a real dataset would recalibrate these weights |
| 6 | **Binary churn: zero volume after `churn_week`** | Real churn is gradual (clients reduce activity before going inactive); the standard definition requires 28 consecutive days of zero activity using daily data | Overstates churn precision; follow-up volume trends and the T vs. C churn rate comparison are directionally correct |
| 7 | **400M+ (top) tier modeled with zero treatment effect** | Top-tier clients might still respond to bundled UX improvements, data features, or relationship manager outreach attached to the same campaign | Conservatively excludes a small population; real effect for this tier could be small but non-zero |
| 8 | **IID volume noise across clients within the same segment** | Market makers' volumes are correlated — the same market microstructure events (spread widening, volatility regimes) move all MMs simultaneously | Understates true within-segment variance; standard errors may be slightly anti-conservative in segment-level HTE |
| 9 | **Only 3 coded confounders (volatility spike, competitor promo, seasonal rally)** | Production data has APAC/EMEA regulatory events, scheduled exchange maintenance, token-listing volatility, and macro events (ETF approvals, Fed meetings) | Additional confounders would add noise and widen CIs; the 3 coded ones are chosen to represent the most common categories |
