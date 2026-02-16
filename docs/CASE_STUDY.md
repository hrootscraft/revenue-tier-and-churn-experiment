# Fee Tier Acceleration: Can Pricing Transparency Drive Institutional Volume?

## Executive Summary

I tested whether showing institutional clients their progress toward the next fee tier would increase trading volume. Volume increased 18% on average, but 40% of treatment clients crossed into a cheaper fee tier, compressing margins. Net revenue impact: +4% overall, but ranging from +2% (clients closest to the threshold) to +11% (clients 10-20% away). When combined with a 3.2 percentage point reduction in 6-month churn, the intervention is positive across all segments. I recommend a targeted rollout to clients 10-30% below the next tier threshold, with the closest clients framed as a retention investment.

---

## Business Context

Coinbase Institutional uses a tiered maker/taker fee structure where clients pay lower rates as their quarterly trading volume increases. Maker orders (limit orders that add liquidity) are cheaper than taker orders (market orders), and each client's effective rate depends on their order mix. Many clients sit just below a tier boundary without knowing how close they are to a better rate.

This creates an information asymmetry: clients who would benefit from trading slightly more don't realize it, and Coinbase leaves both volume and client satisfaction on the table.

The intervention was simple: a progress bar on the client dashboard showing distance to the next tier, paired with a monthly email projecting savings if they increased volume by a specific amount.

The challenge: this is not a straightforward "does the feature increase engagement?" experiment. If clients increase volume (good for Coinbase) but cross into a cheaper tier (bad for Coinbase), the net effect on revenue is ambiguous. This is a real pricing optimization problem.

---

## Experimental Design

**Population:** 850 institutional clients (Hedge Funds, Market Makers, Corporate Treasuries, Family Offices, Asset Managers, Quant Funds)

**Randomization:** Stratified by current tier, distance to next tier, and segment. 50/50 treatment/control split within each stratum.

**Treatment:** Dashboard progress bar + monthly projected savings email

**Control:** Standard experience (no visibility into tier progress)

**Primary outcome:** Net fee revenue (Volume × Fee Rate)

**Secondary outcome:** 6-month client churn

**Guardrails:** Support ticket rate, platform login frequency, Enterprise client churn

**Duration:** 13 weeks (one quarter)

---

## Results

### Randomization Check

All pre-experiment covariates balanced between groups (all SMDs < 0.05). Pre-period volume, AUM, tenure, and engagement were statistically indistinguishable between treatment and control.

### Primary Outcome: Volume and Revenue

Treatment clients increased quarterly trading volume by approximately 18%. Three independent statistical tests confirmed significance (Welch's t-test, Mann-Whitney U, and permutation test, all p < 0.01).

After applying CUPED variance reduction using pre-period volume as a covariate, the confidence interval narrowed by approximately 50%, equivalent to doubling our sample size.

However, 40% of treatment clients crossed into a lower fee tier, triggering the margin compression we anticipated.

### Revenue Decomposition

We decomposed the revenue change into three components:

- **Volume effect** (more trading at the old rate): positive, approximately +$2.1M across treatment
- **Rate effect** (old volume at the new, lower rate): negative, approximately -$1.6M
- **Interaction** (new volume at the new rate): negative, approximately -$0.2M
- **Net:** approximately +$0.3M (+4%)

Notably, the volume increase was disproportionately taker-side — treatment clients traded more aggressively (market orders) to hit tier thresholds, shifting their maker/taker mix ~5-7% toward taker. This means the incremental volume earned a slightly higher per-dollar fee than the client's blended average, partially offsetting the tier compression.

The overall effect is positive but modest, and masks significant variation by client proximity to the next tier.

### Heterogeneous Treatment Effects

Segmenting by distance to the next tier boundary revealed the key insight:

**Clients within 10% of the next tier:** Highest volume response (+25%), but 70% crossed the tier. Net revenue barely positive (+2%). The volume gains are almost entirely offset by margin compression on their existing volume.

**Clients 10-20% away:** The sweet spot. Strong volume response (+15%), but only 30% crossed. Coinbase keeps the higher fee rate on most of the increased volume. Net revenue: +11%.

**Clients 20%+ away:** Modest response (+5%), almost no tier crossing (5%). Net revenue: +4%, driven purely by incremental volume at the existing rate.

Empirical Bayes shrinkage was applied to prevent small-segment noise from driving recommendations.

### Durability

The treatment effect started at approximately 22% in week 1 and decayed to approximately 8% by week 13, suggesting a novelty component. We estimate a weekly decay rate of approximately 2%. Under conservative assumptions (continued decay, effect stabilizes at 30% of initial), the annualized impact is approximately 60% of what a naive projection would suggest.

A market volatility spike in weeks 6-8 affected both groups equally. The treatment effect persisted through the spike, strengthening our confidence that it reflects the intervention rather than market conditions.

### Cannibalization

Control group volume did not decline during the experiment period relative to their pre-experiment baseline. No evidence that treatment clients' volume gains came at the expense of other Coinbase clients.

### Guardrails

All guardrails passed: no increase in support tickets, no decrease in login frequency, and zero Enterprise client churn in either group.

---

## Secondary Outcome: Client Retention

### Causal Result

Treatment reduced 6-month churn by 3.2 percentage points (from approximately 11% to approximately 8%, p < 0.01). This is a causal estimate, supported by randomization.

### Observational Finding

Within the treatment group, clients who crossed a tier churned at approximately 4% versus approximately 9% for those who didn't cross. However, this comparison is observational — clients who crossed are self-selected (growing firms, more engaged) and may have churned less regardless.

An instrumental variable estimate, using treatment assignment as an instrument for tier crossing, suggests that crossing a tier causally reduces churn by approximately 12 percentage points for "compliers" (clients who crossed because of the nudge). This estimate should be interpreted cautiously given sample size constraints.

### Mechanism

We hypothesize two mechanisms: (1) a "golden handcuff" effect — clients who earned a better rate have a switching cost if they leave, and (2) an engagement effect — interacting with the progress bar increases platform familiarity and stickiness.

---

## Recommendation

**Target segment: clients 10-30% below the next tier threshold.**

For the 10-20% segment, the intervention is a clear revenue winner (+11% net revenue, +20% churn reduction). For the within-10% segment, short-term revenue is marginal (+2%), but the retention benefit (60% lower churn among tier-crossers) makes it worthwhile when accounting for client lifetime value.

**Rollout plan:** 10% of eligible clients in month 1, expanding to 50% in month 2 and 100% in month 3 if guardrails hold.

**Projected annual impact:**
- Conservative (10-20% segment only, 30% decay): approximately $1.2M incremental revenue + $750K in retention value
- Base case (all segments within 30%, 15% decay): approximately $3.8M + $1.1M
- Optimistic (no decay): approximately $5.4M + $1.5M

---

## What I Would Do Differently

- Run for two quarters to better characterize durability and separate novelty from sustained behavior change
- Test different progress bar designs (percentage vs. dollar amount vs. countdown) as sub-treatments
- Include a "soft-cross" variant: show the progress bar but don't actually change the tier until next quarter, to isolate the motivational effect from the rate change
- Conduct exit interviews with churned clients to validate the golden handcuff mechanism qualitatively
