## Product DS brain:
1. Before: “Define success, safety, and how long we need to know.”
2. During: “Verify randomization and measurement; don’t break the system.”
3. After: “Quantify impact with uncertainty; correct for multiple looks; recommend action.”

### Quick Steps:

**Example Experiment**: Product wants to add a “Smart Order Retry” feature in Coinbase Prime’s trading API: if an order fails due to transient venue/API issues, the system automatically retries within strict risk limits. Hypothesis: this reduces failed orders and increases executed volume, improving retention.

A. BEFORE THE EXPERIMENT

1. Primary metric, guardrails, decision rule

    Before you run any analysis, decide: What does “better” mean, and what are we not allowed to break?  Pick 1 primary metric (the “north star for this change”) you’re willing to make the ship decision on. For Smart Order Retry: **Primary: Order success rate = successful executions / total submitted orders (within a window)** because the feature directly targets failed orders. You might also choose a primary alternative: Executed notional per client-week (if revenue is the key business goal). But pick one. If you pick 4 primaries, you’ll talk yourself into shipping anything. 
    
    **Guardrails (must-not-hurt)**: These are safety checks. Eg. **Duplicate execution rate (accidentally executing twice), Client support tickets per 1k orders, Risk limit breaches, Latency p95 (retries might increase latency)**.
    
    You're designing a decision: "Ship if primary metric improves by at least X (practical threshold) and no guardrail metric worsens beyond Y; otherwise iterate." This prevents “we shipped because p < 0.05.”

2. Power, runtime, and MDE

    You’re asking: How long must we run to be able to detect an effect that matters? 
    
    - MDE (minimum detectable effect): smallest change you care about.
    - Power: probability you’ll detect that change if it’s real.
    - Runtime: how long until you collect enough data.

    For example, Baseline order success rate = 96%. You care about +0.3 percentage points (96.0% → 96.3%) because that translates into meaningful reduction in failed orders. If you don’t do this step, you’ll end up either stopping too early (“no effect!”) when you just didn’t have enough data or running forever and “finding significance” on tiny meaningless effects
    
    PS: Decide what’s worth detecting before you look.

3. Randomization unit and interference risk

    You must align your design to how the product is exposed. Randomization unit = who gets treatment vs control? 

    For Coinbase Institutional, it’s usually Client-level randomization (each institutional client is either in treatment or control) because if you randomize at order-level within a client, you’ll contaminate behavior: the client adapts to the feature and changes order submission patterns

    About interference risk, standard A/B assumes “one unit doesn’t affect another.” In markets, that can be false. Examples of interference in trading: 
    - If lots of clients get better execution, it can shift liquidity and affect others. 
    - If the retry increases load, it could affect platform latency for everyone.
    
    So you ask: 
    - Is this feature isolated per client? Mostly yes.
    - Does it change shared system behavior? Potentially, so guardrail latency is crucial.

    Key: “Who is treated?” + “Can treated units affect control units?”; there should be no systematic differences in pre-treatment characteristics between groups (on average)

4. Plan for peeking / stopping

    In product teams, people check results every day. If you do that naively, you inflate false positives. If you look 20 times, “something significant” will pop up. So you choose one approach: (a) fixed horizon (no early stopping), (b) sequential plan (advanced - you’re allowed to stop early with proper statistical control (alpha spending / group sequential))

    Key: Decide when you’ll make the decision to analyze, before you see the data.

5. Plan for multiplicity (multiple testing)

    Coinbase Institutional work is full of:
    - many metrics
    - many segments (client tiers, regions, strategies)
    - many time windows

    If you test enough things, you will [“discover” fake wins](!https://youtu.be/K8LQSvtjcEo?si=aFezzZzdXEg7vW2Y).

    So you pre-plan:
    - Primary metric: no adjustment (because it’s pre-specified and singular)
    - Secondary metrics: you can adjust or treat as supportive evidence
    - Segmentation: BH/FDR across segments (because this is discovery-ish)

    Rule of thumb: Shipping decision should not depend on “we found one segment where it worked” unless that segment was pre-planned or adjusted properly. 
    
    Key: Separate confirmatory (decision-driving) from exploratory (learning).

B. DURING THE EXPERIMENT

6. SRM (Sample Ratio Mismatch) / balance checks

    Before analysis, confirm randomization worked. Trust the experiment assignment only after you verify it.

    SRM means the split is not what you intended. Example: You planned 50/50 but got 60/40. That could indicate: logging issues/eligibility rule bug/bucketing bug.

    Balance checks: compare treatment vs control on pre-treatment characteristics like client tier distribution, baseline volume, baseline failure rate

    If balance fails, your causal story is weaker.

7. Logging correctness + metric sanity

    Before you trust any experiment result, you verify that the data is real and consistent. Most “wins” disappear because the data was logged wrong. If measurement is wrong, inference is meaningless.
    - As for logging correctness
        - Do we have data for both groups?
        - Are timestamps sane? If some events are recorded in a different timezone, orders may look like they happened outside the experiment window.
        - Are retries counted properly? (important here) If a retry is logged as a brand new order instead of being linked to the original order, you might inflate total orders (denominator),  inflate failures (if retries fail) or accidentally “create” duplicates
        - Are failure reasons consistent? If treatment started labeling failures differently (e.g., “transient_error” vs “api_error”), it can look like failures dropped when they just got reclassified.
    - and for metric sanity (do the numbers “smell right”?)
        - Baseline matches history? If order success rate is usually ~96% and suddenly your control is 90%, something is off (or there was an incident).
        - Counts look plausible? If one day has 10× more orders than normal, check if you duplicated events.
        - One whale dominates? If one big client produces 60% of all orders, your result might just be their behavior—not the feature.
    
    If you measure wrong, you’ll make a confident decision based on a fake reality.

8. Monitoring guardrails

    Guardrails aren’t just end-of-test; they are safety alarms. If duplicate executions spike, you pause the experiment. So you monitor duplicate rate daily, risk limit violations, p95 latency.

    Product DS is partly “statistics,” partly “safety engineer.”

C. AFTER THE EXPERIMENT

9. Effect size + confidence intervals (not just p-values)

    Your job is not to “get significance.” Your job is to quantify impact and uncertainty. You report:
    - effect size (difference between treatment and control)
    - 95% CI (range of plausible effects)

    Example: success rate +0.25 pp (CI: +0.05 to +0.45 pp)

    Also translate into business terms: “This prevents ~X failed orders/day across Y clients.”

    P-values are a side detail; effect size + CI drives decisions.

10. Multiple testing adjustment, robustness, and ship recommendation

    (a) Multiple testing adjustment (when relevant)
    - If you tested 30 secondary metrics, adjust those p-values (or treat as descriptive).
    - If you sliced 50 segments, apply BH/FDR across segments before claiming wins.

    (b) Robustness checks (make sure it’s not a fluke) 
    
    Common robustness checks in Coinbase context:
    - Remove top 1–2 whale clients and re-check (heavy tails)
    - Use an alternate metric definition (e.g., “success within 5 seconds”)
    - CUPED / pre-period adjustment (reduces variance)
    - Sensitivity checks for logging edge cases (retries counted correctly)

    (c) Ship / iterate / kill + rollout plan
    
    A strong recommendation includes:
    - what happened (effect + CI)
    - risk (guardrails)
    - confidence (robustness)
    - next step (ship to 10% → 50% → 100% with monitoring)

    Key point: Close the loop: analysis → decision → rollout plan.