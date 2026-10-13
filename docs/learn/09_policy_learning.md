# 9. Policy learning: from "who responds" to "who to target"

*Course video 19.* *Where it shows up in this repo:* `src/tierlift/uplift/policy.py`,
`results/policy_comparison.csv`, `results/policy_tree.txt`, figure `08_policy_values.png`.

---

## 9.1 A CATE is not a decision

The forest hands you τ̂(x) for every client. The business asks a yes/no question per client:
*give them the incentive or not?* A **policy** is a function π(x) ∈ {0, 1}. Its **value** is the
average incremental outcome it produces:

```
V(π) = E[ π(X) · (Y(1) − Y(0)) ]        (= E[π(X) τ(X)])
```

If treating costs c per client, replace τ by τ − c. The best policy treats exactly the clients
with τ(x) > c. Simple in principle; the two questions are *how to estimate V(π) honestly* and
*how to find a π that people can act on*.

## 9.2 Evaluating any policy without a model of Y

Use the doubly-robust scores from write-up 6:

```
Γᵢ = μ̂₁(Xᵢ) − μ̂₀(Xᵢ) + Dᵢ (Yᵢ − μ̂₁(Xᵢ))/e − (1−Dᵢ)(Yᵢ − μ̂₀(Xᵢ))/(1−e)
V̂(π) = mean_i [ π(Xᵢ) · Γᵢ ]
```

Because E[Γᵢ | Xᵢ] = τ(Xᵢ), this is unbiased for V(π) for *any* rule π that depends only on
pre-treatment X, and its standard error is just the SE of the mean of π·Γ. The rule can be a
CATE threshold, a hand-written business rule, or "everyone" (blanket rollout). That is what
`compare_policies()` tabulates: every policy is scored with the same Γ, so the comparison is
apples to apples.

**A tiny example.** Five clients with DR scores Γ (in $K):

| client | Γ | near tier? |
|---|---|---|
| A | +8 | yes |
| B | +3 | yes |
| C | −2 | no |
| D | −5 | no |
| E | +1 | no |

* Blanket (treat all): V̂ = (8 + 3 − 2 − 5 + 1)/5 = **+1.0** per client.
* Rule "near tier": V̂ = (8 + 3)/5 = **+2.2** per client (the untreated contribute 0).
* Oracle "Γ > 0": V̂ = (8 + 3 + 1)/5 = **+2.4**.

Note the rule's value is divided by *all five* clients: it is the value *per client in the
population*, which is what you need to multiply by the client count to get dollars. Also note
that "Γ > 0" is cheating: Γ contains the noise of the client's own outcome. A real policy must
decide from X alone, which is why the learned policies use τ̂(x), not Γ.

## 9.3 Interpretable policies: the policy tree

`PolicyTree` (Athey & Wager, 2021; EconML) grows a shallow decision tree whose leaves say
"treat" or "don't", choosing splits to maximise V̂ computed with Γ. We grow it on the
*index-scale* scores (revenue / pre-period revenue, the same outcome the uplift learners use):
dollar scores are dominated by a few whales and a greedy tree then chases their noise. Depth 2,
so the whole policy is four leaves (`results/policy_tree.txt`):

```
if distance to next tier ≤ 20%:
    if maker ratio ≤ 0.43:   treat        (taker-heavy clients near the boundary)
    else:                    don't
else:
    if pre-period volume ≤ $12.6M:  treat   (small, far-from-tier clients)
    else:                           don't
```

The first leaf is the right answer with a refinement an analyst might miss: taker-heavy
clients pay a higher blended fee, so their volume gain is worth more per dollar. The third
leaf is noise: small clients far from any tier have a planted lift of 3–6% and a 4% rate cut,
and the tree found a subgroup where the noisy scores happened to average positive.

## 9.4 What the comparison shows

`results/policy_comparison.csv`, `value_per_client` in dollars with its SE, every row scored
with the same cross-fitted Γ:

| policy | value / client | what it means |
|---|---|---|
| Blanket rollout | −$2,209 ± 904 | the rate cut on 4,000 far-from-boundary clients outweighs their small lift |
| Rule: 10–20% from tier | +$550 ± 201 | the band rule works and is cheap to explain (8% of clients) |
| CATE > 0: S-learner / CausalForestDML | +$645 ± 167 / +$593 ± 164 | the two best learners' own threshold rules, targeting 18–20% of clients |
| CATE > 0: T / X / R / DR-learner | +$170 to −$449, SE ≈ 450–530 | noisier learners target 36–40% of clients and gain nothing |
| Policy tree, **in-sample** | +$642 ± 207 | looks as good as the forest… |
| Policy tree, **cross-fitted** | −$886 ± 605 | …but grown on half the clients and scored on the other half, it is worthless |

Two lessons the table teaches better than any slide:

1. **Honest evaluation is not optional.** The tree's in-sample value is +$642 and its
   cross-fitted value is −$886. The in-sample number is what you get when the same noisy
   scores choose the splits and then grade them (Pitfall 1). With 3,500 clients, a depth-2
   tree is not stable enough to trust; the learners' threshold rules and the hand-written
   band rule are, because they are evaluated on clients they never saw.
2. **The simplest policy is close to the best.** The band rule (+$550) is within one SE of the
   forest's rule (+$593) and the S-learner's (+$645), and it fits on one line of a product
   spec. The learned rules' extra value comes from the refinement the tree also found
   (taker-heavy clients); whether it is worth the explanation cost is a business call.

The experimental rollout table (`results/rollout.csv`, write-up 10) tells the same story with
a different estimator (AIPW on the band): targeted ≈ +12% net revenue on the targeted clients,
blanket ≈ −3% on everyone.

## 9.5 Pitfalls

1. **Learning and evaluating a policy on the same clients.** The policy will look better
   than it is. `compare_policies` scores the CATE-threshold policies with cross-fitted τ̂,
   Γ is cross-fitted too, and the policy tree is reported both ways so you can see the gap
   (`policy_tree_oof`).
2. **Policies that use post-treatment information.** "Treat everyone who crossed a tier" is
   not a policy you can apply before treating.
3. **Ignoring costs.** The incentive's cost (rate compression) is inside Y here; a campaign
   with a fixed cost per client needs `cost_per_client` in `cate_policy()`.
4. **Treating the top-k by score when k is not the business constraint.** If the budget is
   "at most 500 clients", rank by τ̂ and stop at 500; if the constraint is "only positive
   expected value", threshold at the cost. Different questions, different policies.

## 9.6 Exercises

1. In the five-client example, a campaign costs $1.5K per treated client. Recompute the
   values of blanket, rule and oracle policies.
2. Explain why V̂(π) = mean(π·Γ) has a valid SE even when π was chosen *after* seeing the
   data, provided π depends on X only and Γ is cross-fitted.
3. A forest says τ̂ = +$300 ± $900 for a client. The cost is $100. Would you treat them under
   "maximise expected value"? Under "only treat when confident"? Which objective does the
   business usually mean?
4. Depth-2 policy trees can express at most four leaves. Write a rule with five distinct
   segments that *cannot* be represented, and say what you lose.
5. The policy tree's two halves (grown on either half of the data) disagreed on the first
   split. Propose two changes, one to the data and one to the method, that would make the
   tree stable, and say what each costs.

<details><summary>Answers</summary>

1. Subtract 1.5 from Γ for each treated client. Blanket: (6.5 + 1.5 − 3.5 − 6.5 − 0.5)/5 =
   −0.5. Rule: (6.5 + 1.5)/5 = +1.6. Oracle (Γ − 1.5 > 0 → A, B): same +1.6. E is no longer
   worth treating.
2. π(X) is a fixed function of pre-treatment features, so π(Xᵢ)·Γᵢ are i.i.d. draws with mean
   V(π); the CLT applies. If Γ were fitted in-sample it would correlate with π's choice and
   the SE would be understated.
3. Expected value: yes, +300 > 100. Confident-only: no, the CI covers zero. Businesses
   usually mean expected value for a cheap reversible action and confidence for an
   expensive or irreversible one.
4. E.g. tier (4 levels) × near/far = 8 cells, or any rule with three thresholds on one
   feature. A depth-2 tree must merge segments; you lose whatever value lives in the
   distinctions it merges, in exchange for a rule that fits on one slide.
5. Data: more clients, or less noisy scores (a longer experiment, or CUPED-style
   residualisation before computing Γ). Method: grow the tree on *smoothed* scores (a
   learner's cross-fitted τ̂ instead of the raw Γ) or restrict the candidate split variables
   to the ones the business can act on. Each costs either time/money or the guarantee that
   the tree is evaluated on the same scores it was grown on.

</details>
