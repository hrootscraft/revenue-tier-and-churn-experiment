# 1. Potential outcomes: ATE, CATE, ITE, and why randomisation works

*Where it shows up in this repo:* `src/tierlift/simulate/outcomes.py` literally builds
every client's potential outcomes; `results/ate_*.csv` estimates the ATE;
`docs/learn/05`–`09` are all about the CATE.

---

## 1.1 The question we are actually asking

"Did the Tier Accelerator incentive increase revenue?" sounds like one question, but it hides
a comparison between two worlds for the *same* client:

* the world where client *i* got the incentive, with revenue **Y_i(1)**, and
* the world where client *i* did not, with revenue **Y_i(0)**.

The **individual treatment effect** is the gap between those worlds:

```
ITE_i = Y_i(1) − Y_i(0)
```

Here is the whole difficulty of causal inference in one line: **we only ever see one of the
two**. If client *i* got the incentive we observe Y_i(1) and Y_i(0) is a counterfactual. This is
called the *fundamental problem of causal inference* (Holland, 1986).

> **Intuition.** Think of each client as a coin with two faces, "treated" and "untreated",
> each face stamped with a revenue number. The coin lands on one face. You can look at the face
> that is up, never at the face that is down.

In this project we cheat on purpose: the simulator writes *both* faces to `data/truth.parquet`
(`revenue_holdout`, `revenue_incentive`). That is why we can grade every method against the
truth. With real data you never can, and that is why the methods in the later write-ups exist.

## 1.2 Three effects, three questions

| Quantity | Definition | Question it answers | Observable? |
|---|---|---|---|
| **ITE** | Y_i(1) − Y_i(0) | "What did it do for *this* client?" | Never |
| **ATE** | E[Y(1) − Y(0)] | "What does it do *on average*?" | Yes, with randomisation |
| **CATE** | E[Y(1) − Y(0) \| X = x] | "What does it do for clients *like x*?" | Yes, with randomisation + modelling |

CATE is a function (of the client's features), the ATE is a number, the ITE is a number per
client that we can never know. The uplift models in write-up 5 estimate the CATE function;
the "targeting" decision in write-up 9 uses it.

## 1.3 A six-client example you can do by hand

Six clients. Both potential outcomes are written down (quarterly revenue in $K):

| client | Y(0) no incentive | Y(1) incentive | ITE |
|---|---|---|---|
| A | 50 | 56 | +6 |
| B | 80 | 78 | −2 |
| C | 30 | 39 | +9 |
| D | 120 | 112 | −8 |
| E | 40 | 46 | +6 |
| F | 60 | 63 | +3 |

**True ATE** = mean of the ITE column = (6 − 2 + 9 − 8 + 6 + 3) / 6 = **+2.33**.

Now suppose only A, C, E got the incentive and B, D, F did not. We observe:

| client | treated? | observed Y |
|---|---|---|
| A | yes | 56 |
| B | no | 80 |
| C | yes | 39 |
| D | no | 120 |
| E | yes | 46 |
| F | no | 63 |

Difference in means = (56 + 39 + 46)/3 − (80 + 120 + 63)/3 = 47 − 87.67 = **−40.67**.

The true effect is +2.33 and we estimated −40.67. Nothing is wrong with the arithmetic. The
treated group (A, C, E) happened to be the *small* clients. Their untreated outcomes
(50, 30, 40) would have been low anyway. The comparison mixes the treatment effect with a
pre-existing difference between the groups. That pre-existing difference is **selection bias**.

## 1.4 The decomposition that explains everything

Write the difference in means as (treated mean of Y(1)) − (control mean of Y(0)). Add and
subtract the treated group's *untreated* mean, E[Y(0) | D=1]:

```
E[Y | D=1] − E[Y | D=0]
  = E[Y(1) | D=1] − E[Y(0) | D=0]
  = { E[Y(1) | D=1] − E[Y(0) | D=1] }   +   { E[Y(0) | D=1] − E[Y(0) | D=0] }
    \___________ ATT ____________/         \________ selection bias ________/
```

* The first bracket is the effect on the treated (ATT): the treated group's real gain.
* The second bracket compares the treated and control groups *in the untreated world*.
  It is zero only if the two groups would have earned the same without treatment.

In the toy example: ATT = (6 + 9 + 6)/3 = +7; selection bias = (50 + 30 + 40)/3 − (80 + 120 + 63)/3
= 40 − 87.67 = −47.67; and indeed 7 − 47.67 = −40.67. ✔

## 1.5 Why randomisation works

If a coin flip decides who is treated, the flip knows nothing about Y(0) or Y(1). Formally,

```
(Y(0), Y(1)) ⟂ D          (independence from assignment)
```

so E[Y(0) | D=1] = E[Y(0) | D=0] = E[Y(0)]: the selection-bias bracket is **exactly zero in
expectation**, and the difference in means is an unbiased estimate of the ATE. No model, no
assumption about functional form, nothing to get wrong. That is why the experiment in this
project randomises 5,000 clients into cells (`src/tierlift/simulate/assignment.py`) instead
of letting them opt in.

Two refinements you will meet:

* **Stratified randomisation** flips the coin *within* groups (tier × distance bucket ×
  segment), so chance imbalance on those variables is impossible rather than just unlikely.
  It is covariate adjustment done at design time.
* **Unconfoundedness** (write-up 4) is the assumption that lets us *pretend* we randomised
  within levels of X when we did not. It is an assumption; randomisation is a fact.

## 1.6 How it shows up in this repo

* `src/tierlift/simulate/outcomes.py` computes `revenue_holdout`, `revenue_nudge`,
  `revenue_incentive` for every client with the *same* random noise, and then reveals only
  the one for the assigned cell. The function docstring walks through it.
* `tests/test_simulate.py::test_potential_outcomes_consistent` checks that the observed
  outcome equals the potential outcome of the assigned cell: the *consistency* assumption.
* `results/observational.csv` (write-up 4) repeats the toy example at scale: it lets clients
  opt in and shows the naive estimate is off by a factor of several.
* `results/ate_incentive.csv` is the randomised answer.

## 1.7 Pitfalls

1. **ATE ≠ ATT in general.** They coincide under randomisation. Under self-selection the
   people who opt in may be exactly those with large effects.
2. **"We controlled for X" is not randomisation.** It replaces a fact with an assumption
   (unconfoundedness). Sometimes that is all you have; say so.
3. **Interference / SUTVA.** We assumed Y_i depends only on D_i. On a shared order book a
   treated client's extra volume could affect a control client. `docs/SIMPLIFICATIONS.md`
   lists this as a known simplification.
4. **Conditioning on post-treatment variables** (e.g. "among clients who crossed the tier")
   re-introduces selection bias, because crossing is itself an outcome. `features.py` only
   uses pre-treatment variables for this reason.

## 1.8 Exercises

1. In the six-client table, pick the treated group {B, D, F} instead. Compute the difference
   in means, the ATT and the selection bias, and check they add up.
2. There are C(6,3) = 20 ways to choose 3 treated clients. If each is equally likely
   (randomisation), what is the *average* of the difference-in-means over all 20? (You can
   reason it out without enumerating.)
3. A product manager says: "Clients who used the progress bar traded 30% more than clients who
   didn't, so the progress bar causes a 30% lift." Write the decomposition of 1.4 for this
   claim and name the bracket she is ignoring.
4. Explain in one sentence why `features.py` excludes `crossed_tier` from the uplift models.

<details><summary>Answers</summary>

1. Observed: B=78, D=112, F=63 treated; A=50, C=30, E=40 control. DiM = 84.33 − 40 = +44.33.
   ATT = (−2 − 8 + 3)/3 = −2.33. Selection bias = (80+120+60)/3 − (50+30+40)/3 = 86.67 − 40 =
   +46.67. Check: −2.33 + 46.67 = 44.33. ✔ Same data, opposite conclusion from exercise 1.3.
2. Under randomisation every client is treated in exactly half of the 20 assignments, so the
   expected treated mean is the mean of all Y(1) and the expected control mean is the mean of
   all Y(0); the average DiM is the true ATE, +2.33.
3. E[Y | used] − E[Y | didn't] = {effect on users} + {E[Y(0) | used] − E[Y(0) | didn't]}. She
   ignores the second bracket: engaged clients who open the dashboard would trade more
   anyway.
4. Crossing the tier is caused by the treatment, so conditioning on it compares treated
   clients who responded with control clients who happened to grow: selection bias by
   construction.

</details>
