# Decision log

Every non-obvious choice in the project, with the alternatives considered and the reason. Numbered
so the README and the write-ups can point at them. Newest additions at the end of each section.

## A. Scope and honesty

**A1. Simulated data with a planted truth, said up front.**
*Alternatives:* a public dataset (Criteo uplift, Hillstrom) or a fully synthetic benchmark with no
business story.
*Why:* the point of the project is to grade estimators against a known answer; no public dataset has
fee tiers, a holdout kept for 26 weeks, and per-client potential outcomes. The README's first
section says the data is simulated, every "truth" column in the results is labelled, and
`docs/SIMPLIFICATIONS.md` lists what a real exchange would add. A hiring manager should never
discover this on their own.

**A2. Numbers calibrated to the resume bullet, by the planted truth and not by the estimator.**
*Why:* the resume says ~50% variance reduction, ~+11% targeted net revenue, ~2%/week decay, ~40%
ROI overstatement, forest best on Qini/AUUC. The *simulator's* parameters were set so the *true*
effects land there (`configs/default.yaml`, `truth:` block); the estimators are then free to be
right or wrong. The Monte Carlo (A3) is what shows they are right on average. Tuning the
estimators or the seed until the estimate matched would have been the dishonest version.

**A3. A showcase seed chosen by a median rule, not by eye.**
*Why:* any single run of a 5,000-client experiment is one draw. `tierlift montecarlo` runs 40
seeds and writes bias, SD, reported SE and coverage for the headline estimators; the showcase
seed in the config is the one whose targeted-band estimate is closest to the Monte Carlo
*median* (`pick_showcase_seed`). That gives a typical run, not a lucky one, and the rule is
in the code so a reviewer can rerun it. The current showcase seed is 13.

**A4. Both scripts and notebooks, with no logic in the notebooks.**
*Why:* scripts are what gets tested, reviewed and run in CI; notebooks are how an analyst
reads a story. Every notebook cell imports from `tierlift`; `CONTRIBUTING.md` makes this a rule
so the two cannot drift apart.

**A5. Dropped the company branding of the earlier version.**
*Why:* the earlier README named a specific exchange's fee schedule. The schedule here is
illustrative (`configs/default.yaml`) and the text says "an institutional exchange".

## B. Simulator

**B1. Multiplicative volume model with log-normal size.** Quarterly volume is
`latent × drift × planted lift × market multipliers`, with size log-normal (σ = 0.7) across six
client segments. *Why:* real volume is heavy-tailed; most of the estimation difficulty in the
project (whales dominating dollar variance, index vs dollar outcome scales, D-scores chasing
noise) comes from this and would be invisible with Gaussian outcomes.

**B2. Drift split into a covariate-explained part and a noise part** (`drift_flow_coef` 0.57 on
`aum_flow_z`, `drift_log_sd` 0.12). *Why:* this is the lever that sets how much AIPW can beat
CUPED. CUPED only sees pre-period revenue; AIPW's outcome model also sees AUM flows. We assume
a strong CRM signal (most of the drift is predictable), say so in SIMPLIFICATIONS #5, and the
CUPED variance cut is unaffected (it is set by the pre/post correlation, ~0.71). An earlier
split (0.55 / 0.20) gave the same CUPED number and wider AIPW intervals.

**B3. Planted lift as a step function of distance to the next tier** (8% within 10%, 30% at
10–20%, 6% at 20–40%, 3% beyond, 0 at the top tier) with a log-normal client multiplier (σ =
0.25), scaled by engagement and *reduced* for API-heavy clients. *Why:* the business story
needs a band where the volume gain pays for the rate cut and buckets where it does not; the
engagement/API modifiers give the CATE learners something to find beyond the band, and the
Cox model something to explain.

**B4. Treatment also shifts the maker/taker mix** (`treated_maker_ratio_shift` −0.03). *Why:*
clients chasing a volume target trade more aggressively (taker). It makes the rate effect a
function of behaviour, not only of the fee rule, which is why the decomposition needs the
volume-weighted rate.

**B5. Three planted market confounders** (volatility spike in weeks 6–8, a competitor promo
hitting the Starter tier from week 4, a seasonal rally in weeks 10–13). *Why:* to make the
before/after comparison visibly wrong and the treated-vs-holdout comparison visibly right
(`11_weekly_effect.png`, write-up 10 pitfall 1).

**B6. Engagement lapse as a telemetry event with a proportional-hazards clock.** Each treated
client's lapse week is drawn from a PH model (base hazard 2%/week; account manager −, API share
+, engagement −), centred so the population hazard is the base. After lapse the planted lift
is gone. *Why:* this gives the survival chapter a real time-to-event with right-censoring.
*Rejected:* detecting lapse from volume. We tried a "back to baseline for 3 weeks" rule; with
weekly noise of 20–35% and lifts of 3–8% it had zero correlation with the true lapse week.

**B7. Stratified permuted-block randomisation** on tier × distance bucket × segment. *Why:*
the band is 8% of clients; simple randomisation could leave it unbalanced across three cells,
and the permutation test needs the strata anyway.

## C. Experiment analysis

**C1. CUPED with the pre-period revenue as the single covariate; Lin regression and AIPW as
the multi-covariate versions.** *Why:* CUPED is the industry standard and the resume number;
Lin (2013) is the textbook-correct regression with interactions and HC2; AIPW is the course's
double-robust estimator. Reporting all four on one axis (`01_ate_estimators.png`) shows that
the point estimates agree and the standard errors shrink as more pre-treatment information is
used.

**C2. Relative effects by the delta method on the ratio to the control mean.** *Why:* the
business talks in percentages; the delta method gives a standard error for the ratio without a
bootstrap. The Monte Carlo shows a ~2-point upward bias of the ratio for the small band (a
known small-sample property of ratio estimators) with correct coverage; it is documented
rather than "fixed", since the fix (a bootstrap-bias-corrected ratio) would obscure the
estimator.

**C3. The population-level test is reported as non-significant and that is fine.** The blanket
effect is −3% with an MDE of ~12% raw / ~8% CUPED; Welch, Mann–Whitney and the stratified
permutation test all give p > 0.1. *Why:* a real experiment of this size cannot detect a 3%
population effect, and the project says so (`power.csv`). The decision-relevant estimates are
the band (large effect, tight AIPW interval) and the policy values.

**C4. Holm correction across the two treatment cells.** *Why:* two primary contrasts share one
holdout; Holm is the simplest correction that controls the family-wise error without
assuming independence.

**C5. Observational counterfactual uses index-scaled outcome models.** The AIPW outcome models
predict `y / pre_revenue` and scale back. *Why:* with raw dollars the GBM extrapolated badly for
whales and the AIPW correction divided those residuals by small propensities; in the showcase
draw AIPW missed the truth by 5 SE. The index scaling halves the RMSE across repeated draws
(`observational_replication.csv`). The single-draw miss is still shown and explained in
write-up 4, because it is the honest lesson about observational estimates of small effects on
heavy-tailed outcomes.

## D. Uplift modelling

**D1. Outcome for the learners is the revenue index (experiment revenue / pre-period revenue).**
*Why:* in dollars every learner spent its capacity on client size and the S-learner never split
on treatment (regularisation bias); in index units treatment is a visible feature and all six
learners improve. Dollar CATEs for the policy tables are `τ̂_index × pre_revenue`. Write-up 5
section 5.6 has the comparison.

**D2. Known propensity passed as a constant** (`DummyClassifier(strategy="prior")`). *Why:* this
is an RCT; estimating a propensity that is known wastes variance and, for the DR/R-learners,
invites the overlap problems the course warns about.

**D3. Every learner's scores are out-of-fold, 4 folds × 2 repeats.** *Why:* in-sample scores
reward memorising noise (write-up 8 pitfall 1); repeated cross-fitting halves the
Monte-Carlo noise in the scores, which is visible as a steadier leaderboard middle.

**D4. CausalForestDML with 1,000 trees and `min_samples_leaf` 10.** *Why:* EconML's defaults
(100 trees) were noticeably noisier on 3,500 clients; 1,000 trees costs ~45 s. The 5-seed
benchmark replication uses 500 trees and one repeat to fit in a few minutes.

**D5. Oracle metrics (PEHE, rank correlation with the planted ITE) reported next to the
observed ones, labelled as oracle.** *Why:* their job is to validate the validators. They show
the forest is first at recovering the true ranking in every replication, while the observed
AUUC leaderboard reorders between seeds. Without them a reader could not tell whether a
leaderboard change was the model or the noise.

**D6. The policy tree is grown on index-scale DR scores and reported in-sample *and*
cross-fitted.** *Why:* grown on dollar scores it split on whale noise; grown on index scores
it finds the band and the maker-ratio refinement, but half-sample trees disagree on the first
split and the cross-fitted value is negative. We show both numbers because the gap between
them is the clearest demonstration in the project of why honest evaluation matters.
`honest=False` in EconML's tree, because halving 3,500 clients again made the splits worse;
honesty comes from the out-of-fold evaluation instead.

## E. Economics and durability

**E1. Decomposition prices the volume change at the old rate** (Laspeyres convention) and
keeps the interaction term explicit. *Why:* "what the extra trading would have been worth at
the price we used to charge" is the sentence the business understands; the interaction term
is small but dropping it breaks the identity, which a test checks.

**E2. Arm means in the decomposition are AIPW-adjusted** (`decomposition_incentive_adjusted.csv`),
with the raw version kept alongside. *Why:* with 4 top-tier clients and heavy tails, raw arm
means within a bucket can be driven by one whale landing in one arm; the adjusted means use
the covariates to remove that. The ROI projection reads the adjusted table.

**E3. Rollout estimates use AIPW scores fitted once on the full contrast and averaged over the
targeted clients** (`ContrastScores`). *Alternatives:* refit CUPED or AIPW inside the band.
*Why:* the band holds ~290 clients in the contrast; nuisance models refit on 290 clients are
poor and the interval was twice as wide (Monte Carlo: SE 0.084 vs 0.047, with coverage 90% vs
95%). Averaging full-sample influence scores over a pre-treatment-defined subgroup is unbiased
(randomisation holds inside the subgroup) and is exactly the doubly robust policy-value
estimator of course video 19, so the rollout table and the policy table use the same idea.
CUPED inside the band is still reported as the simple estimator.

**E4. ROI decays the volume gain but not the rate cost.** *Why:* a client who crossed a tier
keeps the cheaper rate next quarter whether or not they stay engaged. This is why the *net*
overstatement (90%) is larger than the *benefit* overstatement (41%): a flat cost is
subtracted from a decaying benefit.

**E5. Survival uses statsmodels (Kaplan–Meier via `SurvfuncRight`, Cox via `PHReg`).**
*Alternative:* lifelines. *Why:* lifelines' dependency chain (`autograd-gamma`) did not build in
the target environment; statsmodels is already a dependency and has both estimators.

**E6. The weekly curve's decay fit is the cross-check, the survival hazard is the headline.**
*Why:* the curve's decay rate has SE ≈ 0.5 points on 39 noisy weekly points; the pooled hazard
from 3,500 per-client event times has SE ≈ 0.04 points. Write-up 11 exercise 4 is this
decision.

## F. Engineering

**F1. One typed, validated config (`configs/default.yaml` → frozen dataclasses).** *Why:* every
number a reviewer might question lives in one file with a comment; `load_config(seed=…)`
overrides make Monte Carlo and tests one-liners.

**F2. Truth never leaves `truth.parquet`.** A test asserts no potential-outcome column appears
in the observed tables, and a data-quality check fails the run if a post-treatment column
reaches the feature matrix.

**F3. Small result tables and figures are committed; data and Monte Carlo seeds are not.**
*Why:* the README quotes numbers that a reader should be able to see without running three
minutes of code; the parquet files are reproducible from the seed in under two seconds.

**F4. CI runs lint, the unit tests, and the end-to-end pipeline on a 2,000-client `--fast`
configuration.** *Why:* the full run takes ~3.5 minutes and 4 cores; the fast run exercises
every code path in about a minute.

**F5. Per-worker thread limits in the Monte Carlo** (`threadpool_limits(1)`). *Why:* four
worker processes each spawning OpenMP threads oversubscribed the machine and the run stalled;
pinning each worker to one thread made the 40-seed replication finish in ~2 minutes.
