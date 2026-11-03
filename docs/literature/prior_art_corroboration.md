# Prior art attack on "corroboration-gated consolidation"

Written 2026-11-04. Adversarial review: the goal was to find that the idea is already done or that its source is misread.

## The idea under attack

A surprising sample has an ambiguous cause: a real change (raise plasticity) or a one-off outlier (lower plasticity). At the moment of surprise these cannot be told apart. So hold surprising samples in a fast buffer with their weight updates deferred, and commit them to the slow learner only if later data corroborates them; otherwise discard. Claim: this beats both monotonic surprise-gated learning rates and robust losses on a stream that mixes both causes.

## Source tags (read this first)

- FULL = body text of the paper retrieved as raw page text through alphaXiv `answer_pdf_queries`. The tool returns the most relevant pages, not necessarily every page, so "FULL" means "substantial body text read", not "every page read".
- FULL-WF = fetched with WebFetch, which returns a summary written by a small model. Every number is second-hand and should be re-checked against the PDF before it goes in a paper.
- ABSTRACT = only an abstract or description was seen.
- SNIPPET = only a search-result description was seen.
- Peer-review status is taken from each paper's own header. Several 2025-2026 items are arXiv preprints that I cannot independently check. They are marked PREPRINT.
- Nothing here is filled from memory unless labelled "inference" (my reasoning) or "from memory".

| Id | Source | URL | Tag |
|---|---|---|---|
| S1 | Nassar, Bruckner, Frank 2019, eLife | https://elifesciences.org/articles/46975 | FULL-WF (first 100,000 of 140,822 chars; a second fetch of the remainder returned only references and ethics text) |
| S2 | Altamirano, Briol, Knoblauch 2023, ICML (Dm-BOCD) | https://arxiv.org/abs/2302.04759 | FULL |
| S3 | Duran-Martin et al. 2025, TMLR (BONE) | https://arxiv.org/abs/2411.10153 | FULL |
| S4 | Duran-Martin et al. 2024, ICML (WoLF) | https://arxiv.org/abs/2405.05646 | FULL |
| S5 | Kim et al. 2021, ICCV (SPR) | https://arxiv.org/abs/2110.07735 | FULL |
| S6 | Karim et al. 2022 (CNLL) | https://arxiv.org/abs/2204.09881 | FULL (PREPRINT status not checked) |
| S7 | Bang et al. 2022, CVPR (PuriDivER) | https://arxiv.org/abs/2203.15355 | FULL |
| S8 | Han et al. 2018, NeurIPS (Co-teaching) | https://arxiv.org/abs/1804.06872 | FULL |
| S9 | Shinohara et al. 2026 (BIB, surprise nullification) | https://arxiv.org/abs/2609.10927 | FULL, PREPRINT |
| S10 | Wang, Sun, Wang, Tang 2025 (dual-channel outlier/drift) | https://arxiv.org/abs/2512.12289 | FULL, PREPRINT |
| S11 | Chen and Wu 2025 (BOCD-AR) | https://arxiv.org/abs/2508.06385 | FULL, PREPRINT |
| S12 | Yiu et al. 2026 (BR-iHMM), header says ICML 2026 | https://arxiv.org/abs/2604.14322 | FULL |
| S13 | Oda 2026 (reuse/spawn/defer) | https://arxiv.org/abs/2608.19888 | FULL, PREPRINT, single author |
| S14 | Klymenko 2026 (process vs instrument vs data-path) | https://arxiv.org/abs/2610.18904 | FULL, PREPRINT, independent researcher |
| S15 | Wang et al. 2026 (Age of Learning) | https://arxiv.org/abs/2609.32593 | FULL, PREPRINT |
| S16 | Zhang et al. 2026 (GAIN, gain-guided CTTA) | https://arxiv.org/abs/2609.36655 | FULL, PREPRINT |
| S17 | Mishra 2026 (RDumb++) | https://arxiv.org/abs/2601.15544 | FULL, PREPRINT |
| S18 | Liakoni et al. 2021 (Bayes Factor Surprise) | https://arxiv.org/abs/1907.02936 | FULL |
| S19 | Gershman, Radulescu, Norman, Niv 2014, PLoS Comput Biol | https://pmc.ncbi.nlm.nih.gov/articles/PMC4222636 | FULL-WF (first 100,000 of 122,354 chars) |
| S20 | Song, Jones, Monfils, Niv 2022 (gradual extinction, latent cause) | https://arxiv.org/abs/2205.04670 | FULL |
| S21 | Gershman, Blei, Niv 2010, Psychol Rev | https://www.cs.columbia.edu/~blei/papers/GershmanBleiNiv2010.pdf | SNIPPET (the PDF could not be decoded by the fetch tool; description comes from a web-search result) |
| S22 | Bifet and Gavalda 2007 (ADWIN) | https://paperswithcode.com/paper/learning-from-time-changing-data-with | SNIPPET |
| S23 | BONE code | https://github.com/gerdm/BONE | SNIPPET (URL from S3 and a search result; repository not opened) |

Not opened at all: Adams and MacKay 2007, Wilson/Nassar/Gold 2010, Fearnhead and Rigaill 2019, Sellier and Dellaportas 2023, Reimann 2024, the original ADWIN paper, Yu and Dayan 2005, the COIN contextual-inference model (Heald et al.), any paper named PuriDivER or SPR beyond the arXiv versions above.

## 1. Verdict on novelty

**Partly novel. As a principle it is already done, repeatedly, in 2025-2026 work. The narrow part that survives is empirical and engineering: a deferred weight commit in a deep online learner, tested head to head against a robust loss and a joint robust-changepoint filter on a controlled mixed-cause stream.**

Do not write "no one has deferred attribution of a surprise". At least five items I opened do exactly that in simpler models, and one of them states it as a named principle:

- S9 (Sep 2026): "Deferring attribution and letting subsequent observations differentiate the responses may constitute a principle of adaptive inference in non-stationary environments."
- S12 (ICML 2026 per its header): switching regimes only after pooling evidence over a batch.
- S10 (Dec 2025): outlier confirmed only if the next residual returns to normal.
- S11 (Aug 2025): changepoints declared only after at least five post-change points; collective anomalies confirmed only after 32 more points on real data.
- S13 (Aug 2026): "do not decide yet" as a formally defined action.

## 2. Prior art, one entry per item

For each: what it does; whether it defers commitment; whether it handles both causes in one stream; benchmark; result.

### Robust changepoint and filtering (Bayesian)

**S2 Dm-BOCD (Altamirano 2023).**
- Does: BOCD with a diffusion-score-matching generalised posterior, so outliers do not look like changepoints.
- Defers: no. A robust loss, applied at the moment of the observation.
- Both causes: yes in the sense that it must not fire on outliers and must still fire on real changes.
- Benchmark: synthetic, 600 samples, 2% outliers, 6 changepoints, 10 runs; also Twitter flash crash, well-log, crypto, bond yield.
- Result: PPV 0.907 +/- 0.154 vs 0.600 +/- 0.128 for standard BOCD; TPR 0.883 vs 0.833; delay 1.64 vs 1.05. The paper says robust changepoint methods can have latency and claims it did not observe it. No learner is trained; this is segmentation.

**S4 WoLF (Duran-Martin 2024, ICML).**
- Does: Kalman and extended Kalman filtering with the likelihood scaled by a weight W(y, y_hat) that falls as the innovation grows (IMQ, Mahalanobis, or hard threshold).
- Defers: no. It down-weights or rejects immediately. The hard-threshold variant (WoLF-TMD) is "detect and reject": a large error is simply ignored.
- Both causes: no. It models outliers only. The authors list robustness to outliers in the state process as an open limitation.
- Benchmark: 2D tracking (Student-t and mixture outliers), Lorenz96, online neural-network fitting on corrupted UCI data.
- Result: matches or beats the variational robust filters (KF-B, KF-IW) and Huberised EnKF at KF-like cost. It has a closed form and a bounded posterior influence function. Code URL given in the paper: https://github.com/gerdm/weighted-likelihood-filter (repository not opened).

**S3 BONE (Duran-Martin 2025, TMLR).**
- Does: one framework (measurement model, auxiliary variable, conditional prior, posterior algorithm, weighting algorithm) that covers BOCD, mixtures of experts, Kalman filters, Soft Reset, Titsias et al., and robust variants.
- Defers: no member defers sample commitment. The discrete auxiliary variable keeps K hypotheses (RL[K]-PR), which is a posterior-weight form of non-commitment.
- Both causes: yes, as a combination. Section 4.3.2 combines WoLF-IMQ with BOCD run-length (WoLF+RL[inf]-PR*) on piecewise regression with Student-t errors (df 2.01, change probability 0.01, 30 trials).
- Result: the robust variant has the lowest RMSE; plain BOCD over-segments and resets repeatedly on outliers (figures only, I captured no numbers). The paper says explicitly that "an observation that is unusual may trigger a changepoint unnecessarily".
- Also: its experiments use small MLPs and logistic regression, with MAP point estimates for the network. Its table lists Soft Reset and shrink-and-perturb style priors as members.

**S18 Liakoni 2021, Bayes Factor Surprise.**
- Does: shows exact Bayesian inference in a change model gives learning rate gamma = mS/(1 + mS), and that with an outlier generative model surprise should attenuate learning ("integrating and ignoring"), citing Nassar 2019.
- Defers: no. The posterior carries both hypotheses at once.
- Both causes in one stream: no. The two cases are treated as separate generative models. Return to a previous state is called out of scope.
- Importance for the attack: it is the normative statement that the sign of the surprise effect depends on the assumed generative model. It does not need a buffer.

### Joint outlier and change handling with confirmation (closest prior art)

**S9 Shinohara 2026 (BIB, "surprise nullification").** The nearest match to the idea.
- Does: on a large prediction error it does not choose a cause. It weakens both the belief constraint and the likelihood constraint by a common factor. The resulting "nullification strength" is NOT applied to the update for that observation. It is held as a candidate for the next step, and the next observation decides whether to keep it or reset to ordinary Bayesian updating.
- Defers: yes, by one step. Quote (paper's words): the candidate "does not retrospectively alter the update at time t. Instead, its adoption or rejection is evaluated only when the observation at time t + 1 becomes available."
- Both causes in one stream: yes. Changepoints and outliers occur at equal rates, and the observation at the event is drawn from the same U(0,100) for both, so they are indistinguishable at the event.
- Benchmark: 1-D mean tracking, simulation only, rates 0.001, 0.01, 0.05. Baselines: Sage-Husa adaptive Kalman filter, fixed-strength BIB, belief-forgetting-only filter, and an oracle reduced Bayesian model (the Nassar 2010 plus Nassar 2019 three-hypothesis mixture, given the true rates).
- Result: no scanned setting of the Sage-Husa, fixed-strength or forgetting-only variants beats BIB on both changepoint tracking and post-outlier stability. After a changepoint the reset rate is 0.34 and applied strength 0.58; after an outlier the reset rate is 0.88 and strength 0.09.
- Limits reported by the authors: the oracle model has lower overall RMSE in every condition; at 0.05 event rate some baseline settings beat BIB on overall RMSE; delayed application costs responsiveness at the event; results are simulation only; they say it is unknown whether humans do this.
- Not hit: no neural network, no robust-loss baseline, no label noise.

**S10 Wang et al. 2025 (dual-channel outlier and drift, stream regression).**
- Does: a rapid channel flags a residual above a warning level, then checks whether the next residual returns to normal; if yes it is a confirmed outlier, if not the residual goes to a drift detector (EWMAD-DT) that also separates abrupt from incremental drift. On drift the window and detector reset and a robust learner is refit.
- Defers: yes, one step ("we are unable to definitively classify its state... wait for at least one additional time point").
- Both causes in one stream: yes (datasets labelled Mixed).
- Benchmark: synthetic mixed streams and seven real datasets; baselines ADWIN, KSWIN, Page-Hinkley, DSA as drift detectors; five robust regressors (IPOD, ARLF, Huber, RANSAC, Theil-Sen) as learners.
- Result: higher drift F1 under outlier contamination than the drift-detector baselines; on the Worker dataset MAPE* about 4.5e-2 vs above 6e-2 for ADWIN and KSWIN (read from text, figure-level otherwise). The paper reports that ADWIN and KSWIN treat all residual deviations alike and false-alarm under outliers.
- Gap: the learner is a linear regression refit on a window after drift, not an online weight update that is held back.

**S11 Chen and Wu 2025 (BOCD-AR, collective anomalies vs change points).**
- Does: a Bayesian online model with two binary variables per time step (anomaly, change); an anomaly is short and reversible, a change is persistent. Change points "are declared only after sufficient post-change data have been observed", and detected anomalies are removed.
- Defers: yes, by a hyperparameter (at least 5 post-change points in simulation; 32 subsequent points on real hourly data).
- Both causes: yes.
- Benchmark: 1000 series of length 1000, six changepoints, an anomaly every 100 steps of length 1 or 4; baselines include GOT, COT, TOD, JEPO, BARD, CAPA, BOCPD, RWBS, ARC.
- Result: anomaly precision 0.947 and F1 0.902 vs CAPA 0.599 and 0.667; changepoint precision, F1, false-positive rate and delay more than 10% better than baselines per the paper. Detection accuracy drops when the assumed maximum anomaly length is below the true length (8 sweeps over that hyperparameter).
- Gap: detection, not learner updates.

**S12 Yiu et al. 2026 (BR-iHMM).** The strongest rival conceptually.
- Does: an online infinite HMM with WoLF-robust parameter updates plus "batched" state inference: the model "accumulates evidence over a batch of size B > 1 before allowing a state transition". In the pseudocode, batch likelihoods are computed first, states for the batch are then sampled, and only then are the per-observation parameter updates applied. So parameter updates are in effect held back until the batch assignment is made.
- Theorem 4.1 in the paper: observation-space robustness alone (WoLF) is not sufficient, because an extreme observation can still create a new regime. This is the same point the idea makes.
- Defers: yes, a fixed batch of B, not selective to surprising samples.
- Both causes in one stream: yes.
- Benchmark: synthetic linear regression, d = 100, 3 regimes, 1% outliers uniform on [-600, 600], 100 runs; hourly electricity demand (31,912 points); order-flow imbalance; well-log segmentation.
- Result (one-step RMSE, mean over 100 runs): synthetic 46.1 vs WoLF-iHMM 103.8, iHMM 101.7, BOCD 123.1, offline iHMM 2.9; electricity 0.47 vs 0.63, 0.57, 0.80, offline 0.32. Regime detection: PPV 0.963 and TPR 0.996 at delay 3.4, vs WoLF-iHMM 0.533, 0.344 and delay 0. On order-flow imbalance the tuned batch size was B = 1 (no batching), because regime changes were fast and not outlier driven; RMSE 0.616 vs 0.623 for WoLF-iHMM. Larger B increases detection delay.
- Gap: linear-Gaussian emissions, particle learning, no deep network; code link not found in what I read.

**S13 Oda 2026 (reuse, spawn or defer).** Adjacent.
- Does: for a pool of expert models, each chunk is reused, spawned, or deferred; "defer is exactly the region between accumulated evidence for reuse and accumulated evidence for spawn", with betting e-processes and anytime-valid control.
- Both causes: no. The benchmark has abrupt switches, a covariate-only phase, gradual drift and recurrence. No label noise or outlier condition in what I read.
- Result: zero false spawns and zero missed concepts on a four-regime synthetic stream; 13 experts vs 18-52 on INSECTS; repeated peeking commits 64% of borderline cases to a near coin flip while the anytime-valid scheme defers 90-97% of them.
- Value to the attack: it shows deferral can be given a formal statistical meaning, and that repeated looking is itself a failure mode (relevant to a corroboration test that is checked every step).

**S14 Klymenko 2026 (process change vs instrument fault vs data-path corruption).** Adjacent.
- Does: a bank of classical detectors and a rule that opens a 40-sample decision window and defers up to 240 samples before naming a cause. Ablation: without deferral, level-shift attribution falls from 99.0% to 0%.
- Both causes: yes (three).
- Important negative: Propositions 1 and 2 prove that a persistent bias step or drift in the instrument is not separable from a real process shift by any test on one channel. 2 of 207 detected sensor drifts were attributed to the instrument.
- Consequence for the idea (inference): corroboration by later data from the same channel cannot separate a persistent corruption from a real change. A poisoned or mislabelled run that persists will corroborate itself.

### Deep learning: label noise and continual learning

**S5 SPR (Kim 2021).**
- Does: a "delayed buffer" D holds the incoming stream; when full, a Self-Centered filter (eigenvector centrality on self-supervised features, per class, plus a Beta mixture) moves the likely-clean samples to a purified buffer P. The base network is trained self-supervised on D plus P. Supervised fine-tuning uses only P.
- Defers: yes in a loose sense. Samples wait in D until the buffer fills, then are filtered as a batch. The "corroboration" is agreement among peers in the same buffer (largest cluster per class), not later data.
- Both causes: no. It assumes noisy labels only. Tasks are disjoint class pairs, so a "change" is a new class that never contradicts earlier labels.
- Benchmark: MNIST, CIFAR-10, CIFAR-100 (20 tasks), WebVision.
- Result: CIFAR-100 final accuracy 21.5 / 21.1 / 18.1 at 20 / 40 / 60% random symmetric noise vs 19.3 / 13.8 / 8.8 for GDumb plus SL. Filtered-noise rate in the buffer on CIFAR-10 symmetric 20 / 40 / 60%: 75.5 / 70.5 / 54.3 (E_max 5). The paper itself reports that on abstract WebVision classes ("Cinema", "Spiral") random-sampling methods beat its filter (Table 9, class "Spiral": Self-Replay, which samples at random, 23.2, and GDumb 8.6, vs the Self-Centered filter alone 4.8 and SPR 9.0).

**S6 CNLL (Karim 2022).**
- Does: also a delay buffer, warm-up on it, a class-masked MSE loss, and a split at the mean loss into clean and noisy sets; noisy samples are kept and used semi-supervised (FixMatch).
- Both causes: no (label noise only); tasks are class pairs.
- Result vs SPR on CIFAR-10 symmetric 20 / 40 / 60%: 68.7 / 65.1 / 52.8 vs 43.9 / 43.0 / 40.0. CIFAR-100 random symmetric: 38.7 / 32.1 / 26.2 vs 21.5 / 21.1 / 18.1. Code: https://github.com/nazmul-karim170/CNLL (from the paper).

**S7 PuriDivER (Bang 2022).**
- Does: online blurry continual learning with noisy labels; an episodic memory chosen by a score that trades purity against diversity, and a GMM on loss splitting memory into clean, relabel and unlabeled sets.
- Defers: no, apart from the memory being a buffer.
- Both causes: no; the paper's limitation says no new class is added as tasks progress and task classes overlap but the label function does not change.
- Result: CIFAR-10 symmetric 20 / 40 / 60%: 61.3 / 59.2 / 52.4 vs 61.3 / 50.9 / 34.9 for the best baseline (RM plus DivideMix); CIFAR-100: 35.6 / 33.4 / 28.8. Code: https://github.com/clovaai/puridiver (from the paper).

**S8 Co-teaching (Han 2018).**
- Does: two networks; in each mini-batch each keeps its R(T) smallest-loss samples and trains its peer on them. R(T) = 1 - min(T/T_k * tau, tau), tau set to the noise rate.
- Defers: no. Selection is immediate, within the mini-batch, using the memorisation effect across epochs.
- Both causes: no. Assumes stationary noisy labels over repeated epochs.
- Result: CIFAR-10 pair-45%: 72.62 vs MentorNet 58.14. MNIST pair-45%: 87.63 vs 80.88.
- Inference (not from the paper): in a single-pass stream with real concept change, a genuine change produces large losses and a small-loss rule would drop it. That is the mirror image of the monotone-surprise failure the idea attacks.

**S15 Age of Learning (2026).** Adjacent. Persistence of a prediction error as a training signal (up-weights errors that persist, resets on a correct prediction). Streams of long-tailed class frequency with Period-1 and Period-10 changes. The paper's own limitation: persistently mislabeled or out-of-distribution samples can accumulate large AoL without helping, and it says robust variants for noisy labels are future work. It treats persistence as a reason to learn more, not as a corroboration gate against noise.

**S16 GAIN (2026) and S17 RDumb++ (2026).** Adjacent, in continual test-time adaptation. GAIN: "history proposes, gain decides", i.e. accumulated target statistics propose a correction and a posterior-predictive evaluator decides how strongly to apply it; the source model stays frozen. RDumb++: entropy or KL z-score drift detection triggers a full or soft parameter reset. Neither separates noise from drift; RDumb++ lists pseudo-label noise as an assumption that may fail.

### Drift detectors

**S22 ADWIN (Bifet and Gavalda 2007), SNIPPET only.** Adaptive window with a user-set false-positive probability; it detects a change in the mean of the monitored stream. It does not separate an outlier from a change by design. Evidence from S10 (FULL): ADWIN and KSWIN treat all residual deviations homogeneously and their drift precision drops as the outlier ratio rises. I did not open DDM, EDDM or the original ADWIN paper.

### Latent cause models in AI and psychology

**S21 Gershman, Blei, Niv 2010, SNIPPET only.** Online Bayesian inference over an unbounded number of latent causes; explains renewal and latent inhibition. I could not read the text, so I make no claim about how it treats gradual attribution beyond this description.

**S19 Gershman 2014 (FULL-WF).** A nonparametric switching Kalman filter. Inference is a local approximation that "maintains only a single high probability partition"; I saw no revision of earlier assignments during filtering. A large prediction error makes a new mode more likely, by the same mechanism for jumps and for anything else; outliers are not addressed (sensory noise is Gaussian). Behaviour: 32 students, 12 blocks each (6 jump, 6 gradual); a jump seemed to induce a new memory and protect the old one; the fitted model inferred more modes in the jump condition.

**S20 Song et al. 2022 (FULL).** Reports that the original latent-cause model could not explain the gradual-extinction vs gradual-reverse difference in rats (per a personal communication quoted in the paper). The fix needed a time-dependent CRP prior, Rescorla-Wagner learning, and collapsing the posterior to its mode between sessions. Within a session the belief in the second cause grows as non-shock trials accumulate, so attribution firms up over subsequent observations; it is not a buffered, deferred update.

### Where all this leaves the structure of the idea (inference)

The same skeleton appears under different names: a provisional hypothesis with evidence accounting. BONE's K run-length hypotheses, an iHMM's new state, a Chinese-restaurant-process "new cause", Oda's "spawn", and Shinohara's "candidate" are all a quarantined explanation that is kept, promoted or released as data arrive. Exact Bayesian inference over change and outlier hypotheses (the three-hypothesis mixture in S9's oracle model) is the ideal non-committal learner. A buffer with deferred commit is a cheap approximation to it when K copies of the weights are unaffordable. That is the defensible niche; it is a cost argument, not a new-principle argument.

## 3. What gap survives

Held against everything above, the surviving gap is narrow:

1. **A deferred weight commit in a nonlinear, SGD-trained online learner.** Every deferral method I found works on a mean (S9), a linear regression (S10, S11, S12), or detection only (S11, S14, S13). BR-iHMM defers parameter updates within a batch but with linear-Gaussian emissions. WoLF has a neural-network variant with an extended Kalman filter but no deferral. I found no paper that holds back optimiser steps for flagged samples and commits them to a slow network after later corroboration.
2. **Selective deferral.** BR-iHMM defers every batch; SPR and CNLL buffer everything. Deferring only samples flagged as surprising, and measuring the cost of non-flagged samples, is not in any paper I opened.
3. **A head-to-head on a mixed-cause stream against the right rivals.** No paper compares a deferral learner against (a) a robust-loss learner, (b) a monotone surprise-gated learning rate, and (c) a robust-loss plus changepoint filter, on one stream with a controlled ratio of change to outlier events and with a deep model.
4. **Label-function drift versus label noise in classification.** SPR, CNLL and PuriDivER (all opened) use class-pair or class-overlap tasks; the label function never changes. The ambiguity the idea relies on (a surprising label is noise or a changed concept) does not arise in those benchmarks. A benchmark where P(y|x) really changes while label noise is injected is, in what I read, absent.
5. **The identifiability limit.** S14 proves persistent corruption cannot be told from a real change on one channel. No deep-learning stream paper I opened addresses collective or persistent poisoning against a corroboration rule. This is a risk to the claim, and also a gap worth measuring.

Claims that are now unsafe to make: "novel principle"; "first to defer attribution"; "robust losses ignore the problem" (S12 proves the observation-space robust loss alone is insufficient in the iHMM setting, and S3 combines it with changepoint inference).

Claims that are still open and testable: that deferral helps a deep online learner on a mixed stream; that it beats WoLF plus run-length; that the cost model (buffer of B inputs vs K weight copies) favours it.

## 4. Where prior evidence says the idea will lose (use as falsifiers)

- Frequent changes: in S12 the tuned batch size on the fast-regime order-flow data was B = 1; in S9 some baselines beat BIB at event rate 0.05. If changes arrive faster than the corroboration window, deferral loses.
- Detection delay: S12 reports delay 3.4 steps vs 0 for the non-batched robust model; S9 reports lower learning rate at the event time than the oracle.
- Persistent or collective corruption longer than the window passes the gate (S11 shows accuracy falls when the assumed anomaly length is too short; S14 proves the identifiability limit).
- Rare real events dropped as noise: SPR's own table shows its buffer filter is worse than random sampling on abstract noisy classes.
- Repeated checking of the corroboration test inflates false commits (S13: repeated confidence-interval peeking committed 64% of borderline cases to a coin flip). A corroboration rule needs a sequential-test treatment, not a per-step threshold.
- Baseline strength: the notes in neuromodulation.md already cite Wen et al. (tuned AdamW closes most claimed optimiser gains). Any deep-learning comparison must give the constant-LR Adam baseline and every method the same tuning budget.

## 5. Source check: Nassar, Bruckner, Frank 2019 (S1)

Verified through the eLife full text via WebFetch summaries (second-hand; two separate fetches agreed on every point below, and the tool did not quote the exact instruction sentences, so re-read the Materials and methods paragraph in the PDF before citing).

### Claims in docs/literature/neuromodulation.md

| Claim in the notes | Result |
|---|---|
| n = 39 | Confirmed. 39 participants (22 female, mean age 20.2), all included; the same participants did both conditions, order randomised across participants. |
| Surprise raises learning in changepoint blocks and lowers it in oddball blocks | Confirmed in direction. The abstract (as paraphrased) says larger P300 signals predicted more learning in a changing context and less in an oddball context; surprise was tracked by the P300 in both contexts. |
| beta = 0.71, t = 9.9 | Confirmed numerically (mean/SEM 0.71/0.07, t(38) = 9.9, p = 4e-12). **Needs a definition.** It is the across-subject mean of a per-subject coefficient on prediction error x surprise x condition (condition coded +1 changepoint, -1 oddball), fitted by MAP with a von Mises error model, and tested with a one-sample t-test over the 39 coefficients. It is a difference (interaction) coefficient, not a direct estimate of the oddball effect. |
| Surprise main effect beta = 0.03, p = 0.43 | Not re-verified here. The second fetch did say the P300 had no significant condition-independent relation to learning. |
| P300 conditional-learning effect 0.09, t = 4.7 | Not re-verified here. |

### Did participants know the block type, or infer it?

**They were told.** Reported by both fetches: each condition "was preceded by (1) instructions that included an explicit description of the generative process for that condition", followed by training trials in which the cannon was visible. In test blocks the cannon was hidden. There were four blocks of 60 trials per condition (240 per condition). Changepoint condition: the cannon stays put and re-aims with hazard 0.14. Oddball condition: the cannon drifts and 14% of trials are oddballs at random locations. No awareness check is reported.

**Correction to the notes.** The section "Interpretation of surprise by inferred cause, with sign flip" should read "interpretation of surprise conditioned on an instructed context". Nassar 2019 shows that humans, when told which generative process is active, change the sign of their response to surprise. It does not show that humans infer the cause of an ambiguous surprise in a mixed stream, because the design removes that ambiguity: in a changepoint block every large error is a change; in an oddball block every large error is an oddball. The ambiguity the corroboration idea starts from is absent from this experiment. Liakoni (S18) makes the same observation from theory: the sign depends on the assumed generative model, and the two cases are separate models.

Other caveats in the paper (fetch summary): the study is correlational; the P300 is an EEG signal, not a measured neuromodulator; the authors say the mechanistic latent-state interpretation needs future work.

### Does any evidence show humans defer learning until corroboration?

**No, in what I opened.** The behavioural update in S1 is defined as the shield move from one trial to the next (the "subsequent trial" only). The paper excludes updates immediately after oddballs, because a return to the pre-oddball position could look like learning. Those are exactly the trials where corroboration-dependent behaviour would show. There is no analysis of lags beyond trial t+1. The authors of S9 state independently that it "remains unknown whether a common initial response that defers causal attribution... also occurs in human or biological inferences".

So **deferral is our own extrapolation** from a normative argument plus simulations (S9, S12), not an observed behaviour.

### Gershman's latent-cause work: evidence for delayed or gradual attribution

- Gradual attribution through an accumulating posterior: yes, in model and in rodent behaviour (S20). The cause assigned to a run of trials depends on how abruptly the contingency changed; an abrupt change creates a new cause that protects the old one, a gradual change keeps one cause (S19, S20). Within a session, belief in the new cause grows as observations accumulate (S20).
- Buffered, deferred updates: no. The update goes to whichever cause is currently inferred, and the fitted model needed a commit step (posterior collapsed to the mode between sessions, S20), the opposite of deferral.
- Gershman 2010 (S21): not read; snippet only.
- Evidence strength: S19 has 32 participants. S20 reports 8 to 16 rats per group in the original experiments. The original model could not fit the key contrast without extra assumptions.

## 6. The rival baseline

**Name: robust-loss Kalman or Bayesian update combined with run-length changepoint inference, as implemented in BONE: WoLF-IMQ plus BOCD (BONE's "WoLF+RL[inf]-PR*"; its greedy single-hypothesis variant RL[1]-OUPR* is the cheap version).**
- References: S4 (WoLF, ICML 2024) and S3 (BONE, TMLR 2025).
- Code: JAX library at https://github.com/gerdm/BONE (from S3 and a search result; not opened). WoLF code at https://github.com/gerdm/weighted-likelihood-filter (from S4; not opened).
- Why this one: it is the only released method I found that combines an outlier-robust update (bounded influence) with changepoint inference in one stream, and it can drive a neural network (extended Kalman filter, MAP point estimates in the BONE experiments). It already beats non-robust BOCD on a mixed outlier-plus-changepoint stream (BONE section 4.3.2, figure-level).
- Honest limits of this choice: it is "strongest" by design coverage and availability of code, not by a verified leaderboard, because no benchmark compares these methods head to head. Its adaptation lag under robustness is acknowledged (S2, S12).

**Also to beat or at least re-implement: BR-iHMM (S12).** It is the nearest existing deferral method and it already beats the observation-robust model (WoLF-iHMM) by about 2x on RMSE on a synthetic outlier-plus-regime stream. I did not find released code in what I read, and it uses linear-Gaussian emissions and particle learning.

**For a deep classifier with label noise only:** PuriDivER (https://github.com/clovaai/puridiver) and CNLL (https://github.com/nazmul-karim170/CNLL). They handle noise but not label-function change, so they are baselines for the noise-only arm of a mixed stream, not for the whole stream.

**Reference ceiling:** the exact three-hypothesis Bayesian mixture (change, nominal, outlier) given the true event rates, as the oracle model in S9. If the deferral learner does not approach it, or does not beat it on cost, the claim reduces to a cost argument.

## 7. Search limits

- Searches covered alphaXiv (arXiv-indexed) and a few web searches. Peer-reviewed work not on arXiv, signal-processing and statistics venues, and the quality-control and tracking literature on outlier-robust filtering (for example EMORF/S, cited in S14 and S4) were not opened.
- Unopened leads that could contain closer prior art: Fearnhead and Rigaill 2019; Sellier and Dellaportas 2023; Reimann 2024; Wilson, Nassar and Gold 2010 (joint hazard model); EATA and SAR sample-selection test-time adaptation; the COIN model of contextual inference.
- Several of the closest items (S9, S10, S11, S13, S14) are 2025-2026 preprints, some single-author. I could not assess their quality beyond what the text shows, and S14 was designed and benchmarked by its own author.
- The author-side check on the notes file (docs/literature/neuromodulation.md) covered only the claims listed in section 5. I did not edit that file.
