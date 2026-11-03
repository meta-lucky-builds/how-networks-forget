# Literature thread: gating, sparsity, capacity limits, metacognition

Date of search: 2026-11-04. Thread scope: PFC/basal-ganglia and thalamocortical gating, working-memory (WM) capacity limits, active dendrites and context gating, sparse expansion coding (fly, dentate gyrus), human metacognition (meta-d'), and the AI counterparts, 2024-2026 emphasis.

## 0. How to read this file (provenance conventions)

Every source carries one of these read-level tags. If a source has no tag it was not used.

- [EXCERPTS] I read the relevant pages of the full PDF through alphaXiv `answer_pdf_queries`. This returns the most relevant pages, not necessarily the whole paper, so unread sections may contain caveats I did not see.
- [WEB-PARTIAL] I fetched the page with WebFetch. Its answer is produced by a small model summarising the page, and for the long pages here it covered only the first ~100k characters of 170k-190k. Quotes from these are as the tool returned them.
- [SNIPPET] I saw only a web-search result summary or a discovery-tool abstract. I did not open the paper. Claims tagged this way are leads, not evidence.
- [CITED-ONLY] I saw it only as a citation inside another paper I opened. Not verified.
- [MEMORY] Not verified in this session. Used sparingly and always labelled.

Not opened despite attempts (binary/paywalled/redirect): O'Reilly and Frank 2006 PBWM primary (PDF returned unreadable binary), Dasgupta et al. 2017 Science full text (PDF returned unreadable binary), Rikhye et al. 2018 (Nature auth redirect), Rahnev 2025 (cert error and Nature auth redirect). Everything I say about these four is secondary.

Source-quality warning: most 2026 AI items below are unreviewed preprints, several by one or two authors. alphaXiv vote and view counts are not evidence.

## 1. Summary

1. Biology. The most-cited mechanism in this thread, striatal "go/no-go" gating of WM updating (PBWM), is contested. The only direct fMRI test I opened (7T, n=35) found no evidence of subcortical activation at gate opening in a low-demand task, though the authors say it does not rule out PBWM for high-demand tasks. EEG evidence supports an analogous-computation account but cannot see corticostriatal circuits. Thalamic (mediodorsal) support of PFC maintenance is probable in mice but I only saw summaries. The fly "random" PN-to-KC wiring that many AI papers copy has been revised: the connectome shows non-random structure (z = 13.8 for core-community claws), and the observed network discriminated worse than a random one in the authors' model.
2. AI, sparsity and continual learning. Sparse expansion plus an associative readout looks strong against weak 2021-era baselines (FlyModel: 0.86 vs 0.58 EWC, 5 tasks MNIST-20), but a tiny-paper sparse-activation MLP with a tuned optimiser reaches 78.3% on Split-MNIST class-IL vs 77% quoted for FlyModel, and Fly-CL (ICLR 2026) beats RanPAC by only about 1 point or less in accuracy on most cells (it loses by 0.32 on CIFAR-100 ViT) while being 10x+ faster. The robust benefit I can document is compute, not accuracy. FlyPrompt's "fly-inspired" router is a dense Gaussian projection with ReLU, not sparse binary wiring with winner-take-all.
3. AI, capacity/bottleneck. A limited-capacity shared workspace helps in some settings (bouncing-balls ARI 0.915 at 5 slots vs 0.351 at 10), is within noise on others (CATER top-1 71.07 vs 70.83), and its optimum is tuned, not emergent. In LLMs, the WM-capacity limit is a readout-interference limit, not a failure to locate the item (Gong 2026), reversing the same first author's 2024 attention-dispersion account.
4. AI, metacognition. Calibration is not metacognition. Independent evidence now says (a) LLM confidence is largely a shared difficulty axis with a model-specific threshold (Moran and Whiting 2026, 20 models, binary verbalised confidence), (b) trained confidence tracks output consistency off-distribution and true accuracy only near the training data (Yax et al. 2026), yet (c) confidence is causally used to abstain (Kumaran et al., Nature Machine Intelligence 2026: abstention 66.5% to 7.0% under steering). These are compatible: confidence is used, but mostly encodes shared difficulty or consistency, not private self-knowledge.
5. Dropped property (Section 4). The thing AI consistently drops is a gate or context signal that the system generates itself, online, without block structure. AI versions take the context from a task ID, a nearest-prototype lookup, or a session label. The measurable cost of replacing the oracle with inference in the one paper that reports both is 4.5 points at 100 tasks (81.4 to 76.9) and 0.3 at 10 tasks (94.6 to 94.3). This is my cross-paper inference; no single paper tests it, and Experiment 1 is designed to.

## 2. Evidence table

Grades: E = established (multiple converging sources, direct test), P = probable, C = contested (credible sources disagree or key test is null), H = hypothesis (model-level claim, not tested), U = unverified by me.

| # | Claim | Grade | Source (URL) | What I read |
|---|---|---|---|---|
| B1 | Basal ganglia/thalamus "go/no-go" gating controls WM updating (PBWM; O'Reilly and Frank 2006) | C | Trutti et al., eLife reviewed preprint 97874, https://elifesciences.org/reviewed-preprints/97874 | [WEB-PARTIAL] 7T fMRI, reference-back task, 37 scanned, 35 analysed. Quote: authors "found no evidence of subcortical activation during working memory gate opening"; "moderate evidence" against most basal ganglia and midbrain ROIs; thalamus inconclusive. They say the result "challenges the concept of a striatal working memory gating mechanism in such low-demand working memory tasks" but "do not rule out" PBWM predictions for high-demand tasks, and suggest the striatum may instead support substitution and a ready-to-update mode. |
| B2 | Input gating, output gating and motor gating share a computation (EEG signature, higher decision threshold at gate switches) | P | Rac-Lubashevsky and Frank 2021, PLoS Comput Biol 17(6):e1008971, https://pmc.ncbi.nlm.nih.gov/articles/PMC8211210 | [WEB-PARTIAL] Authors write EEG "does not afford the ability to assess the involvement of corticostriatal circuits specifically"; closing-gate via prefrontal-subthalamic pathway is "speculation"; standard DDM may not fit. Single-lab. |
| B3 | PBWM primary claims (learned gating via dopamine/PVLV, maintenance vs updating separation, advantage over LSTM-like nets) | U | O'Reilly and Frank 2006, Neural Computation | Not opened (PDF unreadable). The search tool's summary called the model "cited as lacking conclusive empirical support"; treat as [SNIPPET]. Any comparison numbers vs LSTM are [MEMORY] and omitted. |
| B4 | Mediodorsal thalamus sustains PFC activity during WM maintenance and rule representation (mouse) | P | Bolkan et al. 2017 (Nat Neurosci, https://www.nature.com/articles/nn.4568); Schmitt et al. 2017 (Nature); Rikhye, Gilra, Halassa 2018 (https://www.nature.com/articles/s41593-018-0269-z) | [SNIPPET] only. Summaries: MD boost during delay improved WM performance; MD amplifies local PFC connectivity; for Rikhye I saw title and a reference-list fragment ("MD responses reflected the statistical regularity of cue presentation and were required for switching between ... cueing contexts"), not verified. |
| B5 | Human WM holds roughly 3-4 items | E (secondary) | Stated as model target in Bouchacourt and Buschman 2019, https://pmc.ncbi.nlm.nih.gov/articles/PMC6613943 | [WEB-PARTIAL] "Stated capacity is roughly 3-4 items"; model correlates r = 0.97 with Luck and Vogel 1997 behaviour using values extracted from figures. Primary behavioural papers not opened. |
| B6 | The capacity limit is the unavoidable price of flexibility (random-connectivity interference) | H | same PMC6613943 | [WEB-PARTIAL] Two-layer Poisson spiking model, sensory layer 8 rings x 512, random layer 1,024 neurons, connection probability 0.35. Authors list limitations: simplified neurons, no short-term plasticity or oscillations, parameters tuned to maximise model performance, comparisons are correlations on figure-extracted values, learning did not generalise beyond trained inputs. Not compared directly with monkey recordings. |
| B7 | WM limit is interference/executive-attention, not storage | P (competing account) | Cited in Gong and Zhang 2024 and Gong 2026 (Engle 2002; Oberauer et al.) | [CITED-ONLY]. |
| B8 | A capacity limit or sparse code improves generalisation in biological systems | U | none found | I found no biological study that tests this causally. The AI papers that assert it (e.g. 2603.03131, Section 3) motivate it by analogy only. |
| B9 | Fly Kenyon cells fire sparsely (~5% of ~2,000) via APL feedback inhibition; PN-to-KC expansion ~40x, ~6 PNs per KC | E (secondary) | Described in Shen et al. 2021, https://arxiv.org/abs/2107.07617 and Zou et al. 2025, https://arxiv.org/abs/2502.01427 | [EXCERPTS] of both. Primary refs (Lin et al. 2014 Nat Neurosci; Caron 2013 Nature; Papadopoulou 2011 Science) are [CITED-ONLY]. |
| B10 | PN-to-KC wiring is random | C (revised) | Zheng et al. 2022, Current Biology 32(15):3334, https://pmc.ncbi.nlm.nih.gov/articles/PMC9413950 | [WEB-PARTIAL] Abstract: connectome "revealed unexpected structure"; main text: input "is not sampled at random by KCs". Core-community claws 1,916 observed vs 1,421.7 +/- 35.7 under random-bouton null (z = 13.8). Concentrated in alpha-beta and alpha'-beta' KCs. But the structure "largely disappears under a local-random null that accounts for arbor geometry", so axon-dendrite overlap explains much. Model: observed network "degraded discrimination performance compared to a random network" except when signal flows only through overconvergent (mostly food-responsive) PN types. One female FAFB brain, ~62% of right-side claws reconstructed. Journal is Current Biology (a search result I first saw said Neuron). |
| B11 | Fly olfactory circuit implements a locality-sensitive-hashing variant with better search efficiency than dense-projection LSH | U | Dasgupta, Stevens, Navlakha 2017, Science 358:793, DOI 10.1126/science.aam9868 | Full text not readable. [SNIPPET] summaries only. A third-party reimplementation (https://github.com/tian-kun/Fly-LSH, [WEB-PARTIAL]) states its results match Fig. 1 trends but "doesn't match results from Fig.3 in Dasgupta et al., 2017"; it uses Hamming distance where the paper used Euclidean, and suspects WTA was implemented differently. It reports Fly-LSH mAPs "roughly 3x bigger than that of conventional LSH" on MNIST. Unreviewed GitHub note, one author. |
| B12 | Dentate-gyrus sparse granule-cell firing supports pattern separation | P (theory); mechanism C | Madar, Ewell, Jones 2019, PLoS Comput Biol, https://pmc.ncbi.nlm.nih.gov/articles/PMC6476466; human fMRI claim Berron 2016 | Madar [WEB-PARTIAL]: granule cells "can exhibit both pattern separation and its opposite computation, pattern convergence" depending on code and input statistics; role "has been hard to test experimentally"; mouse slices with artificial Poisson inputs. Berron [SNIPPET] only. |
| B13 | Dendritic nonlinearities implement context-specific routing in vivo | U | Cited by Iyer et al. 2022 (Yang 2014; Poirazi 2003; Beniaguev 2021) | [CITED-ONLY]. Iyer's text: Beniaguev et al. suggest a pyramidal neuron is "equivalent to a larger ANN with 7 hidden layers". I opened no dendrite neuroscience primary. |
| B14 | Metacognitive sensitivity is dissociable from task performance and domain-specific in humans | P | Fleming et al. 2010 Science; Rouault et al. 2018; Fleming and Lau 2014 | [CITED-ONLY] via Cacioli 2026 Section 5.3, https://arxiv.org/abs/2603.25112. |
| B15 | meta-d' / M-ratio cleanly separates metacognitive efficiency from Type-1 performance | C | Rahnev 2025, Nature Communications 16:701, DOI 10.1038/s41467-025-56117-0 | [SNIPPET] only (search summary of abstract): 17 measures compared; "measures of metacognition depend only weakly on response and metacognitive bias, but show many strong dependencies on task performance"; "no measure of metacognition is perfect". A conference abstract in the same search said M-ratio "outperformed the other 16 measures" with a "slight tendency to overcorrect". Not verified. Separately Cacioli 2026 [EXCERPTS] argues meta-d' is undefined for open-ended QA because there is no two-alternative Type-1 decision. |
| B16 | Human confidence tracks self-consistency more than accuracy | H | Caziot and Mamassian 2021, as cited in Yax et al. 2026 | [CITED-ONLY]. Yax note the two "consistency" definitions differ. |

## 3. AI implementations and effect sizes

### 3.1 Active dendrites + k-winner-take-all (Numenta)

Iyer et al. 2022, Frontiers in Neurorobotics, https://arxiv.org/abs/2201.00042 [EXCERPTS].

- Permuted MNIST, single output head, 2 hidden layers of 2,048 active-dendrite units with kWTA, 8 seeds. Context provided during training: 94.6% (10 tasks), 81.4% (100 tasks). Context inferred during training (online clustering): 94.3% and 76.9%. Active dendrites + SI: 97.2% and 91.6%.
- Versus XdG (Masse 2018): "significantly better than XdG" at large task counts without SI, "slightly worse than XdG combined with SI". XdG numbers are taken from the Masse paper (not re-run).
- Meta-World MT10: active dendrites 87.5% vs 73.1% for a 3-layer MLP with 1,700,000 more non-zero parameters (average over last 500,000 environment steps). Parameter table: ADN 7,449,964 non-zero, MLP 7,994,004, large MLP 9,165,004. The MLP baseline receives the one-hot task ID as extra input (input size 49 vs 39).
- Caveats I can see: "hyperparameters were optimized individually for each setting"; the MT10 context vector is a 10-d one-hot task ID (oracle); the CL inferred-context method assumes "all training examples in a batch correspond to the same task" (their words, Section 6.2.3); no replay or modern pretrained-feature baselines; the ablation (dense dendrites vs sparse no-dendrites vs both, Fig. 12) is in a figure whose values I did not extract.
- Follow-up replication: a 2022-2026 search returned only application or hardware papers (DS-VLA, NeuDW-CIM) [SNIPPET]; I found no independent replication of the permuted-MNIST or MT10 numbers.

### 3.2 Fly-circuit continual learning

| Paper | Setting | Result | Baselines | Read |
|---|---|---|---|---|
| Shen, Dasgupta, Navlakha 2021, https://arxiv.org/abs/2107.07617 | Class-incremental on frozen features. MNIST-20 (MNIST+Fashion-MNIST, LeNet5 trained on KMNIST, 84-d); CIFAR-100 (ImageNet-pretrained ResNet18, 512-d). m = 40d expansion, 5 seeds | MNIST-20 after 5 tasks: FlyModel 0.86 +/- 0.0006 vs BI-R 0.77, GEM 0.69, EWC 0.58 +/- 0.10, Vanilla 0.19. End: at least 0.19 above any CL method, 0.11 below Offline. CIFAR-100: at least 0.15 above, 0.13 below Offline. Ablation at end: sparse+freezing 0.75 vs sparse+logistic regression 0.54 (MNIST-20), 0.41 vs 0.21 (CIFAR-100); dense coding about 0.07 (MNIST-20) | EWC, GEM, BI-R, vanilla, offline; "Parameters ... independently selected using grid search"; perceptron variants where decreasing weights to the wrong class "hurts performance" | [EXCERPTS] |
| Bricken et al. 2023 (quoted by Hard ASH) | Split-MNIST class-IL, 1 hidden layer 1000 | FlyModel 77%; SDMLP 69%; SDMLP+EWC 83% (500 epochs); EWC 61% | | [CITED-ONLY] via Hard ASH Table 1 |
| Keskinen 2024 "Hard ASH", https://arxiv.org/abs/2404.17651 (ICLR 2024 Tiny Paper, unaffiliated author) | 5-task Split-MNIST class-IL, MLP 1x1000, 1 epoch, 5 runs, 95% CI | Hard ASH 78.3 +/- 1.4 (Adagrad); ASH 76.4; Top-K 76.0; ReLU 49.2 (Adam). Sparse beats non-sparse in all optimisers. Adam without bias correction 76.7 vs 71.6 with. Plasticity test (permuted MNIST, 100 tasks): sparsity "helps with stability without significant penalty to plasticity"; latest-task accuracy has "negligible" dependence on sparsity | Optimiser sweep; hyperparameters "heavily tuned towards better performance in Split-MNIST"; i.i.d. MNIST with CL-tuned optimiser 91.3 vs 96.9 with normal optimiser | [EXCERPTS] |
| Zou et al. 2025 "Structural features of the fly olfactory circuit...", https://arxiv.org/abs/2502.01427 (v1 preprint) | Plug-in KC layer (30,000 KCs) on frozen ViT, CIL; CIFAR-100, CUB-200, VTAB | With SGD: +22% (CIFAR-100), +25% (CUB), +7% (VTAB). With SI: +8%, +15%, +5%. EWC in between. Odor-simulation: coding level 1% chosen as best (fly ~5%) | SGD, EWC, SI heads without KC layer; plasticity methods L2-Init and Shrink&Perturb; no analytic classifiers | [EXCERPTS] |
| Zou et al. 2026 "Fly-CL", ICLR 2026, https://arxiv.org/abs/2510.16877 | PTM-based CIL, ViT-B/16 and ResNet-50, 5 seeds, baselines re-run via PILOT code | ViT-B/16 overall accuracy (Table 1): CIFAR-100 Fly-CL 93.89 +/- 0.12 vs RanPAC 94.21 +/- 0.11 (Fly-CL lower by 0.32), EASE 92.96, SEMA 92.04; CUB 93.84 +/- 0.18 vs RanPAC 92.67 +/- 0.20 (+1.17); VTAB 96.54 +/- 0.38 vs F-OAL 94.68 +/- 0.32, RanPAC 94.16. ResNet-50: +1.89 / +1.53 / +1.20 over best baseline. ImageNet-R 83.19 vs RanPAC 83.02; ImageNet-A 67.98 vs 67.28. Post-extraction time on CIFAR-100: 5.38 s vs RanPAC 84.42 s, F-OAL 57.13 s | L2P, DualPrompt, InfLoRA, SEMA, MoE-Adapter, EASE, RanPAC, F-OAL | [EXCERPTS] |
| Yan et al. 2026 "FlyPrompt", ICLR 2026, https://arxiv.org/abs/2602.01976 | General CL (blurry boundaries, single pass), ViT-B/16 Sup-21K, 5 seeds | "up to 11.23%, 12.43%, and 7.62%" over SOTA on CIFAR-100, ImageNet-R, CUB-200. Ablation (CIFAR-100 A_auc): prompt experts only 80.75; + REAR router 81.90; + EMA heads 82.17; all 83.24; RanPAC-style analytic classifier 69.91 | L2P, DualPrompt, CODA-P, MVP, MISA | [EXCERPTS] |

Observations drawn from these (my analysis, labelled as such):

- Fly-CL arithmetic from its own Table 1: the text says VTAB accuracy improves "2.38% over the best-performing methods"; against the best baseline in that table (F-OAL 94.68) the gap is 1.86, and 2.38 is the gap to RanPAC (94.16). The abstract-level claim is "comparable to or exceeding"; the data fit "comparable".
- Fly-CL's defaults are m = 10,000, p = 300, k = 3,000. With ViT d = 768, p = 300 non-zeros per row is about 39% connectivity and k/m = 30% active units, versus about 6/50 and 5% in the fly. Its own sensitivity analysis reports "a broad plateau for optimal k". The ablation I saw (Fig. 4) removes projection, ridge or normalisation; I saw no top-k-only ablation, although the text asserts top-k "simultaneously improv[es] performance". The timing table attributes speedups to sparse projection (0.22 s to 0.08 s) and top-k similarity (0.21 s to 0.08 s) on CUB.
- Fly-CL's "biologically plausible" rule (Hebbian + anti-Hebbian + decay, Eq. 13) is shown to converge in expectation to ridge regression. That includes an error-correcting anti-Hebbian term, which Shen et al. found harmful in sequential class learning (their perceptron variants v1, v3). The two papers are reconcilable: Fly-CL accumulates Gram-matrix sufficient statistics across tasks, so its solution is the joint ridge fit, which does not forget; Shen's rule is online and bounded. I infer this; neither paper states it.
- FlyPrompt's router (Eq. 1): R_ij ~ N(0,1) dense Gaussian, sigma = ReLU, M = 10,000. That is RanPAC-style expansion, not sparse binary PN-to-KC wiring or winner-take-all. Its effect on CIFAR-100 A_auc is 80.75 to 81.90 (+1.15), from Table 3; the hyperparameter plot's y-axis ticks (82.21 to 83.41 for "w/o RP" through M = 20K) suggest roughly a 1-point range, read from axis labels only. Most of the headline gain therefore comes from task-wise prompt experts and EMA heads, not from the fly component. Note also that the router is trained with the session index t as the expert label (Eq. 2-3, Algorithm 1), even in a "general CL" setting.
- Disagreement on plasticity: Zou 2025 reports that in their circuit model memory stability falls with higher coding levels while plasticity "increases steeply in the initial range and then saturates", giving an optimal coding level. Hard ASH reports that sparsity has "negligible effect" on latest-task accuracy on permuted MNIST (100 tasks) until z_k >= 3. Different tasks, architectures and optimisers; unresolved.

### 3.3 Capacity limits, bottlenecks, workspace

- Goyal et al. 2022 (ICLR), https://arxiv.org/abs/2103.01197 [EXCERPTS]. Shared workspace with top-k write competition. CATER top-1: TR 70.83 +/- 0.44, TR+HSW 71.07 +/- 0.04, TR+SSW 71.33 +/- 0.34 (differences smaller than the baseline's own spread). Bouncing balls with SCOFF: ARI 0.276 baseline; +SW with 2 slots 0.154, 4 slots 0.487, 5 slots 0.915, 8 slots 0.891, 10 slots 0.351. The authors read the decline with more slots as validating the bandwidth-limit claim. Atari: median performance ratio 1.13 and 1.11 over RIMs across 36 game pairs. Their abstract says capacity limits "have a rational basis". What I take from it: the benefit appears only at a tuned capacity, and I cannot see from the extracted pages whether slot count was chosen on held-out data.
- Gutheil, Hitzginger, Legenstein 2026, https://arxiv.org/abs/2605.22472 [EXCERPTS]. Hard multi-WTA bottleneck (straight-through Gumbel-softmax) trained on multiple tasks yields symbolic representations; plain softmax gave "blurry output distributions ... insufficient". Synthetic latents, 5 seeds: downstream AUC for new task with 100 training samples 0.993 +/- 0.005 (WTA code) vs worse for entangled input even at 4,400 samples (0.973 +/- 0.01); unseen category pair 0.987 vs 0.917 +/- 0.059 at 4,400. Only 3 of 5 (matched) and 4 of 5 (unmatched) seeds solved the tasks perfectly; symbolic code emerged only in those; needs at least about 20 tasks. dSprites variant: 3 of 5 runs perfect, needed L1 on readout. This is the cleanest positive case in the thread for "a WTA bottleneck improves generalisation", under strict conditions.
- Wang 2026 "Joint Training Across Multiple Activation Sparsity Regimes", https://arxiv.org/abs/2603.03131 [EXCERPTS]. CIFAR-10, no augmentation, WRN-28-4, single seed (3407), 500 epochs: dense best test accuracy 0.869 vs 0.8797 and 0.8802 with two top-k controllers. Author's own limits: hyperparameters not tuned, single runs. I treat this as no evidence either way (+1.1 points, n = 1).
- Gong and Zhang 2024, https://arxiv.org/abs/2409.10715 (NeurIPS 2024 BML workshop) [EXCERPTS]. 1-layer 1-head transformers trained on N-back; accuracy falls with N and total attention entropy rises with N; concluded dispersion of attention scores "might be the cause". Models with more layers or heads reached over 95% on all N.
- Gong 2026, https://arxiv.org/abs/2610.21176 (preprint) [EXCERPTS]. 15 pretrained LLMs (Pythia 70M-6.9B, GPT-2, Qwen2.5 0.5B-14B). Median d' falls from 2.57 (N = 1) to 0.91 (N = 4); at N = 2 models say "match" to a median 79% of lures vs 4% of other non-matches. Task heads attend elsewhere on only 16%, 7%, 8% of misses at N = 2, 3, 4. Linear probes separate match vs lure at median AUC 1.00 (N = 2) while the model's own preference gives 0.63. Interventions: sharpening top-3 heads median change in d' +0.03/+0.05; suppressing non-target letter identity median +0.46 (full) and +0.60 (half), improving d' in 42 of 43 model-by-load cases; random-subspace control about 0. In minimal one-layer transformers precision does limit, but adding a second layer gives d' 6.4-7.2 at every N. LSTM with 512-d state d' >= 7.9 at all N; 32-d LSTM falls from 8.2 to 1.8.
  - Source disagreement: Gong 2024 (dispersion) vs Gong 2026 (interference/readout) from the same first author. The 2026 paper states the precision account holds only "for transformers too shallow to compare two items directly". Treat the 2024 mechanism as superseded for pretrained LLMs.
  - Notable for gating: the beneficial intervention (suppress non-target content) is an oracle input gate: it uses knowledge of which lines are non-target.

### 3.4 Metacognition and calibration in LLMs

- Kadavath et al. 2022, https://arxiv.org/abs/2207.05221 [EXCERPTS]. Large models are well calibrated on multiple choice with lettered options; "none of the above" "reduces accuracy and calibration significantly"; RLHF policies "naively seem miscalibrated" but temperature 2.5 largely fixes it; P(IK) trained on TriviaQA discriminates on other tasks but "OOD calibration of P(IK) is often quite poor". Cross-model test: starting from model A, AUROC 0.8633 on A's ground truth and 0.8460 on B's; starting from B, 0.8631 on A's and 0.8717 on B's ("some signal ... no difference ... on data from model A").
- Cacioli 2026 (independent researcher; v3), https://arxiv.org/abs/2603.25112 [EXCERPTS]. 4 open models (7-9B, Q5_K_M), 224,000 trials, TriviaQA and Natural Questions. Normalised metacognitive information meta-I2r 0.151-0.300 (factor 1.98), not predicted by accuracy; z-ROC slopes 0.78-1.18; weakest domain Science and Technology for all four. Retraction inside the paper: v1/v2 reported an inverse accuracy-efficiency coupling (rho = -1.00); a length-biased automated scorer "missed 53% of one model's human-correct answers"; after correction rho is -0.80 on TriviaQA and +0.00 on NQ. Residual scorer error remains (18.4% of rejected trials in a 630-item audit were actually correct, model-dependent by 2.4x). Lesson for any replication: audit the correctness scorer by model.
- Moran and Whiting 2026 (Pareto AI, preprint), https://arxiv.org/abs/2605.24299 [EXCERPTS]. 20 frontier LLMs, six benchmarks, one binary yes/no confidence judgement per item at temperature 0, about USD 4,000 of API spend. Findings: confidence matrix approximately rank-one on factual benchmarks; mean pairwise Kendall tau 0.037 (SQuAD) and 0.025 (OmniMath) above null, vanishing once performance base rates are matched; MathBench exception traced to reasoning models solving the problem inline while answering the confidence probe. ROC AUC for predicting correctness (Table 3): Internal 0.474 (LegalBench, Gemini Flash), 0.630 (MathBench, GPT-4o), 0.632 (MMLU), 0.590 (SciCode), 0.664 (OmniMath), 0.622/0.731 (SQuAD with Haiku/GPT-4o); trivial logistic regression on the reasoning text (QM) 0.788, 0.703, 0.706, 0.539, 0.619, 0.657/0.683. Limitations they state: one temperature-0 judgement, heuristic reasoning/non-reasoning labels, latent-normality assumption. Their RLVR explanation ("hedging guarantees zero reward") is a hypothesis, not tested.
- Kumaran, Daw, et al., Nature Machine Intelligence 2026, https://arxiv.org/abs/2603.22161 [EXCERPTS]. Four-phase abstention paradigm. GPT-4o 63.7% on Phase 1, temperature-scaled calibrated confidence ECE 0.046, AUROC 0.90; abstention indifference point about 77%; calibrated confidence standardised effect |beta| = 0.99, "approximately an order of magnitude larger" than RAG score, difficulty or embedding features. Activation steering in Gemma 3 27B: abstention fell from 66.5% to 7.0% (59.5 point swing, layers 30-40), accuracy among answered fell from 59.2% to 53.7%, coverage rose from 33.5% to 93%; mediation: 67.1% via confidence redistribution, 26.2% via policy shift, residual direct effect remains. Verbal confidence predicted abstention independently although "objectively less discriminatory". Scope: factual multiple-choice, no chain of thought, SimpleQA-style set, n = 1,000 trials per model. My caution: steering vectors were built from contrasts of high vs low calibrated-confidence margin, and the mediator is also the model's calibrated probabilities, so "confidence" is partly defined by the same quantity that is manipulated; the authors state the assumption of no unmeasured confounding.
- Yax, Palminteri, Oudeyer 2026, https://arxiv.org/abs/2609.33886 [EXCERPTS]. 10 open models fine-tuned (LoRA + linear probe) to predict own accuracy before answering. Verbalised confidence before training r = 0.04 +/- 0.01 with true accuracy; after, r = 0.73 +/- 0.03. But on MMLU-Pro and MedMCQA test splits (category-disjoint) trained confidence tracks output consistency better than accuracy (delta r = -0.12 +/- 0.01), while on the MATH variants (test categories close to training) it tracks accuracy (delta r = +0.27 +/- 0.06); accuracy-tracking declines with cosine distance from training data (r = 0.49 for the end-of-stack probe; 0.07 for the mid-stack probe). Output-consistency tracking appears early in training and transfers across datasets. Models ranged to 9B.
- Park et al. 2026 "ESMA", https://arxiv.org/abs/2602.02605 [EXCERPTS]. Evolution strategy with joint reward R(C, A). Qwen2.5-1.5B d'_type2 0.20 to 0.93 with direct accuracy 42.86% to 41.86%; comparison on same model: SFT 0.40, PPO 0.31, GRPO 0.70, ESMA 0.93; Type-2 AUROC 0.672 to 0.751 (1.5B), 0.601 to 0.755 (3B), 0.676 to 0.747 (7B). External sets without further training: FreebaseQA d' 0.30 to 1.12 (1.5B), NQ Open 0.19 to 0.79, WebQuestions 0.39 to 0.66. Reward-hacking ablation: meta-alignment-only reward collapses to "I don't know". Claims proprietary models: GPT-5.2 0.94, Claude Sonnet 4.5 0.95, Gemini 3 Flash 0.68. Caveat for the thread: given Yax et al., the ESMA gains may be output-consistency tracking; nobody has decomposed them.
- Not opened: Metacognition-in-LLMs survey (https://www.alphaxiv.org/abs/2607.11881), Cognitive Pairwise Training (https://www.alphaxiv.org/abs/2606.00869), Decomposing and Steering Functional Metacognition (https://www.alphaxiv.org/abs/2605.08942) [SNIPPET only].

### 3.5 Claims that failed or weakened (collected)

1. Cacioli v1/v2: inverse accuracy-metacognition coupling retracted after scorer-bias correction.
2. Gong and Zhang 2024: attention-dispersion explanation of LLM WM limits is not supported for pretrained LLMs by Gong 2026.
3. Moran and Whiting: the apparent individuated metacognition in math reduces to inline problem solving.
4. Dasgupta 2017 third-party reproduction: Fig. 3 not matched (unreviewed note).
5. Fly-CL text vs table: 2.38% claim is relative to RanPAC, not to the best baseline in its own Table 1.
6. FlyPrompt: "fly-inspired" component contributes about 1 point of A_auc on CIFAR-100 in its ablation; headline gains come from other modules.
7. Shen 2021 baselines (EWC, GEM, BI-R) are weak by 2025 standards; Hard ASH reaches 78.3% vs 77% quoted for FlyModel with no continual-learning mechanism beyond sparsity and a tuned optimiser.
8. Joint-sparsity-regime generalisation claim: n = 1 seed, +1.1 points.
9. Zheng 2022 vs the "random PN-to-KC" premise of every fly-inspired AI paper above.

## 4. Dropped load-bearing properties

### 4.1 Primary: a self-generated, online gate/context signal

Statement. In the biological accounts, the gating or context signal is produced by the system itself (a learned go/no-go policy in PBWM; thalamic regulation that tracks statistical regularities in cueing contexts) and must work when context changes are not announced. The AI implementations in this thread supply the context from outside or infer it under a block assumption.

Evidence by source:

- Iyer et al. 2022: MT10 context is a "10-dimensional one-hot encoded vector that describes the task ID"; XdG requires "a task ID"; the continual-learning "inferred" context relies on batches that each contain a single task ("one inductive bias in our procedure is that all training examples in a batch correspond to the same task") and nearest-prototype lookup at test time. Cost of replacing the oracle by this inference: 94.6 to 94.3 (10 tasks), 81.4 to 76.9 (100 tasks). [EXCERPTS]
- FlyPrompt: even in the "general CL" benchmark, the analytic router is fitted to session-indexed one-hot expert labels (Eq. 2-3). The authors' own diagnostic (Fig. 2b) shows routers in prior methods are inaccurate, and with an oracle router earlier methods "still exhibit inferior performance", so routing and expert competence are separate bottlenecks. [EXCERPTS]
- Fly-CL, Zou 2025, FlyModel: no gating variable at all; they work by representation separation and class-local writes, with task boundaries used for evaluation.
- Gong 2026: the only intervention that helps LLM working memory substantially (suppress content of non-target items) is an oracle input gate; a model that could produce that suppression itself would be the missing piece. [EXCERPTS]

Biological side, with hedges: PBWM says gating is learned by dopaminergic reinforcement over when to update (primary unread; [SNIPPET]); the direct imaging test of striatal gate opening was null in a low-demand task (B1), so the claim that the striatum is the gate is itself contested. MD-thalamic switching of context is supported in mice at summary level only (B4). Therefore: the AI-side dependence on the source of context is documented; the biological load-bearing status of a specific implementation (basal ganglia) is not.

Confidence: moderate for "AI work supplies or block-assumes the gate signal" (several papers, quoted); low-to-moderate for "this is why AI results differ from biology". Experiment 1 tests the first link causally.

### 4.2 Secondary: class-local associative (non-competitive) update

Shen et al. found the fly's rule (strengthen only synapses from active KCs to the target MBON; no decrease to other outputs) beats the same sparse code with a supervised readout, 0.75 vs 0.54 (MNIST-20) and 0.41 vs 0.21 (CIFAR-100), and that decreasing weights to the wrong class causes "hijacking" of shared neurons (Lemma 2 plus Fig. S3; [EXCERPTS]). Later work replaces it with SGD/EWC/SI (Zou 2025), ridge (Fly-CL, RanPAC), or backprop (Iyer, Hard ASH). Whether this is load-bearing is disputed by the data: Fly-CL matches or beats methods that do not use the rule, and Hard ASH reaches a similar Split-MNIST number with an optimiser alone. Graded: dropped, but not shown to matter on frozen pretrained features.

### 4.3 Secondary: confidence shaped by error costs, and measured separately from ability

Humans' metacognitive sensitivity is assessed separately from Type-1 performance (B14). AI evaluation mostly uses ECE or Brier, which conflate bias and sensitivity (Cacioli), and RLVR-style rewards give no credit for abstaining (Moran and Whiting's hypothesis, untested). ESMA and Kumaran show that putting a price on confident errors or abstention creates and uses confidence signals. This property is only partly dropped (Kumaran shows confidence is causally used) and the biological claim is itself under measurement dispute (B15), so I rate it speculative.

### 4.4 Not dropped (checked)

- Sparse high-dimensional expansion with winner-take-all: retained in FlyModel, Zou 2025, Fly-CL (with defaults far from the fly's 5% coding level); replaced by dense Gaussian + ReLU in FlyPrompt.
- Confidence is used for control: present in LLMs (Kumaran).

## 5. Candidate experiments (free Colab, T4-class GPU, under 1 hour per condition)

General rules for all three: 5 seeds per condition; paired comparisons with a 95% bootstrap CI; fix hyperparameter budgets equally across arms (the same number of tuning runs per arm, tuned on a held-out validation split, not the test split); report the preregistered threshold before running.

### Experiment 1: Does the active-dendrite advantage survive when the gate signal must be self-generated and tasks are not blocked?

Question. Tests Section 4.1 directly.

Setup. Permuted MNIST, T = 10 tasks, single head, 2 hidden layers x 2,048 units (the Iyer architecture), 3 epochs per task. About 30 passes over MNIST for a small MLP per run, which is minutes on a T4. Two stream regimes: (a) sequential blocks (Iyer's regime), (b) blurry stream where each batch is a mixture that drifts from task t to t+1 over the training of task t (for example 80/20 then 50/50 then 20/80), so the batch-equals-task assumption is false.

Arms.
- A: active dendrites + kWTA with oracle task-ID context.
- B: active dendrites + kWTA with Iyer's nearest-prototype context (prototype built per task in regime (a); built by streaming clustering with a novelty threshold, per sample, in regime (b)).
- C: kWTA MLP without dendrites (same sparsity).
- D: dense MLP + SI.
- E: Hard ASH MLP + Adagrad (no context signal at all).

Baselines to reproduce first (sanity gate): at 10 tasks in regime (a), A about 94.6 and B about 94.3 per Iyer; if A and B are not within 2 points of those, stop and debug before reading the main result.

Metric. Final mean accuracy over all 10 tasks, single head, no task ID at test.

Falsifier. Define margin_X = acc(X) - acc(C). In regime (b): if margin_B / margin_A >= 0.9 (and CI on acc(B) - acc(A) includes 0), then the oracle-vs-inferred gate source is not load-bearing and Section 4.1 is falsified for this architecture. Conversely, if acc(B) in regime (b) is within 1 point of acc(C) while acc(A) stays above, the dendritic gain depended on the quality of an externally supplied gate signal. A secondary falsifier: arm E within 2 points of arm A in regime (b) means dendrites bring nothing beyond a tuned optimiser plus sparsity.

Pitfalls. Prototype threshold must be tuned on validation; report cluster purity. Do not tune arm A on the blurry test stream.

### Experiment 2: Is sparsity an accuracy mechanism or only a compute mechanism, and is the fly's associative write rule needed on modern frozen features?

Setup. Cache frozen features once (timm ViT-B/16 and ResNet-50 on CIFAR-100 at 224 px; roughly 10-15 minutes on a T4). Class-incremental, 10 tasks x 10 classes, single pass, no replay. Expansion m = 10,000 for all expansion arms.

Arms (same features, same ridge grid).
1. Prototype/NCM on raw features (floor).
2. Dense Gaussian + ReLU + streaming ridge (RanPAC-style without PETL, FlyPrompt's router form).
3. Sparse Gaussian (p = 300 per row) + top-k (k = 3,000) + streaming ridge (Fly-CL defaults).
4. Same as 3 with k in {100, 500, 1,000} (1%, 5%, 10% coding levels: the fly's regime).
5. Sparse binary PN-to-KC (about 12% connectivity) + top-k 5% + FlyModel partial-freezing associative update (alpha = 0).
6. Same features as 5 + softmax regression by SGD (Shen's "logistic regression" arm).
7. Arm 3 with k/m reassigned at random per sample (random mask of the same cardinality) to separate "which units" from "how many".

Baseline numbers to compare against (for sanity, not as targets): Fly-CL ViT-B/16 CIFAR-100 overall accuracy 93.89 +/- 0.12 and RanPAC 94.21 +/- 0.11 (both with their own training schedules); FlyModel-vs-logistic gap 0.41 vs 0.21 on CIFAR-100 with ResNet18 features and 25 tasks of 4 classes.

Metrics. Mean accuracy over stages, last-stage accuracy, backward transfer, wall-clock of the post-extraction stage.

Falsifiers.
- F1: if arm 3 minus arm 2 is under 0.3 points (paired, 5 seeds, CI includes 0) at matched m, sparsity buys compute only in this regime.
- F2: if arm 5 minus arm 6 is under 5 points of absolute accuracy (Shen reported about 20), the associative rule's advantage does not carry to modern frozen features. If arm 5 is also below arm 3, it is not load-bearing there.
- F3: arm 7 within 0.3 points of arm 3 means selection of the strongest units adds nothing over random sparsity.

Pitfalls. Same ridge-penalty grid per arm; ridge penalty for ViT features is large (Fly-CL searched 1e6-1e9). Report feature-normalisation choice, since Fly-CL's Table 7 shows 87.87 to 93.89 depending on normalisation for ViT on CIFAR.

### Experiment 3: Is LLM confidence individuated once difficulty is controlled, with a continuous confidence signal?

Question. Moran and Whiting (binary verbalised, frontier models, temperature 0) found no individuation; Kadavath found a small own-model advantage (0.8717 vs 0.8460 AUROC in one direction, none in the other). Test with continuous log-probability confidence on open models.

Setup. TriviaQA validation subset of 2,000 questions (plus NQ-Open replication of 1,000). Models: Qwen2.5-1.5B, 3B, 7B (4-bit) and Llama-3.2-3B on a T4. For each model get a greedy answer, normalised answer log-probability (NLP), and a P(True) self-evaluation. Correctness scorer: alias match with containment fallback, audited by hand on 200 items per model to estimate model-dependent error (Cacioli's lesson).

Analysis. For model A's correctness, compare AUROC of A's own confidence with AUROC of B's confidence (pooled over B != A), restricting to accuracy-matched pairs (stratified subsampling so that |acc_A - acc_B| <= 2 points) and also controlling for shared difficulty (the leave-one-out mean correctness of the other models). Report delta = AUROC_own - AUROC_other and a conditional version using partial rank correlation.

Baselines. Own confidence vs "population" confidence: the mean confidence of the other models. Moran's Table 3 shows the population predictor often beats the internal one for frontier models (for example MathBench 0.742 vs 0.630).

Falsifier. The idea "LLM confidence carries model-specific information beyond a shared difficulty axis" is falsified if delta has a 95% bootstrap CI that includes 0 and the point estimate is under 0.02 AUROC for NLP in at least 3 of 4 models. It is supported if delta exceeds 0.03 with CI excluding 0 in at least 3 of 4 models.

Optional extension (needs no extra data). Fine-tune one model with a cheap version of ESMA's joint reward (or a LoRA probe as in Yax) and decompose the gain into output-consistency tracking vs accuracy tracking (delta r) on a held-out category.

Compute check. 2,000 questions x 4 models with short greedy answers: minutes for 1.5B-3B, about 20-30 minutes for 7B 4-bit on a T4. Bootstrap analysis is CPU-only.

## 6. Open questions

1. Is there any direct test of whether capacity limits or sparse codes improve generalisation (as opposed to continual-learning retention) in biology? I found none. In AI the best controlled positive case is the synthetic hard-WTA bottleneck (Gutheil et al.), which required at least about 20 tasks and failed with softmax.
2. Does striatal gating matter only at high WM demand? The 7T null was for a low-demand task; no high-demand fMRI counterpart was found in my searches.
3. Does the structured (non-random) PN-to-KC wiring change what the AI analogue should be? The authors' model says structure helps only for the overconvergent food-responsive channel; no AI paper in this thread tests structured vs random projections.
4. Why do Zou 2025 and Hard ASH disagree on whether sparsity changes plasticity?
5. Can ESMA-type metacognitive gains be decomposed into output-consistency vs accuracy tracking? If consistency dominates, the gains may not transfer to errors made with high consistency, which Yax et al. flag as undetectable from the inside.
6. Is Kumaran's causal-confidence result an artifact of how the steering vector is built? A control would steer along an abstention-propensity direction learned without confidence margins and compare the mediation share.
7. Are PBWM-style reward-trained gates competitive with gated recurrent nets or attention in 2024-2026 work? My two discovery searches for this returned nothing relevant; absence of evidence is weak here.
8. Leads not opened: https://www.alphaxiv.org/abs/2609.16540 (gating in state-space models, memorisation vs in-context learning), https://www.alphaxiv.org/abs/2610.04649 (gated target propagation for compositional generalisation in continual learning), https://www.alphaxiv.org/abs/2604.09670 (in-context superposition and human-like WM interference), https://www.alphaxiv.org/abs/2608.23251 (dendritic structure and plasticity), https://www.alphaxiv.org/abs/2609.37791 (context-based memory network). All [SNIPPET].
