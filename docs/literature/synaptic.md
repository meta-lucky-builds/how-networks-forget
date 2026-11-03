# Thread: synaptic stability and capacity (literature notes)

Written 2026-11-04. No code was run. Nothing committed.

Reading depth key used throughout:
- FULL = full text opened via alphaXiv (`answer_pdf_queries`); I read the pages the tool returned, not necessarily every page.
- ABSTRACT = only the abstract/metadata record, retrieved through the Europe PMC REST API (the `PMID` is given; the PubMed URL is the canonical identifier, but I did not open the PubMed page itself). Publisher pages (nature.com, cell.com, science.org) returned redirects or 403, so no neuroscience primary was read in full.
- SNIPPET = only a search-result blurb. Not citable for claims.
- MEMORY = from memory, unverified.

Preprint caveat: the Dohare et al. numbers below come from the arXiv preprint v3 (Apr 2024), titled "Maintaining Plasticity in Deep Continual Learning". The Nature version (2024, 632:768) was seen only as an abstract record (PMID 39169245). Numbers may differ from the published paper.

---

## 1. Summary

1. **Human adult hippocampal neurogenesis cannot be cited as established biological motivation.** Two 2018 papers reached opposite conclusions on human tissue (Sorrells: not detected in 29 adults aged 18-77; Boldrini: thousands of immature neurons, ages 14-79). Single-nucleus work split the same way (Franjic 2022 negative; Zhou 2022 and Dumitru 2025 positive, using machine-learning classification). I could read only abstracts. The defensible citation is "contested in humans; present in some other mammals; the computational principle (add fresh units with no initial functional effect, protect them while they mature) stands on its own". Note that this principle is already used in AI (continual backprop, NORACL), independent of the biology.
2. **Synaptic cascades/metaplasticity are the best-founded theory in this thread but have almost no AI payoff so far.** Benna-Fusi (Nat Neurosci 2016) is an analytic result (abstract only: memory capacity scaling almost linearly in synapse number, versus square root). The one direct deep-RL test (Kaplanis 2018, FULL) helped on Cart-Pole/Catcher with 3 runs per condition and, by the authors' own statement, failed to match controls on Atari DQN. A 2024 review by Zenke and Laborieux (FULL) says the empirical success of cascade models on deep networks has been limited.
3. **The AI loss-of-plasticity literature is real but its effect sizes and mechanisms are benchmark-dependent, and some of its headline interventions fail to transfer.** Loss of plasticity is shown on permuted/random-label benchmarks (ImageNet binary 89% to 77% over 2000 tasks). A 2026 paper (Liu and Mou) finds the severity depends on how abruptly tasks switch; another 2026 paper (Hernandez-Garcia et al.) finds it also appears under stationary LLM pretraining, but with effect sizes around 1% of a probe AUC. Plasticity injection, which helped on the Atari aggregate (3 seeds), underperformed even the warm-start baseline in on-policy RL.
4. **Dropped load-bearing property (section 4):** in biology, the per-synapse consolidation state that confers stability is also the variable that decides what the plasticity-restoring machinery (downscaling, active forgetting) may touch, on multiple timescales, and forgetting appears to act by reversible suppression rather than erasure. In AI, protection (EWC/SI) and restoration (resets, shrink-and-perturb, L2-init) are mostly separate, and the two methods that do share a signal (UPGD, continual backprop) use a single-timescale utility trace. The biological side of this claim is partly inference from reviews and abstracts; I grade it "plausible, not established".

---

## 2. Evidence table

Grades: **E** = established (replicated, multiple independent sources I saw), **C** = contested (sources I saw disagree), **P** = plausible/partial (one line of evidence or theory-only), **A** = AI empirical result (single study unless stated), **U** = unverified (not opened, or snippet/memory).

| # | Claim | Grade | Source URL | What I read |
|---|---|---|---|---|
| B1 | Adult human hippocampal neurogenesis is absent or extremely rare; young neurons undetected in adults 18-77 (17 postmortem controls + 12 epilepsy surgical resections); few isolated young neurons at ages 7 and 13; macaque neurogenesis present early postnatal, decreasing in juvenile | C | https://pubmed.ncbi.nlm.nih.gov/29513649/ (Sorrells 2018 Nature) | ABSTRACT |
| B2 | Healthy humans aged 14-79 show similar numbers of intermediate neural progenitors and thousands of immature neurons across age; older subjects have less angiogenesis/neuroplasticity and a smaller quiescent progenitor pool in anterior-mid DG | C | https://pubmed.ncbi.nlm.nih.gov/29625071/ (Boldrini 2018 Cell Stem Cell) | ABSTRACT |
| B3 | Thousands of immature neurons in DG of neurologically healthy humans; number/maturation declines in Alzheimer's | C | https://pubmed.ncbi.nlm.nih.gov/30911133/ (Moreno-Jimenez 2019 Nat Med) | ABSTRACT |
| B4 | snRNA-seq across five hippocampal-entorhinal subregions in humans, macaques, pigs: neurogenic signatures in mouse, pig, macaque but not humans | C | https://pubmed.ncbi.nlm.nih.gov/34798047/ (Franjic 2022 Neuron) | ABSTRACT |
| B5 | snRNA-seq plus machine-learning identifies immature dentate granule cells across lifespan; rare proliferating progenitors; fewer in AD | C | https://pubmed.ncbi.nlm.nih.gov/35794479/ (Zhou 2022 Nature) | ABSTRACT. Donor and nuclei counts not in the record. |
| B6 | Ki67 staining plus machine learning identifies proliferating neural progenitors in adult human DG (snRNA-seq of birth through adulthood) | C | https://pubmed.ncbi.nlm.nih.gov/40608919/ (Dumitru 2025 Science) | ABSTRACT. Counts not in the record. |
| B7 | Review argues no reason to abandon the idea that adult-generated neurons contribute to human plasticity | P (opinion) | https://pubmed.ncbi.nlm.nih.gov/29681514/ (Kempermann 2018) | ABSTRACT |
| B8 | "Single-nucleus sequencing finds no adult hippocampal neurogenesis in humans" is a Nat Neurosci comment (Wiseman 2022) on a Neuron paper (PMID 34798047) | context only | https://pubmed.ncbi.nlm.nih.gov/34992287/ | ABSTRACT (title only, no abstract) |
| B9 | The disagreement is partly methodological (sparse nuclei capture, clustering thresholds, marker choice) | U | search-result summary only; a Neuron perspective titled "Mapping human adult hippocampal neurogenesis with single-cell transcriptomics: Reconciling controversy or fueling the debate?" https://www.cell.com/neuron/fulltext/S0896-6273(23)00206-4 | SNIPPET only (title seen). Do not cite for content. |
| B10 | Rodent adult neurogenesis is robust | U | none | MEMORY, unverified |
| M1 | Multi-timescale (cascade) synapses give memory capacity scaling almost linearly in synapse number vs square root for simple synapses; reproduce delayed expression, metaplasticity, spacing effects | P (theory) | https://pubmed.ncbi.nlm.nih.gov/27694992/ (Benna and Fusi 2016) | ABSTRACT. Model structure (chain of N variables, C_k = 2^(k-1), g_k,k+1 proportional to 2^-(k+2)) read from Kaplanis 2018, FULL. |
| M2 | Cascade model outperforms alternatives on storage plus retention | P (theory) | https://pubmed.ncbi.nlm.nih.gov/15721245/ (Fusi 2005 Neuron) | ABSTRACT |
| M3 | Two enablers of continual learning in theory: internal synaptic state over several timescales, and "intelligent" plasticity that uses that state per synapse; BCM sliding threshold had experimental support "with some success" but studies also found synaptic metaplasticity not tied to a neuron-wide threshold; cascade-in-deep-nets success so far limited | P (review, secondary) | https://arxiv.org/abs/2405.16922 (Zenke and Laborieux 2024) | FULL (selected pages) |
| M4 | BCM theory 30-year review | context | https://pubmed.ncbi.nlm.nih.gov/23080416/ (Cooper and Bear 2012) | ABSTRACT, no numbers. Evidence strength for sliding threshold taken only from M3's characterization. |
| M5 | Slow homeostatic plasticity alone cannot stabilize Hebbian plasticity; fast compensatory processes also needed | P | https://pubmed.ncbi.nlm.nih.gov/28093557/ (Zenke and Gerstner 2017) | ABSTRACT |
| S1 | SHY: waking drives net synaptic potentiation, sleep renormalizes | C | https://pubmed.ncbi.nlm.nih.gov/24411729/ (Tononi and Cirelli 2014) | ABSTRACT |
| S2 | Supportive: axon-spine interface decreased about 18% after sleep vs wake, 6,920 synapses measured in mouse cortex | A/E-ish (single lab, one tissue) | https://pubmed.ncbi.nlm.nih.gov/28154076/ (de Vivo 2017 Science) | ABSTRACT |
| S3 | Supportive in Drosophila: synapse size/number increased with waking, decreased only with sleep, three circuits | P | https://pubmed.ncbi.nlm.nih.gov/21700878/ (Bushey 2011) | ABSTRACT |
| S4 | Critical/contradicting: firing-rate homeostasis in rat visual cortex occurred in wake and was suppressed by sleep | C | https://pubmed.ncbi.nlm.nih.gov/26997481/ (Hengen 2016 Cell) | ABSTRACT |
| S5 | Critical: in mouse V1, responses to the experienced orientation grew across post-stimulus sleep but not wake; authors argue conflicts with SHY | C | https://pubmed.ncbi.nlm.nih.gov/26285006/ (Durkin 2016) | ABSTRACT |
| S6 | Critical: most cortical synapses are up-regulated by sleep slow oscillations | C | https://pubmed.ncbi.nlm.nih.gov/28453998/ (Timofeev and Chauvette 2017) | ABSTRACT |
| S7 | Sleep after motor learning promoted branch-specific spine formation in mouse motor cortex (selective, not global) | P | https://pubmed.ncbi.nlm.nih.gov/24904169/ (Yang 2014) | ABSTRACT |
| S8 | Review proposing sleep clears accumulated synaptic information ("sleep is for forgetting"); critical review "Erasing synapses in sleep: is it time to be SHY?" | context | https://pubmed.ncbi.nlm.nih.gov/28100731/ ; https://pubmed.ncbi.nlm.nih.gov/22530156/ | ABSTRACT |
| F1 | Intrinsic active forgetting: dopamine from forgetting cells acts on engram cells via Rac1/cofilin; also neurogenesis-based and interference-based forgetting | P (review) | https://pubmed.ncbi.nlm.nih.gov/28772119/ (Davis and Zhong 2017) | ABSTRACT |
| F2 | Rac activity drives memory decay and interference forgetting in flies; inhibiting it extends early memory from a few hours to more than one day | P | https://pubmed.ncbi.nlm.nih.gov/20178749/ (Shuai 2010 Cell) | ABSTRACT |
| F3 | Dopaminergic activity rises with locomotion, falls at rest; sleep-promoting manipulations reduce it and improve memory retention in Drosophila (the forgetting machinery is itself state-gated) | P | https://pubmed.ncbi.nlm.nih.gov/26073942/ (Berry 2015 Cell) | ABSTRACT |
| F4 | Forgetting is circuit remodeling that makes engram cells inaccessible to recall cues (i.e., suppression of access); rates depend on environment | P (review) | https://pubmed.ncbi.nlm.nih.gov/35027710/ (Ryan and Frankland 2022) | ABSTRACT |
| F5 | Brain-wide regulated decay, mostly during sleep, removes selected memories | P (review) | https://pubmed.ncbi.nlm.nih.gov/23369831/ (Hardt 2013) | ABSTRACT |
| K1 | Critical-period timing and closure set by experience, inhibition, molecular brakes; adult reactivation possible | E (reviews) | https://pubmed.ncbi.nlm.nih.gov/15217343/ (Hensch 2004) ; https://pubmed.ncbi.nlm.nih.gov/32503914/ (Reh 2020) | ABSTRACTS |
| K2 | Degrading chondroitin sulfate proteoglycans in adult rats reactivated ocular-dominance plasticity | P | https://pubmed.ncbi.nlm.nih.gov/12424383/ (Pizzorusso 2002) | ABSTRACT |
| K3 | Environmental enrichment reactivates adult ocular-dominance plasticity, linked to higher serotonin and BDNF and reduced GABA inhibition | P | https://pubmed.ncbi.nlm.nih.gov/20713044/ (Baroncelli 2010) | ABSTRACT |
| A1 | Loss of plasticity on continual ImageNet (binary): 89% on an early task to 77% (about linear-network level) on the 2000th task; Adam and dropout worsen it; L2 and shrink-and-perturb ease it but are hyperparameter-sensitive; continual backprop appears to maintain plasticity, task 5000 better than task 1 | A | https://arxiv.org/abs/2306.13812 | FULL (preprint v3). Nature version abstract only: https://pubmed.ncbi.nlm.nih.gov/39169245/ |
| A2 | Critical periods in DNNs (blur deficit on CIFAR-10): see section 3 | A | https://arxiv.org/abs/1711.08856 | FULL |
| A3 | Plasticity injection on Atari 57 (Double DQN, 200M frames, 3 seeds) | A | https://arxiv.org/abs/2305.15555 | FULL |
| A4 | Benna-Fusi in RL | A | https://arxiv.org/abs/1802.07239 | FULL |
| A5 | Warm-start problem and shrink-and-perturb | A | https://arxiv.org/abs/1910.08475 | FULL |
| A6 | L2-init (regenerative regularization) | A | https://arxiv.org/abs/2308.11958 | FULL |
| A7 | UPGD (utility-gated gradient plus perturbation) | A | https://arxiv.org/abs/2404.00781 | FULL |
| A8 | On-policy RL plasticity loss study | A | https://arxiv.org/abs/2405.19153 | FULL |
| A9 | Gradual-change study (plasticity loss depends on task-switch abruptness) | A | https://arxiv.org/abs/2602.09234 | FULL |
| A10 | Plasticity loss in GPT-style LMs 5M-314M | A | https://arxiv.org/abs/2606.24752 | FULL |
| A11 | FIRE (Frobenius-isometry reinitialization) | A | https://arxiv.org/abs/2602.08040 | FULL |
| A12 | Sleep-inspired replay plus synaptic downscaling | A | https://arxiv.org/abs/2209.05245 | FULL |
| A13 | Growing/elastic networks and plasticity | A | https://arxiv.org/abs/2608.01475 | FULL |
| A14 | NORACL neurogenesis-inspired growth | A | https://arxiv.org/abs/2604.27031 | FULL |
| A15 | Sleep-like unsupervised replay (Tadros 2022 Nat Commun) | A | https://pubmed.ncbi.nlm.nih.gov/36522325/ | ABSTRACT, no numbers |
| A16 | EWC, original | A | https://pubmed.ncbi.nlm.nih.gov/28292907/ | ABSTRACT. Sequential Atari 2600 and digit-dataset tasks; no numbers in the record. |
| A17 | 2026 papers seen only as search snippets (do not cite for claims): Neuroevolution continual RL https://arxiv.org/abs/2610.01583 (snippet says no RL method has yet consistently balanced adaptation and retention); ViTs that never stop learning https://arxiv.org/abs/2603.07787 ; co-observation facet https://arxiv.org/abs/2608.18803 ; Bellman equation for plasticity https://arxiv.org/abs/2609.10776 ; Neurogenesis Deep Learning (Draelos) https://arxiv.org/abs/1612.03770 ; self-regulated neurogenesis https://arxiv.org/abs/2403.14684 | U | as listed | SNIPPET |

---

## 3. AI implementations and effect sizes

Common caution: most results below use permuted-pixel or random-label benchmarks. These isolate trainability but are artificial. Several papers select hyperparameters per method on the same benchmark they report.

### Continual backprop (CBP), Dohare et al. (FULL, preprint v3)
- Mechanism: gradient descent plus, at every step, reinitialize a fraction rho of lowest-utility eligible hidden units; incoming weights resampled from the initial distribution; outgoing weights set to zero; new units protected for a maturity threshold m (example values in the algorithm box: step 1e-4, replacement rate 1e-4, decay 0.99, maturity 100). Utility mixes a contribution term and an adaptation term.
- Benchmarks and numbers: Continual ImageNet (binary classification tasks): baseline backprop 89% to 77% over 2000 tasks; CBP's accuracy on task 5000 exceeded task 1 (read from text; figure values not extractable). Online Permuted MNIST (3 hidden layers x 2000 units, SGD step 0.003, 30 runs, y-axis range 91-96%): only CBP non-degrading; S&P almost no drop; L2 helps; Adam, dropout, online-norm worse than plain backprop.
- Stated limits: S&P and L2 only work in a narrow hyperparameter range; CBP "much less sensitive". Hyperparameter sweeps on Continual ImageNet used 10 runs per config, then 20 more for the best (so 30 total).
- Did not hold up elsewhere:
  - FIRE (FULL) reports node-reset methods (CBP, SNR, ReDo) "show poor overall performance" in warm-start/continual vision generalization (citing Lee et al. 2024 that trainability gains give limited generalization gains). This contradicts the framing that CBP is a general solution; it is consistent with Dohare measuring online training accuracy.
  - Liu and Mou 2026 (FULL) find ReDo fails on a language task and S&P/spectral regularization are volatile on harder vision benchmarks, while L2 works on ResNet-18 but fails on an MLP. Dohare found L2 helpful on an MLP. These two sources disagree on L2.

### Shrink and perturb (Ash and Adams 2020, FULL)
- Warm-start CIFAR-10 validation accuracy, ResNet SGD: random init 56.2 (1.0), warm start 51.7 (0.9); ResNet Adam 78.0 (0.6) vs 74.4 (0.9). Shrink-perturb with lambda 0.6 matched random-init accuracy while training faster in the online experiment; noise-only perturbation closed less of the gap.
- Mechanism note from the paper: shrinking preserves the learned hypothesis (argmax) in bias-free ReLU nets and rebalances gradient magnitudes between old and new data.
- Later papers: lambda and noise scale are sensitive (Dohare); in on-policy RL a soft continuous variant plus LayerNorm was the best (Juliani and Ash).

### L2-init / regenerative regularization (Kumar et al., FULL)
- Five continual supervised problems; 3 seeds for tuning, 10 more for the reported best; performance in top 3 on all five (Adam). Comparable to CBP and ReDo; beats L2 on random-label MNIST/CIFAR.
- Stability data point (Permuted MNIST, 20 tasks, Table 2): L2 Init lambda 1e-3: BWT -63.4%, total online avg accuracy 81.1%. EWC beta 10: BWT -39.0%, 75.8%. L2 Init + EWC: BWT -51.1%, 80.0%. So combining independent stability and plasticity mechanisms landed between the two on forgetting while keeping most of the plasticity; it did not get both. The authors say the study "focused exclusively on maintaining plasticity, leaving aside forgetting".

### Plasticity injection (Nikishin et al. 2023, FULL)
- Double DQN, 57 Atari games, 200M frames, 3 seeds, IQM. One injection at 50M frames. Aggregate improvement claimed as 20% over other dynamic-plasticity methods (shrink-and-perturb, resets, width scaling); the paper says differences from baseline are insignificant for everything except injection.
- Per-game (Table 1, informal visual categories): 12 consistent improvement, 18 minor, 24 negligible, 3 negative (Assault, Demon Attack, Robotank).
- Efficiency claim: matches a larger net from scratch, saving about 20 hours on an A100.
- Failure to transfer: in on-policy PPO (Juliani and Ash, FULL; gridworld, CoinRun, Montezuma RND) plasticity injection underperformed even the warm-start baseline in all three shift conditions, CReLU only worked in window/expand, final-layer reset failed in all three. The authors hypothesize violation of the iid assumption (untested).

### UPGD (Elsayed and Mahmood, ICLR 2024, FULL)
- Update: w <- w - alpha (grad + noise) (1 - U), U in [0,1] a scaled utility (second-order Taylor estimate of loss increase if the weight were zeroed). High-utility weights are protected from both gradient and noise; low-utility weights get both.
- Benchmarks: streaming (no replay, no task boundaries) Input-permuted MNIST, Label-permuted EMNIST, CIFAR-10, mini-ImageNet (frozen ResNet-50 features + 2-layer head), and PPO on MuJoCo (Walker2d, Ant, Humanoid, HumanoidStandup; 40 runs). 20 runs elsewhere.
- Reported pattern: methods addressing only forgetting (streaming EWC/SI/MAS) keep decaying on input-permuted MNIST; methods addressing only plasticity (S&P) hold performance at a lower level on label-permuted problems where forgetting matters (S-EWC best on mini-ImageNet in the text); UPGD "among the best" on both. The numeric forgetting/plasticity figure values did not extract reliably, so I give no effect sizes.
- Ablation: utility gating on top of SGD alone prevents decay on input-permuted MNIST and improves continually on label-permuted problems; weight decay and perturbation matter more on input-permuted MNIST. "Non-protecting" UPGD helps only plasticity. I did not see an ablation using a random mask of equal density in place of utility (see experiment 2).
- Limitation stated: utility computed per weight assuming others fixed; piecewise-stationary assumption.

### Benna-Fusi in RL (Kaplanis et al. 2018, FULL)
- Tabular 10x10 grid with two alternating goals (24 epochs x 10,000 episodes): Benna-Fusi agents relearn faster than control; 3 runs, mean +/- 1 s.d.
- Deep RL: DQN-like, two tasks (Cart-Pole, Catcher), 40 epochs x 20,000 episodes, 30 Benna-Fusi variables per parameter, 1.5-2x runtime; control agents often could not relearn Cart-Pole after Catcher; Benna-Fusi relearned in all epochs; 3 runs per agent. In Catcher it sometimes converged slower in early epochs.
- Within-task without replay: Cart-Pole learned "to perfection in most cases" by BF; controls did not hold a good policy at any tested learning rate (1e-3 to 1e-6); Catcher control learned faster.
- Self-reported failure: initial experiments with larger DQN on Atari "struggled to reach the same level of performance as the control agents".
- No comparison to EWC/SI in that paper. Not replicated on modern benchmarks in what I read.

### Critical learning periods in DNNs (Achille et al., ICLR 2019, FULL)
- Blur deficit (downsample to 8x8, upsample) on CIFAR-10 for the first t0 epochs, then clean training. Two versions of the numbers appear in the document I read: the first paragraph says test error rises from about 6.4% to over 18% when blur persists over 140 epochs (about 300% increase), with an inline editor note that the plot shows 8% and 6.4% is for the later ResNet; the ICLR text says blur not removed within the first 40-60 epochs gives "up to a threefold increase in error". Treat the figures as approximate and flag the internal inconsistency.
- Deficit sensitivity peaks in the early rapid-learning phase (about epoch 30 for a 40-epoch window) and tracks the trace of the Fisher Information. Vertical flip and label permutation (high-level deficits) produce no critical period; noise deficit milder than blur; deeper nets show larger effects; weight decay lengthens the period. Pre-training on blurred data hurts final accuracy.
- Authors' own scoping: not claiming DNNs are a valid model of neurobiology; hypothesize that biological pruning may be a consequence rather than a cause of information-plasticity loss.

### Sleep-inspired and downscaling work (FULL for Robinson 2022)
- Tripartite artificial sleep on split CIFAR-100 (10 tasks of 10 classes, class-incremental, frozen conv features): NREM veridical replay, REM generative replay, and "synaptic downscaling" implemented as zeroing the smallest fraction p of weights per layer at each task start, p in {0, 0.25, 0.5, 0.75, 0.9}. The paper reports p = 0.75 best, p > 0.9 hurts recent tasks, and generative replay is necessary (without it accuracy on previous tasks collapses). No numeric accuracy table appeared in the text I read.
- This is magnitude pruning relabelled as downscaling; the relation to biological size-dependent downscaling is the authors' first-order approximation.
- Tadros 2022 (abstract only): sleep-like offline phase with local Hebbian plasticity and noisy input recovered otherwise forgotten tasks.

### Growth/neurogenesis-inspired
- Kong and Sutton 2026 (FULL): online permuted MNIST/FMNIST, SGD step 0.001, 10,000 and 40,000 samples per task. Fixed MLPs lose plasticity; "staged growing" networks (earlier units frozen) also lose it; "adaptive growing" networks (all connections adaptive, new random unit per task) keep accuracy even though dead-unit fraction keeps rising; "elastic" networks that also prune estimated-dead units hold roughly 0% dormancy and a compact size. No numeric table in text. Notable: dead-unit fraction rising without accuracy loss undercuts dormancy as the cause of plasticity loss.
- NORACL 2026 (FULL, preprint): MLPs on permuted/rotated/binary-split MNIST and binary-split CIFAR-10 (frozen ResNet18 features), 5 seeds. Claims final accuracy at or above oracle-sized static EWC baselines with 10-22% fewer parameters; permuted MNIST 2-layer: about +6% over the large static EWC at about 10% fewer parameters; full-model ablation accuracy 79.4 +/- 0.7 (permuted), 74.9 +/- 1.7 (rotated), 73.9 +/- 2.5 (binary split). CIFAR-10 head: 86.3 +/- 1.1 vs static EWC 84.7-85.4 (within error of the best static, 85.4 +/- 0.8). The authors concede a large static oracle would likely outperform both. Limited to MLPs, MNIST scale.

### Scale, language, and the "is it an artifact" dispute
- Hernandez-Garcia et al. 2026 (FULL): GPT-style LMs 5M-314M non-embedding params, multilingual cycling, probe on held-out Vietnamese; all sizes eventually show rising probe AUC; onset follows T = 1.3e-5 * P^0.8269 task instances; extrapolates 1B params to about 360 task instances (1.8T tokens). Effect magnitudes are small (curves within about -1.5% to +1% of the first-cycle AUC). Also seen in stationary multilingual training for 5M, 12M, 27M. They say they did not find a "smoking gun" correlate (parameter magnitude, dormant units, lazy/collapsed attention heads are correlational). Fit has "several arguable outliers".
- Liu and Mou 2026 (FULL): 4 benchmarks, 5 seeds; unregularized MLP on random-label MNIST collapses from 100% training accuracy to chance after about 50 tasks under abrupt switches; output/input interpolation with step 0.02 holds near-perfect accuracy; step 1 (abrupt) loses it by about 20 tasks (MNIST) and 12 tasks (bigram cipher). Conclusion: loss of plasticity benchmarks may overstate severity relative to gradual real-world change. Direct tension with the stationary-training result above; the two are not contradictory in mechanism but differ in whether abruptness is necessary.

### FIRE (Han et al., ICLR 2026, FULL)
- Defines stability as squared Frobenius distance to past weights (SFE) and plasticity as deviation from isometry (DfI); reinitializes by orthogonal Procrustes projection via Newton-Schulz (5 iterations, under 1% extra training time). Warm-start CIFAR-10/ResNet-18, CIFAR-100/ViT-Tiny, Tiny-ImageNet/VGG-16, GPT-0.1B on OpenWebText, Atari DQN, HumanoidBench SAC. Claims superior or competitive to S&P, DASH, plasticity injection. On CIFAR-100/ViT, DASH was best. No numeric accuracy tables in the text I read. Assumes access to past data. Useful for experiments here as an explicit two-metric (SFE, DfI) protocol, though both are weight-space proxies, and they are not the same as functional retention.

### Claims that failed or are weak (collected)
1. Plasticity injection: on-policy RL worse than warm start (Juliani and Ash); on Atari, 24 of 57 negligible, 3 negative; 3 seeds.
2. Benna-Fusi: Atari DQN failure self-reported; no modern replication seen; deep-net empirical success described as limited by Zenke and Laborieux.
3. Dropout and Adam worsen plasticity in Dohare's setting; the L2-init paper nevertheless uses Adam as the main optimizer and reports its method works there, so Adam's effect is setting-dependent.
4. Dormant-unit fraction is not a reliable causal marker (Kong and Sutton: accuracy maintained while dormancy rises; Hernandez-Garcia: correlational only).
5. Node-reset methods give limited generalization gains in warm-start settings (FIRE citing Lee et al.).
6. Critical-period analogy in DNNs: effect depends on depth, weight decay, deficit type; the paper itself disclaims biological validity.

---

## 4. Dropped load-bearing properties

### Primary candidate: one consolidation state, read by both protection and recycling, across timescales; recycling by reversible suppression
Claim: in biology the quantity that makes a synapse stable is the same quantity that determines how exposed it is to the machinery that restores capacity, and this state evolves on a wide range of timescales. In AI, protection and restoration use different signals or a single timescale.

Biological side (graded "plausible, partial"; it is a synthesis across reviews and abstracts, not one decisive experiment):
- Cascade/metaplastic theory: weights are read from a fast variable that is coupled in both directions to slower hidden variables; deep variables hold old information and restore it to the visible weight (Benna-Fusi; mechanism read in Kaplanis FULL). So consolidation depth and ease of change are the same variable.
- Zenke and Laborieux (FULL) state the two requirements explicitly: internal state over many timescales, and plasticity rules that use that state per synapse.
- Restoration events appear to be state-gated, not uniform: sleep selectively favors spine formation on particular dendritic branches after learning (Yang 2014, abstract); dopamine-driven forgetting activity falls with sleep and retention improves (Berry 2015, abstract); regulated decay "mostly during sleep" removes selected memories (Hardt 2013, abstract); forgetting renders engram cells inaccessible rather than erasing them (Ryan and Frankland 2022, abstract). Critical-period closure is implemented by actively maintained brakes (PV inhibition, perineuronal nets) that can be released (Pizzorusso 2002; Baroncelli 2010; Hensch 2004; abstracts).
- What I did not find in anything I opened: a direct experiment showing that a synapse's consolidation depth predicts its probability of being downscaled or forgotten. This link is inference. Furthermore SHY itself is contested (Hengen 2016, Durkin 2016, Timofeev 2017 abstracts disagree on whether sleep downscales or potentiates). So the "downscaling targets weakly consolidated synapses" half is the least secure; the "forgetting is gated by a state variable and acts by suppression" half has more support but only from reviews/abstracts and Drosophila.

AI side (these are results I read):
- EWC/SI-type protection uses importance fixed at task ends and accumulating; one source notes importance, once attributed, never decreases (Robinson 2022 discussion) and NORACL's theory argues EWC's effective plasticity decays monotonically (authors' theoretical analysis, not an independent measurement).
- Restoration methods that ignore consolidation (shrink-and-perturb uniform shrink; random-init noise) erase what they touch; FIRE's own framing: aggressive reinit erases useful knowledge, conservative reinit gives little plasticity.
- Joint use of independent methods is not additive: L2 Init + EWC gave BWT -51.1% (between -39.0% for EWC and -63.4% for L2 Init) (Kumar Table 2).
- The methods that do share one signal across both jobs are UPGD (utility gate on both the gradient and the noise) and CBP (utility selects what to reset, with a maturity-protection window and zero outgoing weights). UPGD's ablation says gating is what lets it avoid forgetting; "non-protecting" UPGD helps only plasticity. Both use a single exponential-moving-average utility trace (one timescale), computed from current-model sensitivity, so a weight that is unimportant now but important to a task that recurs after a long gap is eligible for recycling. That is the exact failure the power-law cascade is designed to avoid. Whether this actually hurts is untested in what I read (experiment 1).
- Recycling in AI is erasure (reset to init, shrink, noise). Nothing I read implements suppression with a recoverable copy.

So the property AI has mostly dropped is: **a multi-timescale per-synapse (or per-unit) state that simultaneously defines stability and recycling eligibility, with recycling implemented as reversible demotion into slower storage rather than erasure.** UPGD/CBP are the partial exceptions (shared signal, single timescale, erasure).

### Secondary candidates (weaker evidence)
- State-dependent global gate for releasing brakes (neuromodulator/sleep state; critical-period reopening). AI loss of plasticity happens passively with no release control. Evidence: abstract-level only; DNN critical periods show closure emerges without any regulation (Achille), i.e., it may be a property of learning dynamics, with biology adding control on top.
- Bounded synaptic efficacy / finite dynamic range (the setting in which cascades are needed). Related AI findings: weight magnitude growth correlates with plasticity loss (Dohare; Juliani and Ash GLM found weight magnitude predictive of train performance and dead-unit count predictive of test performance; both correlational). L2, S&P, L2-init partially reimpose bounds. Not "dropped" so much as rediscovered piecemeal.
- Structural turnover gated by maturity (adult neurogenesis analog): zero-output-weight insertion plus a maturity period appears in CBP and NORACL. This is implemented in AI but its biological anchor in humans is contested (section 1).

---

## 5. Candidate experiments

All designs are un-run proposals. Time estimates are guesses (MEMORY, unverified): small MLP or CNN on a free Colab T4 with data held in GPU memory. Use at least 10 seeds for MLP experiments and report mean, 95% CI over seeds, and the per-seed paired difference. Give every method an equal hyperparameter budget (same number of trials on a held-out split of tasks). Use the authors' released code where it exists (UPGD: https://github.com/mohmdelsayed/upgd ; CBP: https://github.com/shibhansh/loss-of-plasticity, both URLs taken from the papers I read, not opened).

### Measuring stability and plasticity separately (use in all experiments)
- Plasticity (trainability): from the current weights at the end of each task block, train on a brand-new probe task (a fresh random permutation of the same data, never used for training the main stream; for LM-style variants a held-out language) for a fixed small budget; report area under the online accuracy curve over the first K steps, normalized by the same probe from a fresh initialization. Optimizer state reset before the probe; probe updates discarded. This mirrors the AUC probe in Hernandez-Garcia et al. and the average-online-task-accuracy metric in Kumar et al.
- Stability (retention): (a) zero-shot accuracy on held-out test data of earlier tasks, no further training (backward transfer, as in Kumar's BWT); (b) savings: steps needed to regain 90% of that task's earlier best accuracy, relative to the same measurement from fresh init, since forgetting can be recoverable. Report both because a method that never learns anything also never forgets (a point made in Kumar et al.).
- Never infer plasticity from main-stream accuracy alone; UPGD argues performance reflects plasticity only when forgetting is not also at play.

### Experiment 1: single-timescale vs multi-timescale utility
- Question: does a multi-timescale utility trace improve retention after long gaps without costing plasticity?
- Setup: permuted-MNIST stream (MLP 784-300-150 or similar, ReLU, mini-batch 128 or streaming) over 10 fixed permutations that recur according to a schedule: each permutation reappears after gaps drawn from {5, 20, 100, 400} tasks, interleaved with 200+ one-off random permutations as filler (so recycling pressure is real). 500-1000 tasks, each a short pass (about 100-200 steps).
- Conditions: (a) SGD; (b) UPGD-W with a tuned single trace; (c) UPGD-W variant whose utility is the max (or sum) across K=3-4 traces with decay constants spanning 1 to 3 orders of magnitude, a cheap stand-in for a Benna-Fusi chain; (d) the same two-trace variant with a Benna-Fusi coupling (hidden weight copies that regularize the visible weights, as in Kaplanis) if time permits; (e) CBP; (f) L2-init; (g) streaming EWC.
- Prediction: savings on recurring permutations at gap >= 100 tasks is larger for (c)/(d) than (b) at matched plasticity-probe AUC; (b) drops off beyond about 1/(1 - beta_u) tasks.
- Falsified if: with equal tuning budgets, (c) shows no paired improvement in savings at gaps >= 100 over (b) (95% CI of the paired difference includes 0 across 10 seeds), or the plasticity probe AUC of (c) is below (b) by more than one pooled SE, or the improvement vanishes when (b) is simply given its best slow beta_u (that would mean the benefit is a hyperparameter artifact, not multi-timescale structure).
- Cost: about 10 conditions x 10 seeds; each run is tens of thousands of small-batch steps; plausible within an hour per condition on a T4, not verified.

### Experiment 2: coordinated vs uncoordinated restoration (the missing control)
- Question: is the benefit of utility-gated perturbation due to coordination with protection, or just to perturbing fewer weights?
- Setup: label-permuted EMNIST or CIFAR-10 with frozen features and a 2-layer head (UPGD's setting), streaming, labels permuted every 2500 steps, 400 tasks, with 10% of tasks being exact repeats of earlier label permutations so that retention is measurable.
- 2 x 3 design. Protection: {none, utility gate on gradient}. Restoration: {none, noise gated by the same utility, noise on a random mask with the same expected density and the same total noise energy, redrawn each step}. Add S&P and streaming EWC as external references.
- Prediction (coordination hypothesis): utility-gated noise plus protection beats random-mask noise plus protection on retention (zero-shot BWT and savings) at equal probe AUC.
- Falsified if: the random-mask control matches the utility-gated version within one SE on both stability and plasticity metrics across 10 seeds. Then the coupling is not what matters, and the biological coordination story gains nothing from this evidence.
- Why it is cheap: both arms reuse the UPGD update with the gate swapped; I did not see this control in the UPGD ablation I read.

### Experiment 3: selective vs global "brake release" after a critical-period deficit
- Question: after a deficit has closed learning (Achille-style), does releasing plasticity selectively where the network's information is concentrated recover more clean-data accuracy per unit of disruption than a global release?
- Setup: small CNN or ResNet-ish network on CIFAR-10 (full set or 20k subset), blur deficit (8x8 down/up-sample) for the first t0 epochs with t0 beyond the critical window found in a pilot (find it first: train with t0 in {0, 10, 20, 40, 60}; scaled-down versions may not show the 3x error increase, which would end the experiment honestly). Then clean training for the remaining epochs.
- Conditions at the transition epoch: none; global S&P tuned for best final accuracy; layer-targeted S&P applied only to layers with the highest Fisher trace (as in Achille's layer-wise information plot) at matched total parameter displacement (L2 distance moved); CBP-style unit resets restricted to the same layers; FIRE as an extra baseline. Reference: trained on clean data from scratch for the same total epochs.
- Metrics: plasticity-restoration = final clean test accuracy gap to the clean reference; stability = accuracy on the blurred test distribution right after the intervention and at the end (what was learned during the deficit that should survive if useful), plus L2 displacement.
- Prediction: targeted release closes more of the gap than global release at equal displacement.
- Falsified if: global release is equal or better at matched displacement (paired across 5 seeds; each run is a short CNN training, estimated well under an hour on a T4 though unmeasured), or if no critical-period effect appears in the pilot at this scale.

---

## 6. Open questions

1. Is there any direct measurement linking a synapse's consolidation state to its probability of downscaling or active forgetting? I found only reviews and abstracts that are compatible with it. The SHY literature I saw (abstracts) disagrees on the direction of net change.
2. Full text of Sorrells 2018, Boldrini 2018, Franjic 2022, Zhou 2022, Dumitru 2025 not read: donor numbers, postmortem intervals, fixation, and marker definitions (DCX, PSA-NCAM, Ki67) remain unchecked here. Any citation of human neurogenesis should be revisited with the primaries and with the methodological reviews.
3. Does the loss-of-plasticity effect size matter at LLM scale? Hernandez-Garcia et al. report changes of order 1% in probe AUC over many billions of tokens; Liu and Mou argue abruptness inflates severity. A direct head-to-head on one benchmark with both abrupt and gradual schedules, plus stationary controls, is missing.
4. Dohare et al. Nature version: confirm that the numbers (89% to 77%, 2000 tasks) match the published figures; I read the preprint.
5. Benna-Fusi on modern deep-learning benchmarks: no replication seen beyond Kaplanis (3 runs) and Laborieux (binarized-network metaplasticity, read second-hand through Zenke and Laborieux, reportedly on par with EWC on permuted MNIST). Does a cheap two-to-four-timescale approximation capture the benefit?
6. Are weight-space proxies (SFE, DfI, weight norm) adequate stability and plasticity measures? Gulcehre et al. (cited by Nikishin) question rank as a proxy; FIRE's proxies are unvalidated in my reading. Functional measures (savings, probe AUC) are safer.
7. 2026 papers seen only as snippets (neuroevolution for continual RL, dynamical isometry, SingularClip, "Bellman optimality equation for plasticity", co-observation) were not read and may overlap with experiments 1 to 3.
8. Whether human critical-period reopening (adult amblyopia treatments) can motivate the "brake release" design: no human source opened (Achille cites amblyopia; secondary).
