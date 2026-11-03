# Literature thread: memory consolidation (CLS, replay, replay selection) and AI implementations

Date of search: 2026-11-04. Written by a literature-thread agent. No code was run.

Reading conventions used throughout:
- "FULL" = I read page text of the paper itself (via the alphaXiv PDF reader), though for long papers the tool returns only the most relevant pages, so some sections were not seen. Each entry says which parts.
- "FETCH" = read via WebFetch, which passes the page through a small summariser model. Quotes and numbers from FETCH are second-hand and should be spot-checked against the primary before being relied on.
- "ABSTRACT" = only the abstract (or a paraphrase of it) was seen.
- "SNIPPET" = only a web-search result summary was seen. These are weakest; treat as pointers, not evidence.
- "from memory, unverified" = not opened; not used for any claim below unless labelled.

---

## 1. Summary

**Biology.**
1. Replay/ripples are causally needed for memory. Established at the level of "disrupting ripples or assembly-specific replay impairs later memory" (Girardeau 2009, Gridchyn 2020, Michon 2019; all SNIPPET-level in this thread, but the Gillespie 2021 paper, which I read in full, lists them as the supporting literature). What replay *does* (storage/maintenance vs planning vs retrieval) is contested.
2. Which memories get replayed is the weakest part of the field. Candidate selectors each have support, and some directly conflict:
   - Reward history: Gillespie 2021 (FULL, 4 rats), Michon 2019 (SNIPPET).
   - Not-recently-visited: Gillespie 2021 (FULL; recency effect R2 only 0.011 to 0.028 per subject, p small because of many events).
   - Expected utility, gain x need (Mattar and Daw 2018, FULL for the bioRxiv v4): a normative model, not data. Gillespie argues replay "seems unlikely to implement the value updates that could drive behavior" because replay content changes far slower than behaviour and is decoupled from the next choice.
   - Weak traces prioritised by sleep (Denis 2021 preprint, FULL): n=54 humans, 1-presentation word pairs. But the same paper's own introduction says the direction of the strength effect is unresolved, citing studies that find strong or middle-strength memories favoured.
   - Expected future relevance (Wilhelm 2011, SNIPPET).
   - Overlap with new learning (McClelland 2020): a theoretical proposal, explicitly not simulated as a replay mechanism ("Details not in visible text", and "Ongoing work"); the authors state SWIL's advantage was "not found for some other data sets and neural network architectures".
   No single study I read tests these selectors against one another.
3. Schema-dependent fast consolidation (Tse 2007) is a real, striking result but narrow (one lab, flavour-place pairs, rats) and its interpretation is contested (systems vs cellular consolidation; McClelland et al. note the new locations were adjacent to learned ones so "only the very easy part of neocortical consolidation may have been necessary"). I did not open the Tse primary paper; I read Squire's commentary (FETCH) and McClelland's discussion (FETCH).

**AI.**
4. Replay of real stored data is the robust finding. In van de Ven and Tolias (FULL), on split MNIST Class-IL, EWC/SI sit at about chance-for-this-protocol (20.01 / 19.99) while replay methods reach 90.79 (DGR) to 94.57 (iCaRL, 2000 exemplars).
5. Generative replay works on MNIST-scale data and degrades on natural images. van de Ven 2020 (FETCH) says "straight-forward implementations of GR break down for more challenging problems". The CLS-ER paper (FULL) says generative replay "performs sub-par in comparison to instance-based replay methods on challenging CL settings" (a claim in their related work, not their own experiment).
6. Against a tuned plain-replay baseline, most "smarter replay" gains are small, protocol-dependent, or reverse:
   - GSS and HAL (replay-selection/ replay-regularisation variants) score below plain ER on S-CIFAR-10 Class-IL in Buzzega et al (FULL), and GSS < ER in Mai et al (FULL).
   - Mai et al: simple tricks (NCM classifier, "review" fine-tune) on ER or MIR beat every un-tricked method; GDumb (train only on the buffer) beats ER and MIR on Split CIFAR-100 at M=5k/10k.
   - CLS-ER beats DER++ by +1.3 to +2.5 points on S-CIFAR-10 Class-IL but much more on S-Tiny-ImageNet; its single-semantic-memory ablation (EMA only) already gets most of the gain.
7. 2025-2026 "sleep/hippocampus-inspired" papers (Section 4.6) are almost all ABSTRACT-only here, report few or no numbers in abstracts, and I could not verify any of them. Do not cite them as evidence of effect.

**Dropped property (detail in Section 5).** The best-supported candidate is replay that is selected and dosed as a function of how the new item relates to existing knowledge (similarity-weighted interleaving plus a schema-consistent "fast path" that needs little or no replay), rather than uniform or loss-proxy sampling from a buffer. Honest status: it is dropped by essentially all AI work, but it is NOT demonstrated to be load-bearing in biology; the evidence is one linear-network simulation (positive, with authors' own caveat) plus one rat study (narrow, contested). Two further dropped properties are strongly documented in the AI-side review (Hayes 2021): the fast store is not consolidated-then-purged, and the buffer is used to train the same network that predicts rather than to consolidate into a separate slow system via index-like representations.

---

## 2. Evidence table

Grades: A = replicated or multiple independent lines, B = single solid study or consistent but narrow, C = single study, small n, or preprint, D = theory/model only or I saw only a secondary summary, X = contested.

### 2.1 Biology

| # | Claim | Grade | Source URL | What I read |
|---|---|---|---|---|
| B1 | Disrupting sharp-wave ripples after learning impairs spatial memory (test group n=7, stimulated controls n=7, unimplanted controls n=12) | B | https://pubmed.ncbi.nlm.nih.gov/19749750/ (Girardeau 2009) | SNIPPET (search summary only) |
| B2 | Assembly-specific disruption of sleep replay of one of two learned locations impairs recall of that location only; place-cell code reappears on relearning, so replay may be needed for selecting activity at recall rather than for stabilising the representation | B/X | https://research-explorer.ista.ac.at/record/7684 (Gridchyn 2020) | SNIPPET + institute news page; interpretation attributed to Csicsvari in secondary source |
| B3 | Post-learning ripple disruption impairs memory for the highly rewarded goal but not the low-reward goal; replay proportion correlates with reward size and task demands | B | https://pubmed.ncbi.nlm.nih.gov/31031113/ (Michon 2019, Curr Biol 29:1436) | SNIPPET only (could not open abstract; PubMed returned a cookie wall). Sample size not seen |
| B4 | Awake replay content is decoupled from the next choice; enriched for previously rewarded locations and for locations not recently visited; suppressed for the most recent unrewarded arm. 4 rats, well-trained, one task design | B (narrow) | https://escholarship.org/content/qt0537c2rq/qt0537c2rq.pdf (Gillespie 2021, Neuron 109:3149) | FULL text pages 1-9, 13-18 of the manuscript (intro, results on goal and recency, discussion). Numbers: remote events 35.1/41.1/34.5/47.2% of SWRs per subject; previous-goal replay vs later visits R2 = 0.008 to 0.023 (p = 0.49, 0.32, 0.18, 0.08); recency vs replay rate R2 = 0.011, 0.028, 0.012, 0.018 |
| B5 | Normative account: replay priority = gain x need (EVB). Reproduces forward-before-run and reverse-after-reward replay, initiation bias, reward-site bias, reward-biased sleep replay in simulation, "without parameter fitting" | D (model) | https://www.biorxiv.org/content/10.1101/225664v4.full.pdf (Mattar and Daw 2018, bioRxiv v4; published Nat Neurosci 21:1609) | FULL: abstract, figs 1-4 text, discussion, methods. Authors state gain "requires that the agent knows the effect of the backup on its policy prior to deciding" and call this "admittedly unrealistic"; simulations use two grid worlds |
| B6 | B4 and B5 disagree: Mattar and Daw predict replay prioritised to drive value updates for upcoming behaviour; Gillespie finds replay content slow-changing and not predictive of next choice and argues replay "seems unlikely to implement the value updates that could drive behavior" | X | both URLs above | FULL (both) |
| B7 | Sleep (2 h nap) reduces forgetting of weakly encoded word pairs vs wake: 1-presentation items lost -13.48% (wake) vs -1.53% (nap), B = 11.96 [3.48, 20.40], p = .002; n=54 (36 nap, 18 wake). Fast-spindle density, esp. SO-coupled, tracks consolidation of weak items only | C | https://www.biorxiv.org/content/10.1101/2020.04.03.022434.full.pdf (Denis et al., preprint of J Neurosci 2021) | FULL: abstract, intro, methods overview, behavioural results, discussion |
| B8 | Direction of the memory-strength effect on sleep consolidation is unresolved: weak favoured (Drosopoulos 2007, Schapiro 2017, Denis 2020), strong favoured (Tucker and Fishbein 2008, Wislowska 2017), inverted-U proposed (Stickgold 2009); the authors note good-vs-poor learner confounds | X | same Denis URL | FULL (their intro) |
| B9 | Expectation of a future test determines whether sleep benefits a memory (191 volunteers across two experiments, per press coverage) | C | https://pubmed.ncbi.nlm.nih.gov/21289163/ (Wilhelm 2011) | SNIPPET only |
| B10 | Schema-consistent flavour-place pairs learned in one trial become hippocampus-independent within 48 h (lesion at 48 h spares them); lesion at 3 h abolishes them; rearranging the arena every two sessions removes the fast learning; 6 flavour-place pairs over 6 weeks to build the schema | B / X | https://pmc.ncbi.nlm.nih.gov/articles/PMC2271071 (Squire's Science commentary on Tse 2007, not Tse itself) | FETCH of the commentary. Number of rats not stated there. Tse 2007 primary NOT opened |
| B11 | Caveat on B10: the new targets were adjacent to already-learned places, so possibly "only the very easy part of neocortical consolidation may have been necessary"; unclear how similar a new item must be | X | https://www.biorxiv.org/content/10.1101/2020.01.17.909804v1.full (McClelland et al. 2020 preprint) | FETCH of full text (chars 0-186,739) |
| B12 | In deep linear / Rumelhart-style networks, the projection of a new item onto already-known dimensions is learned rapidly without interleaving; a new dimension requires gradual interleaved learning; sparrowhawk trained focused has initial error 9.0 and "considerable interference with the existing birds" | D (model) | McClelland URL above | FETCH |
| B13 | Similarity-weighted interleaved learning (SWIL): replaying 1 sparrowhawk + known birds + 0.2 fish (3.4 items/epoch) gives "virtually identical results on a per epoch basis compared to full interleaving" (9 items/epoch), i.e. "2.5 times fewer presentations"; a uniform control at the same 3.4 items/epoch is "far less efficient" | D (model) | McClelland URL above | FETCH. 9-item toy hierarchy, linear net, 32 hidden units in one sim. Authors: advantage "not found for some other data sets and neural network architectures"; "important to be cautious in extrapolating" |
| B14 | Hippocampal indexing: hippocampus stores an index to neocortical loci, content lives in cortex (Teyler and Rudy 2007). Pattern separation in DG, completion in CA3 as classic view; Knierim and Neunuebel say CA3 can do either depending on input and attractor strength | A (textbook) but not verified by me | https://pmc.ncbi.nlm.nih.gov/articles/PMC3183227 and others returned by search | SNIPPET only. I did not open Yassa and Stark 2011 or Teyler and Rudy 2007 |
| B15 | Which memories are replayed "remains an open question" (Hayes et al. 2021, a review by AI and neuroscience authors); they cite weak, reward, fear, and overlap-with-new as candidates | secondary | https://arxiv.org/abs/2104.04132 | FULL pages 1-5, 10, 14-15, 18, 24-28 |
| B16 | Discrepancy to flag: Hayes 2021 attributes to "Schapiro et al., 2018" the claim that "weakly learned information was replayed more frequently". Denis's reference list cites Schapiro et al. 2017 (Sci Rep 7:14869) for weak-memory prioritisation, and that work is a behavioural sleep study, not a replay-frequency measurement. I did not find a Schapiro 2018 weak-memory paper | flag | Hayes and Denis URLs | FULL |

### 2.2 What is established vs contested (compact)

Established (multiple labs, causal manipulations):
- Ripple/replay disruption impairs hippocampus-dependent memory (B1, B2, B3; Gillespie 2021 lists further supporting studies, cited there not re-opened by me).
- Replay is not a verbatim copy of experience: it includes shortcuts never taken and remote places (Gillespie intro, citing Gupta 2010, Olafsdottir 2015; Hayes 2021 section 2.2).

Contested:
- Whether awake replay serves planning or storage/maintenance (Gillespie 2021 sets this out explicitly).
- Which memories are prioritised and by what signal (B4 to B9 above).
- Whether schema-consistent consolidation is truly "systems consolidation" (B10, B11).
- Whether any replay-selection rule is *causally* what rescues memories: no study I read manipulates the selection rule itself. Gridchyn 2020 manipulates which assembly is replayed (disruption), which is the closest, and it is SNIPPET-level for me.

Open (nobody I read has the data): whether, after consolidation, hippocampal traces are erased or merely never reactivated (stated as open in Hayes 2021 p.26-27).

---

## 3. AI implementations and effect sizes

All numbers copied from the cited tables. "Baseline" = what the paper itself compared against. I did not rerun anything.

### 3.1 Generative replay

| Paper | Benchmark | Result | Strong-baseline comparison | What I read |
|---|---|---|---|---|
| Shin et al. 2017, arXiv 1705.08690 | Permuted MNIST, MNIST<->SVHN, split MNIST (2-class tasks) | MNIST scholar chain: 98.81, 98.64, 98.58, 98.53, 98.56 % over 5 sequential solvers. Figure-only results otherwise | "ER" in the paper means exact replay of real data with old-solver targets, used as an upper bound, not a tuned modern replay baseline. No CIFAR. Authors: "we observed some performance loss while training the model on SVHN" | FULL |
| van de Ven and Tolias 2019, arXiv 1904.07734 | Split MNIST, permuted MNIST, 3 scenarios, 20 seeds | Split MNIST Class-IL: DGR 90.79 +/- 0.41, DGR+distill 91.79 +/- 0.32, iCaRL (2000 stored) 94.57 +/- 0.11, EWC 20.01, SI 19.99, offline 97.94. Permuted MNIST Class-IL: DGR+distill 96.38 +/- 0.03 vs iCaRL 94.85 +/- 0.03 and offline 97.59. Appendix: on permuted MNIST even 50,000 stored examples were "consistently outperformed by DGR+distill" | The one place generative replay beats stored data. Authors flag the limitation themselves: "MNIST-images are relatively easy to generate" | FULL |
| van de Ven et al. 2020 (Nat Commun) BI-R, via Europe PMC XML | Permuted MNIST 100 tasks; Split CIFAR-100 Class-IL | Text-stated: Split CIFAR-100 Class-IL, BI-R+SI 0.344 +/- 0.002 with replay-through-feedback vs 0.334 +/- 0.002 ablated (a 1.0-point gap, SEM-scale 0.002). Permuted MNIST (100 tasks) 0.904 +/- 0.005 vs 0.892 +/- 0.004 | Summariser reports internal replay uses "pre-trained, frozen convolutional layers", i.e. CIFAR-100 results depend on a pretrained feature extractor (unverified in primary figure/tables; I did not see the tables, which are figures). No comparison to stored-data replay was visible in the text I saw | FETCH, chars 0-200,000 of 235,167; results figures not readable |
| Hayes et al. 2021 (review) | Survey | "The vast majority of existing replay approaches in artificial neural networks replay raw pixel inputs"; feature-level replay (REMIND etc.) is closer to biology | n/a | FULL (partial) |

### 3.2 CLS-inspired dual-memory and tuned replay baselines

| Paper | Benchmark | Numbers | Notes | What I read |
|---|---|---|---|---|
| Buzzega et al. 2020 (DER/DER++), arXiv 2004.07211 | S-CIFAR-10 Class-IL, buffers 200/500/5120 | ER 44.79 / 57.74 / 82.47; DER++ 64.88 / 72.70 / 85.24; GSS 39.07 / 49.73 / 67.27; HAL 32.36 / 41.79 / 59.12; iCaRL 49.02 / 47.55 / 55.07; joint 92.20 | Authors: "a simple Experience Replay baseline ... consistently outperforms cutting-edge methods"; "other ER-based methods (HAL and GSS) show weaker results than ER itself". DER++ gain over ER is +20.1 / +15.0 / +2.8 points. Single-epoch protocol flagged as confounding forgetting with underfitting (joint training single epoch: 56.74 vs 92.20 multi-epoch) | FULL (main results, analysis, appendix tables) |
| Arani et al. 2022 (CLS-ER), arXiv 2201.12604 | S-CIFAR-10 Class-IL | CLS-ER 66.19+/-0.75 / 75.22+/-0.71 / 86.78+/-0.17 vs DER++ 64.88+/-1.17 / 72.70+/-1.36 / 85.24+/-0.49 (gap +1.31 / +2.52 / +1.54) and ER 44.79 / 57.74 / 82.47 | Baselines copied from Buzzega, not re-tuned by the authors ("baseline results are from Buzzega et al. 2020a"). CLS-ER hyperparameters are tuned per setting (Table S4). For S-Tiny-ImageNet they used 50 epochs vs 100 in DER++ | FULL (main tables, appendix) |
| same | S-Tiny-ImageNet Class-IL | CLS-ER 23.47+/-0.80 / 31.03+/-0.56 / 46.74+/-0.31 vs DER++ 10.96 / 19.38 / 39.02 and ER 8.49 / 9.99 / 27.40; joint 59.99 | Largest effect in the paper; authors report +176% over ER and +114% over DER++ at buffer 200 | same |
| same, ablation (Table S2, 10 runs) | S-CIFAR-10 / S-Tiny-ImageNet | Mean-ER (single EMA semantic memory) 61.88 / 70.40 / 84.84 and 17.68 / 24.97 / 45.69, vs full CLS-ER 66.19 / 75.22 / 86.78 and 23.47 / 31.03 / 46.74 | Most of the improvement over ER comes from replaying with an EMA-weight teacher (a mean-teacher mechanism); the second (fast/plastic) memory adds 1.9 to 5.8 points at small buffers. At S-CIFAR-10 buffer 200, Mean-ER (61.88) is below DER++ (64.88). The CLS interpretation is the authors' framing; the paper itself states "mimicking the complementary learning system in the brain more faithfully" is future work | same |
| Mai et al. 2021, arXiv 2101.10423 | Online Split CIFAR-100, Split Mini-ImageNet, CORe50-NC | CIFAR-100 M=1k/5k/10k: ER 7.6/17.0/18.4; MIR 7.6/18.2/19.3; GSS 7.7/11.3/13.4; iCaRL 16.7/19.2/18.8; GDumb 10.4/22.1/28.8; offline 49.7. CORe50-NC: ER 23.5/27.5/28.2; MIR 27.0/32.9/34.5; GDumb 15.1/28.1/32.6 | MIR (selects old samples whose loss rises most under the pending update: the nearest AI analogue to SWIL) is within noise of ER on CIFAR-100 but +5.4 / +6.3 over ER at M=5k/10k on CORe50-NC (SDs 1.7 to 3.5). GSS is below ER everywhere. With tricks: ER+RV (review) M=10k 32.0, MIR+RV 35.2, ER+NCM M=10k 31.0, all beating un-tricked best (28.8) | FULL (results sections and appendix tables) |

### 3.3 Selection / prioritisation in AI

- Hayes et al. 2021 (FULL, p.25): "uniform random sampling ... works well and is widely used in supervised classification"; "sampling strategies have not demonstrated significant success for supervised learning problems", while prioritised experience replay and hindsight replay "have seen more benefit" in RL. They state the efficiency of selection has largely been ignored; papers target accuracy only.
- Mai 2021 and Buzzega 2020 corroborate: GSS < ER. MIR is the exception on a larger dataset (above).
- CARES (arXiv 2610.32799, ABSTRACT only): label-free, geometry-based exemplar selection from a frozen pretrained feature space; reports 2.3 to 8.8x faster than coreset selectors on Tiny-ImageNet and up to 7.58 points over the strongest baseline under class imbalance. I did not see tables.
- No paper I opened implements gain x need (Mattar and Daw) in supervised continual learning. From memory, unverified: it is used in RL via Prioritised Experience Replay (Schaul 2016, cited by Hayes).

### 3.4 Claimed gains that did not hold up (documented in the sources I read)

1. Regularisation methods in the hard scenario: EWC/SI ~20% on split MNIST Class-IL (van de Ven 2019). Their Task-IL competitiveness depended on hyperparameter ranges "several orders of magnitude larger than those typically considered", so earlier reports of EWC failure (or success) were partly tuning artefacts (same paper).
2. Replay-selection variants GSS, HAL below plain ER on S-CIFAR-10 Class-IL (Buzzega; Mai for GSS).
3. A-GEM ~ fine-tuning in online Class-IL on CIFAR-100 (Mai: 3.6 vs finetune 3.7 at M=5k).
4. Evaluation protocol confounds: single-epoch settings confound forgetting with underfitting (Buzzega); reservoir/ring-buffer choice changes ER at low memory (Buzzega appendix); GDumb (no continual learning at all) beats many methods (Mai).
5. Generative replay at natural-image scale: BI-R's headline CIFAR-100 gain from replay-through-feedback is ~1 point and appears to ride on pretrained frozen features (FETCH-level; verify in the primary before citing).
6. Baseline fairness in CLS-ER: baselines inherited from another paper while the proposed method was tuned per setting and trained with fewer epochs on Tiny-ImageNet. I am not claiming the gain is false; it is a caution about reading +1 to +2 point gaps on S-CIFAR-10 as established (gap vs DER++ SD of 1.17 at buffer 200).

### 3.5 Sleep-style methods (no effect sizes obtained)

- Tadros et al. 2022 Nat Commun, "Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks" (SNIPPET; https://par.nsf.gov/biblio/10438306 and DOI 10.1038/s41467-022-34938-7 in search results). Uses an offline phase with local unsupervised Hebbian rules and noisy input in a converted spiking network. I did not read numbers.

### 3.6 2025-2026 items (all ABSTRACT-only; none verified)

| ID | Title (short) | What the abstract says | Numbers in abstract |
|---|---|---|---|
| arXiv 2606.08447 | Not Just After One: sleep-inspired replay | Several tasks trained in sequence, then an unsupervised sleep-like replay phase partially restores old-task performance; task-specific information "gradually decays" with more tasks (Bazhenov, Delanois, Krishnan) | none |
| arXiv 2611.01952 | Hippocampal consolidation without replay in fully fine-tuned networks | Training-only "hippocampal memory" records input geometry per synapse group and limits each synapse's movement; no class labels; reported to lead a frozen-backbone baseline by 17.5 points on TACRED and to cut absorption of established classes by near-duplicates "several-fold"; up to 8B parameters | 17.5 points TACRED |
| arXiv 2610.32799 | CARES | see 3.3 | 2.3-8.8x speed; up to 7.58 pts |
| arXiv 2606.03979 | Language Models Need Sleep (Behrouz, Hashemi, Javanmard, Mirrokni) | "Sleep" paradigm: knowledge-seeding distillation from a smaller to a larger model plus "Dreaming" with RL-generated synthetic curricula | none; no benchmarks named |
| arXiv 2508.16651 | HiCL | DG-style top-k sparse coding, CA3-like autoassociative memory, similarity-gated MoE, similarity-weighted EWC, prioritised replay; "near state-of-the-art" | none; no benchmarks named |

Titles seen in search listings but not opened (so no claims): 2509.00047 (KAIST, brain-inspired replay), 2512.00619 (predictive-coding vs backprop replay), 2606.17493 (robot offline skill consolidation), 2609.31630 (replay in "silent degrees of freedom"), 2611.02216 ("Do agents need to sleep?"), 2610.24453 (Hippo), 2611.02746 (replay with multi-teacher on-policy distillation), 2608.01297 (data augmentation as framework for hippocampal generalisation).

Note on abstract-level claims: three of the five opened 2026 abstracts name no benchmark numbers. A "brain-inspired" label in a title is not evidence the mechanism is what produces the gain (cf. the CLS-ER ablation above).

---

## 4. Dropped load-bearing properties

Ordered by my confidence that the property is (a) really absent from mainstream AI replay and (b) plausibly load-bearing. Confidence in (b) is low-to-moderate for every item; I flag what would change it.

### 4.1 PRIMARY ANSWER: replay targeting and dose conditioned on the relation between the new item and existing knowledge (SWIL plus the schema fast path)

What biology/theory says:
- McClelland et al. 2020 (FETCH): the amount of interleaving needed depends on how the new item overlaps with known structure. Consistent aspects "can be learned rapidly without interleaving"; only new dimensions need gradual interleaving; interleaving "can focus on the previously-known items within this branch". SWIL matched full interleaving with 2.5x fewer presentations and beat a dose-matched uniform control in their toy.
- Tse 2007 (via Squire, FETCH): schema-consistent items consolidate within 48 h; schema-inconsistent arrangement abolishes the effect.
- Gillespie 2021 (FULL): not-recently-visited and reliably rewarded places are replayed; the most recent unrewarded place is suppressed. Whatever the exact rule, replay content is a function of the memory's history and relation to the rest of the map, not a uniform draw.

What AI does: the buffer is sampled uniformly (reservoir) or by a loss/gradient/diversity proxy; the number of replay steps per new item is a fixed ratio chosen by hand. Hayes 2021 (FULL): "uniform random sampling ... works well" and selection "has not demonstrated significant success" in supervised learning. The one near-analogue, MIR, targets old samples that the pending update would interfere with, and it helps on CORe50-NC (+5.4 / +6.3 over ER) but not on CIFAR-100 (within noise).

What is NOT dropped but often mislabelled: iCaRL herding and class-balanced buffers are about buffer composition, not about the consolidation schedule.

What is specifically absent (to my reading of the sources): (i) a replay dose that scales down when the new data is consistent with learned structure; (ii) a measured, per-new-item similarity to the slow system's representation used to pick which old items to interleave; (iii) any test of the Tse prediction that a consistent item needs no replay at all.

Why this is not certain: the authors of SWIL report no advantage "for some other data sets and neural network architectures"; their own model is linear and nine items; Tse is one lab and the adjacency caveat applies. So the honest claim is "dropped, and the one specific theoretical reason to expect it matters has a failure on record".

### 4.2 Consolidate-then-release: capacity-limited fast store with eviction, and a slow learner trained only through replay

Source: Hayes et al. 2021 (FULL, p.25-27): many replay implementations "do not purge their memory buffer, which is not consistent with biology", "do not have a notion of waking (streaming/online) learning", the buffer "memory buffers ... are mostly used to train the neural network that makes predictions" whereas the hippocampal complex serves both replay and inference, "many existing replay approaches have only focused on modeling medial prefrontal cortex directly and do not have a fast learning network", and "none of the aforementioned CLS-inspired models use information from the neocortex-inspired network to influence training of the hippocampal-inspired network".
Counter-example in the sources: CLS-ER does maintain fast/slow weight copies and its plastic and stable models interact (FULL), but its "episodic memory" is still a raw-input buffer with reservoir sampling and no eviction after consolidation, and Mean-ER ablation shows most gain is EMA weight averaging.
Load-bearing status: unknown. Biology says hippocampal traces may be erased or just never reactivated (open question, Hayes p.26-27).

### 4.3 Replay of compressed/index-like representations, not inputs

Hayes (FULL): the brain replays "highly processed representations" consistent with hippocampal indexing; AI mostly replays pixels. Feature-level replay exists (REMIND, BI-R's internal replay) but those still use a frozen or slowly-updated feature extractor, and I did not verify an index-then-cortex-reconstructs mechanism in any paper I opened. Pattern separation/completion as a *retrieval* mechanism is similarly not found in the mainstream replay methods I read (HiCL's DG/CA3 modules are the exception by abstract, with no numbers I could check).

### 4.4 Other properties Hayes 2021 lists as unexplored or ignored (FULL, Table 1)
Partial replay "not explored in deep learning"; temporal structure "largely ignored" (static uncorrelated inputs); replay independent of input novelty ("Replay is largely the same independent of input novelty"); NREM vs REM distinction (DL "focuses on NREM replay and ignores REM replay"); replay and regularisation/normalisation operating independently rather than co-occurring (Hayes p.28).

### 4.5 Replay selection by gain x need
Mattar and Daw is explicit that computing gain needs the post-backup policy, which is "admittedly unrealistic" and needs an approximation. Nobody in the sources implemented a supervised-learning analogue. Its biological support is mixed (B6).

---

## 5. Candidate experiments

All sized for a free Colab T4 (about 15 GB VRAM), under one hour per condition, 5 seeds paired. Run-time figures are my estimates, not measured; the Mammoth code base (the Buzzega et al. repo, `github.com/aimagelab/mammoth`, link taken from the paper) contains ER, DER++ and a CLS-ER repo exists (`github.com/NeurAI-Lab/CLS-ER`, link taken from the paper); I did not open either. To fit the hour budget, cut epochs per task (the published protocols use 50) and use S-CIFAR-10 / S-CIFAR-100 with a small ResNet (ResNet-18 narrow) or cached frozen features; state the departure from published numbers.

### E1. Similarity-weighted replay dose in a deep, nonlinear net (tests SWIL outside the linear toy)
- Setup: S-CIFAR-100 in 10 tasks of 10 classes (Class-IL). At each new task, compute for each new class the cosine similarity of its mean penultimate feature to each old class mean (from the current network). Replay draws from the buffer with probability proportional to softmax(sim / T) over old classes, instead of uniform.
- Conditions at the same total number of replay samples per step: (a) uniform ER, (b) SWIL, (c) anti-SWIL (inverted weights), (d) uniform ER at 40% of the budget, (e) SWIL at 40% of the budget, (f) MIR (the existing interference-based selector, as the strong selector baseline).
- Strong baseline: tuned ER with NCM classifier and the review trick (Mai: these tricks alone gave about +80% relative), at buffer sizes 500 and 2000.
- Falsified if any of: (i) SWIL at equal budget does not exceed uniform ER by more than the paired seed SD over 5 seeds at both buffer sizes; (ii) SWIL-40% fails to match uniform-100% within 1 SD (this is the 2.5x claim); (iii) anti-SWIL performs the same as SWIL (then similarity is not doing the work); (iv) SWIL does not beat MIR (then it adds nothing over what exists).
- Reading the result: a null here is consistent with the McClelland authors' own report that SWIL did not generalise to some datasets/architectures, so it would sharpen rather than overturn the literature.

### E2. Schema fast-path test (Tse analogue): does a schema-consistent new class need no replay?
- Setup: build a schema by pretraining on a hierarchical dataset (e.g. CIFAR-100 with 20 superclasses; train on 4 classes from each of 5 superclasses as "schema"). Then introduce one new class that is (A) in a schema superclass, (B) in a held-out superclass, (C) from an unrelated domain (e.g. SVHN digit class), matched on number of training images. Learn each with a single short focused fine-tune (no replay), measuring (1) new-class accuracy after training and after N intervening steps on unrelated schema-neutral data (the "48 h" analogue), (2) drop in old-class accuracy, (3) the replay dose needed to bring old-class drop below 2 points.
- Baseline: the same protocol with uniform ER at fixed ratio; also a matched-feature-distance control, to separate "semantic consistency" from "low feature distance".
- Falsified if: old-class interference after focused training is not smaller for (A) than (B)/(C) once feature distance is matched, or the replay dose needed does not differ by condition. Also falsified in the Tse sense if the fast path appears only after long, many-epoch schema pretraining of a type that already saturates the network (i.e. it is just "easy class").
- Cost: pretraining ~10 min, each condition a few minutes; well under an hour.

### E3. Replay selection rules taken from the contested biology, at equal compute against a tuned uniform baseline
- Rules: (a) uniform, (b) weak-first (lowest margin/confidence on that buffer item; Denis 2021), (c) strong-first (highest margin; opposite pole of the same contested literature, which gives a direction-of-effect test), (d) non-recent-first (priority grows with steps since the item was last replayed; Gillespie 2021), (e) value-weighted gain x need analogue: priority = |expected loss reduction from one step on that item| x (frequency of its class in a held-out probe stream) (Mattar and Daw), (f) recency-suppressed (down-weight the newest task's items).
- Benchmarks: S-CIFAR-10 (buffer 200/500) and S-CIFAR-100 with a small ResNet, 5 seeds.
- Strong baseline: uniform ER with NCM and review tricks (Mai), plus DER++ as a modern anchor.
- Falsified if: no rule beats uniform ER by more than 1 point average accuracy (paired, 5 seeds) at either buffer size; or any gains vanish once the baseline gets NCM/review (as in the pattern Mai et al. found for fancier methods). A directional result is informative either way: if both (b) and (c) help or both hurt relative to uniform, the "memory strength determines replay priority" story is not what is driving AI gains.
- Note: because this is cheap, it also gives a free calibration of how big a seed SD is on this protocol (Buzzega report SDs of 1.2 to 1.9 at small buffers), which is needed to interpret E1.

### E4 (optional, if time): consolidate-then-release ablation
- Take CLS-ER or Mean-ER style EMA replay and add an eviction rule (remove an item after it has been replayed k times with stable teacher agreement), against never-evict at the same peak buffer. Falsified if never-evict at matched peak memory is not worse, i.e. eviction frees capacity without cost only if the consolidated items are in fact retained by the slow weights.

---

## 6. Open questions

1. What is the replay-selection rule in the brain? The sources I read propose reward, non-recency, weakness, future relevance, overlap with new learning, and expected utility; none tested against each other in one design, and my snippet-only sources (Michon, Wilhelm) may carry caveats I could not see.
2. Is replay content causally what determines what is retained, or does replay mostly maintain representations (Gillespie) or select activity at recall (Gridchyn, as interpreted in a news piece)? Matters because selection rules only help AI if selection is the mechanism.
3. Does SWIL's advantage survive depth, nonlinearity and real data? The authors say it did not on some datasets/architectures; nobody I found has tested deep nets (E1 is that test).
4. Are generative-replay failures on natural images a generator-quality issue only, or does feedback-based and feature-level replay (BI-R, REMIND) just sidestep it by using a pretrained extractor? I could not read BI-R's tables.
5. How much of CLS-ER's gain is the CLS structure vs EMA weight averaging? The authors' own Mean-ER ablation points to the latter for most of it; a decisive control would train a mean-teacher ER without any fast/slow framing.
6. What happens to a hippocampal trace after consolidation (erased vs silent)? Open in the review I read; has a direct AI analogue (E4).
7. Items I could not verify and that would change conclusions: the Tse 2007 primary paper; Michon 2019 sample size and effect; BI-R Table values; any of the 2026 numbers; Mattar and Daw's published (Nat Neurosci) version vs the bioRxiv v4 I read; McClelland 2020's published version vs the preprint; whether Schapiro et al. "2018" exists as Hayes cites it.

---

## 7. Source list (URLs actually opened, with how)

- https://www.biorxiv.org/content/10.1101/2020.01.17.909804v1.full : McClelland, McNaughton, Lampinen preprint (FETCH, whole text)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2271071 : Squire commentary on Tse 2007 (FETCH)
- https://www.biorxiv.org/content/10.1101/225664v4.full.pdf : Mattar and Daw (FULL, selected pages)
- https://escholarship.org/content/qt0537c2rq/qt0537c2rq.pdf : Gillespie et al. 2021 (FULL, selected pages)
- https://www.biorxiv.org/content/10.1101/2020.04.03.022434.full.pdf : Denis et al. (FULL, selected pages)
- https://arxiv.org/abs/1705.08690 : Shin et al. (FULL)
- https://arxiv.org/abs/1904.07734 : van de Ven and Tolias (FULL, selected pages)
- https://arxiv.org/abs/2201.12604 : CLS-ER (FULL, selected pages)
- https://arxiv.org/abs/2004.07211 : DER/DER++ (FULL, selected pages)
- https://arxiv.org/abs/2104.04132 : Hayes et al. replay review (FULL, selected pages)
- https://arxiv.org/abs/2101.10423 : Mai et al. (FULL, selected pages)
- https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7426273/fullTextXML : van de Ven et al. 2020 BI-R (FETCH, chars 0-200,000 of 235,167)
- https://arxiv.org/abs/2606.08447, https://arxiv.org/abs/2611.01952, https://arxiv.org/abs/2610.32799, https://arxiv.org/abs/2606.03979, https://arxiv.org/abs/2508.16651 : abstract pages (FETCH, paraphrased abstracts)
- Search-snippet-only (not opened): Girardeau 2009 (https://pubmed.ncbi.nlm.nih.gov/19749750/), Gridchyn 2020 (https://research-explorer.ista.ac.at/record/7684), Michon 2019 (https://pubmed.ncbi.nlm.nih.gov/31031113/), Wilhelm 2011 (https://pubmed.ncbi.nlm.nih.gov/21289163/), Teyler and Rudy 2007 / Yassa and Stark 2011 (https://pmc.ncbi.nlm.nih.gov/articles/PMC3183227), Tadros 2022 (https://par.nsf.gov/biblio/10438306)
