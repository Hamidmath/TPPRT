Below is what your professor actually told you, separated into action items. I have tried to keep only what he actually said and noted where you said "yes/ok" but he was making a substantive point you may have missed.

---

## A. Things you must DO

### 1. Restructure the whole paper into "baseline → why it fails → our idea → why it works"

This is the single biggest message in the call. He said the paper is "written in the wrong order" right now. The right order:

1. Present a **baseline** approach that a reader would naturally try first (e.g., single-phase PageRank).
2. **Show it does not work** well enough on the data.
3. Introduce the new idea (two-phase split with calibrated β, ρ; diffusion smoothing).
4. **Quantify empirically** that the new idea fixes what the baseline missed.
5. Then a second new idea, and so on.

He stressed that reviewers want to see:
- New contributions identified **clearly**, either in one section or one section per idea.
- Each contribution backed by an experiment showing improvement over a baseline.
- Without this, a reviewer will think "this is just standard stuff" and reject.

**Concrete baselines you need to run and report against:**
- **Single-phase PageRank** vs your two-phase PageRank (this is the headline baseline).
- **PageRank with badly chosen β, ρ** vs calibrated β, ρ.
- He suggests showing the geometric-fit and the prediction error are both better in your two-phase setup than in those baselines.

### 2. Redo the diffusion γ calibration — the "variance retained" plot is the wrong tool

He spent ~10 minutes on this and you said "ok, ok" but I don't think you got the recipe. He explicitly disagreed with using the variance-retained vs γ plot to pick γ. His reasoning:

- "Low variance" is **not** automatically better. At γ = ∞ everything has the same value (variance = 0), and that is obviously useless.
- The plot blends "real road-to-road variation we want to keep" with "high-frequency single-trip noise we want to smooth out", so it cannot tell you which γ is right.

**His recipe** (this is the substantive instruction):

1. For each (bin, link), look at the values across the **4 different weeks** of data.
2. Compute the **cross-week variation** at that (bin, link).
3. Take the **median** of those cross-week variations across all (bin, link).
4. That median is the natural noise floor: things below that scale are high-frequency noise; things above it are real signal.
5. Set diffusion γ so that the smoothing scale is **roughly that level** (within a factor of 2 is fine).

Anything you smooth below this scale is signal that was real but you erased; anything you don't smooth above this scale is noise you kept. The median-cross-week variation is the right place to draw the line.

He acknowledged: "I don't quite know how γ corresponds to a spatial smoothing scale." So you will have to map γ → effective smoothing radius yourself (e.g., run synthetic experiments, or measure on the data).

### 3. The obs_noise (σ_z) choice is **NOT** a major paper section — push it to the appendix or one paragraph

He was very clear: map-matching σ tuning is **not a contribution** of yours. You are using existing technique (Newson-Krumm HMM + Viterbi). His instructions:

- Just **use σ = 4 m**. It is close to Newson-Krumm's reported 4.07 m, and there is a previous paper that says "set it this way" — follow that.
- One sentence in the "experimental setup" or "pipeline" section saying "we set σ = 4 m, calibrated as described in Newson-Krumm and verified for our corpus in Appendix X".
- Move the full sweep / table to an **appendix** if Sigspatial allows appendices (you said you have not checked).
- For the σ outlier instability, use a **trimmed standard deviation**: estimate σ, drop anything outside 3σ (or another threshold), recompute. He called this "fairly simple but it tends to work".

The whole obs_noise discussion is "a detail of implementation". Don't expand it into a chapter.

### 4. Rewrite the Related Work to argue ONE thing: nobody has modeled traffic at this scale

Your current Related Work has been growing into a tools-survey. He said most of it does **not** belong in Related Work. The proper Related Work should answer **one question**:

> Has anyone built a per-link traffic model at the scale of a whole city (100k links), calibrated on data this size?

His exact ranking of what to keep:
- **MATSim**: discuss in detail. He pointed out MATSim is the product of a 30-year research group, so you need to compare your goal with MATSim's goal head-on.
- **OD-pair estimation work**: keep, but mention briefly as a "much coarser modeling" predecessor.
- **PageRank-for-urban-centrality** (the Agryzkov 2023 paper, etc.): keep, but only to argue they did not measure at our scale or calibrate carefully.
- **The PageRank-for-traffic paper you found yesterday**: **read it carefully**. You need to know whether it is in our space or not. He flagged this as "the big question for related work".
- **Sensor/neural-network forecasting (DCRNN, STGCN)**: do NOT include in Related Work. He was explicit (~50:09): "are they trying to predict the sensor values, or are they trying to predict the street traffic on every length? So just the sensor value, right? ... These are tools that we use, and when we introduce the tool, we say this person introduced the tool ... but you don't need to talk about it in related work, because it's not related to our same goal." Since we don't actually use DCRNN/STGCN either, omit entirely.

Things he wants you to **remove** from Related Work:
- Anything that is just a tool you use (HMM Newson-Krumm, diffusion / graph signal processing, etc.). Introduce these in the body when you use them, not in Related Work.

### 5. Add the "repeated routes" justification for the geometric assumption

You mentioned a paper that argues people travel the same routes over and over (a "repeated process"). He liked this. Use it as the **why** for the geometric distribution. (He did flag: the cited paper argues for a **single** geometric; you use two geometric phases, so your story is the natural extension.)

### 6. The figure choice: use the bottom histogram (the one with all routes including short links)

He preferred the bottom panel. The boundary blip at the top is not worth worrying about. Use that one.

### 7. Cross-week prediction is the *main point of the paper*

He said this explicitly: "this is the whole point of the study". The validation that matters is:
- Learn parameters on one week of data.
- Apply to the next week.
- Compute prediction error.
- If your model is good, prediction error should be on the order of the cross-week noise floor (the median cross-week variation from item 2).
- If you do not learn parameters, error should be larger.

This needs to be the headline experiment of the paper, not a side note.

---

## B. Things you must NOT do

1. **Do not run PageRank locally on sub-graphs of the city** (the Beijing-bottleneck idea you brought up). He said "I wouldn't push this idea too far". Drop it.

2. **Do not remove short or low-traffic links from the network.** You suggested this. He said no — it disconnects the graph. The diffusion is supposed to handle these; that is exactly the principled approach.

3. **Do not make obs_noise σ a feature of the paper.** It is an implementation detail.

---

## C. Open question he flagged that you owe him an answer on

The PageRank-for-traffic paper you found "yesterday night". **Read it carefully.** Answer these specifically:
- Did they model **every link** in a city? At what scale?
- Did they **calibrate** the random-walk parameters from data, the way you do?
- Did they **validate** with held-out weeks?
- Most importantly: **what is new in our approach** compared to theirs?

If they did most of what we do, the paper's story has to be reframed around what specifically we add (most likely: the two-phase decomposition + calibration). If they did much less, our novelty argument is much easier.

---

## D. Summary of the "story" he wants the paper to tell

> We want to model traffic on every road link in a whole city, calibrated and validated on real data. Prior work has either modeled coarser units (OD pairs) or modeled sensor outputs (not per-link), or used PageRank without serious calibration at this scale. Our pipeline: map-match (standard, σ=4 m per Newson-Krumm); accumulate into a per-bin × per-link count matrix; smooth with graph diffusion (calibrated to the cross-week noise floor); run a two-phase PageRank chain whose phase-length parameters (β, ρ) are read off the empirical route-length distribution. We show this beats: (i) single-phase PageRank, (ii) untuned PageRank, on next-week prediction.

That is the paper. Most of what is in the draft right now needs to be reordered around this story.
