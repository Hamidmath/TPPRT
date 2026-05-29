# Professor ladder (7 rows per chain) + event-direction 15-link fits

These tables are the most recent results on the 840-pair eval and on the
event-direction 12-pair experiment restricted to the 15 anomaly links
inside the 2 km circle at (40.722348, -111.904691).

Underlying JSON files:
- `results/sp_prof_ladder.json`
- `results/tp_new_prof_ladder.json`
- `results/tp_event_15link_<mode>.json`

## 1. SP ladder, alpha = 0.0508 fixed (840 pairs)

| mode           | Ov. MRE | alpha_s | alpha_l | alpha_len |
|----------------|---------|---------|---------|-----------|
| uniform prior  | 1.3127  | -       | -       | -         |
| diffuse prior  | 1.0697  | -       | -       | -         |
| + lanes only   | 1.0596  | -       | +0.411  | -         |
| + speed only   | 1.0028  | +3.050  | -       | -         |
| + length only  | 1.0696  | -       | -       | -0.013    |
| + both         | 0.9997  | +2.775  | +0.242  | -         |
| + three        | 0.9996  | +2.773  | +0.241  | +0.010    |

## 2. TP-new ladder, (beta, rho) = (0.102, 0.101) fixed (840 pairs)

| mode           | Ov. MRE | alpha_s | alpha_l | alpha_len |
|----------------|---------|---------|---------|-----------|
| uniform prior  | 1.3119  | -       | -       | -         |
| diffuse prior  | 1.0413  | -       | -       | -         |
| + lanes only   | 1.0323  | -       | +0.377  | -         |
| + speed only   | 0.9725  | +3.052  | -       | -         |
| + length only  | 1.0413  | -       | -       | -0.010    |
| + both         | 0.9701  | +2.816  | +0.210  | -         |
| + three        | 0.9701  | +2.814  | +0.209  | +0.010    |

**Key read on the two new rows (length only, three):**
alpha_len pins essentially at zero in both chains. Length adds nothing on
the 840-pair background eval. The "+ three" row reproduces "+ both" to
four decimal places.

**TP vs SP:** TP-new beats SP at every common row by ~0.02 to 0.03.

## 3. Event-direction TP-new fit on the 15-link region

- Chain: TP-new (new-form, mass-conserving)
- (beta, rho) = (0.102, 0.101), FIXED
- Pairs: (Sep 11 16:MM, Sep 18 16:MM) for MM in {00, 05, ..., 55} (12 pairs)
- Input = diffused popularity at Sep 11
- Target = diffused popularity at Sep 18
- Loss = Overall MRE on the 15 anomaly links inside the 2 km circle

| mode          | region MRE (mean) | region MRE (median) | full OV | alpha_s | alpha_l | alpha_len |
|---------------|------------------:|--------------------:|--------:|--------:|--------:|----------:|
| + speed only  | 3.184             | 2.236               | 1.515   | -153    | -       | -         |
| + lanes only  | 4.321             | 3.726               | 1.467   | -       | -10.0   | -         |
| + length only | 4.782             | 2.227               | 1.836   | -       | -       | +20.4     |
| + both        | 2.782             | 1.640               | 1.672   | -114    | +31.0   | -         |
| **+ three**   | **2.624**         | 1.650               | 1.880   | -90     | +14.5   | -23.0     |

**Key read:** All five event-direction fits are bad (region MRE ~2.6 to
4.8). The optimizer drives the road-type exponents to extreme drifting
values. "+ three" is best but still very high. The chain cannot reproduce
the Sep 11 to Sep 18 contrast on this region from a prior that already
encodes the high-traffic week.

## 15-link region IDs

```
58905, 58889, 58864, 58870, 58898, 58888, 58869, 58862,
59159, 59261, 59071, 59157, 58874, 58923, 59260
```

These are the 15 of the top-200 per-slot mean-difference links
(Sep 11 vs Sep 18, 16:00-16:55, Sep 11 >= 5 per-slot filter) whose
midpoints fall inside a 2 km circle at (40.722348 N, -111.904691 W).

## 4. Event-direction fit on the WHOLE network (no region restriction)

- Chain dynamics fixed at the geometric-fit values:
  - SP: alpha = 0.0508
  - TP-new: beta = 0.102, rho = 0.101
- Pairs: (Sep 11 16:MM, Sep 18 16:MM) for MM in {00, 05, ..., 55} (12 pairs)
- Input = diffused popularity at Sep 11, target = diffused popularity at Sep 18
- Loss = Overall MRE on the whole 99,716-link network
- L-BFGS-B, 5 parallel starts per job
- No alpha_len (length parameter dropped from this experiment)

| chain                       | mode         | Ov. MRE (mean) | median  | alpha_s | alpha_l |
|-----------------------------|--------------|---------------:|--------:|--------:|--------:|
| SP (alpha = 0.0508)         | + speed only | **1.0653**     | 1.0642  | +2.367  | -       |
| SP                          | + lanes only | 1.1068         | 1.1039  | -       | +0.356  |
| SP                          | + both       | 1.0653         | 1.0642  | +2.397  | -0.021  |
| TP-new (beta=0.102, rho=0.101) | + speed only | **1.0879**  | 1.0885  | +2.460  | -       |
| TP                          | + lanes only | 1.1324         | 1.1310  | -       | +0.355  |
| TP                          | + both       | 1.0877         | 1.0882  | +2.554  | -0.069  |

**Key reads:**

1. **SP beats TP at every row** in event direction on the whole network --
   the opposite of the 840-pair background result (where TP wins). When
   the chain is asked to predict event-time traffic from a prior that
   already encodes the elevated week-2 state, the simpler single-phase
   form does slightly better.
2. **"+ both" collapses to "+ speed only"** in both chains. The lane
   exponent at the joint optimum is ~zero (-0.02 for SP, -0.07 for TP);
   Ov.MRE matches "+ speed only" to 4 decimals.
3. **Best fit overall: SP + speed only, Ov.MRE = 1.065** with
   alpha_s = +2.4. Median per-pair MRE (1.064) is essentially equal to
   the mean -- the 12 5-min bins contribute uniformly.

Underlying JSONs:
- `results/sp_event_full_<mode>.json`
- `results/tp_event_full_<mode>.json`
for `mode in {speed_only, lanes_only, both}`.
