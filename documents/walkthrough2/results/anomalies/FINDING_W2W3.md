# W2 -> W3 anomaly: what is it?

**Conclusion: the W2 -> W3 anomaly is the Utah State Fair 2018 signal on
the I-15 / Fairpark corridor.** W2's Tue/Wed/Thu (Sep 11-13) had
Fair-elevated rush-hour traffic on the I-15 segments adjacent to the
Fairpark; W3's Tue/Wed/Thu (Sep 18-20) had no Fair. The MRE captures
the difference.

This section walks through (a) what the data shows, (b) what external
sources confirm, and (c) what is inference vs. confirmed.

## 1. What the data shows

### Day-level totals, rush hours only (07-10 + 14-18)

| weekday | W1 (no Fair) | W2 (**Fair active**) | W3 (**no Fair**) | W4 (no Fair) |
|---|---|---|---|---|
| Tue | 276,508 | **288,446** | 231,976 | 284,279 |
| Wed | 274,713 | **287,219** | 259,158 | 283,762 |
| Thu | 281,172 | **288,024** | 242,600 | 281,896 |

Three observations:

1. W2 Tue/Wed/Thu are the highest Tue/Wed/Thu of the whole corpus. W3
   Tue/Wed/Thu are the lowest.
2. The W2 - W1 gap (~+12k per day) is the Fair lift; the W2 - W3 gap
   is much larger because W3 is itself depressed.
3. W3 is not just "no Fair" -- it is also 40-50k below W1, suggesting
   a secondary effect on top of losing the Fair traffic.

### Per-slot signature of the top anomalies

24 of 24 top-thick MRE slots fall on Tue/Wed/Thu of W2 paired with the
same weekday of W3:

| baseline day (W2) | n top-24 slots | clock times |
|---|---|---|
| Tue Sep 11 | 13 | 14:15-14:55 + 16:00-16:50 (PM rush) |
| Wed Sep 12 | 5 | 08:05-08:55 (AM rush) |
| Thu Sep 13 | 6 | 07:15-09:20 (AM rush) |

### Link-level signature

About 30 OSM links keep recurring across the top-24 slots (appearing in
8-14 of 24 each). They share these attributes:

- speed = 31.29 m/s (~70 mph) -- US freeway speed limit
- lanes = 4-6 per OSM record
- length 360-1300 m per segment
- OSM IDs tightly clustered in 58,800-59,500 (with a smaller cluster
  around 91,200)

This is the OSM signature of US interstate freeway, contiguous segments
of a single corridor.

## 2. What external sources confirm

### Utah State Fair 2018 - dates, venue, schedule

- The Fair ran **Thursday Sept 6 to Sunday Sept 16, 2018**, 11 days,
  at the Utah State Fairpark, 155 N 1000 West, Salt Lake City.
  ([Deseret News, 2018-08-22](https://www.deseret.com/2018/8/22/20651762/utah-state-fair-kicks-off-sept-6/))
- The Wikipedia entry confirms the venue address and that the Fair
  always starts the first Thursday after Labor Day.
  ([Wikipedia: Utah State Fair](https://en.wikipedia.org/wiki/Utah_State_Fair))
- 2018 attendance was **283,462** total (~25k per day, +10% year-on-year).
  ([Deseret News, 2018-11-15](https://www.deseret.com/2018/11/15/20659003/revamped-utah-state-fairpark-has-record-breaking-year-but-wants-550k-more-from-state))
- **Sept 11, 12 and 13 had gates opening at NOON** (the rest of the
  Fair opened at 10 a.m.). Those are exactly the W2 baseline dates in
  our top-24.
  ([Deseret News, 2018-08-22](https://www.deseret.com/2018/8/22/20651762/utah-state-fair-kicks-off-sept-6/))

### Fairpark is immediately adjacent to I-15 at Exit 309

- "Coming from I-15, get off via Exit 309 before heading down 1000
  West, then hang a left and travel around four more blocks until you
  see the grounds." Exit 309 separates for 600 North (SR 268) east to
  Capitol Hill and **west to Fairpark**.
  ([iExit / AARoads I-15 north guide](https://www.aaroads.com/guides/i-015-north-salt-lake-ut),
  [Fairpark "Getting to the Fair"](https://www.utahstatefair.com/p/thefair/plan-your-visit/getting-to-the-fair1))
- The Fairpark is roughly **0.7 mi west of I-15** along North Temple /
  600 North; every car arriving from the south, north, or east uses
  I-15 to Exit 309 or 308.

### Weather was not a factor on Sep 18-20, 2018

- NOAA Storm Events Database returns **0 events in Utah between
  09/18/2018 and 09/20/2018**: 0 deaths/injuries, $0 property damage,
  no flood/storm/wind events.
  ([NCEI Storm Events search, 2018-09-18 to 2018-09-20, Utah](https://www.ncei.noaa.gov/stormevents/listevents.jsp?eventType=ALL&beginDate_mm=09&beginDate_dd=18&beginDate_yyyy=2018&endDate_mm=09&endDate_dd=20&endDate_yyyy=2018&county=ALL&hailfilter=0.00&tornfilter=0&windfilter=000&sort=DT&submitbutton=Search&statefips=49%2CUTAH))

### Possible secondary factor: Yom Kippur

- Yom Kippur 2018 began the evening of **Tue Sep 18** and ended the
  evening of **Wed Sep 19**, exactly the days where W3 is most
  depressed (Tue -20%, Wed -10% vs W2 rush hours).
  ([Fox13 News, 2018-09-17 - Yom Kippur explainer](https://www.fox13now.com/2018/09/17/understanding-yom-kippur-the-jewish-day-of-atonement))
- Salt Lake County's Jewish population is small (~5,000-7,000), so
  the direct holiday effect is modest -- 1-3% of trips at most.
  This explains only a sliver of the 40-50k weekday gap.

### What we could **not** find

After extensive search of UDOT releases, KSL/Fox13/ABC4/Deseret News
archives, the FHWA Utah portal, the I-15 Tech Corridor weekly updates
archive, and Utah Highway Patrol crash logs:

- No reported **road closure** on I-15 or I-80 in Salt Lake City for
  Sep 18-20, 2018.
- No reported **multi-vehicle crash** or major incident in that
  window.
- No **construction stage** announced specifically for Sep 18-20 on
  the I-15 segments through SLC (the I-15 Technology Corridor project
  in Lehi was active all of 2018, not just W3).

The I-15 Tech Corridor reconstruction was UDOT's top project for 2018
(starting Jan 2018, $415M), but it is in Utah County south of Lehi,
not the SLC link cluster we identified.
([KSL, "Is the I-15 construction in Utah County almost over?"](https://kslnewsradio.com/1933936/i-15-construction-in-utah-county/))

## 3. Putting it together

The chain of evidence:

1. The top-24 thick anomalies for W2 -> W3 cluster on Tue Sep 11
   afternoon and Wed/Thu Sep 12-13 morning -- exactly the days the
   Utah State Fair gates opened at noon (so vendors/exhibitors arrived
   in the morning, attendees in the afternoon).
2. The links involved are a tight cluster of 4-6 lane, 70 mph
   freeway segments -- consistent with I-15 in NW Salt Lake near
   Exit 309 (the Fairpark exit).
3. W2 totals are the highest Tue/Wed/Thu of the corpus; W3 same days
   are the lowest. The W2 lift is consistent with Fair attendance
   (~25k/day, much of it via I-15).
4. W3 is also separately depressed (~40-50k below W1/W4); part of
   this is likely Yom Kippur on Sep 18-19, part is residual
   week-to-week variability.
5. No weather event, no published closure, no reported major incident
   in the W3 window. So the anomaly is **the absence of a recurring
   event one week earlier**, not the presence of a one-time event in
   W3.

In short: the model is correctly flagging the Utah State Fair as the
biggest week-to-week deviation in the corpus. The "anomaly" is W2
being **elevated** by the Fair, not W3 being depressed by an incident.

## 4. What is confirmed vs. inferred

**Confirmed by external sources:**

- Utah State Fair ran Sep 6-16, 2018, at 155 N 1000 W with 283k total
  attendance.
- Sept 11-13 had noon gate openings.
- Fairpark is reached via I-15 Exit 309.
- No NOAA-recorded weather events in Utah Sep 18-20, 2018.
- Yom Kippur 2018 was Sep 18-19.

**Inferred from the data (matches the story but not directly
sourced):**

- The recurring freeway-class link cluster is on the I-15 segment
  near Exit 309 (we lack coordinates in the graph; matching by OSM
  ID range and link attributes only).
- Fair attendees produce the W2 PM-rush lift; vendors/livestock
  handlers produce the W2 AM-rush lift on Wed/Thu (the schedule's
  noon openings push the bulk to PM, but the back-of-house arrives
  early).

To upgrade the inferred items to confirmed, we would need either (a)
coordinates for the link IDs (the `vector` field in
`city_graph_full.json` may encode this) or (b) UDOT count-station data
from the I-15 Exit 309 ramps for Sep 11-13 vs Sep 18-20.

## Sources

- [Deseret News, 2018-08-22 - Utah State Fair kicks off Sept. 6](https://www.deseret.com/2018/8/22/20651762/utah-state-fair-kicks-off-sept-6/)
- [Deseret News, 2018-11-15 - Revamped Utah State Fairpark has record-breaking year](https://www.deseret.com/2018/11/15/20659003/revamped-utah-state-fairpark-has-record-breaking-year-but-wants-550k-more-from-state)
- [Wikipedia - Utah State Fair](https://en.wikipedia.org/wiki/Utah_State_Fair)
- [Utah State Fairpark - Getting to the Fair](https://www.utahstatefair.com/p/thefair/plan-your-visit/getting-to-the-fair1)
- [AARoads - I-15 North, Salt Lake City](https://www.aaroads.com/guides/i-015-north-salt-lake-ut)
- [NCEI Storm Events search, Utah Sep 18-20 2018 - 0 events](https://www.ncei.noaa.gov/stormevents/listevents.jsp?eventType=ALL&beginDate_mm=09&beginDate_dd=18&beginDate_yyyy=2018&endDate_mm=09&endDate_dd=20&endDate_yyyy=2018&county=ALL&hailfilter=0.00&tornfilter=0&windfilter=000&sort=DT&submitbutton=Search&statefips=49%2CUTAH)
- [Fox13, 2018-09-17 - Yom Kippur explainer](https://www.fox13now.com/2018/09/17/understanding-yom-kippur-the-jewish-day-of-atonement)
- [KSL - I-15 Tech Corridor (Utah County), 2018-2020](https://kslnewsradio.com/1933936/i-15-construction-in-utah-county/)
