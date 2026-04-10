# Video Presentation Script — 4 Minutes

---

## INTRO (0:00 - 0:20)

**[Screen: App loaded, "How It Works" tab visible with architecture diagram]**

> "Hi, we're [team name] from Texas A&M. We built a Public Health Trend Intelligence system for Track C — AI for Social Good.
>
> It answers one question: if you're a Health Minister looking at COVID-19 data for 180+ countries, what's getting worse, what's getting better, and what should you do about it?
>
> Everything runs 100% inside Snowflake — 4 marketplace datasets, ML forecasting, and Cortex AI."

---

## HOW IT WORKS (0:20 - 0:50)

**[Screen: Stay on "How It Works" tab, scroll through diagrams]**

> "First — our architecture. We pull from 4 free Marketplace datasets: ECDC for cases, OWID for vaccinations, Google Mobility for movement patterns, and World Bank for socioeconomic indicators."

**[Point to pipeline diagram]**

> "These get joined through 10+ SQL queries using window functions, CTEs, and a custom ISO country code mapping we built in pure SQL — 150 countries mapped from 2-letter to 3-letter codes to do a cross-database JOIN."

**[Scroll to scoring table]**

> "We mapped every feature to the 5 judging dimensions. Let me show you each one."

---

## COUNTRY RANKINGS (0:50 - 1:05)

**[Click: "Country Rankings" tab]**

> "97 countries ranked by total cases, color-coded by our risk tier system. Red is high risk, yellow moderate, green low. Full detail table below with CFR, cases per 100K, and trend data."

**[Hover over a bar to show tooltip]**

---

## OUTBREAK TRENDS (1:05 - 1:30)

**[Click: "Outbreak Trends" tab]**

> "7-day rolling averages computed with SQL window functions. You can see wave patterns across countries. Below that, we compute doubling time using a logarithmic formula, and track case fatality rate over time."

---

## VACCINATION + MOBILITY (1:30 - 1:55)

**[Click: "Vaccination Tracker" tab]**

> "Vaccination data JOINed with case data — the dual-axis chart shows cases declining as vaccination rises."

**[Click: "Mobility Impact" tab]**

> "Google Mobility JOINed with cases — workplace and retail mobility plotted against surges. This is an INNER JOIN across two Marketplace datasets."

---

## ML FORECAST (1:55 - 2:35) — KEY DEMO

**[Click: "ML Forecast" tab]**

> "Now the ML piece — Snowflake's native Forecasting API."

**[Select "United States", click "Generate Forecast"]**

> "It trains a time-series model on historical data and produces a 30-day projection. Blue is actual, red dashed is forecast. We also get MAPE evaluation metrics automatically."

**[Wait for chart, point to metrics]**

> "This runs entirely inside Snowflake — no external Python, no notebooks."

---

## SOCIOECONOMIC CONTEXT (2:35 - 3:05) — DIFFERENTIATOR

**[Click: "Socioeconomic Context" tab]**

> "Our differentiator — we integrated World Bank development indicators as a 4th dataset. The challenge: COVID data uses 2-letter ISO codes, World Bank uses 3-letter. So we built a 150-country mapping in pure SQL for a cross-database JOIN."

**[Point to scatter plot]**

> "Health spending per capita versus fatality rate — 91 countries matched. And poverty rate versus reported cases — the underreporting hypothesis. Poor countries appear to have fewer cases because they test less."

---

## DATA QUALITY + AI (3:05 - 3:40)

**[Click: "Data Quality" tab]**

> "The problem statement asks for a fairness disclosure. We quantified reporting completeness for every country and flagged geographic bias — testing asymmetry, vaccination data gaps, mobility coverage limits."

**[Click: "AI Insights" tab, select "Executive Summary", click "Generate"]**

> "Cortex AI generates policy-ready briefings. Six analysis types — this Executive Summary passes real data points to mistral-large and produces actionable recommendations."

---

## CLOSE (3:40 - 4:00)

**[Scroll back to "How It Works" tab]**

> "To recap: 4 Marketplace datasets, 10+ advanced SQL queries with window functions and CTEs, cross-database ISO mapping, ML Forecasting API, Cortex AI, a fairness disclosure, and architecture diagrams — all running natively inside Snowflake.
>
> Thank you."

---

## PRE-RECORDING CHECKLIST

Before hitting record:
- [ ] Select countries in sidebar: United States, India, France, Brazil
- [ ] Pre-run the ML forecast for "United States" so it loads fast during demo
- [ ] Have "How It Works" tab open as starting screen
- [ ] Maximize browser window, hide bookmarks bar
- [ ] Test that AI Insights generates without errors

## KEY MOMENTS TO NAIL

1. **Architecture diagrams** (0:20) — judges can pause video here
2. **ISO mapping mention** (0:40) — technically impressive, call it out
3. **ML Forecast live demo** (2:00) — the money shot, show it working
4. **Socioeconomic scatter** (2:40) — your differentiator, 4th dataset
5. **Fairness disclosure** (3:10) — required deliverable, show you care about bias
