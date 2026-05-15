# Part 7 Master Plan: The Chickenstats API (Architecture & Access)

## 1. Metadata, Aesthetics, and Technical Specs
* **Primary Publication:** `chickenandstats.com` (Ghost)
* **Code Documentation:** `chickenstats-api` (FastAPI Swagger Docs / GitHub Pages)
* **Hero Image:** "The Engine Room." A dark-mode render of the actual pipeline: a Mac Studio at the center, connected by glowing lines to a PostgreSQL cylinder on the left (data at rest), a Cloudflare globe at the top (auth + tunnel), and a FastAPI endpoint schema on the right. Tier colors (Cyan, Magenta, Gold) stripe the outgoing data lines.
* **Architecture Diagram:** The full production pipeline: NHL API → `chickenstats` scraper (4 AM daily) / live scraper (every minute) → PostgreSQL (`pbpcs` table) → offline cascade scoring (`compute_pred_goal.py`) → `pred_goal` column in `pbpcs` → FastAPI → Cloudflare Tunnel → End User.
* **Color Standard:** Developer-focused. Syntax highlighting for code blocks. Tier colors (Cyan/Magenta/Gold/Gray) appear in the pipeline diagram and tier-access table.
* **Publishing gate:** This post cannot be published until the API is live, the full historical backfill is complete, and at least one full season of `pred_goal` data is in production. Publish after Parts 1–6 are ready, never before the API is battle-tested.

---

## 2. Narrative Structure (Ghost Post)

### Section 1: From Theory to Production

* **The Hook:** "We didn't build this cascade to write a blog post. We built it to be queried." Six posts proved the architecture. This one is about using it.
* **The gap in public analytics:** Most public xG models release static files — CSVs or parquets updated once a season, or at best once a day. Chickenstats is different: the pre-computed `pred_goal` values are stored in a live PostgreSQL database, refreshed every morning after the nightly scrape, and served through a typed FastAPI endpoint with Auth0 authentication. During active games, a live scraper polls the NHL API every minute and pushes updated play-by-play (with `pred_goal`) to a separate live table.

### Section 2: The Honest Architecture

*This section must be technically accurate. Do not oversell.*

* **The Scraper:** The open-source `chickenstats` Python package handles NHL API ingestion. It runs nightly at 4 AM, scraping the previous day's play-by-play, skater stats, line combinations, and team stats into PostgreSQL. The scraper is MIT-licensed and publicly available regardless of API tier.
* **The Database:** PostgreSQL on a Mac Studio homelab (Apple M-series), accessed via Cloudflare Tunnel. Shot events live in the `pbpcs` table alongside `base_xg` (computed client-side by the `chickenstats` package) and `pred_goal` (computed offline by the cascade pipeline).
* **Offline Cascade Scoring:** `pred_goal` is **not** computed at request time. After the nightly scrape completes, `scripts/compute_pred_goal.py` loads the five cascade models (one per strength state), joins rolling stats and RAPM from the DB, runs the three-tier cascade on every unscored event, and bulk-writes the results back to `pbpcs`. The API reads pre-computed values — it does not run inference per request. This is the right architectural choice: inference once, read many times.
* **FastAPI:** Chosen for auto-generated Swagger documentation at `/docs`, async request handling, and Pydantic schema validation. If you query a shot, you are guaranteed a float for `base_xg` and `pred_goal` — not a null, not a string, not a missing key.
* **Live Games:** A separate live scraper polls the NHL API every minute for in-progress game data. Live play-by-play (with `pred_goal` pre-computed for each new event) is written to `live_pbp` and served through `/live/play_by_play`. Typical latency from event occurrence to API availability: under 90 seconds.
* **Auth:** Three paths to authenticate:
    * **Auth0 Bearer JWT** — for external consumers. Obtain via `POST /api/v1/login/auth0-token` (resource owner password grant). The token carries a custom `tier` claim.
    * **CF Service Token** — for programmatic access. Set `CF-Access-Client-Id` and `CF-Access-Client-Secret` headers. Credentials rotatable via `POST /users/me/programmatic-credentials/rotate`.
    * **Session JWT** — for the web dashboard. Not relevant to API consumers.

### Section 3: The Tier System

*The tier structure is the product. Explain it exactly.*

All three cascade tiers — `base_xg`, `context_xg`, `pred_goal` — are computed and stored. `pred_goal` is **never stripped from responses** regardless of subscription tier. This is the upgrade hook: free users see the full cascade value for their data, recognize it is differentiated signal unavailable anywhere else, and have a concrete reason to upgrade for more seasons.

| Tier | Access | Live Games |
|:---|:---|:---|
| **Free** (default) | 2016–17 playoffs only | ❌ |
| **Premium** (Ghost subscription, lower) | 2019–20 → last completed season | ✅ one team |
| **Pro** (Ghost subscription, higher) | All seasons and sessions | ✅ all teams |
| **Contributor** (manual) | Pro + MLflow / Optuna / MinIO read | ✅ all teams |

Ghost fires `member.created`, `member.updated`, and `member.deleted` webhooks to the API, which syncs the subscriber's tier to Auth0 `app_metadata` in real time. JWT claims reflect the current tier within seconds of a Ghost subscription change.

### Section 4: What You Can Query

Three endpoint groups cover historical data, cascade predictions, and live games:

**`/chicken_nhl/play_by_play`** — Full paginated play-by-play with all `chickenstats` columns, including `base_xg` and `pred_goal`. Use this for season-long spatial analysis, zone entry modeling, or building your own shot maps. Tier-filtered by season.

**`/inference/pred_goal`** — The cascade inference endpoint. Returns pre-computed `base_xg`, `context_xg`, and `pred_goal` for shot, miss, and goal events, joined with player and game identifiers. Same tier filtering as PBP.

**`/live/play_by_play`** — Live PBP for in-progress games with `pred_goal` populated as events arrive. Premium users are restricted to one self-selected team; Pro and above receive all teams.

**`/chicken_nhl/stats/season`** — Season-aggregated skater stats including `ibase_xg` (individual base xG) and `ipred_goal` (individual pred goal), the building blocks for the Archetype Matrix from Part 6.

### Section 5: "Hello, World" — The Quickstart

**Show three complete, copy-pasteable working examples. Not sketches.**

**Example 1 — Historical PBP with cascade xG:**
```python
import requests

response = requests.get(
    "https://api.chickenstats.com/api/v1/chicken_nhl/play_by_play",
    params={
        "season": 20242025,
        "session": "R",
        "event": "GOAL",
        "limit": 100,
    },
    headers={"Authorization": "Bearer YOUR_AUTH0_JWT"}
)

shots = response.json()
# Each record includes: game_id, event_idx, player_1, team, strength_state,
# base_xg (float), pred_goal (float), event_distance, event_angle, shot_type, ...
```

**Example 2 — Cascade inference endpoint:**
```python
response = requests.get(
    "https://api.chickenstats.com/api/v1/inference/pred_goal",
    params={"season": 20242025, "strength_state": "5v5"},
    headers={"Authorization": "Bearer YOUR_AUTH0_JWT"}
)

predictions = response.json()
# Each record: game_id, event_idx, player_name, strength_state,
# base_xg, context_xg, pred_goal (all floats, always present)
```

**Example 3 — CF service token (programmatic access):**
```python
response = requests.get(
    "https://api.chickenstats.com/api/v1/inference/pred_goal",
    params={"season": 20242025},
    headers={
        "CF-Access-Client-Id": "your-client-id",
        "CF-Access-Client-Secret": "your-client-secret",
    }
)
```

Show the exact JSON response for one shot event. Keep it clean — one record, all relevant fields, no truncation. This is the moment a developer decides whether to integrate or move on.

### Section 6: What's Next

Concrete near-term items only. No speculative roadmap.

* **Premium live team filter:** `User.favorite_team` field and the corresponding live-endpoint tier gate are designed but not yet deployed. Premium subscribers will be able to follow one team in real-time with per-event `pred_goal` updates.
* **Season stats endpoints as Archetype Matrix input:** `/chicken_nhl/stats/season` already returns `ibase_xg` and `ipred_goal`. A future convenience endpoint at `/players/{player_id}/season` will serve the per-player Talent Delta aggregation directly, matching the scouting matrix from Part 6.
* **The open-source commitment:** The `chickenstats` scraper and this model training repository remain MIT-licensed and public. The API key gates access to the pre-computed prediction database — not the methodology. Every result in this series is reproducible from the open-source code.

---

## 3. Required Analyses, Charts, and Visuals

**1. The Infrastructure Flowchart**
* *What it is:* Accurate diagram of the actual production pipeline: NHL API → `chickenstats` scraper (4 AM cron via ofelia) → PostgreSQL `pbpcs` → `compute_pred_goal.py` → `pred_goal` column → FastAPI → Cloudflare Tunnel → Auth0-verified end user. Parallel path for live: NHL API → live scraper (1-min cron) → `live_pbp` → `/live/play_by_play`.
* *Color:* Tier colors on the outgoing data lines (Cyan for base_xg, Magenta for context_xg, Gold for pred_goal).

**2. Swagger UI Screenshot**
* *What it is:* Screenshot of the `/docs` page showing the four endpoint groups (`/chicken_nhl`, `/inference`, `/live`, `/users`), their parameters, and example response schemas.

**3. Tier Access Table**
* *What it is:* The clean version of the tier/data-access matrix from Section 3, formatted for the blog post audience. Emphasize that `pred_goal` is visible at all tiers.

**4. JSON Response Snippet**
* *What it is:* One complete, cleanly formatted JSON record from `/inference/pred_goal` — all three cascade outputs, the player identifiers, the game context, and the strength state.

---

## 4. Companion Code Specs

*The companion code lives in the `chickenstats-api` repository. Highlight four specific files:*

**`scripts/compute_pred_goal.py` — Offline cascade scoring:**
```python
# Not loaded at API startup — runs after nightly scrape
booster = xgb.Booster()
booster.load_model(models_dir / f"{variant}_base.ubj")
calibrator = joblib.load(models_dir / f"{variant}_calibrator.joblib")

# Batch scoring: read unscored events, join rolling stats + RAPM, score, write back
events = read_unscored_events(db, variant=variant)
dmatrix = xgb.DMatrix(features, base_margin=sc.logit(events["context_xg"]))
raw_margins = booster.predict(dmatrix, output_margin=True)
pred_goal = calibrator.predict(sc.expit(raw_margins))
bulk_update_pred_goal(db, event_ids, pred_goal)
```
This is intentionally not a runtime inference path. Scoring once and reading many times is the right architecture for pre-play-by-play data.

**`app/api/routes/inference.py` — FastAPI routing:**
The `/inference/pred_goal` endpoint performs a parameterized SQL query against `pbpcs`, returning pre-computed values. No model is loaded at API startup.

**`app/api/schemas.py` — Pydantic models:**
The `PredGoalResponse` schema guarantees `base_xg: float`, `context_xg: float`, and `pred_goal: float` — never null, never string, never missing. Pydantic validation runs before the response is serialized.

**`app/scraper/scrape.py` + `app/live_scraper/live_scrape.py` — Data ingestion:**
Two separate scrapers. The nightly scraper runs at 4 AM and populates `pbpcs` with base_xg (computed by the `chickenstats` package client-side during ingestion). The live scraper polls every minute and writes to `live_pbp` with `pred_goal` populated as events are processed.