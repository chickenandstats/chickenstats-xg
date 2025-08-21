# Python script for scraping raw play-by-play data using the chickenstats library

from pathlib import Path

from chickenstats.chicken_nhl import Season, Scraper

from dotenv import load_dotenv

import os
import cronitor

load_dotenv()

CRONITOR_KEY = os.environ.get("CRONITOR_KEY")
CRONITOR_JOB = os.environ.get("CRONITOR_JOB")

cronitor.api_key = CRONITOR_KEY
monitor = cronitor.Monitor(CRONITOR_JOB)

monitor.ping(
    message="Starting scrape of chickenstats data...", host="macbook", state="run"
)

# Generating a list of years to scrape
years = list(range(2024, 2009, -1))

# Folder for saving the files
SAVE_FOLDER = Path("./raw")

# Iterate through the years
for year in years:
    monitor.ping(message=f"Starting scrape of {year}...", host="macbook", state="run")

    try:
        # Get game IDs using the schedule scraping function

        season = Season(year)

        sched = season.schedule()

        sched = sched.loc[sched.game_state == "OFF"].reset_index(drop=True)

        game_ids = sched.game_id.tolist()

        scraper = Scraper(game_ids)

        # Scrape the play by play for the year
        pbp = scraper.play_by_play

        # Setting filepath
        filepath = SAVE_FOLDER / f"{year}.csv"

        # Saving files
        pbp.to_csv(filepath, index=False)

        monitor.ping(
            message=f"Finishing scrape of {year}...", host="macbook", state="run"
        )

    except:
        monitor.ping(
            message=f"Scrape of {year} failed...", host="macbook", state="fail"
        )

monitor.ping(message="Finished scraping data...", host="macbook", state="complete")
