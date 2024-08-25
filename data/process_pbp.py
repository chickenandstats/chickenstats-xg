# Process data for ML experiments

import pandas as pd

import numpy as np

from pathlib import Path

from tqdm.auto import tqdm

from datetime import datetime

from chickenstats.chicken_nhl.validation import XGSchema


def prep_data(data, strengths):
    """
    Function for prepping play-by-play data for xG experiments

    Data are play-by-play data from the chickenstats function.

    Strengths can be: even, powerplay, shorthanded, empty_for, empty_against
    """

    df = data.copy()

    events = [
        "SHOT",
        "FAC",
        "HIT",
        "BLOCK",
        "MISS",
        "GIVE",
        "TAKE",
        # "PENL",
        "GOAL",
    ]

    conds = np.logical_and.reduce(
        [
            df.event.isin(events),
            df.strength_state != "1v0",
            pd.notnull(df.coords_x),
            pd.notnull(df.coords_y),
        ]
    )

    df = df.loc[conds]

    conds = np.logical_and.reduce(
        [
            df.season == df.season.shift(1),
            df.game_id == df.game_id.shift(1),
            df.period == df.period.shift(1),
        ]
    )
    df["seconds_since_last"] = np.where(
        conds, df.game_seconds - df.game_seconds.shift(1), np.nan
    )
    df["event_type_last"] = np.where(conds, df.event.shift(1), np.nan)
    df["event_team_last"] = np.where(conds, df.event_team.shift(1), np.nan)
    df["event_strength_last"] = np.where(conds, df.strength_state.shift(1), np.nan)
    df["coords_x_last"] = np.where(conds, df.coords_x.shift(1), np.nan)
    df["coords_y_last"] = np.where(conds, df.coords_y.shift(1), np.nan)
    df["zone_last"] = np.where(conds, df.zone.shift(1), np.nan)

    df["same_team_last"] = np.where(np.equal(df.event_team, df.event_team_last), 1, 0)

    df["distance_from_last"] = (
                                       (df.coords_x - df.coords_x_last) ** 2 + (df.coords_y - df.coords_y_last) ** 2
                               ) ** (1 / 2)

    last_is_shot = np.equal(df.event_type_last, "SHOT")
    last_is_miss = np.equal(df.event_type_last, "MISS")
    last_is_block = np.equal(df.event_type_last, "BLOCK")
    last_is_give = np.equal(df.event_type_last, "GIVE")
    last_is_take = np.equal(df.event_type_last, "TAKE")
    last_is_hit = np.equal(df.event_type_last, "HIT")
    last_is_fac = np.equal(df.event_type_last, "FAC")

    same_team_as_last = np.equal(df.same_team_last, 1)
    not_same_team_as_last = np.equal(df.same_team_last, 0)

    df["prior_shot_same"] = np.where((last_is_shot & same_team_as_last), 1, 0)
    df["prior_miss_same"] = np.where((last_is_miss & same_team_as_last), 1, 0)
    df["prior_block_same"] = np.where((last_is_block & same_team_as_last), 1, 0)
    df["prior_give_same"] = np.where((last_is_give & same_team_as_last), 1, 0)
    df["prior_take_same"] = np.where((last_is_take & same_team_as_last), 1, 0)
    df["prior_hit_same"] = np.where((last_is_hit & same_team_as_last), 1, 0)

    df["prior_shot_opp"] = np.where((last_is_shot & not_same_team_as_last), 1, 0)
    df["prior_miss_opp"] = np.where((last_is_miss & not_same_team_as_last), 1, 0)
    df["prior_block_opp"] = np.where((last_is_block & not_same_team_as_last), 1, 0)
    df["prior_give_opp"] = np.where((last_is_give & not_same_team_as_last), 1, 0)
    df["prior_take_opp"] = np.where((last_is_take & not_same_team_as_last), 1, 0)
    df["prior_hit_opp"] = np.where((last_is_hit & not_same_team_as_last), 1, 0)

    df["prior_face"] = np.where(last_is_fac, 1, 0)

    shot_types = pd.get_dummies(df.shot_type, dtype=int)

    shot_types = shot_types.rename(
        columns={
            x: x.lower().replace("-", "_").replace(" ", "_") for x in shot_types.columns
        }
    )

    df = df.copy().merge(shot_types, left_index=True, right_index=True, how="outer")

    conds = [df.score_diff > 4, df.score_diff < -4]

    values = [4, -4]

    df.score_diff = np.select(conds, values, df.score_diff)

    conds = [
        df.player_1_position.isin(["F", "L", "R", "C"]),
        df.player_1_position == "D",
        df.player_1_position == "G",
    ]

    values = ["F", "D", "G"]

    df["position_group"] = np.select(conds, values)

    position_dummies = pd.get_dummies(df.position_group, dtype=int)

    new_cols = {x: f"position_{x.lower()}" for x in values}

    position_dummies = position_dummies.rename(columns=new_cols)

    df = df.merge(position_dummies, left_index=True, right_index=True)

    conds = [
        np.logical_and.reduce(
            [
                df.event.isin(
                    [
                        "GOAL",
                        "SHOT",
                        "BLOCK",
                        "MISS",
                    ]
                ),
                df.event_type_last.isin(["SHOT", "MISS"]),
                df.event_team_last == df.event_team,
                df.game_id == df.game_id.shift(1),
                df.period == df.period.shift(1),
                df.seconds_since_last <= 3,
            ]
        ),
        np.logical_and.reduce(
            [
                df.event.isin(
                    [
                        "GOAL",
                        "SHOT",
                        "BLOCK",
                        "MISS",
                    ]
                ),
                df.event_type_last == "BLOCK",
                df.event_team_last == df.opp_team,
                df.game_id == df.game_id.shift(1),
                df.period == df.period.shift(1),
                df.seconds_since_last <= 3,
            ]
        ),
    ]

    values = [1, 1]

    df["is_rebound"] = np.select(conds, values, 0)

    conds = np.logical_and.reduce(
        [
            df.event.isin(
                [
                    "GOAL",
                    "SHOT",
                    "BLOCK",
                    "MISS",
                ]
            ),
            df.seconds_since_last <= 4,
            df.zone_last == "NEU",
            df.game_id == df.game_id.shift(1),
            df.period == df.period.shift(1),
            df.event != "FAC",
        ]
    )

    df["rush_attempt"] = np.where(conds, 1, 0)

    cat_cols = [
        "strength_state",
        "position_group",
        "event_type_last",
    ]

    for col in cat_cols:
        dummies = pd.get_dummies(df[col], dtype=int)

        new_cols = {x: f"{col}_{x}" for x in dummies.columns}

        dummies = dummies.rename(columns=new_cols)

        df = df.copy().merge(dummies, left_index=True, right_index=True)

        # df = df.drop(col, axis=1)

    if strengths.lower() == "even":
        strengths_list = ["5v5", "4v4", "3v3"]

        conds = np.logical_and.reduce(
            [
                df.event.isin(["GOAL", "SHOT", "MISS"]),
                df.strength_state.isin(strengths_list),
            ]
        )

        df = df.loc[conds]

        drop_cols = [
            x
            for x in df.columns
            if "strength_state_" in x
               and x not in [f"strength_state_{x}" for x in strengths_list]
        ]

        df = df.drop(drop_cols, axis=1, errors="ignore")

    if strengths.lower() == "powerplay" or strengths.lower() == "pp":
        strengths_list = ["5v4", "4v3", "5v3"]
        conds = np.logical_and.reduce(
            [
                df.event.isin(["GOAL", "SHOT", "MISS"]),
                df.strength_state.isin(strengths_list),
            ]
        )

        df = df.loc[conds]

        drop_cols = [
            x
            for x in df.columns
            if "strength_state_" in x
               and x not in [f"strength_state_{x}" for x in strengths_list]
        ]

        df = df.drop(drop_cols, axis=1, errors="ignore")

    if strengths.lower() == "shorthanded" or strengths.lower() == "ss":
        strengths_list = ["4v5", "3v4", "3v5"]
        conds = np.logical_and.reduce(
            [
                df.event.isin(["GOAL", "SHOT", "MISS"]),
                df.strength_state.isin(strengths_list),
            ]
        )

        df = df.loc[conds]

        drop_cols = [
            x
            for x in df.columns
            if "strength_state_" in x
               and x not in [f"strength_state_{x}" for x in strengths_list]
        ]

        df = df.drop(drop_cols, axis=1, errors="ignore")

    if strengths.lower() == "empty_for":
        strengths_list = ["Ev5", "Ev4", "Ev3"]
        conds = np.logical_and.reduce(
            [
                df.event.isin(["GOAL", "SHOT", "MISS"]),
                df.strength_state.isin(strengths_list),
            ]
        )

        df = df.loc[conds]

        drop_cols = [
            x
            for x in df.columns
            if "strength_state_" in x
               and x not in [f"strength_state_{x}" for x in strengths_list]
        ]  # + [x for x in df.columns if "own_goalie" in x]

        df = df.drop(drop_cols, axis=1, errors="ignore")

    if strengths.lower() == "empty_against":
        strengths_list = ["5vE", "4vE", "3vE"]
        conds = np.logical_and.reduce(
            [
                df.event.isin(["GOAL", "SHOT", "MISS"]),
                df.strength_state.isin(strengths_list),
            ]
        )

        df = df.loc[conds]

        drop_cols = [
            x
            for x in df.columns
            if "strength_state_" in x
               and x not in [f"strength_state_{x}" for x in strengths_list]
        ]  # + [x for x in df.columns if "opp_goalie" in x]

        df = df.drop(drop_cols, axis=1, errors="ignore")

    df = df.drop(cat_cols, axis=1, errors="ignore")

    # cols = [
    #     "backhand",
    #     "bat",
    #     "between_legs",
    #     "cradle",
    #     "deflected",
    #     "poke",
    #     "slap",
    #     "snap",
    #     "tip_in",
    #     "wrap_around",
    #     "wrist",
    # ]
    #
    # for col in cols:
    #     if col not in df.columns:
    #         df[col] = 0

    cols = [
        "season",
        "goal",
        "period",
        "period_seconds",
        "score_diff",
        "danger",
        "high_danger",
        "position_f",
        "position_d",
        "position_g",
        "event_distance",
        "event_angle",
        "forwards_percent",
        "forwards_count",
        "opp_forwards_percent",
        "opp_forwards_count",
        "is_rebound",
        "rush_attempt",
        "is_home",
        "seconds_since_last",
        "event_type_last",
        "distance_from_last",
        "prior_shot_same",
        "prior_miss_same",
        "prior_block_same",
        "prior_give_same",
        "prior_take_same",
        "prior_hit_same",
        "prior_shot_opp",
        "prior_miss_opp",
        "prior_block_opp",
        "prior_give_opp",
        "prior_take_opp",
        "prior_hit_opp",
        "prior_face",
        "backhand",
        "bat",
        "between_legs",
        "cradle",
        "deflected",
        "poke",
        "slap",
        "snap",
        "tip_in",
        "wrap_around",
        "wrist",
        "strength_state_3v3",
        "strength_state_4v4",
        "strength_state_5v5",
        "strength_state_3v4",
        "strength_state_3v5",
        "strength_state_4v5",
        "strength_state_4v3",
        "strength_state_5v3",
        "strength_state_5v4",
        "strength_state_Ev3",
        "strength_state_Ev4",
        "strength_state_Ev5",
        "strength_state_3vE",
        "strength_state_4vE",
        "strength_state_5vE",
    ]

    cols = [x for x in cols if x in df.columns]

    # df = df[cols].copy()

    df = XGSchema.validate(df[[x for x in XGSchema.dtypes.keys() if x in df.columns]])

    # if strengths.lower() == "empty_for":
    #     drop_cols = [x for x in df.columns if "own_goalie" in x]
    #
    #     df = df.drop(drop_cols, axis=1)
    #
    # if strengths.lower() == "empty_against":
    #     drop_cols = [x for x in df.columns if "opp_goalie" in x]
    #
    #     df = df.drop(drop_cols, axis=1)

    # for col in df.columns:
    #     df[col] = pd.to_numeric(df[col])
    #
    # conds = [df.goal is True, df.goal is False]
    #
    # values = [1, 0]
    #
    # df.goal = np.select(conds, values, df.goal)
    #
    # df = df.fillna(0)

    return df


years = list(range(2023, 2009, -1))

# List to collect the dataframes
evens = []
powerplays = []
shorthandeds = []
empty_fors = []
empty_againsts = []

# Making a progress bar
pbar = tqdm(iterable=years, desc="PROCESSING PLAY-BY-PLAY DATA")

# Iterate through the years
for year in pbar:
    # Setting filepath
    filepath = Path(f"./raw/{year}.csv")

    # Saving files
    pbp = pd.read_csv(filepath, low_memory=False)

    even = prep_data(pbp, "even")

    powerplay = prep_data(pbp, "powerplay")

    shorthanded = prep_data(pbp, "shorthanded")

    empty_for = prep_data(pbp, "empty_for")

    empty_against = prep_data(pbp, "empty_against")

    if year != 2023:
        evens.append(even)

        powerplays.append(powerplay)

        shorthandeds.append(shorthanded)

        empty_fors.append(empty_for)

        empty_againsts.append(empty_against)

    else:
        even.to_csv(Path("./hold_out/even_strength.csv"))

        powerplay.to_csv(Path("./hold_out/powerplay.csv"))

        shorthanded.to_csv(Path("./hold_out/shorthanded.csv"))

        empty_for.to_csv(Path("./hold_out/empty_for.csv"))

        empty_against.to_csv(Path("./hold_out/empty_against.csv"))

    if year == years[-1]:
        message = "FINISHED PROCESSING DATA"

    else:
        message = f"FINISHED PROCESSING {year}"

    pbar.set_description(message)

    ## Adding current time to the progress bar

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")

    postfix_str = f"{current_time}"

    pbar.set_postfix_str(postfix_str)

even = pd.concat(evens, ignore_index=True)

powerplay = pd.concat(powerplays, ignore_index=True)

shorthanded = pd.concat(shorthandeds, ignore_index=True)

empty_for = pd.concat(empty_fors, ignore_index=True)

empty_against = pd.concat(empty_againsts, ignore_index=True)

even.to_csv(Path("./processed/even_strength.csv"))

powerplay.to_csv(Path("./processed/powerplay.csv"))

shorthanded.to_csv(Path("./processed/shorthanded.csv"))

empty_for.to_csv(Path("./processed/empty_for.csv"))

empty_against.to_csv(Path("./processed/empty_against.csv"))
