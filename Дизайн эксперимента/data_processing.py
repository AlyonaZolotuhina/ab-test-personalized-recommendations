import numpy as np
import pandas as pd

EVENT_TYPES = ["view_banner", "click_banner", "start_watch", "finish_watch", "add_to_favorites"]

def build_daily_user_funnel(events: pd.DataFrame) -> pd.DataFrame:
    required = {"dt", "client_id", "event_type", "watch_time_min"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = events.copy()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["date"] = df["dt"].dt.date
    counts = df.pivot_table(index=["date", "client_id"], columns="event_type",
                            values="dt", aggfunc="count", fill_value=0).reset_index()
    counts.columns.name = None
    for event in EVENT_TYPES:
        if event not in counts:
            counts[event] = 0
    watch = (df[df["event_type"].eq("finish_watch")]
             .groupby(["date", "client_id"])["watch_time_min"]
             .sum().rename("watch_time").reset_index())
    out = counts.merge(watch, on=["date", "client_id"], how="left")
    out["watch_time"] = out["watch_time"].fillna(0)
    out = out.rename(columns={"date":"dt", "client_id":"user_id",
                              "view_banner":"views", "click_banner":"clicks",
                              "add_to_favorites":"adds_to_fav"})
    out["ctr"] = np.where(out["views"] > 0, out["clicks"] / out["views"], np.nan)
    out["avg_watch_time"] = np.where(out["clicks"] > 0, out["watch_time"] / out["clicks"], np.nan)
    out["cr_fav"] = np.where(out["clicks"] > 0, out["adds_to_fav"] / out["clicks"], np.nan)
    out["arpu"] = out["watch_time"]
    return out
