"""
Shared engagement-summary logic for AgentT.
Used by both analytics_summary.py (CLI) and the /admin/analytics endpoint
in main.py, so there's a single source of truth for how the numbers are
computed.
"""

import datetime
from collections import defaultdict


def compute_engagement_summary(rows, roster):
    """
    rows: list of dicts, one per sheet row (as returned by sheet.get_all_records())
    roster: list/set of lowercase emails expected to have participated

    Returns a dict:
        {
            "participants": [ {email, messages, first_seen, last_seen,
                                duration_minutes, interacted}, ... ],
            "unexpected_emails": [ {email, messages}, ... ],
            "roster_size": int,
            "interacted_count": int,
        }
    """
    by_email = defaultdict(list)
    for row in rows:
        email = str(row.get("Participant Email", "")).strip().lower()
        if email:
            by_email[email].append(row)

    roster = sorted({e.strip().lower() for e in roster if e.strip()})
    participants = []

    for email in roster:
        entries = by_email.get(email, [])
        if not entries:
            participants.append({
                "email": email, "messages": 0, "first_seen": None,
                "last_seen": None, "duration_minutes": None, "interacted": False,
            })
            continue

        timestamps = []
        for e in entries:
            ts_raw = str(e.get("Timestamp", "")).strip()
            try:
                timestamps.append(datetime.datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue

        if not timestamps:
            participants.append({
                "email": email, "messages": len(entries), "first_seen": None,
                "last_seen": None, "duration_minutes": None, "interacted": True,
            })
            continue

        first_ts, last_ts = min(timestamps), max(timestamps)
        duration_minutes = round((last_ts - first_ts).total_seconds() / 60, 1)

        participants.append({
            "email": email,
            "messages": len(entries),
            "first_seen": str(first_ts),
            "last_seen": str(last_ts),
            "duration_minutes": duration_minutes,
            "interacted": True,
        })

    unexpected = set(by_email.keys()) - set(roster)
    unexpected_emails = [
        {"email": email, "messages": len(by_email[email])}
        for email in sorted(unexpected)
    ]

    return {
        "participants": participants,
        "unexpected_emails": unexpected_emails,
        "roster_size": len(roster),
        "interacted_count": sum(1 for p in participants if p["interacted"]),
    }
