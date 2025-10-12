import os, sys, json
from datetime import datetime, timedelta, timezone
import requests
import matplotlib.pyplot as plt
from pathlib import Path

OWNER = os.environ.get("OWNER")
REPO  = os.environ.get("REPO")
TOKEN = os.environ.get("GH_TOKEN")

# Fallback: derive from GITHUB_REPOSITORY ("owner/repo")
if (not OWNER or not REPO) and os.environ.get("GITHUB_REPOSITORY"):
    owner_repo = os.environ["GITHUB_REPOSITORY"]
    if "/" in owner_repo:
        OWNER = OWNER or owner_repo.split("/", 1)[0]
        REPO  = REPO  or owner_repo.split("/", 1)[1]

if not all([OWNER, REPO, TOKEN]):
    print("Missing OWNER/REPO/GH_TOKEN env vars"); sys.exit(1)

API = "https://api.github.com"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
STAR_HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github.v3.star+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

METRICS_DIR = Path("metrics")
DATA_DIR    = Path("metrics_data")  # local persistence so we can go beyond 14d for traffic
METRICS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

HIST_FILE = DATA_DIR / "traffic_history.json"  # { "YYYY-MM-DD": {"clones": n, "views": n} }

def _get(url, headers=HEADERS, params=None):
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _pages(url, headers, params=None, max_pages=10):
    page = 1
    while True:
        p = params.copy() if params else {}
        p.update({"per_page": 100, "page": page})
        r = requests.get(url, headers=headers, params=p, timeout=30)
        if r.status_code == 204:
            break
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        yield data
        if len(data) < 100 or page >= max_pages:
            break
        page += 1

def iso_date(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()

def daterange(days):
    today = datetime.now(timezone.utc).date()
    return [today - timedelta(days=i) for i in range(days-1, -1, -1)]

def load_history():
    if HIST_FILE.exists():
        with open(HIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_history(hist):
    # keep ~90 days only
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=90)
    pruned = {k: v for k, v in hist.items() if datetime.fromisoformat(k).date() >= cutoff}
    with open(HIST_FILE, "w", encoding="utf-8") as f:
        json.dump(pruned, f, indent=2, sort_keys=True)

def fetch_traffic():
    clones = _get(f"{API}/repos/{OWNER}/{REPO}/traffic/clones")
    views  = _get(f"{API}/repos/{OWNER}/{REPO}/traffic/views")

    # build day->count dicts for last 14 days
    clones_by_day = {}
    for p in clones.get("clones", []):
        d = iso_date(p["timestamp"]).isoformat()
        clones_by_day[d] = clones_by_day.get(d, 0) + int(p.get("count", 0))

    views_by_day = {}
    for p in views.get("views", []):
        d = iso_date(p["timestamp"]).isoformat()
        views_by_day[d] = views_by_day.get(d, 0) + int(p.get("count", 0))

    return clones_by_day, views_by_day

def update_traffic_history():
    hist = load_history()
    clones14, views14 = fetch_traffic()
    # merge into history (latest 14 days override existing values)
    for d, v in clones14.items():
        rec = hist.get(d, {})
        rec["clones"] = v
        hist[d] = rec
    for d, v in views14.items():
        rec = hist.get(d, {})
        rec["views"] = v
        hist[d] = rec
    save_history(hist)
    return hist

def fetch_stars_per_day(days=28):
    starred_at = []
    for page in _pages(f"{API}/repos/{OWNER}/{REPO}/stargazers", STAR_HEADERS):
        for s in page:
            if "starred_at" in s:
                starred_at.append(iso_date(s["starred_at"]))
    # count per day within window
    window = set(daterange(days))
    counts = {d: 0 for d in window}
    for d in starred_at:
        if d in window:
            counts[d] += 1
    return counts

def fetch_commits_per_day(days=28):
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    commits = []
    page = 1
    while True:
        payload = _get(f"{API}/repos/{OWNER}/{REPO}/commits",
                       params={"since": since, "per_page": 100, "page": page})
        if not payload:
            break
        commits.extend(payload)
        if len(payload) < 100 or page >= 10:
            break
        page += 1
    by_day = {}
    for c in commits:
        try:
            ts = c["commit"]["author"]["date"]
        except Exception:
            continue
        d = iso_date(ts)
        by_day[d] = by_day.get(d, 0) + 1
    # ensure all days present
    window = daterange(days)
    return {d: by_day.get(d, 0) for d in window}

def series_from_dict(window_dates, dct, key=None):
    if key is None:
        return [int(dct.get(d, 0)) for d in window_dates]
    # for traffic history dict where value is {"clones":x,"views":y}
    return [int(dct.get(d.isoformat(), {}).get(key, 0)) for d in window_dates]

def plot_dashboard(window_days=28, outfile="metrics/activity_4w.png"):
    # update/get traffic history
    hist = update_traffic_history()  # merges last 14d into persistent store
    window = daterange(window_days)

    # build series
    clones = series_from_dict(window, hist, "clones")
    views  = series_from_dict(window, hist, "views")
    commits = list(fetch_commits_per_day(window_days).values())
    stars   = list(fetch_stars_per_day(window_days).values())

    # draw single chart in xkcd style (hand-drawn look like your screenshot)
    with plt.xkcd():
        plt.figure(figsize=(10, 4))
        plt.plot(window, clones, label="clones/day")
        plt.plot(window, views,  label="views/day")
        plt.plot(window, commits, label="commits/day")
        plt.plot(window, stars,   label="stars/day")
        plt.title(f"{OWNER}/{REPO} — Activity (last 4 weeks)")
        plt.xlabel("Date")
        plt.ylabel("Daily count")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        plt.legend(loc="upper left", fontsize=8)
        plt.savefig(outfile, dpi=160)
        plt.close()

if __name__ == "__main__":
    plot_dashboard()
    print("Wrote metrics/activity_4w.png and updated metrics_data/traffic_history.json")
