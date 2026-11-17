"""Ball possession, frame by frame, from the automatic tracking and the Smart Tagging.

    python solution/possession.py --name <team_name>

Writes evaluation/results/<team_name>.csv (199,319 rows, frame,label). numpy only. Deterministic:
the same data always gives the same file.

How it works, in four steps:

  1. Read the tracking once and cache it as arrays (positions in metres, speeds in m/s).
  2. Per frame, measure two kinds of evidence:
       - WHO: how much closer the nearest home player is to the ball than the nearest away player,
         counted only when someone is close enough to be playing it. No ball, no evidence.
       - DEAD OR ALIVE: the ball well outside the pitch, the tagged restarts (a restart ends a
         stoppage), and how fast the players are moving. The speed part is a small logistic
         model trained on pseudo-labels the code picks itself (ball clearly out = dead, ball at a
         player's feet in open play = alive). No person labels anything.
  3. Decode the whole half at once with Viterbi over three states (H, A, D). Changing state has a
     fixed cost, so a brief touch or a moment of occlusion does not create a new possession, and
     each change lands where the evidence turns. Then, while the ball is in view and loose, a
     change of team waits for the first frame the new team is at the ball.
  4. Write D before each half's kick-off, X outside the graded halves.

All constants are set from football reasoning, written next to each one. None was chosen by
scoring against the public five minutes.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
CACHE = os.path.join(ROOT, "data", "_cache", "atd_tactical.npz")   # data/ is gitignored
RESULTS = os.path.join(ROOT, "evaluation", "results")

# ---------------------------------------------------------------------------------------------
# Constants. Each is a football or physics judgement, not a fitted value.
# ---------------------------------------------------------------------------------------------
CONTROL_NEAR = 1.0      # m. Within this a player is playing the ball (body + tracking noise).
CONTROL_FAR = 3.0       # m. Beyond this nobody is playing it: loose ball, pass in flight.
TEAM_SCALE = 1.0        # m. How much closer one team must be for 1 unit of log-odds.
TEAM_CAP = 3.0          # max log-odds of team evidence from one frame.

OUT_TOUCH = 0.2         # m past a line: ball has (nearly) crossed it.
OUT_SURE = 1.0          # m past a line: clearly out, beyond tracking noise.
OUT_MIN_FRAMES = 5      # 0.2 s outside, so one bad ball detection is not a stoppage.

RESTART_BEFORE = 2 * FPS        # a set-piece window opens before the restart: dead just before it.
RESTART_AFTER = 1 * FPS         # a kick-off window closes just after the kick: live for 1 s after.
SHORT_WINDOW = 5 * FPS          # kick-off windows are 5 s around the kick. Every other set-piece
                                # window is a fixed 35 s, and its end says nothing about the restart.
RESTART_TEAM_SPAN = 5 * FPS     # the tagged team has the ball at the restart, not for all 35 s.
LIVE_BREAK = 1 * FPS            # a full second of a player running with the ball well inside the
                                # pitch is open play: a stoppage cannot have begun before it.
OUT_LOOKBACK = {                # how long a stoppage of each kind can last, at most
    "throw-in": 60 * FPS, "corner kick": 90 * FPS, "goal kick": 90 * FPS,
    "kick off (after goal)": 180 * FPS,
}

TEMPERATURE = 0.2       # 25 frames a second are not 25 independent looks: ~5 frames per look.
SWITCH_COST = 6.0       # H<->A. At full strength, a lasting change needs ~0.4 s of clear control
                        # by the other team, and a brief one that hands the ball back ~0.8 s.
DEAD_COST = 8.0         # alive<->D, paid on the way in and out: a stoppage needs ~1 s of the
                        # strongest dead evidence, several seconds of weak; occlusion is not one.

BALL_GAP_FILL = 12      # frames. Under 0.5 s a hidden ball has moved in a near-straight line.
STILL_SPEED = 0.3       # m/s. A ball this slow is lying still, not being dribbled.
STILL_FRAMES = 2 * FPS  # a ball lying still for 2 s in open play is very rare: a set piece is near.
FORMATION_WEIGHT = 0.5  # team shape is a hint, never as strong as the ball at a player's feet.
FORMATION_MIN_ACC = 0.6 # use team shape only if it predicts the other half this well.


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------------------------------------
# 1. Tracking -> arrays
# ---------------------------------------------------------------------------------------------
def load_tracking():
    """players (N, slots, 3) and ball (N, 3), normalised units as in the file. Cached."""
    if os.path.exists(CACHE):
        cached = np.load(CACHE)
        return cached["players"], cached["ball"]

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n_frames = source_frames("tactical")
    n_slots = len(team_of_each_track(meta_path))
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

    log("reading tracking (once; cached afterwards) ...")
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            index = int(parts[0]) - 1          # frame_id counts from 1, video frames from 0
            if not 0 <= index < n_frames:
                continue
            groups = parts[1].split(";")
            values = np.array(",".join(groups).split(","), dtype=np.float32)
            if values.size == n_slots * 3:
                players[index] = values.reshape(n_slots, 3)
            ball_values = np.array(parts[2].split(",")[:3], dtype=np.float32)
            ball[index, :ball_values.size] = ball_values

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, players=players, ball=ball)
    return players, ball


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def centred_mean(values, half_width):
    """Mean over +/- half_width frames, ignoring NaN. NaN where the window holds nothing."""
    ok = ~np.isnan(values)
    filled = np.where(ok, values, 0.0)
    total = np.concatenate(([0.0], np.cumsum(filled)))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


# ---------------------------------------------------------------------------------------------
# 2. Evidence
# ---------------------------------------------------------------------------------------------
def fill_short_gaps(values, seen, max_gap):
    """Linear interpolation across runs of missing frames no longer than max_gap."""
    out = values.copy()
    filled = np.zeros(len(values), dtype=bool)
    for start, end in runs(~seen):
        if start == 0 or end == len(values) or end - start > max_gap:
            continue
        a, b = values[start - 1], values[end]
        frac = (np.arange(start, end) - (start - 1)) / float(end - start + 1)
        out[start:end] = a + (b - a) * frac
        filled[start:end] = True
    return out, filled


def frame_features(players, ball, team_of_slot):
    px, py, speed = players[..., 0] * LENGTH, players[..., 1] * WIDTH, players[..., 2]
    bx, by = ball[:, 0] * LENGTH, ball[:, 1] * WIDTH
    seen = ~np.isnan(bx)
    # a ball hidden for under half a second has not gone far: draw it on the straight line
    bx, filled = fill_short_gaps(bx, seen, BALL_GAP_FILL)
    by, _ = fill_short_gaps(by, seen, BALL_GAP_FILL)

    distance = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    d_home = np.where(team_of_slot[None, :] == 0, distance, np.inf).min(axis=1)
    d_away = np.where(team_of_slot[None, :] == 1, distance, np.inf).min(axis=1)

    # how far past the nearest line the ball is (negative = inside)
    past_line = np.max(np.stack([-bx, bx - LENGTH, -by, by - WIDTH]), axis=0)

    # ball speed from its own positions: steadier than any single field in the file
    step = np.full(len(bx), np.nan)
    step[1:-1] = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * FPS / 2.0
    ball_speed = centred_mean(step, 2)

    with np.errstate(all="ignore"):
        median_speed = np.nanmedian(np.where(np.isnan(speed), np.nan, speed), axis=1)
        fast_speed = np.nanpercentile(speed, 80, axis=1)
    n_tracked = np.sum(~np.isnan(px), axis=1)

    # still ball: the ball has barely moved for a while, as when a free kick is being set up
    still = (seen | filled) & (ball_speed < STILL_SPEED) & (past_line < -0.5)
    long_still = np.zeros(len(still), dtype=bool)
    for start, end in runs(still):
        if end - start >= STILL_FRAMES:
            long_still[start:end] = True

    return {
        "seen": seen | filled, "real": seen, "filled": filled,
        "d_home": d_home, "d_away": d_away, "past_line": past_line,
        "ball_speed": ball_speed, "median_speed": median_speed, "fast_speed": fast_speed,
        "n_tracked": n_tracked, "long_still": long_still,
        "px": px, "py": py, "speed": speed,
    }


def attack_sign(px, team_of_slot):
    """Per frame: +1 if home attacks towards x = LENGTH in that half, -1 otherwise.

    Over a whole half each team stands, on average, nearer its own goal, so comparing the two
    teams' mean x per half gives the direction with no labels at all.
    """
    sign = np.ones(len(px))
    home = team_of_slot == 0
    blocks = half_video_blocks(GAME)
    for i, (_, first, last) in enumerate(blocks):
        with np.errstate(all="ignore"):
            hx = np.nanmean(px[first:last + 1][:, home])
            ax = np.nanmean(px[first:last + 1][:, ~home])
        s = 1.0 if hx < ax else -1.0
        lo = first if i else 0
        hi = blocks[i + 1][1] if i + 1 < len(blocks) else len(px)
        sign[lo:hi] = s
    return sign


def formation_features(f, team_of_slot):
    """Team shape, in the home team's attacking direction. Uses no ball information at all."""
    sign = attack_sign(f["px"], team_of_slot)
    x = np.where(sign[:, None] > 0, f["px"], LENGTH - f["px"])
    y = f["py"]
    columns = []
    with np.errstate(all="ignore"):
        for team in (0, 1):
            mine = (team_of_slot == team)[None, :]
            tx = np.where(mine, x, np.nan)
            ty = np.where(mine, y, np.nan)
            ts = np.where(mine, f["speed"], np.nan)
            cx = np.nanmean(tx, axis=1)
            velocity = np.full(len(cx), np.nan)
            velocity[12:-12] = (cx[24:] - cx[:-24]) * FPS / 24.0
            for value in (cx, np.nanstd(tx, axis=1), np.nanstd(ty, axis=1),
                          np.nanpercentile(tx, 10, axis=1), np.nanpercentile(tx, 90, axis=1),
                          np.nanmean(ts, axis=1), velocity):
                columns.append(centred_mean(value, 12))
                columns.append(centred_mean(value, 50))
    X = np.stack(columns, axis=1)
    return np.where(np.isnan(X), np.nanmedian(X, axis=0), X)


def fit_logistic(Z, y, iterations=500, step=0.5, l2=1e-3):
    """Balanced logistic regression by plain gradient descent from zero: deterministic."""
    weight = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        g = weight * (p - y)
        w -= step * (Z.T @ g / len(y) + l2 * w)
        b -= step * g.mean()
    return w, b


def formation_model(f, team_of_slot, dead_seed, graded):
    """log-odds 'home has it' from team shape, learned from frames where the ball says so clearly.

    Pseudo-labels: the ball is at one player's feet (within CONTROL_NEAR) and no opponent is
    within 2.5 m. The code picks them; no person does. Checked by training on one half and
    testing on the other.
    """
    X = formation_features(f, team_of_slot)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mean) / std
    clear = (f["real"] & (f["past_line"] < -1.0) & graded & ~dead_seed
             & (np.minimum(f["d_home"], f["d_away"]) < CONTROL_NEAR)
             & (np.maximum(f["d_home"], f["d_away"]) > 2.5))
    y_all = (f["d_home"] < f["d_away"]).astype(float)

    blocks = half_video_blocks(GAME)
    in_half = [np.zeros(len(Z), dtype=bool) for _ in blocks]
    for mask, (_, first, last) in zip(in_half, blocks):
        mask[first:last + 1] = True
    scores = []
    for test in range(len(blocks)):
        train = clear & ~in_half[test]
        held = clear & in_half[test]
        w, b = fit_logistic(Z[train], y_all[train])
        pred = (Z[held] @ w + b) > 0
        truth = y_all[held] == 1
        scores.append(0.5 * (np.mean(pred[truth]) + np.mean(~pred[~truth])))
    w, b = fit_logistic(Z[clear], y_all[clear])
    log("  formation model: {} clear frames, other-half balanced accuracy {}".format(
        int(clear.sum()), " / ".join("{:.3f}".format(s) for s in scores)))
    return Z @ w + b, min(scores)


def team_logit(f, formation=None):
    """+ = home has it, - = away has it, 0 = no evidence. Before temperature.

    The ball, where someone is playing it, decides. Team shape fills in, weakly, where the ball
    says nothing: hidden, in flight, or loose.
    """
    near = np.minimum(f["d_home"], f["d_away"])
    weight = np.clip((CONTROL_FAR - near) / (CONTROL_FAR - CONTROL_NEAR), 0.0, 1.0)
    weight = np.where(f["filled"], 0.5 * weight, weight)     # drawn ball: half the trust
    with np.errstate(invalid="ignore"):
        margin = np.clip((f["d_away"] - f["d_home"]) / TEAM_SCALE, -TEAM_CAP, TEAM_CAP)
    usable = f["seen"] & (f["past_line"] < OUT_TOUCH) & np.isfinite(near)
    weight = np.where(usable, weight, 0.0)
    t = weight * np.nan_to_num(margin)
    if formation is not None:
        t += (1.0 - weight) * FORMATION_WEIGHT * np.clip(formation, -2.0, 2.0)
    return t


def ball_out(f):
    """Frames where the ball is past a line, in runs long and deep enough to be real."""
    past = f["real"] & (f["past_line"] > OUT_TOUCH)
    out = np.zeros(len(past), dtype=bool)
    for start, end in runs(past):
        if end - start >= OUT_MIN_FRAMES and np.nanmax(f["past_line"][start:end]) > OUT_SURE:
            out[start:end] = True
    return out


def restarts(home_name):
    """Tagged set pieces: [(window_start, window_end, kind, team)] with team 'H', 'A' or None."""
    out = []
    home_key = home_name.lower()[:5]
    for event in load_smart_tagging.load():
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        tags = {k.lower(): (v or "").strip() for k, v in event["tags"].items()}
        kind = tags.get("type", "").lower()
        team_tag = tags.get("team", "").lower()
        if not team_tag or team_tag == "n/a":
            team = None
        else:
            team = "H" if team_tag[:5] == home_key else "A"
        out.append((event["start_frame"], event["end_frame"], kind, team, tags.get("half", "")))
    return out


def speed_model(f, dead_seed, alive_seed):
    """P(dead) from how the players move, learned from pseudo-labels the code chose itself.

    Logistic regression, plain gradient descent, fixed start: deterministic. Ball features are left
    out on purpose so it also works on the 41% of frames with no ball.
    """
    columns = [
        centred_mean(f["median_speed"], 12), centred_mean(f["median_speed"], 50),
        centred_mean(f["fast_speed"], 12), centred_mean(f["fast_speed"], 50),
        centred_mean(f["n_tracked"].astype(float), 25),
    ]
    X = np.stack(columns, axis=1)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mean) / std
    Z = np.concatenate([Z, Z[:, :4] ** 2], axis=1)      # allows a soft threshold shape

    train = dead_seed | alive_seed
    y = dead_seed[train].astype(float)
    w, b = fit_logistic(Z[train], y, iterations=400)
    logit = Z @ w + b
    p_train = 1.0 / (1.0 + np.exp(-logit[train]))
    acc = np.mean((p_train > 0.5) == (y == 1))
    log("  speed model: {} dead / {} alive seed frames, seed accuracy {:.3f}".format(
        int(dead_seed.sum()), int(alive_seed.sum()), acc))
    return np.clip(logit, -4.0, 4.0)


def dead_logit(f, out, restart_list, n_frames):
    """+ = dead, - = alive. Before temperature. Also returns the hard 'pre-kick-off' mask."""
    z = np.zeros(n_frames)
    hard_dead = np.zeros(n_frames, dtype=bool)
    restart_dead = np.zeros(n_frames, dtype=bool)
    restart_alive = np.zeros(n_frames, dtype=bool)
    team_prior = np.zeros(n_frames)
    out_runs = runs(out)
    out_ends = np.array([end for _, end in out_runs]) if out_runs else np.array([], dtype=int)

    # a player running with the ball well inside the pitch: open play beyond doubt
    in_play_ball = (f["real"] & (f["past_line"] < -3.0)
                    & (np.minimum(f["d_home"], f["d_away"]) < CONTROL_NEAR)
                    & (f["ball_speed"] > 1.0) & (f["ball_speed"] < 15.0))

    blocks = half_video_blocks(GAME)
    for start, end, kind, team, half in restart_list:
        lo = max(0, start - RESTART_BEFORE)
        restart_dead[lo:start] = True
        if kind in OUT_LOOKBACK and len(out_ends):
            earlier = out_ends[(out_ends <= start) & (out_ends >= start - OUT_LOOKBACK[kind])]
            if len(earlier):
                # The stoppage began at that ball-out, unless open play came after it. Then the
                # ball-out ended an earlier stoppage, and this one began after that play.
                begin = int(earlier.max())
                play = [stop for go, stop in runs(in_play_ball[begin:start]) if stop - go >= LIVE_BREAK]
                restart_dead[begin + (max(play) if play else 0):start] = True
        if end - start <= SHORT_WINDOW:
            restart_alive[end:end + RESTART_AFTER] = True
        if team is not None:
            team_prior[start:start + RESTART_TEAM_SPAN] += 1.0 if team == "H" else -1.0
        if kind == "kick off (start)":
            # the kick-off frame of each half is published (DATA.md); the window start otherwise
            kickoff = load_smart_tagging.KICKOFF_VIDEO_FRAMES.get(half, start)
            for _, first, last in blocks:
                if first <= start <= last:
                    hard_dead[first:kickoff] = True
                    restart_alive[kickoff:kickoff + RESTART_AFTER] = True

    # pseudo-labels for the speed model
    near_restart = np.zeros(n_frames, dtype=bool)
    for start, end, *_ in restart_list:
        near_restart[max(0, start - 10 * FPS):end + 5 * FPS] = True
    dead_seed = out | restart_dead | hard_dead
    alive_seed = in_play_ball & ~near_restart & ~dead_seed
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    z += speed_model(f, dead_seed & graded, alive_seed & graded)

    z += np.where(out, 8.0, 0.0)
    z += np.where(restart_dead, 6.0, 0.0)
    z -= np.where(restart_alive, 4.0, 0.0)
    live_ball = f["seen"] & (f["past_line"] < -0.5)
    z -= np.where(live_ball & (np.minimum(f["d_home"], f["d_away"]) < 1.5)
                  & (f["ball_speed"] > 2.0), 2.0, 0.0)
    z -= np.where(live_ball & (f["ball_speed"] > 6.0), 2.0, 0.0)
    z += np.where(f["long_still"], 2.0, 0.0)
    return z, hard_dead, team_prior, dead_seed, graded


# ---------------------------------------------------------------------------------------------
# 3. Viterbi over H, A, D
# ---------------------------------------------------------------------------------------------
def viterbi(log_h, log_a, log_d):
    """Most likely H/A/D path. States 0=H, 1=A, 2=D. Costs from SWITCH_COST and DEAD_COST."""
    n = len(log_h)
    emit = np.stack([log_h, log_a, log_d], axis=1)
    cost = np.array([[0.0, SWITCH_COST, DEAD_COST],
                     [SWITCH_COST, 0.0, DEAD_COST],
                     [DEAD_COST, DEAD_COST, 0.0]])
    back = np.zeros((n, 3), dtype=np.int8)
    score = emit[0].copy()
    for t in range(1, n):
        cand = score[:, None] - cost              # from i to j
        best = cand.argmax(axis=0)
        back[t] = best
        score = cand[best, np.arange(3)] + emit[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def hand_over_at_control(path, f, first):
    """Move each H<->A change to the frame the new team reaches the ball.

    The decoder counts a player closing in from 3 m as evidence, so it tends to change team while
    the ball is still loose. A loose ball stays with the last team (CHALLENGE.md). So where the
    ball is in view and nobody is at it, the change waits for the new team's first frame at the
    ball. If the ball is hidden first, or the old team gets back to it, the change stays put.
    """
    path = path.copy()
    i = slice(first, first + len(path))
    d_home, d_away, seen = f["d_home"][i], f["d_away"][i], f["seen"][i]
    near = np.minimum(d_home, d_away)
    at_ball = [(d_home < CONTROL_NEAR) & (d_home <= d_away), (d_away < CONTROL_NEAR) & (d_away < d_home)]
    changes = np.flatnonzero((path[1:] != path[:-1]) & (path[1:] < 2) & (path[:-1] < 2)) + 1
    for k in changes:
        old, new = path[k - 1], path[k]
        end = k
        while end < len(path) and path[end] == new:
            end += 1
        j = k
        while j < end and seen[j] and near[j] >= CONTROL_NEAR:
            j += 1
        if k < j < end and at_ball[new][j]:
            path[k:j] = old
    return path


def label(players, ball, meta_path):
    n_frames = len(ball)
    team_of_slot = team_of_each_track(meta_path)
    home_name, away_name = team_names(meta_path)
    log("teams: H = {}, A = {}".format(home_name, away_name))

    f = frame_features(players, ball, team_of_slot)
    out = ball_out(f)
    restart_list = restarts(home_name)
    log("  {} tagged set pieces, {} ball-out stretches".format(len(restart_list), len(runs(out))))

    z_dead, hard_dead, team_prior, dead_seed, graded = dead_logit(f, out, restart_list, n_frames)
    formation, accuracy = formation_model(f, team_of_slot, dead_seed, graded)
    if accuracy < FORMATION_MIN_ACC:
        log("  team shape not reliable enough here; not used")
        formation = None
    t = team_logit(f, formation) + team_prior

    log_dead = log_sigmoid(z_dead)
    log_alive = log_sigmoid(-z_dead)
    # Team evidence only chooses between H and A: the favoured side pays nothing, the other pays
    # the log-odds. So "a home player is near the ball" never argues that the ball is in play --
    # players stand next to the ball at every set piece. Frame by frame, alive-or-dead comes from
    # z_dead alone (over a sequence, flickering team evidence can still make a short D cheaper).
    log_h = TEMPERATURE * (log_alive + log_sigmoid(t) - log_sigmoid(np.abs(t)))
    log_a = TEMPERATURE * (log_alive + log_sigmoid(-t) - log_sigmoid(np.abs(t)))
    log_d = TEMPERATURE * log_dead
    log_h = np.where(hard_dead, -1e9, log_h)
    log_a = np.where(hard_dead, -1e9, log_a)
    log_d = np.where(hard_dead, 0.0, log_d)

    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        path = viterbi(log_h[first:last + 1], log_a[first:last + 1], log_d[first:last + 1])
        path = hand_over_at_control(path, f, first)
        labels[first:last + 1] = np.array(["H", "A", "D"])[path]
    return labels


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def summary(labels):
    graded = labels != "X"
    alive = np.isin(labels, ["H", "A"])
    spells = [(s, e) for s, e in runs(labels == "H")] + [(s, e) for s, e in runs(labels == "A")]
    durations = np.array([e - s for s, e in spells]) / FPS
    log("  graded frames: {}  H {:.1f}%  A {:.1f}%  D {:.1f}%".format(
        int(graded.sum()), *(100.0 * np.mean(labels[graded] == k) for k in "HAD")))
    log("  possessions: {}  mean {:.1f} s  median {:.1f} s  ball alive {:.1f} min".format(
        len(spells), durations.mean(), np.median(durations), alive.sum() / FPS / 60))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="possession", help="output file name, no extension")
    args = parser.parse_args()

    meta_path, _ = require(["atd_meta", "atd_raw"])
    players, ball = load_tracking()
    labels = label(players, ball, meta_path)
    path = os.path.join(RESULTS, args.name + ".csv")
    write_submission(path, labels)
    summary(labels)
    log("wrote " + os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
