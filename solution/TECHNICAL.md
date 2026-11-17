# Ball possession from automatic tracking — technical note

**Team Total Football** · More than a Hack 2027, Phase 1 · code: https://github.com/vaishnav-197/total-football-mtah2027

One label per frame of the tactical video: `H`, `A`, `D` or `X`.

## 1. Approach

Four steps. Everything is in [`solution/possession.py`](possession.py), numpy only.

1. **Evidence per frame.** We measure two separate things:
   - **Which team.** We compare the nearest home and away players to the ball. Being 1 m closer is worth 1 unit of log-odds, capped at 3. This counts in full when a player is within 1 m of the ball, fades out by 3 m, and is ignored past 3 m (a loose ball or a pass in flight). Wherever the ball does not decide fully (hidden, loose, or 1–3 m from the nearest player), a weaker hint from team shape fills the rest (section 3).
   - **Dead or alive.** Signals: the ball well past a line, the tagged restarts, a ball lying still, and how the players move (section 3).
2. **Decode the whole half at once.** We run Viterbi over three states: H, A and D.
   - Changing team costs 6. Entering or leaving D costs 8 each way. Frame evidence is scaled by 0.2, since 25 frames a second are not 25 independent looks.
   - These are trade-offs between evidence and cost, not hard minimums. At full strength (3 log-odds), a lasting change of team needs about 0.4 s of clear control, and a brief one that hands the ball back about 0.8 s. A stoppage needs about 1 s of strong dead evidence (4 log-odds), or several seconds of weak evidence.
   - Because Viterbi sees the whole half, each change lands where the evidence turns, not after a lag.
   - Team evidence only chooses between H and A: the favoured team pays nothing, the other pays the log-odds. Frame by frame it never counts as evidence that the ball is in play, because players stand next to the ball at every set piece. Over a sequence, team evidence that flips back and forth can still make a short D cheaper than paying for the flips. We have not measured how often that happens.
3. **Hand over at control.** The decoder counts a player closing in from 3 m as evidence, so on its own it changes team while the ball is still loose. The rules say a loose ball stays with the last team. So, while the ball is in view and nobody is within 1 m, each change of team is moved to the first frame the new team is at the ball. If the ball is hidden first, or the old team gets back to it first, the change stays where it is.
4. **Write it out.**
   - Each graded half is D before its kick-off. We use the kick-off frames published in DATA.md (11,185 and 113,742), which the Smart Tagging `Kick Off (Start)` windows contain.
   - X outside the graded halves.

**Why this design.** The ranking metric matches possessions, not frames. The baseline loses on three things:
- **Flicker:** every flip between teams adds fake possessions.
- **"Ball not seen = dead ball":** this splits a possession at every occlusion. The ball is hidden on 41% of graded frames, mostly in long gaps (median 11 s).
- **Missed stoppages:** these merge two possessions into one.

A state model with switch costs addresses all three at once. The structure is also easy to inspect, and every constant has a football or physics meaning.

## 2. Assets used

| asset | used for |
|---|---|
| Tactical ATD (players, ball) | positions in metres (pitch 105.3 × 68.0 from `games.py`), player speeds, ball speed from its own positions |
| ATD metadata | which track slot belongs to which team (as in `starter/baseline.py`) |
| Smart Tagging — `SET PIECES` | the 59 set-piece windows: dead just before each one; the restarting team, where tagged, for the first 5 s; live just after the 5 s kick-off windows. The other 53 windows are a fixed 35 s and their end is not used |
| Published kick-off frames (DATA.md) | where each half's play begins |
| Tactical homography | not used: the ATD is already in pitch coordinates |
| Tactical video | not used, for lack of time; the clearest next step (section 6) |
| Broadcast ATD / video | not used: not synchronised to the tactical timeline, and no mapping is provided |

Smart Tagging phases (`BUILD UP`, `PROGRESSION`, …) were not used. They are phases, not possessions, and their team semantics vary by code.

## 3. Training data

**No human-labelled data of any kind.** There is no hand labelling, nothing from the provided game, no external event data, and nothing derived from either ground truth.

Two small logistic regressions are trained on **pseudo-labels the code selects by fixed rules**. No person decided any of them, which RULES.md allows.

| model | pseudo-labels | features | self-check |
|---|---|---|---|
| dead-ball movement model | dead: ball past a line; the 2 s before each set-piece window; the stretch from the last ball-out to a restart (section 4); the time before kick-off. alive: ball at a player's feet, more than 3 m inside the pitch, moving 1–15 m/s, and not from 10 s before a set-piece window opens to 5 s after it closes | median and 80th-percentile player speed averaged over ±0.5 s and ±2 s (windows of 1 s and 4 s), their squares, and players tracked over ±1 s | trained on one half, tested on the other: balanced accuracy 0.73 and 0.74. A weak signal, used as one input among several |
| team-shape model | the ball observed more than 1 m inside the pitch, within 1 m of one player, no opponent within 2.5 m, outside any dead-ball pseudo-label | per team, in the home side's attacking direction: centroid, spread, 10th/90th percentile depth, mean speed and forward velocity, each averaged over ±0.5 s and ±2 s. **No ball information** | trained on one half, tested on the other: balanced accuracy 0.83 and 0.87 |

- The attacking direction in each half comes from the data itself: over a half, each team stands on average nearer its own goal.
- Both models use plain gradient descent from zero, so they are deterministic.
- Feature scaling uses the whole match. That uses no labels, but the held-out halves are not fully unseen.
- The team-shape accuracy is measured on clear-ball frames. On hidden-ball frames, where it matters most, it is probably lower.

## 4. Possession rules implemented

- **Control.** A team counts as playing the ball only if one of its players is within 3 m, and fully only within 1 m.
- **Loose and contested balls.** Past 3 m the ball gives no team evidence, and team shape gives only a weak hint. While the ball is in view and loose, a change of team waits until the new team reaches it (step 3), so possession stays with the last team, as CHALLENGE.md specifies. Between 1 and 3 m, a contested ball gives partial evidence to whichever team is closer.
- **Touches, deflections, aerial duels.** A brief touch by the other team that hands the ball back needs about 0.8 s of clear control to count. A lasting change needs about 0.4 s. Both come from the switch cost.
- **Ball out.** The ball is out when at least 5 consecutive observed frames (0.2 s) are more than 0.2 m past a line and at least one of them is more than 1 m past it. The stretch counts from the moment it passes 0.2 m.
- **Restarts.**
  - D for the 2 s before each set-piece window. For the kick-offs, where the ball is seen, the window opens about 3 s before the kick.
  - For throw-ins, corners, goal kicks and kick-offs after a goal: D from the last ball-out (at most 60–180 s earlier) until the window opens. If a full second of open play followed that ball-out, the ball-out ended an earlier stoppage, so D starts after that play instead. Open play here means a player running with the ball more than 3 m inside the pitch.
  - Live for 1 s after each 5 s kick-off window, and from each published kick-off frame.
  - Where the restart is tagged with a team (never throw-ins, which are `N/A`), a mild hint toward that team for the first 5 s of the window. The tag names the team taking the restart. At goal kicks it is the side defending that goal (8 of 8), and at corners the side attacking it (5 of 6 where the ball is seen).
- **Other stoppages** (fouls, offsides, injuries) are not tagged. They come from the players' movement and a ball lying still for 2 s. 27 of our 100 stoppages (8.5 min) rest on movement alone.
- **Hidden ball.**
  - Gaps under 0.5 s are filled on a straight line. This covers only 120 frames, because most gaps are long. Filled frames count half for team evidence; their positions are used as they are for ball speed and stillness.
  - Longer gaps give no ball evidence. The team-shape hint carries the decision, at half weight and never as strong as the ball at a player's feet.
- **Dead-ball owner** is never asked for, so it is never computed: a dead ball is D.

## 5. How the public 5 minutes were used

Only to check the pipeline: frame alignment, file format, and labels meaning what we thought. That check found one logic error: team evidence was also counting as evidence that the ball was in play. We fixed it on reasoning alone. **No constant was chosen by scoring against the public window,** and no version was picked over another by its public score.

**This version (v4) was not scored against the public window at all.** We used the window for one check of meaning, which is what the rules intend it for. We compared the answer key with the tracking only, and read no submission:
- **Changes of team.** The answer key has 10 changes where the new team is seen reaching the ball. Its median change is 0.04 s before the new team's first frame within 1 m. In 5 of 7 cases it comes 1.4–2.3 s after the old team's last touch. So a change happens at the reception, not the pass, which is what step 3 assumes.
- **Ball out.** Its one stoppage from a ball-out starts 0.04 s after the ball passes 0.2 m beyond the line, where our ball-out stretch starts. Two of its other stoppages have no ball-out at all: fouls or offsides.
- **Restarts.** In 4 of 4 restarts, the team it gives the ball to is the team at the ball.

v4's changes come from an internal audit that compared our labels with what the tracking itself shows (section 6). None of the audit's checks reads any ground truth:

| finding in v3 | fix in v4 |
|---|---|
| The look-back from a restart to the last ball-out ran through open play in 15 of 30 cases | start the stoppage after the last full second of open play |
| Set-piece windows are a fixed 35 s (kick-offs 5 s), but the restarting-team hint lasted the whole window | hint only for the first 5 s |
| "Live just after the window" assumed every window closes just after the restart; only the kick-off windows do | used only for the 5 s windows |
| The second half went live 3.3 s before the published kick-off | D until the published kick-off frame |
| Of 55 clearly visible turnovers, v3 changed team on 32, a median 0.58 s before the new team reached the ball | step 3: hand over at control. v4 changes team on 37 of them (all fixes together), a median 0.00 s from the new team reaching the ball |

Full match, as written (`validate_submission.py`):

| | |
|---|---|
| possessions | 246, mean 13.3 s, median 8.3 s |
| stoppages | 100 (27 from player movement alone) |
| ball alive | 54.5 min of the graded time |
| graded frames | H 29.7% · A 21.8% · D 48.5% |

## 6. Checks, limitations and next steps

**Checked, on the data and our own output only:**
- **Viterbi** matches brute force on 300 random cases.
- **No near-ties:** moving every frame's evidence by 1e-9 changes no label.
- **Kick-off timing:** both halves go live on the published kick-off frame. After the three goals where the ball is seen on the centre spot, we go live 1 frame before it is seen leaving.
- **Ball crossing a line:** where it ends play, our switch to D lands on the crossing frame (median offset 0).
- **No surprises in the evidence:** no non-finite value on any graded frame.
- **Alignment:** ATD frames are frame_id − 1 and Smart Tagging times are seconds × 25, as DATA.md states. The tagged kick-offs contain the real ones.
- **Teams:** each half's attacking direction comes out opposite, as it should.

**Limitations:**
- **Constants are judgement, not fitted.** No labelled data from other games was used to set them. The clearest next step is to label other matches (allowed) and fit the switch costs and distances there.
- **Balls in the air.** The ATD ball is a ground projection, so a high ball can look near the wrong player. The switch cost absorbs short cases but not long flights.
- **The dead-ball movement model is weak** (0.73–0.74 balanced, held out). Many fouls and offsides are found late or missed, and a few of the 27 movement-only stoppages contain what looks like open play. Slow build-up can look like a stoppage: after one of the four kick-offs following a goal, with the ball hidden, we stay D for about 13 s. Forcing the second after the kick-off live did not help, because the model called a stoppage again straight after, so we left it.
- **Inside the 35 s windows** we do not know exactly when each restart happens. We use only the window's start.
- **Throw-ins carry no team** in the Smart Tagging.
- **Changes of possession inside long ball gaps** usually land where the ball reappears, so they are late by up to the gap.
- **The video is unused.** A ball detector on the tactical video would cut the 41% of frames with no ball, which is where most errors come from.

## 7. Reproduction

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r solution/requirements.txt
# put the data bundle in data/ exactly as DATA.md describes
.venv/bin/python solution/possession.py --name "Total Football"
.venv/bin/python evaluation/validate_submission.py "evaluation/results/Total Football.csv"
```

- The first run reads the tracking and caches it in `data/_cache/`, which git ignores. Each run takes about 20 s.
- With or without the cache, the output is byte-identical: SHA-256 `c8875897b3848ebae27e0b2b8ed5b5b85470f98e05c984c8ffc974cb0fd4d3d8`. We checked this by moving the cache aside and re-running.
- Tested with Python 3.12.3, numpy 2.5.5 and pandas 3.1.1 on macOS.
