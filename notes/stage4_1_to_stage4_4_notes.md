# Stage 4 Notes: Team Construction and Tactical Planning

## Stage 4.1 — Player–Role Suitability

### Purpose

The earlier position models showed which functional roles matched each player's attribute profile. Stage 4.1 converted those predictions into a lineup-selection score. This allowed the project to compare an official-position specialist with a player who could potentially cover the same role flexibly.

### Method

Each player was evaluated for every eligible functional role using three confirmed components:

- **Position Fit Score:** Official positions received full position fit. Predicted flexible roles used the functional-position model's probability as evidence of fit.
- **Role Attribute Score:** Measured how well the player's EAFC attributes matched the requirements of the evaluated role.
- **Overall quality:** The player's EAFC overall rating was retained so that role fit did not completely replace general player strength.

These components produced the final score:

\[
\text{Player Role Suitability}
=0.40(\text{Role Attribute Score})
+0.35(\text{OVR})
+0.25(\text{Position Fit Score}).
\]

For example, Virgil van Dijk's Center Back score was:

\[
0.40(89.55)+0.35(90)+0.25(100)=92.32.
\]

Each row in `eafc26_player_role_suitability.csv` represents one eligible player–role pairing rather than one player alone.

### Results

The input contained **11,872 players and 73 columns**. Stage 4.1 created **20,846 eligible player–role combinations**:

| Role source | Combinations |
|---|---:|
| Official position | 11,872 |
| Flexible position 1 | 6,624 |
| Flexible position 2 | 2,350 |

The number of combinations exceeded the number of players because one player could be evaluated for an official role plus as many as two predicted flexible roles. The largest candidate groups were Wide Midfielder (3,380), Center Back (3,046), and Central Midfielder (2,938). Winger had the smallest group among the eight modeled roles, with 1,981 combinations.

The highest-scoring specialist for each functional role was:

| Functional role | Highest-scoring player | Suitability |
|---|---|---:|
| Center Back | Virgil van Dijk | 92.32 |
| Full Back | Achraf Hakimi | 91.41 |
| Defensive Midfielder | Rodri | 92.06 |
| Central Midfielder | Pedri | 91.02 |
| Attacking Midfielder | Florian Wirtz | 91.27 |
| Wide Midfielder | Mohamed Salah | 92.25 |
| Winger | Vini Jr. | 92.37 |
| Striker | Erling Haaland | 93.25 |

### Interpretation

A high score means that the player is both strong and suitable for that particular role under our scoring definition. It does not prove that the player would perform equally well there in a real match. Predicted flexible roles are possibilities inferred from attributes, not observed coaching decisions or match experience.

---

## Stage 4.2 — Independent Best Lineup for Each Formation

### Purpose

Stage 4.2 found the strongest possible starting XI for each formation independently. This created a specialized baseline before imposing a fixed squad-size restriction.

### Method

The Hungarian assignment algorithm assigned ten unique outfield players to the required functional-role slots in each formation. Every slot received exactly one eligible player, and no player could fill two positions in the same lineup. Alisson was the starting goalkeeper and David Raya was initially retained as the backup goalkeeper.

The three tested formations were:

- 4-3-3
- 4-2-3-1
- 4-4-2

### Results

| Formation | Average outfield suitability | Starting XI average score |
|---|---:|---:|
| 4-3-3 | 91.32 | 90.95 |
| 4-2-3-1 | 91.41 | 91.03 |
| 4-4-2 | 91.52 | 91.13 |

The 4-4-2 received the highest score, but the differences were extremely small. It exceeded the 4-3-3 starting-XI average by only 0.18 points and the 4-2-3-1 by only 0.10 points. These results do not demonstrate that 4-4-2 is objectively superior in real matches; they only show that its assigned players scored slightly higher under the suitability formula.

Every selected player filled an official functional role. Flexible assignments were unnecessary because each formation could search the entire player dataset and choose elite specialists without a squad restriction.

### Main lineups

- **4-3-3:** Alisson; Van Dijk, Gabriel, Hakimi, Nuno Mendes; Rodri, Vitinha, Pedri; Vini Jr., Saka, Haaland.
- **4-2-3-1:** Alisson; Van Dijk, Gabriel, Hakimi, Nuno Mendes; Kimmich, Rodri; Wirtz; Vini Jr., Saka, Haaland.
- **4-4-2:** Alisson; Van Dijk, Gabriel, Hakimi, Nuno Mendes; Vitinha, Pedri; Salah, Raphinha; Kane, Haaland.

### Limitation

The formations were optimized independently, so Stage 4.2 could select a different collection of players for every shape. It did not yet answer whether one fixed team could support all three formations.

---

## Stage 4.3 — Fixed 23-Player Squad

### Purpose

Stage 4.3 selected one squad capable of supporting all three formations. The squad-size assumption was updated from 18 players to 23 players so that the model included a fuller reserve group.

### Method

A mixed-integer optimization model jointly selected:

- 3 goalkeepers
- 20 outfield players
- 23 total squad members

The model rewarded strong formation lineups, useful reserves, multi-role coverage, and players who could be retained when formations changed. Every formation still required ten unique outfield starters and one goalkeeper.

### Selected squad

**Goalkeepers**

- Alisson — default starter
- David Raya — distribution-oriented alternative
- Thibaut Courtois — protection-oriented alternative

**Regular formation starters**

- Virgil van Dijk
- Gabriel
- Achraf Hakimi
- Nuno Mendes
- Rodri
- Pedri
- Vitinha
- Bukayo Saka
- Vini Jr.
- Erling Haaland
- Joshua Kimmich
- Florian Wirtz
- Mohamed Salah
- Raphinha
- Harry Kane

**Additional reserves**

- Kylian Mbappé
- Ousmane Dembélé
- Federico Valverde
- Marquinhos
- Khvicha Kvaratskhelia

### Results

The 23-player squad reproduced all three independently optimized Stage 4.2 lineups with zero suitability loss:

| Formation | Fixed-squad average suitability | Score loss from independent baseline |
|---|---:|---:|
| 4-3-3 | 91.32 | 0.00 |
| 4-2-3-1 | 91.41 | 0.00 |
| 4-4-2 | 91.52 | 0.00 |

Formation switching required:

| Formation change | Outfield players retained | Substitutions required |
|---|---:|---:|
| 4-3-3 → 4-2-3-1 | 8 | 2 |
| 4-3-3 → 4-4-2 | 7 | 3 |
| 4-2-3-1 → 4-4-2 | 5 | 5 |

### Interpretation

The fixed squad is large enough to contain all 15 specialized outfield starters used across the independent lineups, plus five additional reserves. Therefore, the squad restriction did not force a loss in normal lineup quality. Flexibility becomes more useful during substitutions, formation changes, unavailability, and other match-specific situations than during the unrestricted starting-lineup calculation.

### Limitation

The Stage 4.3 optimizer measured coverage and suitability, not chemistry, fatigue, injuries, transfer feasibility, wages, or real coaching preferences. The saved filename may still contain `18_player`, but the validated file contains the corrected 23-player roster.

---

## Stage 4.4 — Default Starting XI and Legal Substitution Plans

### Purpose

Stage 4.4 changed the project from squad construction to match planning. It selected one balanced default XI and created different responses for leading, tied, and trailing match states.

### Default starting XI

The default formation is a balanced 4-3-3:

| Slot | Player |
|---|---|
| GK | Alisson |
| CB | Virgil van Dijk |
| CB | Gabriel |
| FB | Nuno Mendes |
| FB | Achraf Hakimi |
| DM | Rodri |
| CM | Pedri |
| CM | Federico Valverde |
| W | Bukayo Saka |
| W | Vini Jr. |
| ST | Erling Haaland |

The outfield lineup had an average role suitability of **91.31**, average attack of **79.15**, average defense of **72.10**, average physical rating of **80.40**, and average passing of **78.50**.

### Scenario-specific scoring

The tactical plans used the real PAC, SHO, PAS, DRI, DEF, and PHY values from `eafc26_outfield_full_features.csv`. The first version incorrectly used a neutral fallback when these attributes were absent from the role-suitability file; this was detected because all tactical averages became identical and was corrected before simulation.

- **Leading:** prioritized suitability, defense, physical strength, and passing.
- **Tied:** prioritized balance and avoided unnecessary disruption.
- **Trailing:** prioritized pace, shooting, passing, dribbling, and an additional striker.

The optimizer also included a substitution penalty. Leading and tied plans were capped at two planned tactical substitutions, while the trailing plan was capped at three. The match-law maximum remained five, leaving substitutions available for injuries or later decisions.

### Leading plan

- Change from 4-3-3 to 4-2-3-1.
- Federico Valverde and Pedri leave.
- Joshua Kimmich and Florian Wirtz enter.
- Kimmich joins Rodri as the second defensive midfielder.
- Two substitutions are made at halftime. They count toward the five-player limit but use zero in-play substitution opportunities.

The whole-team average defense does not increase, so this should be described as a **more protective positional structure**, not as a universally higher-defensive-attribute lineup. Its two-defensive-midfielder shape and passing emphasis will be represented explicitly in Stage 5.

### Tied plan

- Remain in the original 4-3-3.
- Make no automatic substitutions.
- Preserve all five available substitutions for later information, fatigue, injury, or a change in match state.

This result is important: the model does not force substitutions merely because they are available.

### Trailing plan

- Change from 4-3-3 to 4-4-2.
- Bukayo Saka, Rodri, and Vini Jr. leave.
- Mohamed Salah, Raphinha, and Kylian Mbappé enter.
- Haaland and Mbappé form the two-striker attack.
- Three substitutions are made at halftime. They count toward the five-player limit but use zero in-play substitution opportunities.

The trailing plan raises average attack from **79.15 to 80.53**, while average defense falls from **72.10 to 68.10**. This is the intended risk–reward tradeoff: the team becomes more dangerous offensively but more exposed defensively.

### Rule-compliance results

All three plans passed every implemented rule check:

| Scenario | Ending shape | Substitutions | In-play windows | Halftime used | Legal |
|---|---|---:|---:|---|---|
| Leading | 4-2-3-1 | 2 | 0 | Yes | Yes |
| Tied | 4-3-3 | 0 | 0 | No | Yes |
| Trailing | 4-4-2 | 3 | 0 | Yes | Yes |

Each ending lineup contained exactly 11 players and one goalkeeper. Alisson remained in goal, every player belonged to the selected squad, no removed player returned, and no plan exceeded five substitutions or three substitution windows.

### Limitations

- The halftime decision point is an assumption, not learned from event data.
- The model does not yet include fatigue, injuries, opponent tactics, player chemistry, cards, home advantage, or coaching behavior.
- A formation changes how attributes function together, so a simple average of all outfield DEF ratings cannot fully represent defensive structure.
- The plans are inputs to the Stage 5 simulation, not proof that the substitutions improve real-world win probability.

---

## Stage 4 Overall Conclusion

Stage 4 transformed individual player evaluations into a complete tactical system. The project first scored player–role fit, created specialized formation baselines, selected one 23-player squad without losing lineup quality, and finally created legal match-state responses. The resulting team begins in a balanced 4-3-3, becomes structurally more protective while leading, remains stable while tied, and accepts defensive risk for additional attacking strength while trailing. Stage 5 will test how these choices behave under an explicit set of simulation assumptions.