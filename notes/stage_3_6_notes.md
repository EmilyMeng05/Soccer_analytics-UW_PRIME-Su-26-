# Stage 3.6 — Validating Player Flexibility and Building a Multi-Role Representation

## Goal

The original goal of Stage 3.6 was to expand the player dataset by assigning two additional potential positions to each player using the probabilities produced by the Stage 3.5 functional-role model.

However, a high model probability for a secondary role does not necessarily mean that a player is genuinely versatile. A classifier may assign probability to several roles simply because it is uncertain about the player's primary role.

Therefore, before treating secondary model predictions as flexible positions, Stage 3.6 validates them against the official `Alternative positions` provided in the EAFC dataset.

The final goal became:

> Determine whether learned player attributes contain meaningful information about secondary positional roles, and construct a player representation that separates **official positional eligibility** from **learned role suitability**.

Stage 3.6 was divided into three parts:

- **Stage 3.6A:** Analyze official alternative positions.
- **Stage 3.6B:** Validate learned secondary-role predictions against those alternatives.
- **Stage 3.6C:** Construct a multi-role player representation for Stage 4.


---

# Stage 3.6A — Alternative Position Analysis

## Why analyze alternative positions first?

The dataset contains an `Alternative positions` variable in addition to each player's primary `Position`.

For example:

- Mohamed Salah: RM → RW
- Kylian Mbappé: ST → LW, LM
- Rodri: CDM → CM
- Jude Bellingham: CAM → CM
- Vitinha: CM → CDM, CAM
- Federico Valverde: CM → CDM, RB

These alternative positions provide an independent signal of player flexibility that was not used to train the Stage 3.5 functional-role classifier.

Therefore, they can be used to evaluate whether the model's secondary role predictions actually correspond to known positional flexibility.


## Position normalization

The primary `Position` column contains full position names, while `Alternative positions` uses EAFC abbreviations.

For example:

| Alternative code | Full position |
|---|---|
| CB | Center Back |
| LB | Left Back |
| RB | Right Back |
| CDM | Defensive Midfielder |
| CM | Central Midfielder |
| CAM | Attacking Midfielder |
| LM | Left Midfielder |
| RM | Right Midfielder |
| LW | Left Winger |
| RW | Right Winger |
| ST | Striker |

The alternative positions were first converted into full position names and then mapped into the same eight functional roles used in Stage 3.5:

1. Attacking Midfielder
2. Center Back
3. Central Midfielder
4. Defensive Midfielder
5. Fullback
6. Striker
7. Wide Midfielder
8. Winger


## Exact versus functional flexibility

An important distinction is made between **exact positional flexibility** and **functional-role flexibility**.

For example:

```text
Primary position:
Left Back

Alternative position:
Right Back