from pathlib import Path

# ============================================================
# STAGE 4 SHARED CONFIGURATION
# ============================================================
#
# Formation benchmark:
# We use a representative set of modern formations discussed in:
#
#   SOCCER.com:
#   https://www.soccer.com/guide/most-popular-soccer-formations
#
#   PFSA:
#   https://thepfsa.co.uk/football-formations/
#
# The external sources motivate WHICH formations are considered.
# The exact EAFC slot mapping below is our project's operationalization,
# because EAFC player labels and our eight functional roles do not map
# one-to-one onto every tactical concept (especially wing-backs).
#
# IMPORTANT:
# Every formation below contains 10 outfield slots.
# Goalkeeper selection is handled separately.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"

MULTIROLE_FILE = (
    DATA
    / "processed"
    / "eafc26_players_multirole_representation.csv"
)

GOALKEEPER_FILE = (
    DATA
    / "processed"
    / "eafc26_goalkeepers_with_styles.csv"
)

# Each slot contains:
#
# slot:
#   Display label for the formation position.
#
# accepted_positions:
#   Exact official EAFC primary/alternative positions that make
#   a player eligible for the slot. This preserves laterality.
#
# suitability_roles:
#   Stage 3.6C functional-role suitability signal(s) used to
#   evaluate fit after eligibility has already been established.
#
# For normal slots there is one suitability role.
#
# For LWB/RWB, our data do not contain a separate wing-back
# functional model. We therefore require exact left/right eligibility
# and summarize Fullback + Wide Midfielder suitability with their
# simple mean. This is an explicit modeling choice and should later
# receive a sensitivity check.


def slot(
    name,
    accepted_positions,
    suitability_roles,
):
    return {
        "slot": name,
        "accepted_positions": set(accepted_positions),
        "suitability_roles": list(suitability_roles),
    }


FORMATIONS = {
    "4-3-3": [
        slot("LB", ["Left Back"], ["Fullback"]),
        slot("CB1", ["Center Back"], ["Center Back"]),
        slot("CB2", ["Center Back"], ["Center Back"]),
        slot("RB", ["Right Back"], ["Fullback"]),
        slot("DM", ["Defensive Midfielder"], ["Defensive Midfielder"]),
        slot("CM1", ["Central Midfielder"], ["Central Midfielder"]),
        slot("CM2", ["Central Midfielder"], ["Central Midfielder"]),
        slot("LW", ["Left Winger"], ["Winger"]),
        slot("RW", ["Right Winger"], ["Winger"]),
        slot("ST", ["Striker"], ["Striker"]),
    ],

    "4-2-3-1": [
        slot("LB", ["Left Back"], ["Fullback"]),
        slot("CB1", ["Center Back"], ["Center Back"]),
        slot("CB2", ["Center Back"], ["Center Back"]),
        slot("RB", ["Right Back"], ["Fullback"]),
        slot("DM1", ["Defensive Midfielder"], ["Defensive Midfielder"]),
        slot("DM2", ["Defensive Midfielder"], ["Defensive Midfielder"]),
        slot("AM", ["Attacking Midfielder"], ["Attacking Midfielder"]),
        slot("LW", ["Left Winger"], ["Winger"]),
        slot("RW", ["Right Winger"], ["Winger"]),
        slot("ST", ["Striker"], ["Striker"]),
    ],

    "4-4-2": [
        slot("LB", ["Left Back"], ["Fullback"]),
        slot("CB1", ["Center Back"], ["Center Back"]),
        slot("CB2", ["Center Back"], ["Center Back"]),
        slot("RB", ["Right Back"], ["Fullback"]),
        slot("LM", ["Left Midfielder"], ["Wide Midfielder"]),
        slot("CM1", ["Central Midfielder"], ["Central Midfielder"]),
        slot("CM2", ["Central Midfielder"], ["Central Midfielder"]),
        slot("RM", ["Right Midfielder"], ["Wide Midfielder"]),
        slot("ST1", ["Striker"], ["Striker"]),
        slot("ST2", ["Striker"], ["Striker"]),
    ],

    "4-1-2-1-2": [
        slot("LB", ["Left Back"], ["Fullback"]),
        slot("CB1", ["Center Back"], ["Center Back"]),
        slot("CB2", ["Center Back"], ["Center Back"]),
        slot("RB", ["Right Back"], ["Fullback"]),
        slot("DM", ["Defensive Midfielder"], ["Defensive Midfielder"]),
        slot("CM1", ["Central Midfielder"], ["Central Midfielder"]),
        slot("CM2", ["Central Midfielder"], ["Central Midfielder"]),
        slot("AM", ["Attacking Midfielder"], ["Attacking Midfielder"]),
        slot("ST1", ["Striker"], ["Striker"]),
        slot("ST2", ["Striker"], ["Striker"]),
    ],

    # PFSA describes the 3-5-2 as using three defenders with
    # wide midfielders responsible for the flanks. In our EAFC
    # representation the wing-back-like slots accept either a
    # same-side fullback or same-side wide midfielder.
    "3-5-2": [
        slot("CB1", ["Center Back"], ["Center Back"]),
        slot("CB2", ["Center Back"], ["Center Back"]),
        slot("CB3", ["Center Back"], ["Center Back"]),
        slot(
            "LWB",
            ["Left Back", "Left Midfielder"],
            ["Fullback", "Wide Midfielder"],
        ),
        slot(
            "RWB",
            ["Right Back", "Right Midfielder"],
            ["Fullback", "Wide Midfielder"],
        ),
        slot("DM", ["Defensive Midfielder"], ["Defensive Midfielder"]),
        slot("CM1", ["Central Midfielder"], ["Central Midfielder"]),
        slot("CM2", ["Central Midfielder"], ["Central Midfielder"]),
        slot("ST1", ["Striker"], ["Striker"]),
        slot("ST2", ["Striker"], ["Striker"]),
    ],

    "3-4-3": [
        slot("CB1", ["Center Back"], ["Center Back"]),
        slot("CB2", ["Center Back"], ["Center Back"]),
        slot("CB3", ["Center Back"], ["Center Back"]),
        slot(
            "LWB",
            ["Left Back", "Left Midfielder"],
            ["Fullback", "Wide Midfielder"],
        ),
        slot(
            "RWB",
            ["Right Back", "Right Midfielder"],
            ["Fullback", "Wide Midfielder"],
        ),
        slot("CM1", ["Central Midfielder"], ["Central Midfielder"]),
        slot("CM2", ["Central Midfielder"], ["Central Midfielder"]),
        slot("LW", ["Left Winger"], ["Winger"]),
        slot("RW", ["Right Winger"], ["Winger"]),
        slot("ST", ["Striker"], ["Striker"]),
    ],

    # PFSA describes 5-3-2 as a defensive variation of 3-5-2
    # in which the wide players drop deeper. Here the wide slots
    # therefore use exact LB/RB eligibility.
    "5-3-2": [
        slot("LB", ["Left Back"], ["Fullback"]),
        slot("CB1", ["Center Back"], ["Center Back"]),
        slot("CB2", ["Center Back"], ["Center Back"]),
        slot("CB3", ["Center Back"], ["Center Back"]),
        slot("RB", ["Right Back"], ["Fullback"]),
        slot("DM", ["Defensive Midfielder"], ["Defensive Midfielder"]),
        slot("CM1", ["Central Midfielder"], ["Central Midfielder"]),
        slot("CM2", ["Central Midfielder"], ["Central Midfielder"]),
        slot("ST1", ["Striker"], ["Striker"]),
        slot("ST2", ["Striker"], ["Striker"]),
    ],
}


# Stage 4.2 does NOT lock one arbitrary quality-vs-role-fit weight.
#
# Instead, we run a small sensitivity grid:
#
# role_fit_weight = 0.00 -> quality only
# role_fit_weight = 1.00 -> learned role fit only
#
# Stage 4.1 remains the clean "best player at each exact position"
# OVR baseline.
ROLE_FIT_WEIGHTS = [
    0.25,
    0.50,
    0.75,
]

REFERENCE_ROLE_FIT_WEIGHT = 0.50
