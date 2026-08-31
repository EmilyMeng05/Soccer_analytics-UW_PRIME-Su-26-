from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage4_config import (
    FORMATIONS,
    PROJECT_ROOT,
)


# ============================================================
# STAGE 4.4A — TEAM ATTRIBUTE COVERAGE DIAGNOSTIC
# ============================================================
#
# PURPOSE
# -------
#
# Stage 4.1:
#     OVR-only lineup.
#
# Stage 4.2:
#     OVR quality + learned individual role fit.
#
# Stage 4.3:
#     cross-formation robustness.
#
# Stage 4.4A now asks a DIFFERENT question:
#
#     "What team-level attribute profile did each selection method
#      actually produce?"
#
# IMPORTANT:
# ----------
#
# This script is DIAGNOSTIC ONLY.
#
# It does NOT:
#   - change any selected players,
#   - optimize a complementarity score,
#   - claim one formation is objectively best,
#   - claim that six EAFC attributes capture real-world chemistry.
#
# Instead, it compares the existing Stage 4.1 and Stage 4.2 lineups
# against a data-derived formation demand profile.
#
#
# SIX BROAD ATTRIBUTES
# --------------------
#
#     PAC, SHO, PAS, DRI, DEF, PHY
#
# These are intentionally used instead of the 34 detailed features.
#
# The 34 detailed features already drive Stage 3.6C role suitability.
# Using the six broad dimensions here gives Stage 4.4 a simpler,
# more interpretable team-level representation and reduces direct
# duplication with Stage 4.2.
#
#
# DATA-DERIVED FORMATION DEMAND
# -----------------------------
#
# We do NOT manually say:
#
#     "DEF should be worth 30%"
#     "PAS should be worth 20%"
#
# Instead:
#
# 1. Map each player's OFFICIAL PRIMARY POSITION to one of the
#    eight functional roles used in Stage 3.
#
# 2. For each functional role, calculate the empirical mean
#    PAC/SHO/PAS/DRI/DEF/PHY profile from the player dataset.
#
# 3. For each formation slot, use the empirical profile of the
#    slot's required functional role.
#
# 4. Average the ten outfield slot profiles to obtain the
#    formation's expected six-attribute profile.
#
# For LWB/RWB, Stage 3 did not train a dedicated wing-back role.
# Their formation-demand profile is therefore the equal mean of:
#
#     Fullback
#     Wide Midfielder
#
# This is ONLY the demand-profile construction.
#
# It is deliberately separate from Stage 4.2's role-fit rule, where
# the suitability signal for wing-back-like slots uses:
#
#     max(Fullback suitability, Wide Midfielder suitability)
#
#
# COVERAGE GAP
# ------------
#
# For each attribute a:
#
#     Gap_a = LineupMean_a - FormationDemand_a
#
# Positive:
#     lineup is above the empirical formation-role benchmark.
#
# Negative:
#     lineup is below the empirical formation-role benchmark.
#
# We also report:
#
#     Absolute Gap = |Gap|
#     Shortfall    = max(FormationDemand - LineupMean, 0)
#
# These are DESCRIPTIVE diagnostics.
#
# Stage 4.4A intentionally does NOT collapse the six dimensions into
# a single "chemistry score." The purpose is to inspect the pattern
# first. Stage 4.4B can later define a complementarity objective only
# if these diagnostics behave sensibly.
# ============================================================


ATTRIBUTES = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY",
]

REFERENCE_LAMBDA = 0.25


# ============================================================
# FILE PATHS
# ============================================================

BASELINE_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage_4_1_ovr_baseline"
    / "all_ovr_baseline_lineups.csv"
)

ROLE_ADJUSTED_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage_4_2_role_adjusted_quality"
    / "all_role_adjusted_quality_lineups.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage_4_4a_team_attribute_coverage"
)

PLOT_DIR = (
    RESULTS_DIR
    / "plots"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# We prefer the full processed outfield table because it contains
# primary Position plus the six broad attributes.
#
# The fallbacks make the script a little more portable if your local
# processed-data filenames differ.
ATTRIBUTE_FILE_CANDIDATES = [
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eafc26_outfield_full_features.csv",

    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_eafc26_outfield_players.csv",

    PROJECT_ROOT
    / "data"
    / "processed"
    / "eafc26_players_multirole_representation.csv",
]


# ============================================================
# PRIMARY POSITION -> FUNCTIONAL ROLE
# ============================================================
#
# IMPORTANT:
# This mapping is used only to estimate empirical role profiles.
#
# We use PRIMARY positions here so the formation-demand benchmark
# reflects players who are officially labeled as that role, rather
# than mixing in learned suitability predictions.
# ============================================================

PRIMARY_POSITION_TO_FUNCTIONAL_ROLE = {
    "Left Winger":
        "Winger",
    "Right Winger":
        "Winger",

    "Left Midfielder":
        "Wide Midfielder",
    "Right Midfielder":
        "Wide Midfielder",

    "Left Back":
        "Fullback",
    "Right Back":
        "Fullback",

    "Attacking Midfielder":
        "Attacking Midfielder",

    "Central Midfielder":
        "Central Midfielder",

    "Defensive Midfielder":
        "Defensive Midfielder",

    "Center Back":
        "Center Back",

    "Striker":
        "Striker",
}


# ============================================================
# HELPERS
# ============================================================

def canonical_player_key(value):
    """
    Normalize an ID-like value so that, for example:

        123
        123.0
        "123"

    can match one another.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith(".0"):
        numeric_part = text[:-2]

        if numeric_part.replace("-", "").isdigit():
            return numeric_part

    return text


def safe_filename(text):
    """Create a filesystem-friendly label."""

    return (
        str(text)
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


# ============================================================
# LOAD DATA
# ============================================================

def find_attribute_file():
    """
    Return the first available processed player file containing:

        Position + PAC/SHO/PAS/DRI/DEF/PHY

    We inspect columns rather than assuming that every local copy of
    a processed file contains the same variables.
    """

    checked = []

    for path in ATTRIBUTE_FILE_CANDIDATES:

        if not path.exists():
            checked.append(
                f"{path}  [not found]"
            )
            continue

        columns = pd.read_csv(
            path,
            nrows=0,
        ).columns

        required = {
            "Name",
            "Position",
            *ATTRIBUTES,
        }

        missing = sorted(
            required.difference(
                columns
            )
        )

        if not missing:
            return path

        checked.append(
            f"{path}  [missing: {missing}]"
        )

    raise FileNotFoundError(
        "Could not find a processed outfield file containing "
        "Name, Position, and PAC/SHO/PAS/DRI/DEF/PHY.\n\n"
        "Checked:\n"
        + "\n".join(
            checked
        )
    )


def load_attribute_data():
    """
    Load the outfield attributes used for:
        1. empirical role profiles,
        2. selected-lineup supply profiles.
    """

    attribute_file = (
        find_attribute_file()
    )

    df = pd.read_csv(
        attribute_file
    )

    required = {
        "Name",
        "Position",
        *ATTRIBUTES,
    }

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Attribute data is missing required columns:\n"
            f"{missing}"
        )

    for column in ATTRIBUTES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Name",
            "Position",
            *ATTRIBUTES,
        ]
    ).copy()

    df[
        "Functional Role"
    ] = df[
        "Position"
    ].map(
        PRIMARY_POSITION_TO_FUNCTIONAL_ROLE
    )

    df = df.dropna(
        subset=[
            "Functional Role"
        ]
    ).copy()

    # Build a normalized player key when the original EAFC ID exists.
    if "ID" in df.columns:
        df[
            "Player Key Canonical"
        ] = df[
            "ID"
        ].apply(
            canonical_player_key
        )
    else:
        df[
            "Player Key Canonical"
        ] = None

    print(
        "\nAttribute source:"
    )

    print(
        attribute_file
    )

    return df


def load_lineups():
    """Load Stage 4.1 and Stage 4.2 lineups."""

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            "Could not find Stage 4.1 output:\n"
            f"{BASELINE_FILE}\n\n"
            "Run stage4_1_ovr_baseline.py first."
        )

    if not ROLE_ADJUSTED_FILE.exists():
        raise FileNotFoundError(
            "Could not find Stage 4.2 output:\n"
            f"{ROLE_ADJUSTED_FILE}\n\n"
            "Run stage4_2_role_adjusted_quality.py first."
        )

    baseline = pd.read_csv(
        BASELINE_FILE
    )

    role_adjusted = pd.read_csv(
        ROLE_ADJUSTED_FILE
    )

    required_baseline = {
        "Formation",
        "Formation Slot",
        "Name",
    }

    required_role_adjusted = {
        "Formation",
        "Formation Slot",
        "Name",
        "Role Adjustment Strength",
    }

    missing_baseline = sorted(
        required_baseline.difference(
            baseline.columns
        )
    )

    missing_role_adjusted = sorted(
        required_role_adjusted.difference(
            role_adjusted.columns
        )
    )

    if missing_baseline:
        raise ValueError(
            "Stage 4.1 output is missing columns:\n"
            f"{missing_baseline}"
        )

    if missing_role_adjusted:
        raise ValueError(
            "Stage 4.2 output is missing columns:\n"
            f"{missing_role_adjusted}"
        )

    role_adjusted[
        "Role Adjustment Strength"
    ] = pd.to_numeric(
        role_adjusted[
            "Role Adjustment Strength"
        ],
        errors="coerce",
    )

    if role_adjusted[
        "Role Adjustment Strength"
    ].isna().any():
        raise ValueError(
            "Stage 4.2 contains missing or invalid lambda values."
        )

    reference_role_adjusted = (
        role_adjusted[
            np.isclose(
                role_adjusted[
                    "Role Adjustment Strength"
                ],
                REFERENCE_LAMBDA,
            )
        ]
        .copy()
    )

    if reference_role_adjusted.empty:
        raise ValueError(
            "No Stage 4.2 lineup rows were found for "
            f"lambda={REFERENCE_LAMBDA:.2f}."
        )

    return (
        baseline,
        reference_role_adjusted,
    )


# ============================================================
# EMPIRICAL FUNCTIONAL-ROLE PROFILES
# ============================================================

def calculate_functional_role_profiles(
    attribute_data,
):
    """
    Estimate each functional role's six-attribute profile using
    players whose OFFICIAL PRIMARY POSITION maps to that role.
    """

    role_profiles = (
        attribute_data
        .groupby(
            "Functional Role"
        )[ATTRIBUTES]
        .mean()
        .reset_index()
    )

    role_counts = (
        attribute_data
        .groupby(
            "Functional Role"
        )
        .size()
        .rename(
            "Number of Players"
        )
        .reset_index()
    )

    role_profiles = (
        role_profiles
        .merge(
            role_counts,
            on="Functional Role",
            how="left",
        )
    )

    # Validate that every Stage 4 functional role has an empirical
    # profile available.
    required_roles = sorted(
        {
            role
            for formation_slots
            in FORMATIONS.values()
            for slot_definition
            in formation_slots
            for role
            in slot_definition[
                "suitability_roles"
            ]
        }
    )

    available_roles = set(
        role_profiles[
            "Functional Role"
        ]
    )

    missing_roles = sorted(
        set(
            required_roles
        ).difference(
            available_roles
        )
    )

    if missing_roles:
        raise ValueError(
            "Could not estimate empirical profiles for these "
            "functional roles:\n"
            f"{missing_roles}"
        )

    return (
        role_profiles
        .sort_values(
            "Functional Role"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# FORMATION DEMAND
# ============================================================

def build_role_profile_lookup(
    role_profiles,
):
    """Convert the role-profile table into a simple lookup."""

    lookup = {}

    for _, row in role_profiles.iterrows():

        lookup[
            row[
                "Functional Role"
            ]
        ] = np.array(
            [
                float(
                    row[attribute]
                )
                for attribute
                in ATTRIBUTES
            ],
            dtype=float,
        )

    return lookup


def calculate_formation_demand(
    role_profiles,
):
    """
    Build a six-attribute demand profile for every formation.

    Each outfield slot contributes one empirical role profile.

    Normal slot:
        use its one functional role.

    Hybrid LWB/RWB slot:
        take the equal mean of Fullback + Wide Midfielder profiles.

    Finally, average across the 10 outfield slots.
    """

    role_lookup = (
        build_role_profile_lookup(
            role_profiles
        )
    )

    formation_rows = []
    slot_rows = []

    for (
        formation_name,
        formation_slots,
    ) in FORMATIONS.items():

        slot_vectors = []

        for slot_definition in formation_slots:

            roles = (
                slot_definition[
                    "suitability_roles"
                ]
            )

            role_vectors = [
                role_lookup[
                    role
                ]
                for role
                in roles
            ]

            # For a hybrid slot such as LWB/RWB, this is the equal
            # average of the empirical component-role profiles.
            slot_vector = np.mean(
                role_vectors,
                axis=0,
            )

            slot_vectors.append(
                slot_vector
            )

            slot_record = {
                "Formation":
                    formation_name,

                "Formation Slot":
                    slot_definition[
                        "slot"
                    ],

                "Demand Role(s)":
                    "|".join(
                        roles
                    ),
            }

            for (
                attribute,
                value,
            ) in zip(
                ATTRIBUTES,
                slot_vector,
            ):
                slot_record[
                    attribute
                ] = value

            slot_rows.append(
                slot_record
            )

        formation_vector = np.mean(
            slot_vectors,
            axis=0,
        )

        formation_record = {
            "Formation":
                formation_name,
        }

        for (
            attribute,
            value,
        ) in zip(
            ATTRIBUTES,
            formation_vector,
        ):
            formation_record[
                attribute
            ] = value

        formation_rows.append(
            formation_record
        )

    return (
        pd.DataFrame(
            formation_rows
        ),
        pd.DataFrame(
            slot_rows
        ),
    )


# ============================================================
# ATTACH SIX ATTRIBUTES TO EXISTING LINEUPS
# ============================================================

def build_attribute_lookups(
    attribute_data,
):
    """
    Build both ID-based and name-based lookup tables.

    ID is preferred because names can theoretically collide.
    Name is retained as a careful fallback.
    """

    by_id = {}

    if (
        "Player Key Canonical"
        in attribute_data.columns
    ):

        keyed = attribute_data.dropna(
            subset=[
                "Player Key Canonical"
            ]
        )

        for _, row in keyed.iterrows():

            key = row[
                "Player Key Canonical"
            ]

            if key not in by_id:
                by_id[key] = row

    # Name fallback is allowed only for names that uniquely identify
    # one row in the attribute table.
    name_counts = (
        attribute_data[
            "Name"
        ]
        .value_counts()
    )

    unique_names = set(
        name_counts[
            name_counts == 1
        ].index
    )

    by_name = {}

    for _, row in attribute_data.iterrows():

        name = row[
            "Name"
        ]

        if name in unique_names:
            by_name[name] = row

    return (
        by_id,
        by_name,
    )


def attach_attributes_to_lineup(
    lineup,
    attribute_data,
    method_label,
):
    """
    Attach PAC/SHO/PAS/DRI/DEF/PHY to the already-selected lineup.

    Goalkeepers are excluded because these six broad outfield
    attributes are not a meaningful representation of goalkeeper
    contribution in this project.
    """

    outfield = (
        lineup[
            lineup[
                "Formation Slot"
            ]
            != "GK"
        ]
        .copy()
    )

    by_id, by_name = (
        build_attribute_lookups(
            attribute_data
        )
    )

    matched_records = []
    unmatched = []

    for _, lineup_row in outfield.iterrows():

        source_row = None
        match_source = None

        if (
            "Player Key"
            in outfield.columns
        ):

            key = canonical_player_key(
                lineup_row[
                    "Player Key"
                ]
            )

            if (
                key is not None
                and key in by_id
            ):
                source_row = by_id[
                    key
                ]

                match_source = (
                    "Player Key / ID"
                )

        if source_row is None:

            name = lineup_row[
                "Name"
            ]

            if name in by_name:
                source_row = by_name[
                    name
                ]

                match_source = (
                    "Unique Name Fallback"
                )

        if source_row is None:

            unmatched.append(
                {
                    "Formation":
                        lineup_row[
                            "Formation"
                        ],

                    "Name":
                        lineup_row[
                            "Name"
                        ],

                    "Method":
                        method_label,
                }
            )

            continue

        record = {
            "Method":
                method_label,

            "Formation":
                lineup_row[
                    "Formation"
                ],

            "Formation Slot":
                lineup_row[
                    "Formation Slot"
                ],

            "Name":
                lineup_row[
                    "Name"
                ],

            "Match Source":
                match_source,
        }

        if "Player Key" in outfield.columns:
            record[
                "Player Key"
            ] = lineup_row[
                "Player Key"
            ]

        for attribute in ATTRIBUTES:
            record[
                attribute
            ] = float(
                source_row[
                    attribute
                ]
            )

        matched_records.append(
            record
        )

    if unmatched:

        unmatched_df = pd.DataFrame(
            unmatched
        )

        raise ValueError(
            "Could not attach six-attribute data to some selected "
            "outfield players:\n\n"
            + unmatched_df.to_string(
                index=False
            )
        )

    attached = pd.DataFrame(
        matched_records
    )

    # Every formation should contain exactly ten outfield players.
    counts = (
        attached
        .groupby(
            "Formation"
        )
        .size()
    )

    bad_counts = counts[
        counts != 10
    ]

    if not bad_counts.empty:
        raise ValueError(
            "Expected exactly 10 matched outfield players per "
            f"formation for {method_label}, but found:\n"
            f"{bad_counts}"
        )

    return attached


# ============================================================
# LINEUP SUPPLY + ATTRIBUTE-LEVEL GAPS
# ============================================================

def calculate_lineup_supply(
    attached_lineup,
):
    """
    Calculate the selected XI's mean outfield attribute profile.

    Because every formation has ten outfield players, the mean is
    easier to interpret than a total and does not change the relative
    comparison.
    """

    supply = (
        attached_lineup
        .groupby(
            [
                "Method",
                "Formation",
            ]
        )[ATTRIBUTES]
        .mean()
        .reset_index()
    )

    return supply


def make_coverage_comparison(
    formation_demand,
    baseline_supply,
    role_adjusted_supply,
):
    """
    Produce one row per:

        Formation x Attribute

    with demand, each method's supply, signed gap, absolute gap,
    shortfall, and the change in absolute gap.
    """

    baseline_wide = (
        baseline_supply
        .drop(
            columns=[
                "Method"
            ]
        )
        .set_index(
            "Formation"
        )
    )

    role_wide = (
        role_adjusted_supply
        .drop(
            columns=[
                "Method"
            ]
        )
        .set_index(
            "Formation"
        )
    )

    demand_wide = (
        formation_demand
        .set_index(
            "Formation"
        )
    )

    rows = []

    for formation in FORMATIONS:

        for attribute in ATTRIBUTES:

            demand = float(
                demand_wide.loc[
                    formation,
                    attribute,
                ]
            )

            baseline_value = float(
                baseline_wide.loc[
                    formation,
                    attribute,
                ]
            )

            role_value = float(
                role_wide.loc[
                    formation,
                    attribute,
                ]
            )

            baseline_gap = (
                baseline_value
                - demand
            )

            role_gap = (
                role_value
                - demand
            )

            rows.append(
                {
                    "Formation":
                        formation,

                    "Attribute":
                        attribute,

                    "Formation Demand":
                        demand,

                    "OVR Baseline Supply":
                        baseline_value,

                    "Role-Adjusted Supply":
                        role_value,

                    "OVR Baseline Gap":
                        baseline_gap,

                    "Role-Adjusted Gap":
                        role_gap,

                    "OVR Baseline Absolute Gap":
                        abs(
                            baseline_gap
                        ),

                    "Role-Adjusted Absolute Gap":
                        abs(
                            role_gap
                        ),

                    "OVR Baseline Shortfall":
                        max(
                            demand
                            - baseline_value,
                            0.0,
                        ),

                    "Role-Adjusted Shortfall":
                        max(
                            demand
                            - role_value,
                            0.0,
                        ),

                    # Positive means the role-adjusted lineup is
                    # closer to the empirical formation-demand value
                    # on this attribute.
                    "Absolute Gap Improvement":
                        abs(
                            baseline_gap
                        )
                        - abs(
                            role_gap
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLOTS
# ============================================================

def plot_formation_profiles(
    formation_demand,
    baseline_supply,
    role_adjusted_supply,
):
    """
    Save one plot per formation.

    We intentionally use separate figures rather than a crowded
    multi-panel chart so each formation can be inspected clearly.
    """

    demand_wide = (
        formation_demand
        .set_index(
            "Formation"
        )
    )

    baseline_wide = (
        baseline_supply
        .set_index(
            "Formation"
        )
    )

    role_wide = (
        role_adjusted_supply
        .set_index(
            "Formation"
        )
    )

    for formation in FORMATIONS:

        demand_values = [
            demand_wide.loc[
                formation,
                attribute,
            ]
            for attribute
            in ATTRIBUTES
        ]

        baseline_values = [
            baseline_wide.loc[
                formation,
                attribute,
            ]
            for attribute
            in ATTRIBUTES
        ]

        role_values = [
            role_wide.loc[
                formation,
                attribute,
            ]
            for attribute
            in ATTRIBUTES
        ]

        plt.figure(
            figsize=(
                9,
                5.5,
            )
        )

        plt.plot(
            ATTRIBUTES,
            demand_values,
            marker="o",
            linewidth=2,
            label="Formation demand",
        )

        plt.plot(
            ATTRIBUTES,
            baseline_values,
            marker="o",
            linewidth=2,
            label="OVR baseline XI",
        )

        plt.plot(
            ATTRIBUTES,
            role_values,
            marker="o",
            linewidth=2,
            label=(
                "Role-adjusted XI "
                f"(lambda={REFERENCE_LAMBDA:.2f})"
            ),
        )

        plt.title(
            (
                "Stage 4.4A Team Attribute Coverage — "
                f"{formation}"
            )
        )

        plt.xlabel(
            "EAFC broad attribute"
        )

        plt.ylabel(
            "Mean outfield attribute value"
        )

        plt.ylim(
            0,
            100,
        )

        plt.grid(
            alpha=0.25,
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR
            / (
                "team_attribute_coverage_"
                f"{safe_filename(formation)}.png"
            ),
            dpi=180,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# PRINTING
# ============================================================

def print_role_profiles(
    role_profiles,
):
    print(
        "\n"
        + "=" * 92
    )

    print(
        "EMPIRICAL FUNCTIONAL-ROLE ATTRIBUTE PROFILES"
    )

    print(
        "=" * 92
    )

    print(
        role_profiles[
            [
                "Functional Role",
                "Number of Players",
                *ATTRIBUTES,
            ]
        ]
        .round(
            2
        )
        .to_string(
            index=False
        )
    )


def print_formation_demand(
    formation_demand,
):
    print(
        "\n"
        + "=" * 92
    )

    print(
        "DATA-DERIVED FORMATION DEMAND PROFILES"
    )

    print(
        "=" * 92
    )

    print(
        formation_demand
        .round(
            2
        )
        .to_string(
            index=False
        )
    )


def print_coverage_by_formation(
    coverage_comparison,
):
    for formation in FORMATIONS:

        subset = (
            coverage_comparison[
                coverage_comparison[
                    "Formation"
                ]
                == formation
            ]
            .copy()
        )

        print(
            "\n"
            + "=" * 92
        )

        print(
            "TEAM ATTRIBUTE COVERAGE — "
            f"{formation}"
        )

        print(
            "=" * 92
        )

        print(
            subset[
                [
                    "Attribute",
                    "Formation Demand",
                    "OVR Baseline Supply",
                    "Role-Adjusted Supply",
                    "OVR Baseline Gap",
                    "Role-Adjusted Gap",
                    "Absolute Gap Improvement",
                ]
            ]
            .round(
                2
            )
            .to_string(
                index=False
            )
        )


# ============================================================
# RUN
# ============================================================

def main():

    attribute_data = (
        load_attribute_data()
    )

    baseline, role_adjusted = (
        load_lineups()
    )

    # --------------------------------------------------------
    # 1. Empirical functional-role profiles
    # --------------------------------------------------------

    role_profiles = (
        calculate_functional_role_profiles(
            attribute_data
        )
    )

    # --------------------------------------------------------
    # 2. Formation demand profiles
    # --------------------------------------------------------

    (
        formation_demand,
        slot_demand,
    ) = calculate_formation_demand(
        role_profiles
    )

    # --------------------------------------------------------
    # 3. Attach six broad attributes to EXISTING lineups
    # --------------------------------------------------------

    baseline_attached = (
        attach_attributes_to_lineup(
            baseline,
            attribute_data,
            "OVR Baseline",
        )
    )

    role_adjusted_attached = (
        attach_attributes_to_lineup(
            role_adjusted,
            attribute_data,
            (
                "Role-Adjusted Quality "
                f"(lambda={REFERENCE_LAMBDA:.2f})"
            ),
        )
    )

    # --------------------------------------------------------
    # 4. Calculate selected-lineup attribute supply
    # --------------------------------------------------------

    baseline_supply = (
        calculate_lineup_supply(
            baseline_attached
        )
    )

    role_adjusted_supply = (
        calculate_lineup_supply(
            role_adjusted_attached
        )
    )

    # --------------------------------------------------------
    # 5. Attribute-by-attribute coverage comparison
    # --------------------------------------------------------

    coverage_comparison = (
        make_coverage_comparison(
            formation_demand,
            baseline_supply,
            role_adjusted_supply,
        )
    )

    # --------------------------------------------------------
    # 6. Save outputs
    # --------------------------------------------------------

    role_profiles.to_csv(
        RESULTS_DIR
        / "empirical_functional_role_profiles.csv",
        index=False,
    )

    slot_demand.to_csv(
        RESULTS_DIR
        / "formation_slot_demand_profiles.csv",
        index=False,
    )

    formation_demand.to_csv(
        RESULTS_DIR
        / "formation_attribute_demand_profiles.csv",
        index=False,
    )

    baseline_attached.to_csv(
        RESULTS_DIR
        / "ovr_baseline_players_with_attributes.csv",
        index=False,
    )

    role_adjusted_attached.to_csv(
        RESULTS_DIR
        / "role_adjusted_players_with_attributes.csv",
        index=False,
    )

    baseline_supply.to_csv(
        RESULTS_DIR
        / "ovr_baseline_attribute_supply.csv",
        index=False,
    )

    role_adjusted_supply.to_csv(
        RESULTS_DIR
        / "role_adjusted_attribute_supply.csv",
        index=False,
    )

    coverage_comparison.to_csv(
        RESULTS_DIR
        / "baseline_vs_role_adjusted_attribute_coverage.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. Visualize
    # --------------------------------------------------------

    plot_formation_profiles(
        formation_demand,
        baseline_supply,
        role_adjusted_supply,
    )

    # --------------------------------------------------------
    # 8. Print the main diagnostic tables
    # --------------------------------------------------------

    print_role_profiles(
        role_profiles
    )

    print_formation_demand(
        formation_demand
    )

    print_coverage_by_formation(
        coverage_comparison
    )

    print(
        "\n"
        + "=" * 92
    )

    print(
        "STAGE 4.4A COMPLETE"
    )

    print(
        "=" * 92
    )

    print(
        "\nInterpretation:"
    )

    print(
        "Positive gap = selected lineup is above the "
        "data-derived formation benchmark on that attribute."
    )

    print(
        "Negative gap = selected lineup is below the "
        "data-derived formation benchmark on that attribute."
    )

    print(
        "Positive Absolute Gap Improvement = the role-adjusted XI "
        "is closer to the empirical formation profile than the "
        "OVR-only XI on that attribute."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This stage is diagnostic only. It does not yet define or "
        "optimize a single complementarity/chemistry score."
    )

    print(
        "\nPlots saved to:"
    )

    print(
        PLOT_DIR
    )


if __name__ == "__main__":
    main()
