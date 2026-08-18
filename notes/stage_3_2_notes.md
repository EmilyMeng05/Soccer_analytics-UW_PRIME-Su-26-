# Stage 3.2 — Attacking and Defensive Player Profiles

## Goal

Stage 3.1 showed that the strongest natural clustering pattern roughly separates attack-oriented and defense-oriented players.

However, clustering forces every player into a discrete group.

In reality, soccer ability exists on a spectrum.

A midfielder, for example, may be strong both offensively and defensively.

Therefore, Stage 3.2 asks:

> How strong is each player offensively and defensively?

Rather than assigning players to clusters, each player is placed in a two-dimensional attacking-versus-defensive space.


## Attacking Contribution

The initial attacking score is defined as:

`Attacking Contribution = (PAC + SHO + PAS + DRI) / 4`

This includes:

- Pace
- Shooting
- Passing
- Dribbling

These four attributes broadly represent offensive contribution.


## Defensive Contribution

The initial defensive score is defined as:

`Defensive Contribution = (DEF + PHY) / 2`

This includes:

- Defending
- Physicality

These broadly represent defensive contribution.


## Why Use a 2D Representation?

Instead of assigning players a single label such as "attacker" or "defender", every player receives two scores:

`(Attacking Contribution, Defensive Contribution)`

This allows us to distinguish players who are:

- strong attackers
- strong defenders
- strong at both
- weak at both

Most importantly, it gives us a way to identify midfielders and other versatile players who contribute on both sides of the game.


## Player Categories

The two-dimensional space can broadly be interpreted as four regions:

### High Attack + High Defense

Players who contribute strongly in both dimensions.

These players may be especially useful for versatile or box-to-box roles.


### High Attack + Low Defense

Attack-oriented players.

Examples may include strikers, wingers, and attacking midfielders.


### Low Attack + High Defense

Defense-oriented players.

Examples may include center backs and defensive players.


### Low Attack + Low Defense

Players who are below the selected thresholds in both dimensions.


## Balanced Players

An additional measure is:

`Balance Gap = |Attack - Defense|`

A smaller balance gap means the player's attacking and defensive scores are more similar.

However, balance alone does NOT mean that a player is strong.

For example:

`Attack = 50, Defense = 50`

is more mathematically balanced than:

`Attack = 85, Defense = 80`

but the second player is much stronger overall.

Therefore, balance should be considered together with player quality.


## Overall Quality

A simple combined measure is:

`Overall Quality = (Attack + Defense) / 2`

This allows us to identify players who are:

1. relatively balanced, and
2. strong in both dimensions.

These players are better described as:

**Players strong in both attacking and defending**

rather than simply "balanced players."


## Visualization

Players are plotted on a two-dimensional graph:

- X-axis = Attacking Contribution
- Y-axis = Defensive Contribution

Each point represents one player.

An interactive version of the graph is also created so individual players can be explored.

To avoid overcrowding the graph, only selected top players are highlighted or labeled.


## Why This Stage Is Different From Stage 3.1

Stage 3.1 asks:

> What groups naturally exist?

Stage 3.2 asks:

> Where does each player fall on an attacking-defensive spectrum?

Therefore:

`Stage 3.1 = unsupervised grouping`

while:

`Stage 3.2 = interpretable player scoring`


## Limitation

The attacking and defensive formulas are intentionally simple.

For example:

`Attack = (PAC + SHO + PAS + DRI) / 4`

assumes that pace, shooting, passing, and dribbling contribute equally.

Similarly:

`Defense = (DEF + PHY) / 2`

assumes defending and physicality contribute equally.

These weights are not learned from data.

Therefore, Stage 3.2 should be treated as an interpretable baseline rather than a final measurement of soccer ability.


## Main Takeaway

Stage 3.2 provides a simple way to visualize player versatility.

Instead of forcing players into either an attacking or defensive group, we can identify players who perform strongly in both dimensions.

This becomes important for later stages because players should not necessarily be restricted to their listed EAFC positions.