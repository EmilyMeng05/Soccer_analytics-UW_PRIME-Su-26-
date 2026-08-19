# Stage 3.7 — Goalkeeper Style Analysis

## Goal

The earlier stages focused on outfield players and used attributes such as pace, shooting, passing, dribbling, defending, and physicality.

Goalkeepers were intentionally excluded because they perform a very different role and have their own specialized EAFC attributes.

Stage 3.7 asks:

> What goalkeeper styles appear in the data, and which goalkeeper would best complement a particular outfield lineup?

The goal is not simply to select the goalkeeper with the highest OVR. A team may benefit from a goalkeeper who provides stronger protection, stronger distribution, or a balance between both.


## Dataset

Stage 3.7 uses the cleaned goalkeeper dataset:

`cleaned_eafc26_goalkeepers.csv`

The dataset contains:

`1,274 goalkeepers`

and:

`59 columns`

No goalkeepers were removed because all 1,274 players had values for the five required goalkeeper attributes.


## Goalkeeper Attributes

The analysis uses five specialized EAFC attributes:

- GK Diving
- GK Handling
- GK Kicking
- GK Positioning
- GK Reflexes

These attributes describe different parts of a goalkeeper's job.


### GK Diving

GK Diving measures how well a goalkeeper can launch their body toward a shot and reach the ball.

For example, if an attacker shoots toward the corner of the goal, a goalkeeper with strong diving ability has a better chance of reaching the shot.


### GK Handling

GK Handling measures how securely the goalkeeper catches or controls the ball.

A goalkeeper with strong handling is less likely to drop a catch or push the ball into a dangerous area where an attacker could shoot again.


### GK Kicking

GK Kicking measures the goalkeeper's ability to pass or clear the ball over longer distances.

It does **not** mean that the goalkeeper takes attacking shots.

A goalkeeper with strong kicking can:

- Deliver accurate goal kicks.
- Pass to distant teammates.
- Start counter-attacks quickly.
- Move the ball past an opponent's pressure.


### GK Positioning

GK Positioning measures how well the goalkeeper places themselves before a shot or dangerous play occurs.

Good positioning can make a difficult save easier because the goalkeeper is already standing in a useful location.


### GK Reflexes

GK Reflexes measures how quickly a goalkeeper reacts to sudden shots, rebounds, or changes in the ball's direction.

This is especially important when an attacker shoots from a short distance and the goalkeeper has very little time to respond.


## Protection Contribution

Protection Contribution represents the goalkeeper's ability to prevent goals and safely control dangerous situations near the goal.

It is calculated as:

`Protection Contribution = average(GK Diving, GK Handling, GK Positioning, GK Reflexes)`

These four attributes represent different stages of defending the goal:

1. Positioning helps the goalkeeper prepare for the threat.
2. Reflexes help the goalkeeper react quickly.
3. Diving helps the goalkeeper reach the shot.
4. Handling helps the goalkeeper catch or safely control the ball.

A goalkeeper with high Protection Contribution may be especially helpful when the outfield team:

- Has weaker defenders.
- Allows opponents to take more shots.
- Plays against strong attacking teams.
- Needs the goalkeeper to provide additional defensive security.

Protection does not mean the goalkeeper guarantees that no goals will be conceded. It means their EAFC attributes suggest stronger shot-stopping and ball-control ability.


## Distribution Contribution

Distribution Contribution represents the goalkeeper's ability to move the ball from the defensive end of the field to teammates and begin an attack.

In this first version, it is defined as:

`Distribution Contribution = GK Kicking`

When a goalkeeper gains possession, the team's attack can begin with them. They may pass to a nearby defender, kick toward a midfielder, or send a long ball toward an attacker.

A goalkeeper with high Distribution Contribution can help the team:

- Move the ball forward quickly.
- Begin counter-attacks.
- Keep possession when opponents press near the goal.
- Send accurate long passes toward attacking players.
- Increase the speed of transitions from defense to attack.

Distribution is therefore connected to attacking buildup, even though the goalkeeper normally remains near their own goal.

A strong offensive or possession-focused team may benefit from a goalkeeper with strong distribution because the goalkeeper becomes the first passer in many attacking sequences.


## Protection Versus Distribution

Protection and distribution describe two different ways a goalkeeper can contribute to the team.

| Contribution | Main purpose | Important attribute |
|---|---|---|
| Protection | Prevent and safely control scoring threats | Diving, Handling, Positioning, Reflexes |
| Distribution | Move the ball to teammates and begin attacks | Kicking |

A goalkeeper can be strong in one dimension, both dimensions, or neither dimension relative to the dataset averages.

The purpose of this comparison is not to say that protection and distribution are opposites. An elite goalkeeper can be strong in both. Instead, the comparison helps show where each goalkeeper's relative strengths are concentrated.


## Dataset Averages

The average goalkeeper scores were:

`Average Protection Contribution = 68.06`

`Average Distribution Contribution = 65.57`

These averages divide the interactive graph into four broad regions.


## Goalkeeper Style Groups

| Goalkeeper Style | Number of Players | Meaning |
|---|---:|---|
| High Protection and Distribution | 452 | Above average in both dimensions |
| Protection Focused | 110 | Above-average protection and below-average distribution |
| Distribution Focused | 129 | Above-average distribution and below-average protection |
| Lower Both | 583 | Below average in both dimensions |


### High Protection and Distribution

These goalkeepers are above the dataset average in both dimensions.

Examples include:

- Alisson
- David Raya
- Thibaut Courtois
- Yann Sommer
- Marc-André ter Stegen
- Mike Maignan
- Ederson
- Gianluigi Donnarumma

This category does not automatically mean a goalkeeper is perfectly balanced.

For example, Donnarumma has:

`Protection Contribution = 87.50`

`Distribution Contribution = 70.00`

Both values are above their dataset averages, so he belongs to the high-both group. However, the difference between his two scores shows that his profile still leans strongly toward protection.


### Protection Focused

These goalkeepers have above-average Protection Contribution but below-average Distribution Contribution.

Gregor Kobel is a strong example:

`Protection Contribution = 85.75`

`Distribution Contribution = 64.00`

His profile suggests strong shot-stopping ability but weaker long distribution relative to the other leading goalkeepers.


### Distribution Focused

These goalkeepers have above-average Distribution Contribution but below-average Protection Contribution.

They may help move the ball forward but provide less defensive protection than the average goalkeeper under the current measurements.

This group should not be interpreted as the list of the best distributors overall. Elite distributors such as Ederson and ter Stegen are above average in both dimensions, so they belong to the high-both group instead.


### Lower Both

These goalkeepers are below the dataset average in both dimensions.

This does not mean they have no useful skills. It only means their current Protection and Distribution scores fall below the averages calculated across all 1,274 goalkeepers.


## Goalkeeper Skill

Overall Goalkeeper Skill is calculated using all five specialized attributes:

`Goalkeeper Skill = average(Diving, Handling, Kicking, Positioning, Reflexes)`

OVR remains separate because EAFC's OVR already provides its own general evaluation of a player.

The leading overall goalkeeper results were approximately:

| Goalkeeper | OVR | Protection | Distribution | Goalkeeper Skill |
|---|---:|---:|---:|---:|
| Alisson | 89 | 87.50 | 86 | 87.20 |
| David Raya | 87 | 85.50 | 87 | 85.80 |
| Thibaut Courtois | 89 | 88.00 | 76 | 85.60 |
| Jan Oblak | 88 | 87.00 | 78 | 85.20 |
| Yann Sommer | 87 | 85.25 | 85 | 85.20 |
| Marc-André ter Stegen | 86 | 84.25 | 89 | 85.20 |
| Mike Maignan | 87 | 84.75 | 85 | 84.80 |
| Ederson | 85 | 82.75 | 91 | 84.40 |
| Gianluigi Donnarumma | 89 | 87.50 | 70 | 84.00 |
| Manuel Neuer | 84 | 82.25 | 90 | 83.80 |

These results show why goalkeeper selection should not use OVR alone. Goalkeepers with similar OVR values may have very different protection and distribution profiles.


## Style Gap

Style Gap measures the absolute difference between protection and distribution:

`Style Gap = |Protection Contribution − Distribution Contribution|`

A small Style Gap means the two contribution scores are similar.

For example:

`Protection = 85`

`Distribution = 85`

produces:

`Style Gap = 0`

However, Style Gap does not measure overall quality.

A goalkeeper with Protection = 60 and Distribution = 60 also has a gap of zero. Therefore, balanced goalkeeper selection should consider both a small Style Gap and a high Goalkeeper Skill score.

Strong balanced candidates include:

- Alisson
- David Raya
- Yann Sommer
- Mike Maignan
- Diogo Costa


## Principal Component Analysis

PCA reduces the five standardized goalkeeper attributes into a smaller number of mathematical dimensions.

The first two components explain:

| Component | Explained Variation |
|---|---:|
| PC1 | 85.06% |
| PC2 | 7.61% |
| PC1 + PC2 | 92.67% |

Because PC1 and PC2 explain approximately 92.7% of the variation, the two-dimensional PCA graph preserves most of the information from the five goalkeeper attributes.


## Interpreting PC1

The PC1 loadings were:

| Attribute | PC1 Loading |
|---|---:|
| GK Diving | 0.459 |
| GK Handling | 0.456 |
| GK Kicking | 0.403 |
| GK Positioning | 0.459 |
| GK Reflexes | 0.457 |

All five attributes have similar positive contributions.

Therefore, PC1 can be interpreted as:

**Overall Goalkeeper Quality**

A goalkeeper with a high PC1 score tends to have strong ratings across most or all goalkeeper attributes.


## Interpreting PC2

The PC2 loadings were:

| Attribute | PC2 Loading |
|---|---:|
| GK Diving | −0.311 |
| GK Handling | −0.084 |
| GK Kicking | 0.893 |
| GK Positioning | −0.089 |
| GK Reflexes | −0.301 |

GK Kicking has a very strong positive loading, while Diving and Reflexes have negative loadings.

Therefore, PC2 can be interpreted approximately as:

**Distribution versus Shot-Stopping Style**

The observed direction is:

`Negative PC2 ← Protection and Shot-Stopping`

`Positive PC2 → Distribution and Kicking`

Examples of positive-PC2 distributors include:

- Ederson
- Manuel Neuer
- Marc-André ter Stegen
- Jordan Pickford

Examples of negative-PC2 protectors include:

- Gregor Kobel
- Gianluigi Donnarumma
- Thibaut Courtois
- David de Gea


## Interactive Protection–Distribution Website

The interactive website plots:

`X-axis = Distribution Contribution`

`Y-axis = Protection Contribution`

The vertical line represents average distribution, and the horizontal line represents average protection.

The four graph regions can be interpreted as:

- Upper-right: above average in protection and distribution.
- Upper-left: protection-focused.
- Lower-right: distribution-focused.
- Lower-left: below average in both dimensions.

Hovering over a point shows the goalkeeper's name, team, OVR, individual attributes, contribution scores, and assigned style.

The website also supports zooming, panning, and hiding or displaying goalkeeper groups through the legend.


## Goalkeeper Similarity

PCA is used for visualization, but goalkeeper similarity is calculated using all five standardized goalkeeper attributes.

The `AttributeDistance` value measures the mathematical distance between two goalkeeper profiles:

`Smaller distance = more similar attribute profiles`

Examples include:

- Alisson is most similar to Yann Sommer and David Raya.
- Courtois is most similar to Jan Oblak.
- Ederson is most similar to ter Stegen and Manuel Neuer.
- Donnarumma is most similar to Marco Carnesecchi and Gregor Kobel.

These results support the interpretation that Ederson belongs to a distribution-oriented style, while Donnarumma and Kobel lean more strongly toward protection.


## Important Limitation of the Style Tables

The Protection Focused table contains only players from the protection-focused quadrant. It does not list every goalkeeper with a high Protection Contribution.

For example, Courtois has extremely high protection, but he belongs to the high-both group because his distribution is also above the dataset average.

The same applies to the Distribution Focused table. Ederson, Neuer, and ter Stegen are among the strongest distributors, but they belong to the high-both group because their protection is also above average.

Therefore, the analysis distinguishes between:

- Absolute strength in a contribution.
- A player's relative style or specialization.


## Connection to Team Building

Goalkeeper selection should depend partly on the attributes of the selected outfield team.


### Team With Weaker Defense

If the selected outfield lineup has relatively weak defending, the goalkeeper score can place greater emphasis on:

- Protection Contribution
- Diving
- Reflexes
- Positioning
- Handling

Possible candidates include Courtois, Donnarumma, Oblak, and Kobel.


### Strong Offensive or Possession Team

If the team has strong attacking players and wants to move the ball forward quickly, the goalkeeper score can place greater emphasis on:

- Distribution Contribution
- GK Kicking
- Positive PC2 style

Possible candidates include Ederson, Neuer, ter Stegen, and David Raya.


### Balanced Team

If the outfield lineup is already balanced, the goalkeeper score can emphasize:

- Overall Goalkeeper Skill
- Strong protection and distribution
- A relatively small Style Gap

Possible candidates include Alisson, David Raya, Yann Sommer, and Mike Maignan.


## Main Takeaway

Stage 3.7 shows that goalkeeper selection involves more than choosing the player with the highest OVR.

PCA discovered two clear patterns:

`PC1 = Overall Goalkeeper Quality`

`PC2 = Distribution versus Protection Style`

The contribution analysis also provides two interpretable measurements:

`Protection Contribution = ability to stop and safely control threats`

`Distribution Contribution = ability to move the ball to teammates and begin attacks`

These results allow Stage 4 to select a goalkeeper whose strengths complement the selected outfield lineup rather than using one goalkeeper automatically for every team.


## Output Files

The main expanded goalkeeper dataset is:

`eafc26_goalkeepers_with_styles.csv`

The interactive goalkeeper website is:

`interactive_goalkeeper_protection_distribution.html`

Additional results include:

- PCA explained variance
- PCA loadings
- Overall goalkeeper rankings
- Protection-focused goalkeeper rankings
- Distribution-focused goalkeeper rankings
- High protection-and-distribution goalkeepers
- Balanced goalkeeper profiles
- Similar-goalkeeper results


## Next Stage

Stage 4 will combine:

- The expanded outfield-player dataset from Stage 3.6.
- The expanded goalkeeper dataset from Stage 3.7.

The goal will be to construct one fixed squad that can support the 4-3-3, 4-2-3-1, and 4-4-2 formations while using flexible players and a complementary goalkeeper.

