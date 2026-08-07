# Soccer_analytics-UW_PRIME-Su-26

Building the best soccer team using data and math!

There are so many legendary soccer players who have shaped how people play soccer today. From Ronaldinho, Pelé, Kaká, to Messi, Cristiano Ronaldo, Dembélé, and Mbappé, the list goes on and on. Every generation has players who stand out because of their incredible individual talent.

A natural question then becomes:

> **If we could choose any eleven players in the world, what would be the best possible soccer team?**

The naive approach would be to simply pick the best player at every position. While this definitely provides valuable insights because you always want talented players on your team, I don't think this tells the whole story.

Soccer is a team sport. Having too many players with similar strengths may actually reduce the team's overall performance. Everyone wants to attack, everyone wants to score, and not every player naturally complements one another. Recent teams like Real Madrid have shown that even a roster full of world-class players can struggle to find the right balance.

So instead of asking,

> **"Who are the best players?"**

I want to ask,

> **"How can we mathematically build the best team?"**

Rather than focusing only on player ratings, I want to investigate how player attributes, positions, chemistry, playing styles, and team diversity all contribute to creating a successful lineup.

# Project Overview

Instead of trying to solve the entire problem at once, I divide this project into four stages.

## Stage 1. Representing Each Player

The first step is to represent every player using measurable attributes.

For the current version of this project, I mainly use the FC26 player ratings. These ratings summarize different aspects of a player's abilities.

Current attributes include:

- Pace (PAC)
- Shooting (SHO)
- Passing (PAS)
- Dribbling (DRI)
- Defending (DEF)
- Physicality (PHY)
- Goalkeeping (GK)

In addition to these ratings, I also plan to include information such as

- Preferred positions
- Playing styles
- Club
- Minutes played
- Injury history

The goal of this stage is to convert every player into a numerical feature vector that can later be analyzed mathematically.

Useful links: 

[FC26_data_explain](https://www.fcratings.com/articles/ea-sports-fc-26-attributes-explained)

[FC26_data](https://www.ea.com/games/ea-sports-fc/ratings)

## Stage 2. Understanding Player Similarity

Once every player is represented numerically, the next step is understanding how similar different players actually are.

Instead of manually deciding whether two players have similar playing styles, I want to let the data and machine learning models answer that question.

Some techniques I plan to explore include:

- Correlation analysis 
- Principal Component Analysis (PCA)
- Similarity metrics
- Data visualization

These methods will help reveal which player attributes explain the largest differences between players while also making the data easier to visualize.

## Stage 3. Discovering Player Archetypes

After understanding player similarity, I want to group players into different playing archetypes using clustering algorithms.

Instead of simply saying that Messi and Neymar are similar, I want to mathematically identify groups such as

- Creative playmakers
- Clinical finishers
- Ball-winning midfielders
- Attacking fullbacks
- Physical center backs

The exact groups will depend on what the clustering algorithm discovers from the data.

One idea I am excited to explore is whether selecting multiple players from the same cluster creates redundancy within a team, while selecting players from different clusters produces a more balanced lineup.

## Stage 4. Building the Best Team

Finally, I combine everything together.

Instead of selecting players solely based on their overall ratings, I want to build a team that balances multiple factors simultaneously.

Some of the factors I plan to investigate include

- Overall player quality
- Position constraints
- Playing style diversity
- Team chemistry
- Passing relationships
- Minutes played together

The goal is to compare different team-building strategies and investigate whether diversity and chemistry can outperform simply selecting the highest-rated players.

# Candidate Features

Below are the different features I have brainstormed for this project. Some of these will definitely be included, while others are ideas I would like to experiment with in future versions.

## 1. Overall Player Quality

I think it is important to have a first-round exclusion criterion on who we should pick for the team.

For the current stage, I am using the FC26 player ratings made by EA Sports for getting the overall player quality.

Current thoughts:

- Injury history should probably be considered in future versions.
- Different positions emphasize different skills, so it may make sense to break these ratings down further.


## 2. Most Played Position

We need a way to classify players into positions because we obviously cannot have eleven strikers on the same team.

Since many players can play multiple positions, I plan to rank positions based on how frequently each player has played them throughout the season instead of assigning only one fixed position.


## 3. Passing Relationships

Specifically, I want to examine passing frequency between players.

Questions I would like to investigate include:

- Which players naturally pass to each other?
- Do highly connected passing pairs perform better together?
- Should passing chemistry influence team selection?

## 4. Minutes (or Matches) Played Together

This might be one of the easiest ways to measure chemistry.

Players who have spent more time together may already understand each other's movement and playing style.

One idea is to represent this as a graph where

- Nodes represent players.
- Edges represent shared playing time.
- Edge weights represent minutes played together.

I also want to distinguish between players who have

- Played on the same team.
- Frequently played against one another.


## 5. Playing Style Compatibility

This is probably the hardest feature to define.

There doesn't seem to be a perfect mathematical definition of playing style compatibility, which is exactly why I think it is interesting.

Instead of manually deciding which styles work together, I hope the clustering step can help answer questions such as

- Should we build a team with very similar players?
- Should every player have a completely different style?
- Is the best team somewhere in the middle?

Current thought:

A player's style changes depending on their coach, club, teammates, and even their position, so this feature will probably require the most experimentation.

Useful link:

[FC26_player_style](https://www.fcratings.com/articles/what-are-playstyles-in-ea-sports-fc-26)

## 6. Attacking Contribution *(Potential Future Feature)*

Measure how much each player contributes to attacking opportunities beyond simply scoring goals.

This is a particularly important skill to consider when placing a player in the attacking team.

## 7. Defensive Contribution *(Potential Future Feature)*

Measure how much each player contributes defensively, even if they are not traditional defenders.

This is a particularly important skill to consider when placing a player in the defensing team.

## 8. Team Balance

One of the main goals of this project is to avoid selecting eleven individually great players who all perform the same role.

Instead, I want to investigate whether mathematically encouraging diversity leads to stronger teams.


## 9. Injury History

This is probably one of the most important real-world factors.

Even if a player is world-class, frequent injuries may significantly affect their overall value to the team.