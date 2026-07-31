# Soccer_analytics-UW_PRIME-Su-26

Building the best soccer team using data and math!

There are so many legendary soccer players who have shaped how people play soccer today. From Ronaldinho, Pelé, Kaká, to Messi, C.Ronaldo, Dembélé, and Mbappé. The list goes on and on. This project is to mathematically analyze players’ stats and try to see what factors team managers and coaches need to consider when choosing players for a team.

The naive way of creating the best team will be to pick the best player ever from each position. This approach will definitely provide valuable insights because you always want to choose the good players to play. The issue with this method is that having too many good players on the same team might unfortunately interfere with their performance, as everyone wants to shoot, score, and be the leader. Take the Real Madrid team for this season as an example. So I am curious to explore ways in which one can assess players and how to assess team performance.

# Project brainstorming

First, I build a list of 50 important factors that I think will determine whether a player is good or not. Then I pick the top 10 most relevant and easily measured variables out of the 50.

Here are the top 10 factors that I have listed.

## Top 10 factors for determining a good player under a team setting:

### 1. Overall player quality

I think it is important to have a first-round exclusion criterion on who we should pick for these players.

For the current stage, I am only using the FC26 data, a rating system made by FIFA.

Here is the link: [FC26_data_explained](https://www.fcratings.com/articles/ea-sports-fc-26-attributes-explained) and [FC26_data](https://www.ea.com/games/ea-sports-fc/ratings)

1. Pace (PAC): Measures acceleration and sprint speed.
2. Shooting (SHO): Evaluates finishing, positioning, shot power, and long shots.
3. Passing (PAS): Combines vision, crossing, curve, free kicks, long passing, and short passing accuracy.
4. Dribbling (DRI): Factors in agility, balance, reactions, ball control, dribbling, and composure.
5. Defending (DEF): Assesses defensive awareness, heading, interceptions, standing tackle, and sliding tackle.
6. Physicality (PHY): Measures strength, stamina, aggression, and jumping
7. Goalkeeping (CK): Looks at diving, handling, kicking, positioning, and reflexes

**NOTE:**

1. We need to consider injury when making this data.
2. different position emphasizes on different skills, and so maybe we should break down this rating system a bit more.

### 2. Most played position

We need a way to at least classify players into some categories so it will be easier for the future steps to choose players. After all, we can’t have 11 strikers on the team.

Since a lot of players can play multiple positions, setting a “fixed” position for a player will impact the player’s performance to some degree. So for now, I am going to rate these positions from most frequent to least frequent throughout the season.

### 3. Passing relationship between players

Specifically look at the passing rate for each player and order them from high to low.

**Note:**

We may want to exclude certain positions, like goalie and the striker, because those position weight less on the striking attribute.

### 4. Minutes (or matches) played together

This might be the easiest way to test for chemistry between players, answering questions like whether they are close to each other; maybe from the team they were playing at could infer what type of management systems and playing styles these players like.

I will define two attributes: same team vs different team.

My thought is to represent this factor in a node graph with clustering. I will connect the players with edges if they played with/against each other, and the number on the edge will represent (played time, played against time)

### 5. Playing style compatibility

This is the most confusing factor for the list, and the reason is that there doesn’t exist a perfect matching style list between players. One thing we could use this variable for is to experiment whether we should pick all players with the same style, all with different styles, or the middle. Experimenting a bit on this part will be valuable.

**Note:**

There are a lot of constraints for this factor because a player’s playing style could vary from time to time, with different coaches and clubs, and with different positions.

Link for FC26 player styles: [FC26_player_styles](https://www.fcratings.com/articles/what-are-playstyles-in-ea-sports-fc-26)


### 6. Attacking contribution [potential skip]
### 7. Defensive contribution [potential skip]
### 8. Passing network edge weight / connectivity [potential skip]

### 9. Team balance

Here we want to ensure diversity of the team, in that the model doesn’t just pick only good strikers who can score. We want a mix of attacking, defensive, and positional roles.

### 10. Injury 

This is probably the most important factor in the sense that we do want to keep track of the injuries that players have and whether they are critical injuries or not.

# Helpful links

1. [FC26_most_popular_player_per_position](https://www.fut.gg/tactics/most-used-players/)

2. [Squad_rating](https://fifauteam.com/fc-26-squad-rating-guide/)