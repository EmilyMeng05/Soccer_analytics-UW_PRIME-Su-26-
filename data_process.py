import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# read the dataset
df = pd.read_csv("EAFC26-Men.csv")

# Get a sense of what is the rating distribution like for players 
plt.figure(figsize=(8,5))

plt.hist(df["OVR"], bins=20)

plt.xlabel("Overall Rating")
plt.ylabel("Number of Players")
plt.title("Distribution of All Players")
plt.savefig('before_rating_num players.png', dpi=300, bbox_inches='tight')

plt.show()

# PART 1: general FIFA score
# for this project, we will label elite players with a general score of above 70
# Here is the link: https://fifauteam.com/fc-26-squad-rating-guide/
# This website contains a general FC squad rating system

# based on the previous graph, the median score is around 68-70
# so we do want to keep the data that are slightly below 70
# This cut off point is different than the cut off point listed on the website
elite = df[df["OVR"] > 60]

# number of elite players we have (4048, 59)
# print(elite.shape)

# Out of these elite players, who are among the top
top20 = (
    elite
    .sort_values("OVR", ascending=False)
    [["Name","Position","Team","OVR"]]
)

top20.to_csv("top_players.csv", index=False)

# Have the full name for each position 
position_map = {

    "GK":"Goalkeeper",

    "CB":"Center Back",

    "LB":"Left Back",

    "RB":"Right Back",

    "LWB":"Left Wing Back",

    "RWB":"Right Wing Back",

    "CDM":"Defensive Midfielder",

    "CM":"Central Midfielder",

    "CAM":"Attacking Midfielder",

    "LM":"Left Midfielder",

    "RM":"Right Midfielder",

    "LW":"Left Winger",

    "RW":"Right Winger",

    "CF":"Center Forward",

    "ST":"Striker"

}
elite = elite.copy()
elite["Position"] = elite["Position"].replace(position_map)

# Position distribution 
# how many players do the dataset have for each position
elite["Position"].value_counts().plot(kind="bar")

plt.ylabel("Number of Players")
plt.title("Player Position Distribution")

plt.show()

# Average raing per position 
avg = (
    elite
    .groupby("Position")
    ["OVR"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)

avg.to_csv("average_rating_by_position.csv")

# Correlation heatmap 
# the position and its
cols = ["PAC","SHO","PAS","DRI","DEF","PHY","OVR"]
corr = elite[cols].corr()
print(corr)

# ignore the overal score
features = ["PAC","SHO","PAS","DRI","DEF","PHY"]

# Here we want to group players based on similar overall score 
X = elite[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

elite["PC1"] = components[:,0]
elite["PC2"] = components[:,1]

plt.figure(figsize=(10,8))

positions = elite["Position"].unique()

for pos in positions:

    subset = elite[elite["Position"] == pos]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        alpha=0.6,
        label=pos,
        s=20
    )

# From the results, it looks like PC1 is about attacking and PC2 is about defensive skills
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Player Similarity using FIFA Attributes")

plt.legend(bbox_to_anchor=(1.05,1))

plt.tight_layout()

plt.savefig("PCA_Player_Map.png", dpi=300)

plt.show()

loading = pd.DataFrame(
    pca.components_.T,
    columns=["PC1","PC2"],
    index=features
)

print(loading)