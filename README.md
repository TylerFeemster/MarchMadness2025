# Kaggle's March Madness 2025 Competition

## Overview

This year's [competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/overview) asks competitors to predict the win probability of all possible match-ups in the NCAA tournament, both women's and men's. Since there's never been a perfect bracket, this is a much more robust, *and exciting*, choice. Even when a game doesn't go our way, we can easily "take the L" knowing we already have the next game's predictions in our submission file.

The metric for this competition is the **Brier score**. This is just mean squared error, where the actual target is 1 or 0 depending on whether the first team won or lost. The score spread does not make a difference; a win is a win. The Brier score is a *true* probability metric, i.e. predicting the true probabilities minimizes the espected Brier score. To see this, suppose team A has a probability $p$ of beating team B. The expected score given that we guess $q$ is:

$$p(1-q)^2 + (1-p)q^2.$$

Since this is a strictly convex function in $q$, the score is minimized when its derivative is $0$:

$$0 = -2p(1-q) + 2(1-p)q = -2p + 2pq + 2q - 2pq = 2(q - p),$$

which is satisfied when $p = q$.

Because the Brier score is quadratic, it heavily penalizes predictions that are *confidently wrong*. In fact, predicting a game at 50-50 odds (the baseline) is 4x better than predicting 100-0 odds in the wrong direction.

## Personal Model

### Model Summary

My model was a linear model based on a team embedding that I built by graph smoothing the regular season score outcomes. My cross-validation was simple. I left one season out (2003-2024) and predicted that year's tournament with the resulting model. My average, cross-validated Brier score was 0.161.

### Main Feature

In recent iterations of this competition, Elo has been a fairly strong measure of win-loss probability. One if its advantages is that it naturally biases the most recent performance of teams. This is good for at least a couple reasons:
1. The conference tournaments are the last games to be played leading up to the NCAA tournament, and they give us strong information about the relative strength of teams.
2. If there are team line-up changes (e.g., transfers or major injuries), Elo will bias the updated team's performance.

At the same time, Elo only updates teams based on their current rating. This unfairly discounts early season results, especially since ratings change year-to-year. For instance, suppose Team A beats Team B early in the season, but they are both rated as average teams. Let's say Team B goes on to beat lots of great teams, raising their Elo, while Team A goes on to win almost all of their remaining games against moderate to weak opponents. In this case, Team A won't get the benefit in Elo of beating Team B, a really strong opponent. This points to a critical insight: the results of a week of games not only gives us information about those particular games, but also all previous games that season. Elo score does not do this.

My model's strength comes in its solution to this exact problem. Here's the idea: we treat the regular season as a graph, where each team is a node, and each game is an edge. Each team, or node, is given a single real number to be learned. This number is a proxy for score. The basic version of this model learns these numbers by minimizing

$$\sum_{g \in games}[(\hat x_{team_1} - \hat x_{team_2}) - (x_{team_1}^g - x_{team_2}^g)]^2.$$

In words, we smooth the graph by fitting a single score to each team such that the difference between these team scores is most compatible with the empirical scores throughout the season. Note that the metric is convex, so it is easily solved by standard quadratic programming algorithms. I used CVXPY, but it can also easily be coerced into a linear regression problem and solved that way. Either way, the solution is unique, as long as we enforce some constraint on the total sum, such as

$$\sum_{t \in teams}x_t = 0.$$

Straight away, we see some unique advantages of this team encoding:
1. It sees all regular season games as highly informative.
2. It allows conference differences to emerge naturally and agnostically. Conferences are recognized as graph clusters, structual phenomena that don't have to be explicitly accounted for. Since team encodings within a conference can all change by the same amount without affecting the in-conference spread predictions, two conference clusters are easily compared and rectified by their out-of-conference games.

This is the naive algorithm. My solution has some minor improvements, which I'll describe now. First, we have the problem of blow-out games. If Team A beats Team C by 50 points, and Team B beats Team C by 30 points, does that really suggest Team A is much better than Team B? Maybe Team B was up 30 at halftime, so they benched all their starters early. Further, since the metric is least squares, these outliers would be far too influential in the model fitting. To solve this, I decided to apply the sigmoid function to some constant times the score difference. That way, extreme games are pushed to 0 or 1 while the rest are somewhere in between. Thresholding via clipping would also achieve the same aim, but it would cost us some reduction in information. In the example above, there is still something to be said about Team A winning by 20 more points than Team B. The constant in the sigmoid function was chosen by cross-validated hyperparameter tuning.

There was a second issue with which I dealt: home court advantage. My solution to this was simple: I added a single learnable parameter to the metric to account for it. This looks like

$$\sum_{g \in games}[(\hat x_{team_1} - \hat x_{team_2} + h \cdot x_{hca}) - (x_{team_1}^g - x_{team_2}^g)]^2,$$

where $h$ is 1 when Team 1 is at home, -1 when Team 2 is at home, and 0 when the game is at a neutral site. This single number $x_{hca}$ did not vary depending on the year or whether it was a men's or women's game. Yet, I saw about a 10%+ decrease in overall error after fitting. This number was not used beyond the fitting, but it allowed us to achieve more reliable team embeddings.

### Model Details

My model included 3 ElasticNet regressors for men, and 3 Ridge regressors for women (these also used the scikit-learn ElasticNet class, but the L1 ratio was 0). These regressors partition the games, so only one regressor is used for any given game. The three regressors split the tournament into 3 classes: Round 1, Rounds 2 and 3, and the Elite Eight games and beyond. This allows different features to become more or less important depending on the competition stage. For instance, chalk seeds have high signal in Round 1 but are rarely important in the finals. The partitioning was also chosen with data quantity in mind. With sufficient data, it would probably be better to have a single model for the Final Four games and Championship, but that's only 3 games per year (there'd be a different model for men and women). My model is a linear model with only a few parameters, so 7 games per year for 20+ years is sufficient. This justifies our Elite Eight and beyond model.

There's another advantage to having games partitioned this way. Hyperparameter tuning (alpha and L1-ratio) is much easier since changing one model's alpha leaves the other models untouched. This gives us lots of room for hyperparameter tuning while skirting around the curse of dimensionality. This can be a problem for gradient-boosted trees, where you might want to optimize performance across 10 dependent hyperparameters.

L1-ratio did not have a major effect on score, but alpha tended to increase with the rounds. There are two clear reasons for this: there is less data as the rounds get closer to the championship, and the games become harder to predict.

### Features

As a primary motivator, all features are designed such that the resulting model $f$ satisfies

$$f(\text{Team}_A, \text{Team}_B) = 1 - f(\text{Team}_B, \text{Team}_A),$$ 

i.e. the antisymmetry of probability is implicit.

For both men and women, I used the team embeddings $\{x_{team}\}$ described above to generate 6 features. Five were:

$$\sigma(c \cdot \Delta x_{team}) - 0.5, \quad c \in \{0.01, 0.1, 0.3, 1, 10\},$$

and the other one was $\text{sign}(\Delta x_{team})$.

This is proportional to the previous feature with $c \to \infty$.

I also had an Elo-based feature. With my other features, I found it was most informative to set a team's Elo at the beginning of a season to their previous season's Elo. This is against the standard, where all Elos are average with some base Elo as a partial season reset. I also modified the K factor so that it's more aggresive near the end of the season, utilizes point spread, and considers home court advantage. The resulting feature is just the difference between the Elo scores of the two teams.

I derived two features from the chalk seeds. The first is subtracting the square root of the seeds. So, if seed 1 plays seed 16, the Seed feature holds $\sqrt{16} - \sqrt{1} = 3$. The other feature is just a flag variable: 1 when team A has the better seed, -1 when team B has the better seed, and 0 otherwise.

These are all the features used for the women's tournament. For men, I added two features based on alternative ranking systems with historical predictive ability; I would've done the same for the women's tournament, but this data was not available. The first is Ken Pomeroy's famous, possession-based rating system. In reality, the rankings are secondary to the ratings he generates. The difference in rating is proportional to the logit win chances. Looking at this year's ratings, I found that $\Delta \sqrt{r_{team}}$, where $r$ is the ranking, was almost directly proportional to the difference in rating, at least for the top 100 ranks (most relevant to subset of March Madness teams). So, my feature was $\Delta \sqrt{r_{team}}$.

The second ranking feature was similar. I took the average square root difference of rankings from Pomeroy, Massey, and Moore. Then, I returned the resulting sign value: 1 if Team A was favored, -1 if Team B was favored, and 0 otherwise.

## Submission

My final submission for the Kaggle competition was a simple average of my personal model with [Raddar](https://kaggle.com/raddar)'s notebook, copied here as `vilnius-ncaa.ipynb`. His underlying ML algorithm is gradient-boosted trees plus a 5-parameter spline regressor with his out-of-fold validation results to smooth outputs. The gradient-boosted trees utilized lots of seasonal stat data, as well as Elo ratings and a novel "Team Quality" feature to give an alternative (historically superior) seeding. The Brier score coming from his validation (leave-one-out cross validation by season) is 0.168 from 2003 to 2024.

His model used seasonal stat data and a completely different model to achieve a cross-validation score only slightly worse than mine. For this reason, I went with a 50%-50% ensemble. This should work well given the metric is essentially Ordinary Least Squares, and the models are diverse.

Because competitors are given the opportunity to use two final submissions, I found the Round 1 game with probability closest to 50%. This was the women's match-up between Mississippi State and California. One submission gives California a 100% chance to win, and the other gives Mississippi State a 100% chance to win. This yields the maximum guaranteed Brier score reduction. That's why you'll find `raddar_ensemble_california.csv` and `raddar_ensemble_missst.csv` files in the predictions folder. Because Mississippi State won, the second file is responsible for my final placement.