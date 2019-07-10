import pandas as pd
import numpy as np
import datetime
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn

# required machine learning packages
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV as CCV

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
import xgboost as xgb

df = pd.read_csv("scores.csv")
teams = pd.read_csv("teams.csv")
games_elo = pd.read_csv("nfl_games.csv")

# replacing blank strings with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

# removing rows from specific columns that have null values, resetting index and changing data types
df = df[(df.score_home.isnull() == False) & (df.team_favorite_id.isnull() == False) & (df.over_under_line.isnull() == False) &
        (df.schedule_season >= 1979)]

df.reset_index(drop=True, inplace=True)
df['over_under_line'] = df.over_under_line.astype(float)

# mapping team_id to the correct teams
df['team_home'] = df.team_home.map(teams.set_index('team_name')['team_id'].to_dict())
df['team_away'] = df.team_away.map(teams.set_index('team_name')['team_id'].to_dict())

# fix team_favorite_id for Colts in 1969 and 1971 SB
df.loc[(df.schedule_season == 1968) & (df.schedule_week == 'Superbowl'), 'team_favorite_id'] = 'IND'
df.loc[(df.schedule_season == 1970) & (df.schedule_week == 'Superbowl'), 'team_favorite_id'] = 'IND'

# creating home favorite and away favorite columns (fill na with 0's)
df.loc[df.team_favorite_id == df.team_home, 'home_favorite'] = 1
df.loc[df.team_favorite_id == df.team_away, 'away_favorite'] = 1
df.home_favorite.fillna(0, inplace=True)
df.away_favorite.fillna(0, inplace=True)

# creating over / under column (fill na with 0's)
df.loc[((df.score_home + df.score_away) > df.over_under_line), 'over'] = 1
df.over.fillna(0, inplace=True)

# stadium neutral and schedule playoff as boolean
df['stadium_neutral'] = df.stadium_neutral.astype(int)
df['schedule_playoff'] = df.schedule_playoff.astype(int)

# change data type of date columns
df['schedule_date'] = pd.to_datetime(df['schedule_date'])
games_elo['date'] = pd.to_datetime(games_elo['date'])

# fixing some schedule_week column errors and converting column to integer data type
df.loc[(df.schedule_week == '18'), 'schedule_week'] = '17'
df.loc[(df.schedule_week == 'Wildcard') | (df.schedule_week == 'WildCard'), 'schedule_week'] = '18'
df.loc[(df.schedule_week == 'Division'), 'schedule_week'] = '19'
df.loc[(df.schedule_week == 'Conference'), 'schedule_week'] = '20'
df.loc[(df.schedule_week == 'Superbowl') | (df.schedule_week == 'SuperBowl'), 'schedule_week'] = '21'
df['schedule_week'] = df.schedule_week.astype(int)

# removing extra columns that aren't necessary for analysis
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
       'team_away', 'team_favorite_id', 'spread_favorite',
       'over_under_line', 'weather_temperature',
       'weather_wind_mph', 'score_home', 'score_away',
       'stadium_neutral', 'home_favorite', 'away_favorite',
       'over']]

# Cleaning games_elo and df to merge correctly
wsh_map = {'WSH' : 'WAS'}
games_elo.loc[games_elo.team1 == 'WSH', 'team1'] = 'WAS'
games_elo.loc[games_elo.team2 == 'WSH', 'team2'] = 'WAS'

# fix dates
df.loc[(df.schedule_date == '2016-09-19') & (df.team_home == 'MIN'), 'schedule_date'] = datetime.datetime(2016, 9, 18)
df.loc[(df.schedule_date == '2017-01-22') & (df.schedule_week == 21), 'schedule_date'] = datetime.datetime(2017, 2, 5)
df.loc[(df.schedule_date == '1990-01-27') & (df.schedule_week == 21), 'schedule_date'] = datetime.datetime(1990, 1, 28)
df.loc[(df.schedule_date == '1990-01-13'), 'schedule_date'] = datetime.datetime(1990, 1, 14)
games_elo.loc[(games_elo.date == '2016-01-09'), 'date'] = datetime.datetime(2016, 1, 10)
games_elo.loc[(games_elo.date == '2016-01-08'), 'date'] = datetime.datetime(2016, 1, 9)
games_elo.loc[(games_elo.date == '2016-01-16'), 'date'] = datetime.datetime(2016, 1, 17)
games_elo.loc[(games_elo.date == '2016-01-15'), 'date'] = datetime.datetime(2016, 1, 16)


# merge games_elo with df
df = df.merge(games_elo, left_on=['schedule_date', 'team_home', 'team_away'], right_on=['date', 'team1', 'team2'], how='left')

# merge to fix neutral games where team_home and team_away are switched
games_elo2 = games_elo.rename(columns={'team1' : 'team2', 'team2' : 'team1', 'elo1' : 'elo2', 'elo2' : 'elo1'})
games_elo2['elo_prob1'] = 1 - games_elo2.elo_prob1
df = df.merge(games_elo2, left_on=['schedule_date', 'team_home', 'team_away'], right_on=['date', 'team1', 'team2'], how='left')

# separating merged columns into x and y cols
x_cols = ['date_x', 'season_x', 'neutral_x', 'playoff_x', 'team1_x', 'team2_x', 'elo1_x', 'elo2_x', 'elo_prob1_x', 'score1_x',
         'score2_x', 'result1_x']
y_cols = ['date_y', 'season_y', 'neutral_y', 'playoff_y', 'team1_y', 'team2_y', 'elo1_y', 'elo2_y', 'elo_prob1_y', 'score1_y',
          'score2_y', 'result1_y']

# filling null values for games_elo merged cols
for x, y in zip(x_cols, y_cols):
    df[x] = df[x].fillna(df[y])

# removing y_cols from dataframe
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
       'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
       'weather_temperature', 'weather_wind_mph', 'score_home', 'score_away',
       'stadium_neutral', 'home_favorite', 'away_favorite', 'over', 'neutral_x', 'playoff_x',
         'elo1_x', 'elo2_x', 'elo_prob1_x']]

# remove _x ending from column names
df.columns = df.columns.str.replace('_x', '')

df['result'] = (df.score_home + df.score_away > df.over_under_line).astype(int)

score = df.groupby(['schedule_season', 'schedule_week', 'team_home']).mean()[['score_home', 'score_away']].reset_index()
aw_score = df.groupby(['schedule_season', 'schedule_week', 'team_away']).mean()[['score_home', 'score_away']].reset_index()

# create total pts column
score['point_diff'] = score.score_home - score.score_away
aw_score['point_diff'] = aw_score.score_away - aw_score.score_home

# append the two dataframes
score = score.append(aw_score, ignore_index=True, sort=True)

# fill null values
score.team_home.fillna(score.team_away, inplace=True)

# sort by season and week
score.sort_values(['schedule_season', 'schedule_week'], ascending = [True, True], inplace=True)

# removing unneeded columns & changing column name
score = score[['schedule_season', 'schedule_week', 'team_home', 'point_diff']]
score.rename(columns={'team_home' : 'team'}, inplace=True)

tm_dict = {}
for key in score.team.unique():
    tm_dict[key] = score[score.team == key].reset_index(drop=True)

pts_diff = pd.DataFrame()

# for loop to create a rolling average of the previous games for each season
for yr in score.schedule_season.unique():
    for tm in score.team.unique():
        data = tm_dict[tm].copy()
        data = data[data.schedule_season == yr]

        data.loc[:, 'avg_pts_diff'] = data.point_diff.shift().expanding().mean()

        pts_diff = pts_diff.append(data)

df = df.merge(pts_diff[['schedule_season', 'schedule_week', 'team', 'avg_pts_diff']],
              left_on=['schedule_season', 'schedule_week', 'team_home'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')

df.rename(columns={'avg_pts_diff' : 'hm_avg_pts_diff'}, inplace=True)

df = df.merge(pts_diff[['schedule_season', 'schedule_week', 'team', 'avg_pts_diff']],
              left_on=['schedule_season', 'schedule_week', 'team_away'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')

df.rename(columns={'avg_pts_diff' : 'aw_avg_pts_diff'}, inplace=True)

total_season = pts_diff.groupby(['schedule_season', 'team']).mean()['point_diff'].reset_index()

# adding schedule week for merge and adding one to the season for predictions
total_season['schedule_week'] = 1
total_season['schedule_season'] += 1

# cleaning of columns
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
       'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
       'weather_temperature', 'weather_wind_mph', 'score_home', 'score_away', 'stadium_neutral', 'home_favorite',
       'away_favorite', 'hm_avg_pts_diff','aw_avg_pts_diff', 'elo1', 'elo2', 'elo_prob1', 'over', 'result']]

# merge to have previous seasons average point differential
df = df.merge(total_season[['schedule_season', 'schedule_week', 'team', 'point_diff']],
              left_on=['schedule_season', 'schedule_week', 'team_home'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')

df.rename(columns={'point_diff' : 'hm_avg_diff'}, inplace=True)

df = df.merge(total_season[['schedule_season', 'schedule_week', 'team', 'point_diff']],
              left_on=['schedule_season', 'schedule_week', 'team_away'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')

df.rename(columns={'point_diff' : 'aw_avg_diff'}, inplace=True)

# fill null values
df.hm_avg_pts_diff.fillna(df.hm_avg_diff, inplace=True)
df.aw_avg_pts_diff.fillna(df.aw_avg_diff, inplace=True)

# cleaning of columns
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
       'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
       'weather_temperature', 'weather_wind_mph', 'score_home', 'score_away', 'stadium_neutral', 'home_favorite',
       'away_favorite', 'hm_avg_pts_diff','aw_avg_pts_diff', 'elo1', 'elo2', 'elo_prob1', 'over', 'result']]

# removing all rows with null values
df = df.dropna(how='any',axis=0)

# initial features possible for model
X = df[['schedule_season', 'schedule_week', 'over_under_line', 'spread_favorite', 'weather_temperature', 'weather_wind_mph',
        'home_favorite', 'hm_avg_pts_diff','aw_avg_pts_diff', 'elo1', 'elo2', 'elo_prob1']]

final = X[['over_under_line', 'spread_favorite', 'weather_wind_mph', 'home_favorite', 'hm_avg_pts_diff', 'aw_avg_pts_diff', 'elo_prob1']]
y = df['result']

# base model
base = LDA()

# choose 5 best features
# rfe = RFE(base, 7)
# rfe = rfe.fit(X, y)
#
# # features
# print(rfe.support_)
# print(rfe.ranking_)


models = []

models.append(('LRG', LogisticRegression(solver='liblinear')))
models.append(('KNB', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('XGB', xgb.XGBClassifier(random_state=0)))
models.append(('RFC', RandomForestClassifier(random_state=0, n_estimators=100)))
models.append(('DTC', DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5)))

# evaluate each model by average and standard deviations of roc auc
results = []
names = []

for name, m in models:
    kfold = model_selection.KFold(n_splits=5, random_state=0)
    cv_results = model_selection.cross_val_score(m, final, y, cv=kfold, scoring = 'roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)