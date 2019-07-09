import pandas as pd
import numpy as np
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

df = pd.read_csv('scores.csv')

teams = pd.read_csv('teams.csv')

df = df.replace(r'^\s*$', np.nan, regex=True)

df = df[(df.score_home.isnull() == False) & (df.team_favorite_id.isnull() == False) & (df.over_under_line.isnull() == False) &
        (df.schedule_season >= 1979)]

# add over under
df.reset_index(drop=True, inplace=True)
df['over_under_line'] = df.over_under_line.astype(float)

df['team_home'] = df.team_home.map(teams.set_index('team_name')['team_id'].to_dict())
df['team_away'] = df.team_away.map(teams.set_index('team_name')['team_id'].to_dict())

df.loc[(df.schedule_season == 1968) & (df.schedule_week == 'Superbowl'), 'team_favorite_id'] = 'IND'
df.loc[(df.schedule_season == 1970) & (df.schedule_week == 'Superbowl'), 'team_favorite_id'] = 'IND'

# creating home favorite and away favorite columns (fill na with 0's)
df.loc[df.team_favorite_id == df.team_home, 'home_favorite'] = 1
df.loc[df.team_favorite_id == df.team_away, 'away_favorite'] = 1
df.home_favorite.fillna(0, inplace=True)
df.away_favorite.fillna(0, inplace=True)

df['total'] = df.score_home + df.score_away

# df.groupby('schedule_season', as_index=False)['total'].mean()

threshold = df.over_under_line.mean() + df.over_under_line.std()

#add threshold
df.loc[(df.over_under_line > threshold), 'target'] = 1
df.target.fillna(0, inplace=True)

score = df.groupby(['schedule_season', 'team_home']).mean()[['score_home']].reset_index()
away_score = df.groupby(['schedule_season', 'team_away']).mean()[['score_away']].reset_index()
print(score)
print(away_score)


df.loc[(df.weather_detail == 'DOME'), 'dome'] = 1
df.dome.fillna(0, inplace=True)
df.loc[((df.score_home + df.score_away) > df.over_under_line), 'over'] = 1
df.over.fillna(0, inplace=True)
df.fillna(0, inplace=True)
df.drop({'schedule_playoff', 'stadium_neutral', 'weather_detail', 'team_home',
         'stadium', 'schedule_date', 'team_away', 'team_favorite_id'}, axis=1, inplace=True)

df.to_csv('check.csv')

# df.loc[(df.schedule_week == 'Superbowl'), 'schedule_season'] = 18

X = df.drop({'schedule_season','schedule_week', 'score_home','score_away', 'over'}, axis=1)

# X = X.replace(np.nan, 0)
print(X.columns)
Y = df['over']
base = LDA()


rfe = RFE(base, 5)
rfe = rfe.fit(X, Y)



# features
print(rfe.support_)
print(rfe.ranking_)



x = df[{'over_under_line','weather_wind_mph', 'home_favorite',
       'away_favorite', 'dome', 'schedule_season'}]
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
    cv_results = model_selection.cross_val_score(m, x, Y, cv=kfold, scoring = 'roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


lrg = LogisticRegression(solver='liblinear')
boost = xgb.XGBClassifier()
gnb = GaussianNB()

vote = VotingClassifier(estimators=[('boost', boost), ('gnb', gnb), ('lrg', lrg)], voting='soft')

model = CCV(vote, method='isotonic', cv=3)



train = df[{'over_under_line','weather_wind_mph', 'home_favorite','away_favorite', 'dome', 'schedule_season', 'over' }]

test = df[{'over_under_line','weather_wind_mph', 'home_favorite',
       'away_favorite', 'dome', 'schedule_season', 'over'}]

train = train.loc[train['schedule_season'] < 2017]

test = test.loc[test['schedule_season'] > 2016]

train.drop('schedule_season', axis=1, inplace=True)
test.drop('schedule_season', axis=1, inplace=True)

X_train = train.drop('over', axis=1)
Y_train = train['over']
X_test = test.drop('over', axis=1)
Y_test = test['over']

model.fit(X_train, Y_train)
predicted = model.predict_proba(X_test)[:,1]
print(format(roc_auc_score(Y_test, predicted)))
print(X_train.head(10))
print(Y_train.head(10))
print(X_test.schedule_season)
print(Y_test.head(10))
print(df.describe().transpose())