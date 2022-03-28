from tqdm import tqdm
from tqdm.auto import tqdm
import glob
import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import dtale
import seaborn as sns
import matplotlib.pyplot as plt
from fuzzywuzzy import process
plt.rcParams["figure.figsize"] = (20,8)
tqdm.pandas() # for progress_apply

# Set up a regex to find the year from the filename
import re
year_match = re.compile('(\d{4})')

# Couple of functions to transform the scoreline into useful data
def goals(result: str):
    '''Given a scoreline such as 6-1 returns home and away goals in a dict {home: 6, away: 1}'''
    nums = result.split('-')
    try:
        return {'home': int(nums[0]), 'away': int(nums[1])}
    except Exception:
        return {'home': -1, 'away': -1}
def outcome(result):
    '''Given a string of the result e.g. 6-1 categorise as either a no-score draw (1), score draw (2), home win (3) or away win (4)'''
    goals_result = goals(result)
    if goals_result['home'] == goals_result['away']:
        if goals_result['home'] == 0:
            return 1
        else:
            return 2
    elif goals_result['home'] > goals_result['away']:
        return 3
    else:
        return 4

print('loading data')

# Generate a list of available leagues from the folder names
league_list = glob.glob('/Users/dev/aicore/Football-predictor/Football-Dataset/*')
league_list = pd.DataFrame(league_list)
league_list.columns = ['folder']
league_list['name'] = league_list['folder'].apply(lambda x: x.split('/')[-1])

# Get lists of all files in each folder
def get_files(folder):
    league_file_list = glob.glob(folder + '/*')
    league_file_list.sort()
    return league_file_list
league_list['files'] = league_list['folder'].apply(get_files)

# Now run through the list of leagues (folders) and the files in them (in the files column), read in the data, and combine them together into dfs.
# Results:
# - `overall_team_counts` is a df showing the number of teams involved per league per season (unlikely to be consistent - and indeed not - EPL is a counterexample)
# - `df_full` is the whole data of results across all leagues and seasons
overall_team_counts = pd.DataFrame()
df_league_lists = []
for index, league in tqdm(league_list.iterrows()):
    league_file_list = league['files']
    team_counts = {}
    df_list = []
    for file in league_file_list:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()
        year = year_match.findall(file)[0]
        if len(df.index) == 0:
            # print('No data for {}'.format(year))
            pass
        else:
            df_original = df.copy()
            df_list.append(df)
            team_counts[year] = len(df['home_team'].unique())
            # print('In {} there were {} teams'.format(year, len(df['Home_Team'].unique())))
    overall_team_counts = pd.concat([overall_team_counts, pd.Series(team_counts, name=league['name'])], axis=1)
    df_league_lists.append(pd.concat(df_list, ignore_index=True))
df_full = pd.concat(df_league_lists, ignore_index=True)
overall_team_counts.sort_index()
overall_team_counts = overall_team_counts.fillna(0).astype(int)

# Use the functions defined earlier - `goals` and `outcome` - to transform the scoreline (as a string) into numerical data
df_full['goals'] = df_full['result'].apply(goals)
# in case we're repeating this operation
try:
    df_full = df_full.drop(['home','away'],axis=1) 
except Exception:
    pass
df_full = pd.concat([df_full, pd.json_normalize(df_full['goals'])], axis=1).drop('goals', axis=1)
# drop errors
df_full = df_full.drop(df_full[df_full.home == -1].index)
df_full['total_goals'] = df_full['home'] + df_full['away']
df_full['outcome'] = df_full['result'].apply(outcome)

df = df_full.copy()

# join additional data
print('joining additional data')

print('...matches')
# first up: match info
match_df = pd.read_csv('additional-data/Match_Info.csv')
match_df.columns = match_df.columns.str.lower()
match_df['referee'] = match_df['referee'].str.strip().str.replace('Referee: ','')
match_df['date_dt'] = pd.to_datetime(match_df['date_new'])
match_df['link_parts'] = match_df['link'].str.split('/')
match_df['home_team'] = match_df.apply(lambda row: row['link_parts'][-3], axis=1)
match_df['away_team'] = match_df.apply(lambda row: row['link_parts'][-2], axis=1)
match_df['season'] = match_df.apply(lambda row: row['link_parts'][-1], axis=1)
match_df = match_df.drop(['date_new'], axis=1)

df_teams = df['home_team'].unique()
match_teams = match_df['home_team'].unique()
# needs a tweak for Manchester to give a helping hand on those two teams
matched_teams = [(d,(process.extractOne(d.replace('Man.','Manchester'), match_teams))[0]) for d in tqdm(df_teams)]

def team_name_for_link(t):
    matches = [x[1] for x in matched_teams if x[0] == t]
    return (matches if len(matches)>0 else [''])[0]
df['link'] = df.apply(lambda row: '/'.join(['','match',team_name_for_link(row['home_team']),team_name_for_link(row['away_team']),str(row['season'])]).strip().lower().replace(' ','-'), axis=1)
df.drop_duplicates(subset=['home_team','away_team','season','league'], inplace=True)

# join match data with main results df
df2 = pd.merge(match_df, df, how="inner", on='link')

print('...teams')
# next up: team info
team_df = pd.read_csv('additional-data/Team_Info.csv')
team_df.columns = team_df.columns.str.lower()
team_df['capacity'] = pd.to_numeric(team_df['capacity'].str.replace(',',''))
teams_from_team_data = set(team_df.team)
link_team_data = set(df2.home_team_x)
matched_teams_team_data = [(d,(process.extractOne(d.replace('Man.','Manchester'), teams_from_team_data))[0]) for d in link_team_data]

def team_name_for_link_team_data(t):
    matches = [x[0] for x in matched_teams_team_data if x[1] == t]
    return (matches if len(matches)>0 else [''])[0]
team_df['matched_team'] = team_df.apply(lambda row: team_name_for_link_team_data(row['team']), axis=1)

# join team data with main results
df3 = pd.merge(df2, team_df, how="inner", left_on="home_team_x", right_on="matched_team")

print('...elo')
# ELO import & join
import pickle
elo_dict = pickle.load(open('additional-data/elo_dict.pkl', 'rb'))
elo = pd.DataFrame.from_dict(elo_dict).transpose()
elo.columns = elo.columns.str.lower()
elo['link_parts'] = elo.index.str.split('/')
elo['home_team'] = elo.apply(lambda row: row['link_parts'][-3], axis=1)
elo['away_team'] = elo.apply(lambda row: row['link_parts'][-2], axis=1)
elo['season'] = elo.apply(lambda row: int(str(row['link_parts'][-1])[:4]), axis=1)
elo['link_index'] = elo.apply(lambda row: str(row['link_parts'][-1])[4:], axis=1)
elo['link'] = elo.apply(lambda row: '/'.join(['','match',row['home_team'],row['away_team'],str(row['season'])]), axis=1)
elo = elo.drop(['link_parts'], axis=1)

df3_teams = set(df3.home_team_x)
# lower case fixes sd-Compostela which was the only non-match
elo.home_team = elo.home_team.str.lower()

# now join elo data on to main data
df4 = pd.merge(df3, elo, how="left", on="link")
# clean up - drop some of the columns created by the joins
df4.drop(['link_parts','home_team_y','away_team_y','season_y','team','matched_team','home_team','away_team','season'], axis=1, inplace=True)
df4.rename(columns={'home_team_x':'home_team','away_team_x':'away_team','season_x':'season','home':'home_goals','away':'away_goals'}, inplace=True)

# the result so far
fulldf = df4.copy()

# FEATURE ENGINEERING
print('now for some features')

# Newly-promoted / newly-relegated teams
print('...newly-promoted/-relegated teams')
# we have to connect pairs of leagues so we can look at relegation and promotion between them
league_pairings = [
    ('ligue_1','ligue_2'),
    ('eredivisie','eerste_divisie'),
    ('premier_league','championship'),
    ('primera_division','segunda_division'),
    ('primeira_liga','segunda_liga'),
    ('serie_a','serie_b'),
    ('bundesliga','2_liga')
]

def teams(league,season):
    return set(fulldf[(fulldf['league']==league) & (fulldf['season']==season)].home_team)

# returns the league that's connected (same country) to the one supplied, and t/f as to whether the league supplied as l is the top league
def pairing(l):
    pair = [p for p in league_pairings if l in p][0]
    return ([t for t in pair if t != l][0], l == pair[0]) 

def teams_new_this_season(league, season):
    previous_season = teams(league,str(int(season)-1))
    this_season = teams(league, season)
    return list(this_season.difference(previous_season))

def teams_gone_from_last_season(league, season):
    previous_season = teams(league,str(int(season)-1))
    this_season = teams(league, season)
    return list(previous_season.difference(this_season))

# returns a 2-tuple of (promoted teams, relegated teams) - i.e. lists of teams in this league this season who arrived by either promotion or relegation
def promoted_teams(league, season):
    previous_season = teams(league,str(int(season)-1))
    this_season = teams(league, season)
    if len(previous_season) == 0:
        return ([],[])
    new_this_season = teams_new_this_season(league, season)
    connected = pairing(league)
    if not connected[1]: # league is NOT the top league so we need to look at relegated teams too
        relegated_teams = teams_gone_from_last_season(connected[0], season)
        promoted_teams = list(set(new_this_season).difference(set(relegated_teams)))
        return (promoted_teams, relegated_teams)
    else:
        return (new_this_season, [])

leagues = list(set(fulldf.league))
seasons = list(set(fulldf.season))
df_moved_teams = pd.DataFrame(columns=leagues)
df_moved_teams['season'] = seasons
df_moved_teams.set_index('season', inplace=True)
for league in tqdm(leagues):
    for season in seasons:
        df_moved_teams.at[season,league] = promoted_teams(league, season)

fulldf['home_newly_promoted'] = fulldf.apply(lambda row: row.home_team in df_moved_teams.at[row.season,row.league][0], axis=1)
fulldf['home_newly_relegated'] = fulldf.apply(lambda row: row.home_team in df_moved_teams.at[row.season,row.league][1], axis=1)
fulldf['away_newly_promoted'] = fulldf.apply(lambda row: row.away_team in df_moved_teams.at[row.season,row.league][0], axis=1)
fulldf['away_newly_relegated'] = fulldf.apply(lambda row: row.away_team in df_moved_teams.at[row.season,row.league][1], axis=1)

# drop the items missing elo
print('Before dropping ELO-less matches:',fulldf.size)
fulldf.dropna(subset=['elo_home','elo_away'],inplace=True)
print('After dropping ELO-less matches:',fulldf.size)
fulldf.sort_values('date_dt',inplace=True)

print('...cumulative home/away goals')
# Cumulative home/away goals in the season
fulldf[['home_goals_f_cum','home_goals_a_cum']] = fulldf.groupby(by=['season','home_team']).cumsum()[['home_goals','away_goals']]
fulldf[['away_goals_a_cum','away_goals_f_cum']] = fulldf.groupby(by=['season','away_team']).cumsum()[['home_goals','away_goals']]

# Winning / losing streaks
print('...winning/losing streaks')
df5 = fulldf.copy().set_index(['season','league']).sort_values('date_dt')
df5['home_win'] = df5.apply(lambda row: 1 if row['home_goals'] > row['away_goals'] else 0, axis=1)
df5['away_win'] = df5.apply(lambda row: 1 if row['home_goals'] < row['away_goals'] else 0, axis=1)
df5['draw'] = df5.apply(lambda row: 1 if row['home_goals'] == row['away_goals'] else 0, axis=1)
teams = list(df5['home_team'].unique())

# This construction of team_results_df gives decent raw data of win/lose streaks for each team, regardless of home or away
# it requires a column for every team in the dataset
df6 = df5.copy().reset_index()
teams = list(df6['home_team'].unique())
# Create the new columns, one per team and a row per match, then concat them onto the main data
z = np.zeros(shape=(len(df6),len(teams)))
team_cols = pd.DataFrame(z, columns = teams)
team_results_df = pd.concat([df6, team_cols],axis=1)
# fill in each team's column with 1 for a win and -1 for a defeat against that match
# we aren't gonna count draws for now
# so the majority of the new columns will just contain a zero, and only if there's a non-draw result will there be two cols containing a 1 / -1
print('(match outcomes)')
team_results_df.reset_index(inplace=True)
for i in tqdm(team_results_df.index):
    team_results_df.loc[i, team_results_df.loc[i, 'home_team']] = 1 if team_results_df.loc[i, 'home_goals'] > team_results_df.loc[i, 'away_goals'] else -1 if team_results_df.loc[i, 'home_goals'] < team_results_df.loc[i, 'away_goals'] else 0
    team_results_df.loc[i, team_results_df.loc[i, 'away_team']] = 1 if team_results_df.loc[i, 'home_goals'] < team_results_df.loc[i, 'away_goals'] else -1 if team_results_df.loc[i, 'home_goals'] > team_results_df.loc[i, 'away_goals'] else 0
team_results_df.reset_index(inplace=True)
team_results_df.set_index(['season'],inplace=True)
team_results_df.sort_values('date_dt',inplace=True)
# so this is almost quite nice ... giving a negative count for successive defeats and a positive count for successive wins
# the only flaw really is that draws just give zeros - we'll live with that for now, taking a draw as a neutral thing rather than a streak (which isn't really right)

# create new cols containing 
def count_streaks(season, team):
    filtered = team_results_df.loc[season].query('home_team == @team or away_team == @team')
    filtered = filtered[['link','result',team,'round','date_dt']].sort_values('date_dt')
    # calculate streaks
    filtered['start_of_streak'] = filtered[team].ne(filtered[team].shift())
    filtered['streak_id'] = filtered['start_of_streak'].cumsum()
    filtered['streak_result'] = filtered[team]
    filtered['streak_counter'] = filtered.groupby('streak_id').cumcount()
    filtered = filtered[['link','streak_counter','streak_result',team]]
    filtered.rename({'streak_counter': team+'_streak','streak_result':team+'_streak_result'},axis=1,inplace=True)
    return filtered

# do the count_streaks fn for each team in each season
# and collect the dfs in a list
print('(streaks)')
cumul_dfs = []
for season in tqdm(seasons):
    for team in teams:
        cumul_dfs.append(count_streaks(str(season), team))
# drop the empty ones (dunno why there are so many?? perhaps this is where there aren't any streaks so everything zero, but I suspect it's something more sinister???)
print('streak dfs before dropping empties:',len(cumul_dfs))
streak_dfs = [df for df in cumul_dfs if not df.empty]
print('streak dfs after dropping empties:',len(streak_dfs))
# concat all those dfs together so we've got all the teams' streaks 
# in this combined df we'll get all the different columns back, most of which will be full of NaN, but one set of 3 cols in each row will have content
print('concatenating streak dfs')
all_streaks_df = pd.concat(streak_dfs)
all_streaks_df.reset_index(inplace=True)

# return a pair of values, representing the home streak = length * sign (positive for win, negative for loss) , away streak
# each row will have only one team, and so either home or away will be zero, the other will be real data (possibly zero if either no streak or a streak of draws)
def home_or_away(row):
    # which team cols are non-NaN?
    first_non_null_col = row[2:].first_valid_index()
    team_name = first_non_null_col.split('_')[0]
    link_parts = row['link'].split('/')
    # print(link_parts, team_name)
    home_away = 'home' if team_name==link_parts[2] else 'away'
    # we'll return home_streak * home_streak_result, away_streak * away_streak_result
    if home_away == 'home':
        return row[team_name + '_streak'] * row[team_name + '_streak_result'], 0
    else:
        return 0, row[team_name + '_streak'] * row[team_name + '_streak_result']

# TODO: there's something very very slow taking place in the section below

# use this function to add the home_stream and away_streak cols to the df
all_streaks_df['home_streak'], all_streaks_df['away_streak'] = zip(*all_streaks_df.progress_apply(home_or_away, axis=1))
print('more processing... hang on...')
# and then just keep the key columns, drop the teams now that they've been copied into the home_streak and away_streak cols - we're done with them
just_streaks_df = all_streaks_df[['link','home_streak','away_streak']].copy()
# combine the pairs together - for the same match (=link) we have the home_streak in one row and away_streak in another
merged_streaks_df = just_streaks_df.groupby('link').sum()
# merge that data onto the main df
df7 = df6.merge(merged_streaks_df, on='link')

# Now some feature cleaning so we're ready to put numeric data into some models
print('some final feature cleaning')
# change bools to ints
# see https://stackoverflow.com/questions/17383094/how-can-i-map-true-false-to-1-0-in-a-pandas-dataframe
def bool_to_int(s: pd.Series) -> pd.Series:
    return 1 if s else 0
df7['home_newly_promoted'] = df7['home_newly_promoted'].apply(bool_to_int)
df7['away_newly_promoted'] = df7['away_newly_promoted'].apply(bool_to_int)
df7['home_newly_relegated'] = df7['home_newly_relegated'].apply(bool_to_int)
df7['away_newly_relegated'] = df7['away_newly_relegated'].apply(bool_to_int)
df7_trimmed = df7.drop(columns=['referee','home_yellow','home_red','away_yellow','away_red','date_dt','home_team','away_team','result','round','home_goals','away_goals','total_goals','city','country','stadium','pitch','link_index','home_win','away_win','draw'])
df7_trimmed['season'] = df7_trimmed['season'].apply(lambda s: 2022-int(s))
df7_trimmed['capacity'] = df7_trimmed['capacity']/10000
df_numeric = df7_trimmed.drop(columns='league')
# first two cols should be link then outcome
df_numeric.insert(0, 'outcome', df_numeric.pop('outcome'))
df_numeric.insert(0, 'link', df_numeric.pop('link'))

print('writing CSV')
df_numeric.to_csv('cleaned_dataset.csv', index=False)

print('Done.')
