import pandas as pd
import numpy as np
from fuzzywuzzy import process
from datetime import datetime
import re


from besoccer_scraper import Scraper

league_names = {"E0":"premier_league", 
                "E1":"championship", 
                "D1":"bundesliga", 
                "D2":"2_liga", 
                "SP1":"primera_division", 
                "SP2":"segunda_division", 
                "P1":"primeira_liga", 
                "I1":"serie_a", 
                "I2":"serie_b", 
                "F1":"ligue_1", 
                "F2":"ligue_2", 
                "N1":"eredivisie"
                }

league_id = 'E0'
league_name = 'premier_league'

def res_int(res):
    return 1 if res=='H' else -1 if res=='A' else 0

def result(row, team):
    return res_int(row['FTR']) * (1 if row['HomeTeam']==team else -1)

def league_details(league):
    # return pd.read_excel('https://www.football-data.co.uk/mmz4281/2122/all-euro-data-2021-2022.xlsx', sheet_name=league)
    return latest_results[league]

def streak_feature(team_name, results):
    team_results = results.query('HomeTeam == @team_name or AwayTeam == @team_name').copy()
    team_results['team_res'] = team_results.apply(lambda row: result(row, team_name), axis=1)
    team_results['start_of_streak'] = team_results['team_res'].ne(team_results['team_res'].shift())
    team_results['streak_id'] = team_results['start_of_streak'].cumsum()
    team_results['streak_result'] = team_results['team_res']
    team_results['streak_counter'] = team_results.groupby('streak_id').cumcount()
    team_results['streak_feature'] = team_results['streak_counter'] * team_results['streak_result']
    streak_feature = team_results['streak_feature'].tolist()[-1]
    if streak_feature != 0:
        streak_feature = (abs(streak_feature)+1) * np.sign(streak_feature)
    return streak_feature

def goals(team, results):
    """Returns the total number of home + away goals scored by a team in the season so far"""
    home_matches = results.query('HomeTeam == @team')
    goals_home_for = home_matches['FTHG'].sum()
    goals_home_against = home_matches['FTAG'].sum()
    away_matches = results.query('AwayTeam == @team')
    goals_away_against = away_matches['FTHG'].sum()
    goals_away_for = away_matches['FTAG'].sum()
    return goals_home_for, goals_home_against, goals_away_for, goals_away_against

def streaks(home, away, results):
    match_teams = results['HomeTeam'].unique()
    home_matched_team_name = process.extractOne(home, match_teams)[0]
    home_streak = streak_feature(home_matched_team_name, results)
    home_goals = goals(home_matched_team_name, results)
    away_matched_team_name = process.extractOne(away, match_teams)[0]
    away_streak = streak_feature(away_matched_team_name, results)
    away_goals = goals(away_matched_team_name, results)
    return home_matched_team_name, home_streak, home_goals, away_matched_team_name, away_streak, away_goals


# Turns out these Elo values are completely different from those which the model was trained on (from besoccer.com)
# It might just be a multiplier or something, but we'll revert to using the values scraped from besoccer.com instead
# elo_df = pd.read_csv('http://api.clubelo.com/{}'.format(datetime.now().strftime("%Y-%m-%d")))

# def elos(home, away, league_id):
#     country = {'E':'ENG','D':'GER','S':'ESP','I':'ITA'}[league_id[0]]
#     elo_teams = elo_df.query('Country == @country')
#     match_teams = elo_teams['Club'].unique()
#     home_matched_team_name = process.extractOne(home, match_teams)[0]
#     home_elo = elo_df.query('Club == @home_matched_team_name')['Elo'].to_list()[0]
#     away_matched_team_name = process.extractOne(away, match_teams)[0]
#     away_elo = elo_df.query('Club == @away_matched_team_name')['Elo'].to_list()[0]
#     return home_matched_team_name, home_elo, away_matched_team_name, away_elo

# home_team_elos, home_elo, away_team_elos, away_elo = elos(home_team,away_team,league_id)

be_regex = re.compile(r'^https://www.besoccer.com/match/(.*)/(.*)/.*$')
feature_data = []
# read all sheets
latest_results = pd.read_excel('https://www.football-data.co.uk/mmz4281/2122/all-euro-data-2021-2022.xlsx', sheet_name=None) 

for league_id, league_name in league_names.items():
    print(league_id, league_name)
    all_results = league_details(league_id)
    besoccer_league_data = Scraper(league_name)
    matches, elo_dict = besoccer_league_data.get_next_matches()
    for match_details in elo_dict:
        teams = re.match(be_regex, match_details)
        home_team = teams.group(1)
        away_team = teams.group(2)
        print(home_team, 'vs', away_team)
        home_team_streaks, home_streak, home_goals, away_team_streaks, away_streak, away_goals = streaks(home_team, away_team, all_results)
        home_elo, away_elo = elo_dict[match_details]['Elo_home'], elo_dict[match_details]['Elo_away']
        feature_data.append([league_name, home_team_streaks, away_team_streaks, home_elo, away_elo,  home_goals[0], home_goals[1], away_goals[2], away_goals[3], home_streak, away_streak])
    print('-----------------')

cleaned_data = pd.DataFrame(feature_data, columns=['league','home_team','away_team','home_elo','away_elo','home_goals_f','home_goals_a','away_goals_f','away_goals_a','home_streak','away_streak'])
cleaned_data['home_elo'] = cleaned_data['home_elo'].astype(float)
cleaned_data['away_elo'] = cleaned_data['away_elo'].astype(float)
with open('cleaned_results.csv', 'w') as f:
    cleaned_data.to_csv(f, index=False)
print('saved csv')

