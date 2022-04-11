import pandas as pd
import numpy as np
from fuzzywuzzy import process
from datetime import datetime
import re
import joblib

from besoccer_scraper import Scraper

league_id = 'E0'
league_name = 'premier_league'

prediction_labels = ['','no-score draw','score draw','home win','away win']

def res_int(res):
    return 1 if res=='H' else -1 if res=='A' else 0

def result(row, team):
    return res_int(row['FTR']) * (1 if row['HomeTeam']==team else -1)

latest = pd.ExcelFile('https://www.football-data.co.uk/mmz4281/2122/all-euro-data-2021-2022.xlsx')

def league_details(league):
    return pd.read_excel('https://www.football-data.co.uk/mmz4281/2122/all-euro-data-2021-2022.xlsx', sheet_name=league)

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
    home_matches = results.query('HomeTeam == @team')
    goals_h_f = home_matches['FTHG'].sum()
    goals_h_a = home_matches['FTAG'].sum()
    away_matches = results.query('AwayTeam == @team')
    goals_a_a = away_matches['FTHG'].sum()
    goals_a_f = away_matches['FTAG'].sum()
    return goals_h_f, goals_h_a, goals_a_f, goals_a_a

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

football_classifier = joblib.load('football_classifier.pkl')

all_results = league_details(league_id)

besoccer_league_data = Scraper(league_name)
matches, elo_dict = besoccer_league_data.get_next_matches()

be_regex = re.compile(r'^https://www.besoccer.com/match/(.*)/(.*)/.*$')
for match_details in elo_dict:
    teams = re.match(be_regex, match_details)
    home_team = teams.group(1)
    away_team = teams.group(2)
    home_team_streaks, home_streak, home_goals, away_team_streaks, away_streak, away_goals = streaks(home_team, away_team, all_results)
    home_elo, away_elo = elo_dict[match_details]['Elo_home'], elo_dict[match_details]['Elo_away']
    # print('{} (H) ({}) have a streak of {} and elo of {} - goals at home {} F / {} A, away {} F / {} A'.format(home_team, home_team_streaks, home_streak, home_elo, home_goals[0], home_goals[1], home_goals[2], home_goals[3]))
    # print('{} (H) ({}) have a streak of {} and elo of {} - goals at home {} F / {} A, away {} F / {} A'.format(away_team, away_team_streaks, away_streak, away_elo, away_goals[0], away_goals[1], away_goals[2], away_goals[3]))
    features = np.array([home_elo, away_elo,  home_goals[0], home_goals[1], away_goals[0], away_goals[1], home_streak, away_streak], dtype=float).reshape(1,-1)
    print(home_team, away_team, features)
    prediction = football_classifier.predict(features)
    print('Prediction for {} vs {}: {}'.format(home_team, away_team, prediction_labels[prediction[0]]))
    print('---')