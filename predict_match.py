import pandas as pd
import numpy as np
from fuzzywuzzy import process
from datetime import datetime

league_id = 'E0'
home_team = 'tottenham'
away_team = 'liverpool'

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
    matched_team_name = process.extractOne(home, match_teams)[0]
    home_streak = streak_feature(matched_team_name, results)
    home_goals = goals(matched_team_name, results)
    matched_team_name = process.extractOne(away, match_teams)[0]
    away_streak = streak_feature(matched_team_name, results)
    away_goals = goals(matched_team_name, results)
    return home_streak, home_goals, away_streak, away_goals

elo_df = pd.read_csv('http://api.clubelo.com/{}'.format(datetime.now().strftime("%Y-%m-%d")))

def elos(home, away, league_id):
    country = {'E':'ENG','D':'GER','S':'ESP','I':'ITA'}[league_id[0]]
    elo_teams = elo_df.query('Country == @country')
    match_teams = elo_teams['Club'].unique()
    matched_team_name = process.extractOne(home, match_teams)[0]
    home_elo = elo_df.query('Club == @matched_team_name')['Elo'].to_list()[0]
    matched_team_name = process.extractOne(away, match_teams)[0]
    away_elo = elo_df.query('Club == @matched_team_name')['Elo'].to_list()[0]
    return home_elo, away_elo

all_results = league_details(league_id)
home_streak, home_goals, away_streak, away_goals = streaks(home_team, away_team, all_results)
home_elo, away_elo = elos(home_team,away_team,league_id)
print('{} (H) have a streak of {} and elo of {} - goals at home {} F / {} A, away {} F / {} A'.format(home_team, home_streak, home_elo, home_goals[0], home_goals[1], home_goals[2], home_goals[3]))
print('{} (H) have a streak of {} and elo of {} - goals at home {} F / {} A, away {} F / {} A'.format(away_team, away_streak, away_elo, away_goals[0], away_goals[1], away_goals[2], away_goals[3]))
