from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup as bs
import requests
from tqdm import tqdm
import pandas as pd
import os
import pathlib
import pickle

class Scraper:
    def __init__(self, league:str, url:str='https://www.besoccer.com/competition', year: int=2022) -> None:
        pathlib.Path(f'Data/Results/{league}').mkdir(parents=True, exist_ok=True) 
        pathlib.Path(f'Data/To_Predict/{league}').mkdir(parents=True, exist_ok=True) 
        self.league = league
        self.url = url
        self.year = year
        r = requests.get(f"{self.url}/scores/{self.league}/{self.year}")
        time.sleep(1)
        soup = bs(r.content, 'html.parser')
        matchday_str = soup.find('div', {'class': 'panel-title'}).text
        self.matchday = [int(s) for s in matchday_str.split() if s.isdigit()][0]

    def get_previous_matches(self):
        results = {'Home_Team': [], 'Away_Team': [], 'Result': [], 'Link': [], 'Season': [], 'Round': [], 'League': []}
        for matchday in tqdm(range(1, self.matchday)):
            r = requests.get(f"{self.url}/scores/{self.league}/{self.year}/round{matchday}")
            time.sleep(1)
            soup = bs(r.content, 'html.parser')
            matches_box = soup.find('div', {'class': 'panel-body p0 match-list-new'})
            matches = matches_box.find_all('a', {'class': 'match-link'})
            for match in matches:
                home_team = match.find('div', {'class': 'team-info ta-r'}).find('div', {'class': 'name'}).text.strip()
                away_team = match.find_all('div', {'class': 'team-info'})[1].find('div', {'class': 'name'}).text.strip()
                home_score = match.find('div', {'class': 'marker'}).find('span', {'class': 'r1'}).text.strip()
                away_score = match.find('div', {'class': 'marker'}).find('span', {'class': 'r2'}).text.strip()
                results['Home_Team'].append(home_team)
                results['Away_Team'].append(away_team)
                results['Result'].append(f'{home_score}-{away_score}')
                results['Link'].append(match.get('href'))
                results['Season'].append(self.year)
                results['Round'].append(matchday)
                results['League'].append(self.league)
        df = pd.DataFrame(results)
        df.to_csv(f'Data/Results/{self.league}/Results_{self.year}_{self.league}.csv')
    
    def get_next_matches(self):
        results = {'Home_Team': [], 'Away_Team': [], 'Link': [], 'Season': [], 'Round': [], 'League': []}
        elo_dict = {}
        r = requests.get(f"{self.url}/scores/{self.league}/{self.year}/round{self.matchday + 1}")
        time.sleep(1)
        soup = bs(r.content, 'html.parser')
        matches_box = soup.find('div', {'class': 'panel-body p0 match-list-new'})
        matches = matches_box.find_all('a', {'class': 'match-link'})
        self.matches = matches

        for match in matches:
            home_team = match.find('div', {'class': 'team-info ta-r'}).find('div', {'class': 'name'}).text.strip()
            away_team = match.find_all('div', {'class': 'team-info'})[1].find('div', {'class': 'name'}).text.strip()
            results['Home_Team'].append(home_team)
            results['Away_Team'].append(away_team)
            results['Link'].append(match.get('href'))
            results['Season'].append(self.year)
            results['Round'].append(self.matchday + 1)
            results['League'].append(self.league)
            
        for link in results['Link']:
            time.sleep(3)
            r = requests.get(link + '/analysis')
            soup = bs(r.content, 'html.parser')
            elo_box = soup.find('div', {'class': 'panel-body pn compare-data'})
            elo_row = elo_box.find_all('tr')[1]
            home_elo = elo_row.find('td', {'class': 'team1-c'}).text.strip()
            away_elo = elo_row.find('td', {'class': 'team2-c'}).text.strip()
            elo_dict[link] = {'Elo_home': home_elo, 
                              'Elo_away': away_elo}

        return pd.DataFrame(results), elo_dict

        # we don't need to save to disk because we're only using it to get the elo values
        # df = pd.DataFrame(results)
        # df.to_csv(f'Data/To_Predict/{self.league}/Results_{self.year}_{self.league}.csv')
        # with open(f'Data/To_Predict/{self.league}/elo_dict.pkl', 'wb') as f:
        #     pickle.dump(elo_dict, f)