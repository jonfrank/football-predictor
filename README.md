# Football Prediction project

This is a project to build a model based on results of football matches in various leagues since 1990, to be used to predict the results of future matches. It will use various ML-focused python packages, including these: pandas, seaborn ...

## Milestone 1: EDA and Data Cleaning

- Data was supplied in a zip file, `Football-Dataset-zip`
- I used a regex to extract the season from the filename: `year_match = re.compile('(\d{4})')` to include the season in the data
- Also created two functions, `goals` which gave home and away goals as `int`s from a `str` scoreline, and `outcome` which gave a numerical category to each result, either 1 (no-score draw), 2 (score draw), 3 (home win) or 4 (away win)
- Then created `league_list`, a dataframe listing all the leagues with their league name, the folder name, and a list of all the csv files contained in the folder
- Loop through that df and construct a master dataframe pulling together all the data from the separate files per league and per season. All data that was imported had column names turned to lower case with `df.columns = df.columns.str.lower()`
- Add columns for home / away / total goals and outcome, using the functions defined earlier
- Then graphed the outcomes per league per season:

![image](outcomes.png)

This just gives an overview of the kind of spread of data, the seasons we have in different leagues, any change over time, etc.

###

Some general findings:

- there were no results given for certain leagues in certain seasons
- the number of teams in a particular league was not consistent throughout all seasons
- the patterns of outcomes and total goals across the different leagues does not generally differ much, although there are some that are slightly more consistent across the seasons, and a couple (notably Segunda Liga and Eerste Divisie) which are much more variable)

### Hypotheses as to which factors might be significant

1. capacity - some teams might play better at large stadia
1. pitch surface
1. which round ... some teams might be better earlier or later in the season?
1. how about newly-promoted? (we would need to engineer this feature later on)

## Calculating win/loss streaks

This turned out to be a thorny problem. Various attempts by myself and others to use elegant `groupby`-style methods just didn't work. And it can be pretty hard to debug those with all the pandas alchemy going on behind the scenes.

Instead I took a rather less elegant approach:
1. create a new zero-filled column for each team, named as the team name
1. for each match, if either team won, put a 1 in their column and a -1 in the losing team's column
1. loop through each season and each team, and for each case create streak data on that subframe; `concat` all those together
1. for each row, copy the team's streak info into either the `home_streak` or the `away_streak` column, depending on where they are playing
1. use `groupby('link').sum()` to combine the matching pairs of rows - there'll be two for each match, one for the home team and one away
1. `merge` this resulting frame back into the main dataframe

Checks of subframes for a particular season and team suggest that it's working.

#### Potential improvements:
1. track draws as well rather than effectively ignoring them
1. check that we're accurate on the streak being about games completed **prior** to the game in hand - not sure whether this is right yet.