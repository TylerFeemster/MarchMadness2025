import pandas as pd

class Files:
    def __init__(self):
        self.base = './data/'

        # non-sexed
        self.ordinals = self.base + 'MMasseyOrdinals.csv'

        # sexed
        self.tourney_slots = 'NCAATourneySlots.csv'
        self.teams = 'Teams.csv'
        self.team_conferences = 'TeamConferences.csv'
        self.regular_season_detailed = 'RegularSeasonDetailedResults.csv'
        self.regular_season_compact = 'RegularSeasonCompactResults.csv'
        self.tourney_seeds = 'NCAATourneySeeds.csv'
        self.tourney_compact_results = 'NCAATourneyCompactResults.csv'
        self.tourney_detailed_results = 'NCAATourneyDetailedResults.csv'

    def file(self, string: str, sex: str = 'M'):
        start = self.base + sex
        match string:
            case 'tourney_slots':
                return start + self.tourney_slots
            case 'teams':
                return start + self.teams
            case 'team_conferences':
                return start + self.team_conferences
            case 'regular_season_detailed':
                return start + self.regular_season_detailed
            case 'regular_season_compact':
                return start + self.regular_season_compact
            case 'tourney_seeds':
                return start + self.tourney_seeds
            case 'tourney_detailed_results':
                return start + self.tourney_detailed_results
            case 'tourney_compact_results':
                return start + self.tourney_compact_results
            # non-sexed
            case 'ordinals':
                return self.ordinals
            
    def df(self, string: str, sex: str = 'M'):
        return pd.read_csv(self.file(string, sex))