import pandas as pd
from files import Files
import numpy as np
from sklearn.linear_model import ElasticNet
from pipeline_utils import *
import warnings

class Pipeline:

    def __init__(self):
        reg_M = get_reg('M')
        reg_W = get_reg('W')

        # Hyperparameters
        self.m_alphas = [0, 1e-3, 1e-1]
        self.m_l1_ratios = [0.1] * 3

        self.w_alphas = [0, 1e-2, 0.5]
        self.w_l1_ratios = [0] * 3

        # Season, AID, BID, Score, Seed, Round
        self.trn_mens = tourney('M')
        self.trn_womens = tourney('W')

        # ID, Elo
        self.m_elos = get_elos(sex='M')
        self.w_elos = get_elos(sex='W')

        # ID, Score
        self.m_scores = graph_algorithm(reg_M)
        self.w_scores = graph_algorithm(reg_W)

        self.sub_M = None
        self.sub_W = None


    def submission(self):
        sub = pd.read_csv('./data/SampleSubmissionStage2.csv')[['ID']]
        sub['Season'] = 2025
        sub['ATeamID'] = sub['ID'].apply(lambda x: int(x[5:9]))
        sub['BTeamID'] = sub['ID'].apply(lambda x: int(x[10:14]))

        f = Files()
        m_seeds = f.df('tourney_seeds', sex='M').query('Season == 2025')[['Seed', 'TeamID']]
        w_seeds = f.df('tourney_seeds', sex='W').query('Season == 2025')[['Seed', 'TeamID']]
        seeds = pd.concat([m_seeds, w_seeds]).reset_index(drop=True)
        sub = sub.merge(seeds, how='left', left_on='ATeamID', right_on='TeamID')\
            .drop(columns=['TeamID']).rename(columns={'Seed': 'ASeed'})
        sub = sub.merge(seeds, how='left', left_on='BTeamID', right_on='TeamID')\
            .drop(columns=['TeamID']).rename(columns={'Seed': 'BSeed'})
        subdf = sub.dropna().copy()

        subdf['Men'] = subdf['ATeamID'] // 1000 == 1
        self.sub_M = tourney(sex='M', df=subdf.query('Men == True')[['Season', 'ATeamID', 'BTeamID', 'ASeed', 'BSeed']].copy())
        self.sub_W = tourney(sex='W', df=subdf.query('Men == False')[['Season', 'ATeamID', 'BTeamID', 'ASeed', 'BSeed']].copy())
        # columns: Season, AID, BID, Seed, Round

        self.preprocess()
        # Season, AID, BID, Seed, Round, Score, Elo, Est

        self.predict('M')
        self.predict('W')

        subM = self.sub_M[['AID', 'BID', 'Pred']]
        subW = self.sub_W[['AID', 'BID', 'Pred']]

        subby = pd.concat([subM, subW])
        subby['AID'] = subby['AID'].apply(lambda x: x[5:]).astype(int)
        subby['BID'] = subby['BID'].apply(lambda x: x[5:]).astype(int)
        subby['ID'] = '2025_' + \
            subby['AID'].astype(str) + '_' + subby['BID'].astype(str)

        sub = sub[['ID']].merge(subby[['ID', 'Pred']], how='left', on='ID').fillna(0.5)
        sub.to_csv('./predictions/submission.csv', index=False)

    def predict(self, sex):

        warnings.filterwarnings('ignore')

        X_cols = ['sigm0', 'sigm1', 'sigm2', 'sigm3', 'sigm4', 'Seed', 'Elo', 'Est', 'Seed_Fav']
        
        if sex == 'M':
            X_cols += ['Rank', 'Fav']
            alphas = self.m_alphas
            l1_ratios = self.m_l1_ratios
            tra = self.trn_M
            tst = self.sub_M
        else:
            X_cols += []
            alphas = self.w_alphas
            l1_ratios = self.w_l1_ratios
            tra = self.trn_W
            tst = self.sub_W

        std = tra['Score'].std()
        tra['Score'] /= std
        tst['Score'] /= std

        for i, s in enumerate([0.01, 0.1, 0.3, 1, 10]):
            tra[f'sigm{i}'] = sigm(tra['Score'], s) - 0.5
            tst[f'sigm{i}'] = sigm(tst['Score'], s) - 0.5

        for col in X_cols:
            std = tra[col].std()
            tra[col] /= std
            tst[col] /= std

        results = []
        for round in [0, 1, 2]:
            alpha = alphas[round]
            l1_ratio = l1_ratios[round]
            match round:
                case 0:
                    tra_trn = tra.query('Round == 1')
                    tst_trn = tst.query('Round == 1')
                case 1:
                    tra_trn = tra.query('Round > 1 & Round < 4')
                    tst_trn = tst.query('Round > 1 & Round < 4')
                case 2:
                    tra_trn = tra.query('Round >= 4')
                    tst_trn = tst.query('Round >= 4')

            m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(tra_trn[X_cols], tra_trn['Win'])

            pred = m.predict(tst_trn[X_cols]).clip(0, 1)
            tst_trn['Pred'] = pred
            results.append(tst_trn)
        
        tst = pd.concat(results)
        if sex=='M':
            self.sub_M = tst
        else:
            self.sub_W = tst

    def __leave_one_out(self, season: int, tourn : pd.DataFrame, sex: str):

        warnings.filterwarnings('ignore')

        X_cols = ['sigm0', 'sigm1', 'sigm2', 'sigm3', 'sigm4', 'Seed', 'Elo', 'Est', 'Seed_Fav']
        tourn = tourn.copy()
        tourn['Score'] /= tourn['Score'].std()
        for i, s in enumerate([0.01, 0.1, 0.3, 1, 10]):
            tourn[f'sigm{i}'] = sigm(tourn['Score'], s) - 0.5

        if sex=='M':
            X_cols += ['Rank', 'Fav']
            alphas = self.m_alphas
            l1_ratios = self.m_l1_ratios
        else:
            X_cols += []
            alphas = self.w_alphas
            l1_ratios = self.w_l1_ratios

        # Season, Seed, Round, Score, Est, Win
        val = tourn.query('Season == @season')
        tra = tourn.query('Season != @season')

        for col in X_cols:
            std = tra[col].std()
            tra[col] /= std
            val[col] /= std
        
        val_err = 0
        for round in [0, 1, 2]:
            alpha = alphas[round]
            l1_ratio = l1_ratios[round]
            match round:
                case 0:
                    tra_trn = tra.query('Round == 1')
                    val_trn = val.query('Round == 1')
                case 1:
                    tra_trn = tra.query('Round > 1 and Round < 4')
                    val_trn = val.query('Round > 1 and Round < 4')
                case 2:
                    tra_trn = tra.query('Round >= 4')
                    val_trn = val.query('Round >= 4')

            m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(tra_trn[X_cols], tra_trn['Win'])
            val_err += np.sum(np.pow(m.predict(val_trn[X_cols]).clip(0, 1) - val_trn['Win'], 2))
        
        return val_err / len(val)
        
    
    def preprocess(self):

        warnings.filterwarnings('ignore')

        trn_M = self.trn_mens.copy()
        trn_W = self.trn_womens.copy()

        trn_M['Win'] = (trn_M['Score'] > 0).astype(float)
        trn_W['Win'] = (trn_W['Score'] > 0).astype(float)

        trn_M = trn_M.drop(columns=['Score'])
        trn_W = trn_W.drop(columns=['Score'])
        # Season, Seed, Round, Win, AID, BID

        trn_M = trn_M.rename(columns={'AID': 'ID'}).merge(
            self.m_scores, how='left', on='ID').rename(columns={'Score': 'AScore'})
        trn_M = trn_M.merge(self.m_elos, how='left', on='ID').rename(
            columns={'ID': 'AID', 'Elo': 'AElo'})
        trn_M = trn_M.rename(columns={'BID': 'ID'}).merge(
            self.m_scores, how='left', on='ID').rename(columns={'Score': 'BScore'})
        trn_M = trn_M.merge(self.m_elos, how='left', on='ID').rename(
            columns={'ID': 'BID', 'Elo': 'BElo'})
        trn_M['Score'] = trn_M['AScore'] - trn_M['BScore']
        trn_M['Elo'] = (trn_M['AElo'] - trn_M['BElo']) / 400
        trn_M = trn_M.drop(
            columns=['AScore', 'BScore', 'AElo', 'BElo', 'AID', 'BID'])
        trn_M['Est'] = trn_M['Score'].apply(np.sign)
        # Season, Seed, Round, Score, Elo, Est, Win

        trn_W = trn_W.rename(columns={'AID': 'ID'}).merge(
            self.w_scores, how='left', on='ID').rename(columns={'Score': 'AScore'})
        trn_W = trn_W.merge(self.w_elos, how='left', on='ID').rename(
            columns={'ID': 'AID', 'Elo': 'AElo'})
        trn_W = trn_W.rename(columns={'BID': 'ID'}).merge(
            self.w_scores, how='left', on='ID').rename(columns={'Score': 'BScore'})
        trn_W = trn_W.merge(self.w_elos, how='left', on='ID').rename(
            columns={'ID': 'BID', 'Elo': 'BElo'})
        trn_W['Score'] = trn_W['AScore'] - trn_W['BScore']
        trn_W['Elo'] = (trn_W['AElo'] - trn_W['BElo']) / 400
        trn_W = trn_W.drop(
            columns=['AScore', 'BScore', 'AElo', 'BElo', 'AID', 'BID'])
        trn_W['Est'] = trn_W['Score'].apply(np.sign)
        # Season, Seed, Round, Score, Elo, Est, Win

        self.trn_M = trn_M
        self.trn_W = trn_W

        if self.sub_M is not None:
            sub_M = self.sub_M.rename(columns={'AID': 'ID'}).merge(
                self.m_scores, how='left', on='ID').rename(columns={'Score': 'AScore'})
            sub_M = sub_M.merge(self.m_elos, how='left', on='ID').rename(
            columns={'ID': 'AID', 'Elo': 'AElo'})
            sub_M = sub_M.rename(columns={'BID': 'ID'}).merge(
            self.m_scores, how='left', on='ID').rename(columns={'Score': 'BScore'})
            sub_M = sub_M.merge(self.m_elos, how='left', on='ID').rename(
            columns={'ID': 'BID', 'Elo': 'BElo'})
            sub_M['Score'] = sub_M['AScore'] - sub_M['BScore']
            sub_M['Elo'] = (sub_M['AElo'] - sub_M['BElo']) / 400
            sub_M = sub_M.drop(columns=['AScore', 'BScore', 'AElo', 'BElo'])
            sub_M['Est'] = sub_M['Score'].apply(np.sign)
            # Season, AID, BID, Seed, Round, Score, Elo, Est
            self.sub_M = sub_M
        
        if self.sub_W is not None:
            sub_W = self.sub_W.rename(columns={'AID': 'ID'}).merge(
                self.w_scores, how='left', on='ID').rename(columns={'Score': 'AScore'})
            sub_W = sub_W.merge(self.w_elos, how='left', on='ID').rename(
                columns={'ID': 'AID', 'Elo': 'AElo'})
            sub_W = sub_W.rename(columns={'BID': 'ID'}).merge(
                self.w_scores, how='left', on='ID').rename(columns={'Score': 'BScore'})
            sub_W = sub_W.merge(self.w_elos, how='left', on='ID').rename(
                columns={'ID': 'BID', 'Elo': 'BElo'})
            sub_W['Score'] = sub_W['AScore'] - sub_W['BScore']
            sub_W['Elo'] = (sub_W['AElo'] - sub_W['BElo']) / 400
            sub_W = sub_W.drop(columns=['AScore', 'BScore', 'AElo', 'BElo'])
            sub_W['Est'] = sub_W['Score'].apply(np.sign)
            # Season, AID, BID, Seed, Round, Score, Elo, Est
            self.sub_W = sub_W

    def validation(self, m_alphas, m_l1_ratios, w_alphas, w_l1_ratios):

        self.m_alphas = m_alphas
        self.m_l1_ratios = m_l1_ratios
        self.w_alphas = w_alphas
        self.w_l1_ratios = w_l1_ratios

        warnings.filterwarnings('ignore')

        self.preprocess()        

        m_errors = []
        w_errors = []

        for season in range(2003, 2025):
            if season == 2020: continue
            m_err = self.__leave_one_out(season, self.trn_M, 'M')
            print(f'  Mens {season}: {m_err : .5f}')
            w_err = self.__leave_one_out(season, self.trn_W, 'W')
            print(f'Womens {season}: {w_err : .5f}')
            m_errors.append(m_err)
            w_errors.append(w_err)
            print(f'===================== {season}: {np.mean([m_err, w_err]) : .5f}')

        m_avg = np.sum(m_errors) / len(m_errors)
        w_avg = np.sum(w_errors) / len(w_errors)
        print(f'\n  Mens Mean: {m_avg : .5f}')
        print(f'Womens Mean: {w_avg : .5f}')
        print(f'===================== Mean: {np.mean([m_avg, w_avg]) : .5f}')