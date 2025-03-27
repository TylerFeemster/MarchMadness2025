from files import Files
import pandas as pd
import numpy as np
import cvxpy as cp
import warnings

def get_reg(sex, include_day=False) -> pd.DataFrame:
    f = Files()
    df = process_compact(f.df('regular_season_compact', sex=sex), include_day=include_day).query(
        'Season >= 2003').reset_index(drop=True)
    df['AID'] = df['Season'].astype(str) + '_' + df['ATeamID'].astype(str)
    df['BID'] = df['Season'].astype(str) + '_' + df['BTeamID'].astype(str)
    # columns: Season, AID, BID, Score, HCA
    return df.drop(columns=['ATeamID', 'BTeamID'])


def tourney(sex, df=None) -> pd.DataFrame:
    f = Files()
    if df is None:
        df = process_compact(f.df('tourney_compact_results', sex=sex))[[
            'Season', 'ATeamID', 'BTeamID', 'Score']].query('Season >= 2003').reset_index(drop=True)

    # columns: Season, ATeamID, BTeamID, Score, HCA

        seeds = f.df('tourney_seeds', sex=sex).query('Season >= 2003')
        # df has Season, ATeamID, BTeamID, Score, HCA
        df = df.rename(columns={'ATeamID': 'TeamID'}).merge(seeds, how='left', on=['Season', 'TeamID'])\
            .rename(columns={'TeamID': 'ATeamID', 'Seed': 'ASeed'})
        df = df.rename(columns={'BTeamID': 'TeamID'}).merge(seeds, how='left', on=['Season', 'TeamID'])\
            .rename(columns={'TeamID': 'BTeamID', 'Seed': 'BSeed'})
    
    df['AID'] = df['Season'].astype(str) + '_' + df['ATeamID'].astype(str)
    df['BID'] = df['Season'].astype(str) + '_' + df['BTeamID'].astype(str)

    if sex == 'M':
        ordinals = pd.concat([f.df('ordinals'), pd.read_csv('./data/MMasseyOrdinals_2025.csv')])\
            .query('RankingDayNum == 133')
        ordinals['ID'] = ordinals['Season'].astype(
            str) + '_' + ordinals['TeamID'].astype(str)
        
        rank_list = []
        for col in ['POM', 'MAS', 'MOR']:
            single = ordinals.query(f'SystemName == "{col}"')[['ID', 'OrdinalRank']]
            df = df.rename(columns={'AID': 'ID'}).merge(single, how='left', on='ID').rename(
                columns={'ID': 'AID', 'OrdinalRank': 'ARank'})
            df = df.rename(columns={'BID': 'ID'}).merge(single, how='left', on='ID').rename(
                columns={'ID': 'BID', 'OrdinalRank': 'BRank'})
            delta = np.sqrt(df['BRank']) - np.sqrt(df['ARank'])
            rank_list.append(delta)
            df = df.drop(columns=['ARank', 'BRank'])
            if col=='POM':
                df['Rank'] = delta

        warnings.filterwarnings('ignore')
        df['Fav'] = np.nan_to_num(np.sign(np.nanmean(rank_list, axis=0)))

    df['ANum'] = df['ASeed'].apply(lambda x: x[1:3]).astype(int)
    df['BNum'] = df['BSeed'].apply(lambda x: x[1:3]).astype(int)

    df['Finals'] = df['ASeed'].apply(lambda x: x[0]) \
        != df['BSeed'].apply(lambda x: x[0])

    round_dictionary = {}

    for s1 in range(1, 17):
        for s2 in range(1, 17):
            if s1 == s2:
                # wild card will be considered round 1
                round_dictionary[(s1, s2)] = 1
                continue
            if s1 + s2 == 17:
                round_dictionary[(s1, s2)] = 1
                continue
            v1 = min(s1, 17 - s1)
            v2 = min(s2, 17 - s2)
            if v1 + v2 == 9:
                round_dictionary[(s1, s2)] = 2
                continue
            w1 = min(v1, 9 - v1)
            w2 = min(v2, 9 - v2)
            if w1 + w2 == 5:
                round_dictionary[(s1, s2)] = 3
                continue
            round_dictionary[(s1, s2)] = 4

    id_dict = {}
    for id, game in df.query('Finals == False').iterrows():
        id_dict[id] = round_dictionary[(game['ANum'], game['BNum'])]
    df['Round'] = id_dict
    # Final Four and Championship considered Round 5
    df['Round'] = df['Round'].fillna(5).astype(int)

    df['Seed'] = np.sqrt(df['BNum']) - np.sqrt(df['ANum'])
    df['Seed_Fav'] = (df['BNum'] - df['ANum']).apply(np.sign)

    df = df.drop(columns=['ASeed', 'BSeed', 'Finals', 'ANum',
                 'BNum', 'ATeamID', 'BTeamID']).reset_index(drop=True)

    # columns: Season, AID, BID, Score, Seed, Round
    return df


def process_compact(df: pd.DataFrame, include_day=False) -> pd.DataFrame:

    if include_day:
        df = df[['Season', 'DayNum', 'WTeamID',
                 'LTeamID', 'WScore', 'LScore', 'WLoc']].copy()
    else:
        df = df[['Season', 'WTeamID', 'LTeamID',
                 'WScore', 'LScore', 'WLoc']].copy()

    df['Score'] = df['WScore'] - df['LScore']

    mapper = {'H': 1, 'A': -1, 'N': 0}
    df['HCA'] = df['WLoc'].apply(lambda x: mapper[x])

    df = df.drop(columns=['WScore', 'LScore', 'WLoc'])

    # Season, WTeamID, LTeamID, Score, HCA

    fd = df.copy()
    df = df.rename(columns={'WTeamID': 'ATeamID', 'LTeamID': 'BTeamID'})
    fd = fd.rename(columns={'WTeamID': 'BTeamID', 'LTeamID': 'ATeamID'})
    fd['Score'] *= -1
    fd['HCA'] *= -1

    return pd.concat([df, fd[df.columns]]).sort_values('Season')


def graph_algorithm(df: pd.DataFrame):
    # don't use men's and women's together
    # columns: Season, AID, BID, Score, HCA
    df = df.copy()

    # Normalize by Season
    stds = df[['Season', 'Score']].groupby(['Season']).std()\
        .reset_index().rename(columns={'Score': 'std_Score'})

    df = pd.merge(df, stds, on='Season')
    df['Score'] = df['Score'] / df['std_Score']

    df = df.drop(columns=['std_Score'])

    # Season, AID, BID, Score, HCA

    averages = df[['Season', 'AID', 'Score']].groupby(['Season', 'AID'])\
        .mean().reset_index().rename(columns={'AID': 'ID'}).drop(columns=['Season'])
    averages = averages.set_index('ID').T.drop_duplicates()

    df = df.drop(columns=['Season'])  # AID, BID, Score, HCA

    cof = 1.4
    df['Score'] = sigm(df['Score'], cof) #1 / (1 + np.exp(- cof * df['Score']))

    return solver(averages, df).T.reset_index()


def solver(mus: pd.DataFrame, games: pd.DataFrame):
    idx_to_id = mus.columns.to_list()
    id_to_idx = {id: idx for idx, id in enumerate(idx_to_id)}
    num_teams = len(idx_to_id)

    nus = cp.Variable(num_teams)

    hca = cp.Variable()
    hca_mat = hca * games['HCA'].to_numpy()

    mu_mat = mus.to_numpy().reshape(-1)

    delta_mat = games['Score'].to_numpy()
    a_indices = [id_to_idx[aid] for aid in games['AID']]
    b_indices = [id_to_idx[bid] for bid in games['BID']]

    nu_a_mat = nus[a_indices]
    nu_b_mat = nus[b_indices]

    mu_a_mat = mu_mat[a_indices]
    mu_b_mat = mu_mat[b_indices]

    # Confs Experiment
    id_cid, idx_to_conf = get_id_confs()

    conf_to_idx = {conf: idx for idx, conf in enumerate(idx_to_conf)}
    num_confs = len(idx_to_conf)
    confs = cp.Variable(num_confs)

    a_conf_indices = [conf_to_idx[id_cid[aid]] for aid in games['AID']]
    b_conf_indices = [conf_to_idx[id_cid[bid]] for bid in games['BID']]

    a_conf_mat = confs[a_conf_indices]
    b_conf_mat = confs[b_conf_indices]

    sum = cp.sum_squares(mu_a_mat + nu_a_mat + a_conf_mat -
                         mu_b_mat - nu_b_mat - b_conf_mat - delta_mat + hca_mat)

    # sum = cp.sum_squares(mu_a_mat + nu_a_mat - mu_b_mat - nu_b_mat - delta_mat + hca_mat)

    constraints = [cp.sum(nus) == 0]
    obj = cp.Minimize(sum)

    prob = cp.Problem(obj, constraints)
    prob.solve()

    id_idx_to_cid_idx = {
        id_idx: conf_to_idx[id_cid[id]] for id, id_idx in id_to_idx.items()}
    confs_of_ids = [id_idx_to_cid_idx[id_idx] for id_idx in range(num_teams)]

    return mus + nus.value + confs.value[confs_of_ids]


def sigm(x, sc):
    return 1 / (1 + np.exp(- sc * x))


def tanh(x, tc):
    return (np.tanh(tc * x) + 1) / 2


def __elo_updater(winner_elo, loser_elo, score, day, hca_winner: int):
    hca_amt = 100
    elo_width = 400
    margin_factor = 20
    k = 20
    k_factor = k * (1 + day / 130) * (1 + (score - 1) / margin_factor)
    win_likelihood = 1 / \
        (1 + 10 ** ((loser_elo - winner_elo + hca_winner * hca_amt) / elo_width))
    delta_elo = k_factor * (1 - win_likelihood)
    winner_elo += delta_elo
    loser_elo -= delta_elo
    return winner_elo, loser_elo


def get_elos(sex):
    base_elo = 1000
    df = get_reg(sex, include_day=True).sort_values(['Season', 'DayNum'])
    seasons = sorted(df['Season'].unique().tolist())

    elo = dict()

    for season in seasons:
        season_df = df.query('Season == @season')

        # initialize elos
        ids = list(set(season_df['AID']) | set(season_df['BID']))
        for id in ids:
            prev_id = f'{season-1}_{id[5:]}'
            if prev_id in elo:
                elo[id] = elo[prev_id]
            else:
                elo[id] = base_elo

        # update elos
        season_df = season_df.query('Score > 0').sort_values('DayNum')
        for _, game in season_df.iterrows():
            aid, bid = game['AID'], game['BID']
            elo[aid], elo[bid] = __elo_updater(
                elo[aid], elo[bid], game['Score'], game['DayNum'], game['HCA'])

    elo_df = pd.Series(elo).reset_index()
    elo_df.columns = ['ID', 'Elo']
    return elo_df


def get_id_confs():
    f = Files()
    m_teams = f.df('team_conferences', sex='M')
    m_teams['ConfAbbrev'] = m_teams['ConfAbbrev'] + '_MEN'
    w_teams = f.df('team_conferences', sex='W')
    w_teams['ConfAbbrev'] = w_teams['ConfAbbrev'] + '_WOMEN'
    confs = pd.concat([m_teams, w_teams])

    confs['ID'] = confs['Season'].astype(
        str) + '_' + confs['TeamID'].astype(str)
    confs['CID'] = confs['Season'].astype(str) + '_' + confs['ConfAbbrev']

    conf_list = confs['CID'].unique().tolist()
    return confs[['ID', 'CID']].drop_duplicates().set_index('ID').to_dict()['CID'], conf_list
