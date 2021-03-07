import numpy as np
import pandas as pd
import os
import pickle

from absl import app
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, minmax_scale, StandardScaler
from tree import Tree
import datetime as dt

def main():
    data_path = ('data/favorita/train.csv')
    events_path = ('data/favorita/holidays_events.csv')
    oil_path = ('data/favorita/oil.csv')
    items_path = ('data/favorita/items.csv')
    pkl_path = ('data/favorita/data.pkl')

    start_date = dt.datetime.strptime('2013-01-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2017-08-15', '%Y-%m-%d')
    delta = end_date - start_date
    num_days = delta.days + 1
    print(f'Total time = {num_days} days')
    
    print('Computing features ...')
    oil = pd.read_csv(oil_path)
    oil['date'] = pd.to_datetime(oil['date'])
    
    x_dates = []
    y_prices = []

    idx = 0
    for row in oil.itertuples():
        day_num = (row.date - start_date).days
        price = row.dcoilwtico
        if not np.isnan(price):
            x_dates.append(day_num)
            y_prices.append(price)
    prices = np.interp(range(num_days), x_dates, y_prices)
    wday = []
    month = []
    for i in range(num_days):
        new_date = start_date + dt.timedelta(days=i)
        wday.append(new_date.weekday())
        month.append(new_date.month)
    wday = np.array(wday, dtype=np.float32)
    month = np.array(month, dtype=np.float32)

    feats = np.vstack([prices, wday, month]).T
    feats = minmax_scale(feats)
    global_cont_feats = np.array(feats, dtype=np.float32)
    print(f'global_cont_feats: {global_cont_feats.shape}')

    print('Computing cat features ...')
    events = pd.read_csv(events_path)
    events['date'] = pd.to_datetime(events['date'])
    global_cat_feats = [[''] * 4 for i in range(num_days)]

    cat_feat_list = ['type', 'locale', 'locale_name', 'transferred']
    events = events[['date'] + cat_feat_list].fillna('')

    for row in events.itertuples():
        day_num = (row.date - start_date).days
        if day_num in range(0, num_days):
            global_cat_feats[day_num][0] = row.type
            global_cat_feats[day_num][1] = row.locale
            global_cat_feats[day_num][2] = row.locale_name
            global_cat_feats[day_num][3] = 'yes' if row.transferred else ''

    enc = OrdinalEncoder(dtype=np.int32)
    global_cat_feats = enc.fit_transform(global_cat_feats)
    global_cat_feats = [global_cat_feats[:, i] for i in range(len(cat_feat_list))]
    global_cat_dims = [len(cat) for cat in enc.categories_]
    print('global_cat_feats', len(global_cat_feats[0]), len(global_cat_dims))

    feats = (global_cont_feats, global_cat_feats, global_cat_dims)

    print('Creating tree ...')
    items = pd.read_csv(items_path)
    items.rename(columns={'class': 'clas'}, inplace=True)
    tree = Tree()

    item_to_node_str = {}
    for row in items.itertuples():
        item_nbr = row.item_nbr
        family = row.family
        clas = row.clas
        node_str = f'{family}_{clas}_{item_nbr}'
        item_to_node_str[item_nbr] = node_str
        tree.insert_seq(node_str)
    
    tree.precompute()

    print('Reading time series data from csv ...')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    num_ts = tree.num_nodes
    print(f'Num nodes = {num_ts}')
    ts_data = np.zeros((num_ts, num_days), dtype=np.float32)

    for row in tqdm(df[['date', 'item_nbr', 'unit_sales']].itertuples(), total=len(df.index)):
        day_num = (row.date - start_date).days
        sales = row.unit_sales
        item_id = row.item_nbr
        node_str = item_to_node_str[item_id]
        nid = tree.node_id[node_str]
        ts_data[nid, day_num] += sales

    ts_data = tree.leaf_matrix @ ts_data
    ts_data = ts_data.T
    print(f'ts_data: {ts_data.shape}')

    with open(pkl_path, 'wb') as fout:
        pickle.dump((tree, ts_data, feats), fout)

if __name__ == '__main__':
    main()
