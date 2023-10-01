import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils import construct_graph, data_encoder
from sklearn.preprocessing import MinMaxScaler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--data_path", default="./raw_data", help="Path to save the data")
parser.add_argument("--out_path", default="./processed_data", help="Path to save the data")

args = parser.parse_args()

np.random.seed(args.seed)
tablelist = {}
dname = 'home-credit'

# load data
applications = pd.read_csv(os.path.join(args.data_path, 'application_train.csv'), index_col=False)
bureau = pd.read_csv(os.path.join(args.data_path, 'bureau.csv'), index_col=False)
bureau_balance = pd.read_csv(os.path.join(args.data_path, 'bureau_balance.csv'), index_col=False)
installments_payments = pd.read_csv(os.path.join(args.data_path, 'installments_payments.csv'), index_col=False)
cash_balance = pd.read_csv(os.path.join(args.data_path, 'POS_CASH_balance.csv'), index_col=False)
credit_balance = pd.read_csv(os.path.join(args.data_path, 'credit_card_balance.csv'), index_col=False)
previous_application = pd.read_csv(os.path.join(args.data_path, 'previous_application.csv'), index_col=False)

out_path = os.path.join(args.out_path, 'GFS')
os.makedirs(out_path, exist_ok=True)

# data preprocess
# since we use the value after normalizing to multiply one vector for continuous type columns as
# embedding of these columns, we just set the Nan as 0 after noramlizing.
mapping = {'f1':1, 'f0':0}
applications['TARGET'] = applications['TARGET'].replace(mapping)
application_ID = applications['SK_ID_CURR']

bureau = bureau.merge(application_ID, how='inner', on='SK_ID_CURR')
bureau_balance = bureau_balance.merge(bureau['SK_ID_BUREAU'], how='inner', on='SK_ID_BUREAU')

previous_application = previous_application.merge(application_ID, how='inner', on='SK_ID_CURR')
installments_payments = pd.merge(pd.merge(installments_payments, application_ID, how='inner', on='SK_ID_CURR'), previous_application['SK_ID_PREV'], how='inner', on='SK_ID_PREV')
cash_balance = pd.merge(pd.merge(cash_balance, application_ID, how='inner', on='SK_ID_CURR'), previous_application['SK_ID_PREV'], how='inner', on='SK_ID_PREV')
credit_balance = pd.merge(pd.merge(credit_balance, application_ID, how='inner', on='SK_ID_CURR'), previous_application['SK_ID_PREV'], how='inner', on='SK_ID_PREV')

# normalize
# we do the following step
# 1. we normalize the continuous columns to [0, 1]
# 2. we fill the Nan with 0 for continuous columns
# 3. data encoder steps we will encode the Nan for categorical columns
scalar = MinMaxScaler()
con_applications = applications.select_dtypes(include='number').columns
applications[con_applications] = scalar.fit_transform(applications[con_applications])
applications[con_applications] = applications[con_applications].fillna(0)

con_bureau = bureau.select_dtypes(include='number').columns
bureau[con_bureau] = scalar.fit_transform(bureau[con_bureau])
bureau[con_bureau] = bureau[con_bureau].fillna(0)

con_bureau_balance = bureau_balance.select_dtypes(include='number').columns
bureau_balance[con_bureau_balance] = scalar.fit_transform(bureau_balance[con_bureau_balance])
bureau_balance[con_bureau_balance] = bureau_balance[con_bureau_balance].fillna(0)

con_install = installments_payments.select_dtypes(include='number').columns
installments_payments[con_install] = scalar.fit_transform(installments_payments[con_install])
installments_payments[con_install] = installments_payments[con_install].fillna(0)

con_cash = cash_balance.select_dtypes(include='number').columns
cash_balance[con_cash] = scalar.fit_transform(cash_balance[con_cash])
cash_balance[con_cash] = cash_balance[con_cash].fillna(0)

con_credit = credit_balance.select_dtypes(include='number').columns
credit_balance[con_credit] = scalar.fit_transform(credit_balance[con_credit])
credit_balance[con_credit] = credit_balance[con_credit].fillna(0)

con_prev = previous_application.select_dtypes(include='number').columns
previous_application[con_prev] = scalar.fit_transform(previous_application[con_prev])
previous_application[con_prev] = previous_application[con_prev].fillna(0)

# encode data
tablelist['applications'] = applications
tablelist['bureau'] = bureau
tablelist['bureau_balance'] = bureau_balance
tablelist['installments_payments'] = installments_payments
tablelist['cash_balance'] = cash_balance
tablelist['credit_balance'] = credit_balance
tablelist['previous_application'] = previous_application
tablelist = data_encoder(dname, tablelist, is_align=True, dummy_table=False)
print('Successfully encode the data')

# build graph
construct_graph(dname, tablelist, out_path, dummy_table=False)