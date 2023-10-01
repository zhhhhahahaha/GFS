# PK means primary key, MK means this column appears in multiple table
# The encoding process requires that same domain in different table must have same name
# For do not have these kinds of things:
# 1. 'Date': []
# 2. 'PK': None,
# 3. 'MK': None,
encode_config = {
    'acquire-valued-shoppers' :{
        'History' :{
            'PK': 'id',
            'Date': ['offerdate'],
            'MK': ['chain'],
        },
        'offers' :{
            'PK': 'offer',
            'Date': [],
            'MK': ['category', 'company', 'brand'],
        },
        'transactions':{
            'PK': None,
            'Date': ['date'],
            'MK': ['chain', 'category', 'company', 'brand'],
        }
    },
    'outbrain':{
        'clicks':{
            'PK': None,
            'Date': [],
            'MK': None,
        },
        'events':{
            'PK': 'display_id',
            'Date': ['timestamp'],
            'MK': None,
        },
        'ad_content':{
            'PK': 'ad_id',
            'Date': [],
            'MK': None,
        },
        'doc_meta':{
            'PK': 'document_id',
            'Date': [],
            'MK': None,
        },
        'doc_ent':{
            'PK': None,
            'Date': [],
            'MK': None,
        },
        'doc_topics':{
            'PK': None,
            'Date': [],
            'MK': None,
        },
        'doc_cat':{
            'PK': None,
            'Date': [],
            'MK': None,
        }
    },
    'outbrain-full':{
        'clicks':{
            'PK': None,
            'Date': [],
            'MK': None,
        },
        'events':{
            'PK': 'display_id',
            'Date': ['timestamp'],
            'MK': ['platform', 'geo_location'],
        },
        'user':{
            'PK': 'uuid',
            'Date': [],
            'MK': None,
        },
        'page_views':{
            'PK': None,
            'Date': ['timestamp'],
            'MK': ['platform', 'geo_location'],
        },
        'ad_content':{
            'PK': 'ad_id',
            'Date': [],
            'MK': None,
        },
        'doc_meta':{
            'PK': 'document_id',
            'Date': [],
            'MK': None,
        },
        'doc_ent':{
            'PK': None,
            'Date': [],
            'MK': None,
        },
        'doc_topics':{
            'PK': None,
            'Date': [],
            'MK': None,
        },
        'doc_cat':{
            'PK': None,
            'Date': [],
            'MK': None,
        }
    },
    'diginetica':{
        'clicks':{
            'PK': None,
            'Date': [],
            'MK': None,
        },
        'queries_meta':{
            'PK': 'queryId',
            'Date': ['eventdate'],
            'MK': ['categoryId'],
        },
        'products':{
            'PK': 'itemId',
            'Date': [],
            'MK': ['categoryId'],
        },
        'session':{
            'PK': 'sessionId',
            'Date': [],
            'MK': None,
        },
        'item_views':{
            'PK': None,
            'Date': ['eventdate'],
            'MK': None,
        },
        'purchases':{
            'PK': None,
            'Date': ['eventdate'],
            'MK': None,
        },
    },
    'home-credit':{
        'applications':{
            'PK': 'SK_ID_CURR',
            'Date': [],
            'MK': ['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_TYPE_SUITE'],
        },
        'bureau':{
            'PK': 'SK_ID_BUREAU',
            'Date': [],
            'MK': None,
        },
        'bureau_balance':{
            'PK': None,
            'Date': [],
            'MK': None,
        },
        'installments_payments':{
            'PK': None,
            'Date': [],
            'MK': None,
        },
        'cash_balance':{
            'PK': None,
            'Date': [],
            'MK': ['NAME_CONTRACT_STATUS'], 
        },
        'credit_balance':{
            'PK': None,
            'Date': [],
            'MK': ['NAME_CONTRACT_STATUS'],
        },
        'previous_application':{
            'PK': 'SK_ID_PREV',
            'Date': [],
            'MK': ['NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS', 'WEEKDAY_APPR_PROCESS_START', 'NAME_TYPE_SUITE'],
        },
    },
    'kdd15':{
        'course':{
            'PK': 'course_id',
            'Date': ['from', 'to'],
            'MK': None,
        },
        'enrollment':{
            'PK': 'enrollment_id',
            'Date': [],
            'MK': None,
        },
        'log': {
            'PK': None,
            'Date': ['time'],
            'MK': ['module_id'],
        },
        'object': {
            'PK': None,
            'Date': ['start'],
            'MK': ['module_id'],
        },
    }
}

# is_table = False means it is dummy table
# The first schema is for not creating dummy table
graph_config = {
    #/** acquire-valued-shoppers **/
    'acquire-valued-shoppers' :[
        {
        'History' :{
            'PK': 'id',
            'FK': {
                'offers' : 'offer',},# table name(nodename) : column name
            'Cat': ['market', 'year', 'month', 'day', 'wday'],
            'Con': None,
            'is_table': True,
        },
        'offers' :{
            'PK': 'offer',
            'FK': None,
            'Cat': ['category', 'company', 'brand'],
            'Con': ['quantity', 'offervalue'],
            'is_table': True,
        },
        'transactions': {
            'PK': None,
            'FK': {
                'History': 'id', },
            'Cat': ['chain', 'dept', 'category', 'company', 'brand', 'productmeasure', 'year', 'month', 'day', 'wday'],
            'Con': ['productsize', 'purchasequantity', 'purchaseamount'],
            'is_table': True,
        }},
    ],
    #/** home-credit **/
    'home-credit': [{
        'applications':{
            'PK': 'SK_ID_CURR',
            'FK': None,
            'Cat': ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', \
                    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', \
                    'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', \
                    'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', \
                    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', \
                    'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', \
                    'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', \
                    'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'],
            'Con': ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', \
                    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', \
                    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START', 'EXT_SOURCE_1', 'EXT_SOURCE_2', \
                    'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', \
                    'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', \
                    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', \
                    'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', \
                    'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', \
                    'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', \
                    'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', \
                    'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', \
                    'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', \
                    'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'],
            'is_table': True,},
        'bureau': {
            'PK': 'SK_ID_BUREAU',
            'FK': {
                'applications': 'SK_ID_CURR',
            },
            'Cat': ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'],
            'Con': ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', \
                    'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY'],
            'is_table': True,},
        'bureau_balance':{
            'PK': 'SK_ID_BUREAU',
            'FK': {
                'bureau': 'SK_ID_BUREAU',
            },
            'Cat': ['STATUS'],
            'Con': ['MONTHS_BALANCE'],
            'is_table': True,},
        'installments_payments':{
            'PK': None,
            'FK': {
                'applications': 'SK_ID_CURR',
                'previous_application': 'SK_ID_PREV',
            },
            'Cat': None,
            'Con': ['NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'],
            'is_table': True,},
        'cash_balance': {
            'PK': None,
            'FK': {
                'applications': 'SK_ID_CURR',
                'previous_application': 'SK_ID_PREV',
            },
            'Cat': ['NAME_CONTRACT_STATUS'],
            'Con': ['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF'],
            'is_table': True,},
        'credit_balance': {
            'PK': None,
            'FK': {
                'applications': 'SK_ID_CURR',
                'previous_application': 'SK_ID_PREV',
            },
            'Cat': ['NAME_CONTRACT_STATUS'],
            'Con': ['MONTHS_BALANCE', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', \
                    'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL', \
                    'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', \
                    'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF'],
            'is_table': True,},
        'previous_application': {
            'PK': 'SK_ID_PREV',
            'FK': {
                'applications': 'SK_ID_CURR',
            },
            'Cat': ['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'NAME_CASH_LOAN_PURPOSE', \
                    'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', \
                    'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION', 'NFLAG_INSURED_ON_APPROVAL'],
            'Con': ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START', 'RATE_DOWN_PAYMENT', \
                    'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'DAYS_DECISION', 'SELLERPLACE_AREA', 'CNT_PAYMENT', 'DAYS_FIRST_DRAWING', \
                    'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION'],
            'is_table': True,},
        }
    ],
    'outbrain': [{
        'clicks': {
            'PK': None,
            'FK': {
                'events': 'display_id',
                'ad_content': 'ad_id',},
            'Cat': None,
            'Con': None,
            'is_table': True,},
        'events': {
            'PK': 'display_id',
            'FK': {
                'doc_meta': 'document_id',},
            'Cat': ['platform', 'timestamp_year', 'timestamp_month', 'timestamp_day', 'timestamp_wday', 'geo_location', 'uuid'],
            'Con': None,
            'is_table': True,},
        'ad_content':{
            'PK': 'ad_id',
            'FK':{
                'doc_meta': 'document_id',},
            'Cat': ['campaign_id', 'advertiser_id'],
            'Con': None,
            'is_table': True,},
        'doc_meta':{
            'PK': 'document_id',
            'FK': None,
            'Cat': ['source_id', 'publisher_id'],
            'Con': None,
            'is_table': True,},
        'doc_ent':{
            'PK': None,
            'FK': {
                'doc_meta': 'document_id',},
            'Cat': ['entity_id'],
            'Con': ['confidence_level'],
            'is_table': True,},
        'doc_topics':{
            'PK': None,
            'FK': {
                'doc_meta': 'document_id',},
            'Cat': ['topic_id'],
            'Con': ['confidence_level'],
            'is_table': True,},
        'doc_cat':{
            'PK': None,
            'FK': {
                'doc_meta': 'document_id',},
            'Cat': ['category_id'],
            'Con': ['confidence_level'],
            'is_table': True,},
        },
    ],
    'outbrain-full': [{
        'clicks': {
            'PK': None,
            'FK': {
                'events': 'display_id',
                'ad_content': 'ad_id',},
            'Cat': None,
            'Con': None,
            'is_table': True,},
        'events': {
            'PK': 'display_id',
            'FK': {
                'doc_meta': 'document_id',
                'user': 'uuid',},
            'Cat': ['platform', 'timestamp_year', 'timestamp_month', 'timestamp_day', 'timestamp_wday', 'geo_location'],
            'Con': None,
            'is_table': True,},
        'user':{
            'PK': 'uuid',
            'FK': None,
            'Cat': None,
            'Con': None,
            'is_table': True},
        'page_views':{
            'PK': None,
            'FK': {
                'user': 'uuid',
                'doc_meta': 'document_id',},
            'Cat': ['platform', 'timestamp_year', 'timestamp_month', 'timestamp_day', 'timestamp_wday', 'geo_location', 'traffic_source'],
            'Con': None,
            'is_table': True,},
        'ad_content':{
            'PK': 'ad_id',
            'FK':{
                'doc_meta': 'document_id',},
            'Cat': ['campaign_id', 'advertiser_id'],
            'Con': None,
            'is_table': True,},
        'doc_meta':{
            'PK': 'document_id',
            'FK': None,
            'Cat': ['source_id', 'publisher_id'],
            'Con': None,
            'is_table': True,},
        'doc_ent':{
            'PK': None,
            'FK': {
                'doc_meta': 'document_id',},
            'Cat': ['entity_id'],
            'Con': ['confidence_level'],
            'is_table': True,},
        'doc_topics':{
            'PK': None,
            'FK': {
                'doc_meta': 'document_id',},
            'Cat': ['topic_id'],
            'Con': ['confidence_level'],
            'is_table': True,},
        'doc_cat':{
            'PK': None,
            'FK': {
                'doc_meta': 'document_id',},
            'Cat': ['category_id'],
            'Con': ['confidence_level'],
            'is_table': True,},
        },
    ],
    'diginetica':[{
        'clicks':{
            'PK': None,
            'FK':{
                'products': 'itemId',
                'queries_meta': 'queryId',
            },
            'Cat': None,
            'Con': None,
            'is_table': True,},
        'products':{
            'PK': 'itemId',
            'FK': None,
            'Cat': ['categoryId'],
            'Con': ['pricelog2'],
            'is_table': True,},
        'queries_meta':{
            'PK': 'queyId',
            'FK': {
                'session': 'sessionId',
            },
            'Cat': ['categoryId', 'year', 'month', 'day', 'wday'],
            'Con': ['timeframe', 'duration'],
            'is_table': True,},
        'session':{
            'PK': 'sessionId',
            'FK': None,
            'Cat': None,
            'Con': None,
            'is_table': True,},
        'item_views':{
            'PK': None,
            'FK': {
                'products': 'itemId',
                'session': 'sessionId',
            },
            'Cat': ['year', 'month', 'day', 'wday'],
            'Con': ['timeframe'],
            'is_table': True,},
        'purchases':{
            'PK': None,
            'FK': {
                'products': 'itemId',
                'session': 'sessionId',
            },
            'Cat': ['ordernumber', 'year', 'month', 'day', 'wday'],
            'Con': ['timeframe'],
            'is_table': True,},
        },
    ],
    'kdd15':[{
        'course':{
            'PK': 'course_id',
            'FK': None,
            'Cat': ['from_year', 'from_month', 'from_day', 'from_wday', 'to_year', 'to_month', 'to_day', 'to_wday'],
            'Con': None,
            'is_table': True,},
        'enrollment':{
            'PK': 'enrollment_id',
            'FK': {
                'course': 'course_id',
            },
            'Cat': ['username'],
            'Con': None,
            'is_table': True,},
        'log':{
            'PK': None,
            'FK': {
                'enrollment': 'enrollment_id',
            },
            'Cat': ['time_year', 'time_month', 'time_day', 'time_wday', 'source', 'event', 'module_id'],
            'Con': None,
            'is_table': True,},
        'object':{
            'PK': None,
            'FK': {
                'course': 'course_id',
            },
            'Cat': ['module_id', 'category', 'children'],
            'Con': None,
            'is_table': True,},
    },],
    'synthetic':[{
        'a':{
            'PK': 'aId',
            'FK':{
                'b': 'bId',
                'c': 'cId',
            },
            'Cat': ['data1'],
            'Con': None,
            'is_table': True,},
        'b':{
            'PK': 'bId',
            'FK':{
                'c': 'cId',
            },
            'Cat': None,
            'Con': ['data1'],
            'is_table': True},
        'c': {
            'PK': 'cId',
            'FK': None,
            'Cat': None,
            'Con': ['data1'],
            'is_table': True},
        },
    ],
    'synthetic2':[{
        'a':{
            'PK': 'aId',
            'FK':{
                'b': 'bId',
                'c': 'cId',
            },
            'Cat': ['data1'],
            'Con': None,
            'is_table': True,},
        'b':{
            'PK': 'bId',
            'FK':{
                'c': 'cId',
                'd': 'dId',
            },
            'Cat': None,
            'Con': None,
            'is_table': True},
        'c': {
            'PK': 'cId',
            'FK': None,
            'Cat': None,
            'Con': ['data1'],
            'is_table': True},
        'd': {
            'PK': 'dId',
            'FK': None,
            'Cat': None,
            'Con': ['data1'],
            'is_table': True,},
        },
    ],
}
