#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 18:25:27 2017

@author: dave
"""

#import urllib, json
import requests
import os
import time
from os.path import join as pjoin

import pandas as pd

from openbenchmarking import xml2df

class TweakersPricewatch:

    def __init__(self, fpath=None):

        self.fpath = fpath
        xml = xml2df()
        if fpath is None:
            self.fpath = pjoin(xml.db_path, '..', 'tweakers-pricewatch.h5')

        self.df = pd.DataFrame()
        if os.path.isfile(self.fpath):
            self.df = pd.read_hdf(self.fpath, 'table')

        self.cols = {'search':[], 'name':[], 'minPrice':[], 'id':[],
                     'link':[], 'rating':[], 'thumbnail':[], 'type':[]}
        self.cols_less = set(list(self.cols.keys())) - set(['minPrice', 'search'])

        self.df_dict_all = {k:[] for k in self.cols}

    def request(self, search):
        """JSON return object can contain various keys, but these seems to be
        consistent:
            id, link, name, thumbnail, type
        If available:
            minPrice
        """
        uri = 'https://tweakers.net/xmlhttp/xmlHttp.php'
        params = {'application' : 'sitewidesearch',
                  'type' : 'search',
                  'action' : 'pricewatch',
                  'keyword' : search,
                  'output' : 'json'}
        r = requests.get(uri, params=params)
        self.search = search
        # if response type was set to JSON, then you'll automatically have a JSON
        #response here...
        return r.json()

    def convert(self, data):
        """Convert Tweakers json response to a df_dict ordered structure.
        """

        df_dict = {k:[] for k in self.cols}

        items = []
        if 'entities' in data:
            items += data['entities']
        if 'articles' in data:
            items += data['articles']

        for item in items:

            df_dict['search'].append(self.search)

            min_price = -1
            if 'minPrice' in item:
                tmp = item['minPrice'].split('&euro; ')[1].split('<')[0]
                # remove . as thousand separator, replace comma with dot decimal
                # and remove possible - sign after decimal separator
                tmp = tmp.replace('.', '').replace(',', '.').replace('-', '')
                min_price = float(tmp)
            df_dict['minPrice'].append(min_price)

            for key in self.cols_less:
                val = ''
                if key in item:
                    val = item[key]
                df_dict[key].append(val)

        return df_dict

    def append2df_dict(self, df_dict):

        for key, val in df_dict.items():
            self.df_dict_all[key].extend(val)

    def df_append(self, save=True):
        """Add search results to database and save
        """
        self.df = self.df.append(pd.DataFrame(self.df_dict_all),
                                 ignore_index=True)
        if save:
            self.df.to_hdf(self.fpath, 'table')

    def search_keys(self, search_keys, save=True, sleeptime=0.8):
        """
        """

        for key in search_keys:
            time.sleep(sleeptime)
            data = self.request(key)
            df_dict = self.convert(data)
            self.append2df_dict(df_dict)
        self.df_append(save=save)


#url = 'https://tweakers.net/xmlhttp/xmlHttp.php?application=sitewidesearch&' \
#      'type=search&action=pricewatch&keyword=7990&output=json'
#response = urllib.request.urlopen(url)
#data_bytes = response.read()
#data = json.loads(data_bytes.decode('utf-8'))

if __name__ == '__main__':

    xml = xml2df()

#    uri = 'https://tweakers.net/xmlhttp/xmlHttp.php'
#    params = {'application' : 'sitewidesearch',
#              'type' : 'search',
#              'action' : 'pricewatch',
#              'keyword' : 'AMD Opteron 6174',
#              'output' : 'json'}
#    r = requests.get(uri, params=params)
#    # if response type was set to JSON, then you'll automatically have a JSON
#    #response here...
#    data = r.json()
#    min_price = data['entities'][0]['minPrice'].split('&euro; ')[1].split('<')[0]
#    min_price = float(min_price.replace(',', '.'))
#    print(params['keyword'])
#    print(min_price)

#    df = pd.read_hdf(pjoin(xml.pts_local, 'database.h5'), 'table')
#    sel = df[df['ResultIdentifier']=='pts/graphics-magick-1.4.1']
#    sorted(sel['ProcessorName'].unique())[6:10]
#    tpw = TweakersPricewatch()
#    search_keys = sorted(sel['ProcessorName'].unique())[6:10]
#    for key in search_keys:
#        data = tpw.request(key)
#        df_dict = tpw.convert(data)
#        tpw.append2df_dict(df_dict)
#    tpw.df_append(save=False)

#    tpw.search_keys(search_keys, save=False)
#    tpw.df[['search', 'name', 'minPrice']]

    # using lxml is much slower
    #tree = fromstring(data['entities'][0]['minPrice'])
    #min_price = tree.cssselect('a')[0].text.split('€ ')[1]
    #min_price = fromstring(data['entities'][0]['minPrice']).text.split('€ ')[1]
