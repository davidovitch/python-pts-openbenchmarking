#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 22:31:03 2017

@author: dave
"""

from os.path import join as pjoin
from datetime import date

import numpy as np
import pandas as pd

from openbenchmarking import (EditXML, xml2df, explore_dataset, plot_barh,
                              plot_barh_groups, search_openbm,
                              load_local_testids, find_items_in_field,
                              find_results_with_items_in_field)


def example_xml():
    """Manually select a bunch of cases and select only those systems and
    resuls that are of importance.
    """

    search_hardware = 'RX 480'
    search_descr = '1920 x 1080'

#    # download all of them seperately
#    for test_result in cases:
#        print('='*10, test_result)
#        obm = EditXML()
#        obm.load(obm.url_base.format(test_result))
#        id_rename = obm.remove(search_tests, search_hardware, search_descr)
#        obm.cleanup(id_rename)
#        obm.write_local()

    # heaven and RX 480, ALL
    #for k in url.split(','): print("'"+k+"',")
    cases1 = ['1608171-LO-AMDGPUPRO13',
            '1606281-HA-RX480LINU80',
            '1608101-KH-1608099KH32',
            '1606304-HA-1606297HA97',
            '1606302-HA-RADEONRX435',
            '1608200-PTS-AMDGPUPR97',
            '1608186-KH-1608171LO49',
            '1610315-TA-AMDGPUPRO08',
            '1606288-HA-RX480LINU56',
            '1607160-LO-AMDGPUVSM87',
            '1607103-LO-RX480OVER58',
            '1608286-LO-WINDOWS1070',
            '1606302-HA-RADEONRX476',
            '1607091-LO-LINUX48DR92',
            '1606297-HA-RADEONRX427',
            '1606299-HA-RX480DRIV74',
            '1607171-LO-GTX1060BE75',
            '1608099-KH-1606281HA62',
            '1607015-HA-1606297HA02',
            '1611267-LO-HEAVEN35158',
            '1607173-LO-GTX1060PO50',
            '1606300-HA-1606297HA15',
            '1607014-HA-1606297HA25',
            '1606300-HA-1606300HA08',
            '1608192-KH-1608101KH50',
            '1606306-GA-1606297HA13',
            '1609136-LO-MULTICARD22',
            '1606309-KH-1606297HA78',
            '1608274-LO-44444267033',
            '1609257-LO-NEWSYSTEM60',
            '1609180-LO-RX370MESA29',
            '1610142-LO-TEST0114153']
    # load all from the same source
#    search_tests = 'unigine-heaven'
#    obm = EditXML()
#    url = obm.url_base.format(','.join(cases))
#    obm.load(url)
#    id_rename = obm.remove(search_tests, search_hardware, search_descr)
#    obm.cleanup(id_rename)
#    obm.write_local(test_result='{}-rx-480-1920x1080'.format(search_tests))

    # tropics, rx 480
    cases2 = ['1609109-KH-1609104KH60',
             '1609136-LO-MULTICARD22',
             '1607147-HA-TROPICS6107',
             '1609104-KH-A10UNIGIN40']
    # load all from the same source
#    search_tests = 'unigine-tropics'
#    obm = EditXML()
#    url = obm.url_base.format(','.join(cases))
#    obm.load(url)
#    id_rename = obm.remove(search_tests, search_hardware, search_descr)
#    obm.cleanup(id_rename)
#    obm.write_local(test_result='{}-rx-480-1920x1080'.format(search_tests))

    # valley, rx 480
    cases3 = ['1608171-LO-AMDGPUPRO13',
            '1606281-HA-RX480LINU80',
            '1608101-KH-1608099KH32',
            '1606304-HA-1606297HA97',
            '1606302-HA-RADEONRX435',
            '1608200-PTS-AMDGPUPR97',
            '1608186-KH-1608171LO49',
            '1611237-KH-NOV22270868',
            '1607160-LO-AMDGPUVSM87',
            '1607103-LO-RX480OVER58',
            '1608286-LO-WINDOWS1070',
            '1606302-HA-RADEONRX476',
            '1606297-HA-RADEONRX427',
            '1606299-HA-RX480DRIV74',
            '1606287-HA-RX480DOLL44',
            '1608099-KH-1606281HA62',
            '1607171-LO-GTX1060BE75',
            '1607015-HA-1606297HA02',
            '1607178-LO-GTX1060MO36',
            '1606300-HA-1606297HA15',
            '1607014-HA-1606297HA25',
            '1608192-KH-1608101KH50',
            '1606300-HA-1606300HA08',
            '1611233-KH-TGM74906897',
            '1606306-GA-1606297HA13',
            '1606309-KH-1606297HA78',
            '1608278-GNAR-160827263',
            '1607134-KH-1606112KH22',
            '1608272-LO-55555650923',
            '1609180-LO-RX370MESA29',
            '1607040-KH-JULIEN50266',
            '1608238-HA-UNIGINEVA24']
#    search_tests = 'unigine-valley'
#    obm = EditXML()
#    url = obm.url_base.format(','.join(cases))
#    obm.load(url)
#    id_rename = obm.remove(search_tests, search_hardware, search_descr)
#    obm.cleanup(id_rename)
#    obm.write_local(test_result='{}-rx-480-1920x1080'.format(search_tests))

    # sanctuary, rx 480
    cases4 = ['1609109-KH-1609104KH60',
            '1609107-KH-A1062869437',
            '1609104-KH-A10UNIGIN40',
            '1607125-HA-SANTUARY143']
#    search_tests = 'unigine-sanctuary'
#    obm = EditXML()
#    url = obm.url_base.format(','.join(cases))
#    obm.load(url)
#    id_rename = obm.remove(search_tests, search_hardware, search_descr)
#    obm.cleanup(id_rename)
#    obm.write_local(test_result='{}-rx-480-1920x1080'.format(search_tests))

    # ========================================
    cases = cases1 + cases2 + cases3 + cases4
    cases=set(cases)
    search_tests = 'unigine-'
    obm = EditXML()
    url = obm.url_base.format(','.join(cases))
    obm.load(url)
    id_rename = obm.remove(search_tests, search_hardware, search_descr)
    obm.cleanup(id_rename)
    obm.write_local(test_result='{}-rx-480-1920x1080'.format('unigine-all'))


def example1_dataframe():

    # load previously donwload data
    xml = xml2df()
    df = pd.read_hdf(pjoin(xml.res_path, 'search_rx_470.h5'), 'table')

    df.drop(xml.user_cols, inplace=True, axis=1)
    df.drop_duplicates(inplace=True)

    # only R470 graphic cards
    res_find = df['Graphics'].str.lower().str.find('rx 470')
    # grp_lwr holds -1 for entries that do not contain the search string
    # we are only interested in taking the indeces of those entries that do
    # contain our search term, so antyhing above -1
    df_sel = df.loc[(res_find > -1).values]

    # now see for which tests we have sufficient data
    explore_dataset(df_sel, 'ResultIdentifier', 'ResultDescription', 'Processor')

    # select only a certain test
    df_sel = df_sel[df_sel['ResultIdentifier'] == 'pts/unigine-valley-1.1.4']

    # and the same version/resultion of said test
    seltext = 'Resolution: 1920 x 1080 - Mode: Fullscreen'
    sel = df_sel[df_sel['ResultDescription']==seltext].copy()
    # cast Value to a float64
    sel['Value'] = sel['Value'].astype(np.float64)
    # remove close to zero measurements
    sel = sel[(sel['Display Driver']!='None') &
              (sel['Value']>0.5)]

    # now we need to pivot the table into a different form:
    # each column is a different hardware/software combination, and each row
    # is another different variable (test/hardware/software)

    plot_barh(sel, 'Processor', label_xval='Value')
    plot_barh_groups(sel, 'Graphics', 'Processor', label_xval='Value')
#    plot_barh_groups(df, label_yval, label_grousp, label_xval='Value')

    # -------------------------------------------------------------------------
#    pp=df[:10]
#    grp_lwr = pp['Scale'].str.lower().str.find('watts')
#    isel = grp_lwr[grp_lwr > -1].index
#    pp['Scale'].loc[isel]


def example2_dataframe():

    # load previously donwload data
    xml = xml2df()
    df = pd.read_hdf(pjoin(xml.res_path, 'search_rx_470.h5'), 'table')
    df.drop(xml.user_cols, inplace=True, axis=1)
    df.drop_duplicates(inplace=True)

    # only RX 470 graphic cards
    df_find = df['Graphics'].str.lower().str.find('rx 470')
    # grp_lwr holds -1 for entries that do not contain the search string
    # we are only interested in taking the indeces of those entries that do
    # contain our search term, so antyhing above -1
    df_sel = df.loc[(df_find > -1).values]

    # now see for which tests we have sufficient data
    explore_dataset(df_sel, 'ResultIdentifier', 'ResultDescription', 'Processor')

    # select only a certain test
    df_sel = df_sel[df_sel['ResultIdentifier'] == 'pts/xonotic-1.4.0']

    # and the same version/resultion of said test
    seltext = 'Resolution: 1920 x 1080 - Effects Quality: Ultimate'
    sel = df_sel[df_sel['ResultDescription']==seltext].copy()
    # cast Value to a float64
    sel['Value'] = sel['Value'].astype(np.float64)
    # remove close to zero measurements, and those cases where the Display
    # Driver field got lost
#    sel = sel[(sel['Display Driver']!='None') &
#              (sel['Value']>0.5)]

    qq = sel[sel['Processor']==' Intel Core i5-4670K @ 3.80GHz (4 Cores)']
    for col in qq:
        print(len(qq[col].unique()), col)
        if len(qq[col].unique()) > 1:
            print('******', qq[col].unique())

    # now we need to pivot the table into a different form:
    # each column is a different hardware/software combination, and each row
    # is another different variable (test/hardware/software)

#    plot_barh(sel, 'Processor', label_xval='Value')
    plot_barh_groups(sel, 'Graphics', 'Processor', label_xval='Value')


def example3_dataframe():

    # load previously donwload data
    xml = xml2df()
    df = pd.read_hdf(pjoin(xml.res_path, 'search_rx_470.h5'), 'table')

    df.drop(xml.user_cols, inplace=True, axis=1)
    df.drop_duplicates(inplace=True)

    # select only subset of data, and plot
    # only R470 graphic cards
    res_find = df['Graphics'].str.lower().str.find('rx 470')
    # grp_lwr holds -1 for entries that do not contain the search string
    # we are only interested in taking the indeces of those entries that do
    # contain our search term, so antyhing above -1
    df_sel = df.loc[(res_find > -1).values]

    explore_dataset(df_sel, 'ResultIdentifier', 'ResultDescription', 'Processor')

    # select only a certain test
    df_sel = df_sel[df_sel['ResultIdentifier'] == 'pts/xonotic-1.4.0']

    # and the same version/resultion of said test
    seltext = 'Resolution: 3840 x 2160 - Effects Quality: Ultimate'
    sel = df_sel[df_sel['ResultDescription']==seltext].copy()
    # cast Value to a float64
    sel['Value'] = sel['Value'].astype(np.float64)
    # remove close to zero measurements
#    sel = sel[(sel['Display Driver']!='None') &
#              (sel['Value']>0.5)]

    # now we need to pivot the table into a different form:
    # each column is a different hardware/software combination, and each row
    # is another different variable (test/hardware/software)

#    plot_barh(sel, 'Processor', label_xval='Value')
    plot_barh_groups(sel, 'Graphics', 'Processor', label_xval='Value')


def example(search_string):

    # load previously donwload data
    xml = xml2df()
#    df = pd.read_hdf(pjoin(xml.res_path, 'search_rx_470.h5'), 'table')
    testids = search_openbm(search_string, save_xml=True)

    # TODO: create method to convert a list of testids into a DataFrame
    # either load locally or from remote

    df.drop(xml.user_cols, inplace=True, axis=1)
    df.drop_duplicates(inplace=True)

    # select only subset of data, and plot
    # only R470 graphic cards
    res_find = df['Graphics'].str.lower().str.find('rx 470')
    # grp_lwr holds -1 for entries that do not contain the search string
    # we are only interested in taking the indeces of those entries that do
    # contain our search term, so antyhing above -1
    df_sel = df.loc[(res_find > -1).values]

    explore_dataset(df_sel, 'ResultIdentifier', 'ResultDescription', 'Processor')

    # select only a certain test
    df_sel = df_sel[df_sel['ResultIdentifier'] == 'pts/xonotic-1.4.0']

    # and the same version/resultion of said test
    seltext = 'Resolution: 3840 x 2160 - Effects Quality: Ultimate'
    sel = df_sel[df_sel['ResultDescription']==seltext].copy()
    # cast Value to a float64
    sel['Value'] = sel['Value'].astype(np.float64)
    # remove close to zero measurements
#    sel = sel[(sel['Display Driver']!='None') &
#              (sel['Value']>0.5)]

    # now we need to pivot the table into a different form:
    # each column is a different hardware/software combination, and each row
    # is another different variable (test/hardware/software)

#    plot_barh(sel, 'Processor', label_xval='Value')
    plot_barh_groups(sel, 'Graphics', 'Processor', label_xval='Value')


def testA_vs_testB():
    """Find the corrolation between two benchmark results
    """

    xml = xml2df()
    df = pd.read_hdf(pjoin(xml.db_path, 'database.h5'), 'table')
    df['ProcessorFrequency'] = df['ProcessorFrequency'].astype(str)
    df['ProcessorFrequency'] = df['ProcessorFrequency'].str.replace('GHz', '')
    df['ProcessorFrequency'] = df['ProcessorFrequency'].astype(np.float32)
    resid1 = 'pts/graphics-magick-1.4.1'
    resid2 = 'pts/c-ray-1.1.0'
    sel = df[((df['ResultIdentifier']==resid1) |
              (df['ResultIdentifier']==resid2)) & (df['ResultOf']=='no') &
             (df['ProcessorFrequency']<20)]
    sel = sel.copy()
    sel['Value'] = sel['Value'].astype(np.float32)
    sel['ProcessorCores'] = sel['ProcessorCores'].astype(np.float32)

    test1, test2 = [], []
    for hwhash, gr_hw in sel.groupby('HardwareHash'):
        if len(gr_hw['ResultIdentifier'].unique()) < 2:
            continue
        print(hwhash)
        for test, gr_test in gr_hw.groupby('ResultIdentifier'):
            print('   ', test.ljust(30), '{: 3d}'.format(len(gr_test)),
                  '{: 9.02f}'.format(gr_test['Value'].mean()),
                  '{: 9.02f}'.format(gr_test['Value'].std()),
                  '{: 9.02f}'.format(gr_test['Value'].min()),
                  '{: 9.02f}'.format(gr_test['Value'].max()))

    # create a pivot table
    pivot = pd.pivot_table(sel, values='Value',
                           index='HardwareHash',
                           columns='ResultIdentifier')#,
#                           aggfunc=[np.mean, np.std, np.min, np.max, np.sum])
    # remove all harware configurations that only have one test result
    data = pivot.dropna(axis=0)


def cross_plot():

    # create a plot like with joint distributions and their cross distributions

    # for example: 25 systems, 21 tests
    # http://openbenchmarking.org/result/1603119-GA-1401227SO30
    # AMD vs Intel, non gaming
    xml = xml2df()
    testid = '1701157-RI-CLEARPATC00' # 2 systems, 11 tests
    testid = '1603119-GA-1401227SO30' # 25 systems, 21 tests
    xml.load_testid_from_obm(testid, save_xml=True, use_cache=True)
    df_dict = xml.xml2dict()
    df = xml.dict2df(df_dict)
#    df = pd.DataFrame(df_dict)
    pivot = pd.pivot_table(df, values='Value', index='SystemIdentifier',
                           columns='ResultIdentifier')

    # 73 systems, 23 tests
    # games, 12/2016
    # http://openbenchmarking.org/result/1612283-DARK-HD7950R28


def find_hardware():
    """Example to look for hardware and compare results from a given test
    """

    xml = xml2df()
    df = pd.read_hdf(pjoin(xml.db_path, 'database.h5'), 'table')

    field = 'ProcessorName'
    search_items = ['4850e', 'x2 555']
    df_sel = find_items_in_field(df, search_items, field)
    # find tests for which we have all targets have a data point
    resids = find_results_with_items_in_field(df_sel, search_items, field)

    # list all results
    for key, values in resids.items():
        print(key)
        for value in values:
            print('    ', value)

    explore_dataset(df_sel, 'ResultIdentifier', 'ResultDescription',
                    'Processor', min_cases=2)

    # plot a single ResultIdentifier. Note that this might contain several
    # variations of the test, as defined in the ResultDescription
    sel = df_sel[df_sel['ResultIdentifier']=='pts/encode-mp3-1.3.1']
    fig, ax = plot_barh_groups(sel, 'Graphics', 'Processor', label_xval='Value')

    # plot all common results
    for resid, values in resids.items():
        sel1 = df_sel[df_sel['ResultIdentifier']==resid]
        for resdescr in values:
            sel2 = sel1[sel1['ResultDescription']==resdescr]
            fig, ax = plot_barh_groups(sel2, 'Graphics', 'Processor',
                                       label_xval='Value')
            ax.set_title(resid + '\n' + resdescr)


def database():
    """
    """

    # manual XML file changes:

    # there is one case that has (Total Cores: 4) instead of (4 Cores)
    # for the Processor tag.
    # http://openbenchmarking.org/result/1108293-IV-ZAREASONL67


    xml = xml2df()
    df = pd.read_hdf(pjoin(xml.db_path, 'database_results.h5'), 'table')
    df_sys = pd.read_hdf(pjoin(xml.db_path, 'database_systems.h5'), 'table')

    field = 'Disk'
    search_items = ['960 evo', '950 evo', 'KC1000', 'RD400']
    df_sel = find_items_in_field(df_sys, search_items, field)

#     field = 'ResultTitle'
#     search_items = ['flexible io tester']
#     df_sel = find_items_in_field(df, search_items, field)

    df_sel = pd.merge(df_sel, df, left_on='SystemHash', right_on='SystemHash',
                      how='left', sort=False)
    resids = find_results_with_items_in_field(df_sel, search_items, field,
                                              gr1='ResultTitle')

    for resid, values in resids.items():
        sel1 = df_sel[df_sel['ResultTitle']==resid]
        print(resid, sel1['ResultIdentifier'].unique())
        for resdescr in values:
            sel2 = sel1[sel1['ResultDescription']==resdescr]
            if len(sel2) > 1:
                print('    % 3i  %s' % (len(sel2), resdescr))

    # plot some results
    resid_manual = {'pts/xonotic-1.3.1':['Resolution: 1920 x 1080 - Effects Quality: High'],
                    'pts/unigine-sanctuary-1.5.2':['Resolution: 1920 x 1080']}
                #'pts/c-ray-1.1.0':['Total Time'],
                #'pts/dota2-1.2.1':['Resolution: 1920 x 1080 - Renderer: OpenGL'],
                #'pts/unigine-heaven-1.6.2':['Resolution: 1920 x 1080 - Mode: Fullscreen'],
                #'pts/unigine-valley-1.1.4':['Resolution: 1920 x 1080 - Mode: Fullscreen']}
    for resid, values in resid_manual.items():
        sel1 = df_sel[df_sel['ResultIdentifier']==resid]
        for resdescr in values:
            sel2 = sel1[sel1['ResultDescription']==resdescr]
            if len(sel2) == 0:
                continue
            elif len(sel2) > 2:
                fig, ax = openbm.plot_barh_groups(sel2, 'ProcessorName',
                                                  'Graphics', label_xval='Value')
            else:
                fig, ax = openbm.plot_barh(sel2, 'ProcessorName',
                                           label_xval='Value')
            ax.set_title(resid + '\n' + resdescr)


def example_db_split_gpu():

    field = 'Graphics'
    search_items = ['rx 460', 'R7 260X', 'hd 6950', 'hd 7950', 'hd 7790']
    search_items = ['rx 460', 'R7 260X']
    search_items = ['rx 460', 'hd 7790', 'hd 7950', 'R7 260X']

    gr1 = 'ResultIdentifier'

    xml = xml2df()
    #df = pd.read_hdf(pjoin(xml.db_path, 'database_categoricals.h5'), 'table')
    df = pd.read_hdf(pjoin(xml.db_path, 'database_results.h5'), 'table')
    df_sys = pd.read_hdf(pjoin(xml.db_path, 'database_systems.h5'), 'table')

    #search_items = ['r7 250']
    df_sel = find_items_in_field(df_sys, search_items, field)
    df_sel = pd.merge(df_sel, df, left_on='SystemHash', right_on='SystemHash',
                      how='left', sort=False)
    resids = find_results_with_items_in_field(df_sel, search_items, field,
                                              gr1=gr1)

    for key, values in resids.items():
        print(key)
        for value in values:
            print('    ', value)

    for resid, values in resids.items():
        sel1 = df_sel[df_sel['ResultIdentifier']==resid]
        print(resid, sel1['ResultIdentifier'].unique())
        for resdescr in values:
            sel2 = sel1[sel1['ResultDescription']==resdescr]
            if len(sel2) > 1:
                print('    % 3i  %s' % (len(sel2), resdescr))

    resid_manual = {
        'pts/xonotic-1.3.1':['Resolution: 1920 x 1080 - Effects Quality: High',
                             'Resolution: 3840 x 2160 - Effects Quality: Ultimate'],
        'pts/unigine-heaven-1.6.2':['Resolution: 1920 x 1080 - Mode: Fullscreen'],
        'pts/c-ray-1.1.0':['Total Time'],
        'pts/gputest-1.3.1':['pts/gputest-1.3.1'],
        'pts/bioshock-infinite-1.0.1':['Resolution: 1920 x 1080 - Effects Quality: Ultra'],
        }
        #'pts/unigine-sanctuary-1.5.2':['Resolution: 1920 x 1080']}
        #'pts/c-ray-1.1.0':['Total Time'],
        #'pts/dota2-1.2.1':['Resolution: 1920 x 1080 - Renderer: OpenGL'],
        #'pts/unigine-heaven-1.6.2':['Resolution: 1920 x 1080 - Mode: Fullscreen'],
        #'pts/unigine-valley-1.1.4':['Resolution: 1920 x 1080 - Mode: Fullscreen']}

    for resid, values in resid_manual.items():
        sel1 = df_sel[df_sel['ResultIdentifier']==resid]
        for resdescr in values:
            sel2 = sel1[sel1['ResultDescription']==resdescr]
            if len(sel2) == 0:
                continue
            elif len(sel2) > 2:
                fig, ax = plot_barh_groups(sel2, 'ProcessorName', 'Graphics',
                                           label_xval='Value')
            else:
                fig, ax = plot_barh(sel2, 'ProcessorName', label_xval='Value')
            ax.set_title(resid + '\n' + resdescr)

    # Plot all results
    for resid, values in resids.items():
        sel1 = df_sel[df_sel[gr1]==resid]
        for resdescr in values:
            sel2 = sel1[sel1['ResultDescription']==resdescr]
            if len(sel2) <= 1:
                continue
            fig, ax = plot_barh_groups(sel2, 'ProcessorName', 'Graphics',
                                       label_xval='Value')
            ax.set_title(resid + '\n' + resdescr)

if __name__ == '__main__':

    xml = xml2df()

    # -------------------------------------------------------------------------
    # MERGE ALL LOCALLY SAVED XML FILES INTO A DataFrame
#    df, failed = load_local_testids()
#    df.to_hdf(pjoin(xml.db_path, 'database.h5'), 'table')
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
#    obm = xml2df()
#    # load a locally saved testid XML file
#    io = pjoin(obm.res_path, "1606281-HA-RX480LINU80/composite.xml")
#    obm.load(io)
#    dict_sys = obm.generated_system2dict()
#    dict_res = obm.data_entry2dict()

#    obm = xml2df()
#    # download testid XML file from OpenBenchmarking
#    obm.load_testid('1606281-HA-RX480LINU80')
#    dict_sys = obm.generated_system2dict()
#    dict_res = obm.data_entry2dict()
