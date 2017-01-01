# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:03:26 2016

@author: davidovitch
"""

import os
from os.path import join as pjoin

from lxml.html import fromstring
from lxml import etree
import urllib.request

from tqdm import tqdm
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


class OpenBenchMarking:

    def __init__(self):
        self.pts_local = pjoin(os.environ['HOME'],
                               '.phoronix-test-suite/test-results/')
        self.url_base = 'http://openbenchmarking.org/result/{}&export=xml'
        self.hard_soft_tags = set(['Hardware', 'Software'])
        self.testid = 'unknown'
        self.user_cols = ['User', 'SystemDescription', 'testid', 'Notes',
                          'SystemIdentifier', 'GeneratedTitle', 'LastModified']

    def load_testid(self, testid):
        """

        Parameters
        ----------

        testid : str
            OpenBenchemarking.org testid, for example: 1606281-HA-RX480LINU80
        """
        self.testid = testid
        self.load(self.url_base.format(testid))

    def load(self, io):

        tree = etree.parse(io)
        self.root = tree.getroot()
        self.io = io

    def get_all_profiles(self, search_string):
        """Return a list of test profile id's
        """

        url = 'http://openbenchmarking.org/s/{}&show_more'.format(search_string)
        response = urllib.request.urlopen(urllib.parse.quote(url, safe='/:'))
        data = response.read()      # a bytes object
        text = data.decode('utf-8') # str; can't be used if data is binary

        tree = fromstring(text)
        # all profiles names are in h4 elements, and nothing else is, nice and easy
        # but if a title is given, the id is in the parent link
        # url starts with /result/
        ids = [k.getparent().attrib['href'][8:] for k in tree.cssselect('h4')]

        return ids


class EditXML(OpenBenchMarking):

    def __init__(self):
        super().__init__()

    def merge(self, list_test_results):
        """DOESN'T MERGE ANYTHING YET
        """
        self.root = etree.Element('PhoronixTestSuite')
        for test_result in list_test_results:
            fpath = os.path.join(self.pts_local, test_result, 'composite.xml')
            tree = etree.parse(fpath)
            root = tree.getroot()

    def write_local(self, test_result=None):
        if test_result is None:
            test_result = self.test_result
        fpath = os.path.join(self.pts_local, test_result)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        fname = os.path.join(fpath, 'composite.xml')
        with open(fname, 'w') as f:
            f.write(etree.tostring(self.root).decode())

    def remove(self, search_tests, search_hardware, search_descr):

        sys_rem = []
        id_rename = {}

        # root.Generated, root.System, root.Result
        for el in self.root.findall('System'):
            print('--------', el.tag)
            # only keep RX 480 systems
            hardware = el.find('Hardware')
            hardware_els = hardware.text.split(', ')
            hardware_dict = {k.split(':')[0]:k.split(':')[1] for k in hardware_els}
#            for k in hardware_els:
#                print("'" + k.split(':')[0] + "', ", end='')

            software = el.find('Software')
            software_els = software.text.split(', ')
            software_dict = {k.split(':')[0]:k.split(':')[1] for k in software_els}
#            for k in software_els:
#                print("'" + k.split(':')[0] + "', ", end='')

#            cpu = hardware_dict['Processor'].strip()
#            try:
#                kernel = ' - ' + hardware_dict['Kernel'].strip()
#            except:
#                kernel = ''
#            id_rename[el.find('Identifier').text] = cpu + kernel
            cpu = hardware_dict['Processor'].strip() + ' - '
            identifier = el.find('Identifier').text
            id_rename[identifier] = cpu.split(' @ ')[0] + ' // ' + identifier
            if hardware.text.find(search_hardware) > -1:
                print(hardware_dict['Processor'])
            else:
                sys_rem.append(el.find('Identifier').text)
                el.getparent().remove(el)

        sys_rem = set(sys_rem)
        for el in self.root.findall('Result'):
            print('--------', el.tag)
            # is this the right test at the right description?
            try:
                find_id = el.find('Identifier').text.find(search_tests)
                find_descr = el.find('Description').text.find(search_descr)
            except:
                find_id = -1
                find_descr = -1
            if find_id > -1 and find_descr > -1:
                print(el.find('Identifier').text)
                # only keep systems that have not been filtered out
                for entry in el.find('Data').getchildren():
                    sys_id = entry.find('Identifier').text
                    if sys_id in sys_rem:
                        entry.getparent().remove(entry)
            else:
                el.getparent().remove(el)

        return id_rename

    def cleanup(self, id_rename):
        # keep track of all systems that have test results, remove the ones
        # that have been filtered out
        sys_list = []
        for el in self.root.findall('Result'):
            for entry in el.find('Data').getchildren():
                identifier = entry.find('Identifier')
                sys_list.append(identifier.text)
                # rename identifiers
#                entry.set('Identifier', id_rename[identifier.text])
                identifier.text = id_rename[identifier.text]
        sys_list = set(sys_list)
        for el in self.root.findall('System'):
            identifier = el.find('Identifier')
            if identifier.text not in sys_list:
                el.getparent().remove(el)
            else:
                # rename identifier
#                el.set('Identifier', id_rename[identifier.text])
                identifier.text = id_rename[identifier.text]


class xml2df(OpenBenchMarking):

    def __init__(self, io=None, testid=None):
        super().__init__()

        if io is not None:
            self.load(io)
        elif testid is not None:
            self.load_testid(testid)

    def convert(self):

        df_sys = self.generated_system2df()
        df_sys.rename(columns={'Description':'SystemDescription',
                               'Identifier':'SystemIdentifier',
                               'JSON':'SystemJSON',
                               'Title':'GeneratedTitle'}, inplace=True)
        df_res = self.data_entry2df()
        df_res.rename(columns={'Description':'ResultDescription',
                               'Identifier':'ResultIdentifier',
                               'JSON':'DataEntryJSON',
                               'Title':'ResultTitle'}, inplace=True)
        # drop the SystemIdentifier column since it is already part of df_sys
        df_res.drop('SystemIdentifier', inplace=True, axis=1)

        df = pd.merge(df_sys, df_res, left_index=True, right_on='SystemIndex')
        # after merging both, SystemIndex is now obsolete
        df.drop('SystemIndex', inplace=True, axis=1)

        # convert object columns to string, but leave other data types as is
        for col, dtype in df.dtypes.items():
            if isinstance(dtype, object):
                df[col] = df[col].values.astype(np.str)

        return df

    def _split2dict(self, string):
        """Convert following string to dictionary:
        key1: value1, key2: value2, ...
        """
        elements = string.split(', ')
        return {k.split(':')[0]:k.split(':')[1] for k in elements}

    def _add2row(self, elements, columns, df_dict, missing_val=None,
                 rename={}):
        """

        Elements with the tag Hardware and Software are split into multiple
        columns.

        Parameters
        ----------

        elements : list of lmxl elements

        columns : set
            columns names present in df_dict

        df_dict : dict
            pandas.DataFrame dictionary

        missing_val : str, default=None
            When an tag occurs in columns but not in elements, it is added to
            df_dict with missing_val as value. Rename is applied after the
            missing keys from columns are checked

        Returns
        ------

        df_dict : dict
            pandas.DataFrame dictionary with one added row for all the columns
            of the set columns. Elements should be a sub-set of columns.
            Values occuring in columns but not in elements are added as None.

        """

        # make sure that all containing elements are used, and that
        # missing ones are filled in as empty to preserve a valid
        # DataFrame dictionary
        found_els = []

        for el in elements:
            if el.tag in self.hard_soft_tags:
                # split the Hardware and Software tags into the columns
                tmp = self._split2dict(el.text)
                for key, value in tmp.items():
                    df_dict[key].append(value)
                    found_els.append(key)
            else:
                df_dict[el.tag].append(el.text)
                found_els.append(el.tag)

        # populate missing keys with an empty value
        for key in columns - set(found_els):
            df_dict[key].append(missing_val)

        # rename a certain column
        for key, value in rename.items():
            df_dict[value] = df_dict[key]
            df_dict.pop(key)

        return df_dict

    def generated_system2df(self):
        """For a given xml result file from pts/OpenBenchmarking.org, convert
        the Generated and System tags to a Pandas DataFrame. This means that
        the data contained in the Generated tag will now be repeated for each
        of the systems contained in the System tag.

        Now we duplicated data among different rows, which helps when
        searching/selecting.

        The Hardware and Software tags are split into multiple columns to
        facilitate a more fine grained searching and selection process.
        """

        generated = ['Title', 'LastModified', 'TestClient', 'Description',
                     'Notes', 'InternalTags', 'ReferenceID',
                     'PreSetEnvironmentVariables']
        system = ['Identifier', 'Hardware', 'Software', 'User', 'TimeStamp',
                  'TestClientVersion', 'Notes', 'JSON']
        hardware = ['Processor', 'Motherboard', 'Chipset', 'Memory', 'Disk',
                    'Graphics', 'Audio', 'Network', 'Monitor']
        software = ['OS', 'Kernel', 'Desktop', 'Display Server',
                    'Display Driver', 'OpenGL', 'OpenCL', 'Vulkan', 'Compiler',
                    'File-System', 'Screen Resolution']

        cols_sys = system + hardware + software
        cols_sys.remove('Hardware')
        cols_sys.remove('Software')

        generated_set = set(generated)
        system_set = set(cols_sys)
#        hardware_set = set(hardware)
#        software_set = set(software)

        dict_sys = {k:[] for k in cols_sys}

#        els_generated = self.root.findall('Generated')
#        dict_sys = self._add2row(els_generated, generated_set, dict_sys)
#
#        for key, value in dict_sys.items():
#            print(key.rjust(28), len(value))
#
#        els_generated = self.root.findall('System')
#        dict_sys = self._add2row(els_generated, system_set, dict_sys)
#
#        for key, value in dict_sys.items():
#            print(key.rjust(28), len(value))

        # there should only be one Generated element
        gen_elements = self.root.findall('Generated')
        assert len(gen_elements) == 1

        dict_gen = {el.tag : el.text for el in gen_elements[0]}
        # add empty values for possible missing keys
        for key in generated_set - set(dict_gen.keys()):
            dict_gen[key] = None
        # also include the URL testid identifier which is unique for each
        # test entry on OpenBenchmarking.org
        dict_gen['testid'] = self.testid

        # For each system create a row in the df_dict
        systems = self.root.findall('System')
        for sys_els in systems:
            dict_sys = self._add2row(sys_els, system_set, dict_sys)

        # sanity checks
        for key, value in dict_sys.items():
            if not len(systems) == len(value):
                rpl = [key, len(value), len(systems)]
                raise AssertionError('{} has {} elements instead of {}'.format(*rpl))

        # expand with the same values for Generated columns, this duplication
        # of data will make searching/selecting in the DataFrame later easier
        for key, value in dict_gen.items():
            dict_sys[key] = [value]*len(systems)

#        for key, value in dict_sys.items():
#            print(key.rjust(28), len(value))

        # the indices for Identifier in Results/Data/Entry
        self.ids = {key:i for i,key in enumerate(dict_sys['Identifier'])}

        return pd.DataFrame(dict_sys)

    def data_entry2df(self):
        """For a given xml result file from pts/OpenBenchmarking.org, convert
        the Result tags to a Pandas DataFrame. This means that the data
        contained in the Result tag will now be replicated for each of the
        Data/Entry tags for that given test result.
        """

        result = ['Identifier', 'Title', 'AppVersion', 'Arguments',
                  'Description', 'Scale', 'Proportion', 'DisplayFormat']
        data_entry = ['Identifier', 'Value', 'RawString', 'JSON']
        data_entry_ = ['SystemIdentifier', 'Value', 'RawString', 'JSON']
        data_set = set(data_entry)
        rename = {'Identifier':'SystemIdentifier'}

        dict_res = {k:[] for k in result+data_entry_}

        res_elements = list(self.root.findall('Result'))

        for res_el in res_elements:
            # get all the details of the test in question
            res_id = {k.tag:k.text for k in res_el}
            res_id.pop('Data')
            data_entries = res_el.find('Data')
#            # add each result to the collection
#            for entry in data_entries:
#                # start with empty but with all required keys dict
#                row = {k:None for k in data_entry}
#                # update with all tags found in the xml element
#                row.update({k.tag:k.text for k in entry})
#                # Identifier is used in both Result and Data, rename to
#                # SystemIdentifier, and remove Identifier
#                row['SystemIdentifier'] = row['Identifier']
#                row.pop('Identifier')
#            for key, value in row:
#                dict_res[key].append(value)
            # add each result to the collection
            tmp = {k:[] for k in data_entry}
            for entry in data_entries:
                tmp = self._add2row(entry, data_set, tmp)
            # before merging, rename Identifier
            for key, value in rename.items():
                tmp[value] = tmp[key]
                tmp.pop(key)
            # add with the rest
            for key, value in tmp.items():
                dict_res[key].extend(value)
            # and add the Result element columns to all the data entries
            for key, value in res_id.items():
                dict_res[key].extend([value]*len(data_entries))

        # add the index of the System as a column because SystemIdentifier is
        # not unique! Will be used when merging/joining with Generated/system
        dict_res['SystemIndex'] = []
        for identifier in dict_res['SystemIdentifier']:
            dict_res['SystemIndex'].append(self.ids[identifier])

#        for key, value in dict_res.items():
#            print(key.rjust(28), len(value))

        return pd.DataFrame(dict_res)


def download_from_openbm(search_string):

    # TODO: another de-duplication strategy: if the results are the same,
    # and some of the other differences are due to a None or something we
    # can also be confident we are talking about the same test case.

    df = pd.DataFrame()
    xml = xml2df()
    # get a list of test result id's from using the search function on obm.org
    testids = xml.get_all_profiles(search_string)
    for testid in tqdm(testids):
#        print(testid)
        # download each result xml file and convert to df
        xml.load_testid(testid)
        # save in one big dataframe
        df = df.append(xml.convert())
    # create a new unique index
    df.index = np.arange(len(df))

    # there are probably going to be more duplicates
    df.drop_duplicates(inplace=True)
    # columns that can hold different values but still could refer to the same
    # test data. So basically all user defined columns should be ignored.
    # But do not drop the columns, just ignore them for de-duplication
    cols = list(set(df.columns) - set(xml.user_cols))
    # mark True for values that are NOT duplicates
    df = df.loc[np.invert(df.duplicated(subset=cols).values)]

    # TODO: Value can also be a time series of measurements
    # convert mean values from float to text
#    df['Value'] = df['Value'].values.astype(np.float64)

    # save the DataFrame
#    df.to_hdf(pjoin(xml.pts_local, 'search_{}.h5'.format(search_string)), 'table')
#    df.to_excel(pjoin(xml.pts_local, 'search_rx_470.xlsx'))

    return df


def explore_dataset(df, label1, label2, label3):

    # how many cases per test, and how many per resolution
    df_dict = {'nr':[], 'test':[]}

    for grname, gr in df.groupby(label1):
        df_dict['nr'].append(len(gr))
        df_dict['test'].append(grname)
        print('{:5d} : {}'.format(len(gr), grname))

    df_tests = pd.DataFrame(df_dict)
    df_tests.sort_values('nr', inplace=True)
    for col in df_tests['test']:
        df_sel = df[df[label1]==col]
        print()
        print('{:5d} : {}'.format(len(df_sel), col))
        for grname2, gr2 in df_sel.groupby(label2):
            print(' '*8 + '{:5d} : {}'.format(len(gr2), grname2))
            for grname3, gr3 in gr2.groupby(label3):
                print(' '*16 + '{:5d} : {}'.format(len(gr3), grname3))


# turn off upper axis tick labels, rotate the lower ones, etc
#for ax in ax1, ax2, ax2t, ax3:
#    if ax != ax3:
#        for label in ax.get_xticklabels():
#            label.set_visible(False)
#    else:
#        for label in ax.get_xticklabels():
#            label.set_rotation(30)
#            label.set_horizontalalignment('right')
#    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')


def plot_barh(df, label_yval, label_xval='Value'):

    fig_heigth_inches = len(df) * (6/15)
    figsize = (12, fig_heigth_inches)

    height = 0.8
    tick_height = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    yticks = np.arange(0, len(df)*tick_height, tick_height)
    ax.barh(yticks, df[label_xval].values, align='center', height=height)
    ax.set_yticks(yticks)
    ax.set_yticklabels(df[label_yval].values.astype(str))
    ax.set_ylim(-height/2, yticks[-1] + height/2)
    fig.tight_layout()

    return fig, ax


def plot_barh_groups(df, label_yval, label_group, label_xval='Value'):

    fig_heigth_inches = len(df) * (6/15)
    figsize = (12, fig_heigth_inches)

    height = 1.0
    tick_height = 1.0
    gr_spacing = height/2
    nr_groups = len(df[label_group].unique())
    nr_cases = len(df)
    nr_bars = nr_groups + nr_cases/2
    fig_heigth_inches = nr_bars * tick_height * (6/15)
    y0 = 0
    yticks, yticklabels = np.array([]), np.array([])
    yticks_center, yticklabels_center = [], []
    fig, ax = plt.subplots(figsize=figsize)

    for igr, (grname, gr) in enumerate(df.groupby(label_group)):
        gr_yticks = np.arange(y0, y0+len(gr)*tick_height, tick_height)
        ax.barh(gr_yticks, gr[label_xval], align='center', height=height)
        yticks = np.append(yticks, gr_yticks)
        yticklabels = np.append(yticklabels, gr[label_yval].values.astype(str))
        yticks_center.append(y0 + len(gr)*tick_height/2 - tick_height/2)
        yticklabels_center.append(grname)
        y0 += (len(gr) + gr_spacing)

    ax.set_ylim(-tick_height, yticks[-1] + tick_height)
    ax.set_yticks(yticks_center)
    ax.set_yticklabels(yticklabels_center)
    fig.tight_layout()

    return fig, ax


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
    df = pd.read_hdf(pjoin(xml.pts_local, 'search_rx_470.h5'), 'table')

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
    df = pd.read_hdf(pjoin(xml.pts_local, 'search_rx_470.h5'), 'table')
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
    df = pd.read_hdf(pjoin(xml.pts_local, 'search_rx_470.h5'), 'table')

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


if __name__ == '__main__':

    dummy = None

#    df = download_from_openbm('RX 480')

#    obm = xml2df()
#    io = pjoin(obm.pts_local, "1606281-HA-RX480LINU80/composite.xml")
#    obm.load(io)
#    df_sys = obm.generated_system2df()
#    df_res = obm.data_entry2df()

#    obm = xml2df()
#    io = pjoin(obm.pts_local, "1606281-HA-RX480LINU80/composite.xml")
#    df = obm.convert(io)
