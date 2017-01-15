# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:03:26 2016

@author: davidovitch
"""

import os
from os.path import join as pjoin
from glob import glob
import re
import gc
import hashlib

from lxml.html import fromstring
from lxml import etree
import urllib.request

from tqdm import tqdm
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


class OpenBenchMarking:

    def __init__(self):
        self.pts_path = pjoin(os.environ['HOME'], '.phoronix-test-suite')
        self.res_path = pjoin(self.pts_path, 'test-results/')
        self.db_path = pjoin(os.environ['HOME'], '.phoronix-test-suite')
        self.url_base = 'http://openbenchmarking.org/result/{}&export=xml'
        self.url_search = 'http://openbenchmarking.org/s/{}&show_more'
        self.url_latest = 'http://openbenchmarking.org/results/latest'
        self.hard_soft_tags = set(['Hardware', 'Software'])
        self.testid = 'unknown'
        self.user_cols = ['User', 'SystemDescription', 'testid', 'Notes',
                          'SystemIdentifier', 'GeneratedTitle', 'LastModified']
        self.testid_cache = None

    def make_cache_set(self):
        """Create set of all testids present in the res_path directory.
        """
        fpath = os.path.join(self.res_path, '*')
        self.testid_cache = set([os.path.basename(k) for k in glob(fpath)])

    def load_testid_from_obm(self, testid, use_cache=True, save_xml=False):
        """Download a given testid from OpenBenchmarking.org.

        Parameters
        ----------

        testid : str
            OpenBenchemarking.org testid, for example: 1606281-HA-RX480LINU80
        """
        self.testid = testid

        if not use_cache:
            self.testid_cache = set([])
        elif self.testid_cache is None:
            self.make_cache_set()

        if testid not in self.testid_cache:
            self.load(self.url_base.format(testid))
            in_cache = False
        else:
            self.load(pjoin(self.res_path, testid, 'composite.xml'))
            in_cache = True

        if save_xml and not in_cache:
            self.write_testid_xml()

    def load(self, io):

        tree = etree.parse(io)
        self.root = tree.getroot()
        self.io = io

    def get_profiles(self, search_string):
        """Return a list of test profile id's
        """

        if search_string is None:
            url = self.url_latest
        else:
            url = self.url_search.format(search_string)
        response = urllib.request.urlopen(urllib.parse.quote(url, safe='/:'))
        data = response.read()      # a bytes object
        text = data.decode('utf-8') # str; can't be used if data is binary

        tree = fromstring(text)
        # all profiles names are in h4 elements, and nothing else is, nice and
        # easy but if a title is given, the id is in the parent link
        # url starts with /result/
        ids = [k.getparent().attrib['href'][8:] for k in tree.cssselect('h4')]

        return ids

    def write_testid_xml(self):
        fpath = pjoin(self.res_path, self.testid)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        fname = pjoin(fpath, 'composite.xml')
        with open(fname, 'w') as f:
            f.write(etree.tostring(self.root).decode())


class EditXML(OpenBenchMarking):

    def __init__(self):
        super().__init__()

    def merge(self, list_test_results):
        """DOESN'T MERGE ANYTHING YET
        """
        self.root = etree.Element('PhoronixTestSuite')
        for test_result in list_test_results:
            fpath = os.path.join(self.res_path, test_result, 'composite.xml')
            tree = etree.parse(fpath)
            root = tree.getroot()

    def write_local(self, test_result=None):
        if test_result is None:
            test_result = self.test_result
        fpath = os.path.join(self.res_path, test_result)
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
            self.load_testid_from_obm(testid, use_cache=True)

    def _rename_dict_key(self, df_dict, rename):
        """rename a key in a dictionary"""
        for key, value in rename.items():
            df_dict[value] = df_dict[key]
            df_dict.pop(key)
        return df_dict

    def convert2dict(self):
        """Convert the loaded XML file into a DataFrame ready dictionary of
        lists.
        """

        dict_sys = self.generated_system2dict()
        rename = {'Description':'SystemDescription',
                  'Identifier':'SystemIdentifier',
                  'JSON':'SystemJSON',
                  'Title':'GeneratedTitle'}
        dict_sys = self._rename_dict_key(dict_sys, rename)

        dict_res = self.data_entry2dict()
        rename = {'Description':'ResultDescription',
                  'Identifier':'ResultIdentifier',
                  'JSON':'DataEntryJSON',
                  'Title':'ResultTitle'}
        dict_res = self._rename_dict_key(dict_res, rename)

        # add columns for the system data
        for col in dict_sys:
            if col in dict_res:
                raise KeyError('{} already in df_dict'.format(col))
            dict_res[col] = []

        # for each data result entry, add the system col values
        for sys_index in dict_res['SystemIndex']:
            for col, val in dict_sys.items():
                dict_res[col].append(dict_sys[col][sys_index])
        dict_res.pop('SystemIndex')

        # doing this as DataFrames is very expensive considering the small
        # amount of data per XML file, and the large number of XML files.
#        df = pd.merge(df_sys, df_res, left_index=True, right_on='SystemIndex')
#        # after merging both, SystemIndex is now obsolete
#        df.drop('SystemIndex', inplace=True, axis=1)

        return dict_res

    def convert2df(self, dict_res):
        """Convert a df_dict to a DataFrame and convert columns to proper
        c-type variable names and values.

        RawString is a series of values separated by :
        Value can be a series of values sperated by ,
        """

        # split the Value column into a float and array part
        dict_res['ValueArray'] = []
        for i, valstring in enumerate(dict_res['Value']):
            valarray = np.fromstring(valstring, sep=',')
            # if we have more then one element it is a series, otherwise
            # just a single value
            if len(valarray) > 1:
                dict_res['Value'][i] = np.nan
                dict_res['ValueArray'].append(valarray)
            elif len(valarray)==0:
                dict_res['Value'][i] = np.nan
                dict_res['ValueArray'].append(np.array([np.nan]))
            else:
                dict_res['Value'][i] = valarray[0]
                dict_res['ValueArray'].append(np.array([np.nan]))

        # RawString will allways (?) hold more than one value
        for i, valstring in enumerate(dict_res['RawString']):
            # FIXME: reading empty field in xml is set to None seems?
            if valstring is None:
                valarray = np.array([np.nan])
            else:
                valarray = np.fromstring(valstring, sep=':')
            dict_res['RawString'][i] = valarray

        # convert to dataframe, set datatypes
        df = pd.DataFrame(dict_res)

        # convert all column names to c-name compatible
        df.rename(columns=lambda x: x.replace('-', '').replace(' ', ''),
                  inplace=True)

        return df

    def _split2dict(self, string):
        """Convert following string to dictionary:
        key1: value1, key2: value2, ...
        """
        # some old cases have (Total Cores: #) instead of (# Cores)
        elements = string.replace('Total Cores:', 'Total Cores').split(', ')
        return {k.split(':')[0]:k.split(':')[1] for k in elements}

    def _add2row(self, elements, columns, df_dict, missing_val='',
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

        missing_val : str, default=''
            When an tag occurs in columns but not in elements, it is added to
            df_dict with missing_val as value. Rename is applied after the
            missing keys from columns are checked

        Returns
        ------

        df_dict : dict
            pandas.DataFrame dictionary with one added row for all the columns
            of the set columns. Elements should be a sub-set of columns.
            Values occuring in columns but not in elements are added with the
            value as set in the missing_val variable.

        """

        # make sure that all containing elements are used, and that
        # missing ones are filled in as empty to preserve a valid
        # DataFrame dictionary
        found_els = []

        for el in elements:
            if el.tag in self.hard_soft_tags:
                # split the Hardware and Software tags into the columns
                tmp = self._split2dict(el.text)
                # add hash to have a unique identifier for each configuration
                md5hash = hashlib.md5(el.text.encode('utf-8')).hexdigest()
                tmp[el.tag + 'Hash'] = md5hash
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

    def generated_system2dict(self, missing_val=''):
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
                  'TestClientVersion', 'Notes', 'JSON', 'System Layer']
        hardware = ['Processor', 'Motherboard', 'Chipset', 'Memory', 'Disk',
                    'Graphics', 'Audio', 'Network', 'Monitor', 'HardwareHash']
        software = ['OS', 'Kernel', 'Desktop', 'Display Server',
                    'Display Driver', 'OpenGL', 'OpenCL', 'Vulkan', 'Compiler',
                    'File-System', 'Screen Resolution', 'SoftwareHash']

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
            dict_gen[key] = missing_val
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
                msg = '{} has {} elements instead of {}'.format(*rpl)
                raise AssertionError(msg)

        # expand with the same values for Generated columns, this duplication
        # of data will make searching/selecting in the DataFrame later easier
        for key, value in dict_gen.items():
            dict_sys[key] = [value]*len(systems)

#        for key, value in dict_sys.items():
#            print(key.rjust(28), len(value))

        # the indices for Identifier in Results/Data/Entry
        self.ids = {key:i for i,key in enumerate(dict_sys['Identifier'])}

        return dict_sys

    def data_entry2dict(self, missing_val=''):
        """For a given xml result file from pts/OpenBenchmarking.org, convert
        the Result tags to a Pandas DataFrame. This means that the data
        contained in the Result tag will now be replicated for each of the
        Data/Entry tags for that given test result.
        """
        # ResultOf indicates whether the result belongs to another result
        # for example, corresponding CPU usage, render time per frame.
        # ResultOf is not defined in the XML source.
        result = ['Identifier', 'Title', 'AppVersion', 'Arguments', 'ResultOf',
                  'Description', 'Scale', 'Proportion', 'DisplayFormat']
        data_entry = ['Identifier', 'Value', 'RawString', 'JSON']
        data_entry_ = ['DataEntryIdentifier'] + data_entry[1:]
        data_set = set(data_entry)
        rename = {'Identifier':'DataEntryIdentifier'}

        dict_res = {k:[] for k in result+data_entry_}

        res_elements = list(self.root.findall('Result'))

        res_title = ''
        for res_el in res_elements:
            # get all the details of the test in question
            res_id = {k.tag:k.text for k in res_el}
            res_id.pop('Data')
            data_entries = res_el.find('Data')
            # if the result identifier is empty, it corresponds to the previous
            # result (usually CPU/frame time for LINE_GRAPH). Add one
            # additional check: result title should be the same
            if res_id['Identifier'] is not missing_val:
                res_id_val = res_id['Identifier']
                res_title = res_id['Title']
                res_id['ResultOf'] = 'no'
            elif res_id['Title'] == res_title:
                res_id['Identifier'] = res_id_val
                res_id['Title'] = res_title
                res_id['ResultOf'] = 'yes'
            # some cases just have no result identifier and do not belong to
            # another test
            else:
                res_id['ResultOf'] = 'missing'

            # add each result to the collection
            tmp = {k:[] for k in data_entry}
            for entry in data_entries:
                tmp = self._add2row(entry, data_set, tmp)
            # before merging, rename Identifier
            tmp = self._rename_dict_key(tmp, rename)
            # add with the rest
            for key, value in tmp.items():
                dict_res[key].extend(value)
            # and add the Result element columns to all the data entries
            for key, value in res_id.items():
                dict_res[key].extend([value]*len(data_entries))

        # add the index of the System as a column because SystemIdentifier is
        # not unique! Will be used when merging/joining with Generated/system
        dict_res['SystemIndex'] = []
        # when having multiple bars per group, the data/entry identifier
        # can be shorter compared to the system identifier, and also contains
        # an additional label: "EXTRA LABEL: SHORT SYSTEM ID"
        dict_res['DataEntryIdentifierExtra'] = []
        dict_res['DataEntryIdentifierShort'] = []
        for identifier in dict_res['DataEntryIdentifier']:
            idf_split = identifier.split(': ')
            if len(idf_split) == 2:
                idf_short = idf_split[1]
                dict_res['DataEntryIdentifierExtra'].append(idf_split[0])
                dict_res['DataEntryIdentifierShort'].append(idf_split[1])
                # find the long version of the identifier
                for idf in self.ids:
                    if idf.find(idf_short) > -1:
                        dict_res['SystemIndex'].append(self.ids[idf])
                        break
            else:
                dict_res['SystemIndex'].append(self.ids[identifier])
                dict_res['DataEntryIdentifierExtra'].append(missing_val)
                dict_res['DataEntryIdentifierShort'].append(missing_val)

#        for key, value in dict_res.items():
#            print(key.rjust(28), len(value))

        return dict_res


def search_openbm(search_string, save_xml=True, use_cache=True):
    """

    Parameters
    ----------

    search_string : str
        Search string used on the OpenBenchmarking.org search page. If set to
        None, the latest test results page is loaded instead.

    save_xml : boolean, default=False
        If set to True, all test id's XML files are saved in the user's
        phoronix test suite home directory.

    use_cache : boolean, default=True
        Do not download results that are already there.

    """

    # TODO: another de-duplication strategy: if the results are the same,
    # and some of the other differences are due to a None or something we
    # can also be confident we are talking about the same test case.

    xml = xml2df()
    # get a list of test result id's from using the search function on obm.org
    testids = xml.get_profiles(search_string)
    nr_testids = len(testids)
    if nr_testids < 1:
        print('no tests found for search query: {}'.format(search_string))
        return None
    else:
        print('found {} testids on search page'.format(nr_testids))

    # save list
    fname = pjoin(xml.pts_path, 'openbenchmarking.org-searches',
                  'testids_{}.txt'.format(search_string))
    np.savetxt(fname, np.array(testids, dtype=np.str), fmt='%22s')

    # load the cache list
    xml.make_cache_set()

    # if cache is used, only download new cases
    if use_cache:
        testids = set(testids) - xml.testid_cache
        nr_testids = len(testids)
        if nr_testids < 1:
            print('all testids have already been downloaded')
            return None
        print('')
        print('start downloading {} test id\'s'.format(nr_testids))
        print('')

    for testid in tqdm(testids):
        # download xml file
        xml.load_testid_from_obm(testid, use_cache=False, save_xml=save_xml)

    return testids


def split_cpu_info(df):
    """Processor tag also contains number cores and CPU frequency, split them
    off into their own columns. Format: Intel Core i7-6700K @ 4.20GHz (8 Cores)
    This string can further be preceded by the number of CPU's: 2 x Intel ...
    """

    df_add = df['Processor'].str.split(' @ ', expand=True)
    freq = df_add[1].str.split(' \(', expand=True)
    cores = freq[1].str.split(' Core', expand=True)
    # set cores to -1 if not present instead of None
    cores[0] = cores[0].astype(str)
    sel = cores[cores[0]=='None'].index
    cores.loc[sel, 0] = -1

    count_name = df_add[0].str.split(' x ', expand=True)
    # however, if ' x ' is missing the name will be on the first column instead
    # of the second
    count_name[1] = count_name[1].values.astype(str)
    nocount = count_name[count_name[1]=='None'].index
    # move the name one column right
    count_name.loc[nocount, 1] = count_name.loc[nocount, 0]
    # and the Processor count is one
    count_name.loc[nocount, 0] = 1

    df['ProcessorFrequency'] = freq[0].str.replace('GHz', '').astype(np.float32)
    try:
        df['ProcessorCores'] = cores[0].astype(np.int16)
    except ValueError as e:
        # some old results have: AMD Athlon @ 1.10GHz (Total Cores 1)
        # note that the (Total Cores: 1) the colon has been removed in
        # xml2df._split2dict earlier
        sel = cores[0].str.find('Total') > -1
        cores.loc[sel, 0] = freq[1].loc[sel].str.split('Cores ', expand=True)[1]
        cores.loc[sel, 0] = cores.loc[sel, 0].str.replace(')', '')
        df['ProcessorCores'] = cores[0].astype(np.int16)
    df['ProcessorCount'] = count_name[0].astype(np.int16)
    df['ProcessorName'] = count_name[1]

    return df


def load_local_testids():
    """Load all local test id xml files and merge into one pandas.DataFrame.
    """
    df_dict = None
    xml = xml2df()

    # make a list of all available test id folders
    base = os.path.join(xml.res_path, '*')
    failed = []

    regex = re.compile(r'^[0-9]{7}\-[A-Za-z0-9]*\-[A-Za-z0-9]*$')
    i = 0

    for fpath in tqdm(glob(base)):
        testid = os.path.basename(fpath)
        regex.findall(testid)
        if len(regex.findall(testid)) != 1:
            continue
        fpath = pjoin(xml.res_path, testid, 'composite.xml')
        xml.load(fpath)
        xml.testid = testid
        i += 1

        try:
            _df_dict = xml.convert2dict()
        except Exception as e:
            print('')
            print('conversion to df_dict of {} failed.'.format(testid))
            print(e)
            continue

        if df_dict is None:
            df_dict = {key:[] for key in _df_dict}

        for key, val in _df_dict.items():
            df_dict[key].extend(val)

        # Very expensive to append many DataFrames to an ever growing DataFrame
#        try:
#            df = df.append(xml.convert2dict())
#            i += 1
#        except Exception as e:
#            failed.append(testid)
##            print('*'*79)
#            print('conversion to df of {} failed.'.format(testid))
##            print('*'*79)
#            raise e

    print('loaded {} xml files locally'.format(i))

    df = xml.convert2df(df_dict)
#    df = pd.DataFrame(df_dict)
    del df_dict
    gc.collect()

    # create a new unique index
    df.index = np.arange(len(df))

    # FIXME: is it safe to ignore array columns when looking for duplicates?
    # there are probably going to be more duplicates
    # doesn't work for ndarray columns
    arraycols = set(['RawString', 'ValueArray'])
    cols = list(set(df.columns) - arraycols)
    df.drop_duplicates(inplace=True, subset=cols)
    # columns that can hold different values but still could refer to the same
    # test data. So basically all user defined columns should be ignored.
    # But do not drop the columns, just ignore them for de-duplication
    cols = list(set(df.columns) - set(xml.user_cols) - arraycols)
    # mark True for values that are NOT duplicates
    df = df.loc[np.invert(df.duplicated(subset=cols).values)]
    # split the Processer column in Processor info, frequency and cores
    df = split_cpu_info(df)

    # convert object columns to string, but leave other data types as is
    # ignore columns with very long strings to avoid out of memory errors
    # RawString and Value can contain a time series
    ignore = set(['RawString', 'Value', 'ValueArray'])
    for col, dtype in df.dtypes.items():
        if dtype==np.object and col not in ignore:
            try:
                df[col] = df[col].values.astype(np.str)
            except Exception as e:
                print(col)
                raise e

    # trim all columns
    for col in df:
        if df[col].dtype==np.object:
            df[col] = df[col].str.strip()

    # remove all spaces in column names
#    new_cols = {k:k.replace(' ', '').replace('-', '') for k in df.columns}
#    df.rename(columns=new_cols, inplace=True)

    return df, failed


def explore_dataset(df, label1, label2, label3, min_cases=3):

    # how many cases per test, and how many per resolution
    df_dict = {'nr':[], 'test':[]}

    for grname, gr in df.groupby(label1):
        df_dict['nr'].append(len(gr))
        df_dict['test'].append(grname)
#        print('{:5d} : {}'.format(len(gr), grname))

    df_tests = pd.DataFrame(df_dict)
    df_tests.sort_values('nr', inplace=True)

    print('top 10 in nr of cases per {}'.format(label1))
    print(df_tests[-10:])

    for col in df_tests['test']:
        df_sel = df[df[label1]==col]
#        print()
#        print('{:5d} : {}'.format(len(df_sel), col))
        for grname2, gr2 in df_sel.groupby(label2):
#            print(' '*8 + '{:5d} : {}'.format(len(gr2), grname2))
            if len(gr2[label3].unique()) >= min_cases:
                print()
                print(col)
                print(grname2)
                for grname3, gr3 in gr2.groupby(label3):
                    print(' '*8 + '{:5d} : {}'.format(len(gr3), grname3))


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

        # write the label_yval into the bar
        for y, label in zip(gr_yticks, gr[label_yval].values.astype(str)):
            ax.text(0, y, label, fontsize=8, verticalalignment='center',
                    horizontalalignment='left', color='w')

        y0 += (len(gr) + gr_spacing)

    ax.set_ylim(-tick_height, yticks[-1] + tick_height)
    ax.set_yticks(yticks_center)
    ax.set_yticklabels(yticklabels_center)
    fig.tight_layout()

    return fig, ax


if __name__ == '__main__':

    xml = xml2df()

#    search_string = 'RX 480'
#    df = search_openbm(search_string, save_xml=False, use_cache=True)
#    df.to_hdf(pjoin(xml.db_path, 'search_{}.h5'.format(search_string)), 'table')
#    df.to_excel(pjoin(xml.db_path, 'search_{}.xlsx'.format(search_string)))
#    df.to_csv(pjoin(xml.db_path, 'search_{}.csv'.format(search_string)))

#    df = search_openbm(None, save_xml=True)
#    dd = date.today()
#    today = '{}-{:02}-{:02}'.format(dd.year, dd.month, dd.day)
#    df.to_hdf(pjoin(xml.db_path, 'latest_{}.h5'.format(today)), 'table')

#    xml = xml2df()
#    io = pjoin(xml.res_path, "1510217-HA-BPPADOKA880/composite.xml")
#    xml.load(io)
#    df_dict = xml.convert2dict()
#    dict_sys = xml.generated_system2dict()
#    dict_res = xml.data_entry2dict()

#    obm = xml2df()
#    io = pjoin(obm.res_path, "1606281-HA-RX480LINU80/composite.xml")
#    df_dict = obm.convert2dict(io)
