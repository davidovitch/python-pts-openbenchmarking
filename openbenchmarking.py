# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:03:26 2016

@author: davidovitch
"""

import os
from os.path import join as pjoin
from glob import glob
import shutil
import re
import gc
import hashlib
import json

from lxml.html import fromstring
from lxml import etree
import urllib.request

from tqdm import tqdm
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


class DataFrameDict:
    """Utilities for handling and checking the consistancy of a dictionary
    in DataFrame format
    """

    def check_column_length(df_dict):
        """Make sure all columns have the same number of rows
        """
        collens = {}
        for col, values in df_dict.items():
            collens[col] = len(values)

        if len(set(collens.values())) > 1:
            for col, val in collens.items():
                print('%6i : %s' % (val, col))

        return collens

    def trim_cell_len(df_dict, maxlens):
        """Trim maximum length of a certain columns/cells
        """
        col0 = list(df_dict.keys())[0]
        for i in range(len(df_dict[col0])):
            for col, maxlen in maxlens.items():
                if isinstance(df_dict[col][i], str):
                    df_dict[col][i] = df_dict[col][i][:maxlen]
        return df_dict

    def check_cell_len(df_dict):
        """List maximum cell length per column
        """
        maxcellen, celltype = {}, {}
        for name, column in df_dict.items():
            maxcellen[name], celltype[name] = [], []
            for cell in column:
                celltype[name].append(str(type(cell)))
                if isinstance(cell, str):
                    maxcellen[name].append(len(cell))
                else:
                    maxcellen[name].append(0)
        return maxcellen, celltype


class OpenBenchMarking:

    # add an optional HTML header when downloading content with urllib
    header = {}

    def __init__(self):
        self.pts_path = pjoin(os.environ['HOME'], '.phoronix-test-suite')
        self.res_path = pjoin(self.pts_path, 'test-results-all-obm/')
        self.db_path = pjoin(os.environ['HOME'], '.phoronix-test-suite')
        self.url_base = 'http://openbenchmarking.org/result/{}&export=xml'
        self.url_search = 'http://openbenchmarking.org/s/{}&show_more'
        self.url_latest = 'http://openbenchmarking.org/results/latest'
        self.url_test = 'http://openbenchmarking.org/test/pts/{}'
        self.url_test_search = 'http://openbenchmarking.org/test/pts/{}&search'
        self.url_test_base = 'http://openbenchmarking.org/tests/pts'
        self.hard_soft_tags = set(['Hardware', 'Software'])
        self.testid = 'unknown'
        self.user_cols = ['User', 'SystemDescription', 'testid', 'Notes',
                          'SystemIdentifier', 'GeneratedTitle', 'LastModified']
        self.testid_cache = None

    def make_cache_set(self):
        """Create set of all testids present in the res_path directory.
        """
        fpath = os.path.join(self.res_path, '**', '*.xml')
        self.testid_cache = set(
            [os.path.basename(k)[:-4] for k in glob(fpath, recursive=True)])

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
            fname = self.get_testid_fname()
            self.load(fname)
            in_cache = True

        if save_xml and not in_cache:
            self.write_testid_xml()

    def load(self, io):

        tree = etree.parse(io)
        self.root = tree.getroot()
        self.io = io

    def get_profiles(self, url):
        """Return a list of test profile id's from a given OBM url. URL is
        parsed with safe.

        Parameters
        ----------

        url : str


        Returns
        -------

        ids : list
            List of test profile id's
        """

        url = urllib.parse.quote(url, safe='/:')
        req = urllib.request.Request(url=url, headers=self.header)
        response = urllib.request.urlopen(req)

        data = response.read()      # a bytes object
        text = data.decode('utf-8') # str; can't be used if data is binary

        tree = fromstring(text)
        # all profiles names are in h4 elements, and nothing else is, nice
        # and easy but if a title is given, the id is in the parent link
        # url starts with /result/
        ids = [k.getparent().attrib['href'][8:] for k in tree.cssselect('h4')]

        return ids

    def get_testid_fname(self, testid=None):
        """
        """
        if testid is None:
            testid = self.testid
        yy = testid[:2]
        mm = testid[2:4]
        return pjoin(self.res_path, yy, mm, testid + '.xml')

    def get_tests(self):
        """Return a list with all PTS tests.
        """
        url = urllib.parse.quote(self.url_test_base, safe='/:')
        req = urllib.request.Request(url=url, headers=self.header)
        response = urllib.request.urlopen(req)
        data = response.read()      # a bytes object
        text = data.decode('utf-8') # str; can't be used if data is binary
        tree = fromstring(text)
        tests = []
        for h4 in tree.cssselect('h4'):
            if len(h4) < 1:
                continue
            # link starts with /test/pts/
            tests.append(h4.getchildren()[0].attrib['href'][10:])
        return tests

    def write_testid_xml(self):
        """Write testid xml file to res_path/yy/mm/testid.xml
        """
        fname = self.get_testid_fname()
        fpath = os.path.dirname(fname)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
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
            fname = self.get_testid_fname(testid=test_result)
            tree = etree.parse(fname)
            root = tree.getroot()

    def write_local(self, test_result=None):
        if test_result is None:
            test_result = self.test_result
        fname = self.get_testid_fname(testid=test_result)
        fpath = os.path.dirname(fname)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
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
    """Class for converting a PTS/OpenBenchmarking.org XML result file into a
    table formatted as a Pandas.DataFrame object.
    """

    def __init__(self, io=None, testid=None):
        """

        Parameters
        ----------

        io : path, default=None
            File path to the testid file to be loaded, or anything else that
            lxml.etree.parse(io) can handle.

        testid : str, default=None
            testid of the result file to be loaded. If not available locally
            it will be downloaded from OpenBenchmarking.org.
        """
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

    def xml2dict(self):
        """Convert the loaded XML file into a DataFrame ready dictionary of
        lists.
        """

        dict_sys = self.generated_system2dict()
        rename = {'Identifier':'SystemIdentifier',
                  'Title':'GeneratedTitle'}
                  #'JSON':'SystemJSON', 'Description':'GeneratedDescription',
        dict_sys = self._rename_dict_key(dict_sys, rename)

        dict_res = self.data_entry2dict()
        rename = {'JSON':'DataEntryJSON',
                  'Description':'ResultDescription',
                  'Identifier':'ResultIdentifier',
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

    def xml2dict_split(self):
        """Convert the loaded XML file into a DataFrame ready dictionary of
        lists, but keep system and results separate and create a unique
        overlapping index.
        """

        dict_sys = self.generated_system2dict() #maxlen=200
        rename = {'Identifier':'SystemIdentifier',
                  'Title':'GeneratedTitle'}
                  #'JSON':'SystemJSON', 'Description':'GeneratedDescription',
        dict_sys = self._rename_dict_key(dict_sys, rename)
        maxlens = {'Memory' : 100,
                   'Disk' : 100,
                   'GeneratedTitle' : 115}
        dict_sys = self._trim_cell_len(dict_sys, maxlens)

        dict_res = self.data_entry2dict() #maxlen=60
        dict_res = split_json(dict_res)
        rename = {'Description':'ResultDescription',
                  'Identifier':'ResultIdentifier',
                  'Title':'ResultTitle',
                  'compiler':'DataEntryCompiler',
                  'compiler-type':'DataEntryCompilerType',
                  'max-result':'DataEntryMaxResult',
                  'min-result':'DataEntryMinResult'}
        dict_res = self._rename_dict_key(dict_res, rename)
        maxlens = {'AppVersion' : 50,
                   'Scale' : 50,
                   'ResultDescription' : 150}
        dict_res = self._trim_cell_len(dict_res, maxlens)

        # prepare full length but empty SystemHash column in dict_sys
        nr_systems = len(dict_sys['SystemIdentifier'])
        dict_sys['SystemHash'] = ['']*nr_systems
        # Add a system hash column for each row
        text = ''
        for irow in range(nr_systems):
            text = ''
            for col in dict_sys.keys():
                text += str(dict_sys[col][irow])
            md5hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            dict_sys['SystemHash'][irow] = md5hash

        # add SystemHash column to results
        dict_res['SystemHash'] = []

        # for each data result entry, add the system col values
        for sys_index in dict_res['SystemIndex']:
            md5hash = dict_sys['SystemHash'][sys_index]
            dict_res['SystemHash'].append(md5hash)
        dict_res.pop('SystemIndex')

        return dict_res, dict_sys

    def dict2df(self, dict_res):
        """Convert a df_dict to a DataFrame and convert columns to proper
        c-type variable names and values.

        RawString is a series of values separated by :
        Value can be a series of values sperated by ,
        """

        # split the Value column into a float and array part
        if 'Value' in dict_res:
            dict_res['ValueArray'] = []
            for i, valstring in enumerate(dict_res['Value']):
                if valstring is None:
                    valstring = ''
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
        if 'RawString' in dict_res:
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
        # some old result files have (Total Cores: #) instead of (# Cores)
        elements = string.replace('Total Cores:', 'Total Cores').split(', ')
        return {k.split(':')[0]:k.split(':')[1] for k in elements}

    def _add2row(self, elements, columns, df_dict, missing_val='',
                 rename={}, ignore=set([])):
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

        rename : dict, default={}

        ignore : set, default=set([])
            Ignore elements.

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
            if el.tag in ignore:
                continue
            # TODO: should be like split_info methods for cpu, gpu, memory
            if el.tag in self.hard_soft_tags:
                # Here the columns HardwareHash and SoftwareHash are created.
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

    def _trim_cell_len(self, df_dict, maxlens):
        """Trim maximum length of a certain columns/cells
        """
        col0 = list(df_dict.keys())[0]
        for i in range(len(df_dict[col0])):
            for col, maxlen in maxlens.items():
                if isinstance(df_dict[col][i], str):
                    df_dict[col][i] = df_dict[col][i][:maxlen]
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

        Following columns are ignored from the Generated group to avoid too
        long column values: Description

        Following columns are ingored from the System group to avoid too long
        columns values: JSON
        """

        generated = ['Title', 'LastModified', 'TestClient', 'Description',
                     'Notes', 'InternalTags', 'ReferenceID',
                     'PreSetEnvironmentVariables']
        gen_ignore = set(['Description'])
        system = ['Identifier', 'Hardware', 'Software', 'User', 'TimeStamp',
                  'TestClientVersion', 'Notes', 'JSON', 'System Layer']
        sys_ignore = set(['JSON']) # 'Notes'
        hardware = ['Processor', 'Motherboard', 'Chipset', 'Memory', 'Disk',
                    'Graphics', 'Audio', 'Network', 'Monitor', 'HardwareHash']
        software = ['OS', 'Kernel', 'Desktop', 'Display Server',
                    'Display Driver', 'OpenGL', 'OpenCL', 'Vulkan', 'Compiler',
                    'File-System', 'Screen Resolution', 'SoftwareHash']

        cols_sys = system + hardware + software
        # Remove columns we do not want to use because they are too long
        cols_sys.remove('Hardware')
        cols_sys.remove('Software')
        for k in sys_ignore:
            cols_sys.remove(k)

        generated_set = set(generated) - set(gen_ignore)
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
        # FIXME: data checks should take place in a xml check method/class!
        assert len(gen_elements) == 1
        # create dictionary of the tags/values in Generated
        dict_gen = {el.tag : el.text for el in gen_elements[0]
                    if el.tag not in gen_ignore}
        # Are there any suprises in the tags? Tags we haven't seen before?
        assert len(dict_gen.keys() - (generated_set | gen_ignore)) == 0

        # add empty values for possible missing keys
        for key in generated_set - set(dict_gen.keys()):
            dict_gen[key] = missing_val
        # also include the URL testid identifier which is unique for each
        # test entry on OpenBenchmarking.org
        dict_gen['testid'] = self.testid

        # For each system create a row in the df_dict
        systems = self.root.findall('System')
        for sys_els in systems:
            dict_sys = self._add2row(sys_els, system_set, dict_sys,
                                     ignore=sys_ignore)

        # sanity checks
        for key, value in dict_sys.items():
            if not len(systems) == len(value):
                rpl = [key, len(value), len(systems)]
                msg = '{} has {} elements instead of {}'.format(*rpl)
                raise AssertionError(msg)

        # expand with the same values for Generated columns, this duplication
        # of data will make searching/selecting in the DataFrame later easier
        # FIXME: data duplication of the Generated tags for all test runs
        for key, value in dict_gen.items():
            dict_sys[key] = [value]*len(systems)

#        for key, value in dict_sys.items():
#            print(key.rjust(28), len(value))

        # the indices for Identifier in Results/Data/Entry
        self.ids = {key:i for i,key in enumerate(dict_sys['Identifier'])}

        return dict_sys#, dict_gen

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
        res_ignore = set(['Arguments']) # , 'Description'
        data_entry = ['Identifier', 'Value', 'RawString', 'JSON']
        dat_ignore = set([])#'JSON'])
        data_entry_ = set(['DataEntryIdentifier'] + data_entry[1:]) - dat_ignore
        data_set = set(data_entry) - dat_ignore
        rename = {'Identifier':'DataEntryIdentifier'}

        dict_res = {k:[] for k in set(result) - res_ignore | data_entry_}

        res_elements = list(self.root.findall('Result'))

        res_title = ''
        for res_el in res_elements:
            # get all the details of the test in question
            res_id = {k.tag:k.text for k in res_el if k.tag not in res_ignore}
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
                res_id['ResultOf'] = 'na'

            # add each result to the collection
            tmp = {k:[] for k in data_set}
            for entry in data_entries:
                tmp = self._add2row(entry, data_set, tmp, ignore=dat_ignore)
            # before merging, rename Identifier
            tmp = self._rename_dict_key(tmp, rename)
            # add with the rest
            for key, value in tmp.items():
                dict_res[key].extend(value)
            # and add the Result element columns to all the data entries
            # FIXME: Data duplication for of the Result tags for each run
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


class DataBase:

    def __init__(self):
        self.regex = re.compile(r'^[0-9]{7}\-[A-Za-z0-9]*\-[A-Za-z0-9]*$')
        self.db_path = pjoin(os.environ['HOME'], '.phoronix-test-suite')
        self.xml = xml2df()

    def load(self):

        fname = pjoin(self.xml.db_path, 'database_results.h5')
        df = pd.read_hdf(fname, 'table')
        fname = pjoin(self.xml.db_path, 'database_systems.h5')
        df_sys = pd.read_hdf(fname, 'table')

        return df, df_sys

    def get_hdf_stores(self):
        fname = pjoin(self.xml.db_path, 'database_results.h5')
        self.store = pd.HDFStore(fname, mode='a', format='table',
                                 complib='blosc', compression=9)
        fname = pjoin(self.xml.db_path, 'database_systems.h5')
        self.store_sys = pd.HDFStore(fname, mode='a', format='table',
                                     complib='blosc', compression=9)

    def build(self, debug=False):
        """Build complete database from scratch, over write existing.
        """

        df_dict, df_dict_sys = self.testids2dict()
        if debug:
            DataFrameDict.check_column_length(df_dict)
            DataFrameDict.check_column_length(df_dict_sys)

            maxcellen1, celltype1 = DataFrameDict.check_cell_len(df_dict)
            maxcellen2, celltype2 = DataFrameDict.check_cell_len(df_dict_sys)
            return

        df = self.xml.dict2df(df_dict)
        df = self.cleanup(df)

        fname = pjoin(self.xml.db_path, 'database_results.h5')
#        df.drop(['ValueArray', 'RawString'], axis=1, inplace=True)
        df.to_hdf(fname, 'table', format='table', complib='blosc', mode='w',
                  compression=9, data_columns=True)
        del df_dict
        del df
        gc.collect()

        df = self.xml.dict2df(df_dict_sys)
        df = self.cleanup(df)
        fname = pjoin(self.xml.db_path, 'database_systems.h5')
#        df.drop(['ValueArray', 'RawString'], axis=1, inplace=True)
        df.to_hdf(fname, 'table', format='table', complib='blosc', mode='w',
                  compression=9, data_columns=True)
        del df_dict_sys
        del df
        gc.collect()

    def update(self, testids=None):
        """Load existing database and add testids that haven't been added.
        """
        df, df_sys = self.load()

        if testids is None:
            # already included testid's
            testids_df = set(df_sys['testid'].unique())
            # all downloaded testids
            base = os.path.join(self.xml.res_path, '*')
            testids_disk = set([os.path.basename(k) for k in glob(base)])
            # only add testids that are on disk but not in the database
            testids = testids_disk - testids_df

        print('\nupdating with %i new testids' % len(testids))
        df_dict, df_dict_sys = self.testids2dict(testids=testids)

        self.get_hdf_stores()

        df = self.xml.dict2df(df_dict)
        df = self.cleanup(df, i0=len(df))
        # https://stackoverflow.com/a/15499291
        # use data_columns=True for robustness: process column by column and
        # raise when a data type is being offended
        self.store.append('table', df, data_columns=True)
        self.store.close()

        df = self.xml.dict2df(df_dict_sys)
        df = self.cleanup(df, i0=len(df))
        self.store_sys.append('table', df, data_columns=True)
        self.store_sys.close()

        gc.collect()

    def testids2dict(self, testids=None):
        """Load all local test id xml files and convert to pandas.DataFrame
        dictionaries.

        Parameters
        ----------

        testids : iterable, default=None
            Iterable holding of those testid's to be converted to dicts.
            If None, all testid's stored at xml.res_path are considered.

        Returns
        -------

        df_dict, df_dict_sys : pandas.DataFrame dictionary

        """
        df_dict = None
        df_dict_sys = None

        # consider all testids if None
        if testids is None:
            # make a list of all available test id folders
            base = pjoin(self.xml.res_path, '**', '*.xml')
            testids = glob(base, recursive=True)

        regex = re.compile(r'^[0-9]{7}\-[A-Za-z0-9]*\-[A-Za-z0-9]*$')
        i = 0

        for fpath in tqdm(testids):

#             if i > 10000:
#                 break

            testid = os.path.basename(fpath)[:-4]
            regex.findall(testid)
            if len(regex.findall(testid)) != 1:
                continue
            self.xml.load(fpath)
            self.xml.testid = testid
            i += 1

            try:
#                _df_dict = xml.xml2dict()
                _df_dict, _df_dict_sys = self.xml.xml2dict_split()
            except Exception as e:
                print('')
                print('conversion to df_dict of {} failed.'.format(testid))
                print(e)
                continue
            # make sure we have a consistant df
            k1 = set([len(val) for key, val in _df_dict.items()])
            k2 = set([len(val) for key, val in _df_dict_sys.items()])
            if len(k1) > 1 or len(k2) > 1:
                DataFrameDict.check_column_length(_df_dict)
                DataFrameDict.check_column_length(_df_dict_sys)
                print('conversion to df_dict of {} failed.'.format(testid))
                continue

            if df_dict is None:
                df_dict = {key:[] for key in _df_dict}
            if df_dict_sys is None:
                df_dict_sys = {key:[] for key in _df_dict_sys}

            for key, val in _df_dict.items():
                df_dict[key].extend(val)
            for key, val in _df_dict_sys.items():
                df_dict_sys[key].extend(val)

        return df_dict, df_dict_sys

    def cleanup(self, df, i0=0):

        # FIXME: is it safe to ignore array columns when looking for duplicates?
        # there are probably going to be more duplicates
        # doesn't work for ndarray columns
        arraycols = set(['RawString', 'ValueArray'])
        cols = list(set(df.columns) - arraycols)
        df.drop_duplicates(inplace=True, subset=cols)
        # columns that can hold different values but still could refer to the same
        # test data. So basically all user defined columns should be ignored.
        # But do not drop the columns, just ignore them for de-duplication
        cols = list(set(df.columns) - set(self.xml.user_cols) - arraycols)
        # mark True for values that are NOT duplicates
        df = df.loc[np.invert(df.duplicated(subset=cols).values)]
        # split the Processer column in Processor info, frequency and cores
        if 'Processor' in df.columns:
            df = split_cpu_info(df)

        # trim all columns
        for col in df:
            if df[col].dtype==np.object:
                df[col] = df[col].str.strip()

        # convert object columns to string, but leave other data types as is
        # ignore columns with very long strings to avoid out of memory errors
        # RawString and ValueArray can contain a time series
        ignore = set(['Value']) | arraycols
        for col, dtype in df.dtypes.items():
            if dtype==np.object and col not in ignore:
                try:
#                    df[col] = df[col].astype('category')
                    df[col] = df[col].values.astype(np.str)
                except Exception as e:
                    print(col)
                    raise e
#        # leave array data in different dataframe
#        df_arr = df[['testid', 'ValueArray', 'Rawstring']]

        # trim all columns
        for col in df:
            if df[col].dtype==np.object:
                df[col] = df[col].str.strip()

        # create a new unique index
        df.index = i0 + np.arange(len(df))

        # remove all spaces in column names
#        new_cols = {k:k.replace(' ', '').replace('-', '') for k in df.columns}
#        df.rename(columns=new_cols, inplace=True)

        return df


def search_openbm(search=None, save_xml=True, use_cache=True, tests=[],
                  latest=False):
    """

    Parameters
    ----------

    search : str, default=None
        Search string used on the OpenBenchmarking.org search page. Skipped
        when None.

    tests : 'ALL' or list of str, default=[]
        Get result ids from the test and test search pages, should be an
        existing/valid PTS test name. Use all existing tests when tests=ALL.

    latest : boolean, default=False
        Set to True to get latest results.

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
    testids, urls = [], []

    if latest:
        urls.append(xml.url_latest)
    elif isinstance(search, str):
        urls.append(xml.url_search.format(search))

    test_ = str(tests)
    if len(tests) > 3:
        test_ = str(len(tests))
    if tests == 'ALL':
        tests = xml.get_tests()
        test_ = 'ALL'
    print('including {} tests'.format(len(tests)))
    for test in tests:
        urls.append(xml.url_test.format(test))
        urls.append(xml.url_test_search.format(test))

    # get a list of test result id's from given urls
    print('start extracting testids from {} urls'.format(len(urls)))
    for url in tqdm(urls):
        testids.extend(xml.get_profiles(url))
    testids = list(set(testids))
    nr_testids = len(testids)
    print('search: {}, tests: {}, latest: {}'.format(search, test_, latest))
    print('found {} testids'.format(nr_testids))
    if nr_testids < 1:
        return

    # save all tests ids from a lot of results
    if len(testids) > 150:
        fname = 'testids_tmp.txt'
        fpath = pjoin(xml.pts_path, 'openbenchmarking.org-searches', fname)
        np.savetxt(fpath, np.array(testids, dtype=np.str), fmt='%22s')

    # save testids when using search or latest
    if isinstance(search, str) or latest:
        if search is None:
            search = 'latest'
        fname = 'testids_{}.txt'.format(search.replace('/', '_slash_'))
        fpath = pjoin(xml.pts_path, 'openbenchmarking.org-searches', fname)
        np.savetxt(fpath, np.array(testids, dtype=np.str), fmt='%22s')

    # load the cache list
    xml.make_cache_set()

    # if cache is used, only download new cases
    if use_cache:
        testids = set(testids) - xml.testid_cache
        nr_testids = len(testids)
        if nr_testids < 1:
            print('all testids have already been downloaded')
            return
        print('')
        print('start downloading {} new test id\'s'.format(nr_testids))
        print('')

    for testid in tqdm(testids):
        # download xml file
        try:
            xml.load_testid_from_obm(testid, use_cache=False, save_xml=save_xml)
        except OSError as e:
            print(f'failed downloading: {testid}')
            print(e)

    return testids


def split_json(df_dict):
    """
    """
    cols = set(['compiler', 'compiler-options', 'compiler-type',
                'install-footnote', 'max-result', 'min-result'])
    cols_float = set(['max-result', 'min-result'])
    for col in cols:
        df_dict[col] = []

    for val in df_dict['JSON']:
        # don't even try to decode JSON if it is too short
        if len(val) < 2:
            json_dict = {}
        else:
            json_dict = json.loads(val)

        # seems the data could have one level of hierarchy less
        if 'compiler-options' in json_dict:
            json_dict.update(json_dict.pop('compiler-options'))
        # add splitted JSON data to current row
        for col, val in json_dict.items():
            df_dict[col].append(val)
        # empty values for missing JSON data at current row
        for col in cols - set(json_dict.keys()) - cols_float:
            df_dict[col].append('')
        for col in cols_float - set(json_dict.keys()):
            df_dict[col].append(np.nan)
    df_dict.pop('JSON')
    df_dict.pop('compiler-options')
    df_dict.pop('install-footnote')

    return df_dict


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
        sel = cores[0].str.contains('Total').values.astype(bool)
        cores.loc[sel, 0] = freq[1].loc[sel].str.split('Cores ', expand=True)[1]
        cores.loc[sel, 0] = cores.loc[sel, 0].str.replace(')', '')
        # we might have None values if nothing has been found in any of the
        # above steps. None only gets converted to nan via float32, not int16
        cores[0] = cores[0].astype(np.float32)
        cores.loc[np.isnan(cores[0].values), 0] = -1
        df['ProcessorCores'] = cores[0].astype(np.int16)
    df['ProcessorCount'] = count_name[0].astype(np.int16)
    df['ProcessorName'] = count_name[1]

    return df


def split_gpu_info(graphics):
    """Graphics tag contains optionally additionall information.

    AMD Radeon HD 6450/7450/8450 / R5 230 OEM 1024MB (625/800MHz)
    MSI AMD Radeon R7 370 / R9 270/370 OEM 4096MB (350/1400MHz)
    AMD Radeon HD 6770 1024MB (850/1200MHz)
    ATI Radeon HD 5450 512MB (650/333MHz)
    AMD Radeon HD 5450 (650/400MHz)
    Sapphire AMD Radeon HD 5450 512MB (650/400MHz)
    Gigabyte AMD Radeon HD 6450/7450/8450 / R5 230 OEM 1024MB
    ATI Radeon HD 5450 ATI Radeon HD 4290 CrossFire 1024MBMB (650/667MHz)
    (1150MHz)
    (450/532MHz)
    1 x ATI Radeon HD 4870 X2 1024MB CrossFire (750/900MHz)
    1024MB (400/399MHz)
    2 x AMD Radeon HD 6800 1024MB CrossFire (790/1000MHz)
    AMD FireGL V5600 512MB (800/1100MHz)
    AMD FirePro 2270
    XFX AMD Radeon HD 5000/6000/7350/8350 1024MB

    MSI NVIDIA GeForce GTX 460 768MB (675/1804MHz)
    MSI NVIDIA GeForce GTX 550 Ti 1024MB (50/135MHz)
    NVIDIA GeForce GT 220 /3DNOW! 1024MB (625/405MHz)
    NVIDIA GeForce GT 440/PCIe/SSE2 1024MB (810/900MHz)
    """

    '?:([0-9])(?: x ))?'
    '([a-zA-Z]*) ?'
    '(ATI|AMD|NVIDIA)? ?'
    '(R[0-9]|Radeon(?: HD)?(?: R[0-9])?|FireGL|FirePro|GeForce( GTX| GT)?) ?'
    '((([A-Z]?[0-9]{3,4}( Ti)?)\/?){0,4})'
    '*(OEM)* *(CrossFire)* *'
    '([0-9]*MB|MBMB|GB)* *'
    '(CrossFire)* *'
    '(\((([0-9]*)/([0-9]*))MHz\))*'


    x1 = r'(?:([0-9])(?: x ))?'
    brand = r'([a-zA-Z]*) ?'
    chip = r'(ATI|AMD|NVIDIA) ?'
    series = r'(R[0-9]|Radeon(?: HD)?(?: R[0-9])?|FireGL|FirePro|'\
             r'GeForce(?: GTX| GT)?) ?'
    productid = r'((?:(?:[A-Z]?[0-9]{3,4}(?: Ti)?)\/?){0,4})'
    p2 = r' *(OEM)* *(CrossFire)* ?'
    memory = r'(?:([0-9]*MB|MBMB|GB) )?'
    p2 = r'(CrossFire)* *'
    freq = r'(?:\((?:([0-9]*)/([0-9]*))MHz\))*'
    regex = re.compile(x1+brand+chip+series+productid+p2+memory+p2+freq)
    r2 = re.compile(series)
    r2.findall('R[0-9]')

    q='1 x MSI ATI Radeon HD 4870 1024MB CrossFire (750/900MHz)'
    q='Gigabyte AMD Radeon HD 6450/7450/8450 OEM / R5 230 OEM 1024MB'

    for k in q.split(' / '):
        print(regex.findall(k))

    if graphics.find('Radeon HD') > -1:
        chip = 'AMD'
        vendor = graphics.split('Radeon HD')[0].replace('AMD').replace('ATI')


def split_memory_info():
    """
    1 x 8192 MB DDR3-1600MHz RMT3160ME68FAF1600
    1 x 8192 MB DDR3-1600MHz Kingston
    1 x 8192 MB DDR3-1600MHz Kingston KHX1600C9S3L
    1 x 8192 MB DDR4-2400MHz
    1 x 8192 MB DRAM
    1 x 8192 MB RAM
    1 x 8192 MB RAM QEMU
    1 x 16384 MB 2133MHz
    1024 MB + 256 MB + 256 MB + 512 MB DDR-266MHz
    2 GB + 2 GB + 4 GB + 4 GB DDR3-1333MHz
    MB
    Unknown + 16384 MB + Unknown + Unknown + Unknown + Unknown DDR3
    x 0 DDR2-667MHz
    x 0 Empty-Empty
    """
    pass


def explore_dataset(df, label1, label2, label3, min_cases=3):

    # how many cases per test, and how many per resolution
    df_dict = {'nr':[], 'l1-l2':[]}

    for l1, gr1 in df.groupby(label1):
        for l2, gr2 in gr1.groupby(label2):
            df_dict['nr'].append(len(gr2))
            df_dict['l1-l2'].append(l1 + ' // ' + l2)
#        print('{:5d} : {}'.format(len(gr), grname))

    df_tests = pd.DataFrame(df_dict)
    df_tests.sort_values('nr', inplace=True)

    print('top 10 in nr of cases per {}'.format(label1))
    print(df_tests[-10:])

    for col in df_tests['l1-l2']:
        l1 = col.split(' // ')[0]
        df_sel = df[df[label1]==l1]
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

    return df_tests


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


def find_items_in_field(df, search_items, field):
    """Find those data points for which field contains any item in search_items

    Parameters
    ----------

    df : pd.DataFrame

    search_items : list

    field : str

    Returns
    -------

    df_sel : pd.DataFrame
        Selection of the results that contain results that contains any entry
        defined in search_items.

    """

    locsel = False
    for item in search_items:
        ksel = df[field].str.contains(item, case=False).values
        df['%s %s' % (field, item)] = ksel
        locsel += ksel
    return df.loc[locsel]


def find_results_with_items_in_field(df, search_items, field,
                                     gr1='ResultIdentifier',
                                     gr2='ResultDescription'):
    """Find the tests that contain results for all search parameters.
    ResultIdentifier's equal to None are ignored. Requires find_items to be
    run first because it relies on the added columns of find_i.

    Parameters
    ----------

    df : pd.DataFrame

    search_items : list

    field : str

    Returns
    -------

    resids : dict
        Dictionary with ResultIdentifier as key and a list of
        ResultDescription's as value.
    """

    # find tests for which all search_items have a data point
    resids = {}
    for resid, gr_resid in df.groupby(gr1):
        if resid == 'None': continue
        for resdesc, gr_resdesc in gr_resid.groupby(gr2):
            gpu_found = []
            for i in search_items:
                gpu_found.append(gr_resdesc['%s %s' % (field, i)].values.any())
            if set(gpu_found) == set([True]):
                if resid in resids:
                    resids[resid].append(resdesc)
                else:
                    resids[resid] = [resdesc]

    return resids


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
#    fig.tight_layout()

    return fig, ax


def plot_barh_groups(df, label_yval, label_group, label_xval='Value'):

    if len(df) < 1:
        return None, None

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
                    horizontalalignment='left', color='k')

        y0 += (len(gr) + gr_spacing)

    ax.set_ylim(-tick_height, yticks[-1] + tick_height)
    ax.set_yticks(yticks_center)
    ax.set_yticklabels(yticklabels_center)
    fig.tight_layout()

    return fig, ax


def move_to_folders():
    """migrate results to res_path/yy/mm/testid.xml from
    res_path/testid/composite.xml
    """

    obm = OpenBenchMarking()

    res_path = obm.res_path
    res_path2 = pjoin(obm.pts_path, 'test-results-all-obm-2/')
    res_path3 = pjoin(obm.pts_path, 'test-results-all-obm-leftover/')

    if not os.path.isdir(res_path3):
        os.makedirs(res_path3)

    base = os.path.join(res_path, '*')
    testids = glob(base)

    regex = re.compile(r'^[0-9]{7}\-[A-Za-z0-9]*\-[A-Za-z0-9]*$')

    for fpath in tqdm(testids):

        testid = os.path.basename(fpath)
        regex.findall(testid)
        if len(regex.findall(testid)) != 1:
            print('regex fail:', testid)
            shutil.move(fpath, res_path3)
            continue
        fpath = pjoin(res_path, testid, 'composite.xml')
        yy = testid[:2]
        mm = testid[2:4]
        if not os.path.isdir(pjoin(res_path2, yy, mm)):
            os.makedirs(pjoin(res_path2, yy, mm))
        shutil.copy2(fpath, pjoin(res_path2, yy, mm, testid + '.xml'))


if __name__ == '__main__':
    dummy = None

#    xml = xml2df()

#    search = 'RX 480'
#    testids = search_openbm(search=search, save_xml=False, use_cache=True)
#    testids = search_openbm(latest=True, save_xml=True)

#    xml = xml2df()
#    io = pjoin(xml.res_path, "1510217-HA-BPPADOKA880/composite.xml")
#    xml.load(io)
#    df_dict = xml.xml2dict()
#    dict_sys = xml.generated_system2dict()
#    dict_res = xml.data_entry2dict()

#    obm = xml2df()
#    io = pjoin(obm.res_path, "1606281-HA-RX480LINU80/composite.xml")
#    df_dict = obm.xml2dict(io)
