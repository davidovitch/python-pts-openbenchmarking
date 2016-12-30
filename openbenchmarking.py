# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:03:26 2016

@author: davidovitch
"""

#import urllib2
#response = urllib2.urlopen('http://www.example.com/')
#html = response.read()

import os
from os.path import join as pjoin

from lxml.html import fromstring
#from lxml import objectify
from lxml import etree
import urllib.request

import pandas as pd


class EditXML:

    def __init__(self):
        self.flocal = pjoin(os.environ['HOME'],
                            '.phoronix-test-suite/test-results/')
        self.url_base = 'http://openbenchmarking.org/result/{}&export=xml'
        self.hard_soft_tags = set(['Hardware', 'Software'])

    def merge(self, list_test_results):
        """DOESN'T MERGE ANYTHING YET
        """
        self.root = etree.Element('PhoronixTestSuite')
        for test_result in list_test_results:
            fpath = os.path.join(self.flocal, test_result, 'composite.xml')
            tree = etree.parse(fpath)
            root = tree.getroot()

    def load(self, io):

        tree = etree.parse(io)
        self.root = tree.getroot()
        self.io = io

    def write_local(self, test_result=None):
        if test_result is None:
            test_result = self.test_result
        fpath = os.path.join(self.flocal, test_result)
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


class xml2df:

    def __init__(self):
        self.flocal = os.path.join(os.environ['HOME'],
                                   '.phoronix-test-suite/test-results/')
        self.url_base = 'http://openbenchmarking.org/result/{}&export=xml'
        self.hard_soft_tags = set(['Hardware', 'Software'])

    def load(self, io):
        tree = etree.parse(io)
        self.root = tree.getroot()
        self.io = io

    def _split2dict(self, string):
        """Convert following string to dictionary:
        key1: value1, key2: value2, ...
        """
        elements = string.split(', ')
        return {k.split(':')[0]:k.split(':')[1] for k in elements}

    def _add2row(self, elements, columns, df_dict, missing_val=None):
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
            df_dict with missing_val as value.

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

        return df_dict

    def generated_system2df(self):
        """Convert to Pandas DataFrame. We will split all the data into two
        tables:
            * Generated + System
            * Result + Data + Entry

        This will result in duplicated data among different rows, but it avoids
        joining tables time and again when searching/selecting.
        """

        generated = ["Title", "LastModified", "TestClient", "Description",
                     "Notes", "InternalTags", "ReferenceID",
                     "PreSetEnvironmentVariables", 'XXX']
        system = ["Identifier", "Hardware", "Software", "User", "TimeStamp",
                  "TestClientVersion", "Notes", "JSON"]
        hardware = ['Processor', 'Motherboard', 'Chipset', 'Memory', 'Disk',
                    'Graphics', 'Audio', 'Network']
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
        dict_gen = {k:[] for k in generated}

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

        for key, value in dict_sys.items():
            print(key.rjust(28), len(value))

        # the indices for Identifier in Results/Data/Entry
        ids = {key:i for i,key in enumerate(dict_sys['Identifier'])}

        return pd.DataFrame(dict_sys)

    def date_entry2df(self):

        result = ["Identifier", "Title", "AppVersion", "Arguments",
                  "Description", "Scale", "Proportion", "DisplayFormat"]
        data_entry = ["Label", "Value", "RawString", "JSON"]

        dict_res = {k:[] for k in result+data_entry}

        res_elements = list(self.root.findall('Result'))

        for el in res_elements:
            for entry in el.find('Data').getchildren():
                identifier = entry.find('Identifier')


def dostuff():

    search_hardware = 'RX 480'
    search_descr = '1920 x 1080'

#    # download all of them seperately
#    for test_result in cases:
#        print('='*10, test_result)
#        obm = EditXML()
#        obm.load(test_result)
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
#    obm.load(','.join(cases))
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
#    obm.load(','.join(cases))
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
#    obm.load(','.join(cases))
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
#    obm.load(','.join(cases))
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

    # ========================================
    testname = 'unigine-heaven-rx-480'
    fname = pjoin(obm.flocal, testname, 'composite-original.xml')
    with open(fname, 'w') as f:
        f.write(text)

    response = urllib.request.urlopen(url)
    data = response.read()      # a `bytes` object
    text = data.decode('utf-8') # a `str`; this step can't be used if data is binary
    #xml = objectify.fromstring(text)
    #root = etree.fromstring(text)
    #tree = etree.parse(url)
    #root = tree.getroot()

    fname = pjoin(obm.flocal, testname, 'composite.xml')
    with open(fname, 'w') as f:
        f.write(etree.tostring(root).decode())

def get_all_profiles():

    url = 'http://openbenchmarking.org/s/AMD%20Radeon%20RX%20470&show_more'
    response = urllib.request.urlopen(url)
    data = response.read()      # a `bytes` object
    text = data.decode('utf-8') # a `str`; this step can't be used if data is binary

    tree = fromstring(text)
    # all profiles names are in h4 elements, and nothing else is, nice and easy
    # but if a title is given, the id is in the parent link
#    cases = [k.text for k in tree.cssselect('h4')]
    # url starts with /result/
    ids = [k.getparent().attrib['href'][8:] for k in tree.cssselect('h4')]



if __name__ == '__main__':

    obm = xml2df()
    obm.load(pjoin(obm.flocal, "1606281-HA-RX480LINU80/composite.xml"))
#    self = obm
    df_sys = obm.generated_system2df()

# Generated

# System
# System.Identifier

# Result
# Result.Identifier: test ID
# Result.Identifier.Data.Entry
# Result.Identifier.Data.Entry.Identifier
# Result.Identifier.Data.Entry.Value
# Result.Identifier.Data.Entry.Rawstring (: seperated)