#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:39:36 2018

@author: dave
"""

from openbenchmarking import search_openbm


if __name__ == '__main__':
    ids = search_openbm(latest=True, save_xml=True)
