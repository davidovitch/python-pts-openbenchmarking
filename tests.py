#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:48:45 2018

@author: dave
"""


import unittest

from openbenchmarking import split_cpu_str


class Tests(unittest.TestCase):

    def setUp(self):
        pass

    def test_split_cpu_str(self):
        cpu_strs = ['Intel Core i9-7980XE @ 4.40GHz (18 Cores / 36 Threads)',
                    'Intel Core i9-7980XE @ 4.40GHz (18 Cores)',
                    'AMD Phenom II X4 965 @ 3.40GHz (Total Cores: 40)',
                    '2 x AMD EPYC 7601 32-Core @ 2.20GHz (64 Cores)']
        expected = [(1, 4.4, 18, 36), (1, 4.4, 18, -1), (1, 3.4, 40, -1),
                    (2, 2.2, 64, -1)]

        for cpu_str, answer in zip(cpu_strs, expected):
            cpus, descr, freq, cores, threads = split_cpu_str(cpu_str)
            self.assertEqual(answer, (cpus, freq, cores, threads))


if __name__ == "__main__":
    unittest.main()
