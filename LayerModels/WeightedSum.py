#!/usr/bin/env python
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
import os.path
import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *
from BaseDendriteModel import BaseDendriteModel

class WeightedSum(BaseDendriteModel):
    accesses = ['individual_input']
    updates = ['I']
    params = ['dummy']

    def __init__(self, params_dict, access_buffers, dt, debug=False,
                 LPU_id=None, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.LPU_id = LPU_id
        self.access_buffers = access_buffers
        self.params_dict = params_dict
        print(params_dict)
        self.debug = debug
        self.dt = dt
        self.LPU_id = LPU_id

        self.num_comps = params_dict['dummy'].size
        self.inputs = {
            k: garray.empty(self.num_comps, dtype = self.access_buffers[k].dtype)\
            for k in self.accesses}

    def run_step(self, update_pointers, st=None):
        self.sum_in_variable('individual_input',
                             self.inputs['individual_input'],
                             st = st)
        cuda.memcpy_dtod(int(update_pointers['I']), self.inputs['individual_input'].gpudata,
                         self.inputs['individual_input'].nbytes)
