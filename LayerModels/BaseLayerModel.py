#!/usr/bin/env python
from abc import ABCMeta, abstractmethod, abstractproperty 
from neurokernel.LPU.NDComponents.NDComponent import NDComponent


class BaseLayerModel(NDComponent):
        __metaclass__ = ABCMeta

        accesses = []
        updates = []
