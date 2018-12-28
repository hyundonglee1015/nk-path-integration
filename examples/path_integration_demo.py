#!/usr/bin/env python

"""
Generic LPU demo for path-integration

Notes
-----
Generate input file and LPU configuration by running

cd data
python gen_generic_lpu.py
"""

import argparse
import itertools
import os, sys

import networkx as nx

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

from neurokernel.LPU.LPU import LPU

from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

import neurokernel.mpi_relaunch

from data import gen_path_integration as gpi


dt = 1e-4
'''
dur = 1.0
steps = int(dur/dt)
'''
steps = 1500

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=steps, type=int,
                    help='Number of steps [default: %s]' % steps)
parser.add_argument('-g', '--gpu_dev', default=0, type=int,
                    help='GPU device number [default: 0]')
args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = setup_logger(file_name=file_name, screen=screen)

man = core.Manager()

(comp_dict, conns) = LPU.lpu_parser('./data/generic_path_integration.gexf.gz')

fl_input_processor_h = FileInputProcessor('./data/h.h5')
fl_input_processor_vx = FileInputProcessor('./data/vx.h5')
fl_input_processor_vy = FileInputProcessor('./data/vy.h5')
fl_output_processor = FileOutputProcessor([('r',None)], 'r_output.h5', sample_interval=1)


man.add(LPU, 'ge', dt, comp_dict, conns,
        device=args.gpu_dev, input_processors = [fl_input_processor_vx, fl_input_processor_vy, fl_input_processor_h],
        output_processors = [fl_output_processor], debug=args.debug)

man.spawn()
man.start(steps=args.steps)
man.wait()
