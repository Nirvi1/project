# -*- coding:utf-8 -*-
"""
File: config.py
File Created: 2021-02-24
Author: Nirvi Badyal
"""
import os
import json
from dotmap import DotMap


PATH_SRC_LOCAL = "src/"


def prepare_opt(prj_path=PATH_SRC_LOCAL, prj_name='exp', conf_name='config.json'):
	with open(os.path.join(prj_path, prj_name, conf_name), 'r') as config_file:
		opt_config = json.load(config_file)
	# Merge dicts to dotmap
	return DotMap({**opt_config})
