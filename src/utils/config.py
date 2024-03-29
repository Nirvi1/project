# -*- coding:utf-8 -*-
"""
File: config.py
File Created: 2021-02-24
Author: Nirvi Badyal
"""
import os
import json
from dotmap import DotMap


PATH_TO_SRC = "src/"


def generate_settings(prj_path=PATH_TO_SRC, prj_name='exp', conf_name='config.json'):
	with open(os.path.join(prj_path, prj_name, conf_name), 'r') as config_file:
		opt_config = json.load(config_file)
	return DotMap({**opt_config})
