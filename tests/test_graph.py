import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import graph_util

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_annette_from_json(network="cf_resnet50"):
    json_file = Path('tests','data',network+'.json')
    annette_graph = graph_util.AnnetteGraph(network, json_file)

    assert True

#TODO: test fuse Layers -> test delete Layers

def main():
    network = "densenet121"
    test_annette_from_json(network=network)

if __name__ == '__main__':
    main()
