"""Annette Graph Utils

This Module contains the Graph Utilities to generate Annette readable graphs from MMDNN
or read directly from json.
"""
from __future__ import print_function

import json
import logging
import numpy as np
import sys

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache-2.0"


class AnnetteGraph():
    """Annette DNN Graph Description object.

    Args:
        name (str): network name
        json_file (str, optional): location of the .json file the network
            will be generated from. Alternatively an empty network graph will
            be generated

    Attributes:
        :var model_spec (dict): model_spec dictionary with the following
        model_spec[name] (str): network name
        model_spec[layers] (dict): mmdnn style description of the layers
        model_spec[input_layers] (list): list of the input layer names
        model_spec[output_layers] (list): list of the output layer names
    """

    def __init__(self, name, json_file=None, in_dict=None):
        self.model_spec = dict()
        self.model_spec['name'] = name
        self.model_spec['layers'] = dict()

        if json_file:
            print("Loading from file: ", json_file)
            with open(json_file, 'r') as f:
                self.model_spec = json.load(f)
        if not 'input_layers' in self.model_spec:
            self.model_spec['input_layers'] = []
        if not 'output_layers' in self.model_spec:
            self.model_spec['output_layers'] = []
        self._make_input_layers()
        self._make_output_layers()
        self.topological_sort = self._get_topological_sort()

    def add_layer(self, layer_name, layer_attr, resort = False):
        """Add a layer to the network description graph -> model_spec[layers]

        Args:
            layer_name (str): Name of the Layer to add
            layer_attr (dict): mmdnn style layer attributes

        Returns:
            True if successful, False otherwise.

        TODO:
            * check for existing parents
            * check for duplicate layer_names
            * check for valid attributes
        """
        self.model_spec['layers'][layer_name] = layer_attr.copy()
        if resort:
            for p in layer_attr['parents']:
                if p in self.model_spec['layers'].keys():
                    self.model_spec['layers'][p]['children'].append(layer_name)
            self._make_input_layers()
            self._make_output_layers()
            self.topological_sort = self._get_topological_sort()
        return True

    def _make_output_layers(self):
        self.model_spec['output_layers'] = []
        for name, layer in self.model_spec['layers'].items():
            if len(layer['children']) == 0:
                self.model_spec['output_layers'].append(name)

    def _make_input_layers(self, rebuild=False):
        self.model_spec['input_layers'] = []
        for name, layer in self.model_spec['layers'].items():
            self.model_spec['layers'][name]['left_parents'] = len(
                layer['parents'])
            if len(layer['parents']) == 0:
                self.model_spec['input_layers'].append(name)

    def fuse_layer(self, primary, secondary):
        """Fuse secondary to primary layer

        Args:
            primary (str): Name of the primary layer
            secondary (str): Name of the secondary layer
        Returns:
            True if successful, False otherwise.
        """
        if secondary in self.model_spec['layers'] and primary in self.model_spec['layers']:
            logging.debug("fuse layer %s to layer %s" % (secondary, primary))
            self.model_spec['layers'][primary][self.model_spec['layers'][secondary]['type']] = \
                self.model_spec['layers'][secondary]
            self.delete_layer(secondary)
            return True
        else:
            return False

    def split_layer(self, primary, secondary):
        """Split primary to seconday layers

        Args:
            primary (str): Name of the primary layer
            secondary (list): Names of the secondary layers
        Returns:
            True if successful, False otherwise.
        """
        if primary in self.model_spec['layers']:
            logging.debug("Split layer %s" % (primary))
            logging.debug(self.model_spec['layers'][primary])
            logging.debug(secondary)
            new_layers = []
            for n, in_layer in enumerate(secondary):
                new_name = primary+"_"+in_layer
                new_layers.append(new_name)
                logging.debug(n, in_layer)
                logging.debug("Add layer")
                if n == 0:
                    # change layer name
                    self.model_spec['layers'][new_name] = self.model_spec['layers'].pop(
                        primary)
                else:
                    self.add_layer(
                        new_name, self.model_spec['layers'][new_layers[0]])
                # change layer type
                self.model_spec['layers'][new_name]['type'] = in_layer
                logging.debug(self.model_spec['layers'][new_name])
            logging.debug(new_layers)
            for n, in_layer in enumerate(new_layers):
                logging.debug("Edit %s" % in_layer)
                logging.debug(self.model_spec['layers'][in_layer])
                if n == 0:
                    # change parents
                    for p in self.model_spec['layers'][in_layer]['parents']:
                        logging.debug("parent: %s" %
                                      self.model_spec['layers'][p]['children'])
                        self.model_spec['layers'][p]['children'] = \
                            [in_layer if x == primary else x for x in self.model_spec['layers'][p]['children']]
                        logging.debug("new_parent: %s" %
                                      self.model_spec['layers'][p]['children'])
                    logging.debug("new layer name: %s" % in_layer)
                    # change children
                    logging.debug("new_child:" + new_layers[n+1])
                    self.model_spec['layers'][in_layer]['children'] = [
                        new_layers[n+1]]
                elif n == len(new_layers)-1:
                    logging.debug(n)
                    # change parents
                    logging.debug("new_parent:" + new_layers[n-1])
                    self.model_spec['layers'][in_layer]['parents'] = [
                        new_layers[n-1]]
                    # change children
                    for c in self.model_spec['layers'][in_layer]['children']:
                        logging.debug(c)
                        logging.debug("children: %s" %
                                      self.model_spec['layers'][c]['parents'])
                        self.model_spec['layers'][c]['parents'] = \
                            [in_layer if x ==
                                primary else x for x in self.model_spec['layers'][c]['parents']]
                        logging.debug("children: %s" %
                                      self.model_spec['layers'][c]['parents'])
                    logging.debug("current layer name: %s" % new_name)
                else:
                    self.model_spec['layers'][in_layer]['parents'] = [
                        new_layers[n-1]]
                    self.model_spec['layers'][in_layer]['children'] = [
                        new_layers[n+1]]
            return True

    def delete_layer(self, name):
        """Delete layer from Graph

        Args:
            name (str): Name of the layer to delete
        Returns:
            True if successful, False otherwise.
        """
        if name in self.model_spec['layers']:
            logging.info("Deleting layer %s" % name)

            if len(self.model_spec['layers'][name]['parents']) > 0:
                parents = (self.model_spec['layers'][name]['parents'])
            else:
                parents = []
                logging.debug("No parents")

            if len(self.model_spec['layers'][name]['children']) > 0:
                children = (self.model_spec['layers'][name]['children'])
            else:
                children = []
                logging.debug("No children")

            for p in parents:
                logging.debug("parents: %s" % p)
                logging.debug(self.model_spec['layers'][p]['children'])
                self.model_spec['layers'][p]['children'].remove(name)
                for c in children:
                    self.model_spec['layers'][p]['children'].append(c)
                logging.debug(self.model_spec['layers'][p]['children'])

            for c in children:
                logging.debug("children: %s" % c)
                logging.debug(self.model_spec['layers'][c]['parents'])
                self.model_spec['layers'][c]['parents'].remove(name)
                for p in parents:
                    self.model_spec['layers'][c]['parents'].append(p)
                logging.debug(self.model_spec['layers'][c]['parents'])

            if name in self.model_spec['input_layers']:
                for c in children:
                    self.model_spec['input_layers'].append(c)
                self.model_spec['input_layers'].remove(name)

            if name in self.model_spec['output_layers']:
                for p in parents:
                    self.model_spec['output_layers'].append(p)
                self.model_spec['output_layers'].remove(name)

            del self.model_spec['layers'][name]
            self._get_topological_sort()
            return True
        else:
            logging.warning("Layer %s does not exists" % name)
            return False

    def _get_topological_sort(self):
        """Resort Graph

        Returns:
            Topological Sort
        """
        self.topological_sort = self.model_spec['input_layers'][:]
        idx = 0
        for n in self.model_spec['layers']:
            self.model_spec['layers'][n]['left_parents'] = len(
                self.model_spec['layers'][n]['parents'])
        while idx < len(self.topological_sort):
            name = self.topological_sort[idx]
            current_node = self.model_spec['layers'][name]
            for next_node in current_node['children']:
                next_node_info = self.model_spec['layers'][next_node]
                # one node may connect another node by more than one edge.
                self.model_spec['layers'][next_node]['left_parents'] -= \
                    self._check_left_parents(name, next_node_info)
                if next_node_info['left_parents'] == 0:
                    self.topological_sort.append(next_node)
            idx += 1
        logging.debug(self.topological_sort)
        return self.topological_sort

    def _check_left_parents(self, in_node_name, node):
        count = 0
        for in_edge in node['parents']:
            if in_node_name == in_edge.split(':')[0]:
                count += 1
        return count

    def to_json(self, filename):
        with open(filename, 'w') as f:
            f.write(json.dumps(self.model_spec, indent=4))
        print("Stored to %s" % filename)