"""
Created on March 21, 2018

@author: Alejandro Molina
"""
import ast

from spn.io.Text import spn_to_str_equation
from spn.io.Text import add_str_to_spn, add_node_to_str
from collections import OrderedDict
from lark.lexer import Token
import inspect
import numpy as np

from spn.structure.leaves.histogram.Histograms import Histogram
import logging

logger = logging.getLogger(__name__)


def histogram_to_str(node, feature_names=None, node_to_str=None):
    decimals = 4
    if feature_names is None:
        fname = "V" + str(node.scope[0])
    else:
        fname = feature_names[node.scope[0]]

    breaks = np.array2string(np.array(node.breaks), precision=decimals, separator=",")
    densities = np.array2string(
        np.array(node.densities), precision=decimals, separator=","  # formatter={"float_kind": lambda x: "%.10f" % x}
    )
    bin_repr_points = np.array2string(
        np.array(node.bin_repr_points),
        precision=decimals,
        separator=",",
        # formatter={"float_kind": lambda x: "%.10f" % x},
    )

    return "Histogram(%s|%s;%s;%s)" % (fname, breaks, densities, bin_repr_points)


def histogram_tree_to_spn(tree, features, obj_type, tree_to_spn):
    # Note: tree.children consists of 1 or 2 Tokens (Node-Name [PARAMNAME], Feature-Name [FNAME])
    # followed by 2 or 3 Trees (breaks, densities, bin_repr_points)
    # If first child is a Node-Name (e.g. 'HistogramNode_1'), the first index is being ignored
    index_shift = 0
    if type(tree.children[0]) is Token and tree.children[0].type == "PARAMNAME":
        index_shift = 1

    # We may assume there are at least two Trees
    feature = str(tree.children[index_shift])
    breaks = list(map(ast.literal_eval, tree.children[1 + index_shift].children))
    densities = list(map(ast.literal_eval, tree.children[2 + index_shift].children))

    # Check if there is a third Tree (i.e. 'bin_repr_points' is available)
    tree_count = len(tree.children) - index_shift - 1
    if tree_count == 3:
        bin_repr_points = list(map(ast.literal_eval, tree.children[3 + index_shift].children))
    else:
        # ToDo: bin_repr_points might not generated correctly. Confirmation and/or fix needed. (2020-JAN-16)
        # 'bin_repr_points' is not available, reconstruct it
        bin_repr_points = ((breaks + np.roll(breaks, -1)) / 2.0)[:-1]
        # Default is MetaType.DISCRETE, hence set type to 'int' and finally store as list
        bin_repr_points = bin_repr_points.astype(int).tolist()

    node = Histogram(breaks, densities, bin_repr_points)

    if features is not None:
        node.scope.append(features.index(feature))
    else:
        node.scope.append(int(feature[1:]))

    return node


def add_histogram_text_support():
    add_node_to_str(Histogram, histogram_to_str)

    add_str_to_spn(
        "histogram",
        histogram_tree_to_spn,
        # Grammar change (legacy support): 'bin_repr_points' / 3rd list has been marked optional. (2020-JAN-13)
        """
                   histogram: "Histogram(" FNAME "|" list ";" list (";" list)? ")"  """,
        None,
    )
