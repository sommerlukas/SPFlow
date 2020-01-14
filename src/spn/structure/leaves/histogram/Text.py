"""
Created on March 21, 2018

@author: Alejandro Molina
"""
import ast

from spn.io.Text import spn_to_str_equation
from spn.io.Text import add_str_to_spn, add_node_to_str
from collections import OrderedDict
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

    return "Histogram(%s|%s;%s)" % (fname, breaks, densities)
    # ToDo: Revert ASAP -- Output of 'bin_repr_points' has been suppressed to be supported by SPNC. (2019-DEC-10)
    # return "Histogram(%s|%s;%s;%s)" % (fname, breaks, densities, bin_repr_points)


def histogram_tree_to_spn(tree, features, obj_type, tree_to_spn):
    # ToDo: Revert ASAP -- Parse-Tree structure has been changed for now; forced alternate processing. (2020-JAN-13)
    # Revert by removing True-branch, but note that it might be necessary to account for other grammar changes.
    if True:
        if len(tree.children) < 4:
            # Case: equation string to SPN
            breaks = list(map(ast.literal_eval, tree.children[1].children))
            densities = list(map(ast.literal_eval, tree.children[2].children))
            feature = str(tree.children[0])
        else:
            # Case: str-ref-graph to SPN
            # Since we use one function for equation and str_ref_graph conversion, there is an index shift
            # Caused by modifying the grammar w.r.t. the additional PARAMNAME e.g. "HistogramNode_1" (tree.children[0])
            breaks = list(map(ast.literal_eval, tree.children[2].children))
            densities = list(map(ast.literal_eval, tree.children[3].children))
            feature = str(tree.children[1])
        maxx = breaks[0]
        minx = breaks[-1]
        # ToDo: bin_repr_points is probably not generated correctly. Confirmation and/or fix needed. (2020-JAN-13)
        bin_repr_points = np.array([minx + (maxx - minx) / 2])
    else:
        breaks = list(map(ast.literal_eval, tree.children[1].children))
        densities = list(map(ast.literal_eval, tree.children[2].children))
        bin_repr_points = list(map(ast.literal_eval, tree.children[3].children))
        feature = str(tree.children[0])

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
        # ToDo: Revert ASAP -- Grammar change: 'bin_repr_points' / 3rd list has been marked optional. (2020-JAN-13)
        """
                   histogram: "Histogram(" FNAME "|" list ";" list (";" list)? ")"  """,
        None,
    )
