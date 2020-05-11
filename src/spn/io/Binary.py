
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, rebuild_scopes_bottom_up, assign_ids
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.StatisticalTypes import Type, MetaType
from spn.io.Graphics import plot_spn
import logging

import os
import sys

import capnp
sys.path.append("./capnproto")
import spflow_capnp

logger = logging.getLogger(__name__)

metaType2Enum = {MetaType.REAL : "real", MetaType.BINARY : "binary", MetaType.DISCRETE : "discrete"}

enum2MetaType = {v : k for k, v in metaType2Enum.items()}

type2Enum = {Type.REAL : "real", Type.INTERVAL : "interval", Type.POSITIVE : "positive", Type.CATEGORICAL : "categorical",
             Type.ORDINAL : "ordinal", Type.COUNT : "count", Type.BINARY : "binary"}

enum2Type = {v : k for k, v in type2Enum.items()}

def binary_serialize_product(product, file, is_rootNode):
    for c in product.children:
        binary_serialize(c, file, False)
    prod_msg = spflow_capnp.ProductNode.new_message()
    children = prod_msg.init("children", len(product.children))
    for i, child in enumerate(product.children):
        children[i] = child.id
    node = spflow_capnp.Node.new_message()
    node.id = product.id
    scope = node.init("scope", len(product.scope))
    for i, v in enumerate(product.scope):
        scope[i] = v
    node.product = prod_msg
    node.rootNode = is_rootNode
    node.write(file)

def binary_deserialize_product(node, node_map):
    child_ids = node.product.children
    children = [node_map.get(id) for id in child_ids]
    product = Product(children = children)
    product.id = node.id
    product.scope = node.scope
    return product

def binary_serialize_sum(sum, file, is_rootNode):
    for c in sum.children:
        binary_serialize(c, file, False)
    sum_msg = spflow_capnp.SumNode.new_message()
    children = sum_msg.init("children", len(sum.children))
    for i, child in enumerate(sum.children):
        children[i] = child.id
    weights = sum_msg.init("weights", len(sum.weights))
    for i, w in enumerate(sum.weights):
        weights[i] = w
    node = spflow_capnp.Node.new_message()
    node.id = sum.id
    scope = node.init("scope", len(sum.scope))
    for i, v in enumerate(sum.scope):
        scope[i] = v
    node.sum = sum_msg
    node.rootNode = is_rootNode
    node.write(file)

def binary_deserialize_sum(node, node_map):
    child_ids = node.sum.children
    children = [node_map.get(id) for id in child_ids]
    sum = Sum(children = children, weights=node.sum.weights)
    sum.id = node.id
    sum.scope = node.scope
    return sum

def binary_serialize_gaussian(gauss, file, is_rootNode):
    gauss_msg = spflow_capnp.GaussianLeaf.new_message()
    gauss_msg.mean = gauss.mean
    gauss_msg.stddev = gauss.stdev
    node = spflow_capnp.Node.new_message()
    scope = node.init("scope", len(gauss.scope))
    for i, v in enumerate(gauss.scope):
        scope[i] = v
    node.gaussian = gauss_msg
    node.rootNode = is_rootNode
    node.id = gauss.id
    node.write(file)

def binary_deserialize_gaussian(node, node_map):
    mean = node.gaussian.mean
    stdev = node.gaussian.stddev
    gauss = Gaussian(mean, stdev, node.scope[0])
    gauss.id = node.id
    return gauss

def binary_serialize_histogram(hist, file, is_rootNode):
    hist_msg = spflow_capnp.HistogramLeaf.new_message()
    breaks = hist_msg.init("breaks", len(hist.breaks))
    for i,b in enumerate(hist.breaks):
        breaks[i] = int(hist.breaks[i])
    densities = hist_msg.init("densities", len(hist.densities))
    for i,d in enumerate(hist.densities):
        densities[i] = hist.densities[i]
    reprPoints = hist_msg.init("binReprPoints", len(hist.bin_repr_points))
    for i,r in enumerate(hist.bin_repr_points):
        reprPoints[i] = hist.bin_repr_points[i]
    hist_msg.type = type2Enum.get(hist.type)
    hist_msg.metaType = metaType2Enum.get(hist.meta_type)
    node = spflow_capnp.Node.new_message()
    scope = node.init("scope", len(hist.scope))
    for i, v in enumerate(hist.scope):
        scope[i] = v
    node.hist = hist_msg
    node.rootNode = is_rootNode
    node.id = hist.id
    node.write(file)

def binary_deserialize_histogram(node, node_map):
    breaks = node.hist.breaks
    densities = node.hist.densities
    reprPoints = node.hist.binReprPoints
    type = enum2Type.get(node.hist.type)
    metaType = enum2MetaType.get(node.hist.metaType)
    hist = Histogram(breaks=breaks, densities=densities, bin_repr_points=reprPoints, scope=node.scope[0],
                     type_=type, meta_type=metaType)
    hist.id = node.id
    return hist


def binary_serialize(node, file, is_rootNode = True):
    if isinstance(node, Product):
        binary_serialize_product(node, file, is_rootNode)
    elif isinstance(node, Sum):
        binary_serialize_sum(node, file, is_rootNode)
    elif isinstance(node, Histogram):
        binary_serialize_histogram(node, file, is_rootNode)
    elif isinstance(node,Gaussian):
        binary_serialize_gaussian(node, file, is_rootNode)


def binary_deserialize(file):
    node_map = {}
    nodes = []
    for node in spflow_capnp.Node.read_multiple(file):
        which = node.which()
        deserialized = None
        if which == "product":
           deserialized = binary_deserialize_product(node, node_map)
        elif which == "sum":
            deserialized = binary_deserialize_sum(node, node_map)
        elif which == "hist":
            deserialized = binary_deserialize_histogram(node, node_map)
        elif which == "gaussian":
            deserialized = binary_deserialize_gaussian(node, node_map)
        node_map[node.id] = deserialized
        if node.rootNode:
            nodes.append(deserialized)
    return nodes

def binary_serialize_to_file(spn, fileName):
    with open(fileName, "w+b") as outFile:
        binary_serialize(spn, outFile)

def binary_deserialize_from_file(fileName):
    with open(fileName, "rb") as inFile:
        return binary_deserialize(inFile)

if __name__ == "__main__":

    h1 = Histogram([0., 1., 2.], [0.25, 0.75], [1, 1], scope=1)
    h2 = Histogram([0., 1., 2.], [0.25, 0.75], [1, 1], scope=2)
    h3 = Histogram([0., 1., 2.], [0.25, 0.75], [1, 1], scope=1)
    h4 = Histogram([0., 1., 2.], [0.25, 0.75], [1, 1], scope=2)

    p0 = Product(children=[h1, h2])
    p1 = Product(children=[h3, h4])
    spn = Sum([0.3, 0.7], [p0, p1])

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    plot_spn(spn, "before.png")

    binary_serialize_to_file(spn, "test.bin")

    deserialized = binary_deserialize_from_file("test.bin")








