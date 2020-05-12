import logging
import os
import numpy as np
import capnp

from spn.io.Graphics import plot_spn
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, rebuild_scopes_bottom_up, assign_ids
from spn.structure.StatisticalTypes import Type, MetaType
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.parametric.Parametric import Gaussian

capnp.remove_import_hook()
capnp_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "capnproto/spflow.capnp"))
spflow_capnp = capnp.load(capnp_file)

logger = logging.getLogger(__name__)

metaType2Enum = {MetaType.REAL : "real", MetaType.BINARY : "binary", MetaType.DISCRETE : "discrete"}

enum2MetaType = {v : k for k, v in metaType2Enum.items()}

type2Enum = {Type.REAL : "real", Type.INTERVAL : "interval", Type.POSITIVE : "positive", Type.CATEGORICAL : "categorical",
             Type.ORDINAL : "ordinal", Type.COUNT : "count", Type.BINARY : "binary"}

enum2Type = {v : k for k, v in type2Enum.items()}

def unwrap_value(value):
    # If the value was defined in the module numpy, convert it to a
    # Python primitive type for serialization.
    if type(value).__module__ == np.__name__:
        return value.item()
    return value

def binary_serialize_product(product, file, is_rootNode, visited_nodes):
    # Serialize child nodes before node itself
    for c in product.children:
        binary_serialize(c, file, False, visited_nodes)
    # Construct inner product node message.
    prod_msg = spflow_capnp.ProductNode.new_message()
    children = prod_msg.init("children", len(product.children))
    for i, child in enumerate(product.children):
        children[i] = child.id
    # Construct surrounding node message
    node = spflow_capnp.Node.new_message()
    node.id = product.id
    node.product = prod_msg
    node.rootNode = is_rootNode
    node.write(file)

def binary_deserialize_product(node, node_map):
    child_ids = node.product.children
    # Resolve references to child nodes by ID.
    children = [node_map.get(id) for id in child_ids]
    # Check all childs have been resolved.
    assert None not in children, "Child node ID could not be resolved"
    product = Product(children = children)
    product.id = node.id
    return product

def binary_serialize_sum(sum, file, is_rootNode, visited_nodes):
    # Serialize child nodes before node itself
    for c in sum.children:
        binary_serialize(c, file, False, visited_nodes)
    # Construct innner sum node message
    sum_msg = spflow_capnp.SumNode.new_message()
    children = sum_msg.init("children", len(sum.children))
    for i, child in enumerate(sum.children):
        children[i] = child.id
    weights = sum_msg.init("weights", len(sum.weights))
    for i, w in enumerate(sum.weights):
        weights[i] = unwrap_value(w)
    # Construct surrounding node message
    node = spflow_capnp.Node.new_message()
    node.id = sum.id
    node.sum = sum_msg
    node.rootNode = is_rootNode
    node.write(file)

def binary_deserialize_sum(node, node_map):
    child_ids = node.sum.children
    # Resolve references to child nodes by ID.
    children = [node_map.get(id) for id in child_ids]
    # Check all childs have been resolved.
    assert None not in children, "Child node ID could not be resolved"
    sum = Sum(children = children, weights=node.sum.weights)
    sum.id = node.id
    return sum

def binary_serialize_gaussian(gauss, file, is_rootNode, visited_nodes):
    # Construct inner Gaussian leaf message
    gauss_msg = spflow_capnp.GaussianLeaf.new_message()
    gauss_msg.mean = unwrap_value(gauss.mean)
    gauss_msg.stddev = unwrap_value(gauss.stdev)
    # Check that scope is defined over a single variable
    assert len(gauss.scope) == 1, "Expecting Gauss to be univariate"
    gauss_msg.scope = unwrap_value(gauss.scope[0])
    # Construct surrounding node message.
    node = spflow_capnp.Node.new_message()
    node.gaussian = gauss_msg
    node.rootNode = is_rootNode
    node.id = gauss.id
    node.write(file)

def binary_deserialize_gaussian(node, node_map):
    gauss = Gaussian(node.gaussian.mean, node.gaussian.stddev, node.gaussian.scope)
    gauss.id = node.id
    return gauss

def binary_serialize_histogram(hist, file, is_rootNode, visited_nodes):
    # Construct inner histogram leaf message.
    hist_msg = spflow_capnp.HistogramLeaf.new_message()
    breaks = hist_msg.init("breaks", len(hist.breaks))
    for i,b in enumerate(hist.breaks):
        breaks[i] = int(hist.breaks[i])
    densities = hist_msg.init("densities", len(hist.densities))
    for i,d in enumerate(hist.densities):
        densities[i] = unwrap_value(hist.densities[i])
    reprPoints = hist_msg.init("binReprPoints", len(hist.bin_repr_points))
    for i,r in enumerate(hist.bin_repr_points):
        reprPoints[i] = unwrap_value(hist.bin_repr_points[i])
    hist_msg.type = type2Enum.get(hist.type)
    hist_msg.metaType = metaType2Enum.get(hist.meta_type)
    # Check that scope is defined over a single variable
    assert len(hist.scope) == 1, "Expecting Gauss to be univariate"
    hist_msg.scope = unwrap_value(hist.scope[0])
    # Construct surrounding node message.
    node = spflow_capnp.Node.new_message()
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
    hist = Histogram(breaks=breaks, densities=densities, bin_repr_points=reprPoints, scope=node.hist.scope,
                     type_=type, meta_type=metaType)
    hist.id = node.id
    return hist


def binary_serialize(node, file, is_rootNode, visited_nodes):
    if node.id not in visited_nodes:
        if isinstance(node, Product):
            binary_serialize_product(node, file, is_rootNode, visited_nodes)
        elif isinstance(node, Sum):
            binary_serialize_sum(node, file, is_rootNode, visited_nodes)
        elif isinstance(node, Histogram):
            binary_serialize_histogram(node, file, is_rootNode, visited_nodes)
        elif isinstance(node,Gaussian):
            binary_serialize_gaussian(node, file, is_rootNode, visited_nodes)
        else:
            raise NotImplementedError(f"No serialization defined for node {node}")
        visited_nodes.add(node.id)



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
        else:
            raise NotImplementedError(f"No deserialization defined for {which}")
        node_map[node.id] = deserialized
        if node.rootNode:
            nodes.append(deserialized)
    return nodes

def binary_serialize_to_file(rootNodes, fileName):
    # Buffering write, buffers up to 100 MiB.
    with open(fileName, "w+b", buffering=100*(2**20)) as outFile:
        numNodes = 0
        for spn in rootNodes:
            assert is_valid(spn), "SPN invalid before serialization"
            visited = set()
            binary_serialize(spn, outFile, True, visited)
            numNodes += len(visited)
        print(f"Serialized {numNodes} nodes to {fileName}")

def binary_deserialize_from_file(fileName):
    with open(fileName, "rb") as inFile:
        rootNodes = binary_deserialize(inFile)
        for root in rootNodes:
            rebuild_scopes_bottom_up(root)
            assert is_valid(root), "SPN invalid after deserialization"
        return rootNodes

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

    binary_serialize_to_file([spn], "test.bin")

    deserialized = binary_deserialize_from_file("test.bin")








