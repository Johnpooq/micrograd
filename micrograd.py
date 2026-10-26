import os
import tempfile
import webbrowser

try:
    from engine import Value
except ModuleNotFoundError:  # pragma: no cover - supports package-style imports
    from karpathy.engine import Value


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    try:
        from graphviz import Digraph
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "graphviz is required to render the computation graph"
        ) from exc

    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label="{ %s | data %.4f| grad %.4f}" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


if __name__ == "__main__":
    # inputs x1,x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights w1,w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bias of the neuron
    b = Value(6.8813735870195432, label="b")
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"
    o.grad = 1.0
    o.backward()

    dot = draw_dot(o)
    svg = dot.pipe(format="svg").decode("utf-8")

    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(svg)
        path = f.name

    webbrowser.open("file://" + os.path.abspath(path))
