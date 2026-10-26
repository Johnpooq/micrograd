# Micrograd: a tiny autograd engine

A small, educational implementation of scalar reverse-mode automatic
differentiation and a multilayer perceptron (MLP). The project is intended to
make the mechanics of a computation graph, gradient propagation, and a basic
training loop easy to inspect.

This is a personal learning project inspired by Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd). It is not an official
Karpathy project or a production-ready machine-learning framework.

## What it demonstrates

- Scalar values connected in a computation graph
- Reverse-mode gradient propagation through arithmetic and common activations
- Neurons, layers, and an MLP built on the scalar autodiff engine
- A small training loop that learns a four-example classification task
- Tests comparing the engine's results with PyTorch autograd

## Project structure

| File | Purpose |
| --- | --- |
| `engine.py` | Scalar `Value` type, computation graph, and backward pass |
| `micrograd.py` | Computation graph tracing and optional SVG visualization |
| `nn.py` | Neuron, layer, MLP, and a small training demo |
| `test_engine.py` | Numerical and gradient checks against PyTorch |
| `requirements-dev.txt` | Dependencies for running the test suite |

## How it works

1. Each `Value` stores a scalar, its parent values, and the operation that
   produced it.
2. Arithmetic and activation methods attach local gradient rules to the
   computation graph.
3. `Value.backward()` visits the graph in reverse topological order to
   accumulate gradients from the output back to the inputs.
4. The MLP demo computes a squared-error loss, backpropagates it, and updates
   each parameter with a basic gradient-descent step.

## Requirements

- Python 3.10 or newer
- PyTorch and pytest to run the tests

The implementation and training demo use only the Python standard library.
Graph visualization is optional and additionally requires the Python `graphviz`
package and the Graphviz system executable.

## Setup

Create and activate a virtual environment, then install the test dependencies:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Run the training demo

```bash
python nn.py
```

The script prints the loss after each of 20 training steps. Importing the
modules does not start training.

## Run the tests

```bash
python -m pytest -q
```

## Optional computation graph visualization

Install the Graphviz Python package and the Graphviz system application, then
run `python micrograd.py` to open an example computation graph. The
`draw_dot(value)` helper can also be used to render a graph for another `Value`.
The Python package alone is not enough; the Graphviz executable must also be
available on your `PATH`.

## Attribution and license

This educational project was developed while studying the ideas in
[Andrej Karpathy's micrograd repository](https://github.com/karpathy/micrograd).
The upstream MIT license and its copyright notice are included in `LICENSE`.
