# XOR Gate using Multi-Layer Perceptron

The **XOR (Exclusive OR)** gate is a classic example used to demonstrate the limitation of a single-layer perceptron.

Unlike AND and OR gates, XOR **cannot be solved using a single perceptron** because its data points are not linearly separable.

To solve XOR, we use a **Multi-Layer Perceptron (MLP)** with a hidden layer and train it using **backpropagation**.

---

## 🧠 What is XOR?

The XOR gate produces `1` when the two inputs are **different**, and `0` when they are **the same**.

| Input 1 | Input 2 | XOR Output |
| :-----: | :-----: | :--------: |
|    0    |    0    |      0     |
|    0    |    1    |      1     |
|    1    |    0    |      1     |
|    1    |    1    |      0     |

In other words:

```text
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

---

## ❌ Why doesn't a single perceptron work?

A single perceptron can only learn a **linear decision boundary**.

For XOR, the points are:

```text
        x2
        ↑
    1   ● (0,1) → 1        ● (1,1) → 0
       
        
    0   ● (0,0) → 0        ● (1,0) → 1
        └────────────────────────────→ x1
```

There is no single straight line that can separate the `0` outputs from the `1` outputs.

Therefore:

> **XOR is not linearly separable.**

This is the reason we introduce a **hidden layer**.

---

## 🏗️ Neural Network Architecture

The network used here contains:

* **2 input neurons**
* **2 hidden neurons**
* **1 output neuron**

```text
                 Hidden Layer
              ┌───────────────┐
              │               │
x1 ──────────►│      h1       │──────►
              │               │       \
              └───────────────┘        \
                                         ► Output
              ┌───────────────┐        /    ŷ
              │               │       /
x2 ──────────►│      h2       │──────►
              │               │
              └───────────────┘
```

The network parameters are:

```text
Hidden neuron 1:
W1, W2, b1

Hidden neuron 2:
W3, W4, b2

Output neuron:
W5, W6, b3
```

So there are **6 weights and 3 biases**.

---

## ⚙️ Forward Propagation

### Hidden neuron 1

$$
z_1 = W_1x_1 + W_2x_2 + b_1
$$

$$
h_1 = \sigma(z_1)
$$

### Hidden neuron 2

$$
z_2 = W_3x_1 + W_4x_2 + b_2
$$

$$
h_2 = \sigma(z_2)
$$

### Output neuron

$$
z_3 = W_5h_1 + W_6h_2 + b_3
$$

$$
\hat{y} = \sigma(z_3)
$$

where the sigmoid activation function is:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

---

## 📉 Error Function

The model uses **Mean Squared Error (MSE)**.

For one training example:

$$
E=\frac{1}{2}(y-\hat{y})^2
$$

where:

* \(y\) = actual output
* \(\hat{y}\) = predicted output

---

## 🔄 Backpropagation

The error is propagated backward through the network to calculate how much each weight contributed to the error.

The output-layer error term is:

$$
\delta_3=(\hat{y}-y)\hat{y}(1-\hat{y})
$$

For hidden neuron 1:

$$
\delta_1=
\delta_3W_5h_1(1-h_1)
$$

For hidden neuron 2:

$$
\delta_2=
\delta_3W_6h_2(1-h_2)
$$

The weights are then updated using:

$$
W_{new}=W_{old}-\eta\frac{\partial E}{\partial W}
$$

where \(\eta\) is the learning rate.

---

## 🔑 Weight Updates

The gradients for the six weights are:

|  Weight | Gradient        |
| :-----: | :-------------- |
| \(W_1\) | \(\delta_1x_1\) |
| \(W_2\) | \(\delta_1x_2\) |
| \(W_3\) | \(\delta_2x_1\) |
| \(W_4\) | \(\delta_2x_2\) |
| \(W_5\) | \(\delta_3h_1\) |
| \(W_6\) | \(\delta_3h_2\) |

The biases are updated using:

$$
b_1=b_1-\eta\delta_1
$$

$$
b_2=b_2-\eta\delta_2
$$

$$
b_3=b_3-\eta\delta_3
$$

---

## 🧪 Training Configuration

This implementation uses:

| Parameter       |              Value |
| --------------- | -----------------: |
| Inputs          |                  2 |
| Hidden neurons  |                  2 |
| Output neurons  |                  1 |
| Activation      |            Sigmoid |
| Loss            | Mean Squared Error |
| Learning Rate   |                0.5 |
| Training Epochs |             10,000 |
| Optimizer       |   Gradient Descent |

---

### 📁 Files

```text
02-mlp-XOR_GATE/
│
├── xor_mlp.py      # XOR implementation
└── README.md       # Explanation and theory
```
