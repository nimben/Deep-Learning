# AND Gate Perceptron

A simple implementation of a **Perceptron** trained to learn the logic of an **AND gate**.

This project was created as part of my journey into understanding the fundamentals of **Machine Learning and Neural Networks**, starting from a basic perceptron before moving towards more complex models.

---

## 🧠 What is a Perceptron?

A perceptron is one of the simplest forms of an artificial neuron.

It takes inputs, multiplies them by weights, adds a bias, and passes the result through an activation function.

The basic calculation is:

$$
z = w_1x_1 + w_2x_2 + b
$$

The activation function then converts this value into the predicted output.

---

## 🔢 AND Gate

An AND gate produces `1` only when **both inputs are 1**.

| Input 1 | Input 2 | Target |
| ------: | ------: | -----: |
|       0 |       0 |      0 |
|       0 |       1 |      0 |
|       1 |       0 |      0 |
|       1 |       1 |      1 |

The perceptron learns this relationship by adjusting its weights and bias during training.

---

## ⚙️ How Training Works

For every training example:

1. Calculate the weighted sum.
2. Apply the activation function.
3. Compare the prediction with the target.
4. Calculate the error.

$$
error = target - prediction
$$

5. If the prediction is incorrect, update the weights.

$$
w_{new} = w_{old} + \eta(error)(x)
$$

The bias is updated similarly:

$$
b_{new} = b_{old} + \eta(error)
$$

where \(\eta\) is the learning rate.

---

## 🔄 Training Parameters

```text
Learning Rate : 0.1
Epochs        : 1000
Initial w1    : 0.0
Initial w2    : 0.0
Initial bias   : 0.0
```

An **epoch** represents one complete pass through the training dataset.

---

## 💻 Implementation

The implementation is written in Python and executed using **Google Colab**.

The notebook contains:

* AND gate dataset
* Weight and bias initialization
* Step activation function
* Forward calculation
* Error calculation
* Perceptron weight update
* Training over multiple epochs
* Final prediction testing

---

## ✅ Expected Output

After training, the perceptron should learn the AND relationship:

```text
[0, 0] -> 0
[0, 1] -> 0
[1, 0] -> 0
[1, 1] -> 1
```

---





