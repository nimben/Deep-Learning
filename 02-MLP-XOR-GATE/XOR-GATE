import math

X=[[0, 0],
   [0,1],
   [1,0],
   [1,1]]

Y=[0,1,1,0]

w11 = 0.5
w12 = 0.5

w21 = 0.5
w22 = 0.5

w31 = 0.5
w32 = 0

b1 = 0.5
b2 = 0.5
b3 = 0.5

l_rate = 0.5

def sigmoid(x):
  return 1/(1 + math.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

def err_der(output, target):
  return output - target

for epoch in range(100000000):
  for i in range(len(X)):
     x1 = X[i][0]
     x2 = X[i][1]
     target = Y[i]

     z1 = x1*w11 +x2*w21 + b1
     h1 = sigmoid(z1)

     z2 = x1*w12 +x2*w22 + b2
     h2 = sigmoid(z2)

     z3 = h1*w31 + h2*w32 + b3
     output = sigmoid(z3)

     error = 0.5 * (target - output) ** 2

     delta3 = err_der(output, target) * sigmoid_derivative(output)
     delta2 = delta3 * w31 * sigmoid_derivative(h1)
     delta1 = delta3 * w32 * sigmoid_derivative(h2)

     w31_new = delta3 * h1
     w32_new =  delta3 * h2

     w11_new = delta2 * x1
     w12_mew = delta1 * x1
     w21_new = delta2  * x2
     w22_new = delta1 * x2

     b1_new = delta2
     b2_new = delta1
     b3_new = delta3

     w31 = w31 - (l_rate * w31_new)
     w32 = w32 - (l_rate * w32_new)
     w11 = w11 - (l_rate * w11_new)
     w12 = w12 - (l_rate * w12_mew)
     w21 = w21 - (l_rate * w21_new)
     w22 = w22 - (l_rate * w22_new)

     b1 = b1 - (l_rate * b1_new)
     b2 = b2 - (l_rate * b2_new)
     b3 = b3 - (l_rate * b3_new)



print("Final weights:")
print("w1 =", w11)
print("w2 =", w12)
print("Bias =", b1)
print("w3 =", w21)
print("w4 =", w22)
print("Bias =", b2)
print("w5 =", w31)
print("w6 =", w32)
print("Bias =", b3)

print("\nXOR Gate Output:")

for i in range(len(X)):
  x1=X[i][0]
  x2=X[i][1]

  z1 = x1*w11 +x2*w21 + b1
  h1= sigmoid(z1)
  z2 = x1*w12 +x2*w22 + b2
  h2= sigmoid(z2)
  z3 = h1*w31 + h2*w32 + b3
  pred = sigmoid(z3)

  print(x1, "XOR", x2, "=", pred)



