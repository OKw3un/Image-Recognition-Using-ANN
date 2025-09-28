# Image Recognition Using Artificial Neural Networks

This project implements a simple Artificial Neural Network (ANN) that can recognize 5x5 images of the digits 1 and 2. The network is trained using 10 matrices of each digit, learning to correctly classify input matrices.

---

## How It's Made
**Technologies:** C#, Artificial Neural Networks, Basic Linear Algebra  

The project starts by generating 5x5 matrices representing the digits 1 and 2. These matrices serve as the training dataset. A **Neuron** class was implemented to represent an artificial neuron, storing inputs and weights. All weights were initialized randomly between 0 and 1.  

Using the Neuron class, a **Neural Network** structure was created with two output neurons (N1 and N2). Each input matrix contains 25 pixels, and the network is fully connected, meaning every pixel is connected to both neurons.  

Training follows a simplified learning rule:  
- For input matrices representing digit 1, the expected output of N1 is 1, and for digit 2, N2 should output 1.  
- If the neuron with the highest output does not match the target, weights are updated using: `w = w ± (λ * x)`  
- The network is trained for 40 epochs with a learning rate of λ = 0.03.

---

## Optimizations
- Random weight initialization with iterative updates ensures gradual learning.  
- Simple and small dataset allows efficient computation without additional optimization.  
- Matrix and vector operations are structured for readability and performance.

---

## Lessons Learned
- This project helped understand the core mechanics of neural networks and weight updates in practice.  
- Even with a small dataset, network accuracy varies based on learning rate and number of epochs.  
- Testing with unseen matrices highlights the importance of generalization and potential overfitting.  
- Building and training a neural network from scratch provides hands-on experience that is very educational.

---

### Sample Outputs
| Input | Target | Prediction |
|-------|--------|------------|
| Digit 1 matrix example | 1 | 1 |
| Digit 2 matrix example | 2 | 2 |
| ...   | ...    | ... |
