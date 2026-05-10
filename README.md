# Neural Network From Scratch: A Public Health Application

Built a single-layer neural network entirely from scratch in R (no Keras, 
no PyTorch) implementing forward pass, backpropagation, stochastic gradient 
descent, and L2 regularization from first principles.

**Application:** Predicting systolic blood pressure from behavioral predictors 
(age, BMI, sleep) using NHANES 2017–2018 data (n=5,142).

**Key finding:** The neural network marginally outperformed linear regression 
(1.3% improvement), demonstrating that when relationships are approximately 
linear and key predictors are missing, model complexity doesn't compensate 
for data limitations. This finding has direct implications for when to use 
ML vs. simpler methods in applied health settings.

**Tools:** R (base), NHANES data  
**Methods:** Neural networks, gradient descent, L2 regularization, train/test validation  
**Course:** STP 540 — Computational Statistics, ASU (Spring 2026)  
**Portfolio submission:** Passed — approved for publication on math.asu.edu
