##################################################
### Single Layer Neural Network - From Scratch
### Step 1: Simulated Data, 1 x, 2 units
##################################################

set.seed(42)

## simulate data
## true relationship is nonlinear: ytrue = x^2 + noise
n = 100
xdat = runif(n, -3, 3)
ydat = xdat^2 + rnorm(n, 0, 0.5)

## plot to see what we're working with
plot(xdat, ydat, pch=16, col="blue",
     main="Simulated data: y = x^2 + noise",
     xlab="x", ylab="y")
##################################################
### Forward Pass
###
### Three steps:
### 1. linear combinations -- z = W %*% x
### 2. activation function -- A = sigmoid(z)
### 3. output -- yhat = beta0 + A %*% beta
##################################################

## sigmoid activation function
## squishes any number into (0,1)
sigmoid = function(z) {
  1 / (1 + exp(-z))
}

## forward pass function
## X is n x (p+1) design matrix with column of 1s
## W is K x (p+1) matrix of input weights (includes bias)
## beta is (K+1) vector of output weights (includes intercept)
forward_pass = function(X, W, beta) {
  
  ## step 1: linear combinations
  ## W is K x (p+1), X is n x (p+1)
  ## Z is K x n -- one row per unit, one column per observation
  Z = W %*% t(X)
  
  ## step 2: activation
  ## A is K x n
  A = sigmoid(Z)
  
  ## step 3: output
  ## add row of 1s to A for the output intercept
  ## A_aug is (K+1) x n
  A_aug = rbind(1, A)
  
  ## yhat is n x 1
  yhat = t(A_aug) %*% beta
  
  return(list(yhat=yhat, Z=Z, A=A))
}
backward_pass = function(X, ydat, yhat, Z, A, W, beta) {
  
  nn = nrow(X)
  K = nrow(W)
  
  ## error for each observation
  ## n x 1
  error = ydat - yhat
  
  ## sigmoid derivative
  ## F'(z) = A*(1-A)
  ## K x n
  Fprime = A * (1 - A)
  
  ## gradient for output weights beta
  ## (K+1) x 1
  ## add row of 1s for intercept gradient
  A_aug = rbind(1, A)
  grad_beta = -2 * (A_aug %*% error) / nn
  
  ## gradient for input weights W
  ## K x (p+1)
  ## beta without intercept is beta[-1]
  beta_noint = beta[-1]
  
  ## delta is K x n
  ## error flowing back through output weights
  ## and through sigmoid derivative
  delta = (beta_noint %*% t(error)) * Fprime
  
  ## grad_W is K x (p+1)
  grad_W = -2 * (delta %*% X) / nn
  
  return(list(grad_beta=grad_beta, grad_W=grad_W))
}

#plot(nn_result$loss_history,
 #    type="l", col="blue", lwd=2,
  #   xlab="Epoch", ylab="MSE Loss",
   #  main="Training Loss over Epochs")
#early stopping= one of the three main ways to prevent overfitting, if the loss isnt really changing at a certain point we dont need as many
# iterations
#early stopping=stop training whent he loss on held out validation data stops improving, loss curve tells us this
#l2 regularization- adds a penalty to the loss function that shrinks weights toward 0. forces the netowkr to find
# simpler solutions and generalize better (adding below)
#reduce model complexity - fewer units K means less capacity to memorize noise
#the loss curve on our initial simulated data and network suggests convergence around epoch 300, indicating fewer epochs may be sufficient
#in practice, early stopping based on validation loss would be implemented to prevent overfitting

##################################################
### Initialize weights and run forward pass
##################################################

K = 2       ## number of units
p = 1       ## number of predictors

## add column of 1s to x for input bias
X = cbind(1, xdat)   ## n x (p+1)

## input weights: K x (p+1)
## one row per unit, one column per predictor plus bias
set.seed(42)
W = matrix(rnorm(K*(p+1)), nrow=K, ncol=p+1)

## output weights: (K+1) vector
## K weights plus one intercept
beta = rnorm(K+1)

## run forward pass
result = forward_pass(X, W, beta)
yhat = result$yhat

## check -- should be n predictions
cat("Number of predictions:", length(yhat), "\n")
cat("First few predictions:", round(head(yhat), 3), "\n")
cat("First few true values:", round(head(ydat), 3), "\n")

## compute initial loss
loss = mean((ydat - yhat)^2)
cat("Initial loss (MSE):", round(loss, 3), "\n")

## run backward pass using initial forward pass results
grad = backward_pass(X, ydat, yhat, 
                     result$Z, result$A, W, beta)

cat("Gradient for output weights (beta):\n")
print(round(grad$grad_beta, 4))
cat("\nGradient for input weights (W):\n")
print(round(grad$grad_W, 4))

##################################################
### Training Loop
###
### Forward pass -> loss -> backward pass -> 
### update weights -> repeat
##################################################
##################################################
### Full Neural Network with SGD and L2 Regularization
###
### Key additions vs basic version:
### 1. SGD -- compute gradient on batches not all data
### 2. L2 regularization -- add penalty to loss
###    penalizes large weights, prevents overfitting
###    new loss = MSE + lambda * sum(W^2 + beta^2)
###    gradient just adds 2*lambda*weight to each grad
##################################################

train_nn_sgd = function(X, ydat, 
                        K = 5,
                        learning_rate = 0.1,
                        n_epochs = 1000,
                        batch_size = 20,
                        lambda = 0.001) {
  
  nn = nrow(X)
  p = ncol(X) - 1
  
  ## initialize weights randomly
  set.seed(42)
  W = matrix(rnorm(K*(p+1)), nrow=K, ncol=p+1)
  beta = rnorm(K+1)
  
  ## storage for loss history
  ## one loss per epoch
  loss_history = rep(0, n_epochs)
  
  for(epoch in 1:n_epochs) {
    
    ## shuffle data at start of each epoch
    shuffle_idx = sample(1:nn)
    X_shuf = X[shuffle_idx, ]
    y_shuf = ydat[shuffle_idx]
    
    ## split into batches
    n_batches = floor(nn / batch_size)
    
    for(k in 1:n_batches) {
      
      ## get batch
      batch_idx = ((k-1)*batch_size + 1):(k*batch_size)
      X_batch = X_shuf[batch_idx, ]
      y_batch = y_shuf[batch_idx]
      
      ## forward pass on batch
      result = forward_pass(X_batch, W, beta)
      yhat_batch = result$yhat
      
      ## backward pass on batch
      grad = backward_pass(X_batch, y_batch, 
                           yhat_batch,
                           result$Z, result$A,
                           W, beta)
      
      ## add L2 regularization to gradients
      ## penalty doesn't apply to intercepts
      ## so we only penalize W and beta[-1]
      grad$grad_W = grad$grad_W + 2*lambda*W
      grad$grad_beta[-1] = grad$grad_beta[-1] + 
        2*lambda*beta[-1]
      
      ## update weights
      beta = beta - learning_rate * grad$grad_beta
      W = W - learning_rate * grad$grad_W
    }
    
    ## compute loss on full data after each epoch
    result_full = forward_pass(X, W, beta)
    yhat_full = result_full$yhat
    mse = mean((ydat - yhat_full)^2)
    
    ## L2 penalty term
    penalty = lambda * (sum(W^2) + sum(beta[-1]^2))
    loss_history[epoch] = mse + penalty
  }
  
  ## final predictions
  result_final = forward_pass(X, W, beta)
  yhat_final = result_final$yhat
  
  return(list(
    W = W,
    beta = beta,
    yhat = yhat_final,
    loss_history = loss_history
  ))
}

##################################################
### run and compare different settings
##################################################

## baseline: K=2, no regularization
nn_basic = train_nn_sgd(X, ydat,
                        K = 2,
                        learning_rate = 0.1,
                        n_epochs = 500,
                        batch_size = 20,
                        lambda = 0.0)

## more units: K=10
nn_K10 = train_nn_sgd(X, ydat,
                      K = 10,
                      learning_rate = 0.1,
                      n_epochs = 500,
                      batch_size = 20,
                      lambda = 0.0)

## more units with regularization
nn_K10_reg = train_nn_sgd(X, ydat,
                          K = 10,
                          learning_rate = 0.1,
                          n_epochs = 500,
                          batch_size = 20,
                          lambda = 0.01)
## K=2 with regularization
nn_K2_reg = train_nn_sgd(X, ydat,
                         K = 2,
                         learning_rate = 0.1,
                         n_epochs = 500,
                         batch_size = 20,
                         lambda = 0.01)

cat("Final MSE K=2 no reg:      ", 
    round(mean((ydat-nn_basic$yhat)^2), 3), "\n")
cat("Final MSE K=2 lambda=0.01: ", 
    round(mean((ydat-nn_K2_reg$yhat)^2), 3), "\n")
cat("Final MSE K=10 no reg:     ", 
    round(mean((ydat-nn_K10$yhat)^2), 3), "\n")
cat("Final MSE K=10 lambda=0.01:", 
    round(mean((ydat-nn_K10_reg$yhat)^2), 3), "\n")

## plot K=2 with reg fit
plot(xdat, ydat, pch=16, col="gray",
     main="Fit: K=2, lambda=0.01", xlab="x", ylab="y")
lines(sort(xdat), nn_K2_reg$yhat[order(xdat)],
      col="purple", lwd=2)
##################################################
### plot results
##################################################

par(mfrow=c(2,3))

## loss curves
plot(nn_basic$loss_history, type="l", col="blue", lwd=2,
     main="Loss: K=2, no reg", xlab="Epoch", ylab="MSE")

plot(nn_K10$loss_history, type="l", col="red", lwd=2,
     main="Loss: K=10, no reg", xlab="Epoch", ylab="MSE")

plot(nn_K10_reg$loss_history, type="l", col="darkgreen", lwd=2,
     main="Loss: K=10, lambda=0.01", xlab="Epoch", ylab="MSE")

## fits
plot(xdat, ydat, pch=16, col="gray",
     main="Fit: K=2, no reg", xlab="x", ylab="y")
lines(sort(xdat), nn_basic$yhat[order(xdat)], 
      col="blue", lwd=2)

plot(xdat, ydat, pch=16, col="gray",
     main="Fit: K=10, no reg", xlab="x", ylab="y")
lines(sort(xdat), nn_K10$yhat[order(xdat)], 
      col="red", lwd=2)

plot(xdat, ydat, pch=16, col="gray",
     main="Fit: K=10, lambda=0.01", xlab="x", ylab="y")
lines(sort(xdat), nn_K10_reg$yhat[order(xdat)], 
      col="darkgreen", lwd=2)

par(mfrow=c(1,1))

## final losses
cat("Final MSE K=2 no reg:      ", 
    round(mean((ydat-nn_basic$yhat)^2), 3), "\n")
cat("Final MSE K=10 no reg:     ", 
    round(mean((ydat-nn_K10$yhat)^2), 3), "\n")
cat("Final MSE K=10 lambda=0.01:", 
    round(mean((ydat-nn_K10_reg$yhat)^2), 3), "\n")
# train_nn = function(X, ydat, K=2, 
#                     learning_rate=0.1,
#                     n_epochs=1000) {
#   
#   nn = nrow(X)
#   p = ncol(X) - 1
#   
#   ## initialize weights randomly
#   set.seed(42)
#   W = matrix(rnorm(K*(p+1)), nrow=K, ncol=p+1)
#   beta = rnorm(K+1)
#   
#   ## storage for loss history
#   loss_history = rep(0, n_epochs)
#   
#   for(epoch in 1:n_epochs) {
#     
#     ## forward pass
#     result = forward_pass(X, W, beta)
#     yhat = result$yhat
#     
#     ## compute loss
#     loss_history[epoch] = mean((ydat - yhat)^2)
#     
#     ## backward pass
#     grad = backward_pass(X, ydat, yhat,
#                          result$Z, result$A, 
#                          W, beta)
#     
#     ## update weights
#     ## move in opposite direction of gradient
#     beta = beta - learning_rate * grad$grad_beta
#     W = W - learning_rate * grad$grad_W
#   }
#   
#   ## final forward pass for predictions
#   result = forward_pass(X, W, beta)
#   yhat = result$yhat
#   
#   return(list(
#     W = W,
#     beta = beta,
#     yhat = yhat,
#     loss_history = loss_history
#   ))
# }

## run the network
nn_result = train_nn_sgd(X, ydat, 
                     K=2,
                     learning_rate=0.1,
                     n_epochs=1000)

## plot loss over epochs
plot(nn_result$loss_history,
     type="l", col="blue", lwd=2,
     xlab="Epoch", ylab="MSE Loss",
     main="Training Loss over Epochs")

## plot predictions vs true values
plot(xdat, ydat, pch=16, col="blue",
     main="Neural Network Fit",
     xlab="x", ylab="y")
lines(sort(xdat), 
      nn_result$yhat[order(xdat)],
      col="red", lwd=2)
legend("top", 
       legend=c("true data", "nn fit"),
       col=c("blue","red"),
       pch=c(16,NA), lty=c(NA,1))

## final loss
cat("Initial loss:", round(nn_result$loss_history[1], 3), "\n")
cat("Final loss:", round(tail(nn_result$loss_history,1), 3), "\n")

##################################################
### Train/Test Split
###
### 75% train, 25% test
### same split for all models so comparison is fair
##################################################

set.seed(42)
nn = nrow(X)
train_idx = sample(1:nn, floor(0.75*nn))
test_idx = setdiff(1:nn, train_idx)

## split data
X_train = X[train_idx, ]
y_train = ydat[train_idx]
X_test = X[test_idx, ]
y_test = ydat[test_idx]

cat("Training observations:", nrow(X_train), "\n")
cat("Test observations:", nrow(X_test), "\n")

##################################################
### train all four models on training data only
##################################################

nn_K2 = train_nn_sgd(X_train, y_train,
                     K = 2,
                     learning_rate = 0.1,
                     n_epochs = 500,
                     batch_size = 20,
                     lambda = 0.0)

nn_K2_reg = train_nn_sgd(X_train, y_train,
                         K = 2,
                         learning_rate = 0.1,
                         n_epochs = 500,
                         batch_size = 20,
                         lambda = 0.01)

nn_K10 = train_nn_sgd(X_train, y_train,
                      K = 10,
                      learning_rate = 0.1,
                      n_epochs = 500,
                      batch_size = 20,
                      lambda = 0.0)

nn_K10_reg = train_nn_sgd(X_train, y_train,
                          K = 10,
                          learning_rate = 0.1,
                          n_epochs = 500,
                          batch_size = 20,
                          lambda = 0.01)

##################################################
### evaluate on TEST data
### this is the real performance measure
##################################################

## get test predictions for each model
## need to run forward pass on test data

get_test_mse = function(model_result, X_test, y_test) {
  result = forward_pass(X_test, 
                        model_result$W, 
                        model_result$beta)
  mse = mean((y_test - result$yhat)^2)
  return(round(mse, 3))
}

## compare train vs test MSE for each model
cat("\n=== Model Comparison ===\n")
cat(sprintf("%-20s %-12s %-12s\n", 
            "Model", "Train MSE", "Test MSE"))
cat(sprintf("%-20s %-12s %-12s\n",
            "-----", "---------", "--------"))

models = list(nn_K2, nn_K2_reg, nn_K10, nn_K10_reg)
names = c("K=2  no reg", "K=2  reg", 
          "K=10 no reg", "K=10 reg")

for(i in 1:4) {
  train_mse = round(mean((y_train - models[[i]]$yhat)^2), 3)
  test_mse = get_test_mse(models[[i]], X_test, y_test)
  cat(sprintf("%-20s %-12s %-12s\n", 
              names[i], train_mse, test_mse))
}

##################################################
### plot test fits
##################################################

par(mfrow=c(2,2))

plot_fit = function(model, title, color) {
  result = forward_pass(X_test, model$W, model$beta)
  yhat_test = result$yhat
  plot(X_test[,2], y_test, pch=16, col="gray",
       main=title, xlab="x", ylab="y")
  lines(sort(X_test[,2]), 
        yhat_test[order(X_test[,2])],
        col=color, lwd=2)
}

plot_fit(nn_K2,     "K=2  no reg",  "blue")
plot_fit(nn_K2_reg, "K=2  reg",     "purple")
plot_fit(nn_K10,    "K=10 no reg",  "red")
plot_fit(nn_K10_reg,"K=10 reg",     "darkgreen")

par(mfrow=c(1,1))0

##################################################
### Linear Regression Baseline
###
### fit linear regression on same train data
### compare test MSE to neural network
### linear regression should struggle with U-shape
### because it can only fit straight lines
##################################################

## extract raw x values for train and test
x_train_raw = X_train[, 2]
x_test_raw  = X_test[, 2]

## build clean dataframes
train_df = data.frame(y = y_train, x = x_train_raw)
test_df  = data.frame(y = y_test,  x = x_test_raw)

## check
cat("train_df dimensions:", nrow(train_df), ncol(train_df), "\n")
cat("test_df dimensions:", nrow(test_df), ncol(test_df), "\n")
head(train_df)

## fit linear regression
lm_fit = lm(y ~ x, data = train_df)

## predict on test
yhat_lm_test = predict(lm_fit, newdata = test_df)
cat("Length of predictions:", length(yhat_lm_test), "\n")

## test MSE
lm_test_mse = round(mean((test_df$y - yhat_lm_test)^2), 3)
cat("Linear Regression Test MSE:", lm_test_mse, "\n")
cat("Neural Network K=2 Test MSE: 0.220\n")

result_K2 = forward_pass(X_test, nn_K2$W, nn_K2$beta)

plot(test_df$x, test_df$y,
     pch=16, col="gray",
     main="Neural Network vs Linear Regression\non Test Data",
     xlab="x", ylab="y")

lines(sort(test_df$x),
      result_K2$yhat[order(test_df$x)],
      col="blue", lwd=2)

lines(sort(test_df$x),
      yhat_lm_test[order(test_df$x)],
      col="red", lwd=2, lty=2)

legend("top",
       legend=c("true data",
                "NN K=2 (MSE=0.220)",
                paste0("Linear Reg (MSE=9.323)")),
       col=c("gray","blue","red"),
       pch=c(16,NA,NA),
       lty=c(NA,1,2),
       cex=0.8)



##################################################
### NHANES Application
### Predicting Systolic Blood Pressure from
### behavioral and demographic predictors
##################################################

## install if needed
install.packages("nhanesA")
library(nhanesA)

##################################################
### Pull NHANES data
### We need two tables:
### BPX - blood pressure measurements (outcome)
### BMX - body measures (BMI)
### DIQ - diabetes/health indicators
### PAQ - physical activity
### ALQ - alcohol use
### SLQ - sleep
### DEMO - demographics (age)
##################################################

## pull each table for 2017-2018 cycle
bp   = nhanes("BPX_J")    ## blood pressure
demo = nhanes("DEMO_J")   ## demographics  
bmx  = nhanes("BMX_J")    ## body measures
paq  = nhanes("PAQ_J")    ## physical activity
alq  = nhanes("ALQ_J")    ## alcohol
slq  = nhanes("SLQ_J")    ## sleep

## check what we have
cat("Blood pressure rows:", nrow(bp), "\n")
cat("Demographics rows:", nrow(demo), "\n")
cat("Body measures rows:", nrow(bmx), "\n")
cat("Physical activity rows:", nrow(paq), "\n")
cat("Alcohol rows:", nrow(alq), "\n")
cat("Sleep rows:", nrow(slq), "\n")

## look at blood pressure columns
cat("\nBlood pressure columns:\n")
print(names(bp))
## we'll avergae across the four bp columns for our output
##################################################
### Select relevant columns and merge
### SEQN is the unique participant ID
### we merge everything on SEQN
##################################################

## select columns we need from each table
bp_clean = bp[, c("SEQN", "BPXSY1", "BPXSY2", 
                  "BPXSY3", "BPXSY4")]

demo_clean = demo[, c("SEQN", "RIDAGEYR")]  ## age

bmx_clean = bmx[, c("SEQN", "BMXBMI")]     ## BMI

paq_clean = paq[, c("SEQN", "PAD615")]     ## vigorous activity minutes

alq_clean = alq[, c("SEQN", "ALQ130")]     ## avg alcohol drinks per day

slq_clean = slq[, c("SEQN", "SLD012")]     ## sleep hours

## average systolic readings for stable outcome
bp_clean$systolic = rowMeans(bp_clean[, c("BPXSY1","BPXSY2",
                                          "BPXSY3","BPXSY4")],
                             na.rm=TRUE)

## keep only SEQN and systolic
bp_clean = bp_clean[, c("SEQN", "systolic")]

## merge all tables on SEQN
dat = merge(bp_clean, demo_clean, by="SEQN")
dat = merge(dat, bmx_clean, by="SEQN")
dat = merge(dat, paq_clean, by="SEQN")
dat = merge(dat, alq_clean, by="SEQN")
dat = merge(dat, slq_clean, by="SEQN")

## rename columns for clarity
names(dat) = c("SEQN", "systolic", "age", 
               "bmi", "phys_act", "alcohol", "sleep")

cat("Merged dataset rows:", nrow(dat), "\n")
cat("Columns:", names(dat), "\n")
cat("\nFirst few rows:\n")
head(dat)
cat("\nSummary:\n")
summary(dat)

#dropping the predictors that aren't going to add to our model, keeping the strong 3
##################################################
### Clean dataset
### drop physical activity and alcohol
### handle missing values
### keep: systolic, age, bmi, sleep
##################################################

## select only the variables we're keeping
dat_clean = dat[, c("SEQN", "systolic", "age", 
                    "bmi", "sleep")]

## remove rows with any NA
dat_clean = na.omit(dat_clean)

cat("Rows after removing NAs:", nrow(dat_clean), "\n")

## check ranges look sensible
summary(dat_clean)

##################################################
### check for outliers
##################################################

## systolic over 200 is extremely high but possible
## BMI over 60 is extremely rare
## sleep under 3 or over 12 is unusual

par(mfrow=c(2,2))
hist(dat_clean$systolic, main="Systolic BP", 
     col="lightblue", xlab="mmHg")
hist(dat_clean$age, main="Age", 
     col="lightblue", xlab="Years")
hist(dat_clean$bmi, main="BMI", 
     col="lightblue", xlab="kg/m2")
hist(dat_clean$sleep, main="Sleep Hours", 
     col="lightblue", xlab="Hours")
par(mfrow=c(1,1))

##################################################
### Scale predictors to mean 0 variance 1
### critical for neural networks
### large unscaled values cause gradient problems
### do NOT scale the outcome y
##################################################

## store scaling parameters for later interpretation
age_mean = mean(dat_clean$age)
age_sd   = sd(dat_clean$age)
bmi_mean = mean(dat_clean$bmi)
bmi_sd   = sd(dat_clean$bmi)
slp_mean = mean(dat_clean$sleep)
slp_sd   = sd(dat_clean$sleep)

## scale predictors
dat_clean$age_s   = (dat_clean$age   - age_mean) / age_sd
dat_clean$bmi_s   = (dat_clean$bmi   - bmi_mean) / bmi_sd
dat_clean$sleep_s = (dat_clean$sleep - slp_mean) / slp_sd

## verify scaling worked
cat("Scaled age   -- mean:", round(mean(dat_clean$age_s),3),
    "sd:", round(sd(dat_clean$age_s),3), "\n")
cat("Scaled bmi   -- mean:", round(mean(dat_clean$bmi_s),3),
    "sd:", round(sd(dat_clean$bmi_s),3), "\n")
cat("Scaled sleep -- mean:", round(mean(dat_clean$sleep_s),3),
    "sd:", round(sd(dat_clean$sleep_s),3), "\n")

##################################################
### Build X matrix and y vector
##################################################

## X with column of 1s for intercept
X_nhanes = cbind(1, 
                 dat_clean$age_s,
                 dat_clean$bmi_s,
                 dat_clean$sleep_s)

colnames(X_nhanes) = c("intercept", "age", "bmi", "sleep")

## outcome
y_nhanes = dat_clean$systolic

cat("\nX dimensions:", dim(X_nhanes), "\n")
cat("y length:", length(y_nhanes), "\n")

##################################################
### Train/test split -- 75/25
##################################################

set.seed(42)
nn_obs = nrow(X_nhanes)
train_idx_n = sample(1:nn_obs, floor(0.75*nn_obs))
test_idx_n  = setdiff(1:nn_obs, train_idx_n)

X_tr = X_nhanes[train_idx_n, ]
y_tr = y_nhanes[train_idx_n]
X_te = X_nhanes[test_idx_n, ]
y_te = y_nhanes[test_idx_n]

cat("Train observations:", nrow(X_tr), "\n")
cat("Test observations:", nrow(X_te), "\n")


##################################################
### Train Neural Network on NHANES data
##################################################

## K=2 no regularization
nhanes_K2 = train_nn_sgd(X_tr, y_tr,
                         K = 2,
                         learning_rate = 0.1,
                         n_epochs = 500,
                         batch_size = 32,
                         lambda = 0.0)

## K=10 no regularization
nhanes_K10 = train_nn_sgd(X_tr, y_tr,
                          K = 10,
                          learning_rate = 0.1,
                          n_epochs = 500,
                          batch_size = 32,
                          lambda = 0.0)

## K=10 with regularization
nhanes_K10_reg = train_nn_sgd(X_tr, y_tr,
                              K = 10,
                              learning_rate = 0.1,
                              n_epochs = 500,
                              batch_size = 32,
                              lambda = 0.001)

## K=25 with regularization
## more units since we now have 3 predictors
nhanes_K25_reg = train_nn_sgd(X_tr, y_tr,
                              K = 25,
                              learning_rate = 0.1,
                              n_epochs = 500,
                              batch_size = 32,
                              lambda = 0.001)

##################################################
### Linear regression baseline
##################################################

nhanes_df_tr = data.frame(y   = y_tr,
                          age = X_tr[,2],
                          bmi = X_tr[,3],
                          slp = X_tr[,4])

nhanes_df_te = data.frame(y   = y_te,
                          age = X_te[,2],
                          bmi = X_te[,3],
                          slp = X_te[,4])

lm_nhanes = lm(y ~ age + bmi + slp, 
               data = nhanes_df_tr)

yhat_lm_te = predict(lm_nhanes, 
                     newdata = nhanes_df_te)

lm_mse = round(mean((y_te - yhat_lm_te)^2), 3)

##################################################
### Compare all models on test data
##################################################

get_test_mse2 = function(model, X_te, y_te) {
  result = forward_pass(X_te, model$W, model$beta)
  round(mean((y_te - result$yhat)^2), 3)
}

cat("=== NHANES Model Comparison ===\n")
cat(sprintf("%-25s %-12s %-12s\n",
            "Model", "Train MSE", "Test MSE"))
cat(sprintf("%-25s %-12s %-12s\n",
            "-----", "---------", "--------"))

## linear regression
lm_train_mse = round(mean((y_tr - predict(lm_nhanes))^2), 3)
cat(sprintf("%-25s %-12s %-12s\n",
            "Linear Regression", lm_train_mse, lm_mse))

## neural networks
nn_models = list(nhanes_K2, nhanes_K10, 
                 nhanes_K10_reg, nhanes_K25_reg)
nn_names  = c("NN K=2 no reg", "NN K=10 no reg",
              "NN K=10 reg", "NN K=25 reg")

for(i in 1:4) {
  tr_mse = round(mean((y_tr - nn_models[[i]]$yhat)^2), 3)
  te_mse = get_test_mse2(nn_models[[i]], X_te, y_te)
  cat(sprintf("%-25s %-12s %-12s\n",
              nn_names[i], tr_mse, te_mse))
}

##################################################
### loss curves
##################################################

par(mfrow=c(2,2))
plot(nhanes_K2$loss_history, type="l", col="blue", lwd=2,
     main="Loss: K=2 no reg", xlab="Epoch", ylab="MSE")
plot(nhanes_K10$loss_history, type="l", col="red", lwd=2,
     main="Loss: K=10 no reg", xlab="Epoch", ylab="MSE")
plot(nhanes_K10_reg$loss_history, type="l", col="darkgreen", lwd=2,
     main="Loss: K=10 reg", xlab="Epoch", ylab="MSE")
plot(nhanes_K25_reg$loss_history, type="l", col="purple", lwd=2,
     main="Loss: K=25 reg", xlab="Epoch", ylab="MSE")
par(mfrow=c(1,1))

## predicted vs actual plot
## works for any number of dimensions
## shows how well predictions track true values

result_K25 = forward_pass(X_te, 
                          nhanes_K25_reg$W,
                          nhanes_K25_reg$beta)

par(mfrow=c(1,2))

## best neural network
plot(y_te, result_K25$yhat,
     pch=16, col=rgb(0,0,1,0.3),
     main="NN K=25 reg: Predicted vs Actual",
     xlab="Actual Systolic BP",
     ylab="Predicted Systolic BP")
abline(0, 1, col="red", lwd=2)

## linear regression
plot(y_te, yhat_lm_te,
     pch=16, col=rgb(0,0,0,0.3),
     main="Linear Regression: Predicted vs Actual",
     xlab="Actual Systolic BP",
     ylab="Predicted Systolic BP")
abline(0, 1, col="red", lwd=2)

par(mfrow=c(1,1))