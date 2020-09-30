#-----------------------------------------------------------
# IDS - ICA project
#-----------------------------------------------------------

# 2 components

setwd('C:/Users/rosan/Desktop/miniproject/mixed_audios/2components')

library(tuneR)
library(fastICA)
mixed_1 <- readWave('mixed_1.wav')
mixed_2 <- readWave('mixed_2.wav')

play(mixed_1)
play(mixed_2)

mixed_1@samp.rate
# 44100
mixed_2@samp.rate
# 44100

mixed_1@left
# 559104 sample points
mixed_2@left
# 559104 sample points

# structure of the mixed_1
str(mixed_1)
# structure of the mixed_2
str(mixed_2)

X <- cbind(mixed_1@left, mixed_2@left)
a <- fastICA(X, 2, alg.typ = "parallel", fun = "logcosh", alpha = 1,
             method = "R", row.norm = FALSE, maxit = 200,
             tol = 0.0001, verbose = TRUE)

par(mfcol = c(2, 2))

plot(1:559104, X[, 1], type = "l", main = "Mixed Signals",
     xlab = "", ylab = "")
plot(1:559104, X[, 2], type = "l", xlab = "", ylab = "")
plot(1:559104, a$S[, 1], type = "l", main = "ICA source estimates",
     xlab = "", ylab = "")
plot(1:559104, a$S[, 2], type = "l", xlab = "", ylab = "")

sep_1 <- mixed_1
sep_2 <- mixed_2
sep_1@left <- a$S[, 1]*1000
sep_2@left <- a$S[, 2]*100
play(sep_1)
play(sep_2)

#-----------------------------------------------------------
# three components
#-----------------------------------------------------------
setwd('C:/Users/rosan/Desktop/miniproject/mixed_audios/3components')

mixed_1 <- readWave('mixed_1.wav')
mixed_2 <- readWave('mixed_2.wav')
mixed_3 <- readWave('mixed_3.wav')

mixed_1@samp.rate
# 44100

mixed_1@left
# 559104 sample points

X <- cbind(mixed_1@left, mixed_2@left, mixed_3@left)
a_3 <- fastICA(X, 3, alg.typ = "parallel", fun = "logcosh", alpha = 1,
             method = "R", row.norm = FALSE, maxit = 200,
             tol = 0.0001, verbose = TRUE)

par(mfcol = c(2, 2))

plot(1:559104, X[, 1], type = "l", main = "Mixed Signals",
     xlab = "", ylab = "")
plot(1:559104, X[, 2], type = "l", xlab = "", ylab = "")
plot(1:559104, a$S[, 1], type = "l", main = "ICA source estimates",
     xlab = "", ylab = "")
plot(1:559104, a$S[, 2], type = "l", xlab = "", ylab = "")

sep_1 <- mixed_1
sep_2 <- mixed_2
sep_3 <- mixed_3
sep_1@left <- a_3$S[, 1]*1000
sep_2@left <- a_3$S[, 2]*1000
sep_3@left <- a_3$S[, 3]*1000
play(sep_1)
play(sep_2)
play(sep_3)


#-----------------------------------------------------------
# 5 components
#-----------------------------------------------------------
setwd('C:/Users/rosan/Desktop/miniproject/mixed_audios/5components')

mixed_1 <- readWave('mixed_1.wav')
mixed_2 <- readWave('mixed_2.wav')
mixed_3 <- readWave('mixed_3.wav')
mixed_4 <- readWave('mixed_4.wav')
mixed_5 <- readWave('mixed_5.wav')

# play(mixed_1)
# play(mixed_5)

mixed_1@samp.rate
# 44100

mixed_1@left
# 559104 sample points

X <- cbind(mixed_1@left, mixed_2@left, mixed_3@left, mixed_4@left, mixed_5@left)
a_5 <- fastICA(X, 5, alg.typ = "parallel", fun = "logcosh", alpha = 1,
               method = "R", row.norm = FALSE, maxit = 200,
               tol = 0.0001, verbose = TRUE)

sep_4 <- mixed_1
sep_5 <- mixed_1
sep_1@left <- a_5$S[, 1]*1000
sep_2@left <- a_5$S[, 2]*1000
sep_3@left <- a_5$S[, 3]*1000
sep_4@left <- a_5$S[, 4]*1000
sep_5@left <- a_5$S[, 5]*1000
play(sep_1)
play(sep_2)
play(sep_3)
play(sep_4)
play(sep_5)




#-----------------------------------------------------------
# fastICA example codes
#-----------------------------------------------------------


#-----------------------------------------------------------
#Example 1: un-mixing two mixed independent uniforms
#-----------------------------------------------------------
S <- matrix(runif(10000), 5000, 2)
A <- matrix(c(1, 1, -1, 3), 2, 2, byrow = TRUE)
X <- S %*% A
a <- fastICA(X, 2, alg.typ = "parallel", fun = "logcosh", alpha = 1,
             method = "C", row.norm = FALSE, maxit = 200,
             tol = 0.0001, verbose = TRUE)
par(mfrow = c(1, 3))
plot(a$X, main = "Pre-processed data")
plot(a$X %*% a$K, main = "PCA components")
plot(a$S, main = "ICA components")
#-----------------------------------------------------------
#Example 2: un-mixing two independent signals
#-----------------------------------------------------------
S <- cbind(sin((1:1000)/20), rep((((1:200)-100)/100), 5))
A <- matrix(c(0.291, 0.6557, -0.5439, 0.5572), 2, 2)
X <- S %*% A
a <- fastICA(X, 2, alg.typ = "parallel", fun = "logcosh", alpha = 1,
             method = "R", row.norm = FALSE, maxit = 200,
             tol = 0.0001, verbose = TRUE)
par(mfcol = c(2, 3))
plot(1:1000, S[,1 ], type = "l", main = "Original Signals",
     xlab = "", ylab = "")
plot(1:1000, S[,2 ], type = "l", xlab = "", ylab = "")
plot(1:1000, X[,1 ], type = "l", main = "Mixed Signals",
     xlab = "", ylab = "")
plot(1:1000, X[,2 ], type = "l", xlab = "", ylab = "")
plot(1:1000, a$S[,1 ], type = "l", main = "ICA source estimates",
     xlab = "", ylab = "")
plot(1:1000, a$S[, 2], type = "l", xlab = "", ylab = "")
#-----------------------------------------------------------
#Example 3: using FastICA to perform projection pursuit on a
# mixture of bivariate normal distributions
#-----------------------------------------------------------
if(require(MASS)){
  x <- mvrnorm(n = 1000, mu = c(0, 0), Sigma = matrix(c(10, 3, 3, 1), 2, 2))
  x1 <- mvrnorm(n = 1000, mu = c(-1, 2), Sigma = matrix(c(10, 3, 3, 1), 2, 2))
  X <- rbind(x, x1)
  a <- fastICA(X, 2, alg.typ = "deflation", fun = "logcosh", alpha = 1,
               method = "R", row.norm = FALSE, maxit = 200,
               tol = 0.0001, verbose = TRUE)
  par(mfrow = c(1, 3))
  plot(a$X, main = "Pre-processed data")
  plot(a$X %*% a$K, main = "PCA components")
  plot(a$S, main = "ICA components")
}

#-----------------------------------------------------------
