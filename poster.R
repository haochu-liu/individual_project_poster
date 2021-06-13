library(MASS)
library(class)
library(e1071)
library(ggplot2)
library(GGally)
library(readr)
library(randomForest)

diabetes_dataset <- read_csv("diabetes-dataset.csv")
df <- diabetes_dataset[, c(2, 6, 9)]
df <- df[df$Glucose != 0, ]
df <- df[df$BMI != 0, ]
df <- df[!duplicated(df), ]
df <- df[sample(nrow(df), 715), ]
ggpairs(data = df[, 1:2], mapping = ggplot2::aes(color = as.character(t(df[, 3])), alpha = 0.3))

# Percentage Split 75:25
choosing <- sample(nrow(df), 536)
df.tr <- df[choosing, ]
df.te <- df[-choosing, ]
ps <- data.frame(matrix(0, nrow = 5, ncol = 2), row.names = c("LDA", "LR", "QDA", "SVM", "KNN"))
colnames(ps) <- c("Train", "Test")

# Cross-validation
V <- 5
V.frac <- nrow(df)/V
cv <- data.frame(matrix(0, nrow = 5, ncol = 5), row.names = c("LDA", "LR", "QDA", "SVM", "KNN"))
colnames(cv) <- c("Mean", "SE", "Precision", "Recall", "F1-score")

# model1 - LDA
# PS
model1 <- lda(Outcome~., data = df.tr)
model1.test.pred <- predict(model1, df.te)$posterior[,2]

table(df.te$Outcome, model1.test.pred > 0.5)
ps[1, 2] <- sum(df.te$Outcome != (model1.test.pred > 0.5))/nrow(df.te)

model1.train.pred <- predict(model1, df.tr)$posterior[,2]
table(df.tr$Outcome, model1.train.pred > 0.5)
ps[1, 1] <- sum(df.tr$Outcome != (model1.train.pred > 0.5))/nrow(df.tr)

# CV
v.fold.lda.res <- numeric(V)
v.fold.lda.precision <- numeric(V)
v.fold.lda.recall <- numeric(V)
v.fold.lda.f1 <- numeric(V)
for (i in 1:V){
  v.test.idx <- ((i - 1) * V.frac + 1):(V.frac * i)
  v.train.idx <- setdiff(1:nrow(df), v.test.idx)
  rip.lda <-  lda(Outcome~., data = df, subset = v.train.idx)
  rip.lda.pred <- predict(rip.lda, df[v.test.idx, ])$posterior[,2]
  err.rate <- sum(df[v.test.idx, "Outcome"] != (rip.lda.pred > 0.5))/length(v.test.idx)
  t <- table(t(df[v.test.idx, "Outcome"]), rip.lda.pred > 0.5)
  v.fold.lda.res[i] <- err.rate
  v.fold.lda.precision[i] <- t[2, 2] / (t[2, 2] + t[1, 2])
  v.fold.lda.recall[i] <- t[2, 2] / (t[2, 2] + t[2, 1])
  v.fold.lda.f1[i] <- 2 * v.fold.lda.precision[i] * v.fold.lda.recall[i] / (v.fold.lda.precision[i] + v.fold.lda.recall[i])
}

cv[1, 1] <- mean(v.fold.lda.res)
cv[1, 2] <- sqrt(var(v.fold.lda.res))/sqrt(V)
cv[1, 3] <- mean(v.fold.lda.precision)
cv[1, 4] <- mean(v.fold.lda.recall)
cv[1, 5] <- mean(v.fold.lda.f1)

# model2 - LR
# PS
model2 <- glm(Outcome~., data = df.tr, family = "binomial")
model2.test.pred <- predict(model2, df.te, type="response")

table(df.te$Outcome, model2.test.pred > 0.5)
ps[2, 2] <- sum(df.te$Outcome != (model2.test.pred > 0.5))/nrow(df.te)

model2.train.pred <- predict(model2, df.tr, type="response")
table(df.tr$Outcome, model2.train.pred > 0.5)
ps[2, 1] <- sum(df.tr$Outcome != (model2.train.pred > 0.5))/nrow(df.tr)

# CV
v.fold.lr.res <- numeric(V)
v.fold.lr.precision <- numeric(V)
v.fold.lr.recall <- numeric(V)
v.fold.lr.f1 <- numeric(V)
for (i in 1:V){
  v.test.idx <- ((i - 1) * V.frac + 1):(V.frac * i)
  v.train.idx <- setdiff(1:nrow(df), v.test.idx)
  rip.lr <-  glm(Outcome~., data = df, subset = v.train.idx, family = "binomial")
  rip.lr.pred <- predict(rip.lr, df[v.test.idx, ], type = "response")
  err.rate <- sum(df[v.test.idx, "Outcome"] != (rip.lr.pred > 0.5))/length(v.test.idx)
  t <- table(t(df[v.test.idx, "Outcome"]), rip.lr.pred > 0.5)
  v.fold.lr.res[i] <- err.rate
  v.fold.lr.precision[i] <- t[2, 2] / (t[2, 2] + t[1, 2])
  v.fold.lr.recall[i] <- t[2, 2] / (t[2, 2] + t[2, 1])
  v.fold.lr.f1[i] <- 2 * v.fold.lr.precision[i] * v.fold.lr.recall[i] / (v.fold.lr.precision[i] + v.fold.lr.recall[i])
}

cv[2, 1] <- mean(v.fold.lr.res)
cv[2, 2] <- sqrt(var(v.fold.lr.res))/sqrt(V)
cv[2, 3] <- mean(v.fold.lr.precision)
cv[2, 4] <- mean(v.fold.lr.recall)
cv[2, 5] <- mean(v.fold.lr.f1)

# model3 - QDA
# PS
model3 <- qda(Outcome~., data = df.tr)
model3.test.pred <- predict(model3, df.te)$posterior[,2]

table(df.te$Outcome, model3.test.pred > 0.5)
ps[3, 2] <- sum(df.te$Outcome != (model3.test.pred > 0.5))/nrow(df.te)

model3.train.pred <- predict(model3, df.tr)$posterior[,2]
table(df.tr$Outcome, model3.train.pred > 0.5)
ps[3, 1] <- sum(df.tr$Outcome != (model3.train.pred > 0.5))/nrow(df.tr)

# CV
v.fold.qda.res <- numeric(V)
v.fold.qda.precision <- numeric(V)
v.fold.qda.recall <- numeric(V)
v.fold.qda.f1 <- numeric(V)
for (i in 1:V){
  v.test.idx <- ((i - 1) * V.frac + 1):(V.frac * i)
  v.train.idx <- setdiff(1:nrow(df), v.test.idx)
  rip.qda <-  qda(Outcome~., data = df, subset = v.train.idx)
  rip.qda.pred <- predict(rip.qda, df[v.test.idx, ])$posterior[,2]
  err.rate <- sum(df[v.test.idx, "Outcome"] != (rip.qda.pred > 0.5))/length(v.test.idx)
  t <- table(t(df[v.test.idx, "Outcome"]), rip.qda.pred > 0.5)
  v.fold.qda.res[i] <- err.rate
  v.fold.qda.precision[i] <- t[2, 2] / (t[2, 2] + t[1, 2])
  v.fold.qda.recall[i] <- t[2, 2] / (t[2, 2] + t[2, 1])
  v.fold.qda.f1[i] <- 2 * v.fold.qda.precision[i] * v.fold.qda.recall[i] / (v.fold.qda.precision[i] + v.fold.qda.recall[i])
}

cv[3, 1] <- mean(v.fold.qda.res)
cv[3, 2] <- sqrt(var(v.fold.qda.res))/sqrt(V)
cv[3, 3] <- mean(v.fold.qda.precision)
cv[3, 4] <- mean(v.fold.qda.recall)
cv[3, 5] <- mean(v.fold.qda.f1)

# model4 - SVM
# PS
model4 <- svm(Outcome~., data = df.tr)
model4.test.pred <- predict(model4, df.te, type="response")

table(df.te$Outcome, model4.test.pred > 0.5)
ps[4, 2] <- sum(df.te$Outcome != (model4.test.pred > 0.5))/nrow(df.te)

model4.train.pred <- predict(model4, df.tr, type="response")
table(df.tr$Outcome, model4.train.pred > 0.5)
ps[4, 1] <- sum(df.tr$Outcome != (model4.train.pred > 0.5))/nrow(df.tr)

# CV
v.fold.svm.res <- numeric(V)
v.fold.svm.precision <- numeric(V)
v.fold.svm.recall <- numeric(V)
v.fold.svm.f1 <- numeric(V)
for (i in 1:V){
  v.test.idx <- ((i - 1) * V.frac + 1):(V.frac * i)
  v.train.idx <- setdiff(1:nrow(df), v.test.idx)
  rip.svm <-  svm(Outcome~., data = df, subset = v.train.idx)
  rip.svm.pred <- predict(rip.svm, df[v.test.idx, ], type = "response")
  err.rate <- sum(df[v.test.idx, "Outcome"] != (rip.svm.pred > 0.5))/length(v.test.idx)
  t <- table(t(df[v.test.idx, "Outcome"]), rip.svm.pred > 0.5)
  v.fold.svm.res[i] <- err.rate
  v.fold.svm.precision[i] <- t[2, 2] / (t[2, 2] + t[1, 2])
  v.fold.svm.recall[i] <- t[2, 2] / (t[2, 2] + t[2, 1])
  v.fold.svm.f1[i] <- 2 * v.fold.svm.precision[i] * v.fold.svm.recall[i] / (v.fold.svm.precision[i] + v.fold.svm.recall[i])
}

cv[4, 1] <- mean(v.fold.svm.res)
cv[4, 2] <- sqrt(var(v.fold.svm.res))/sqrt(V)
cv[4, 3] <- mean(v.fold.svm.precision)
cv[4, 4] <- mean(v.fold.svm.recall)
cv[4, 5] <- mean(v.fold.svm.f1)

# model5 -KNN
# control parameters, estimating by CV
krange <- 2:75
v.fold.knn.mat <- matrix(0, nrow = length(krange), ncol = V)

for (i in 1:V){
  v.test.idx <- ((i - 1) * V.frac + 1):(V.frac * i)
  v.train.idx <- setdiff(1:nrow(df), v.test.idx)
  k.test.res <- numeric(length(krange))
  # loop over k
  for (j in krange){
    my.knn <-  knn(df[v.train.idx, 1:2], df[v.test.idx, 1:2], as.integer(t(df[v.train.idx, 3])), k = j)
    knn.cl <- as.numeric(my.knn) - 1
    knn.prob <- (attributes(knn(df[v.train.idx, 1:2], df[v.test.idx, 1:2], as.integer(t(df[v.train.idx, 3])), k = j, prob = T)))$prob
    my.test.pred <- knn.cl * knn.prob + (1 - knn.cl) * (1 - knn.prob)
    err.rate <- sum((my.test.pred >= 0.5) != df$Outcome[v.test.idx])/length(v.test.idx) 
    k.test.res[j - 1] <- err.rate
  }
  v.fold.knn.mat[, i] <- k.test.res 
}

boxplot(t(v.fold.knn.mat), xlab = "krange", ylab = "error rate", main = "Boxplot for k values", names = 2:75)

k.means <- apply(v.fold.knn.mat, 1, mean)
min(k.means)

ks.std.errs <- apply(v.fold.knn.mat, 1, function(x) sqrt(var(x)))/sqrt(V)
plot(krange, k.means, ylab = "mean error rate", main = "Mean error rates for k values")

# compare the minimum mean choices
min.ks <- which(k.means == min(k.means))
cbind(krange[min.ks], k.means[min.ks], ks.std.errs[min.ks])
cv[5, 1] <- k.means[min.ks][1]
cv[5, 2] <- ks.std.errs[min.ks][1]
best_k <- min(krange[min.ks])

v.fold.knn.precision <- numeric(V)
v.fold.knn.recall <- numeric(V)
v.fold.knn.f1 <- numeric(V)
for (i in 1:V){
  v.test.idx <- ((i - 1) * V.frac + 1):(V.frac * i)
  v.train.idx <- setdiff(1:nrow(df), v.test.idx)
  rip.knn <- knn(df[v.train.idx, 1:2], df[v.test.idx, 1:2], as.integer(t(df[v.train.idx, 3])), k = best_k)
  rip.cl <- as.numeric(rip.knn) - 1
  rip.prob <- (attributes(knn(df[v.train.idx, 1:2], df[v.test.idx, 1:2], as.integer(t(df[v.train.idx, 3])), k = best_k, prob = T)))$prob
  rip.knn.pred <- rip.cl * rip.prob + (1 - rip.cl) * (1 - rip.prob)
  t <- table(t(df[v.test.idx, "Outcome"]), rip.knn.pred > 0.5)
  v.fold.knn.precision[i] <- t[2, 2] / (t[2, 2] + t[1, 2])
  v.fold.knn.recall[i] <- t[2, 2] / (t[2, 2] + t[2, 1])
  v.fold.knn.f1[i] <- 2 * v.fold.knn.precision[i] * v.fold.knn.recall[i] / (v.fold.knn.precision[i] + v.fold.knn.recall[i])
}
cv[5, 3] <- mean(v.fold.knn.precision)
cv[5, 4] <- mean(v.fold.knn.recall)
cv[5, 5] <- mean(v.fold.knn.f1)

model5 <- knn(df.tr[, 1:2], df.te[, 1:2], as.integer(t(df.tr[, 3])), k = best_k)
model5.cl <- as.numeric(model5) - 1
model5.prob <- (attributes(knn(df.tr[, 1:2], df.te[, 1:2], as.integer(t(df.tr[, 3])), k = best_k, prob = T)))$prob
model5.pred <- model5.cl * model5.prob + (1 - model5.cl) * (1 - model5.prob)
table(df.te$Outcome, model5.pred > 0.5)
ps[5, 2] <- sum((model5.pred >= 0.5) != df.te$Outcome)/nrow(df.te)

model5.tr.cl <- as.numeric(knn(df.tr[, 1:2], df.tr[, 1:2], as.integer(t(df.tr[, 3])), k = best_k)) - 1
model5.tr.prob <- (attributes(knn(df.tr[, 1:2], df.tr[, 1:2], as.integer(t(df.tr[, 3])), k = best_k, prob = T)))$prob
model5.tr.pred <- model5.tr.cl * model5.tr.prob + (1 - model5.tr.cl) * (1 - model5.tr.prob)
ps[5, 1] <- sum((model5.tr.pred >= 0.5) != df.tr$Outcome)/nrow(df.tr)

# ensemble
ensemble.mat <- cbind(model1.test.pred, model2.test.pred, model3.test.pred, model4.test.pred, model5.pred)
ensemble.pred <- apply(ensemble.mat, 1, mean)
ensemble.err <- sum((ensemble.pred >= 0.5) != df.te$Outcome)/nrow(df.te)

# Visualization
m <- 500
x <- seq(0, 210, length=m)
y <- seq(0, 90, length=m)
gr <- expand.grid(x, y)
colnames(gr) <- c("Glucose", "BMI")

gr.knn <- knn(df.tr[, 1:2], gr, as.integer(t(df.tr[, 3])), k = best_k)
gr.knn.cl <- as.numeric(gr.knn) - 1
gr.knn.prob <- (attributes(knn(df.tr[, 1:2], gr, as.integer(t(df.tr[, 3])), k = best_k, prob = T)))$prob
gr.pred <- gr.knn.cl * gr.knn.prob + (1 - gr.knn.cl) * (1 - gr.knn.prob)
gr.pred.mat <- matrix(gr.pred, ncol = m)

plot(df.tr[, 1:2])
points(df.tr[df.tr$Outcome == 0, 1:2], col = "red")
contour(x, y, gr.pred.mat, levels = 0.5, add = T, col = "blue", drawlabels = F)

model1.gr.pred <- matrix(predict(model1, gr)$posterior[,2], nrow = m)
contour(x, y, model1.gr.pred, levels = 0.5, add = T, col = "orange", drawlabels = F)

model2.gr.pred <- matrix(predict(model2, gr, type="response"), nrow = m)
contour(x, y, model2.gr.pred, levels = 0.5, add = T, col = "purple", drawlabels = F)

model3.gr.pred <- matrix(predict(model3, gr)$posterior[,2], nrow = m)
contour(x, y, model3.gr.pred, levels = 0.5, add = T, col = "yellow", drawlabels = F)

model4.gr.pred <- matrix(predict(model4, gr, type="response"), nrow = m)
contour(x, y, model4.gr.pred, levels = 0.5, add = T, col = "green", drawlabels = F)

# plotting
gr.knn <- knn(df.tr[, 1:2], gr, as.integer(t(df.tr[, 3])), k = 10)
gr.knn.cl <- as.numeric(gr.knn) - 1
gr.knn.prob <- (attributes(knn(df.tr[, 1:2], gr, as.integer(t(df.tr[, 3])), k = 10, prob = T)))$prob
gr.pred <- gr.knn.cl * gr.knn.prob + (1 - gr.knn.cl) * (1 - gr.knn.prob)
gr.pred.mat <- matrix(gr.pred, ncol = m)

plot(df.tr[, 1:2])
points(df.tr[df.tr$Outcome == 0, 1:2], col = "red")
contour(x, y, gr.pred.mat, levels = 0.5, add = T, col = "blue")

# recommendation
row.names(cv) <- c("LDA", "LR", "QDA", "SVM", "57-NN")
ggplot(cv) +
  geom_bar(aes(x = rownames(cv), y = Mean), stat = "identity", fill = "skyblue", alpha = 0.7) +
  geom_errorbar(aes(x = rownames(cv), ymin = Mean - SE, ymax = Mean + SE), width = 0.4, colour = "orange", alpha = 0.9, size = 1.3) +
  xlab("Models") +
  theme(text = element_text(size = 18))

barplot(as.matrix(t(ps)), beside = T, legend.text = T, args.legend = list(x = "topright", inset=c(-0.05, -0.45)), col = c("blue", "skyblue"), ylab = "error rates")
  



