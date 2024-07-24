library(gam)
library(xgboost)
library(tidyverse)
library(glmnet)
library(caret)
library(ggplot2)
library(ggrepel)
library(ggpubr)
library(dplyr)
library(plyr)
library(maps)
library(cowplot)
library(MASS)
library(rpart)
library(rpart.plot)
library(gglasso)

Data = read.csv("BrazHousesRent.csv", sep = ",", dec = ".", header = T, colClasses = "character")

Data <- unique(Data)
count_zero <- sum(Data$floor == "-")
print(count_zero)

floor_abs <- subset(Data, floor == "-", select = hoa..R..)
floor_abs[,1] = as.numeric(floor_abs[,1])
hist(floor_abs[, 1], xlab = "HOA", main = "Histogram of hoa..R.. for houses at '-' floor" )
count_zero <- sum(floor_abs[, 1] == 0)
print(count_zero)
print(count_zero/length(floor_abs[, 1]))

Data[Data == "-"] <- 0
categ <- c("city","animal","furniture")
cols <- colnames(Data)
for(i in 1:ncol(Data)){
  if (cols[i] %in% categ){
    Data[,i] = as.factor(Data[,i])
  }else{
    Data[,i] = as.numeric(Data[,i])
  }
}

average_rent_price <- function(City) {
  city_rent_amounts <- subset(Data, city == City, select = rent.amount..R..)
  average_rent <- mean(city_rent_amounts$rent.amount..R..)
  return(average_rent)
}
city_data <- data.frame(
  City = c("Belo Horizonte", "Campinas", "Porto Alegre", "Rio de Janeiro", "Sao Paulo"),
  Latitude = c(-19.9167, -22.9071, -30.0346, -22.9068, -23.5505),
  Longitude = c(-43.9345, -47.0632, -51.2177, -43.1729, -46.6333),
  Average_Rent = sapply(c("Belo Horizonte", "Campinas", "Porto Alegre", "Rio de Janeiro", "São Paulo"), 
                        function(x) average_rent_price(x))
)

city_data$Alpha <- (city_data$Average_Rent -
                      min(city_data$Average_Rent)) / (max(city_data$Average_Rent) 
                                                      - min(city_data$Average_Rent))

# Plot the geospatial data for Brazil only
map_brazil <- map_data("world", region = "Brazil")

# Create the plot
ggplot() +
  geom_polygon(data = map_brazil, aes(x = long, y = lat, group = group), fill = "lightgray", color = "white") +
  geom_label_repel(data = city_data, aes(x = Longitude, y = Latitude, label = City), color = "black", size = 3,
                   box.padding = 0.5, point.padding = 0.2, force = 1, segment.color = "transparent") +
  geom_point(data = city_data, aes(x = Longitude, y = Latitude, color = Average_Rent), alpha = 0.8, size = 5) +
  labs(title = "Average Rent in Different Cities in Brazil",
       x = "Longitude",
       y = "Latitude") +
  scale_color_gradient(low = "blue", high = "red") +
  theme_minimal()

cor(Data$fire.insurance..R..,Data$rent.amount..R..)

cont <- c("area","fire.insurance..R..","property.tax..R..","hoa..R..", "floor")
for(i in 1:ncol(Data)){
  if (cols[i] %in% cont){
    Q3 <- quantile(Data[,i], .75)
    IQR <- IQR(Data[,i])
    Data <- subset(Data, Data[,i]< (Q3 + 3.5*IQR))
  }
}

Data <- subset(Data, floor < 40)
Data <- subset(Data, rooms < 10)
Data <- subset(Data, bathroom < 7)

fit1 <- lm(log(rent.amount..R..) ~ area,data = Data)
summary(fit1)$coefficients
dtree <- rpart(rent.amount..R.. ~ rooms + furniture + parking.spaces, data = Data, method = "anova")
rpart.plot(dtree)


# Getting the datasets ready

### Turning variables animal and furniture to 0s and 1s

#Animal
vec <- Data$animal
vec <- as.character(vec)
vec[vec == "acept"] <- 1      # Replace "acept" by 1
vec[vec == "not acept"] <- 0       # Replace "not acept" by 0
vec <- as.factor(vec)
Data$animal <- vec

#Furniture
vec <- Data$furniture
vec <- as.character(vec)
vec[vec == "furnished"] <- 1      # Replace "furnished" by 1
vec[vec == "not furnished"] <- 0       # Replace "not furnished" by 0
vec <- as.factor(vec)
Data$furniture <- vec

### Train test split

set.seed(1)
library(caret)
trainrows <- createDataPartition(Data$rent.amount..R.., p=0.8, list=FALSE)
training_set <- Data[trainrows,]
d_test <- Data[-trainrows,]
trainrows <- createDataPartition(training_set$rent.amount..R.., p=0.8, list=FALSE)
d_train <- training_set[trainrows,]
d_val <- training_set[-trainrows,]

### Scaling

scale_data <- function(dataset, dataset2) {
  for(i in 1:ncol(dataset)){
    if (is.numeric(dataset[1,i])){
      dataset[,i] = scale(dataset[,i], center = mean(dataset2[,i]), scale = sd(dataset2[,i]))
    }
  }
  return(dataset)
} 
d_val <- scale_data(d_val, d_train)
d_test <- scale_data(d_test, training_set)
d_train <- scale_data(d_train, d_train) #Training set last since we use its mean and sd to scale the val set

### Encoding categorical variables

d_train_unenc <- d_train
d_val_unenc <- d_val

encode <- function(dataset, excluded=c()) {
  excluded_cols <- dataset[, names(dataset) %in% excluded]
  dataset <- dataset[,!names(dataset) %in% excluded]
  dmy <- dummyVars(" ~ .", data = dataset)
  dataset <- data.frame(predict(dmy, newdata = dataset))
  dataset <- cbind(dataset, excluded_cols)
  return(dataset)
}  
d_train <- encode(d_train, c("animal", "furniture"))
d_val <- encode(d_val, c("animal", "furniture"))
d_test <- encode(d_test, c("animal", "furniture"))
```

# Testing the models' performances, Model Selection

## 1.5 - AIC and BIC 

full.model <- lm(rent.amount..R.. ~ ., data = d_train)
step.model.aic <- stepAIC(full.model, direction = "both", trace = 0)

step.model.bic <- stepAIC(full.model, direction = "both", trace = 0, k = log1p(nrow(d_train)))

predictions.aic <- predict(step.model.aic, newdata = d_val)
predictions.bic <- predict(step.model.bic, newdata = d_val)

aic_mse <- mean((predictions.aic - d_val$rent.amount..R..)^2)
bic_mse <- mean((predictions.bic - d_val$rent.amount..R..)^2)
aic_rmse <- sqrt(aic_mse)
bic_rmse <- sqrt(bic_mse)
aic_rsquared <- summary(step.model.aic)$r.squared
bic_rsquared <- summary(step.model.bic)$r.squared

#Data frame to store the evaluation metrics
table <- data.frame(
  Model = character(),  
  MSE = numeric(),
  RMSE = numeric(),
  R_squared = numeric()
)
table <- rbind(table, c("AIC", round(aic_mse, 5), round(aic_rmse, 5), round(aic_rsquared, 5)))
table <- rbind(table, c("BIC", round(bic_mse, 5), round(bic_rmse, 5), round(bic_rsquared, 5)))
colnames(table) <- c("Model", "MSE", "RMSE", "R_squared")

print(table)

coef(step.model.bic)

bicplot <- ggplot(data.frame(Fitted = step.model.bic$fitted.values, Residuals = step.model.bic$residuals), aes(x = Fitted, y = Residuals)) +
  geom_point(color = "blue") + 
  theme_minimal() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Res v Fitted BIC", x = "Fitted Values", y = "Residuals")
aicplot <- ggplot(data.frame(Fitted = step.model.aic$fitted.values, Residuals = step.model.aic$residuals), aes(x = Fitted, y = Residuals)) +
  geom_point(color = "blue") + 
  theme_minimal() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Re Fitted AIC", x = "Fitted Values", y = "Residuals")
ggarrange(aicplot,bicplot,nrow = 1,ncol = 2)

## Lasso and Group-Lasso

X = model.matrix(rent.amount..R.. ~ ., data=d_train)[,-1]
y = d_train$rent.amount..R..
set.seed(333)
cvlasso = cv.glmnet(x = X, y = y,nfolds = 10)
groups <- c(1,1,1,1,1,2,3,4,5,6,7,8,9,10,11)
glasso = cv.gglasso(x=X, y=y, group = groups,nfolds = 10)

pen_val <- model.matrix(rent.amount..R.. ~ ., data=d_val)[,-1]
lassopredmin <- predict(cvlasso,pen_val, s = "lambda.min")
lmin_mse <- mean((lassopredmin - d_val$rent.amount..R..)^2)
lassopred1se <- predict(cvlasso,pen_val, s = "lambda.1se")
lse_mse <- mean((lassopred1se - d_val$rent.amount..R..)^2)
lmin_Rsq <- cor(lassopredmin,d_val$rent.amount..R..)^2

glpred1 <- predict(glasso,pen_val, s="lambda.1se")
gl1_MSE <- mean((glpred1 - d_val$rent.amount..R..)^2)
gl1_Rsq <- cor(glpred1,d_val$rent.amount..R..)^2
glpred2 <- predict(glasso,pen_val, s="lambda.min")
gl2_MSE <- mean((glpred2 - d_val$rent.amount..R..)^2)
gl2_Rsq <- cor(glpred2,d_val$rent.amount..R..)^2

table <- rbind(table, c("Lasso", round(lmin_mse, 5), round(sqrt(lmin_mse), 5), round(lmin_Rsq, 5)))
table <- rbind(table, c("grLasso", round(gl2_MSE, 5), round(sqrt(gl2_MSE), 5), round(gl2_Rsq, 5)))
colnames(table) <- c("Model", "MSE", "RMSE", "R_squared")
print(table)

## GAM

### Smoothing terms

num_names = names(d_train_unenc)[d_train_unenc %>% map_lgl(is.numeric)]
num_names = num_names %>% 
  discard(~.x %in% c("rent.amount..R.."))
num_feat = num_names %>% 
  map_chr(~paste0("s(", .x, ", 10)")) %>%
  paste(collapse = "+")

cat_feat = names(d_train_unenc)[d_train_unenc %>% map_lgl(is.factor)] %>% 
  paste(collapse = "+")

gam_form = as.formula(paste0("rent.amount..R.. ~", num_feat, "+", cat_feat))

### Model fitting

fit_gam = gam(formula = gam_form, family = "gaussian", data = d_train_unenc)

ggplot(data.frame(Fitted = fit_gam$fitted.values, Residuals = fit_gam$residuals), aes(x = Fitted, y = Residuals)) +
  geom_point(color = "red") + 
  theme_minimal() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "green") +
  labs(title = "Residuals vs Fitted Values", x = "Fitted Values", y = "Residuals")


### Predicting the validation set

predicted_values = predict(fit_gam, d_val_unenc) 

observed_values = d_val_unenc$rent.amount..R..
gam_mse = mean((predicted_values - observed_values)^2)
gam_rmse = sqrt(gam_mse)
gam_rsquared = cor(predicted_values, observed_values)^2

## XGBoost

### Setting up

X_train <- d_train[, !(colnames(d_train) %in% c("rent.amount..R.."))]
y_train <- as.numeric(d_train$rent.amount..R..)

X_val <- d_val[, !(colnames(d_val) %in% c("rent.amount..R.."))]
y_val <- as.numeric(d_val$rent.amount..R..)

#Make sure the columns are numeric before making the xgb.DMatrix that XGBoost is gonna use
for(i in 1:ncol(X_train)){
  X_train[,i] = as.numeric(X_train[,i])
}

for(i in 1:ncol(X_val)){
  X_val[,i] = as.numeric(X_val[,i])
}

xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_val <- xgb.DMatrix(data = as.matrix(X_val))

param_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 4, 5),             
  eta = c(0.01, 0.05, 0.1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,  
  subsample = 1             
)

xgb_model <- train(
  X_train, y_train, 
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = param_grid,
  metric = 'RMSE',
  verbosity = 0
)

print(xgb_model$bestTune)


### Model fitting

final_model <- xgboost(
  data = xgb_train, 
  nrounds = xgb_model$bestTune$nrounds,
  max_depth = xgb_model$bestTune$max_depth,
  eta = xgb_model$bestTune$eta,
  verbose = 0
)

### Predicting the validation set

#Errors
predictions <- predict(final_model, newdata = xgb_val)
xgb_mse <- mean((predictions - y_val)^2)
xgb_rmse <- sqrt(xgb_mse)
xgb_rsquared <- 1 - (sum((y_val - predictions)^2) / sum((y_val - mean(y_val))^2))

## Model comparison

#Data frame to store the evaluation metrics
table <- data.frame(
  Model = character(),  
  MSE = numeric(),
  RMSE = numeric(),
  R_squared = numeric()
)

#table <- rbind(table, c("Multilinear regression", round(mlr_mse, 3), round(mlr_rmse, 3), round(mlr_rsquared, 3)))
#table <- rbind(table, c("LASSO reression", round(lasso_mse, 3), round(lasso_rmse, 3), round(lasso_rsquared, 3)))
table <- rbind(table, c("GAM", round(gam_mse, 5), round(gam_rmse, 5), round(gam_rsquared, 5)))
table <- rbind(table, c("XGBoost", round(xgb_mse, 5), round(xgb_rmse, 5), round(xgb_rsquared, 5)))
colnames(table) <- c("Model", "MSE", "RMSE", "R_squared")

print(table)

# Prediction on the test set

## Setting up

#Scale ans encode training set
training_set <- scale_data(training_set, training_set)
training_set <- encode(training_set, c("animal", "furniture"))

#Splitting dependent and independent variables
X_train <- training_set[, !(colnames(training_set) %in% c("rent.amount..R.."))]
y_train <- as.numeric(training_set$rent.amount..R..)

X_test <- d_test[, !(colnames(d_test) %in% c("rent.amount..R.."))]
y_test <- as.numeric(d_test$rent.amount..R..)

#Make sure the columns are numeric before making the xgb.DMatrix that XGBoost is gonna use
for(i in 1:ncol(X_train)){
  X_train[,i] = as.numeric(X_train[,i])
}

for(i in 1:ncol(X_val)){
  X_test[,i] = as.numeric(X_test[,i])
}

xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test))


## Model fitting

final_model <- xgboost(
  data = xgb_train, 
  nrounds = xgb_model$bestTune$nrounds,
  max_depth = xgb_model$bestTune$max_depth,
  eta = xgb_model$bestTune$eta,
  verbose = 0
)


## Predicting on the test set and evaluation of the performance

predictions <- predict(final_model, newdata = xgb_test)
test_mse <- mean((predictions - y_test)^2)
test_rmse <- sqrt(xgb_mse)
test_rsquared <- 1 - (sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2))
print(round(test_mse, 5))
print(round(test_rmse, 5))
print(round(xgb_rsquared, 5))

#Feature importance scores
importance_scores <- xgb.importance(
  feature_names = colnames(X_train),
  model = final_model
)

#Plot feature importance
xgb.plot.importance(importance_matrix = importance_scores)

plot(Data$rent.amount..R.., Data$fire.insurance..R.., 
     xlab = "Rent Amount", ylab = "Fire Insurance",
     main = "Scatter Plot of Rent Amount vs Fire Insurance")

# Assuming 'Data' is your dataset and 'Data$city' contains the city names
library(ggplot2)

# Filter the dataset for each city
rio_data <- subset(Data, Data$city == "Rio de Janeiro")
sao_paulo_data <- subset(Data, Data$city == "São Paulo")
porto_alegre_data <- subset(Data, Data$city == "Porto Alegre")
campinas_data <- subset(Data, Data$city == "Campinas")
belo_horizonte_data <- subset(Data, Data$city == "Belo Horizonte")

combined_data <- rbind(rio_data, sao_paulo_data, porto_alegre_data, campinas_data, belo_horizonte_data)

# Create a scatter plot with different colors for each city
ggplot(combined_data, aes(x = rent.amount..R.., y = fire.insurance..R.., color = city)) +
  geom_point() +
  labs(title = "Rent Amount vs. Insurance by City") +
  xlab("Rent Amount") +
  ylab("Insurance")


hoa_zero_data <- subset(Data, Data$city == "Rio de Janeiro" & Data$hoa..R.. == 0 & Data$floor == 0)
hoa_nonzero_data <- subset(Data, Data$city == "Rio de Janeiro" & Data$hoa..R.. != 0)
hoa_data <- rbind(transform(hoa_zero_data, Group = "Independent houses"),
                  transform(hoa_nonzero_data, Group = "Houses in condominium"))
ggplot(hoa_data, aes(x = rent.amount..R.., y = fire.insurance..R.., color = Group)) +
  geom_point() +
  labs(title = "Rent Amount vs. Insurance in Rio de Janeiro",
       subtitle = "") +
  xlab("Rent Amount") +
  ylab("Insurance")

## Task 2 

#Scale or not?
library(factoextra)
library(dplyr)
library(cluster)

#REMOVE CATEGORICAL FEATURES
Data_sc <- Data %>% select_if(is.numeric)
Data_sc <- as.data.frame(scale(Data_sc))

dist_p <- factoextra::get_dist(Data_sc,method = "pearson")
dist_e <- factoextra::get_dist(Data_sc,method = "euclidean")

#Optimal K
set.seed(444)
sil_k <- fviz_nbclust(
  Data_sc,
  FUNcluster = kmeans,
  diss = dist_e,
  method = "silhouette",
  print.summary = TRUE,
  k.max = 10
)
 max(sil_k$data$y)
wssk <-fviz_nbclust(
  Data_sc,
  FUNcluster = kmeans,
  diss = dist_e,
  method = "wss",
  print.summary = TRUE,
  k.max = 10
)
silh <-fviz_nbclust(
  Data_sc,
  FUNcluster = factoextra::hcut,
  diss = dist_e,
  method = "silhouette",
  print.summary = TRUE,
  k.max = 10
)
wssh <- fviz_nbclust(
  Data_sc,
  FUNcluster = factoextra::hcut,
  diss = dist_e,
  method = "wss",
  print.summary = TRUE,
  k.max = 20
)
library(ggpubr)#Should we keep this?
sil_k#silhouette and elbow for kmeans

ggarrange(silh,wssh, nrow = 1,ncol = 2)#sil and elbow for HC

#Trying with 2 clusters
#kmeans
km2 = kmeans(Data_sc, 2, nstart = 1, iter.max = 1e2)
kmv2 <- factoextra::fviz_cluster(km2, data = Data_sc, geom = "point", 
                                 ggtheme = theme_minimal(), main = "K-Means")
kmv2
km2$withinss
km2$betweenss
km2$tot.withinss

kmv2
#Evaluating silhouette
library(cluster)
sk2 = silhouette(km2$cluster, 
                 dist = dist_e)
sk2v <- fviz_silhouette(sk2)
which(sil_k$data$y == max(sil_k$data$y))

hc2c <- factoextra::hcut(x = dist_e, 
                         k = 4,
                         hc_method = "ward.D2")
hc2c$silinfo$avg.width

e2c <-factoextra::fviz_cluster(list(data = Data_sc, cluster = hc2c$cluster), main = "Hierarchical",labelsize = 0)
e2c

sh = silhouette(hc2c$cluster,dist_e)
shv <- fviz_silhouette(sh)

ggarrange(e2c,shv,nrow = 2,ncol = 2)#this plots partitions and silhouettes for both

mclust::adjustedRandIndex(km2$cluster,hc2c$cluster)

#Making 2 dfs for each cluster
clust1 <- Data[which(km2$cluster == 1),]
print("Average rent in the first K-Means cluster")
mean(clust1$rent.amount..R..)#avg rent in clust1 (cheaper housing)
clust2 <- Data[which(km2$cluster == 2),]
mean(clust2$rent.amount..R..)#clust2 is more expensive

table(Data$city)#Most houses are in Sao Paulo, most frequent city in bth clusters is sao paulo
summary(clust1$city)
summary(clust2$city)
#In percentages
as.numeric(summary(clust1$city))/nrow(Data)#cluster 1 has 70% of houses
as.numeric(summary(clust2$city))/nrow(Data)


colw <- c("red","coral", "darkorange", "gold", "darkred", "firebrick")

pct1 <- as.numeric(summary(clust1$city))/nrow(clust1)
names(pct1) <- levels(clust1$city)
library(plotly)
pie1 <- plot_ly(clust1, labels = levels(clust1$city), values = pct1, type = 'pie')
pie1

pct2 <- as.numeric(summary(clust2$city))/nrow(clust2)
names(pct2) <- levels(clust2$city)
pie2 <- plot_ly(clust2, labels = levels(clust2$city), values = pct2, type = 'pie')
pie2

display_city_pie_chart <- function(Data, data_name) {
  
  # Create a dataframe with the count of each city
  city_df <- as.data.frame(table(Data$city))
  colnames(city_df) <- c("city", "count")
  
  # Compute the percentages for each city
  city_df$percentage <- city_df$count / sum(city_df$count)
  
  # Create pie chart
  ggplot(city_df, aes(x = "", y = percentage, fill = city)) +
    geom_bar(width = 1, stat = "identity") +
    coord_polar("y", start = 0) + 
    scale_fill_brewer(palette = "Set3") +
    theme_void() +
    labs(title = paste("Pie Chart of Cities in", data_name), fill = "City")
}

pie1 <- display_city_pie_chart(clust1,"Cluster1")
pie2 <- display_city_pie_chart(clust2,"Cluster2")
ggarrange(pie1,pie2,nrow = 1,ncol = 2)

?plot_#Furniture by Cluster
#summary(clust1$furniture)
#summary(clust2$furniture)

#Median number of rooms by cluster

table(clust1$rooms)
table(clust2$rooms)
#table(clust1$rooms)
#table(clust2$rooms)

#Average property tax by cluster
mean(clust1$property.tax..R..)
mean(clust2$property.tax..R..)

calculate_mode <- function(x) {
  frequency_table <- table(x)
  mode_value <- as.numeric(names(frequency_table)[which.max(frequency_table)])
  return(mode_value)
}
r1 <- calculate_mode(clust1$rooms)
r1
r2 <- calculate_mode(clust2$rooms)
r2
mean(clust1$rent.amount..R../clust1$rooms)
mean(clust2$rent.amount..R../clust2$rooms)

print(paste("Average rent per room in Cluster 1:",round(mean(clust1$rent.amount..R../clust1$rooms),3),", in Cluster 2:",round(mean(clust2$rent.amount..R../clust2$rooms),3)))
