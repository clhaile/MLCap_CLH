###
# Movielens code
###

## initial project code supplied by EdX

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

#set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
#Note I ran this code on R 3.5.1
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##beginning my code

##preamble items
## my typical libraries used

library(tidyverse)
library(dslabs)
library(caret)
library(dslabs)
library(tidyverse)
library(pdftools)
library(lubridate)
library(matrixStats)
library(data.table)


## PREPROCESSING AND CLEANING

#get a view of the edx data set structure
head(edx)

#check for missing (NA) values in edx and validation

edx %>% map_df(~sum(is.na(.))) %>% knitr::kable()
validation %>% map_df(~sum(is.na(.))) %>% knitr::kable()

## Preprocessing on the date of the rating (timestamp)
## Alter timestamp to more understandable date
## Reduce to year only to avoid too much granularity (yr_rate)

edx <- mutate(edx, date = as_datetime(timestamp))
edx$yr_rate <- format(edx$date,"%Y")
edx$yr_rate <-as.numeric(edx$yr_rate)

validation <- mutate(validation, date = as_datetime(timestamp))
validation$yr_rate <- format(validation$date,"%Y")
validation$yr_rate <-as.numeric(validation$yr_rate)

##Preprocessing on genres
## Multiple genres have a great deal of overlap, to much processing to split out into
## separate variables for each possible genre.  Take the first listed genre (alphabetical)
## which although crude may give some differentiation in ratings
## save variable genre_single

edx$genre_single<-sub("\\|.*", "", edx$genres)
validation$genre_single<-sub("\\|.*", "", validation$genres)

##Preprocessing on the year of release (extracted from title)
## save variable yr_release

edx$yr_release<-gsub(".*\\((.*)\\).*", "\\1", edx$title)
edx$yr_release <-as.numeric(edx$yr_release)

validation$yr_release<-gsub(".*\\((.*)\\).*", "\\1", validation$title)
validation$yr_release <-as.numeric(validation$yr_release)

edx <- edx %>% select(userId, movieId, rating, yr_rate,genre_single,yr_release)
validation <- validation %>% select(userId, movieId, rating, yr_rate,genre_single,yr_release)

### END PREPROCESSING AND CLEANING

##count the number of ratings given in edx
edx %>% group_by(rating) %>% tally()

### BASIC DESCRIPTIVES

## count the number of unique movie titles
n_distinct(edx$movieId)
##count the number of movie reviewers
n_distinct(edx$userId)

## table of number of movies and raters
edx %>% summarize(movies=n_distinct(edx$movieId),raters=n_distinct(edx$userId)) %>% knitr::kable()

## Bar chart of ratings count
edx %>%ggplot(aes(rating)) +geom_bar()+ggtitle("Count of each type of movie rating")

## histogram of average rating per movie
edx %>% group_by(movieId) %>% summarise(mean=mean(rating)) %>% ggplot(aes(mean)) +geom_histogram(binwidth=.5)

### VISUALIZATION OF THE EFFECT OF DIFFERENT PREDICTORS ON MEAN MOVIE RATING

# movieId effect
pl<-edx %>% group_by(movieId) %>% summarise(mean=mean(rating)) 
ggplot(pl,aes(x=movieId,y=mean)) + geom_bar(stat='identity')

# userId effect

pl<-edx %>% group_by(userId) %>% summarise(mean=mean(rating)) 
ggplot(pl,aes(x=userId,y=mean)) + geom_bar(stat='identity')

# Year of rating effect

pl<-edx %>% group_by(yr_rate) %>% summarise(mean=mean(rating)) 
ggplot(pl,aes(x=yr_rate,y=mean)) + geom_bar(stat='identity')


# Year of release effect

pl<-edx %>% group_by(yr_release) %>% summarise(mean=mean(rating)) 
ggplot(pl,aes(x=yr_release,y=mean)) + geom_bar(stat='identity')

# First listed genre effect

pl<-edx %>% group_by(genre_single) %>% summarise(mean=mean(rating)) 
ggplot(pl,aes(x=genre_single,y=mean)) + geom_bar(stat='identity')


##Movie Rating predictions
## RMSE (error) function

RMSE <- function(true_ratings,predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2))}

## Naive movie rating prediction using only the average movie rating
mu<-mean(edx$rating)
naive_rmse<-RMSE(validation$rating,mu)
rmse_results<-data_frame(method="Naive Model",RMSE=naive_rmse)

## Add "movieId" as a predictive variable (b_i)
movie_avgs<-edx %>% group_by(movieId) %>% summarize(b_i=mean(rating-mu))
predicted_ratings<-mu+validation %>% left_join(movie_avgs,by='movieId') %>% .$b_i
model_1_rmse<-RMSE(predicted_ratings,validation$rating)
rmse_results<-bind_rows(rmse_results,data_frame(method="Movie Effect Model",
                                                RMSE = model_1_rmse))
rmse_results %>% knitr::kable()

## Add "userId" to model as predictive variable (b_u)
user_avgs<-edx %>% left_join(movie_avgs,
                             by='movieId') %>% group_by(userId) %>% summarize(b_u=mean(rating-mu-b_i))

predicted_ratings<-validation %>% left_join(movie_avgs,by='movieId') %>%
  left_join(user_avgs,by='userId') %>%
  mutate(pred=mu+b_i+b_u) %>% .$pred

model_2_rmse <- RMSE(predicted_ratings,validation$rating)
rmse_results<-bind_rows(rmse_results,
                        data_frame(method="Movie +User effects model",RMSE=model_2_rmse))
rmse_results %>% knitr::kable()


## Add "yr_release", "genre_single", and "yr_rate" to model as 
## predictive variables (b_r, b_g, b_w)

mu<-mean(edx$rating)
b_i<-edx %>% group_by(movieId) %>%
  summarize(b_i=sum(rating-mu)/(n()))
b_u<-edx %>% 
  left_join(b_i,by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u=sum(rating-b_i-mu)/(n()))
b_r<-edx %>% 
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  group_by(yr_release) %>%
  summarize(b_r=sum(rating-b_i-b_u-mu)/(n()))
b_g<-edx %>% 
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  left_join(b_r,by="yr_release") %>%
  group_by(genre_single) %>%
  summarize(b_g=sum(rating-b_i-b_u-b_r-mu)/(n()))
b_w<-edx %>% 
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  left_join(b_r,by="yr_release") %>%
  left_join(b_g,by="genre_single") %>%
  group_by(yr_rate) %>%
  summarize(b_w=sum(rating-b_i-b_u-b_r-b_g-mu)/(n()))
predicted_ratings<-validation %>%
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  left_join(b_r,by="yr_release") %>%
  left_join(b_g,by="genre_single") %>%
  left_join(b_w,by="yr_rate") %>%
  mutate(pred=mu+b_i+b_u+b_r+b_g+b_w) %>% .$pred

model_3_rmse <- RMSE(predicted_ratings,validation$rating)
rmse_results<-bind_rows(rmse_results,
                        data_frame(method="Movie+User+ReleaseYear+Genre+RateYear model",RMSE=model_3_rmse))
rmse_results %>% knitr::kable()

## Add Regularization to penalize effect of small number of ratings
## penalty = lambda
## Run over several values of lambda to see which minimizes RMSE

lambdas<-seq(0,10,.5)

rmses<-sapply(lambdas,function(l){
  mu<-mean(edx$rating)
  b_i<-edx %>% group_by(movieId) %>%
    summarize(b_i=sum(rating-mu)/(n()+l))
  b_u<-edx %>% 
    left_join(b_i,by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u=sum(rating-b_i-mu)/(n()+l))
  b_r<-edx %>% 
    left_join(b_i,by="movieId") %>%
    left_join(b_u,by="userId") %>%
    group_by(yr_release) %>%
    summarize(b_r=sum(rating-b_i-b_u-mu)/(n()+l))
  b_g<-edx %>% 
    left_join(b_i,by="movieId") %>%
    left_join(b_u,by="userId") %>%
    left_join(b_r,by="yr_release") %>%
    group_by(genre_single) %>%
    summarize(b_g=sum(rating-b_i-b_u-b_r-mu)/(n()+l))
  b_w<-edx %>% 
    left_join(b_i,by="movieId") %>%
    left_join(b_u,by="userId") %>%
    left_join(b_r,by="yr_release") %>%
    left_join(b_g,by="genre_single") %>%
    group_by(yr_rate) %>%
    summarize(b_w=sum(rating-b_i-b_u-b_r-b_g-mu)/(n()+l))
  predicted_ratings<-validation %>%
    left_join(b_i,by="movieId") %>%
    left_join(b_u,by="userId") %>%
    left_join(b_r,by="yr_release") %>%
    left_join(b_g,by="genre_single") %>%
    left_join(b_w,by="yr_rate") %>%
    mutate(pred=mu+b_i+b_u+b_r+b_g+b_w) %>% .$pred
  return(RMSE(predicted_ratings,validation$rating))})


## Plot of lambda vs RMSE
plot(lambdas,rmses)

## Choose optimal lambda

lambda<-lambdas[which.min(rmses)]


## Re-run the model with the lambda minimizing RMSE

mu<-mean(edx$rating)
b_i<-edx %>% group_by(movieId) %>%
  summarize(b_i=sum(rating-mu)/(n()+lambda))
b_u<-edx %>% 
  left_join(b_i,by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u=sum(rating-b_i-mu)/(n()+lambda))
b_r<-edx %>% 
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  group_by(yr_release) %>%
  summarize(b_r=sum(rating-b_i-b_u-mu)/(n()+lambda))
b_g<-edx %>% 
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  left_join(b_r,by="yr_release") %>%
  group_by(genre_single) %>%
  summarize(b_g=sum(rating-b_i-b_u-b_r-mu)/(n()+lambda))
b_w<-edx %>% 
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  left_join(b_r,by="yr_release") %>%
  left_join(b_g,by="genre_single") %>%
  group_by(yr_rate) %>%
  summarize(b_w=sum(rating-b_i-b_u-b_r-b_g-mu)/(n()+lambda))
predicted_ratings<-validation %>%
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  left_join(b_r,by="yr_release") %>%
  left_join(b_g,by="genre_single") %>%
  left_join(b_w,by="yr_rate") %>%
  mutate(pred=mu+b_i+b_u+b_r+b_g+b_w) %>% .$pred


model_4_rmse <- RMSE(predicted_ratings,validation$rating)
rmse_results<-bind_rows(rmse_results,
                        data_frame(method="Regularized Movie+User+ReleaseYear+Genre+RateYear model",RMSE=model_4_rmse))
rmse_results %>% knitr::kable()

