# In this series of blog posts, I will compare different machine and deep learning methods 
# to predict clothing categories from images using the Fashion MNIST data. In this first blog 
# of the series, we will explore and prepare the data for analysis. I will also show you how 
# to predict the clothing categories of the Fashion MNIST data using my go-to model: an 
# artificial neural network. To show you how to use one of RStudios incredible features to 
# run Python from RStudio, I build my neural network in Python. In the second blog post 
# (https://github.com/fverkroost/RStudio-Blogs/blob/master/machine_learning_fashion_mnist_post2.Rmd), 
# we will experiment with tree-based methods (single tree, random forests and boosting) and 
# support vector machines to see whether we can beat the neural network in terms of performance. 

# To start, we first set our seed to make sure the results are reproducible.
set.seed(1234)

# The `keras` package contains the Fashion MNIST data, so we can easily import the data into 
# RStudio from this package directly after installing it from Github and loading it.

library(devtools)
devtools::install_github("rstudio/keras")
library(keras)        
install_keras()  
fashion_mnist <- keras::dataset_fashion_mnist()

# The resulting object named `fashion_mnist` is a nested list, consisting of lists `train` and 
# `test`. Each of these lists in turn consists of arrays `x` and `y`. To look at the dimentions
# of these elements, we recursively apply the `dim()` function to the `fashion_mnist` list.

rapply(fashion_mnist, dim)

# From the result, we observe that the `x` array in the training data contains `dim(fashion_mnist$train$x)[3]` matrices each of `r nrow(fashion_mnist$train$x)` rows and `r ncol(fashion_mnist$train$x)` columns, or in other words `r nrow(fashion_mnist$train$x)` images each of `r ncol(fashion_mnist$train$x)` by `r dim(fashion_mnist$train$x)[3]` pixels. The `y` array in the training data contains `r nrow(fashion_mnist$train$y)` labels for each of the images in the `x` array of the training data. The test data has a similar structure but only contains `r nrow(fashion_mnist$test$x)` images rather than `r nrow(fashion_mnist$train$x)`. For simplicity, we rename these lists elements to something more intuitive (where `x` now represents images and `y` represents labels):
  
c(train.images, train.labels) %<-% fashion_mnist$train
c(test.images, test.labels) %<-% fashion_mnist$test

# Every image is captured by a `r ncol(fashion_mnist$train$x)` by `r dim(fashion_mnist$train$x)[3]` 
# matrix, where entry [i, j] represents the opacity of that pixel on an integer scale from 
# `r min(fashion_mnist$train$x)` (white) to `r max(fashion_mnist$train$x)` (black). The labels 
# consist of integers between zero and nine, each representing a unique clothing category. As the 
# category names are not contained in the data itself, we have to store and add them manually. 
# Note that the categories are evenly distributed in the data.

cloth_cats = data.frame(category = c('Top', 'Trouser', 'Pullover', 'Dress', 'Coat',  
                                     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'), 
                        label = seq(0, 9))

# To get an idea of what the data entail and look like, we plot the first ten images of the test 
# data. To do so, we first need to reshape the data slightly such that it becomes compatible with 
# `ggplot2`. We select the first ten test images, convert them to data frames, rename the columns 
# into digits 1 to `r ncol(fashion_mnist$train$x)`, create a variable named `y` with digits 1 to 
# `r ncol(fashion_mnist$train$x)` and then we melt by variable `y`. We need package `reshape2` 
# to access the `melt()` function. This results in a `r ncol(fashion_mnist$train$x)` times 
# `r ncol(fashion_mnist$train$x)` equals `r ncol(fashion_mnist$train$x)*ncol(fashion_mnist$train$x)` 
# by 3 (y pixels (= y), x pixels (= variable) and the opacity (= value)) data frame. We bind 
# these all together by rows using the `rbind.fill()` function from the `plyr` package and 
# add a variable `Image`, which is a unique string repeated 
# `r ncol(fashion_mnist$train$x)*ncol(fashion_mnist$train$x)` times for each of the nine 
# images containing the image number and corresponding test set label.

library(reshape2)
library(plyr)
subarray <- apply(test.images[1:10, , ], 1, as.data.frame)
subarray <- lapply(subarray, function(df){
  colnames(df) <- seq_len(ncol(df))
  df['y'] <- seq_len(nrow(df))
  df <- melt(df, id = 'y')
  return(df)
})
plotdf <- rbind.fill(subarray)
first_ten_labels <- cloth_cats$category[match(test.labels[1:10], cloth_cats$label)]
first_ten_categories <- paste0('Image ', 1:10, ': ', first_ten_labels)
plotdf['Image'] <- factor(rep(first_ten_categories, unlist(lapply(subarray, nrow))), 
                          levels = unique(first_ten_categories))

# We then plot these first ten test images using package `ggplot2`. Note that we reverse 
# the scale of the y-axis because the original dataset contains the images upside-down. 
# We further remove the legend and axis labels and change the tick labels.
library(ggplot2)
ggplot() + 
  geom_raster(data = plotdf, aes(x = variable, y = y, fill = value)) + 
  facet_wrap(~ Image, nrow = 2, ncol = 5) + 
  scale_fill_gradient(low = "white", high = "black", na.value = NA) + 
  theme(aspect.ratio = 1, legend.position = "none") + 
  labs(x = NULL, y = NULL) + 
  scale_x_discrete(breaks = seq(0, 28, 7), expand = c(0, 0)) + 
  scale_y_reverse(breaks = seq(0, 28, 7), expand = c(0, 0))

# Next, it's time to start the more technical work of predicting the labels from the image data. 
# First, we need to reshape our data to convert it from a multidimensional array into a two-dimensional 
# matrix. To do so, we vectorize each `r ncol(fashion_mnist$train$x)` by `r ncol(fashion_mnist$train$x)` 
# matrix into a column of length `r ncol(fashion_mnist$train$x)*ncol(fashion_mnist$train$x)`, and then 
# we bind the columns for all images on top of each other, finally taking the transpose of the resulting 
# matrix. This way, we can convert a `r ncol(fashion_mnist$train$x)` by `r ncol(fashion_mnist$train$x)` 
# by `r nrow(fashion_mnist$train$x)` array into a `r nrow(fashion_mnist$train$x)` by 
# `r ncol(fashion_mnist$train$x)*ncol(fashion_mnist$train$x)` matrix. We also normalize the data by 
# dividing between the maximum opacity of `r max(fashion_mnist$train$x)`.

train.images <- data.frame(t(apply(train.images, 1, c))) / max(fashion_mnist$train$x)
test.images <- data.frame(t(apply(test.images, 1, c))) / max(fashion_mnist$train$x)

# We also create two data frames that include all training and test data (images and labels), respectively.

pixs <- 1:ncol(fashion_mnist$train$x)^2
names(train.images) <- names(test.images) <- paste0('pixel', pixs)
train.labels <- data.frame(label = factor(train.labels))
test.labels <- data.frame(label = factor(test.labels))
train.data <- cbind(train.labels, train.images)
test.data <- cbind(test.labels, test.images)

# Now, let's continue by building a simple neural network model to predict our clothing categories. 
# Neural networks are artificial computing systems that were built with human neural networks in mind. 
# Neural networks contain nodes, which transmit signals amongst one another. Usually the input at each 
# node is a number, which is transformed according to a non-linear function of the input and weights, 
# the latter being the parameters that are tuned while training the model. Sets of neurons are collected 
# in different layers; neural networks are reffered to as 'deep' when they contain at least two hidden 
# layers. If you're not familiar with artificial neural networks, then [this](www.google.com) is a good source to 
# start learning about them.

# In this post, I will show you how artificial neural networks with different numbers of hidden layers 
# compare, and I will also compare these networks to a convolutional network, which often performs 
# better in the case of visual imagery. I will show you some basic models and how to code these, but 
# will not spend too much time on tuning neural networks, for example when it comes to choosing the 
# right amount of hidden layers or the number of nodes in each hidden layer. In essence, what it comes 
# down to is that these parameters largely depend on your data structure, magnitude and complexity. 
# The more hidden layers one adds, the more complex non-linear relationships can be modelled. Often, 
# in my experience, adding hidden layers to a neural network increases their performance up to a certain 
# number of layers, after which the increase becomes non-significant while the computational requirements 
# and interpretation become more infeasible. It is up to you to play around a bit with your specific data 
# and test how this trade-off works. For a more detailed explanation and framework for tuning neural networks, 
# I refer you to [this source](www.google.com).


# Although neural networks can easily built in RStudio using Tensorflow and Keras, I really want to show 
# you one of the incredible features of RStudio where you can run Python in RStudio. This can be done in 
# two ways: either we choose "Terminal" on the top of the output console in RStudio and run Python via 
# Terminal, or we use the base `system2()` function to run Python in RStudio. 

# For the second option, to use the `system2()` command, it's important to first check what version of 
# Python should be used. You can check which versions of Python are installed on your machine by running 
# `python --version` in Terminal. Note that with RStudio 1.1 (1.1.383 or higher), you can run in Terminal 
# directly from RStudio on the "Terminal" tab. You can also run `python3 --version` to check if you have 
# Python version 3 installed. On my machine, `python --version` and `python3 --version` return Python 2.7.16 
# and Python 3.7.0, respectively. You can then run `which python` (or `which python3` if you have Python 
# version 3 installed) in Terminal, which will return the path where Python is installed. In my case, 
# these respective commands return `/usr/bin/python` and `/Library/Frameworks/Python.framework/Versions/3.7/bin/python3`. 
# As I will make use of Python version 3, I specify the latter as the path to Python in the `use_python()` 
# function from the `reticulate` package. We can check whether the desired version of Python is used by 
# using the `sys` package from Python. Just make sure to change the path in the code below to what 
# version of Python you desire using and where that version in installed.

library(reticulate)
use_python(python = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3')
sys <- import("sys")
sys$version

# Now that we've specified the correct version of Python to be used, we can run our Python script from 
# RStudio using the `system2()` function. This function also takes an argument for the version of Python 
# used, which in my case is Python version 3. If you are using an older version of Python, make sure to 
# change `"python3"` in the command below to `"python2"`.

python_file <- "simple_neural_network_fashion_mnist.py"
system2("python3", args = c(python_file), stdout = NULL, stderr = "")