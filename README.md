An easy to use neural networks tool for learning tree-like feedforward architectures with support for dense and sparse inputs.

##Usage

Assume that we want to classify pairs of texts into matching and not matching. This problem may arise in many contexts, for example the first text may be a query, and the second - text of a related or a nonrelated document. In this case we may come up with the following solution. First, we convert the texts to sparse vectors (build a bag of words or trigrams representation). Then, create the learn and validation files (don't forget to shuffle the lines!) in the following format:

1 \t 1 \t 1:1 5:4 6:4 \t 1:4 2:1 5:3

0 \t 1 \t 4:1 1:4 10:4 \t 12:4.1 2:3.1 5:1.5

1 \t 1 \t 1:1 3:4 1:4 \t 3:2 2:1.4 19:1

Each line contains tab-separated columns. The first two columns are target and weight correspondingly. They may be followed by any number of columns. Each column contains space-separated values that represent different inputs to the neural network. In this example we have two sparse inputs obtained from the text.

Now, we can start training a neural network. The tool provides an easy way to specify the architecture of the neural network directly in the command line. For example, we may run the following command:

python unn.py learn -m model.net -f train -t val -i query:sparse:700000,doc:sparse:700000 -a 'query_embedding=linear(rlu(rlu(query,100),300),300)|doc_embedding=linear(rlu(rlu(doc,100),300),300)|output=sigmoid(rlu(concat(query_embedding,doc_embedding),300),1) --loss cross_entropy

Here we specify where to save the model (file 'model.net') - it is where the best model will be occasionally saved, train and validation files, what inputs the train file provide (the first input is a sparse vector 'query' of length 700000, the second is a sparse vector 'doc' of length 700000), loss function (cross entropy - good for classification) and the architecture of the neural network.

A neural network consists of several submodules separated by '|' in the specification. The string 'query_embedding=linear(rlu(rlu(query,100),300), 300)' means that a variable 'query_embedding' is created that is a function of input variable 'query', obtained by applying the affine transformations of size 100 and 300, and rlu nonlinearities. The same is done for 'doc_embedding' as specified in string 'doc_embedding=linear(rlu(rlu(doc,100),300),300)'. The last submodule is always the default output of the neural network. The string 'output=sigmoid(dot(left_embedding, right_embedding))' means that we concatenate query_embedding and doc_embedding, apply affine transformation with 300 outputs, apply rlu nonlinearity, then again affine nonlinearity of size 1, and finally take a sigmoid. Thus, the output is a single float named 'output' in the range 0-1, associated with the probability of a match between the texts.

When we want to apply a trained neural network we can do the following:

python unn.py apply -m model/model.net -o output -d val -i nothing1,nothing2,query,doc -f val.res

Here -o is which variable to output (it can be any named variable, for example 'doc_embedding'), -d - where the file with data is, -i - describes the columns of the input file (the columns may be arbitrary, the model will look only at the columns with the names required for computations), -f specifies where to save the result. If you have a model, but don't know the names of its inputs, or its named variables, run 'unn.py describe' to find this out.

It is possible to create a function with shared parameters. For example, the previous example could look as follows:

python unn.py learn -m model.net -f train -t val -i query:sparse:700000,doc:sparse:700000 -a 'embed(input:sparse:700000)=linear(rlu(rlu(input,100),300),300)|query_embedding=embed(query)|doc_embedding=embed(doc)|output=sigmoid(rlu(concat(query_embedding, doc_embedding),300),1) --loss cross_entropy

Here function 'embed' is applied to query and doc using the same parameters.

Currently the tool supports 2 loss functions for learning: cross_entropy and mse. You may specify a different loss function for validation (for example, learn using cross entropy, validate using accuracy). The main learning algorithm is sgd. In the learn mode the inputs can be specified in several formats. The full format is name:type:num_features@preprocessor. Preprocessor is valid only for dense inputs and may be 'none', 'scale', 'pca', and "minmax". If you don't know what to choose, use 'scale' (this is the default option when @preprocessor is omitted). When the type is dense, you can actually drop the 'type' and 'num_features' field, they will be inferred from the data. It is not possible to find the number of features for sparse inputs from data, therefore a full format should be used. When shared functions are defined, the input is always specified as name:type:num_features.

All options are described in the source code, or can be found by calling unn.py -h, unn.py learn -h, unn.py apply -h. Some important options for unn.py learn are:
- -a - architecture of the network
- -i - specification of the inputs
- --loss - train loss (usually cross-entropy for classification and mse for regression). The library also supports pairwise learning - it is easy to construct the required architecture in -a.
- --vloss - validation loss (cross-entropy, mse, or accuracy) - if not specified, has the same value as --loss
- --mf, --mt - keep train, validation in memory (faster learning if they are not too big)
- --lr - learning rate - a very important parameter of sgd learning. The default value of 0.1 may work or not work for you. Small values of sgd lead to too slow progress, large values may make the network never converge to a good state. It is a good practice to try several values (like 0.1, 0.3, 0.03, 0.01, 0.003, etc.) and wait several thousand batches to see which works better. During training you may want to decrease learning rate (when the network seems to make no further progress). In this case send keyboard interrupt (press ctrl-c) to half the learning rate.
- --momentum - another hyperparameter of sgd. The default value 0.9 is mostly OK, but you may want to tune it somehow (for example try 0.99, 0.95). The larger momentum, the smaller learning rate should be.
- -l - l2 regularization - when you see that a network overfits (validation error goes up, train error goes down), you can get more data, or try to set this value. The larger it is, the less the network will overfit. However, it can cause underfitting (train error is high, validation error is high), so better keep it close to zero.

Some of the implemented operations are (see the source code for more information):
- rlu(input_variable[, num_outputs]) - returns input_variable > 0 ? input_variable : 0. If num_outputs is specified, an affine transformation is applied before the nonlinearity
- sigmoid(input_variable[, num_outputs]) - returns sigmoid(input_variable). If num_outputs is specified, an affine transformation is applied before the nonlinearity
- tanh(input_variable[, num_outputs]) - returns tanh(input_variable). If num_outputs is specified, an affine transformation is applied before the nonlinearity
- linear(input_variable, num_outputs) - apply an affine transformation
- concat(input_variable1, input_varable2, ...) - concatenates the inputs
- dot(input_variable1, input_varable2) - returns sum(input_variable1 * input_variable2, axis=1)
- scale(input_variable, const) - returns const * input_variable
- sub(input_variable1, input_variable2) - returns input_variable1 - input_variable2

The code is rather short and most mathematical issues are resolved with theano, thus it is very easy to introduce a new operation if it is missing.

##Installation
Currently it is just one file, so you can just copy it and put wherever you want. It requires theano, numpy and scipy. It is highly recommended that the libraries were configured to use optimized blas libraries, like openblas or mkl. Otherwise you may get a 10x slowdown.
