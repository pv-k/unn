"""Copyright (c) 2015, Pavel Kalinin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

import argparse
import sys
import pickle
import os.path
import math

import numpy
import scipy.sparse
import scipy.stats
import shutil
import time
import collections
import tempfile
import copy
import bisect

import theano
import theano.sparse
import theano.tensor as T
import theano.tensor.nnet
from theano.compile.ops import as_op
import sklearn.preprocessing

class unique_tempdir(object):
    def __enter__(self):
        if not (os.path.isdir("./tmp")):
            os.makedirs("./tmp")
        self.dir = tempfile.mkdtemp(dir="./tmp/")
        return self.dir
    def __exit__(self, etype, value, traceback):
        shutil.rmtree(self.dir)

#------------------------------------------learning--------------------

@as_op(itypes=[T.ivector],
       otypes=[T.ivector])
def uniq(a):
    return numpy.unique(a)

def sgd_update(energy, params, learning_rates, momentums):
    '''
    Ordinary sgd with momentum. (not Nesterov momentum :( )
    energy - what to minimize (theano symbolic),
    params - minimize with respect to these params (theano symbolic)
    learning_rates - learning rate per each parameters (theano symbolic)
    momentums - momentum per each parameter (theano symbolic)
    '''

    assert len(params) == len(learning_rates)
    assert len(momentums) == len(learning_rates)

    all_params = []
    for param in params:
        if isinstance(param, SparseDotParams):
            all_params.append(param.compactW)
        else:
            all_params.append(param)
    updates = []
    grads = T.grad(energy, all_params)

    sparse_params_data = collections.defaultdict(list)
    for param, grad, lr, momentum in zip(params, grads, learning_rates, momentums):
        if isinstance(param, SparseDotParams):
            sparse_params_data[param.W].append((lr, momentum, grad, param))
        else:
            if momentum is not None:
                param_momentum = theano.shared(numpy.zeros_like(param.get_value()),
                    broadcastable=param.broadcastable)
                updated_param_momentum = momentum * param_momentum + \
                    (numpy.float32(1.0) - momentum) * grad
                updates.append((param, param - lr * updated_param_momentum))
                updates.append((param_momentum, updated_param_momentum))
            else:
                updates.append((param, param - lr * grad))

    for param_W in sparse_params_data:
        param_data = sparse_params_data[param_W]

        momentum = None
        lr = None
        for lr_, momentum_, grad, param in param_data:
            if momentum is None:
                momentum = momentum_
                lr = lr_
        for lr_, momentum_, grad, param in param_data:
            if momentum is not None and (momentum_ is None or momentum != momentum_):
                raise Exception("Different momentums for the same "
                    "parameter are not allowed")
            if lr is not None and (lr_ is None or lr != lr_):
                raise Exception("Different learning rates for the same "
                    "parameter are not allowed")

        if momentum is not None:
            param_momentum = theano.shared(numpy.zeros_like(param_W.get_value()),
                    broadcastable=param_W.broadcastable)

            updated_indices_vars = []
            for lr, momentum, grad, param in param_data:
                updated_indices_vars.append(param.nonzeroes)
            updated_indices = uniq(T.concatenate(updated_indices_vars))

            # decay momentum and add updates
            updated_param_momentum = T.set_subtensor(param_momentum[updated_indices],
                param_momentum[updated_indices] * momentum)
            for lr, momentum, grad, param in param_data:
                updated_param_momentum = T.inc_subtensor(
                    updated_param_momentum[param.nonzeroes],
                    (numpy.float32(1.0) - momentum) * grad)
            updates.append((param_momentum, updated_param_momentum))

            new_value = T.inc_subtensor(param_W[updated_indices],
                -lr * updated_param_momentum[updated_indices])
            updates.append((param_W, new_value))
        else:
            new_value = param_W
            for lr, momentum, grad, param in param_data:
                new_value = T.inc_subtensor(new_value[param.nonzeroes],
                    -lr * grad).get_output()
            updates.append((param_W, new_value))
    return updates

def ravel(flat_list, deep_list):
    def sub_ravel(flat_iter, deep_list):
        res = []
        for item in deep_list:
            if isinstance(item, (list, tuple)):
                res.append(sub_ravel(flat_iter, item))
            else:
                res.append(flat_iter.next())
        return res
    return sub_ravel(flat_list.__iter__(), deep_list)
def flatten(deep_list):
    res = []
    for item in deep_list:
        if isinstance(item, (list, tuple)):
            res.extend(flatten(item))
        else:
            res.append(item)
    return res

class Function(object):
    def __init__(self, input_transformers, theano_inputs, outputs, updates=None):
        self.ordered_names = [var_name for var_name in theano_inputs]
        ordered_theano_inputs = [theano_inputs[var_name] for var_name in self.ordered_names]
        self.func = theano.function(inputs=ordered_theano_inputs,
            outputs=outputs, updates=updates)
        self.transformers_map = input_transformers
    def __call__(self, inputs_map):
        for var_name, var_data in inputs_map.items():
            if var_name in self.transformers_map:
                transformer = self.transformers_map[var_name]
                inputs_map[var_name] = transformer.transform(var_data)
        inputs = [inputs_map[var_name] for var_name in self.ordered_names]
        return self.func(*inputs)

class Trainer(object):
    def __init__(self, learning_rate, momentum, num_epochs, reg_lambda, batch_size,
            validation_frequency, loss_function):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.validation_frequency = validation_frequency
        self.loss_function = loss_function
    def fit(self, model, train_energy, validation_energy, save_path,
            train_dataset, validation_dataset, continue_learning):
        input_vars = {var.name: var for var in flatten(train_energy.get_inputs())}
        for var in flatten(validation_energy.get_inputs()):
            assert var.name in input_vars
        transformers_map = {var.name: var.transformer for var in input_vars.values() if
            var.transformer is not None}
        input_theano_vars = {}
        for var in input_vars.values():
            if var.type == "dense":
                input_theano_vars[var.name] = T.matrix(name=var.name, dtype="float32")
            elif var.type == "sparse":
                input_theano_vars[var.name] = theano.sparse.csr_matrix(
                    name=var.name, dtype='float32')
            else:
                assert False

        train_inputs = ravel([input_theano_vars[var.name] for var in
            flatten(train_energy.get_inputs())], train_energy.get_inputs())
        train_loss, dynamic_params, dynamic_penalized_params = train_energy.train_fprop(train_inputs)
        params = train_energy.get_params() + dynamic_params
        penalized_params = train_energy.get_penalized_params() + dynamic_penalized_params
        assert len(params) == len(set(params))
        assert len(penalized_params) == len(set(penalized_params))
        print "Number of static parameters: " + str(len(train_energy.get_params()))
        print "Number of dynamic parameters: " + str(len(dynamic_params))
        print "Number of static penalized parameters: " + str(len(train_energy.get_penalized_params()))
        print "Number of dynamic penalized parameters: " + str(len(dynamic_penalized_params))

        lr = theano.shared(numpy.float32(self.learning_rate))
        if self.momentum is not None:
            momentum = theano.shared(numpy.float32(self.momentum))
        else:
            momentum = None
        learning_rates = [lr] * len(params)
        momentums = [momentum] * len(params)

        l2_penalty = 0
        if self.reg_lambda != 0:
            for param in penalized_params:
                l2_penalty = l2_penalty + T.sum(T.sqr(param))
        l2_penalty = l2_penalty * numpy.float32(self.reg_lambda)
        train_loss = train_loss + l2_penalty

        assert "weights" in input_vars
        sum_weights = T.sum(abs(input_theano_vars["weights"]))

        train = Function(
            input_transformers=transformers_map,
            theano_inputs=input_theano_vars,
            outputs=[train_loss, sum_weights],
            updates=sgd_update(
                energy=train_loss,
                params=params,
                learning_rates=learning_rates,
                momentums=momentums))
        score_train = Function(input_transformers=transformers_map,
            theano_inputs=input_theano_vars,
            outputs=[train_energy.fprop(train_inputs) + l2_penalty, sum_weights])

        if validation_energy is not None:
            validation_inputs = {var.name: input_theano_vars[var.name] for var in
                flatten(validation_energy.get_inputs())}
            flat_validation_inputs = [input_theano_vars[var.name] for var in
                flatten(validation_energy.get_inputs())]
            validation_loss = validation_energy.fprop(
                ravel(flat_validation_inputs, validation_energy.get_inputs()))
            score_validation = Function(input_transformers=transformers_map,
                theano_inputs=validation_inputs,
                outputs=[validation_loss, sum_weights])

        if not continue_learning:
            have_transformers = False
            for var_name, var in input_vars.items():
                if var.transformer is not None:
                    have_transformers = True
                    break

            if have_transformers:
                print "Fitting transformers"
                for data_map in train_dataset.read(1000000):
                    for var_name, var in input_vars.items():
                        if var.transformer is not None:
                            var.transformer.fit(data_map[var.name])
                    break
            else:
                print "Input data is used as is"

        print "Estimating initial performance"
        # estimate training score
        train_score = 0
        weights_total = 0
        for batch_idx, data_map in enumerate(train_dataset.read(10000)):
            batch_score, suw_weights = score_train(data_map)
            weights_total += suw_weights
            train_score += suw_weights * batch_score
            if batch_idx > 10:
                break
        train_score *= 1.0 / weights_total

        if validation_energy is not None:
            # compute validation score
            best_validation_score = 0
            weights_total = 0
            for batch_idx, data_map in enumerate(validation_dataset.read(10000)):
                batch_score, suw_weights = score_validation(data_map)
                weights_total += suw_weights
                best_validation_score += suw_weights * batch_score
            best_validation_score *= 1.0 / weights_total

            print "Initital validation score: {}".format(best_validation_score)

        print "Learning"
        # training
        train_decay = 0.999
        batch_idx = 0
        epoch_ind = 0
        interrupts_budget = 10
        data_iter = train_dataset.read_train(self.batch_size)
        while True:
            try:
                while epoch_ind < self.num_epochs:
                    start_time = time.time()
                    for data_map in data_iter:
                        batch_score = train(data_map)[0]
                        train_score = train_decay * train_score + (1 - train_decay) * batch_score

                        if (batch_idx + 1) % 100 == 0:
                            elapsed = time.time() - start_time
                            sys.stdout.write("{}: Train score: {}; Elapsed: {}\n".format(
                                (batch_idx + 1), train_score, elapsed) )
                            start_time = time.time()
                        if (batch_idx + 1) % self.validation_frequency == 0:
                            if validation_energy is not None:
                                validation_score = 0
                                weights_total = 0
                                for data_map in validation_dataset.read(10000):
                                    batch_score, suw_weights = score_validation(data_map)
                                    weights_total += suw_weights
                                    validation_score += suw_weights * batch_score
                                validation_score *= 1.0 / weights_total
                                if validation_score < best_validation_score:
                                    best_validation_score = validation_score
                                    with unique_tempdir() as tmpdir:
                                        tmp_model_path = os.path.join(tmpdir, "model.net")
                                        with open(tmp_model_path, "w") as f:
                                            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
                                        os.rename(tmp_model_path, save_path)
                                sys.stdout.write("Epoch {}: validation score: {}; best validation score: {}\n".format(
                                    epoch_ind, validation_score, best_validation_score))
                            else:
                                with unique_tempdir() as tmpdir:
                                    tmp_model_path = os.path.join(tmpdir, "model.net")
                                    with open(tmp_model_path, "w") as f:
                                        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
                                    os.rename(tmp_model_path, save_path)
                                sys.stdout.write("Model saved\n".format(
                                    epoch_ind, validation_score, best_validation_score))

                        batch_idx += 1
                    data_iter = train_dataset.read_train(self.batch_size)
                    epoch_ind += 1
                break
            except KeyboardInterrupt:
                if interrupts_budget == 0:
                    raise
                interrupts_budget -= 1
                lr.set_value(lr.get_value() / numpy.float32(2.0))
                print "New learning rate set {}. Interrupts remaining: {}".format(lr.get_value(), interrupts_budget)

# ------------------------------apply---------------------------------------

def apply_model(model, dataset, output_stream):
    input_vars = {var.name: var for var in flatten(model.get_inputs())}
    transformers_map = {var.name: var.transformer for var in input_vars.values() if
        var.transformer is not None}
    input_theano_vars = {}
    for var in input_vars.values():
        if var.type == "dense":
            input_theano_vars[var.name] = T.matrix(name=var.name, dtype="float32")
        elif var.type == "sparse":
            input_theano_vars[var.name] = theano.sparse.csr_matrix(
                name=var.name, dtype='float32')
        else:
            assert False

    inputs = ravel([input_theano_vars[var.name] for var in
        flatten(model.get_inputs())], model.get_inputs())
    model_output = model.fprop(inputs)

    eval_data = Function(
        input_transformers=transformers_map,
        theano_inputs=input_theano_vars,
        outputs=[model_output])

    for batch_idx, data_map in enumerate(dataset.read(10000)):
        output = eval_data(data_map)[0]
        output_stream.write("\n".join(("\t".join(str(val) for val in vals)) for vals in output) + "\n")

# -----------------------------------modules-----------------------
class Variable(object):
    def __init__(self, type, num_features=None, producer=None, name=None,
            transformer=None):
        self.producer = producer
        self.name = name
        self.type = type
        self.num_features = num_features
        self.transformer = transformer

class Module(object):
    def __init__(self, input_vars, output_var):
        self.input_vars = input_vars
        self.output = output_var
        self.output.producer = self
    def train_fprop(self, inputs):
        raise NotImplementedError()
    def fprop(self, inputs):
        return self.train_fprop(inputs)[0]
    def get_params(self):
        return []
    def get_penalized_params(self):
        return []
    def get_inputs(self):
        return self.input_vars
    def get_output(self):
        return self.output

# efficient sparse dot for affine module
@as_op(itypes=[T.ivector],
       otypes=[T.ivector])
def range_len(e):
    return numpy.asarray(range(len(e)), dtype=numpy.int32)
@as_op(itypes=[T.fmatrix, T.fmatrix, T.ivector],
       otypes=[T.fmatrix])
def subtensor_add(X, update, inds):
    X[inds] += update
    return X
class SparseDotParams(object):
    def __init__(self, out, W, compactW, nonzeroes):
        self.out = out
        self.compactW = compactW
        self.nonzeroes = nonzeroes
        self.W = W
def sparse_dot(X, W, W_numrows=None):
    assert isinstance(X.type, theano.sparse.type.SparseType)
    assert X.format == "csr"
    columns_map = theano.shared(numpy.zeros(W_numrows,
        dtype=numpy.int32), name="columns_map")
    (data, indices, indptr, shape) = theano.sparse.basic.csm_properties(X)
    nonzeroes = uniq(indices)
    columns_map = T.set_subtensor(columns_map[nonzeroes], range_len(nonzeroes))
    new_indices = columns_map[indices]
    new_shape = T.set_subtensor(shape[1], nonzeroes.shape[0])
    newX = theano.sparse.basic.CSR(data, new_indices, indptr, new_shape)
    compactW = W[nonzeroes]
    res = theano.sparse.structured_dot(newX, compactW)
    return SparseDotParams(res, W, compactW, nonzeroes)

class AffineModule(Module):
    def __init__(self, rng, input_var, num_outputs, output_activation_name=None):
        Module.__init__(self, [input_var], Variable("dense", num_outputs, self))

        assert input_var.num_features is not None
        assert input_var.type is not None

        self.num_outputs = num_outputs
        self.input_var = input_var
        self.output_activation_name = output_activation_name

        expected_input_size = 50 if input_var.type == "sparse" else input_var.num_features
        bound = numpy.sqrt(6. / (expected_input_size + num_outputs))
        if output_activation_name == "sigmoid":
            bound = 4 * numpy.sqrt(6. / (expected_input_size + num_outputs))
        elif output_activation_name == "rlu":
            bound = numpy.sqrt(2. / expected_input_size)

        self.num_outputs = num_outputs
        self.output_activation_name = output_activation_name
        W_values = numpy.asarray(
            rng.uniform(low=-bound, high=bound,
                size=(input_var.num_features, self.num_outputs)),
            dtype=numpy.float32)
        self.W = theano.shared(value=W_values, name="W")
        b_values = numpy.zeros((1, num_outputs), dtype=numpy.float32)
        self.b = theano.shared(value=b_values, name="b", broadcastable=(True, False))
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        input = inputs[0]
        dynamic_params = []
        dynamic_penalized_params = []
        if self.input_var.type == "sparse":
            res = sparse_dot(input, self.W, self.input_var.num_features)
            output = res.out
            dynamic_params.append(res)
            dynamic_penalized_params.append(res.compactW)
        else:
            output = T.dot(input, self.W)
        output = output + self.b
        return output, dynamic_params, dynamic_penalized_params
    def fprop(self, inputs):
        assert len(inputs) == 1
        input = inputs[0]
        if self.input_var.type == "sparse":
            return theano.sparse.structured_dot(input, self.W) + self.b
        else:
            return T.dot(input, self.W) + self.b
    def get_params(self):
        params = [self.b]
        if self.input_var.type != "sparse":
            params.append(self.W)
        return params
    def get_penalized_params(self):
        penalized_params = []
        if self.input_var.type != "sparse":
            penalized_params.append(self.W)
        return penalized_params
    def __getinitargs__(self):
        return (self.rng, self.input_var, self.num_outputs, self.output_activation_name)
    def __get_state__(self):
        data = {
            "W" : self.W.get_value(),
            "b" : self.b.get_value()
        }
        return data
    def __set_state__(self, state):
        self.W.set_value(state["W"])
        self.b.set_value(state["b"])

class RluModule(Module):
    def __init__(self, input):
        Module.__init__(self, [input], Variable("dense", input.num_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        var = inputs[0]
        return var * (var > 0), [], []
def rlu(rng, input_variable, num_outputs=None):
    modules = []
    if not isinstance(input_variable, Variable):
        raise Exception("Unknown input: " + str(input_variable))
    if num_outputs is not None:
        num_outputs = int(num_outputs)
        modules.append(AffineModule(rng, input_variable,
            num_outputs, "rlu"))
        input_variable = modules[-1].get_output()
    modules.append(RluModule(input_variable))
    return modules

class SigmoidModule(Module):
    def __init__(self, input):
        Module.__init__(self, [input], Variable("dense", input.num_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        var = inputs[0]
        return T.nnet.sigmoid(var), [], []
def sigmoid(rng, input_variable, num_outputs=None):
    modules = []
    if not isinstance(input_variable, Variable):
        raise Exception("Unknown input: " + str(input_variable))
    if num_outputs is not None:
        num_outputs = int(num_outputs)
        modules.append(AffineModule(rng, input_variable,
            num_outputs, "sigmoid"))
        input_variable = modules[-1].get_output()
    modules.append(SigmoidModule(input_variable))
    return modules

class TanhModule(Module):
    def __init__(self, input):
        Module.__init__(self, [input], Variable("dense", input.num_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        var = inputs[0]
        return T.tanh(var), [], []
def tanh(rng, input_variable, num_outputs=None):
    if not isinstance(input_variable, Variable):
        raise Exception("Unknown input: " + str(input_variable))
    modules = []
    if num_outputs is not None:
        num_outputs = int(num_outputs)
        modules.append(AffineModule(rng, input_variable,
            num_outputs, "tanh"))
        input_variable = modules[-1].get_output()
    modules.append(TanhModule(input_variable))
    return modules
class SignedSqrtModule(Module):
    def __init__(self, input):
        Module.__init__(self, [input], Variable("dense", input.num_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        var = inputs[0]
        return T.sgn(var) * T.sqrt(abs(var)), [], []
def signed_sqrt(rng, input_variable, num_outputs=None):
    if not isinstance(input_variable, Variable):
        raise Exception("Unknown input: " + str(input_variable))
    modules = []
    if num_outputs is not None:
        num_outputs = int(num_outputs)
        modules.append(AffineModule(rng, input_variable,
            num_outputs, "signed_sqrt"))
        input_variable = modules[-1].get_output()
    modules.append(SignedSqrtModule(input_variable))
    return modules

def linear(rng, input_variable, num_outputs):
    if not isinstance(input_variable, Variable):
        raise Exception("Unknown input: " + str(input_variable))
    modules = []
    num_outputs = int(num_outputs)
    modules.append(AffineModule(rng, input_variable,
        num_outputs, "linear"))
    return modules

class ConcatModule(Module):
    def __init__(self, inputs):
        num_output_features = sum([input.num_features for input in inputs])
        Module.__init__(self, inputs, Variable("dense", num_output_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == len(self.get_inputs())
        return T.concatenate(inputs, axis=1), [], []
def concat(*inputs):
    if len(inputs) < 2:
        raise Exception("Concat function: nothing to concatenate")
    for input in inputs:
        if not isinstance(input, Variable):
            raise Exception("Unknown input: " + str(input))
    return [ConcatModule(inputs)]

class DotModule(Module):
    def __init__(self, inputs):
        num_output_features = 1
        Module.__init__(self, inputs, Variable("dense", num_output_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 2
        return T.sum(inputs[0] * inputs[1], axis=1).dimshuffle((0, 'x')), [], []
def dot(*inputs):
    if len(inputs) != 2:
        raise Exception("Wrong number of inputs to dot: {}. Should be 2".format(len(inputs)))
    for input in inputs:
        if not isinstance(input, Variable):
            raise Exception("Unknown input: " + str(input))
    return [DotModule(inputs)]

class ScaleModule(Module):
    def __init__(self, input, scaler):
        Module.__init__(self, [input], Variable(input.type, input.num_features, self))
        self.scaler = scaler
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        return inputs[0] * self.scaler, [], []
def scale(*inputs):
    if len(inputs) != 2:
        raise Exception("Invalid number of inputs in scale function")
    if not isinstance(inputs[0], Variable):
        raise Exception("Unknown input: " + str(input))
    try:
        float(inputs[1])
    except TypeError:
        raise Exception("Cannot parse scalin constat as float")
    return [ScaleModule(inputs[0], float(inputs[1]))]

class SubModule(Module):
    def __init__(self, inputs):
        num_output_features = 1
        Module.__init__(self, inputs, Variable("dense", num_output_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 2
        return inputs[0] - inputs[1], [], []
def sub(*inputs):
    if len(inputs) != 2:
        raise Exception("Wrong number of inputs to dot: {}. Should be 2".format(len(inputs)))
    for input in inputs:
        if not isinstance(input, Variable):
            raise Exception("Unknown input: " + str(input))
    return [SubModule(inputs)]

class MseEnergy(Module):
    def __init__(self, model, transformer=None):
        inputs = (model.get_inputs(), Variable("dense", name="targets",
            transformer=transformer), Variable("dense", name="weights"))
        Module.__init__(self, inputs, Variable("dense"))
        self.model = model
    def train_fprop(self, inputs):
        model_inputs = inputs[0]
        model_output, model_dynamic_params, model_dynamic_penalized_params =\
            self.model.train_fprop(model_inputs)
        targets = inputs[1]
        weights = inputs[2]
        return T.sum(weights * T.sqr(targets - model_output)) / T.sum(abs(weights)), \
            model_dynamic_params, model_dynamic_penalized_params
    def fprop(self, inputs):
        model_inputs = inputs[0]
        targets = inputs[1]
        weights = inputs[2]
        model_output = self.model.fprop(model_inputs)
        return T.sum(weights * T.sqr(targets - model_output)) / T.sum(abs(weights))
    def get_params(self):
        return self.model.get_params()
    def get_penalized_params(self):
        return self.model.get_penalized_params()
class CrossEntropyEnergy(Module):
    def __init__(self, model, transformer=None):
        inputs = (model.get_inputs(), Variable("dense", name="targets",
            transformer=transformer), Variable("dense", name="weights"))
        Module.__init__(self, inputs, Variable("dense"))
        self.model = model
    def train_fprop(self, inputs):
        model_inputs = inputs[0]
        model_output, model_dynamic_params, model_dynamic_penalized_params =\
            self.model.train_fprop(model_inputs)
        targets = inputs[1]
        weights = inputs[2]
        return -T.sum(weights * (targets * T.log(model_output + numpy.float32(1e-10)) +
            (numpy.float32(1) - targets) * T.log(numpy.float32(1.0000001) - model_output))) / T.sum(abs(weights)), \
            model_dynamic_params, model_dynamic_penalized_params
    def fprop(self, inputs):
        model_inputs = inputs[0]
        targets = inputs[1]
        weights = inputs[2]
        model_output = self.model.fprop(model_inputs)
        return -T.sum(weights * (targets * T.log(model_output + numpy.float32(1e-10)) +
            (numpy.float32(1) - targets) * T.log(numpy.float32(1.0000001) - model_output) )) / T.sum(abs(weights))
    def get_params(self):
        return self.model.get_params()
    def get_penalized_params(self):
        return self.model.get_penalized_params()
class AccuracyEnergy(Module):
    def __init__(self, model, transformer=None):
        inputs = (model.get_inputs(), Variable("dense", name="targets",
            transformer=transformer), Variable("dense", name="weights"))
        Module.__init__(self, inputs, Variable("dense"))
        self.model = model
    def train_fprop(self, inputs):
        model_inputs = inputs[0]
        model_output, model_dynamic_params, model_dynamic_penalized_params =\
            self.model.train_fprop(model_inputs)
        targets = inputs[1]
        weights = inputs[2]

        # this expression assumes that there is just one output with everything
        #   above 0.5 positive (common assumption for binary classification)
        # if there are more outputs, there are more possible options - currently this is not implemented
        return T.sum(weights * (targets == (model_output > numpy.float32(0.5)))) / T.sum(abs(weights)), \
            model_dynamic_params, model_dynamic_penalized_params
    def fprop(self, inputs):
        model_inputs = inputs[0]
        targets = inputs[1]
        weights = inputs[2]
        model_output = self.model.fprop(model_inputs)
        return T.sum(weights * (targets == (model_output > numpy.float32(0.5)))) / T.sum(abs(weights))
    def get_params(self):
        return self.model.get_params()
    def get_penalized_params(self):
        return self.model.get_penalized_params()

# --------------------------core model---------------------------------------

class FuncWrapperModule(Module):
    def __init__(self, func_module, inputs):
        assert len(inputs) == len(func_module.get_inputs())
        output = func_module.get_output()
        Module.__init__(self, inputs, Variable(output.type, output.num_features, self))
        self.func_module = func_module
        self.inputs = inputs
    def train_fprop(self, inputs):
        assert len(inputs) == len(self.get_inputs())
        return self.func_module.train_fprop(inputs)
    def fprop(self, inputs):
        return self.func_module.fprop(inputs)
    def get_params(self):
        return self.func_module.get_params()
    def get_penalized_params(self):
        return self.func_module.get_penalized_params()

class FuncModule(Module):
    def __init__(self, inputs, output, toposorted_modules):
        assert isinstance(inputs, (list, tuple))
        Module.__init__(self, inputs, output)
        # just a validity check: all real inputs are in inputs
        inputs_map = {var.name: var for var in inputs}
        intermediates = set()
        for module in toposorted_modules:
            for input in module.get_inputs():
                if not isinstance(input, Variable):
                    raise Exception("Incorrect input specified: {}".format(input))
                if input not in intermediates:
                    assert input.name in inputs_map
            intermediates.add(module.get_output())

        self.toposorted_modules = toposorted_modules
    def train_fprop(self, inputs):
        assert len(inputs) == len(self.get_inputs())
        context = {}
        for input, input_var in zip(inputs, self.get_inputs()):
            context[input_var] = input
        dynamic_params = []
        dynamic_penalized_params = []
        for module in self.toposorted_modules:
            inputs = [context[input] for input in module.get_inputs()]
            module_output, module_dynamic_params, module_dynamic_penalized_params = module.train_fprop(inputs)
            context[module.get_output()] = module_output
            dynamic_params.extend(module_dynamic_params)
            dynamic_penalized_params.extend(module_dynamic_penalized_params)
        return module_output, dynamic_params, dynamic_penalized_params
    def fprop(self, inputs):
        assert len(inputs) == len(self.get_inputs())
        context = {}
        for input, input_var in zip(inputs, self.get_inputs()):
            context[input_var] = input
        for module in self.toposorted_modules:
            inputs = [context[input] for input in module.get_inputs()]
            module_output = module.fprop(inputs)
            context[module.get_output()] = module_output
        return module_output
    def get_params(self):
        params = set()
        for module in self.toposorted_modules:
            params |= set(module.get_params())
        return list(params)
    def get_penalized_params(self):
        params = set()
        for module in self.toposorted_modules:
            params |= set(module.get_penalized_params())
        return list(params)

class UnnComputer(Module):
    def __init__(self, func):
        Module.__init__(self, func.get_inputs(), func.get_output())
        self.func = func
    def train_fprop(self, inputs):
        return self.func.train_fprop(inputs)
    def fprop(self, inputs):
        return self.func.fprop(inputs)
    def get_params(self):
        return self.func.get_params()
    def get_penalized_params(self):
        return self.func.get_penalized_params()
    def get_inputs(self):
        return self.func.get_inputs()

class Unn(object):
    def __init__(self, funcs, inputs, output_name, architecture_string):
        self.funcs = funcs
        self.inputs = inputs
        self.output_name = output_name
        self.architecture_string = architecture_string
    def get_computer(self, output_name=None):
        if output_name is None:
            output_name = self.output_name
        if output_name not in self.funcs:
            raise Exception(output_name + " is not in model variables")
        inputs = set()
        used_modules = set()
        toposorted_modules = []
        var2producer = {func.get_output() : func for func in self.funcs.values()}
        def unravel_computer(var):
            if var in var2producer:
                producer = var2producer[var]
                for input_var in producer.get_inputs():
                    unravel_computer(input_var)
                if producer not in used_modules:
                    toposorted_modules.append(producer)
                    used_modules.add(producer)
            else:
                inputs.add(var)
        output_var = self.funcs[output_name].get_output()
        unravel_computer(output_var)
        func = FuncModule(list(inputs), output_var, toposorted_modules)
        return UnnComputer(func)

# --------------------------- expression parsing -----------------------
def get_name_and_inputs(func_definition):
    func_name, inputs = func_definition.split("(", 1)
    if not inputs.endswith(")"):
        raise Exception("Parentheses mismatch in function definition " + func_definition)
    inputs = inputs[:-1]
    input_expressions = []
    last_pos = 0
    level = 0
    for i, c in enumerate(inputs):
        if c == "(":
            level += 1
        elif c == ")":
            level -= 1
            if level < 0:
                raise Exception("Parentheses mismatch in function expression "
                    "'{}' at position {}".format(func_definition, len(func_name) + 1 + i))
        else:
            if level == 0 and c == ",":
                input_expressions.append(inputs[last_pos:i])
                last_pos = i + 1
    if last_pos != len(inputs):
        input_expressions.append(inputs[last_pos:len(inputs)])
    return func_name, input_expressions
def make_function(function_string, known_funcs, rng, external_variables=None):
    if function_string.count("=") != 1:
        raise Exception("Invalid function definition: " + function_string)
    if external_variables is None:
        external_variables = {}
    function_string = ''.join(function_string.replace("[", "(").replace(
        "]", ")").replace("{", "(").replace("}", ")").replace(
        "<", "(").replace(">", ")").split())
    func_definition, function_expression = function_string.split("=")

    # parse func definition
    if "(" in func_definition:
        new_func_name, func_inputs = func_definition.strip(")").split("(")
        if "(" in new_func_name:
            raise Exception("Invalid function name: " + new_func_name)
        if new_func_name == "":
            raise Exception("Function name should not be empty!")
        if new_func_name in known_funcs:
            raise Exception("Duplicate function name: " + new_func_name)
        if new_func_name in external_variables:
            raise Exception("Duplicate function name: " + new_func_name)
        func_inputs = func_inputs.split(",")
        inputs_names = set()
        for idx, input in enumerate(func_inputs):
            if "(" in input:
                raise Exception("Invalid function input: {}. Function name: {}".format(input, new_func_name))
            items = input.split(":")
            if len(items) != 3:
                raise Exception("Invalid specification of function input. "
                    "Should be <input_name>:<type>:<num_features>. Have: " + input)
            name = items[0]
            type = items[1]
            if type not in ["dense", "sparse"]:
                raise Exception("Invalid specification of function input. "
                    "Should contain: <input_name>:<type>:<num_features>. Have: {}. "
                    "Invalid type: {}".format(input, type))
            try:
                num_features = int(items[2])
            except TypeError:
                raise Exception("Cannot parse {} as integer (should be number of features). "
                    "Function name: {}, input {}. Input should have format "
                    "<input_name>:<type>:<num_features>".format(
                        items[2], name, input))

            if name in inputs_names or name in external_variables:
                raise Exception("Duplicate input name: " + name)
            inputs_names.add(name)

            func_inputs[idx] = Variable(name=name, type=type, num_features=num_features)
    else:
        new_func_name = func_definition
        func_inputs = []

    # parse func expression
    input_map = {var.name: var for var in func_inputs}
    toposorted_modules = []
    def sub_make_function(expression):
        inputs = []
        func_name, inputs_strs = get_name_and_inputs(expression)
        for input_str in inputs_strs:
            if "(" in input_str:
                subfunction_output = sub_make_function(input_str)
                inputs.append(subfunction_output)
            elif input_str in input_map:
                inputs.append(input_map[input_str])
            elif input_str in external_variables:
                var_name = input_str
                inputs.append(external_variables[var_name])
                input_map[var_name] = external_variables[var_name]
                func_inputs.append(input_map[var_name])
            else:
                inputs.append(input_str)
        if func_name in known_funcs:
            modules = [FuncWrapperModule(known_funcs[func_name], inputs)]
        elif func_name == "rlu":
            modules = rlu(rng, *inputs)
        elif func_name == "sigmoid":
            modules = sigmoid(rng, *inputs)
        elif func_name == "tanh":
            modules = tanh(rng, *inputs)
        elif func_name == "linear":
            modules = linear(rng, *inputs)
        elif func_name == "concat":
            modules = concat(*inputs)
        elif func_name == "dot":
            modules = dot(*inputs)
        elif func_name == "scale":
            modules = scale(*inputs)
        elif func_name == "signed_sqrt":
            modules = signed_sqrt(rng, *inputs)
        elif func_name == "sub":
            modules = sub(*inputs)
        else:
            raise Exception("Unknown function: {}".format(func_name))
        toposorted_modules.extend(modules)
        return modules[-1].get_output()
    output = sub_make_function(function_expression)
    return new_func_name, FuncModule(func_inputs, output, toposorted_modules)
def build_unn(input_variables, architecture_string):
    modules = architecture_string.split("|")
    inputs_map = {}
    for variable in input_variables:
        inputs_map[variable.name] = variable
    funcs = {}
    rng = numpy.random.RandomState(0)
    for function_string in modules:
        func_name, func_module = make_function(function_string, funcs, rng, inputs_map)
        assert func_name not in funcs
        funcs[func_name] = func_module
        assert func_name not in inputs_map
        inputs_map[func_name] = func_module.get_output()
    output_name = func_name
    return Unn(funcs, input_variables, output_name, architecture_string)

# --------------------------- data processing---------------------

class SparseMatrixBuilder(object):
    def __init__(self, num_features, capacity):
        self.rows = numpy.zeros([capacity], dtype=numpy.int32)
        self.columns = numpy.zeros([capacity], dtype=numpy.int32)
        self.vals = numpy.zeros([capacity], dtype=numpy.float32)
        self.row = 0
        self.pos = 0
        self.num_features = num_features
    def append_row(self, str_):
        if str_ == "":
            self.row += 1
            return
        # the actual format is idx:val idx:val
        # but this way is faster
        entries = str_.replace(":", " ").split(" ")
        num_elements = len(entries) / 2
        assert num_elements * 2 == len(entries)

        while self.pos + num_elements >= len(self.rows):
            self.rows.resize([2 * len(self.rows)])
            self.columns.resize([2 * len(self.columns)])
            self.vals.resize([2 * len(self.vals)])
        self.columns[self.pos: self.pos + num_elements] = map(int, entries[0::2])
        self.vals[self.pos: self.pos + num_elements] = map(float, entries[1::2])
        self.rows[self.pos: self.pos + num_elements] = self.row
        self.row += 1
        self.pos += num_elements
    def get(self):
        return scipy.sparse.csr_matrix((self.vals[:self.pos],
            (self.rows[:self.pos], self.columns[:self.pos])),
            shape=(self.row, self.num_features))
class DenseMatrixBuilder(object):
    def __init__(self, num_features, num_samples, dtype=numpy.float32):
        self.array = numpy.zeros([num_samples, num_features], dtype=dtype)
        self.dtype = dtype
        self.pos = 0
    def append_row(self, str_):
        self.array[self.pos] = numpy.fromstring(str_, dtype=numpy.float32, sep=" ")
        self.pos += 1
    def get(self):
        return self.array[:self.pos]

# Weights in sgd are a problem. What if one sample is extremely important?
# If samples are taken uniformly, this will lead to an extremely large
# update when this sample finally appears that may destroy the model.
# Here we solve this by sampling from the weights distribution.
# Thus, learning may be slower, and the dataset is kept in memory
class RandomDatasetIterator(object):
    def __init__(self, dataset, batch_size):
        weights_pos = -1
        for idx, var in enumerate(dataset.input_vars):
            if var.name == "weights":
                weights_pos =  idx
                break
        if (weights_pos < 0):
            raise Exception("No weights in dataset")

        self.cumsum = numpy.cumsum(abs(dataset.arrays[weights_pos]) * 1.0 / sum(abs(dataset.arrays[weights_pos])))
        self.cumsum[-1] = 1
        self.num_batches = math.ceil(dataset.arrays[0].shape[0] * 1.0 / batch_size)
        self.batch_size = batch_size
        self.dataset = dataset
        self.pos = 0
    def __iter__(self):
        return self
    def next(self):
        if self.pos >= self.num_batches:
            raise StopIteration()
        else:
            self.pos += 1
            samples_ids = [bisect.bisect_left(self.cumsum, numpy.random.random()) for _ in range(self.batch_size)]
            res = {}
            for var, array in zip(self.dataset.input_vars, self.dataset.arrays):
                if var.name == "weights":
                    weights = (2 * (array[samples_ids] > 0) - 1.0).astype(numpy.float32)
                    res[var.name] = weights
                else:
                    res[var.name] = array[samples_ids]
            return res

class SequentialDatasetIterator(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.num_samples = self.dataset.arrays[0].shape[0]
        self.batch_size = batch_size
    def __iter__(self):
        self.pos = 0
        return self
    def next(self):
        if self.pos >= self.num_samples:
            raise StopIteration()
        else:
            next_pos = min(self.pos + self.batch_size, self.num_samples)
            res = {}
            for var, array in zip(self.dataset.input_vars, self.dataset.arrays):
                res[var.name] = array[self.pos:next_pos]
            self.pos = next_pos
            return res

class MemoryDataset(object):
    def __init__(self, arrays, input_vars, use_random_iterator):
        self.arrays = arrays
        self.input_vars = input_vars
        self.use_random_iterator = use_random_iterator
    def read_train(self, batch_size):
        if self.use_random_iterator:
            return RandomDatasetIterator(self, batch_size)
        else:
            return SequentialDatasetIterator(self, batch_size)
    def read(self, batch_size):
        return SequentialDatasetIterator(self, batch_size)

class FileIterator(object):
    def __init__(self, input_file, input_vars, batch_size):
        self.input_file = open(input_file)
        self.input_vars = input_vars
        self.batch_size = batch_size
    def __iter__(self):
        return self
    def next(self):
        arrays = []
        for input_var in self.input_vars:
            if input_var.type == "sparse":
                arrays.append(SparseMatrixBuilder(input_var.num_features, self.batch_size))
            elif input_var.type == "dense":
                arrays.append(DenseMatrixBuilder(input_var.num_features, self.batch_size))
            else:
                arrays.append(None)

        have_data = False
        for idx, line in enumerate(self.input_file):
            have_data = True
            entries = line.strip("\n").split("\t")
            for input_idx, entry in enumerate(entries):
                if arrays[input_idx] is not None:
                    arrays[input_idx].append_row(entry)
            if idx + 1 >= self.batch_size:
                break
        if have_data:
            res = {}
            for var, data in zip(self.input_vars, arrays):
                if data is not None:
                    res[var.name] = data.get()
            return res
        else:
            raise StopIteration()

class FileDataset(object):
    def __init__(self, input_file, input_vars):
        self.input_file = input_file
        self.input_vars = input_vars
    def read_train(self, batch_size):
        return FileIterator(self.input_file, self.input_vars, batch_size)
    def read(self, batch_size):
        return FileIterator(self.input_file, self.input_vars, batch_size)

def get_transformer(transformer_name):
    if transformer_name == "scale":
        return sklearn.preprocessing.StandardScaler()
    elif transformer_name == "minmax":
        return sklearn.preprocessing.MinMaxScaler()
    elif transformer_name == "pca":
        return sklearn.decomposition.PCA(whiten=True)
    elif transformer_name == "none":
        return None
    else:
        assert False

def get_input_vars(inputs_specification, dataset_file):
    input_vars = []
    inputs_names = set()
    inputs_specs = inputs_specification.split(",")
    for spec in inputs_specs:
        entries = spec.split("@")[0].split(":")
        if len(entries) > 3:
            raise Exception("Frong number of fields in input specification: {}. Should be: name[:type:num_features@transformer]".format(spec))
        name = entries[0]
        if name in inputs_names:
            raise Exception("Duplicate input: " + name)
        inputs_names.add(name)
        type = None
        if len(entries) > 1:
            type = entries[1]
        num_features = None
        if len(entries) > 2:
            num_features = int(entries[2])

        input_vars.append(Variable(name=name, type=type,
            num_features=num_features, transformer=None))

    if dataset_file is not None:
        with open(dataset_file) as f:
            items = f.readline().split("\t")
            if len(items) != len(input_vars):
                raise Exception("Number of inputs in {} is not equal to the number "
                    "of inputs specified in the command line".format(dataset_file))

            for idx, (input_var, item) in enumerate(zip(input_vars, items)):
                if item == "" or ":" in item:
                    if input_var.type is not None and input_var.type != "sparse":
                        raise Exception("The provided data type is inconsistent with "
                            "data: input '{}' should be sparse" .format(input_var.name))
                    input_var.type = "sparse"
                    if input_var.num_features is None:
                        raise Exception("Number of features should be specified for "
                            "sparse input '{}'".format(input_var.name))
                else:
                    if input_var.type is not None and input_var.type != "dense":
                        raise Exception("The provided data type is inconsistent with "
                            "data: input '{}' should be dense" .format(input_var.name))
                    input_var.type = "dense"
                    if input_var.num_features is not None and input_var.num_features != len(item.split(" ")):
                        raise Exception("Number of features mismatch for input {}. Got {}, specified {}".format(
                            input_var.name, len(items.split(" ")), input_var.num_features))
                    input_var.num_features = len(item.split(" "))

    for idx, var in enumerate(input_vars):
        if "@" in inputs_specs[idx]:
            entries = spec.split("@")
            if len(entries) != 2:
                raise Exception("Invalid input specification: multiple @ found for input {}".format(spec))
            transform_name = entries[-1]
            spec = entries[0]
        else:
            if var.type == "dense":
                transform_name = "scale"
            else:
                transform_name = "none"
        if transform_name not in ["none", "pca", "scale", "minmax"]:
            raise Exception("Unknown transformer name: " + transform_name)
        var.transformer = get_transformer(transform_name)
        if var.type == "sparse" and var.transformer is not None:
            raise Exception("Transformer cannot be used with a sparse input '{}'".format(input_var.name))
    return input_vars

def load_dataset_from_file(file_, input_vars, use_random_iterator=False):
    with open(file_) as f:
        num_samples = sum(1 for line in f)

    arrays = []
    for input_var in input_vars:
        if input_var.type == "sparse":
            arrays.append(SparseMatrixBuilder(input_var.num_features, num_samples))
        else:
            arrays.append(DenseMatrixBuilder(input_var.num_features, num_samples))
    with open(file_) as f:
        for idx, line in enumerate(f):
            if idx % 10000 == 0:
                sys.stdout.write("\rreading sample {}".format(idx))
                sys.stdout.flush()
            entries = line.strip("\n").split("\t")
            if len(entries) != len(arrays):
                raise Exception("Wrong number of inputs in " + file_)

            for builder, entry in zip(arrays, entries):
                builder.append_row(entry)
    sys.stdout.write("\r" + " " * 100 + "\r")
    arrays = map(lambda x: x.get(), arrays)

    return MemoryDataset(arrays, input_vars, use_random_iterator)

#--------------------------------modes-----------------------------------

def learn():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", default="model.net")
    parser.add_argument("-c", "--continue_learning", action="store_const", const=True,
        default=False)

    # data specification
    parser.add_argument('-f', "--train", required=True,
        help="path to the learn dataset (format: target \\t weight \\t space-separated_input [\\t space-separated_input ...])")
    parser.add_argument("--mf", dest="keep_train_in_memory", help="read train data from memory", action="store_const", const=True, default=False)
    parser.add_argument('-t', "--test",
        help="path to the validation dataset (format: target \\t weight \\t space-separated_input [\\t space-separated_input ...])")
    parser.add_argument("--mt", dest="keep_validation_in_memory", help="read validation data from memory", action="store_const", const=True, default=False)
    parser.add_argument('-i', "--inputs",
        help="Inputs names. Format: <name>[:<type>:<num_features>]@<preprocessor>,<name>[:<type>:<num_features>]@<preprocessor>. "
            "Names 'targets' and 'weights' are reserved."
            "Entry 'num_features' is optional for dense input. Entry 'type' for dense input is optional - it will be inferred from "
            "the data. Entry 'preprocessor' (valid options: pca, scale, minmax, none. Default - scale) is valid only for dense input")

    # unn architecture
    parser.add_argument("-a", "--architecture", help="architecture of the network. Format <func>|<func>|...|<func>"
        " where <func> can be <func_name>(<input_name>:<type>:<num_features>,...)=func_expression(input_name) (use this to create shared modules)"
        " or <output_name>=func_expression. The last <func> is model output")
    parser.add_argument("--loss", choices=["mse", "cross_entropy"], help="Loss function", default="mse")
    parser.add_argument("--vloss", dest="validation_loss", choices=["accuracy", "mse", "cross_entropy"],
        help="Validation loss (specify if you want it be different from train loss). Currently 'accuracy' supports just one output", default=None)

    # learning parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs. Each epoch is one pass through the train dataset")
    parser.add_argument("--batch_size", type=int, default=30,
                        help="Batch size")
    parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum")
    parser.add_argument("-l", dest="reg_lambda", type=float, default=0,
                        help="l2 regularization lambda")
    parser.add_argument("--val_freq", type=int, default=10000,
        help="Number of batches between validations")
    parser.add_argument("--rand", dest="use_random_iterator", action="store_const",
        default=False, const=True, help="Sample train batches (otherwise they will be consumed sequentially)")

    args = parser.parse_args()

    if args.validation_loss is None:
        args.validation_loss = args.loss

    if not os.path.exists(args.train):
        raise Exception("Train file is missing")
    if not os.path.exists(args.test):
        raise Exception("Validation file is missing")

    args.inputs = "".join(args.inputs.split())
    args.architecture = "".join(args.architecture.split())

    with open(args.train) as f:
        entries = f.readline().split("\t")
        if len(entries) < 3:
            raise Exception("Not enough inputs in the train file")
        num_targets = len(entries[0].split(" "))
        num_weights = len(entries[1].split(" "))
        if num_weights != 1:
            raise Exception("There should be 1 weight for each input sample in the train file")

    print "Loading the model"
    if args.continue_learning:
        assert os.path.exists(args.model_path)
        try:
            with open(args.model_path) as f:
                unn = pickle.load(f)
            print "Model was loaded from path " + args.model_path
        except Exception as ex:
            sys.stderr.write("Cannot load the model: {}".format(ex))
            raise
        print "Warning: input definitions are reused from the previous runs."
        model_input_vars = unn.inputs
    else:
        if os.path.exists(args.model_path):
            raise Exception("Model file exists! The old file will not be overriden")

        input_specs = "targets:dense:{},weights:dense:1,".format(num_targets) + args.inputs
        model_input_vars = get_input_vars(input_specs, args.train)[2:]
        unn = build_unn(model_input_vars, args.architecture)

    actual_input_vars = copy.copy(model_input_vars)
    actual_input_vars.insert(0, Variable(name="weights", type="dense", num_features=1))
    actual_input_vars.insert(0, Variable(name="targets", type="dense", num_features=num_targets))

    print "reading train"
    assert not args.use_random_iterator or args.keep_train_in_memory
    if args.keep_train_in_memory:
        train_dataset = load_dataset_from_file(args.train, actual_input_vars, args.use_random_iterator)
    else:
        train_dataset = FileDataset(args.train, actual_input_vars)
    print "reading validation"
    if args.test is not None:
        if args.keep_validation_in_memory:
            validation_dataset = load_dataset_from_file(args.test, actual_input_vars)
        else:
            validation_dataset = FileDataset(args.test, actual_input_vars)
    else:
        validation_dataset = None

    learn_model = unn.get_computer()
    if args.loss == "mse":
        train_energy = MseEnergy(learn_model)
    elif args.loss == "cross_entropy":
        train_energy = CrossEntropyEnergy(learn_model)
    else:
        assert False

    if validation_dataset is not None:
        if args.validation_loss == "mse":
            validation_energy = MseEnergy(learn_model)
        elif args.validation_loss == "cross_entropy":
            validation_energy = CrossEntropyEnergy(learn_model)
        elif args.validation_loss == "accuracy":
            validation_energy = AccuracyEnergy(learn_model)
        else:
            assert False
    else:
        validation_energy = None

    trainer = Trainer(learning_rate=args.learning_rate, momentum=args.momentum,
        num_epochs=args.epochs, reg_lambda=args.reg_lambda, batch_size=args.batch_size,
        validation_frequency=args.val_freq, loss_function=args.loss)

    print "Learning"
    trainer.fit(model=unn, train_energy=train_energy,
        validation_energy=validation_energy, save_path=args.model_path,
        train_dataset=train_dataset, validation_dataset=validation_dataset,
        continue_learning=args.continue_learning)
    print "Finished"

def apply_model_to_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", default="model.net", required=True)
    parser.add_argument("-o", "--output", default=None,
        help="Which variable to output")
    parser.add_argument('-d', "--data", required=True,
        help="path to the dataset file")
    parser.add_argument('-i', "--inputs", help="csv inputs names")
    parser.add_argument('-f', "--output_file", help="file where to put the result", required=True)
    args = parser.parse_args()

    assert os.path.exists(args.model_path)
    try:
        with open(args.model_path) as f:
            unn = pickle.load(f)
        print "model was loaded from path " + args.model_path
    except Exception as ex:
        sys.stderr.write("Cannot load the model: {}".format(ex))
        raise

    model = unn.get_computer(args.output)
    model_inputs = {var.name: var for var in model.get_inputs()}
    user_inputs = ''.join(args.inputs.split()).split(",")
    for input in model_inputs:
        if input not in user_inputs:
            raise Exception("Input '{}' is required".format(input))
    inputs = []
    for input in user_inputs:
        if input in model_inputs:
            inputs.append(model_inputs[input])
        else:
            inputs.append(Variable(type=None, name="input"))
    dataset = FileDataset(args.data, inputs)
    with open(args.output_file, "w") as of:
        apply_model(model, dataset, of)

def describe_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", default="model.net", required=True)
    args = parser.parse_args()

    assert os.path.exists(args.model_path)
    try:
        with open(args.model_path) as f:
            unn = pickle.load(f)
        print "model was loaded from path " + args.model_path
    except Exception as ex:
        sys.stderr.write("Cannot load the model: {}".format(ex))
        raise

    print "Model archtecture:"
    for module in unn.architecture_string.split("|"):
        print "\t{}".format(module)
    print "Model inputs:"
    for var in unn.inputs:
        print "\tName={}; Type={}; Transformer={}; Number of features={}".format(
            var.name, var.type, var.transformer, var.num_features)

def print_help():
    print "Neural network-based sparse vectors matcher. Usage:"
    print "\tunn.py learn ... - learn a model"
    print "\tunn.py apply ... - apply model to data"
    print "\tunn.py describe ... - print model architecture"
def dispatch():
    if len(sys.argv) == 1 or len(sys.argv) == 2 and sys.argv[1] == "-h":
        print_help()
        sys.exit()
    else:
        mode = sys.argv[1]
        del sys.argv[1]

        if mode == "learn":
            learn()
        elif mode == "apply":
            apply_model_to_data()
        elif mode == "describe":
            describe_model()
        else:
            print "Unknown mode: {}".format(mode)
            print_help()
            sys.exit(1)

if __name__ == "__main__":
    dispatch()
