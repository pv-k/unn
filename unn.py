#!/usr/bin/env python

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
import math
import scipy.sparse
import scipy.stats
import shutil
import time
import collections
import tempfile
import copy
import bisect
import threading
import Queue

import theano
import theano.sparse
import theano.tensor as T
import theano.tensor.nnet
from theano.compile.ops import as_op
import sklearn.preprocessing
import signal
import concurrent.futures

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

def sgd_update(energy, params, learning_rates, momentums, nesterov=True):
    '''
    Sgd with momentum.
        It is a bit complicated because of support for efficient sparse updates
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
                if nesterov:
                    updates.append((param, param - lr * ((numpy.float32(1.0) - momentum) * grad +
                        momentum * updated_param_momentum)))
                else:
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

            new_param_W = param_W
            # decay momentum and add updates
            updated_param_momentum = T.set_subtensor(param_momentum[updated_indices],
                param_momentum[updated_indices] * momentum)
            for lr, momentum, grad, param in param_data:
                updated_param_momentum = T.inc_subtensor(
                    updated_param_momentum[param.nonzeroes],
                    (numpy.float32(1.0) - momentum) * grad)
                if nesterov:
                    new_param_W = T.inc_subtensor(
                        new_param_W[param.nonzeroes], -lr * (numpy.float32(1.0) - momentum) * grad)
            updates.append((param_momentum, updated_param_momentum))

            if nesterov:
                new_param_W = T.inc_subtensor(new_param_W[updated_indices],
                    -lr * momentum * updated_param_momentum[updated_indices])
            else:
                new_param_W = T.inc_subtensor(param_W[updated_indices],
                    -lr * updated_param_momentum[updated_indices])
            updates.append((param_W, new_param_W))
        else:
            new_value = param_W
            for lr, momentum, grad, param in param_data:
                new_value = T.inc_subtensor(new_value[param.nonzeroes],
                    -lr * grad).get_output()
            updates.append((param_W, new_value))
    return updates


def adadelta_update(energy, params, decay, eps):
    decay = numpy.float32(decay)
    eps = numpy.float32(eps)

    all_params = []
    for param in params:
        if isinstance(param, SparseDotParams):
            all_params.append(param.compactW)
        else:
            all_params.append(param)
    updates = []
    grads = T.grad(energy, all_params)

    sparse_params_data = collections.defaultdict(list)
    for param, grad in zip(params, grads):
        if isinstance(param, SparseDotParams):
            sparse_params_data[param.W].append((grad, param))
        else:
            sqr_grad = theano.shared(
                numpy.zeros_like(param.get_value()),
                broadcastable=param.broadcastable)
            sqr_delta = theano.shared(
                numpy.zeros_like(param.get_value()),
                broadcastable=param.broadcastable)
            updated_sqr_grad = sqr_grad * decay + \
                numpy.float32(1.0 - decay) * T.sqr(grad)
            delta = -T.sqrt(sqr_delta + numpy.float32(eps)) / T.sqrt(updated_sqr_grad + eps) * grad
            updated_sqr_delta = sqr_delta * decay + numpy.float32(1 - decay) * T.sqr(delta)
            updates.append((param, param + delta))
            updates.append((sqr_grad, updated_sqr_grad))
            updates.append((sqr_delta, updated_sqr_delta))

    for param_W in sparse_params_data:
        param_data = sparse_params_data[param_W]

        sqr_grad = theano.shared(
            numpy.zeros_like(param_W.get_value()),
            broadcastable=param_W.broadcastable)
        cum_grad = theano.shared(
            numpy.zeros_like(param_W.get_value()),
            broadcastable=param_W.broadcastable)
        sqr_delta = theano.shared(
            numpy.zeros_like(param_W.get_value()),
            broadcastable=param_W.broadcastable)

        updated_indices_vars = []
        for grad, param in param_data:
            updated_indices_vars.append(param.nonzeroes)
        updated_indices = uniq(T.concatenate(updated_indices_vars))

        T.set_subtensor(cum_grad[updated_indices], numpy.float32(0))
        for grad, param in param_data:
            cum_grad = T.inc_subtensor(
                cum_grad[param.nonzeroes], grad)
        updated_sqr_grad = T.set_subtensor(
                sqr_grad[updated_indices],
                sqr_grad[updated_indices] * decay + (numpy.float32(1.0) - decay) * T.sqr(cum_grad[updated_indices]))

        new_param_W = T.inc_subtensor(param_W[updated_indices],
            -T.sqrt(sqr_delta[updated_indices] + eps) / T.sqrt(updated_sqr_grad[updated_indices] + eps) * cum_grad[updated_indices])
        updated_sqr_delta = T.set_subtensor(
                sqr_delta[updated_indices],
                sqr_delta[updated_indices] * decay + numpy.float32(1.0 - decay) *
                    (sqr_delta[updated_indices] + eps) / (updated_sqr_grad[updated_indices] + eps) * T.sqr(cum_grad[updated_indices]))

        updates.append((sqr_grad, updated_sqr_grad))
        updates.append((sqr_delta, updated_sqr_delta))
        updates.append((param_W, new_param_W))
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
            outputs=outputs, updates=updates, on_unused_input="warn")
        self.transformers_map = input_transformers
    def __call__(self, inputs_map):
        for var_name, var_data in inputs_map.items():
            if var_name in self.transformers_map:
                transformer = self.transformers_map[var_name]
                inputs_map[var_name] = transformer.transform(var_data)
        inputs = [inputs_map[var_name] for var_name in self.ordered_names]
        return self.func(*inputs)

class Trainer(object):
    def __init__(self, updater_params, num_epochs, reg_lambda, batch_size,
            validation_frequency, optimizer):
        self.updater_params = updater_params
        self.num_epochs = num_epochs
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.validation_frequency = validation_frequency
        self.optimizer = optimizer
    def fit(self, model, train_energy, validation_energy, save_path,
            train_dataset, validation_dataset, continue_learning, always_save, start_batch=0):
        input_vars = {var.name: var for var in flatten(train_energy.get_inputs())}
        if validation_energy is not None:
            for var in flatten(validation_energy.get_inputs()):
                if var.name not in input_vars:
                    raise Exception("Validation energy has an unknown input: {}".format(var.name))

        have_new_transformers = False
        for var_name, var in input_vars.items():
            if hasattr(var, "transformer_name") and var.transformer_name is not None and var.transformer is None:
                have_new_transformers = True
                break
        if have_new_transformers:
            print "Fitting transformers"
            for num_samples, data_map in train_dataset.read(1000000):
                for var_name, var in input_vars.items():
                    if hasattr(var, "transformer_name") and var.transformer_name is not None and var.transformer is None:
                        var.transformer = get_transformer(var.transformer_name)
                        print "... for " + var_name
                        if var.transformer is not None:
                            var.transformer.fit(data_map[var.name])
                break

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
                raise Exception("Unknown variable type: {}".format(var.type))

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

        l2_penalty = 0
        if self.reg_lambda != 0:
            for param in penalized_params:
                l2_penalty = l2_penalty + T.sum(T.sqr(param))
        l2_penalty = l2_penalty * numpy.float32(self.reg_lambda)
        train_loss = train_loss + l2_penalty

        if self.optimizer in ["nesterov", "sgd"]:
            lr = theano.shared(numpy.float32(self.updater_params.learning_rate))
            if self.updater_params.momentum is not None:
                momentum = theano.shared(numpy.float32(self.updater_params.momentum))
            else:
                momentum = None
            learning_rates = [lr] * len(params)
            momentums = [momentum] * len(params)
            if self.optimizer == "nesterov":
                train = Function(
                    input_transformers=transformers_map,
                    theano_inputs=input_theano_vars,
                    outputs=[train_loss],
                    updates=sgd_update(
                        energy=train_loss,
                        params=params,
                        learning_rates=learning_rates,
                        momentums=momentums,
                        nesterov=True))
            else:
                train = Function(
                    input_transformers=transformers_map,
                    theano_inputs=input_theano_vars,
                    outputs=[train_loss],
                    updates=sgd_update(
                        energy=train_loss,
                        params=params,
                        learning_rates=learning_rates,
                        momentums=momentums,
                        nesterov=False))
        elif self.optimizer == "adadelta":
            train = Function(
                input_transformers=transformers_map,
                theano_inputs=input_theano_vars,
                outputs=[train_loss],
                updates=adadelta_update(
                    energy=train_loss,
                    params=params,
                    decay=self.updater_params.decay,
                    eps=self.updater_params.eps))

        score_train = Function(input_transformers=transformers_map,
            theano_inputs=input_theano_vars,
            outputs=[train_loss])

        if validation_energy is not None:
            validation_inputs = {var.name: input_theano_vars[var.name] for var in
                flatten(validation_energy.get_inputs())}
            flat_validation_inputs = [input_theano_vars[var.name] for var in
                flatten(validation_energy.get_inputs())]
            validation_loss = validation_energy.fprop(
                ravel(flat_validation_inputs, validation_energy.get_inputs()))
            score_validation = Function(input_transformers=transformers_map,
                theano_inputs=validation_inputs,
                outputs=[validation_loss])

        print "Estimating initial performance"
        if validation_energy is not None:
            # compute validation score
            best_validation_score = 0
            samples_total = 0
            for batch_idx, (num_samples, data_map) in enumerate(validation_dataset.read(10000)):
                batch_score = score_validation(data_map)[0]
                samples_total += num_samples
                best_validation_score += num_samples * batch_score
            best_validation_score *= 1.0 / samples_total

            print "Initial validation score: {}".format(best_validation_score)

        # estimate training score
        train_score = 0
        samples_total = 0
        for batch_idx, (num_samples, data_map) in enumerate(train_dataset.read(10000)):
            batch_score = score_train(data_map)[0]
            samples_total += num_samples
            train_score += num_samples * batch_score
            if batch_idx > 10:
                break
        train_score *= 1.0 / samples_total

        print "Learning"
        # training
        train_decay = 0.999
        batch_idx = 0
        epoch_ind = 0

        if self.optimizer in ["nesterov", "sgd"]:
            interrupts_budget = 10
        data_iter = train_dataset.read_train(self.batch_size)
        start_time = time.time()
        while True:
            try:
                while epoch_ind < self.num_epochs:
                    for num_samples, data_map in data_iter:
                        if batch_idx < start_batch:
                            batch_idx += 1
                            if (batch_idx + 1) % 100 == 0:
                                sys.stdout.write("\rCurrent batch:  {}".format(batch_idx))
                                sys.stdout.flush()
                            continue
                        batch_score = train(data_map)[0]
                        train_score = train_decay * train_score + (1 - train_decay) * batch_score

                        if (batch_idx + 1) % 100 == 0:
                            elapsed = time.time() - start_time
                            sys.stdout.write("{}: Train score: {}; Elapsed: {}\n".format(
                                (batch_idx + 1), train_score, elapsed))
                            start_time = time.time()
                        batch_idx += 1
                        if batch_idx % self.validation_frequency == 0:
                            if validation_energy is not None:
                                validation_score = 0
                                samples_total = 0
                                for num_samples, data_map in validation_dataset.read(10000):
                                    batch_score = score_validation(data_map)[0]
                                    samples_total += num_samples
                                    validation_score += num_samples * batch_score
                                validation_score *= 1.0 / samples_total
                                model_saved = False
                                if validation_score < best_validation_score or always_save:
                                    model.save(save_path)
                                    model_saved = True
                                if validation_score < best_validation_score:
                                    best_validation_score = validation_score
                                sys.stdout.write("Epoch {}: validation score: {}; best validation score: {} {}\n".format(
                                    epoch_ind, validation_score, best_validation_score, "model saved" if model_saved else ""))
                            else:
                                model.save(save_path)
                    data_iter = train_dataset.read_train(self.batch_size)
                    epoch_ind += 1
                break
            except KeyboardInterrupt:
                if self.optimizer in ["nesterov", "sgd"]:
                    if interrupts_budget == 0:
                        raise
                    interrupts_budget -= 1
                    lr.set_value(lr.get_value() / numpy.float32(2.0))
                    print "New learning rate set {}. Interrupts remaining: {}".format(lr.get_value(), interrupts_budget)
                else:
                    raise

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

    for batch_idx, (num_samples, data_map) in enumerate(dataset.read(10000)):
        output = eval_data(data_map)[0]
        output_stream.write("\n".join(("\t".join(str(val) for val in vals)) for vals in output) + "\n")

# -----------------------------------modules-----------------------
modules_map = {}
def register_module(name, func):
    assert name not in modules_map
    modules_map[name] = func

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
    # returns 3 variables: output, dynamic params, dynamic penalized params
    # Currently, dynamic params are the result of sparse affine ops. These are
    # the parameters that are not persistent (exist only during function execution).
    # As can be seen in sgd_update, we may still want to know their values
    def train_fprop(self, inputs):
        raise NotImplementedError()
    def fprop(self, inputs):
        return self.train_fprop(inputs)[0]
    def get_params(self):
        return []
    # some parameters are not penalized, like biases in affine transformations
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
register_module("rlu", rlu)

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
register_module("sigmoid", sigmoid)

class ExpModule(Module):
    def __init__(self, input):
        Module.__init__(self, [input], Variable("dense", input.num_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        var = inputs[0]
        return T.exp(var), [], []
def exp(rng, input_variable, num_outputs=None):
    modules = []
    if not isinstance(input_variable, Variable):
        raise Exception("Unknown input: " + str(input_variable))
    if num_outputs is not None:
        num_outputs = int(num_outputs)
        modules.append(AffineModule(rng, input_variable,
            num_outputs, "exp"))
        input_variable = modules[-1].get_output()
    modules.append(ExpModule(input_variable))
    return modules
register_module("exp", exp)

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
register_module("tanh", tanh)

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
register_module("signed_sqrt", signed_sqrt)

def linear(rng, input_variable, num_outputs):
    if not isinstance(input_variable, Variable):
        raise Exception("Unknown input: " + str(input_variable))
    modules = []
    num_outputs = int(num_outputs)
    modules.append(AffineModule(rng, input_variable,
        num_outputs, "linear"))
    return modules
register_module("linear", linear)

class ConcatModule(Module):
    def __init__(self, inputs):
        num_output_features = sum([input.num_features for input in inputs])
        Module.__init__(self, inputs, Variable("dense", num_output_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == len(self.get_inputs())
        return T.concatenate(inputs, axis=1), [], []
def concat(rng, *inputs):
    if len(inputs) < 2:
        raise Exception("Concat function: nothing to concatenate")
    for input in inputs:
        if not isinstance(input, Variable):
            raise Exception("Unknown input: " + str(input))
    return [ConcatModule(inputs)]
register_module("concat", concat)

class DotModule(Module):
    def __init__(self, inputs):
        num_output_features = 1
        Module.__init__(self, inputs, Variable("dense", num_output_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 2
        return T.sum(inputs[0] * inputs[1], axis=1).dimshuffle((0, 'x')), [], []
def dot(rng, *inputs):
    if len(inputs) != 2:
        raise Exception("Wrong number of inputs to dot: {}. Should be 2".format(len(inputs)))
    for input in inputs:
        if not isinstance(input, Variable):
            raise Exception("Unknown input: " + str(input))
    return [DotModule(inputs)]
register_module("dot", dot)

class CosineDistanceModule(Module):
    def __init__(self, inputs):
        num_output_features = 1
        Module.__init__(self, inputs, Variable("dense", num_output_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 2
        return (T.sum(inputs[0] * inputs[1], axis=1).dimshuffle((0, 'x')) / (1e-10 + T.sqrt(T.sum(inputs[0] * inputs[0], axis=1))).dimshuffle((0, 'x')) / \
            (1e-10 + T.sqrt(T.sum(inputs[1] * inputs[1], axis=1))).dimshuffle((0, 'x')) ), [], []
def cosine_distance(rng, *inputs):
    if len(inputs) != 2:
        raise Exception("Wrong number of inputs to cosine_distance: {}. Should be 2".format(len(inputs)))
    for input in inputs:
        if not isinstance(input, Variable):
            raise Exception("Unknown input: " + str(input))
    return [CosineDistanceModule(inputs)]
register_module("cosine_distance", cosine_distance)

class ScaleModule(Module):
    def __init__(self, input, scaler):
        Module.__init__(self, [input], Variable(input.type, input.num_features, self))
        self.scaler = scaler
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        return inputs[0] * numpy.float32(self.scaler), [], []
def scale(rng, *inputs):
    if len(inputs) != 2:
        raise Exception("Invalid number of inputs in scale function")
    if not isinstance(inputs[0], Variable):
        raise Exception("Unknown input: " + str(input))
    try:
        float(inputs[1])
    except TypeError:
        raise Exception("Cannot parse scaling constat as float")
    return [ScaleModule(inputs[0], float(inputs[1]))]
register_module("scale", scale)

class AutoScaleModule(Module):
    def __init__(self, input):
        Module.__init__(self, [input], Variable(input.type, input.num_features, self))
        self.scaler = theano.shared(numpy.float32(0.0), name="autoscaler")
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        return T.log(numpy.float32(1.0) + T.exp(self.scaler)) / numpy.float32(0.6931471805599453) * inputs[0], [], []
    def get_params(self):
        params = [self.scaler]
        return params
    def __get_state__(self):
        data = {
            "scaler" : self.scaler.get_value()
        }
        return data
    def __set_state__(self, state):
        self.scaler.set_value(state["scaler"])
def autoscale(rng, *inputs):
    if len(inputs) != 1:
        raise Exception("Invalid number of inputs in scale function")
    if not isinstance(inputs[0], Variable):
        raise Exception("Unknown input: " + str(input))
    return [AutoScaleModule(inputs[0])]
register_module("autoscale", autoscale)

class DivideModule(Module):
    def __init__(self, input1, input2):
        Module.__init__(self, [input1, input2], Variable(input1.type, input1.num_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 2
        return inputs[0] / inputs[1], [], []
def divide(rng, *inputs):
    if len(inputs) != 2:
        raise Exception("Invalid number of inputs in divide function")
    if not isinstance(inputs[0], Variable):
        raise Exception("Unknown input: " + str(inputs[0]))
    if not isinstance(inputs[1], Variable):
        raise Exception("Unknown input: " + str(inputs[1]))
    return [DivideModule(*inputs)]
register_module("divide", divide)

class AddModule(Module):
    def __init__(self, inputs):
        Module.__init__(self, inputs, Variable(inputs[0].type, inputs[0].num_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) > 0
        res = inputs[0]
        for input in inputs[1:]:
            res += input
        return res, [], []
def add(rng, *inputs):
    if len(inputs) < 1:
        raise Exception("Invalid number of inputs in add function")
    for input in inputs:
        if not isinstance(input, Variable):
            raise Exception("Unknown input: " + str(input))
    return [AddModule(inputs)]
register_module("add", add)

class UnitNormModule(Module):
    def __init__(self, input):
        Module.__init__(self, [input], Variable(input.type, input.num_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 1
        return inputs[0] / (numpy.float32(1e-10) + T.sqrt(T.sum(inputs[0] * inputs[0], axis=1)).dimshuffle((0, 'x'))), [], []
def unit_norm(rng, *inputs):
    if len(inputs) != 1:
        raise Exception("Invalid number of inputs in unit_norm function")
    if not isinstance(inputs[0], Variable):
        raise Exception("Unknown input: " + str(input))
    return [UnitNormModule(inputs[0])]
register_module("unit_norm", unit_norm)

class SubModule(Module):
    def __init__(self, inputs):
        num_output_features = 1
        Module.__init__(self, inputs, Variable("dense", num_output_features, self))
    def train_fprop(self, inputs):
        assert len(inputs) == 2
        return inputs[0] - inputs[1], [], []
def sub(rng, *inputs):
    if len(inputs) != 2:
        raise Exception("Wrong number of inputs to dot: {}. Should be 2".format(len(inputs)))
    for input in inputs:
        if not isinstance(input, Variable):
            raise Exception("Unknown input: " + str(input))
    return [SubModule(inputs)]
register_module("sub", sub)


class GaussianEnergy(Module):
    def __init__(self, predictions, targets, weights):
        inputs = [predictions]
        if isinstance(targets, float):
            self.target = numpy.float32(targets)
        else:
            inputs.append(targets)
            self.target = None
        if weights is not None:
            self.weights = "not none"
            inputs.append(weights)
        else:
            self.weights = None
        Module.__init__(self, inputs, Variable("dense", 1, self))
    def train_fprop(self, inputs):
        predictions = inputs[0]
        means = predictions[:, 0].ravel()
        stds = predictions[:, 1].ravel()
        if self.target is not None:
            targets = self.target
        else:
            targets = inputs[1]
        energies = numpy.float32(0.5) * T.sqr((targets - means) / (numpy.float32(1e-10) + stds)) + T.log(stds + 1e-10)
        if self.weights is not None:
            weights = inputs[-1]
            return T.sum(weights * energies / T.sum(abs(weights))), [], []
        else:
            return T.mean(energies), [], []
def gaussian_energy(rng, predictions, targets, weights):
    if not isinstance(predictions, Variable):
        raise Exception("Unknown input (predictions in gaussian_energy): " + str(predictions))
    if not isinstance(targets, Variable):
        try:
            targets = float(targets)
        except ValueError:
            raise Exception("Unknown input (targets in gaussian_energy): " + str(targets))
    if weights is not None and not isinstance(weights, Variable):
        raise Exception("Unknown input: " + str(weights))
register_module("gaussian_energy", gaussian_energy)


class Mse(Module):
    def __init__(self, predictions, targets, weights):
        inputs = [predictions]
        if isinstance(targets, float):
            self.target = numpy.float32(targets)
        else:
            inputs.append(targets)
            self.target = None
        if weights is not None:
            self.weights = "not none"
            inputs.append(weights)
        else:
            self.weights = None
        Module.__init__(self, inputs, Variable("dense", 1, self))
    def train_fprop(self, inputs):
        predictions = inputs[0]
        if self.target is not None:
            targets = self.target
        else:
            targets = inputs[1]
        if self.weights is not None:
            weights = inputs[-1]
            return T.sum(weights * T.sqr(targets - predictions)) / T.sum(abs(weights)), [], []
        else:
            return T.mean(T.sqr(targets - predictions)), [], []
def mse(rng, predictions, targets, weights=None):
    if not isinstance(predictions, Variable):
        raise Exception("Unknown input (predictions in mse): " + str(predictions))
    if not isinstance(targets, Variable):
        try:
            targets = float(targets)
        except ValueError:
            raise Exception("Unknown input (targets in mse): " + str(targets))
    if weights is not None and not isinstance(weights, Variable):
        raise Exception("Unknown input: " + str(weights))
    return [Mse(predictions, targets, weights)]
register_module("mse", mse)


class CrossEntropy(Module):
    def __init__(self, predictions, targets, weights):
        inputs = [predictions]
        if isinstance(targets, float):
            self.target = numpy.float32(targets)
        else:
            inputs.append(targets)
            self.target = None
        if weights is not None:
            self.weights = "not none"
            inputs.append(weights)
        else:
            self.weights = None
        Module.__init__(self, inputs, Variable("dense", 1, self))
    def train_fprop(self, inputs):
        predictions = inputs[0]
        if self.target is not None:
            targets = self.target
        else:
            targets = inputs[1]
        if self.weights is not None:
            weights = inputs[-1]
            return -T.sum(weights * (targets * T.log(predictions + numpy.float32(1e-10)) +
                (numpy.float32(1) - targets) * T.log(numpy.float32(1.0000001) - predictions))) / T.sum(abs(weights)),[],[]
        else:
            return -T.mean((targets * T.log(predictions + numpy.float32(1e-10)) +
                (numpy.float32(1) - targets) * T.log(numpy.float32(1.0000001) - predictions))), [], []

def cross_entropy(rng, predictions, targets, weights=None):
    if not isinstance(predictions, Variable):
        raise Exception("Unknown input (predictions in cross_entropy): " + str(predictions))
    if not isinstance(targets, Variable):
        try:
            targets = float(targets)
        except ValueError:
            raise Exception("Unknown input (targets in cross_entropy): " + str(targets))
    if weights is not None and not isinstance(weights, Variable):
        raise Exception("Unknown input: " + str(weights))
    return [CrossEntropy(predictions, targets, weights)]
register_module("cross_entropy", cross_entropy)


class ErrorRate(Module):
    def __init__(self, predictions, targets, weights=None):
        inputs = [predictions]
        if isinstance(targets, float):
            self.target = numpy.float32(targets)
        else:
            inputs.append(targets)
            self.target = None
        if weights is not None:
            self.weights = "not none"
            inputs.append(weights)
        else:
            self.weights = None
        Module.__init__(self, inputs, Variable("dense", 1, self))
    def train_fprop(self, inputs):
        predictions = inputs[0]
        if self.target is not None:
            targets = self.target
        else:
            targets = inputs[1]

        # this expression assumes that there is just one output with everything
        #   above 0.5 positive (common assumption for binary classification)
        # if there are more outputs, there are more possible options - currently this is not implemented
        if self.weights is not None:
            weights = inputs[-1]
            return numpy.float32(1.0) - T.sum(weights * (targets == (predictions > numpy.float32(0.5)))) / T.sum(abs(weights)), [], []
        else:
            return numpy.float32(1.0) - T.mean(numpy.float32(1.0) * (targets == (predictions > numpy.float32(0.5)))), [], []
def error_rate(rng, predictions, targets, weights=None):
    if not isinstance(predictions, Variable):
        raise Exception("Unknown input (predictions in error_rate): " + str(predictions))
    if not isinstance(targets, Variable):
        try:
            targets = float(targets)
        except ValueError:
            raise Exception("Unknown input (targets in error_rate): " + str(targets))
    if weights is not None and not isinstance(weights, Variable):
        raise Exception("Unknown input: " + str(weights))
    return [Accuracy(predictions, targets, weights)]
register_module("error_rate", error_rate)


# --------------------------core components---------------------------------------


class FuncWrapperModule(Module):
    def __init__(self, func_module, inputs):
        output = func_module.get_output()
        Module.__init__(self, inputs, Variable(output.type, output.num_features, self))
        self.func_module = func_module
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
    def __init__(self, inputs, output, toposorted_modules, defined_inputs=None):
        assert isinstance(inputs, (list, tuple))
        # just a validity check: all real inputs are in inputs
        inputs_map = {var.name: var for var in inputs}
        intermediates = set()
        for module in toposorted_modules:
            for input in module.get_inputs():
                if not isinstance(input, Variable):
                    raise Exception("Incorrect input specified: {}".format(input))
                if input.name not in inputs_map and input not in intermediates:
                    raise Exception("Strange input: name {}, variable {}, module {}".format(input.name, input, module))
            intermediates.add(module.get_output())
        Module.__init__(self, inputs, output)
        self.defined_inputs = defined_inputs
        self.toposorted_modules = toposorted_modules
    def get_defined_inputs(self):
        return self.defined_inputs
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
    def save(self, path):
        with unique_tempdir() as tmpdir:
            tmp_model_path = os.path.join(tmpdir, "model.net")
            for i in range(100):
                try:
                    with open(tmp_model_path, "w") as f:
                        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                    for _ in range(10):
                        try:
                            shutil.move(tmp_model_path, path)
                            break
                        except Exception:
                            pass
                    # check if the model can be loaded
                    with open(path) as f:
                        pickle.load(f)
                    break
                except Exception as ex:
                    sys.stderr.write("Saving failed: " + str(ex))
                    time.sleep(1)
                    pass
        sys.stdout.write("Model saved\n")


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
        if not func_definition.endswith(")") or func_definition.count("(") != 1 or func_definition.count(")") != 1:
            raise Exception("Invalid function definition: " + func_definition)
        new_func_name, func_inputs = func_definition[:-1].split("(")
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
            if name in external_variables:
                raise Exception("Variable " + name + " specified as input to function " + new_func_name + \
                    " hides a variable or function with the same name in the outer scope. Use another name - this reduces the probability of error")
            inputs_names.add(name)

            func_inputs[idx] = Variable(name=name, type=type, num_features=num_features)
    else:
        new_func_name = func_definition
        func_inputs = []

    # function has implicit inputs that are variables from the outer scope and explicit - specified in the function definition
    defined_func_inputs = list(func_inputs)

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
                assert input_map[var_name].name == var_name
            else:
                inputs.append(input_str)
        if func_name in known_funcs:
            assert len(inputs) == len(known_funcs[func_name].get_defined_inputs())
            subfunc_inputs = []
            for input in known_funcs[func_name].get_inputs():
                if input in known_funcs[func_name].get_defined_inputs():
                    subfunc_inputs.append(inputs[0])
                    inputs = inputs[1:]
                else:
                    assert input.name is not None
                    assert input.name in external_variables
                    if input.name not in input_map:
                        assert input.name is not None
                        input_map[input.name] = input
                        func_inputs.append(input)
                    subfunc_inputs.append(input)
            modules = [FuncWrapperModule(known_funcs[func_name], subfunc_inputs)]
        elif func_name in modules_map:
            modules = modules_map[func_name](rng, *inputs)
        else:
            raise Exception("Unknown function: {}".format(func_name))
        toposorted_modules.extend(modules)
        return modules[-1].get_output()
    output = sub_make_function(function_expression)
    return new_func_name, FuncModule(func_inputs, output, toposorted_modules, defined_func_inputs)


def build_unn(input_variables, architecture_string, existing_unn=None):
    modules = architecture_string.split("|")
    var_map = {}
    for variable in input_variables:
        var_map[variable.name] = variable
    if existing_unn is not None:
        funcs = existing_unn.funcs
        for func_name in funcs:
            # support for old models with this problem
            funcs[func_name].get_output().name = func_name

            var_map[func_name] = funcs[func_name].get_output()
        modules = architecture_string.split("|")
        architecture_string = existing_unn.architecture_string + "|" + architecture_string
    else:
        funcs = {}
    rng = numpy.random.RandomState(0)
    for function_string in modules:
        func_name, func_module = make_function(function_string, funcs, rng, var_map)
        if func_name in funcs:
            raise Exception("Duplicate function definition: " + func_name)
        funcs[func_name] = func_module
        if func_name in var_map:
            raise Exception("Duplicate function definition: " + func_name)
        func_module.get_output().name = func_name
        var_map[func_name] = func_module.get_output()
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
        num_samples = dataset.arrays[0].shape[0]

        if weights_pos < 0:
            self.cumsum = range(1,num_samples + 1) * 1.0 / num_samples
        else:
            self.cumsum = numpy.cumsum(abs(dataset.arrays[weights_pos]) * 1.0 / sum(abs(dataset.arrays[weights_pos])))

        self.cumsum[-1] = 1
        self.num_batches = math.ceil(num_samples * 1.0 / batch_size)
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
            return (len(samples_ids), res)


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
            num_samples = next_pos - self.pos
            self.pos = next_pos
            return (num_samples, res)

class Dataset(object):
    def read_train(self, batch_size):
        raise NotImplementedError()
    def read(self, batch_size):
        raise NotImplementedError()

class MemoryDataset(Dataset):
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
        self.input_file = input_file
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

        num_samples = 0
        for idx, line in enumerate(self.input_file):
            num_samples += 1
            entries = line.strip("\n").split("\t")
            for input_idx, entry in enumerate(entries):
                if arrays[input_idx] is not None:
                    arrays[input_idx].append_row(entry)
            if idx + 1 >= self.batch_size:
                break
        if num_samples != 0:
            res = {}
            for var, data in zip(self.input_vars, arrays):
                if data is not None:
                    res[var.name] = data.get()
            return (num_samples, res)
        else:
            raise StopIteration()


def prepare_batch(input_vars, lines):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    arrays = []
    for input_var in input_vars:
        if input_var.type == "sparse":
            arrays.append(SparseMatrixBuilder(input_var.num_features, len(lines)))
        elif input_var.type == "dense":
            arrays.append(DenseMatrixBuilder(input_var.num_features, len(lines)))
        else:
            arrays.append(None)

    for idx, line in enumerate(lines):
        entries = line.strip("\n").split("\t")
        for input_idx, entry in enumerate(entries):
            if arrays[input_idx] is not None:
                arrays[input_idx].append_row(entry)
    res = {}
    for var, data in zip(input_vars, arrays):
        if data is not None:
            res[var.name] = data.get()
    return (len(lines), res)
class AsyncFileIterator(object):
    def __init__(self, input_file, input_vars, batch_size, num_threads=1):
        self.init(input_file, input_vars, batch_size)
        self.workers = concurrent.futures.ProcessPoolExecutor(num_threads)
    def init(self, input_file, input_vars, batch_size):
        self.input_file = input_file
        self.input_vars = input_vars
        self.batch_size = batch_size
        self.cache_size = max(1, math.ceil(1000000.0 / batch_size))

        self.batch_lines = Queue.Queue(maxsize=self.cache_size)
        self.batches = Queue.Queue(maxsize=self.cache_size)
        self.tasks = []
        self.stopped = False

        self.lines_producer = threading.Thread(target=self.read_batch_lines)
        self.lines_producer.daemon = True
        self.lines_producer.start()
        self.batch_manager = threading.Thread(target=self.manage_producers)
        self.batch_manager.daemon = True
        self.batch_manager.start()
    def reset(self, input_file, input_vars, batch_size):
        self.stopped = True
        while self.lines_producer.is_alive() or self.batch_manager.is_alive():
            time.sleep(1)
        self.init(input_file, input_vars, batch_size)
    def read_batch_lines(self):
        lines = []
        for line in self.input_file:
            if self.stopped:
                return
            lines.append(line)
            if len(lines) == self.batch_size:
                while True:
                    try:
                        self.batch_lines.put(lines, block=True, timeout=3)
                        break
                    except Queue.Full:
                        if self.stopped:
                            return
                lines = []
        if len(lines) > 0:
            while True:
                try:
                    self.batch_lines.put(lines, block=True, timeout=3)
                    break
                except Queue.Full:
                    if self.stopped:
                        return
    def manage_producers(self):
        while self.lines_producer.is_alive() or len(self.tasks) > 0 or not self.batch_lines.empty():
            active_tasks = []
            for task in self.tasks:
                if task.done():
                    while True:
                        try:
                            if not task.cancelled():
                                self.batches.put(task.result(), block=True, timeout=3)
                            break
                        except Queue.Full:
                            if self.stopped:
                                for task in self.tasks:
                                    task.cancel()
                                    return
                else:
                    active_tasks.append(task)
            while len(active_tasks) < self.cache_size and not self.batch_lines.empty():
                lines = self.batch_lines.get(block=True, timeout=3)
                try:
                    active_tasks.append(self.workers.submit(prepare_batch,
                        self.input_vars, lines))
                except Exception as ex:
                    if self.stopped:
                        for task in self.tasks:
                            task.cancel()
                            return
                        return
                    else:
                        raise
            self.tasks = active_tasks
            if self.stopped:
                for task in self.tasks:
                    task.cancel()
            time.sleep(1)
    def __iter__(self):
        return self
    def next(self):
        while True:
            if not self.batches.empty():
                return self.batches.get()
            else:
                time.sleep(1)
                if not self.batch_manager.is_alive():
                    raise StopIteration()


class AsyncFileDataset(Dataset):
    def __init__(self, input_file, input_vars, num_threads=1):
        import concurrent.futures
        self.input_file = input_file
        self.input_vars = input_vars
        self.iter = None
        self.num_threads = num_threads
    def read_train(self, batch_size):
        # only one iterator may be active
        if self.iter is not None:
            self.iter.reset(self.input_file, self.input_vars, batch_size)
        else:
            self.iter = AsyncFileIterator(self.input_file, self.input_vars, batch_size, self.num_threads)
        return self.iter
    def read(self, batch_size):
        # only one iterator may be active
        if self.iter is not None:
            self.iter.reset(self.input_file, self.input_vars, batch_size)
        else:
            self.iter = AsyncFileIterator(self.input_file, self.input_vars, batch_size, self.num_threads)
        return self.iter

class AsyncStdinDataset(Dataset):
    def __init__(self, input_vars, num_threads=1):
        import concurrent.futures
        self.input_vars = input_vars
        self.iter = None
        self.num_threads = num_threads
    def read_train(self, batch_size):
        # only one iterator may be active
        assert self.iter is None
        self.iter = AsyncFileIterator(sys.stdin, self.input_vars, batch_size, self.num_threads)
        return self.iter
    def read(self, batch_size):
        assert self.iter is None
        self.iter = AsyncFileIterator(sys.stdin, self.input_vars, batch_size, self.num_threads)
        return self.iter


class FileDataset(Dataset):
    def __init__(self, input_file, input_vars):
        self.input_file = input_file
        self.input_vars = input_vars
    def read_train(self, batch_size):
        return FileIterator(open(self.input_file), self.input_vars, batch_size)
    def read(self, batch_size):
        return FileIterator(open(self.input_file), self.input_vars, batch_size)

class StdinDataset(Dataset):
    def __init__(self, input_vars):
        self.input_vars = input_vars
    def read_train(self, batch_size):
        return FileIterator(sys.stdin, self.input_vars, batch_size)
    def read(self, batch_size):
        return FileIterator(sys.stdin, self.input_vars, batch_size)


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


def get_input_vars(inputs_specification, dataset_file=None):
    input_vars = []
    inputs_names = set()
    inputs_specs = inputs_specification.split(",")
    for spec in inputs_specs:
        entries = spec.split("@")[0].split(":")
        if len(entries) > 3:
            raise Exception("Wrong number of fields in input specification: {}. Should be: name[:type:num_features@transformer]".format(spec))
        if dataset_file is None and len(entries) != 3:
            raise Exception("Wrong number of fields in input specification: {}. Should be: name:type:num_features[@transformer]".format(spec))
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
                # was scale, changed to none so that targets or weights were not accidentaly scaled
                transform_name = "none"
            else:
                transform_name = "none"
        if transform_name not in ["none", "pca", "scale", "minmax"]:
            raise Exception("Unknown transformer name: " + transform_name)
        var.transformer_name = None if transform_name == "none" else transform_name
        if var.type == "sparse" and var.transformer_name is not None:
            raise Exception("Transformer cannot be used with a sparse input '{}'".format(input_var.name))
    return input_vars


def load_dataset_from_file(file_, input_vars, use_random_iterator=False):
    arrays = []
    for input_var in input_vars:
        if input_var.type == "sparse":
            arrays.append(SparseMatrixBuilder(input_var.num_features, 100))
        else:
            arrays.append(DenseMatrixBuilder(input_var.num_features, 100))
    if file_ is not None:
        file_ = open(file_)
    else:
        file_ = sys.stdin
    for line in file_:
        for idx, line in enumerate(sys.stdin):
            if idx % 10000 == 0:
                sys.stdout.write("\rreading sample {}".format(idx))
                sys.stdout.flush()
            entries = line.strip("\n").split("\t")
            if len(entries) != len(arrays):
                print "idx=" + str(idx)
                print "num_entries="+ str(len(entries))
                raise Exception("Wrong number of inputs in " + file_)

            for builder, entry in zip(arrays, entries):
                builder.append_row(entry)
    sys.stdout.write("\r" + " " * 100 + "\r")
    arrays = map(lambda x: x.get(), arrays)

    return MemoryDataset(arrays, input_vars, use_random_iterator)


#--------------------------------modes-----------------------------------


class Object(object):
    pass
def learn():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", default="model.net")
    parser.add_argument("-c", "--continue_learning", action="store_const", const=True,
        default=False)

    # data specification
    parser.add_argument('-f', "--train", default=None,
        help="path to the learn dataset (format: target \\t weight \\t space-separated_input [\\t space-separated_input ...]). If not specfied - read from stdin")
    parser.add_argument("--mf", dest="keep_train_in_memory", help="read train data from memory", action="store_const", const=True, default=False)
    parser.add_argument("--aff", dest="train_async_file_reader_num_threads",
        help="Experimental option: read data from file asyncroniously using this number of threads", type=int, default=0)
    parser.add_argument('-t', "--test",
        help="path to the validation dataset (format: target \\t weight \\t space-separated_input [\\t space-separated_input ...])")
    parser.add_argument("--mt", dest="keep_validation_in_memory", help="read validation data from memory", action="store_const", const=True, default=False)
    parser.add_argument("--aft", dest="test_async_file_reader_num_threads",
        help="Experimental option: read data from file asyncroniously using this number of threads", type=int, default=0)
    parser.add_argument('-i', "--inputs",
        help="Inputs names. Format: <name>[:<type>:<num_features>]@<preprocessor>,<name>[:<type>:<num_features>]@<preprocessor>. "
            "Entry 'num_features' is optional for dense input. Entries 'type', 'num_features' for dense input are optional - they will be inferred from "
            "the data (if train file is specified). Entry 'preprocessor' (valid options: pca, scale, minmax, none. Default - none) is valid only for dense input")
    parser.add_argument('--sb', "--start_batch", dest="start_batch",
        help="Start batch. In case previous run failed", type=int, default=0)

    # unn architecture
    parser.add_argument("-a", "--architecture", help="architecture of the network. Format <func>|<func>|...|<func>"
        " where <func> can be <func_name>(<input_name>:<type>:<num_features>,...)=func_expression(input_name) (use this to create shared modules)"
        " or <output_name>=func_expression. The last <func> is the default model output")
    parser.add_argument("--te", "--train_energy", dest="train_energy", help="variable to minimize by tuning the parameters", default="energy")
    parser.add_argument("--ve", "--val_energy", dest="val_energy", help="variable to use for validation", default=None)
    parser.add_argument("--save", help="Ignore validation and always save the newest model", action="store_const", const=True, default=False)

    # learning parameters
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Number of training epochs. Each epoch is one pass through the train dataset")
    parser.add_argument("--batch_size", type=int, default=30,
                        help="Batch size")
    parser.add_argument("--val_freq", type=int, default=10000,
        help="Number of batches between validations")
    parser.add_argument("--rand", dest="use_random_iterator", action="store_const",
        default=False, const=True, help="Sample train batches (otherwise they will be consumed sequentially)")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "nesterov", "adadelta"], default="nesterov")
    parser.add_argument("-l", dest="reg_lambda", type=float, default=0,
                        help="l2 regularization lambda")

    # optimizers params
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum. Valid for nesterov and sgd optimizers")
    parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=0.1,
                        help="learning rate. Valid for nesterov and sgd optimizers")
    # thanks to theano adadelta is extremely slow (reason not clear)
    parser.add_argument("--decay", type=float, default=0.95,
                        help="Valid for adadelta optimizer")
    parser.add_argument("--eps", "--eps", type=float, default=1e-5,
                        help="Valid for adadelta optimizer")

    args = parser.parse_args()

    if args.val_energy is None:
        args.val_energy = args.train_energy

    if not args.train is not None and not os.path.exists(args.train):
        raise Exception("Train file is missing")
    if args.test is None:
        print "Warning: no validation. The model will be saved each {} batches".format(args.val_freq)
    elif not os.path.exists(args.test):
        raise Exception("Validation file does not exist")

    args.inputs = "".join(args.inputs.split())
    args.architecture = "".join(args.architecture.split())
    if args.train is not None:
        model_input_vars = get_input_vars(args.inputs)
    else:
        model_input_vars = get_input_vars(args.inputs, args.train)

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
        unn_inputs = {}
        for input in unn.inputs:
            unn_inputs[input.name] = input
        for input in model_input_vars:
            assert input.name in unn_inputs
            assert input.type == unn_inputs[input.name].type
            assert input.num_features == unn_inputs[input.name].num_features
            assert not hasattr(unn_inputs[input.name], "transformer_name") or input.transformer_name == unn_inputs[input.name].transformer_name
    else:
        if os.path.exists(args.model_path):
            raise Exception("Model file exists! The old file will not be overriden")
        unn = build_unn(model_input_vars, args.architecture)

    if args.train_energy not in unn.funcs:
        raise Exception("Train energy variable is unknown")
    if args.test is not None:
        if args.val_energy not in unn.funcs:
            raise Exception("Train energy variable is unknown")
    # if unn is large, it may cause MemoryError while forking
    del unn

    print "reading train"
    assert not args.use_random_iterator or args.keep_train_in_memory
    if args.keep_train_in_memory:
        train_dataset = load_dataset_from_file(args.train, model_input_vars, args.use_random_iterator)
    elif args.train_async_file_reader_num_threads != 0:
        if args.train is not None:
            train_dataset = AsyncStdinDataset(model_input_vars, args.train_async_file_reader_num_threads)
        else:
            train_dataset = AsyncFileDataset(args.train, model_input_vars, args.train_async_file_reader_num_threads)
    else:
        if args.train is not None:
            train_dataset = StdinDataset(model_input_vars)
        else:
            train_dataset = FileDataset(args.train, model_input_vars)
    print "reading validation"
    if args.test is not None:
        if args.keep_validation_in_memory:
            validation_dataset = load_dataset_from_file(args.test, model_input_vars)
        elif args.test_async_file_reader_num_threads != 0:
            validation_dataset = AsyncFileDataset(args.test, model_input_vars, args.test_async_file_reader_num_threads)
        else:
            validation_dataset = FileDataset(args.test, model_input_vars)
    else:
        validation_dataset = None

    # reload the model
    if args.continue_learning:
        try:
            with open(args.model_path) as f:
                unn = pickle.load(f)
        except Exception as ex:
            sys.stderr.write("Cannot load the model: {}".format(ex))
            raise
    else:
        unn = build_unn(model_input_vars, args.architecture)
    train_energy = unn.get_computer(args.train_energy)
    if validation_dataset is not None:
        validation_energy = unn.get_computer(args.val_energy)
    else:
        validation_energy = None

    if args.optimizer in ["nesterov", "sgd"]:
        updater_params = Object()
        updater_params.momentum = args.momentum
        updater_params.learning_rate = args.learning_rate
    elif args.optimizer == "adadelta":
        updater_params = Object()
        updater_params.decay = args.decay
        updater_params.eps = args.eps

    trainer = Trainer(updater_params=updater_params,
        num_epochs=args.epochs, reg_lambda=args.reg_lambda, batch_size=args.batch_size,
        validation_frequency=args.val_freq, optimizer=args.optimizer)

    print "Learning with {} optimizer".format(args.optimizer)
    trainer.fit(model=unn, train_energy=train_energy,
        validation_energy=validation_energy, save_path=args.model_path,
        train_dataset=train_dataset, validation_dataset=validation_dataset,
        continue_learning=args.continue_learning, always_save=args.save, start_batch=args.start_batch)
    print "Finished"


def add_new_modules():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", "--src_model_path", dest="src_model_path", required=True)
    parser.add_argument("--dst", "--dst_model_path", dest="dst_model_path", required=True)
    parser.add_argument('-i', "--inputs",
            help="Additional inputs names. Format: <name>:<type>:<num_features>@<preprocessor>,<name>:<type>:<num_features>@<preprocessor>. "
                "Entry 'preprocessor' (valid options: pca, scale, minmax, none. Default - none) is valid only for dense input")

    # unn architecture
    parser.add_argument("-a", "--architecture", help="New operations to add to the model. Format <func>|<func>|...|<func>"
        " where <func> can be <func_name>(<input_name>:<type>:<num_features>,...)=func_expression(input_name) (use this to create shared modules)"
        " or <output_name>=func_expression)")

    args = parser.parse_args()
    assert os.path.exists(args.src_model_path)
    try:
        with open(args.src_model_path) as f:
            unn = pickle.load(f)
        print "model was loaded from path " + args.src_model_path
    except Exception as ex:
        sys.stderr.write("Cannot load the model: {}".format(ex))
        raise

    new_inputs = get_input_vars(args.inputs)
    model_input_vars = unn.inputs
    model_inputs = set()
    for input in model_input_vars:
        model_inputs.add(input.name)
    for input in new_inputs:
        if input in model_inputs:
            raise Exception("Input {} already defined".format(input.name))
        else:
            model_input_vars.append(input)

    new_unn = build_unn(model_input_vars, args.architecture, unn)
    new_unn.save(args.dst_model_path)


def apply_model_to_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-o", "--output", default=None,
        help="Which variable to output")
    parser.add_argument('-d', "--data", help="path to the dataset file. If not specified - read from stdin", default=None)
    parser.add_argument('-i', "--inputs", help="csv inputs names")
    parser.add_argument('-f', "--output_file", help="file where to put the result", required=True)
    parser.add_argument("--aff", dest="async_file_reader_num_threads",
        help="Experimental option: read data from file asyncroniously using this number of threads", type=int, default=0)
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

    if args.data is not None:
        dataset = AsyncFileDataset(args.data, inputs, num_threads=3)
    else:
        dataset = AsyncStdinDataset(inputs, num_threads=3)
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

    print "Model architecture:"
    for module in unn.architecture_string.split("|"):
        print "\t{}".format(module)
    print "Model inputs:"
    for var in unn.inputs:
        print "\tName={}; Type={}; Transformer={}; Number of features={}".format(
            var.name, var.type, var.transformer, var.num_features)


def print_help():
    print "Usage:"
    print "\tunn.py learn ... - tune the parameters to minimize the output of a model"
    print "\tunn.py apply ... - apply model to data"
    print "\tunn.py describe ... - print model architecture"
    print "\tunn.py add ... - add new operations to the architecture"


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
        elif mode == "add":
            add_new_modules()
        else:
            print "Unknown mode: {}".format(mode)
            print_help()
            sys.exit(1)

if __name__ == "__main__":
    dispatch()
