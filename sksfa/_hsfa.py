import numpy as np
from tqdm import tqdm
from sksfa.utils import ReceptiveRebuilder, ReceptiveSlicer
from sklearn.preprocessing import PolynomialFeatures
from sksfa import SFA
from time import time

class Flatten:
    def fit(self, X, y=None):
        pass

    def partial_fit(self, X, y=None):
        pass

    def transform(self, X):
        return X.reshape((X.shape[0], -1))

class AdditiveNoise:
    def __init__(self, std=0.01):
        self.std = std

    def fit(self, X, y=None):
        pass

    def partial_fit(self, X, y=None):
        pass

    def transform(self, X):
        return X + np.random.normal(0.0, self.std, X.shape)

class Clipper:
    def __init__(self, val_min=-4, val_max=+4):
        self.val_min = val_min
        self.val_max = val_max

    def fit(self, X, y=None):
        pass

    def partial_fit(self, X, y=None):
        pass

    def transform(self, X):
        return np.clip(X, self.val_min, self.val_max)


class HSFA:
    def __init__(self, n_components, input_shape, layer_configurations, internal_batch_size=50, noise_std=0.05, verbose=False):
        """ Layer configurations need to contain: (field_w, field_h, stride_w, stride_h, n_components) """
        self.layer_configurations = layer_configurations
        self.verbose = verbose
        self.input_shape = input_shape
        self.internal_batch_size = internal_batch_size
        self.n_components = n_components
        self.noise_std = noise_std
        self.sequence = []
        self.layer_outputs = []
        self.initialize_layers()

    def initialize_layers(self):
        # First layer does not need reconstructor
        field_w, field_h, stride_w, stride_h, n_components = self.layer_configurations[0]
        try:
            slicer = ReceptiveSlicer(input_shape=self.input_shape, field_size=(field_w, field_h), strides=(stride_w, stride_h))
        except AssertionError:
            raise ValueError(f"Layer 1: Field ({field_w}, {field_h}) with stride ({stride_w}, {stride_h}) does not fit data dimension ({self.input_shape[0]}, {self.input_shape[1]})")
        self.sequence.append(slicer)
        sfa = SFA(n_components, batch_size=self.internal_batch_size)
        self.sequence.append(sfa)
        reconstructor = ReceptiveRebuilder((slicer.reconstruction_shape))
        if self.verbose:
            print(slicer.reconstruction_shape)
        self.layer_outputs.append(slicer.reconstruction_shape)
        self.sequence.append(reconstructor)
        for build_idx, (field_w, field_h, stride_w, stride_h, n_components) in enumerate(self.layer_configurations[1:]):
            if (field_w == field_h == -1):
                field_w = slicer.reconstruction_shape[0]
                field_h = slicer.reconstruction_shape[1]
            try:
                slicer = ReceptiveSlicer(input_shape=slicer.reconstruction_shape, field_size=(field_w, field_h), strides=(stride_w, stride_h))
            except AssertionError:
                raise ValueError(f"Layer {2 + build_idx}: Field ({field_w}, {field_h}) with stride ({stride_w}, {stride_h}) does not fit data dimension ({slicer.reconstruction_shape[0]}, {slicer.reconstruction_shape[1]})")
            if self.verbose:
                print(slicer.reconstruction_shape)
            self.layer_outputs.append(slicer.reconstruction_shape)
            self.sequence.append(slicer)
            pre_expansion_sfa = SFA(n_components, batch_size=self.internal_batch_size, fill_mode=None)
            self.sequence.append(pre_expansion_sfa)
            expansion = PolynomialFeatures(2)
            expansion.partial_fit = expansion.fit
            self.sequence.append(expansion)
            self.sequence.append(AdditiveNoise(self.noise_std))
            post_expansion_sfa = SFA(n_components, batch_size=self.internal_batch_size, fill_mode=None)
            self.sequence.append(post_expansion_sfa)
            self.sequence.append(Clipper(-4, 4))
            reconstructor = ReceptiveRebuilder((slicer.reconstruction_shape))
            self.sequence.append(reconstructor)
        self.sequence.append(Flatten())
        pre_expansion_sfa = SFA(n_components, batch_size=self.internal_batch_size, fill_mode=None)
        self.sequence.append(pre_expansion_sfa)
        expansion = PolynomialFeatures(2)
        expansion.partial_fit = expansion.fit
        self.sequence.append(expansion)
        self.sequence.append(AdditiveNoise(self.noise_std))
        post_expansion_sfa = SFA(self.n_components, batch_size=self.internal_batch_size, fill_mode=None)
        if self.verbose:
            print((self.n_components,))
        self.sequence.append(post_expansion_sfa)
        self.sequence.append(Clipper(-4, 4))

    def fit(self, X):
        X = np.copy(X)
        n_samples = X.shape[0]
        batch_size = self.internal_batch_size
        n_batches = int(np.ceil(n_samples / batch_size))
        accumulating_indices = [idx for idx, member in enumerate(self.sequence) if type(member) == SFA]
        accumulating_indices += [len(self.sequence)]
        last_idx = -1
        for idx in tqdm(accumulating_indices):
            transform_only = self.sequence[:last_idx+1]
            partial_sequence = self.sequence[last_idx+1:idx]
            for batch_idx in range(n_batches):
                current_batch = X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                for member in transform_only:
                    current_batch = member.transform(current_batch)
                for member in partial_sequence:
                    member.partial_fit(current_batch)
                    current_batch = member.transform(current_batch)
                if idx < len(self.sequence):
                    self.sequence[idx].partial_fit(current_batch)
            last_idx = idx
        return self

    def transform(self, X, seq_end=None):
        n_samples = X.shape[0]
        batch_size = self.internal_batch_size
        n_batches = int(np.ceil(n_samples / batch_size))
        result = None
        sequence = self.sequence if seq_end is None else self.sequence[:seq_end]
        for batch_idx in tqdm(range(n_batches)):
            current_batch = X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            for transformer in sequence:
                if type(transformer) == AdditiveNoise:
                    continue
                current_batch = transformer.transform(current_batch)
            if result is None:
                result = np.empty((n_samples,) + current_batch.shape[1:])
            result[batch_idx * batch_size: (batch_idx + 1) * batch_size] = current_batch
        return result

    def summary(self):
        print()
        print(" = = = = NETWORK ARCHITECTURE = = = = ")
        print()
        print("Input Layer:")
        print(f"\tinput shape: \t\t{self.input_shape}")
        for layer_idx, (field_w, field_h, stride_w, stride_h, n_components) in enumerate(self.layer_configurations):
            print(f"Layer {layer_idx + 1}:")
            print(f"\treceptive field: \t({field_w}, {field_h})\n\tstrides: \t\t({stride_w}, {stride_h})")
            output_shape = self.layer_outputs[layer_idx]
            print(f"\toutput shape: \t\t{output_shape + (n_components,)}")
        print(f"Final Layer:")
        print("\tfully connected")
        print(f"\toutput shape \t\t({self.n_components},)")
        print()
        print()


