from autoencoder_model import AutoEncoderModel
from normality_indexes import log_cw, wae_normality_index, swae_normality_index


class CWAEFactory:

    def create(self, dataset, z_dim) -> AutoEncoderModel:
        x_dim = dataset.x_dim
        iterator = dataset.get_iterator()
        print(f'Creating CWAE {x_dim}, {z_dim}')
        return AutoEncoderModel(x_dim, z_dim, iterator, log_cw)


class WAEFactory:

    def create(self, dataset, z_dim) -> AutoEncoderModel:
        x_dim = dataset.x_dim
        iterator = dataset.get_iterator()
        print(f'Creating WAE {x_dim}, {z_dim}')
        return AutoEncoderModel(x_dim, z_dim, iterator, lambda Z: wae_normality_index(Z, z_dim))


class SWAEFactory:

    def create(self, dataset, z_dim) -> AutoEncoderModel:
        x_dim = dataset.x_dim
        iterator = dataset.get_iterator()
        print(f'Creating SWAE {x_dim}, {z_dim}')
        return AutoEncoderModel(x_dim, z_dim, iterator, lambda Z: swae_normality_index(Z, z_dim))
