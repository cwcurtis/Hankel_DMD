"""
    Author:
        Jay Lago, NIWC/SDSU, 2021
"""
import tensorflow as tf
import numpy as np
import scipy as sp

from tensorflow import keras
from tensorflow.keras.layers import *


class DLDMD(keras.Model):
    def __init__(self, hyp_params, **kwargs):
        super(DLDMD, self).__init__(**kwargs)

        # Parameters
        self.batch_size = hyp_params['batch_size']
        self.phys_dim = hyp_params['phys_dim']
        self.latent_dim = hyp_params['latent_dim']
        self.num_time_steps = int(hyp_params['num_time_steps'])
        self.num_pred_steps = int(hyp_params['num_pred_steps'])
        self.time_final = hyp_params['time_final']
        self.num_en_layers = hyp_params['num_en_layers']
        self.num_neurons = hyp_params['num_en_neurons']
        self.delta_t = hyp_params['delta_t']
        self.precision = hyp_params['precision']

        self.num_observables = hyp_params['num_observables']
        self.threshhold = hyp_params['threshhold']
        self.observation_dimension = hyp_params['observation_dimension']
        self.window = self.num_time_steps - (self.num_observables - 1)

        self.enc_input = (self.num_time_steps, self.phys_dim)
        self.dec_input = (self.window-1, self.latent_dim)

        if self.precision == 'float32':
            self.precision_complex = tf.complex64
        else:
            self.precision_complex = tf.complex128

        # Construct the ENCODER network
        self.encoder = keras.Sequential(name="encoder")
        self.encoder.add(Dense(self.num_neurons,
                               input_shape=self.enc_input,
                               activation=hyp_params['hidden_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='enc_in'))
        for ii in range(self.num_en_layers):
            self.encoder.add(Dense(self.num_neurons,
                                   activation=hyp_params['hidden_activation'],
                                   kernel_initializer=hyp_params['kernel_init_enc'],
                                   bias_initializer=hyp_params['bias_initializer'],
                                   trainable=True, name='enc_' + str(ii)))
        self.encoder.add(Dense(self.latent_dim,
                               activation=hyp_params['ae_output_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='enc_out'))

        # Construct the DECODER network
        self.decoder = keras.Sequential(name="decoder")
        self.decoder.add(Dense(self.num_neurons,
                               input_shape=self.dec_input,
                               activation=hyp_params['hidden_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='dec_in'))
        for ii in range(self.num_en_layers):
            self.decoder.add(Dense(self.num_neurons,
                                   activation=hyp_params['hidden_activation'],
                                   kernel_initializer=hyp_params['kernel_init_dec'],
                                   bias_initializer=hyp_params['bias_initializer'],
                                   trainable=True, name='dec_' + str(ii)))
        self.decoder.add(Dense(self.phys_dim,
                               activation=hyp_params['ae_output_activation'],
                               kernel_initializer=hyp_params['kernel_init_dec'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='dec_out'))

    def call(self, x):
        # Encode the entire time series
        y = self.encoder(x)
        x_ae = self.decoder(y[:, :(self.window-1), :])

        # Reshape for DMD step
        yt = tf.transpose(y, [0, 2, 1])
        y_adv, evals, evecs, phi, dmdloss = self.hankel_dmd(yt)

        # Decode the latent trajectories
        x_adv = self.decoder(y_adv)

        # Model weights
        weights = self.trainable_weights

        return [y, x_ae, x_adv, y_adv, weights, evals, evecs, phi, dmdloss]

    def edmd(self, Y):
        Y_m = Y[:, :, :-1]
        Y_p = Y[:, :, 1:]

        sig, U, V = tf.linalg.svd(Y_m, compute_uv=True, full_matrices=False)
        sigr_inv = tf.linalg.diag(1.0 / sig)
        Uh = tf.linalg.adjoint(U)

        A = Y_p @ V @ sigr_inv @ Uh
        evals, evecs = tf.linalg.eig(A)
        phi = tf.linalg.solve(evecs, tf.cast(Y_m, dtype=self.precision_complex))
        y0 = phi[:, :, 0]
        y0 = y0[:, :, tf.newaxis]

        recon = tf.TensorArray(self.precision_complex, size=self.num_pred_steps)
        recon = recon.write(0, evecs @ y0)
        evals_k = tf.identity(evals)
        for ii in tf.range(1, self.num_pred_steps):
            tmp = evecs @ (tf.linalg.diag(evals_k) @ y0)
            recon = recon.write(ii, tmp)
            evals_k = evals_k * evals
        recon = tf.math.real(tf.transpose(tf.squeeze(recon.stack()), perm=[1, 0, 2]))
        return recon, evals, evecs, phi

    def hankel_dmd(self, Y):
        winsize = self.window
        nobs = self.num_observables
        # Perform DMD method.  Note, we need to be careful about how we break the concantenated Hankel matrix apart.

        gm = tf.Variable(tf.zeros([self.num_observables*self.phys_dim, self.batch_size * (self.window - 1)], dtype=self.precision))
        gp = tf.Variable(tf.zeros([self.num_observables*self.phys_dim, self.batch_size * (self.window - 1)], dtype=self.precision))

        for jj in range(self.phys_dim):
            Yobserved = (tf.squeeze(Y[:, jj, :])).numpy()
            for ll in range(self.batch_size):
                tseries = Yobserved[ll, :]
                tcol = tseries[:nobs]
                trow = tseries[(nobs - 1):]
                hmat = np.flipud(sp.linalg.toeplitz(tcol[::-1], trow))

                gm[jj*nobs:(jj+1)*nobs, ll * (winsize - 1):(ll + 1) * (winsize - 1)].assign(hmat[:, :-1])
                gp[jj*nobs:(jj+1)*nobs, ll * (winsize - 1):(ll + 1) * (winsize - 1)].assign(hmat[:, 1:])

        sig, U, V = tf.linalg.svd(gm, compute_uv=True, full_matrices=False)
        sigr_inv = tf.linalg.diag(1.0 / sig)
        Uh = tf.linalg.adjoint(U)
        A = gp @ V @ sigr_inv @ Uh
        evals, evecs = tf.linalg.eig(A)
        phi = tf.linalg.solve(evecs, tf.cast(gm, dtype=self.precision_complex))

        gpV = gp @ V
        gpVVh = gpV @ tf.linalg.adjoint(V)
        dmdloss = tf.norm(gp - gpVVh, ord='fro', axis=[-2, -1])/tf.math.sqrt(tf.cast(self.batch_size*(self.window-1), dtype=self.precision))

        # Build reconstruction
        phiinit = phi[:, ::(self.window-1)]
        initconds = tf.cast(tf.transpose(tf.squeeze(Y[:, :, 0])), dtype=self.precision_complex)
        sigp, Up, Vp = tf.linalg.svd(phiinit, compute_uv=True, full_matrices=False)
        sigp_inv = tf.cast(tf.linalg.diag(1.0 / sigp), dtype=self.precision_complex)
        kmat = initconds @ Vp @ sigp_inv @ tf.linalg.adjoint(Up)
        recon = tf.reshape(tf.transpose(tf.math.real(kmat @ phi)), [self.batch_size, self.window-1, self.phys_dim])
        return recon, evals, evecs, phi, dmdloss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'encoder': self.encoder,
                'decoder': self.decoder}
