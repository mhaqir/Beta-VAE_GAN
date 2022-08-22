### Disentangled Representation Learning Using ($\beta$)-VAE and GAN
See the paper at https://arxiv.org/pdf/2208.04549.pdf.

The first thing to train is beta-VAE, the following command can be used for this:

```python
python main_VAE.py <z_dim>  <lr>  <ne>  <beta>  <tr>  <inference>  <batch_size>
```

z_dim: number of latent space <br>
lr: learning rate <br>
ne: number of epochs <br>
beta: beta value <br>
tr: threshold on the x-axis position <br>
inference: yes/no depending if it inference mode or not <br>
batch_size: size of the input batch <br>


A model with a name containing these parameters will be saved in the model_VAE folder. In the next step the GAN module will be trained using the following command:

```python
python main_GAN.py <z_dim>  <lr>  <ne>  <beta>  <tr>  <inference>  <batch_size>
```

The arguments in this command are like the ones for VAE, and need to be the same for the pretrained VAE model. For training each of the models, there are a few
directories at the beginning of the file that should be set and are self explanatory.


After training the network, the same command can be used in inference mode.

Also dSprite dataset needs to be dwnloaded: https://github.com/deepmind/dsprites-dataset
