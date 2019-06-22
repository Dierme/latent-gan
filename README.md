# Latent-GAN

An implementation of the Latent-GAN from the [publication](https://chemrxiv.org/s/29e0d04638865e1e0aff)

This project is done by Oleksii Prykhodko and Simon Johansson at AstraZeneca under the supervision of 
Hongming Chen.

The extensive documentation and the _requirements.txt_ are coming soon.

**Note!** currently the project does not containg the autoencoder because
we used one from currently unpublished work. But the Latent-gan can work with any autoencoder.
To run the project now you would have to tinker the _encode.py_ and the _decode.py_ a bit
to match the interface of the of your autoencoder.
A stub and an interface will be added soon to give an example of autencoder and how it's used. 

## Installation
todo
## Requirements
todo
## Description

### Workflow
To begin working with Latent-GAN you should have a trained autoencoder
and a SMILES file. Put the autoencoder in the _autoencoder_ folder and the SMILES file into the
_data_ folder.

First, encode SMILES using the _encode.py_. That will give you a file with
 latent vectors. Now you are ready to train a GAN.
 
Use _create_model.py_ to create and save the model. After that run _train_model.py_
providing the latent vector path and the model path. When model is trained the
script will sample default amount of latent vectors. However, you may sample more using
the _sample.py_

Once you have a file with generated latent vectors, you can decode it to SMILES using the
_decode.py_ script.

