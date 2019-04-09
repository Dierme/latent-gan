import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Suppress UserWarning of TensorFlow while loading the model
import copy
import numpy as np
from datetime import datetime

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

import keras
from keras.layers import Input, Concatenate, Dense, Flatten, RepeatVector, TimeDistributed
from keras.layers import Bidirectional, GaussianNoise, BatchNormalization
from keras.layers import CuDNNLSTM as LSTM #Faster drop-in for LSTM using CuDNN on TF backend on GPU
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.utils import multi_gpu_model #For parallel gpu training

from sklearn.preprocessing import StandardScaler #For scaling of the descriptors

import shutil, zipfile, tempfile, pickle

# Custom dependencies
from molvecgen import SmilesVectorizer
from generators import CodeGenerator as DescriptorGenerator
from generators import HetSmilesGenerator
from custom_callbacks import ModelAndHistoryCheckpoint2, exp_lr_decay


class DDC:
    
    def __init__(self, **kwargs):
        '''
        # Arguments
            kwargs:
                x            : model input - np.ndarray of np.bytes_ or np.float64
                y            : model output - np.ndarray of np.bytes_
                input_scaling: flag for scaling of the input descriptors - boolean
                model_name   : model filename to load - string
                dataset_info : dataset information including name, maxlen and charset - hdf5 dataset
                noise_std    : standard deviation of the noise layer in the latent space - float
                lstm_dim     : size of LSTM RNN layers - int
                dec_layers   : number of decoder layers - int
                td_dense_dim : size of TD Dense layers inbetween the LSTM ones
                               to suppress network size - int
                batch_size   : the network's batch size - int
                codelayer_dim: dimentionality of the latent space - int
                
                
        # Examples of __init__
            To *train* a blank model with encoder (autoencoder):
                model = ddc.DDC(x              = mols,
                                y              = mols,
                                input_scaling  = True,
                                dataset_info   = info,
                                noise_std      = 0.1,
                                lstm_dim       = 256,
                                dec_layers     = 3,
                                td_dense_dim   = 0,
                                batch_size     = 128,
                                codelayer_dim  = 128)
            
            To *train* a blank model without encoder:
                model = ddc.DDC(x              = descriptors,
                                y              = mols,
                                input_scaling  = True,
                                dataset_info   = info,
                                noise_std      = 0.1,
                                lstm_dim       = 256,
                                dec_layers     = 3,
                                td_dense_dim   = 0,
                                batch_size     = 128)
                                
            To *re-train* a saved model with encoder (autoencoder):
                model = ddc.DDC(x              = mols,
                                y              = mols,
                                model_name     = saved_model_name)
            
            To *re-train* a saved model without encoder:
                model = ddc.DDC(x              = descriptors,
                                y              = mols,
                                model_name     = saved_model_name)
                
            To *test* a saved model:
                model = ddc.DDC(model_name     = saved_model_name)

        '''

        # Identify the mode to start the model in
        if "x" in kwargs and "y" in kwargs:
            x = kwargs.get("x")
            y = kwargs.get("y")
            if "model_name" not in kwargs:
                self.__mode = "train"
            else:
                self.__mode = "retrain"
        elif "model_name" in kwargs:
            self.__mode = "test"
        else:
            self.__mode = "unknown"
        
        print("Initializing model in %s mode." % self.__mode)
        
        if self.mode == "train":         
            # Infer input type from the type(x)
            if type(x[0]) == np.bytes_:
                self.__input_type = "mols" # binary RDKit mols
            else:
                self.__input_type = "descriptors" # other molecular descriptors
                if kwargs.get("scaling", False) is True:
                    # Normalize the input
                    self.__scaler = StandardScaler()
                    x = self.__scaler.fit_transform(x)
                else:
                    self.__scaler = None
            
            # Extend maxlen to avoid annoying breaks in training
            self.__maxlen = kwargs.get("dataset_info")["maxlen"] + 10

            self.__charset        = kwargs.get("dataset_info")["charset"]
            self.__dataset_name   = kwargs.get("dataset_info")["name"]
            self.__lstm_dim       = kwargs.get("lstm_dim", 256)
            self.__h_activation   = kwargs.get("h_activation", "relu")
            self.__bn_momentum    = kwargs.get("bn_momentum", 0.9)
            self.__noise_std      = kwargs.get("noise_std", 0.01)
            self.__td_dense_dim   = kwargs.get("td_dense_dim", 0) # >0 squeezes RNN connections with Dense sandwiches
            self.__batch_size     = kwargs.get("batch_size", 256)
            self.__dec_layers     = kwargs.get("dec_layers", 2)

            if self.input_type == "descriptors":
                self.__codelayer_dim = kwargs.get("x").shape[1] #TODO
                if "codelayer_dim" in kwargs:
                    print("Ignoring requested codelayer_dim because it is inferred from the cardinality of the descriptors.")
            else:
                self.__codelayer_dim = kwargs.get("codelayer_dim", 128)

            # Create the left/right-padding vectorizers
            self.__smilesvec1 = SmilesVectorizer(canonical=False, 
                                                 augment=True, 
                                                 maxlength=self.maxlen, 
                                                 charset=self.charset,
                                                 binary=True)

            self.__smilesvec2 = SmilesVectorizer(canonical=False, 
                                                 augment=True, 
                                                 maxlength=self.maxlen, 
                                                 charset=self.charset,
                                                 binary=True, 
                                                 leftpad=False)

            #self.train_gen.next() #This line is needed to set train_gen.dims (to be fixed in HetSmilesGenerator)
            self.__input_shape     = self.smilesvec1.dims
            self.__dec_dims        = list(self.smilesvec1.dims)
            self.__dec_dims[0]     = self.dec_dims[0]-1
            self.__dec_input_shape = self.dec_dims
            self.__output_len      = self.smilesvec1.dims[0]-1
            self.__output_dims     = self.smilesvec1.dims[-1]   

            # Build all sub-models as untrained models
            if self.input_type == "mols":
                self.__build_mol_to_latent_model()
            else:
                self.__mol_to_latent_model = None

            self.__build_latent_to_states_model()
            self.__build_batch_model()

            # Build data generators
            self.__build_generators(kwargs.get("x"), kwargs.get("y"))
        
        # Retrain or Test mode
        else:
            self.__model_name = kwargs.get("model_name")
            
            # Load the model
            self.__load(self.model_name)
            
            if self.mode == "retrain":
                # Build data generators
                self.__build_generators(kwargs.get("x"), kwargs.get("y"))
                
        # Build full model out of the sub-models
        self.__build_model()
        
        # Show the resulting full model
        print(self.model.summary())
    
    
    '''
    Architecture properties.
    '''
    @property
    def lstm_dim(self):
        return self.__lstm_dim
    
    @property
    def h_activation(self):
        return self.__h_activation
    
    @property
    def bn_momentum(self):
        return self.__bn_momentum
    
    @property
    def noise_std(self):
        return self.__noise_std
    
    @property
    def td_dense_dim(self):
        return self.__td_dense_dim
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    @property
    def dec_layers(self):
        return self.__dec_layers
    
    @property
    def codelayer_dim(self):
        return self.__codelayer_dim
    
    @property
    def steps_per_epoch(self):
        return self.__steps_per_epoch
    
    @property
    def validation_steps(self):
        return self.__validation_steps
    
    @property
    def input_shape(self):
        return self.__input_shape
    
    @property
    def dec_dims(self):
        return self.__dec_dims
    
    @property
    def dec_input_shape(self):
        return self.__dec_input_shape
    
    @property
    def output_len(self):
        return self.__output_len
    
    @property
    def output_dims(self):
        return self.__output_dims
    
    '''
    Models.
    '''
    @property
    def mol_to_latent_model(self):
        return self.__mol_to_latent_model
    
    @property
    def latent_to_states_model(self):
        return self.__latent_to_states_model
    
    @property
    def batch_model(self):
        return self.__batch_model
    
    @property
    def sample_model(self):
        return self.__sample_model
    
    @property
    def model(self):
        return self.__model
    
    '''
    Train properties.
    '''
    @property
    def epochs(self):
        return self.__epochs
    
    @property
    def clipvalue(self):
        return self.__clipvalue
    
    @property
    def lr(self):
        return self.__lr
    
    @property
    def h(self):
        return self.__h
    
    '''
    Other properties.
    '''
    @property
    def mode(self):
        return self.__mode
    
    @property
    def dataset_name(self):
        return self.__dataset_name
    
    @property
    def model_name(self):
        return self.__model_name
    
    @property
    def input_type(self):
        return self.__input_type
    
    @property
    def maxlen(self):
        return self.__maxlen
    
    @property
    def charset(self):
        return self.__charset
    
    @property
    def smilesvec1(self):
        return self.__smilesvec1
    
    @property
    def smilesvec2(self):
        return self.__smilesvec2
    
    @property
    def train_gen(self):
        return self.__train_gen
    
    @property
    def valid_gen(self):
        return self.__valid_gen
    
    @property
    def scaler(self):
        return self.__scaler
    
    
    '''
    Private methods.
    '''
    
    def __build_generators(self, x, y, split=0.9):
            '''
            Build data generators to be used in (re)training.
            '''
            
            # Sanity check
            assert len(x) == len(y)
            
            # Split dataset into train and validation sets
            cut = int(split * len(x))
            x_train = x[:cut]
            x_valid = x[cut:]
            y_train = y[:cut]
            y_valid = y[cut:]
            
            if self.input_type == "mols":
                self.__train_gen = HetSmilesGenerator(x_train, 
                                                      None, 
                                                      self.smilesvec1, 
                                                      self.smilesvec2, 
                                                      batch_size=self.batch_size, 
                                                      shuffle=True)

                self.__valid_gen = HetSmilesGenerator(x_valid, 
                                                      None, 
                                                      self.smilesvec1, 
                                                      self.smilesvec2, 
                                                      batch_size=self.batch_size,
                                                      shuffle=True)                    
            
            else:
                self.__train_gen = DescriptorGenerator(x_train,
                                                       y_train,
                                                       self.smilesvec1,
                                                       self.smilesvec2,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)

                self.__valid_gen = DescriptorGenerator(x_valid,
                                                       y_valid,
                                                       self.smilesvec1,
                                                       self.smilesvec2,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
            
            # Calculate number of batches per training/validation epoch
            train_samples = len(x_train)
            valid_samples = len(x_valid)
            self.__steps_per_epoch  = train_samples // self.batch_size
            self.__validation_steps  = valid_samples // self.batch_size

            print("Model received %d train samples and %d validation samples." % (train_samples,
                                                                                  valid_samples))

            
    def __build_mol_to_latent_model(self):
        '''
        Model that transforms binary molecules to their latent representation.
        Only used if input is mols.
        Other input types may be either ECFP4 or QSAR properties; in this case this model is not used.
        '''
        
        # Input tensor (MANDATORY)
        encoder_inputs = Input(shape=self.input_shape, name="Encoder_Inputs")
        
        x = encoder_inputs
        
        # The two encoder layers, number of cells are halved as Bidirectional
        encoder = Bidirectional(LSTM(self.lstm_dim//2,
                                     return_sequences=True,
                                     return_state=True, # Return the states at end of the batch
                                     name="Encoder_LSTM_1"))
        x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)
        
        x = BatchNormalization(momentum=self.bn_momentum, name="BN_1")(x)
        
        encoder2 = Bidirectional(LSTM(self.lstm_dim//2,
                   return_state=True, # Return the states at end of the batch
                   name="Encoder_LSTM_2"))
        
        encoder_outputs, state_h2, state_c2 , state_h2_reverse, state_c2_reverse = encoder2(x)
        
        # The concatenate states
        states = Concatenate(axis=-1, name="Concatenate_1")([state_h,          state_c, 
                                                             state_h2,         state_c2,
                                                             state_h_reverse,  state_c_reverse, 
                                                             state_h2_reverse, state_c2_reverse])
        
        states = BatchNormalization(momentum=self.bn_momentum, name="BN_2")(states)

        # A non-linear recombination
        neck_relu = Dense(self.codelayer_dim, activation=self.h_activation, name="Codelayer_Relu")
        neck_outputs = neck_relu(states)
        
        neck_outputs = BatchNormalization(momentum=self.bn_momentum, name="BN_Codelayer")(neck_outputs)
        
        # Add Gaussian noise to "spread" the distribution of the latent variables during training
        neck_outputs  = GaussianNoise(self.noise_std, name="Gaussian_Noise")(neck_outputs)
        
        # Define the model
        self.__mol_to_latent_model = Model(encoder_inputs, neck_outputs)
        # Name it!
        self.mol_to_latent_model.name = "mol_to_latent_model"

        
    def __build_latent_to_states_model(self):      
        '''
        Model that constructs the initial states of the decoder from a latent molecular representation.
        '''
    
        # Input tensor (MANDATORY)
        latent_input = Input(shape=(self.codelayer_dim,), name="Latent_Input")
    
        # Initialize list of state tensors for the decoder
        decoder_state_list = []
        
        for dec_layer in range(self.dec_layers):
            # The tensors for the initial states of the decoder
            name = "Dense_h_" + str(dec_layer)
            h_decoder = Dense(self.lstm_dim, activation="relu", name=name)(latent_input)
            
            name = "BN_h_" + str(dec_layer)
            decoder_state_list.append(BatchNormalization(momentum=self.bn_momentum, name=name)(h_decoder))
            
            name = "Dense_c_" + str(dec_layer)
            c_decoder = Dense(self.lstm_dim, activation="relu", name=name)(latent_input)
            
            name = "BN_c_" + str(dec_layer)
            decoder_state_list.append(BatchNormalization(momentum=self.bn_momentum, name=name)(c_decoder))
        
        # Define the model
        self.__latent_to_states_model = Model(latent_input, decoder_state_list)
        # Name it!
        self.latent_to_states_model.name = "latent_to_states_model"
        
        
    def __build_batch_model(self):
        '''
        Model that returns the output characters in batch.
        '''
        
        # List of input tensors to batch_model
        inputs = []
        
        # This is the start character padded OHE smiles for teacher forcing
        decoder_inputs = Input(shape=self.dec_input_shape, name="Decoder_Inputs")
        inputs.append(decoder_inputs)
        
        # I/O tensor of the LSTM layers
        x = decoder_inputs
        
        for dec_layer in range(self.dec_layers):
            name = "Decoder_State_h_" + str(dec_layer)
            state_h = Input(shape=[self.lstm_dim], name=name)
            inputs.append(state_h)
            
            name = "Decoder_State_c_" + str(dec_layer)
            state_c = Input(shape=[self.lstm_dim], name=name)
            inputs.append(state_c)
            
            # RNN layer
            decoder_lstm = LSTM(self.lstm_dim,
                           return_sequences=True,
                           name="Decoder_LSTM_" + str(dec_layer))
            
            x = decoder_lstm(x, initial_state=[state_h, state_c])
            x = BatchNormalization(momentum=self.bn_momentum, name="BN_Decoder_" + str(dec_layer))(x)
            
            # Squeeze LSTM interconnections using Dense layers
            if self.td_dense_dim > 0: 
                x = TimeDistributed(Dense(self.td_dense_dim), name="Time_Distributed_" + str(dec_layer))(x)
        
        # Final Dense layer to return soft labels (probabilities)
        outputs = Dense(self.output_dims, activation='softmax', name="Dense_Decoder")(x)
        
        # Define the batch_model
        self.__batch_model = Model(inputs=inputs, outputs=[outputs])
        # Name it!
        self.batch_model.name = "batch_model"
        
        
    def __build_model(self):
        '''
        Full model that constitutes the complete pipeline.
        '''
        
        # IFF input is not encoded, stack the encoder (mol_to_latent_model)
        if self.input_type == "mols":        
            # Input tensor (MANDATORY) - Same as the mol_to_latent_model input!
            encoder_inputs = Input(shape=self.input_shape, name="Encoder_Inputs")
            # Input tensor (MANDATORY) - Same as the batch_model input for teacher's forcing!
            decoder_inputs = Input(shape=self.dec_input_shape, name="Decoder_Inputs")

            # Stack the three models
            # Propagate tensors through 1st model
            x = self.mol_to_latent_model(encoder_inputs)
            # Propagate tensors through 2nd model
            x = self.latent_to_states_model(x)
            # Append the first input of the third model to be the one for teacher's forcing
            x = [decoder_inputs] + x
            # Propagate tensors through 3rd model
            x = self.batch_model(x)

            # Define full model (SMILES -> SMILES)
            self.__model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[x])
        
        # Input is pre-encoded, no need for encoder
        else:
            # Input tensor (MANDATORY)
            latent_input = Input(shape=(self.codelayer_dim,), name="Latent_Input")
            # Input tensor (MANDATORY) - Same as the batch_model input for teacher's forcing!
            decoder_inputs = Input(shape=self.dec_input_shape, name="Decoder_Inputs")
            
            # Stack the two models
            # Propagate tensors through 1st model
            x = self.latent_to_states_model(latent_input)
            # Append the first input of the 2nd model to be the one for teacher's forcing
            x = [decoder_inputs] + x
            # Propagate tensors through 2nd model
            x = self.batch_model(x)

            # Define full model (latent -> SMILES)
            self.__model = Model(inputs=[latent_input, decoder_inputs], outputs=[x])

            
    def __build_sample_model(self) -> dict:
        '''
        Model that predicts a single character of the output.
        This model is generated from the modified config file of the self.batch_model.
        
        Returns:
            The dictionary of the configuration.
        '''
        
        # Get the configuration of the batch_model
        config = self.batch_model.get_config()
        
        # Keep only the "Decoder_Inputs" as single input to the sample_model
        config["input_layers"] = [config["input_layers"][0]]
        
        # Find decoder states that are used as inputs in batch_model and remove them
        idx_list = []
        for idx, layer in enumerate(config["layers"]):
            
            if "Decoder_State_" in layer["name"]:
                idx_list.append(idx)
        
        # Pop the layer from the layer list
        # Revert indices to avoid re-arranging after deleting elements
        for idx in sorted(idx_list, reverse=True):
            config["layers"].pop(idx)
    
        # Remove inbound_nodes dependencies of remaining layers on deleted ones
        for layer in config["layers"]:
            idx_list = []

            try:
                for idx, inbound_node in enumerate(layer["inbound_nodes"][0]):
                    if "Decoder_State_" in inbound_node[0]: 
                        idx_list.append(idx)
            # Catch the exception for first layer (Decoder_Inputs) that has empty list of inbound_nodes[0]
            except:
                pass

            # Pop the inbound_nodes from the list
            # Revert indices to avoid re-arranging
            for idx in sorted(idx_list, reverse=True):
                layer["inbound_nodes"][0].pop(idx)

        # Change the batch_shape of input layer
        config["layers"][0]["config"]["batch_input_shape"] = (1, 1, self.dec_input_shape[-1])
        
        # Finally, change the statefulness of the RNN layers
        for layer in config["layers"]:
            if "Decoder_LSTM_" in layer["name"]:
                layer["config"]["stateful"] = True
                
        # Define the sample_model using the modified config file
        self.__sample_model = Model.from_config(config)
        
        # Copy the trained weights from the trained batch_model to the untrained sample_model
        for layer in self.sample_model.layers:
            # Get weights from the batch_model
            weights = self.batch_model.get_layer(layer.name).get_weights()
            # Set the weights to the sample_model
            self.sample_model.get_layer(layer.name).set_weights(weights)
            
        # Return the config for further inspection
        return config
    
    
    def __load(self, model_name):
        '''
        Load complete model from a zip file.
        To be called within __init__.
        '''
        
        print("Loading model.")
        tstart = datetime.now()

        # Temporary directory to extract the zipped information
        with tempfile.TemporaryDirectory() as dirpath:

            # Unzip the directory that contains the saved model(s)
            with zipfile.ZipFile(model_name+".zip","r") as zip_ref:
                zip_ref.extractall(dirpath)

            # Load metadata
            metadata = pickle.load(open(dirpath+"/metadata.pickle", "rb"))

            # Re-load metadata
            self.__dict__.update(metadata)

            # Load all sub-models
            try:
                self.__mol_to_latent_model = load_model(dirpath+"/mol_to_latent_model.h5")
            except:
                print("'mol_to_latent_model' not found, setting to None.")
                self.__mol_to_latent_model = None

            self.__latent_to_states_model = load_model(dirpath+"/latent_to_states_model.h5")
            self.__batch_model            = load_model(dirpath+"/batch_model.h5")
            # Build sample_model out of the trained batch_model
            self.__build_sample_model()
            
        print("Loading finished in %i seconds." %((datetime.now()-tstart).seconds))

        
    '''
    Public methods.
    '''
    
    def fit(self, model_name, epochs, lr, mini_epochs, patience, gpus=1, workers=1, use_multiprocessing=False, verbose=2, 
            max_queue_size=10, clipvalue=0, save_period=5, checkpoint_dir="/projects/cc/kjmv588/models/checkpoints/",
            lr_decay=False):
        '''
        Fit the full model to the training data.
        Supports multi-gpu training if gpus set to >1.
        
        # Arguments
            kwargs:
                model_name         : base name for the checkpoints - string
                epochs             : number of epochs to train in total - int
                lr                 : initial learning rate of the training - float
                mini_epochs        : number of dividends of an epoch (==1 means no mini_epochs) - int
                patience           : minimum consecutive mini_epochs of stagnated learning rate to consider 
                                     before lowering it - int
                gpus               : number of gpus to use for multi-gpu training (==1 means single gpu) - int
                workers            : number of CPU workers - int
                use_multiprocessing: flag for Keras multiprocessing - boolean
                verbose            : verbosity of the training - int
                max_queue_size     : max size of the generator queue - int
                clipvalue          : value of gradient clipping - float
                save_period        : mini_epochs every which to checkpoint the model - int
                checkpoint_dir     : directory to store the checkpoints - string
                lr_decay           : flag to use exponential decay of learning rate - boolean
        '''

        # Get parameter values if specified
        self.__epochs    = epochs
        self.__lr        = lr
        self.__clipvalue = clipvalue
        
        # Optimizer
        if clipvalue > 0:
            print("Using gradient clipping %.2f." % clipvalue)
            opt = Adam(lr=self.lr, clipvalue=clipvalue)
            
        else:
            opt = Adam(lr=self.lr)        
        
        # Callbacks        
        rlr = ReduceLROnPlateau(monitor="val_loss", 
                                factor=0.5, 
                                patience=patience, 
                                min_lr=1e-6, 
                                verbose=1, 
                                min_delta=1e-4)
        
        lr_scheduler = LearningRateScheduler(schedule=exp_lr_decay, verbose=1)
                        
        checkpoint_file = checkpoint_dir + "%s--{epoch:02d}--{val_loss:.4f}--{lr:.7f}" % model_name
        mhcp = ModelAndHistoryCheckpoint2(filepath=checkpoint_file,
                                         model_dict=self.__dict__,
                                         monitor="val_loss",
                                         verbose=1,
                                         mode="min",
                                         period=save_period)
        
        if lr_decay:
            callbacks = [lr_scheduler, mhcp]
        else:
            callbacks = [rlr, mhcp]
        
        # Inspect training parameters at the start of the training
        self.summary()
        
        # Parallel training on multiple GPUs
        if gpus > 1:
            parallel_model = multi_gpu_model(self.model, gpus=gpus)
            parallel_model.compile(loss="categorical_crossentropy", optimizer=opt)
            # This `fit` call will be distributed on all GPUs.
            # Each GPU will process (batch_size/gpus) samples per batch.
            parallel_model.fit_generator(self.train_gen,
                                         steps_per_epoch=self.steps_per_epoch / mini_epochs,
                                         epochs=mini_epochs * self.epochs,
                                         validation_data=self.valid_gen,
                                         validation_steps=self.validation_steps / mini_epochs,
                                         callbacks=callbacks,
                                         max_queue_size=max_queue_size,
                                         workers=workers,
                                         use_multiprocessing=use_multiprocessing,
                                         verbose=verbose) #1 to show progress bar
            
        elif gpus == 1:
            self.model.compile(loss="categorical_crossentropy", optimizer=opt)
            self.model.fit_generator(self.train_gen,
                                     steps_per_epoch=self.steps_per_epoch / mini_epochs,
                                     epochs=mini_epochs * self.epochs,
                                     validation_data=self.valid_gen,
                                     validation_steps=self.validation_steps / mini_epochs,
                                     callbacks=callbacks,
                                     max_queue_size=10,
                                     workers=workers,
                                     use_multiprocessing=use_multiprocessing,
                                     verbose=verbose) #1 to show progress bar
            
        # Build sample_model out of the trained batch_model
        self.__build_sample_model()
        
        # Training history
        self.__h = mhcp.history
    

    def vectorize(self, mols_test, leftpad=True):
        '''
        Perform One-Hot Encoding (OHE) on a binary molecule.
        '''
    
        if leftpad:
            return self.smilesvec1.transform(mols_test)
        else:
            return self.smilesvec2.transform(mols_test)
    
    
    def transform(self, mols_ohe):
        '''
        Encode a batch of OHE molecules into their latent representations.
        '''
    
        return self.mol_to_latent_model.predict(mols_ohe)

    
    def predict(self, latent, temp=0):
        '''
        Predict a *single* SMILES string from a latent representation.
        Careful, "latent" must be the output of self.transform().
        If temp>0, multinomial sampling is used instead of selecting the single most probable character at each step.
        If temp=1, multinomial sampling without temperature scaling is used.
        '''
    
        # Decode states and reset the LSTM cells with them to bias the generation towards the desired properties
        if self.scaler is not None:
            latent = self.scaler.predict(latent)
            
        states = self.latent_to_states_model.predict(latent)
        
        for dec_layer in range(self.dec_layers):
            self.sample_model.get_layer("Decoder_LSTM_" + str(dec_layer)).reset_states(states=[states[2*dec_layer],states[2*dec_layer+1]])
        
        # Prepare the input char
        startidx = self.smilesvec1._char_to_int[self.smilesvec1.startchar]
        samplevec = np.zeros((1,1,self.smilesvec1.dims[-1]))
        samplevec[0,0,startidx] = 1
        smiles = ""
        # Initialize Negative Log-Likelihood (NLL)
        NLL = 0
        # Loop and predict next char
        for i in range(1000):
            o = self.sample_model.predict(samplevec)
            # Multinomial sampling with temperature scaling
            if temp:
                temp=abs(temp) # Handle negative values
                nextCharProbs = np.log(o) / temp
                nextCharProbs = np.exp(nextCharProbs)
                nextCharProbs = nextCharProbs / nextCharProbs.sum() - 1e-8 # Re-normalize for float64 to make exactly 1.0 for np.random.multinomial
                sampleidx = np.random.multinomial(1, nextCharProbs.squeeze(), 1).argmax()               
                
            # Else, select the most probable character
            else:
                sampleidx = np.argmax(o)
            
            samplechar = self.smilesvec1._int_to_char[sampleidx]
            if samplechar != self.smilesvec1.endchar:
                smiles = smiles + self.smilesvec1._int_to_char[sampleidx]
                samplevec = np.zeros((1,1,self.smilesvec1.dims[-1]))
                samplevec[0,0,sampleidx] = 1
                # Calculate negative log likelihood for the selected character given the previous characters
                NLL -= np.log(o[0][0][sampleidx]) 
            else:
                break
        return smiles, NLL 
    

    def summary(self):
        '''
        Echo the training configuration for inspection.
        '''
        
        print("\nModel trained with dataset %s that has maxlen=%d and charset=%s for %d epochs." % (self.dataset_name,
                                                                                                    self.maxlen,
                                                                                                    self.charset,
                                                                                                    self.epochs))

        print("noise_std: %.6f, lstm_dim: %d, dec_layers: %d, td_dense_dim: %d, batch_size: %d, codelayer_dim: %d, lr: %.6f." % (self.noise_std, 
                                                                                                                                 self.lstm_dim, 
                                                                                                                                 self.dec_layers, 
                                                                                                                                 self.td_dense_dim, 
                                                                                                                                 self.batch_size, 
                                                                                                                                 self.codelayer_dim, 
                                                                                                                                 self.lr))

            
    def save(self, model_name):
        '''
        Save model in a zip file.
        '''
        
        with tempfile.TemporaryDirectory() as dirpath:
            
            # Save the Keras models
            if self.mol_to_latent_model is not None:
                self.mol_to_latent_model.save(dirpath+"/mol_to_latent_model.h5")
                
            self.latent_to_states_model.save(dirpath+"/latent_to_states_model.h5")
            self.batch_model.save(dirpath+"/batch_model.h5")
            
            # Exclude un-picklable and un-wanted attributes
            excl_attr = ["_DDC__mode", # mode is excluded because it is identified within __init__
                         "_DDC__train_gen",
                         "_DDC__valid_gen",
                         "_DDC__mol_to_latent_model",
                         "_DDC__latent_to_states_model",
                         "_DDC__batch_model",
                         "_DDC__sample_model",
                         "_DDC__model"]

            # Cannot deepcopy self.__dict__ because of Keras' thread lock so this is
            # bypassed by popping, saving and re-inserting the un-picklable attributes
            to_add = {}
            # Remove un-picklable attributes
            for attr in excl_attr:
                to_add[attr] = self.__dict__.pop(attr, None)
            
            # Pickle metadata, i.e. almost everything but the Keras models and generators
            pickle.dump(self.__dict__, open(dirpath+"/metadata.pickle", "wb"))

            # Zip directory with its contents
            shutil.make_archive(model_name, 'zip', dirpath)
            
            # Finally, re-load the popped elements
            for attr in excl_attr:
                #if attr == "_DDC__mol_to_latent_model" and to_add[attr] is None:
                self.__dict__[attr] = to_add[attr]
            
            print("Model saved.")
        