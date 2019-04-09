from keras.callbacks import ModelCheckpoint
import tempfile, pickle, shutil
import numpy as np

def exp_lr_decay(epoch, lr): 
    '''
    Custom learning rate decay schedule.
    '''
    epoch_to_start = 500
    last_epoch = 999
    lr_init = 1e-3
    lr_final = 1e-6

    decay_duration = last_epoch - epoch_to_start
    
    if epoch < epoch_to_start:
        return lr
    else:
        # Slope of the decay
        k = - (1/decay_duration) * np.log(lr_final / lr_init)
        
        lr = lr_init * np.exp(-k*(epoch-epoch_to_start))
        return lr
    
class ModelAndHistoryCheckpoint2(ModelCheckpoint):
    """
    Callback to save all sub-models and training history.
    """
    
    def __init__(self, 
                 filepath,
                 model_dict,
                 monitor='val_loss', 
                 verbose=0,
                 save_best_only=False, 
                 save_weights_only=False,
                 mode='auto', 
                 period=1):
        
        super().__init__(filepath,
                         monitor='val_loss', 
                         verbose=0,
                         save_best_only=False, 
                         save_weights_only=False,
                         mode='auto', 
                         period=1)
        
        self.period = period
        self.model_dict = model_dict
        
    def save_models(self, filepath):
        '''
        Save everything in a zip file.
        '''

        with tempfile.TemporaryDirectory() as dirpath:

            # Save the Keras models
            if self.model_dict["_DDC__mol_to_latent_model"] is not None:
                self.model_dict["_DDC__mol_to_latent_model"].save(dirpath+"/mol_to_latent_model.h5")

            self.model_dict["_DDC__latent_to_states_model"].save(dirpath+"/latent_to_states_model.h5")
            self.model_dict["_DDC__batch_model"].save(dirpath+"/batch_model.h5")

            # Exclude un-picklable and un-wanted attributes
            excl_attr = [ "_DDC__mode", # mode is excluded because it is defined inside __init__
                          "_DDC__train_gen",
                          "_DDC__valid_gen",
                          "_DDC__mol_to_latent_model",
                          "_DDC__latent_to_states_model",
                          "_DDC__batch_model",
                          "_DDC__sample_model",
                          "_DDC__model" ]

            # Cannot deepcopy self.__dict__ because of Keras' thread lock so this is
            # bypassed by popping, saving and re-inserting the un-picklable attributes
            to_add = {}
            # Remove un-picklable attributes
            for attr in excl_attr:
                to_add[attr] = self.model_dict.pop(attr, None)

            # Pickle metadata
            pickle.dump(self.model_dict, open(dirpath+"/metadata.pickle", "wb"))

            # Zip directory with its contents
            shutil.make_archive(filepath, 'zip', dirpath)

            # Finally, re-load the popped elements
            for attr in excl_attr:
                #if attr == "_DDC__mol_to_latent_model" and to_add[attr] is None:
                #    continue
                self.model_dict[attr] = to_add[attr]

            print("Model saved.")
            
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        # Save training history
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)    
        
        # Save model(s)
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    print("Saving weights of full model, ONLY.")
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.save_models(filepath)

class ModelAndHistoryCheckpoint(ModelCheckpoint):
    """
    Callback to save all sub-models and training history.
    """
    
    def __init__(self, 
                 filepath,
                 mol_to_latent_model,
                 latent_to_states_model,
                 batch_model,
                 metadata,
                 monitor='val_loss', 
                 verbose=0,
                 save_best_only=False, 
                 save_weights_only=False,
                 mode='auto', 
                 period=1):
        
        super().__init__(filepath,
                         monitor='val_loss', 
                         verbose=0,
                         save_best_only=False, 
                         save_weights_only=False,
                         mode='auto', 
                         period=1)
        
        self.period = period
        
        self.mol_to_latent_model = mol_to_latent_model
        self.latent_to_states_model = latent_to_states_model
        self.batch_model = batch_model
        
        self.metadata = metadata
        
    def save_models(self, filepath):
        # Create temporary directory to store everything
        with tempfile.TemporaryDirectory() as dirpath:
            
            if self.mol_to_latent_model is not None:
                # Save mol_to_latent_model
                self.mol_to_latent_model.save(dirpath+"/mol_to_latent_model.h5")

            # Save latent_to_states_model
            self.latent_to_states_model.save(dirpath+"/latent_to_states_model.h5")

            # Save the batch_model
            self.batch_model.save(dirpath+"/batch_model.h5")
                                                        
            # Add history to metadata and pickle
            self.metadata["history"] = self.history
            pickle.dump(self.metadata, open(dirpath+"/metadata.pickle", "wb"))

            # Zip directory with its contents
            shutil.make_archive(filepath, 'zip', dirpath)
            
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        # Save training history
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)    
        
        # Save model(s)
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    print("Saving weights of full model, ONLY.")
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.save_models(filepath)