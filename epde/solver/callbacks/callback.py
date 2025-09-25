from abc import ABC, abstractmethod

class Callback(ABC):
    """
    Base class used to build new callbacks.
    """


    def __init__(self):
        """
        Initializes the KerasClassifierTrainer.
        
        This method sets up the trainer with default configurations, preparing it for fitting a Keras model.
        It's crucial for ensuring a clean state before training, allowing for consistent and reproducible results
        when discovering differential equations from data using neural networks.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Class Fields:
            print_every (None): Frequency of printing training metrics. Initialized to None.
            verbose (int): Verbosity level during training. Initialized to 0.
            validation_data (None): Validation data to be used during training. Initialized to None.
            _model (None): The Keras model to be trained. Initialized to None.
        """
        self.print_every = None
        self.verbose = 0
        self.validation_data = None
        self._model = None

    def set_params(self, params):
        """
        Sets the parameters of the callback.
        
        This allows customization of the callback's behavior during the equation discovery process.
        
        Args:
            params (dict): A dictionary containing the parameters to set for the callback.
        
        Returns:
            None
        """
        self.params = params

    def set_model(self, model):
        """
        Sets the model to be used by the callback during training.
        
        This allows the callback to interact with and monitor the training process of a specific model.
        The model is stored internally for later use by the callback methods.
        
        Args:
            model: The model instance to be associated with the callback.
        
        Returns:
            None.
        """
        self._model = model

    @property
    def model(self):
        """
        Gets the model associated with this callback.
        
        This allows access to the underlying model being used for equation discovery,
        enabling inspection of its architecture and parameters during the evolutionary process.
        
        Args:
            None
        
        Returns:
            str: The model.
        """
        return self._model

    def on_epoch_begin(self, logs=None):
        """
        Called at the start of each training epoch.
        
        Subclasses can override this method to implement custom logic that needs to be executed at the beginning of each epoch.
        This is useful for tasks such as adjusting learning rates, modifying training data, or performing other epoch-specific initializations.
        
        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_epoch_end(self, logs=None):
        """
        Called at the end of an epoch to perform actions that contribute to finding the best differential equation.
        
        Subclasses should override this method to implement custom logic that needs to be executed after each training epoch.
        This function is designed to be called only during the training phase. It allows for actions such as updating the population of equation candidates,
        evaluating their performance, and adjusting search parameters based on the epoch's results.
        
        Args:
            epoch: Integer, index of the epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        
        Returns:
            None
        """
        pass

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of the training process.
        
        This method is designed to be overridden by subclasses to implement custom initialization or setup procedures before the training loop starts.
        It allows users to define specific actions that should be executed once at the beginning of the training process, such as initializing variables, setting up logging, or preparing data.
        
        Args:
            logs (dict, optional): A dictionary containing information about the training process.
                Currently, no data is passed to this argument, but it may be populated in future versions. Defaults to None.
        
        Returns:
            None
        """
        pass

    def on_train_end(self, logs=None):
        """
        Called at the end of training.
        
        Subclasses should override for any actions to run after the training process is complete.
        This can be useful for tasks such as saving the discovered equation or performing final evaluations.
        
        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        
        Returns:
            None
        """
        pass

    def during_epoch(self, logs=None):
        """
        This method is called at the end of each epoch to perform any necessary updates or evaluations after the model has been trained on the entire dataset for that epoch.
        
        Args:
            logs (dict, optional): The logs returned from the previous epoch, containing information such as loss and metrics. Defaults to None.
        
        Returns:
            None
        """
        pass