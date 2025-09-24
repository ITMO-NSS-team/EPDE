from epde.solver.callbacks.callback import Callback

# import tree

class CallbackList(Callback):
    """
    Container abstracting a list of callbacks.
    """

    def __init__(
        self,
        callbacks=None,
        model=None,
        **params,
    ):
        """
        Initializes a container for managing and invoking a set of callbacks.
        
        This class streamlines the execution of multiple callbacks during
        different stages of a process (e.g., training epochs). It aggregates
        a list of `Callback` instances, enabling their collective invocation
        through a single method call (e.g., `callback_list.on_epoch_end(...)`).
        This ensures that all relevant actions are performed at each stage,
        simplifying the management of complex workflows.
        
        Args:
            callbacks: A list of `Callback` instances to be managed. Defaults to an empty list if None is provided.
            model: The associated `Model` instance that these callbacks will interact with.
            **params: Optional keyword arguments containing parameters to be passed to each `Callback` instance via the `set_params` method.
        
        Returns:
            None
        """
        self.callbacks = callbacks if callbacks else []

        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

    def set_model(self, model):
        """
        Sets the Keras model for the callback and its children.
        
        This method propagates the Keras model instance to the callback and all its children,
        ensuring that all callbacks have access to the model for tasks such as monitoring
        training progress, modifying training behavior, or accessing model internals.
        
        Args:
            model: The Keras model instance.
        
        Returns:
            None.
        """
        super().set_model(model)
        for callback in self.callbacks:
            callback.set_model(model)

    def append(self, callback):
        """
        Appends a callback function to the list of callbacks.
        
        This allows to execute custom code during the equation discovery process, for example, to track progress or modify search parameters.
        
        Args:
            callback: The callback function to append.
        
        Returns:
            None.
        """
        self.callbacks.append(callback)

    def set_params(self, params):
        """
        Sets the parameters for the object and propagates them to the individual callbacks.
        
        This ensures that all callbacks have access to the same set of parameters,
        allowing them to consistently tailor their behavior during the equation discovery process.
        
        Args:
            params (dict): The parameters to set. These parameters might control aspects
                           of the evolutionary search, data preprocessing, or equation evaluation.
        
        Returns:
            None.
        
        Class fields (object properties) that are initialized:
            params (dict): The parameters of the object.
        """
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def on_epoch_begin(self, logs=None):
        """
        Called at the beginning of an epoch.
        
        This method iterates through the registered callbacks and calls their `on_epoch_begin` methods, allowing each callback to perform actions at the start of each epoch, such as resetting metrics or updating internal states. This ensures that all necessary preparations are made before the training epoch begins.
        
        Args:
            logs: Log data.
        
        Returns:
            None
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

    def on_epoch_end(self, logs=None):
        """
        Called at the end of an epoch to notify callbacks.
        
        This method iterates through the registered callbacks and calls their `on_epoch_end` methods,
        allowing them to perform actions or record information at the end of each epoch. This is
        crucial for tasks such as tracking metrics, adjusting hyperparameters, or saving model
        checkpoints during the training process.
        
        Args:
            logs (dict, optional): Metric results for this epoch. Defaults to None.
        
        Returns:
            None
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(logs)

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.
        
        This method is called to signal the start of the training process,
        allowing callbacks to prepare or initialize any resources they need
        before the training loop begins. It iterates through the list of callbacks
        and calls their respective `on_train_begin` methods.
        
        Args:
            logs (dict, optional): Dictionary of logs. Defaults to None.
        
        Returns:
            None
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """
        Propagates the `on_train_end` event to all callbacks in the list.
        
        This method is called at the very end of the training process. It ensures that each callback has the opportunity to perform any final actions or cleanup operations, such as logging final results or saving model parameters.
        
        Args:
            logs (dict, optional): Dictionary of logs collected during training. Defaults to None.
        
        Returns:
            None
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)
