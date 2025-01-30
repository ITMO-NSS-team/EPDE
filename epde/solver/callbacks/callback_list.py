from epde.solver.callbacks.callback import Callback

# import tree

class CallbackList(Callback):
    """Container abstracting a list of callbacks."""
    def __init__(
        self,
        callbacks=None,
        model=None,
        **params,
    ):
        """Container for `Callback` instances.

        This object wraps a list of `Callback` instances, making it possible
        to call them all at once via a single endpoint
        (e.g. `callback_list.on_epoch_end(...)`).

        Args:
            callbacks: List of `Callback` instances.
            model: The `Model` these callbacks are used with.
            **params: If provided, parameters will be passed to each `Callback`
                via `Callback.set_params`.
        """
        self.callbacks = callbacks if callbacks else []

        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

    def set_model(self, model):
        super().set_model(model)
        for callback in self.callbacks:
            callback.set_model(model)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def on_epoch_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

    def on_epoch_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)
