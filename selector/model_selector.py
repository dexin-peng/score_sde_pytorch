class ModelRegistry(dict):
    def __getitem__(self, key):
        if not isinstance(key, str):
            name = key.model.name
            return super().__getitem__(name)(key)
        return super().__getitem__(key)
    
    def __call__(self, config):
        return self[config]

_MODELS = ModelRegistry()

def register_model(cls=None, *, name=None):
    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls
    return _register(cls) if cls is not None else _register
