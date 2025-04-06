_SDEs = {}

def register_sde(cls=None, *, name=None):
    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _SDEs:
            raise ValueError(f'Already registered SDE with name: {local_name}')
        _SDEs[local_name] = cls
        return cls
    return _register(cls) if cls is not None else _register
