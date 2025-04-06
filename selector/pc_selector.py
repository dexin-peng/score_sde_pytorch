class PredictorRegistry(dict):
    def __getitem__(self, key):
        if not isinstance(key, str):
            name = key.sampling.predictor
            return super().__getitem__(name)(key)
        return super().__getitem__(key)
    
    def __call__(self, config):
        return self[config]

class CorrectorRegistry(dict):
    def __getitem__(self, key):
        if not isinstance(key, str):
            name = key.sampling.corrector
            return super().__getitem__(name)(key)
        return super().__getitem__(key)
    
    def __call__(self, config):
        return self[config]

_PREDICTORS = PredictorRegistry()
_CORRECTORS = CorrectorRegistry()

def register_predictor(cls=None, *, name=None):
    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered predictor with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls
    return _register(cls) if cls is not None else _register

def register_corrector(cls=None, *, name=None):
    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered corrector with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls
    return _register(cls) if cls is not None else _register
