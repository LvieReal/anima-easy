class AnimaError(Exception):
    pass


class ValidationError(AnimaError):
    pass


class StorageError(AnimaError):
    pass
