import inspect


def get_name(obj):
    if inspect.isfunction(obj) or inspect.isclass(obj):
        return obj.__qualname__
    else:
        return repr(obj)
