import warnings

from .overrides import CONTAINER_MODULES


def get_all_nn_modules():
    import inspect
    import pkgutil
    import importlib
    import torch.nn as nn

    try:
        import torchvision
    except ImportError:
        torchvision = None
    
    try:
        import torchaudio
    except ImportError:
        torchaudio = None
    except Exception:
        print('[warning] torchaudio available, but import failed and hence torchvista cannot trace torchaudio operations.\
               If you need torchaudio tracing, run `import torchaudio` separately to debug what is wrong.')
        torchaudio = None
    
    try:
        import torchtext
    except ImportError:
        torchtext = None
    except Exception:
        print('[warning] torchtext available, but import failed and hence torchvista cannot trace torchtext operations.\
               If you need torchtext tracing, run `import torchtext` separately to debug what is wrong.')
        torchtext = None

    modules_to_scan = [nn, torchvision, torchaudio, torchtext]

    visited = set()
    module_classes = set()

    def walk_module(mod):
        if mod in visited:
            return
        visited.add(mod)

        try:
            for _, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, nn.Module):
                    module_classes.add(obj)
        except Exception:
            return  # Skip modules that can't be introspected

        # Recursively explore submodules
        if hasattr(mod, '__path__'):
            for _, subname, ispkg in pkgutil.iter_modules(mod.__path__, mod.__name__ + "."):
                try:
                    submod = importlib.import_module(subname)
                    walk_module(submod)
                except Exception:
                    continue  # skip if can't import

    for mod in modules_to_scan:
        if mod is not None:
            walk_module(mod)

    return module_classes

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    MODULES = get_all_nn_modules() - CONTAINER_MODULES

