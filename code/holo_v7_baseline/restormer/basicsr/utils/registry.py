"""Registry shim for upstream BasicSR arch files.

This tree loads architectures by scanning ``basicsr/models/archs/*_arch.py``
(see ``basicsr/models/archs/__init__.py``), so ``@ARCH_REGISTRY.register()``
only has to be a no-op that returns the class unchanged.
"""


class _NoOpRegistry:
    def register(self, name=None, **_):
        def decorator(cls):
            return cls
        return decorator

    def get(self, name):
        raise KeyError(name)


ARCH_REGISTRY = _NoOpRegistry()
