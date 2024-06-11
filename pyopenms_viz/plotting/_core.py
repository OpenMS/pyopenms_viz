
class PlotAccessor:
    """
    Make plots of MassSpec data using dataframes
    
    """

def _load_backend(backend: str) -> Any:
    pass

def _get_plot_backend(backend: str | None = None):
    
    backend_str: str = backend or "auto"
    
    if backend_str in _backends:
        return _backends[backend_str]

    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module