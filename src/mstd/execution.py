"""One execution policy shared by CLI, model config and experiment tools."""


def execution_settings(backend=None,device=None):
    """GPU by default; a lone --device cpu chooses the NumPy reference.

    Explicit torch/cpu remains useful for deterministic-event backend checks.
    No CUDA availability probing or silent fallback occurs here.
    """
    if device=='gpu':device='cuda'
    if backend is None:backend='numpy' if device=='cpu' else 'torch'
    if device is None:device='cpu' if backend=='numpy' else 'cuda'
    if backend not in ('numpy','torch') or not isinstance(device,str):raise ValueError('invalid execution backend/device')
    if device!='cpu' and device!='cuda' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('device must be cpu, cuda or cuda:N')
    if backend=='numpy' and device!='cpu':raise ValueError('NumPy requires CPU; use torch for CUDA')
    return dict(backend=backend,device=device)
