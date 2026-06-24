"""Auto-imported by Python at startup (when repo root is on sys.path).

Forces all torch deserialization to map to CPU. CADDreamer's cached tensors
(the dill-pickled semantic-label maps and intermediate caches) were saved on a
CUDA device; loading them on a CPU-only box otherwise raises
'Attempting to deserialize object on a CUDA device but torch.cuda.is_available()
is False'. We make torch.load default to map_location='cpu' and patch the
low-level storage loader that dill/pickle use for bare tensors.
"""
import os

if os.environ.get("CADDREAMER_FORCE_CPU", "1") == "1":
    try:
        import torch

        if not torch.cuda.is_available():
            # 1) torch.load -> default map_location='cpu'
            _orig_load = torch.load

            def _cpu_load(*args, **kwargs):
                kwargs.setdefault("map_location", "cpu")
                return _orig_load(*args, **kwargs)

            torch.load = _cpu_load

            # 2) bare-tensor unpickling (what dill.load hits) goes through
            #    torch.storage._load_from_bytes -> torch.load(BytesIO); patched
            #    above. But also force the legacy rebuild path to CPU:
            try:
                import torch._utils as _tu
                _orig_rebuild = _tu._rebuild_tensor_v2

                def _rebuild_cpu(storage, *a, **k):
                    try:
                        if hasattr(storage, "is_cuda") and storage.is_cuda:
                            storage = storage.cpu()
                    except Exception:
                        pass
                    return _orig_rebuild(storage, *a, **k)

                _tu._rebuild_tensor_v2 = _rebuild_cpu
            except Exception:
                pass
    except Exception:
        pass
