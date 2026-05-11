# Known Limitations (Public Beta)

- Not all model repos expose complete metadata; parameter/license detection may be heuristic.
- Conversion and quantization require compatible local llama.cpp tools.
- Fine-tuning backends depend on optional ML stacks and local hardware availability.
- Multi-model serving performance is hardware-dependent and not yet auto-tuned per workload.
- Some workflows require internet access (model resolution/download), while inference is offline after download.
