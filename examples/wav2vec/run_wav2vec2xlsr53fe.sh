#!bin/bash

# Run wav2vec 2.0 XLSR-53 as feature extractor.

conda activate fairseq

PYTHONPATH=$PYTHONPATH:/media/data1/ewaldvdw/projects/fairseq

# Commit 4fed0beca64a52aa718371dc3b2cf1fd979197a4 works with the xlsr_53_56k.pt model without the "mask..._before" and "quantize_depth" changes.

python examples/wav2vec/extract_features.py

conda deactivate
