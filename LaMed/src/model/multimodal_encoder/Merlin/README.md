# Merlin: Vision Language Foundation Model for 3D Computed Tomography

[![arXiv](https://img.shields.io/badge/arXiv-2406.06512-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2406.06512)&nbsp;&nbsp;&nbsp;&nbsp;[![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/stanfordmimi/Merlin)&nbsp;&nbsp;&nbsp;&nbsp;[![pypi](https://img.shields.io/pypi/v/merlin-vlm?style=for-the-badge)](https://pypi.org/project/merlin-vlm/)&nbsp;&nbsp;&nbsp;&nbsp;![License](https://img.shields.io/github/license/stanfordmimi/merlin?style=for-the-badge)

*Merlin is a 3D VLM for computed tomography that leverages both structured electronic health records (EHR) and unstructured radiology reports for pretraining.*

![Key Graphic](documentation/assets/overview.png)

## ⚡️ Installation

To install Merlin (Python 3.9 required), you can simply run:

```python
pip install merlin-vlm
```

For an editable installation, use the following commands to clone and install this repository.
```bash
git clone https://github.com/StanfordMIMI/Merlin.git
cd merlin
pip install -e .
```

## 🚀 Inference with Merlin

To create a Merlin model with both image and text embeddings enabled, use the following:
```python
from merlin import Merlin

model = Merlin()
```

To initialize the model with **only image embeddings** active, use:
```python
from merlin import Merlin

model = Merlin(ImageEmbedding=True)
```

#### For inference on a demo CT scan, please check out the [demo](documentation/demo.py)

#### For additional information, please read the [documentation](documentation/inference.md).

## 📎 Citation
If you find this repository useful for your work, please cite the cite the [original paper](https://arxiv.org/abs/2406.06512):

```bibtex
@article{blankemeier2024merlin,
  title={Merlin: A vision language foundation model for 3d computed tomography},
  author={Blankemeier, Louis and Cohen, Joseph Paul and Kumar, Ashwin and Van Veen, Dave and Gardezi, Syed Jamal Safdar and Paschali, Magdalini and Chen, Zhihong and Delbrouck, Jean-Benoit and Reis, Eduardo and Truyts, Cesar and others},
  journal={Research Square},
  pages={rs--3},
  year={2024}
}
```
