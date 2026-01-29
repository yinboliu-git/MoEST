---
license: cc-by-nc-nd-4.0
language:
- en
tags:
- histology
- pathology
- vision
- pytorch
- self-supervised
- vit
extra_gated_prompt: >-
  This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution.
  Any commercial use, sale, or other monetization of the UNI model and its derivatives, which include models trained on outputs from the UNI model or datasets created from the UNI model, is prohibited and requires prior approval.
  Please note that the primary email used to sign up for your Hugging Face account must match your institutional email to receive approval. By downloading the model, you attest that all information (affiliation, research use) is correct and up-to-date.
  Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use.
  By downloading this model, you agree not to distribute, publish or reproduce a copy of the model.
  If another user within your organization wishes to use the UNI model, they must register as an individual user and agree to comply with the terms of use.
  Users may not attempt to re-identify the deidentified data used to develop the underlying model.
  If you are a commercial entity, please contact the corresponding author.
extra_gated_fields:
  Full name (first and last): text
  Current affiliation (no abbreviations): text
  Type of Affiliation:
    type: select
    options: 
      - Academia
      - Industry
      - label: Other
        value: other
  Current and official institutional email (**this must match your primary email in your Hugging Face account, @gmail/@hotmail/@qq email domains will be denied**): text
  Please explain your intended research use: text
  I agree to all terms outlined above: checkbox
  I agree to use this model for non-commercial, academic purposes only: checkbox
  I agree not to distribute the model, if another user within your organization wishes to use the UNI model, they must register as an individual user: checkbox
metrics:
- accuracy
pipeline_tag: image-feature-extraction
library_name: timm
---
# Updates:
UNI2, a successor to UNI, trained on over 200 million images from over 350k diverse H&E and IHC slides has been released! Model weights and instructions are available at: \[[Huggingface Repo](https://huggingface.co/MahmoodLab/UNI2-h)\]



# Model Card for UNI
\[[Journal Link](https://www.nature.com/articles/s41591-024-02857-3)\] | \[[Open Access Read Link](https://rdcu.be/dBMgh)\] | \[[Github Repo](https://github.com/mahmoodlab/uni)\] | \[[Cite](#bibtex)\] 


## What is UNI?

UNI is the largest pretrained vision encoder for histopathology (100M images, 100K WSIs) _**developed on internal neoplastic, infectious, inflamatory and normal tissue and also made publicly available**_. We show state-of-the-art performance across 34 clinical tasks, with strong performance gains on rare and underrepresented cancer types.
 - _**Why use UNI?**_: UNI does not use open datasets and large public histology slide collections (TCGA, CPTAC, PAIP, CAMELYON, PANDA, and others in TCIA) for pretraining, which are routinely used in benchmark development in computational pathology. We make UNI available for the research community in building and evaluating pathology AI models without risk of data contamination on public benchmarks or private histopathology slide collections.

![](https://huggingface.co/MahmoodLab/UNI/resolve/main/uni.jpg)

## Requesting Access

As mentioned in the gated prompt, you must agree to the outlined terms of use, _**with the primary email for your HuggingFace account matching your institutional email**_. If your primary email is a personal email (@gmail/@hotmail/@qq) **your request will be denied**. To fix this, you can: (1) add your official institutional email to your HF account, and confirm your email address to verify, and (2) set your institutional email as your primary email in your HF account. Other reasons for your request access being denied include other mistakes in the form submitted, for example: full name includes abbreviations, affiliation is not spelled out, the described research use is not sufficient, or email domain address not recognized.

![](https://huggingface.co/MahmoodLab/UNI/resolve/main/requesting_access.png)

## Model Description

- **Developed by:** Mahmood Lab AI for Pathology @ Harvard/BWH
- **Model type:** Pretrained vision backbone (ViT-L/16 via DINOv2) for multi-purpose evaluation on histopathology images
- **Pretraining dataset:** Mass-100K, sourced from private histology collections (BWH / MGH), in addition to slides from the public GTEx consortium.
- **Repository:** https://github.com/mahmoodlab/UNI
- **Paper:** https://www.nature.com/articles/s41591-024-02857-3
- **License:** CC-BY-NC-ND-4.0

### How To Use (Feature Extraction)

Following authentication (using ```huggingface_hub```), the ViT-L/16 model architecture with pretrained weights and image transforms for UNI can be directly loaded using the [timm](https://huggingface.co/docs/hub/en/timm) library. This method automatically downloads the model weights to the [huggingface_hub cache](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) in your home directory (```~/.cache/huggingface/hub/models--MahmoodLab--UNI```), which ```timm``` will automatically find when using the commands below:

```python
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# pretrained=True needed to load UNI weights (and download weights for the first time)
# init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()
```

You can also download the model weights to a specified checkpoint location in your local directory. The ```timm``` library is still used for defining the ViT-L/16 model architecture. Pretrained weights and image transforms for UNI need to be manually loaded and defined.
```python
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model.eval()
```

You can use the UNI pretrained encoder to extract features from histopathology ROIs, as follows:

```python
from PIL import Image
image = Image.open("uni.jpg")
image = transform(image).unsqueeze(dim=0) # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
with torch.inference_mode():
    feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,1024]
```

These pre-extracted features can then be used ROI classification (via linear probing), slide classification (via multiple instance learning), and other machine learning settings.




### Direct Use (with Pre-Extracted and Frozen Features)

The models can be used without fine-tuning to obtain competitive results on:
- ROI classification, with logistic regression classifiers applied on the class token.
- ROI classification, with k-nearest neighbors (k-NN) classifiers applied on the class token.
- ROI classification, with nearest centroid classifiers (SimpleShot) applied on the class token.
- ROI retrieval, using nearest neighbors classifiers
- slide classification, with multiple instance learning (MIL) classifiers applied on a bag of class tokens extracted from the WSI

### Downstream Use (Finetuning)

It is also possible to perform fine-tuning on the models, and recommended for competitive performance on segmentation tasks. We recommend finetuning using frameworks specialized for adapting ViTs for dense prediction tasks, such as ViTDet or ViT-Adapter (which depends on Mask2Former).

## Training Details

- **Training data:** Mass-100K, a pretraining dataset (sourced from MGH, BWH, and GTEx) composed of 75,832,905  [256×256] and 24,297,995 [512×512] histology images at 20× resolution, sampled from 100,402 H&E WSIs (100,130,900 images in total).
- **Training regime:** fp16 using PyTorch-FSDP mixed-precision.
- **Training objective:** DINOv2 SSL recipe with the following losses:
  - DINO self-distillation loss with multi-crop
  - iBOT masked-image modeling loss
  - KoLeo regularization on [CLS] tokens
- **Training length:** 125,000 iterations with a batch size of 3072
- **Model architecture:** ViT-Large (0.3B params): Patch size 16, embedding dimension 1024, 16 heads, MLP FFN
- **Hardware used:** 4x8 Nvidia A100 80GB
- **Hours trained:** Approx 1024 GPU hours (32 hours total)
- **Cloud provider:** MGB ERIS Research Computing Core

## Software Dependencies

**Python Packages**
- torch>=2.0: https://pytorch.org
- xformers>=0.0.18: https://github.com/facebookresearch/xformers
- timm>=0.9.8: https://github.com/huggingface/pytorch-image-models

**Repositories**
- DINOv2 (self-supervised learning): https://github.com/facebookresearch/dinov2
- CLAM (slide classification): https://github.com/mahmoodlab/CLAM
- Mask2Former (cell and tissue segmentation): https://github.com/facebookresearch/Mask2Former
- ViT-Adapter (cell and tissue segmentation): https://github.com/czczup/ViT-Adapter
- LGSSL (Linear Probe & Few-Shot Eval): https://github.com/mbanani/lgssl

## License and Terms of Use
This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the UNI model and its derivatives, which include models trained on outputs from the UNI model or datasets created from the UNI model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the UNI model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author.


## Contact
For any additional questions or comments, contact Faisal Mahmood (`faisalmahmood@bwh.harvard.edu`), 
Richard J. Chen (`richardchen@g.harvard.edu`), 
Tong Ding (`tong_ding@g.harvard.edu`),
or Ming Y. Lu (`mlu16@bwh.harvard.edu`).


## Acknowledgements
The project was built on top of amazing repositories such as [ViT](https://github.com/google-research/big_vision), [DINOv2](https://github.com/facebookresearch/dinov2), [LGSSL](https://github.com/mbanani/lgssl),  and [Timm](https://github.com/huggingface/pytorch-image-models/) (ViT model implementation). We thank the authors and developers for their contribution. 


## BibTeX
If you found our work useful in your research, please consider citing our work at:

Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al. Towards a general-purpose foundation model for computational pathology. Nat Med (2024). https://doi.org/10.1038/s41591-024-02857-3

```
@article{chen2024uni,
  title={Towards a General-Purpose Foundation Model for Computational Pathology},
  author={Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Song, Andrew H and Shaban, Muhammad and others},
  journal={Nature Medicine},
  publisher={Nature Publishing Group},
  year={2024}
}
```
Works that use UNI should also attribute ViT and DINOv2.