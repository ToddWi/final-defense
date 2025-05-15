
## Installation
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

#### SAM ViT-H weights
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

### Training
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LLaVA" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="1" \
  --exp_name="lisa-7b"
```
When training is finished, to get the full model weight:
```
cd ./runs/lisa-7b/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```

For example:
```
CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="./LLaVA/LLaVA-Lightning-7B-v1-1" \
  --weight="lisa-7b/pytorch_model.bin" \
  --save_path="./LISA-7B"
```

### Validation
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LISA_HF_Model_Directory" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --exp_name="lisa-7b" \
  --eval_only
```

Note: the `v1` model is trained using both `train+val` sets, so please use the `v0` model to reproduce the validation results. (To use the `v0` models, please first checkout to the legacy version repo with `git checkout 0e26916`.)


## Inference 

To chat with [LISA-13B-llama2-v1](https://huggingface.co/xinlai/LISA-13B-llama2-v1) or [LISA-13B-llama2-v1-explanatory](https://huggingface.co/xinlai/LISA-13B-llama2-v1-explanatory):
(Note that `chat.py` currently does not support `v0` models (i.e., `LISA-13B-llama2-v0` and `LISA-13B-llama2-v0-explanatory`), if you want to use the `v0` models, please first checkout to the legacy version repo `git checkout 0e26916`.)
```
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1'
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1-explanatory'
```
To use `bf16` or `fp16` data type for inference:
```
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1' --precision='bf16'
```
To use `8bit` or `4bit` data type for inference (this enables running 13B model on a single 24G or 12G GPU at some cost of generation quality):
```
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1' --precision='fp16' --load_in_8bit
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1' --precision='fp16' --load_in_4bit
```
Hint: for 13B model, 16-bit inference consumes 30G VRAM with a single GPU, 8-bit inference consumes 16G, and 4-bit inference consumes 9G.

After that, input the text prompt and then the image path. For example，
```
- Please input your prompt: Where can the driver see the car speed in this image? Please output segmentation mask.
- Please input the image path: imgs/example1.jpg

- Please input your prompt: Can you segment the food that tastes spicy and hot?
- Please input the image path: imgs/example2.jpg
```
The results should be like:
<p align="center"> <img src="imgs/example1.jpg" width="22%"> <img src="vis_output/example1_masked_img_0.jpg" width="22%"> <img src="imgs/example2.jpg" width="25%"> <img src="vis_output/example2_masked_img_0.jpg" width="25%"> </p>

## Deployment
```
CUDA_VISIBLE_DEVICES=0 python app.py --version='xinlai/LISA-13B-llama2-v1 --load_in_4bit'
CUDA_VISIBLE_DEVICES=0 python app.py --version='xinlai/LISA-13B-llama2-v1-explanatory --load_in_4bit'
```
By default, we use 4-bit quantization. Feel free to delete the `--load_in_4bit` argument for 16-bit inference or replace it with `--load_in_8bit` argument for 8-bit inference.


## Dataset
In Logi-CoT ReasonSeg, we have collected 503 images (243 train, 60 val, and 200 test). The training and validation sets can be download from <a href="https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy?usp=sharing">**this link**</a>. 

Each image is provided with an annotation JSON file:
```
image_1.jpg, image_1.json
image_2.jpg, image_2.json
...
image_n.jpg, image_n.json
```
Important keys contained in JSON files:
```
- "text": text instructions.
- "is_sentence": whether the text instructions are long sentences.
- "shapes": target polygons.
```

The elements of the "shapes" exhibit two categories, namely **"target"** and **"ignore"**. The former category is indispensable for evaluation, while the latter category denotes the ambiguous region and hence disregarded during the evaluation process. 

We provide a <a href="https://github.com/dvlab-research/LISA/blob/main/utils/data_processing.py">**script**</a> that demonstrates how to process the annotations:
```
python3 utils/data_processing.py
```

Besides, we leveraged GPT-3.5 for rephrasing instructions, so images in the training set may have **more than one instructions (but fewer than six)** in the "text" field. During training, users may randomly select one as the text query to obtain a better model.

