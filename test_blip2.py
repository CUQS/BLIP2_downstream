import torch
from models.encoder_decoder.BLIP2 import load_blip2_and_preprocess

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# vitL
model, vis_processors, txt_processors = load_blip2_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
# model, vis_processors, txt_processors = load_blip2_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

print(model)
