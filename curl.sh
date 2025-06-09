# prepare dataset for the model to train
# python data/frankenstein/prepare.py
# # train custom model from scratch 
# python train.py config/train_frankenstein.py --device=cpu
# # run api
# set MODEL_DIR=shakespeare && set DEVICE=cpu &&  uvicorn api:app --reload --port 8000

# test api
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello World!!", "num_samples": 1, "max_new_tokens": 50, "temperature": 0.8, "top_k": 200}'
