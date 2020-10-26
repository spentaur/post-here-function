import azure.functions as func
from transformers import DistilBertTokenizerFast
import onnxruntime
import json
import time
import logging
import heapq
from pathlib import Path

with open(Path(__file__).parent / "labels.json", "r") as f:
    labels = json.load(f)

model_onnx = onnxruntime.InferenceSession(
    f"{Path(__file__).parent}/model.onnx")
tokenizer = DistilBertTokenizerFast.from_pretrained(
    f"{Path(__file__).parent}/tokenizer")


def predict_onnx(text, k=500):
    start = time.time()
    inputs = tokenizer(
        text,
        return_tensors="np",
        max_length=512,
        truncation=True,
        padding=True)
    inputs = {onnx_input.name: inputs[onnx_input.name]
              for onnx_input in model_onnx.get_inputs()}
    logging.info(f"Tokenizing took {time.time() - start}")
    start1 = time.time()
    preds = model_onnx.run(None, inputs)[0]
    logging.info(f"Prediction took {time.time() - start1}")
    start2 = time.time()
    top_k = [
        heapq.nlargest(
            k,
            range(
                len(pred)),
            pred.__getitem__) for pred in preds]
    preds_labels = [[labels[str(pred)]
                     for pred in top_n if pred] for top_n in top_k]
    logging.info(f"Getting Labels took {time.time() - start2}")
    logging.info(f"Total prediction function took {time.time() - start}")
    return preds_labels


def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == 'GET':
        return func.HttpResponse("success", status_code=200)
    req_body = req.get_json()
    text = req_body.get('text')
    k = req_body.get('k') or 500

    if text:
        preds = predict_onnx(text, k)
        return func.HttpResponse(json.dumps(preds))
    else:
        return func.HttpResponse("Please enter text.")
