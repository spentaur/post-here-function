import json
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import heapq
from boto3 import client


s3 = client('s3')
s3_bucket = "post-here"
file_prefix = "model-optimized.onnx"

with open("./labels.json", "r") as f:
    labels = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("./model", use_fast=True)
obj = s3.get_object(Bucket=s3_bucket, Key=file_prefix)
model_onnx = InferenceSession(obj['Body'].read())


def predict(event, contenxt):
    try:
        body = json.loads(event['body'])
        if 'text' not in body:
            return {
                "statusCode": 200,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True

                },
                "body": json.dumps("Please include text.")
            }
        text = body['text']
        if 'k' in body:
            k = body['k']
        else:
            k = 16
        inputs = tokenizer(text, padding=True, truncation=True,
                           return_tensors="np")
        inputs = {onnx_input.name: inputs[onnx_input.name]
                  for onnx_input in model_onnx.get_inputs()}
        preds = model_onnx.run(None, inputs)[0]
        top_k = [
            heapq.nlargest(
                k,
                range(
                    len(pred)),
                pred.__getitem__) for pred in preds]
        preds_labels = [[labels[str(pred)]
                         for pred in top_n if pred] for top_n in top_k]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps(preds_labels)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
