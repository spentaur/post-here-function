import json
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np
from boto3 import client


s3 = client('s3')
s3_bucket = "post-here"
file_prefix = "model-optimized.onnx"

labels = np.loadtxt('./labels.txt', dtype='str')

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
            k = 24

        inputs = tokenizer(
            text,
            return_tensors="np",
            max_length=512,
            padding=True,
            truncation=True
        )
        inputs = {onnx_input.name: inputs[onnx_input.name]
                  for onnx_input in model_onnx.get_inputs()}
        preds = model_onnx.run(None, inputs)[0]
        ind = preds.argsort()[:, ::-1][:, :k]
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps(labels[ind].tolist())
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
