import json
import numpy as np
import io
import cv2
import boto3

s3_resource = boto3.resource('s3')
sagemaker_runtime = boto3.client('runtime.sagemaker')
sns_client = boto3.client('sns')

ENDPOINT_NAME = "#SAGEMAKER_ENDPOINT_NAME#"
SNS_TOPIC_ARN = "#SNS_TOPIC_ARN#"

def read_image(s3_image_name):
    print(s3_image_name)
    cv2_image = cv2.imread(s3_image_name)
    print(cv2_image)
    # image height/width
    image_height_rs = 324
    image_width_rs = 486
    # resize image
    cv2_resized_image = cv2.resize(cv2_image, (image_height_rs, image_width_rs))
    print(cv2_resized_image)
    # reshape image
    cv2_reshaped_image = np.reshape(cv2_resized_image, cv2_resized_image.shape[0]*cv2_resized_image.shape[1]*cv2_resized_image.shape[2])
    print(cv2_reshaped_image)
    return cv2_reshaped_image

def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()

def lambda_handler(event, context):
    s3_record = event['Records'][0] 
    s3_bucket = s3_record['s3']['bucket']['name']    
    s3_image_key = s3_record['s3']['object']['key']

    # read from s3    
    print(s3_bucket)
    print(s3_image_key)
    bucket = s3_resource.Bucket(s3_bucket)
    tmp_image_name = '/tmp/' + s3_image_key
    bucket.download_file(s3_image_key, tmp_image_name)

    image_vectors = read_image(tmp_image_name)
    
    payload = np2csv([image_vectors])
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME, 
        ContentType='text/csv',
        Body=payload)
    
    result = json.loads(response['Body'].read().decode())
    predicted_label = result['predictions'][0]['predicted_label']
    score = result['predictions'][0]['score']
    print(predicted_label)
    print(score)
    do_notify_sns_topic = (predicted_label==1 and score < 0.95) or (predicted_label==0)
    if(do_notify_sns_topic):
        message={
            "image_name": s3_image_key,
            "body": json.dumps(result)
        }
        sns_client.publish(TargetArn=SNS_TOPIC_ARN,Message=json.dumps(message)
)


    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(result)
    }

