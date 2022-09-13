from io import BytesIO
import os
import uuid
import boto3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dotenv import dotenv_values
from botocore.exceptions import ClientError


class AwsOperations:
    def __init__(self):
        self.setup_s3_storage()

    def setup_s3_storage(self):
        self.config = dotenv_values(".env")
        self.BUCKET = self.config.get("BUCKET_NAME")
        self.session = boto3.Session(
            aws_access_key_id=self.config.get("ACCESS_KEY_ID"),
            aws_secret_access_key=self.config.get("SECRET_ACCESS_KEY")
        )
        self.s3 = self.session.client('s3')

    def upload_file(self, file_name, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_name)

        try:
            self.s3.upload_file(
                file_name, self.BUCKET, object_name,
                ExtraArgs={'ACL': 'public-read-write', 'ContentType': 'image/jpeg'})

        except ClientError as e:
            print(e)
            return False
        return True

    def get_random_object(self):
        list_response = self.s3.list_objects_v2(
            Bucket=self.BUCKET,
            MaxKeys=1,
            StartAfter=str(uuid.uuid4()),
        )
        if 'Contents' in list_response:
            key = list_response['Contents'][0]['Key']
            item_response = self.s3.get_object(
                Bucket=self.BUCKET,
                Key=key
            )
            image = mpimg.imread(BytesIO(item_response['Body'].read()), 'jp2')
            plt.figure(0)
            plt.imshow(image)
            plt.show()
        else:
            print("Didn't find an item. Please try again.")


aws_operations = AwsOperations()
aws_operations.get_random_object()
