import os
import boto3
from dotenv import dotenv_values
from botocore.exceptions import ClientError


class AwsOperations:
    def __init__(self):
        self.setup_s3_storage()

    def setup_s3_storage(self):
        self.config = dotenv_values(".env")
        self.session = boto3.Session(
            aws_access_key_id=self.config.get("ACCESS_KEY_ID"),
            aws_secret_access_key=self.config.get("SECRET_ACCESS_KEY")
        )
        self.s3 = self.session.client('s3')

    def upload_file(self, file_name, object_name=None):
        BUCKET = self.config.get("BUCKET_NAME")

        if object_name is None:
            object_name = os.path.basename(file_name)

        try:
            self.s3.upload_file(
                file_name, BUCKET, object_name,
                ExtraArgs={'ACL': 'public-read-write', 'ContentType': 'image/jpeg'})

        except ClientError as e:
            print(e)
            return False
        return True
