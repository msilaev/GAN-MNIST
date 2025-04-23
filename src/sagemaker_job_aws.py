import os
import sagemaker
from dotenv import load_dotenv
from sagemaker.pytorch import PyTorch
from sagemaker.local import LocalSession

load_dotenv()

#  Set AWS dummy credentials BEFORE anything else
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] =  os.getenv("SAGEMAKER_REGION")

#  Set file paths (using local mode)
#s3_input_path = "file://./data"
#s3_output_path = "file://./logs"

s3_input_path = os.getenv("S3_INPUT_PATH")
s3_output_path = os.getenv("S3_OUTPUT_PATH")

#  Define estimator
estimator = PyTorch(
    entry_point="train_conditional_gan_sagemaker.py",
    source_dir=".",
    role=os.getenv("SAGEMAKER_ROLE_ARN"),  # Replace with your IAM role ARN
#    role="dummy-role",  # anything works in local mode
    framework_version="1.9.1",
    py_version="py38",
    instance_count=1,
    #instance_type="local",
    instance_type="ml.m5.medium",  # Use a valid SageMaker instance type
    hyperparameters={
        "batch-size": 64,
        "n-epochs": 10,
        "lr": 0.0002,
        "latent-dim": 100,
        "num-classes": 10,
        "image-size": 28
    },
    output_path=s3_output_path,
)

#  Fit the model using local path
estimator.fit({"train": s3_input_path})