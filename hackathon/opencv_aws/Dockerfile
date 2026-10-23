# AWS Lambda Python base image for x86_64
# See: https://docs.aws.amazon.com/lambda/latest/dg/python-image.html
FROM public.ecr.aws/lambda/python:3.12

# Install OpenCV and NumPy dependencies for MRI processing
# Uses opencv-python-headless (no GUI/X11 dependencies) matching local project major version 5.0.0
# numpy 2.5.2 matches the local validated environment
RUN pip install --no-cache-dir \
    opencv-python-headless==5.0.0.93 \
    numpy==2.5.2

# Copy the OpenCV AWS project into the Lambda task root
COPY hackathon/opencv_aws/ ${LAMBDA_TASK_ROOT}/hackathon/opencv_aws/
COPY mri_core/ ${LAMBDA_TASK_ROOT}/mri_core/

# Set the Lambda handler
# The handler format is: path.to.module.function_name
CMD [ "hackathon.opencv_aws.evidence_manifest.lambda_handler" ]
