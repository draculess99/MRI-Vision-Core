"""
AWS Lambda transport for execution router.

Implements Transport interface with lazy boto3 import.
Only imported/used when aws_lambda mode is active.
"""

import json
from typing import Any, Dict

from mri_core.execution import Transport


class AwsLambdaTransport(Transport):
    """Transport for AWS Lambda invocation."""

    def invoke_remote(
        self, function_name: str, payload: Dict[str, Any], region: str
    ) -> Dict[str, Any]:
        """
        Invoke AWS Lambda function.

        Raises ImportError if boto3 not installed.
        Raises any exception from boto3 client.

        Args:
            function_name: Name of Lambda function
            payload: Input payload dict
            region: AWS region

        Returns:
            Response dict with statusCode and data
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 required for AWS_LAMBDA mode. "
                "Install with: pip install boto3"
            )

        client = boto3.client("lambda", region_name=region)

        response = client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )

        payload_dict = json.loads(response["Payload"].read())
        return payload_dict
