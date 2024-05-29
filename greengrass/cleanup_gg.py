DEV_IOT_COMPONENTS=[
    "com.aws.jetson.docker",
    "com.aws.edge.genai"
]

import boto3, os

AWS_ACCOUNT_NUM=os.environ["AWS_ACCOUNT_NUM"]
AWS_REGION=os.environ["AWS_REGION"]
DEV_IOT_THING=os.environ["DEV_IOT_THING"]
DEV_IOT_THING_GROUP=os.environ["DEV_IOT_THING_GROUP"]

gg_client = boto3.client('greengrassv2')

# Delete all the AWS IoT Greengrass Components
component_arn = None
response = gg_client.list_components()
for component in response['components']:
    if component['componentName'] in DEV_IOT_COMPONENTS:
        component_arn = component['arn']
if component_arn!=None:
    response = gg_client.list_component_versions(arn=component_arn)
    for component_ver in response['componentVersions']:
        gg_client.delete_component(arn=component_ver['arn'])
        print(f"Deleted component: {component_ver['arn']}")

# Delete all the AWS IoT Greengrass Deployments
deployments = gg_client.list_deployments(targetArn=f"arn:aws:iot:{AWS_REGION}:{AWS_ACCOUNT_NUM}:thing/{DEV_IOT_THING}", historyFilter='ALL')['deployments']
for deployment in deployments:
    if deployment['deploymentName']==DEV_IOT_THING_GROUP:
        deployment_arn = deployment['targetArn']
        deployment_id = deployment['deploymentId']
        if deployment['deploymentStatus'] == "COMPLETED" or deployment['deploymentStatus'] == "ACTIVE":
            gg_client.cancel_deployment(deploymentId=deployment_id)
        gg_client.delete_deployment(deploymentId=deployment_id)
        print(f"Deleted deployment: {deployment_arn} : {deployment_id} : {deployment['revisionId']}")