export TRACKING_SERVER_ARN=$(aws sagemaker describe-mlflow-tracking-server --tracking-server-name=SageMakerMLFlowDemoTS | jq -r '.TrackingServerArn')

cat <<EOF> pv_s3.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: s3-pv
spec:
  capacity:
    storage: 1200Gi # ignored, required
  accessModes:
    - ReadWriteMany # supported options: ReadWriteMany / ReadOnlyMany
  mountOptions:
    - allow-delete
    - region $AWS_REGION
    - prefix /
  csi:
    driver: s3.csi.aws.com # required
    volumeHandle: s3-csi-driver-volume
    volumeAttributes:
      bucketName: hpto-training-bucket-17nov
EOF

kubectl apply -f pv_s3.yaml