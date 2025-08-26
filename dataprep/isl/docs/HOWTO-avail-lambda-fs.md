# Lambda Cloud File System

- Lambda cloud provides cloud storage, which is supposed to be mounted on lambda instances.
- They also provide S3 compatible APIs which allows us to upload and download esily.
- In our solution, we follow this methodology
    1. Upload the downsampled and trimmed mp4s to lambda storage, using `awscli`, from BCS-VM after running CPU-pipeline.
    2. From colab instance, download the videos to it from lambda storage, again using `awscli`.
    3. After processing the videos with dwpose on colab, upload the generated `.npz` files to lambda cloud.
    4. Later dowmload the output files to BCS-
    VM. These files are included in the dataset, before making tar files for HuggingFace push.
- Lambda cloud storage serves as a persistant storage solution for the colab instance.
- Also lambda cloud storage facilitates data transfer between the BCS-VM, which is within VPN, using aws.

> In the solution architecture, lambda cloud can easily be replaced by another cloud storage like AWS S3 or Cloudflare R2. If it has s3 api combatibility, the better.

## How to avail a cloud storage in lambda
1. Login to https://cloud.lambda.ai . BCS has an account and users can be added there.
2. Generate API key at https://cloud.lambda.ai/api-keys/s3-adapter . It will generate AWS_ACCESS_ID, AWS_SECRET_KEY, AWS_REGION and AWS_ENPOINTURL. Copy and keep these safe for use from BCS-VM, COLAB etc.
3. Upon choosing the region for storage select one with s3 API support. (Only us-east-2 and us-east-3 had it on Aug, 2025).
4. The bucket name came be copies from the table at https://cloud.lambda.ai/file-systems .

> This storage will be charged based on the size of data kept in it on an hourly rate. It will be charged whether or not it is mounted on an instance.


## Useful commands for data transfer
To setup access:
```bash
pip install awscli boto3
aws configure
```
For avoiding trouble when using lambda, add this to the `~/.aws/config` file:
```
[default]
region =us-east-3 
endpoint-url =https://files.us-east-3.lambda.ai 

request_checksum_calculation =when_required 
response_checksum_validation =when_required 
```

To view the contents:

```bash
aws s3 s3://<bucket-name> --endpoint-url https://files.us-east-3.lambda.ai

aws s3 s3://<bucket-name>/<folder-name> --endpoint-url https://files.us-east-3.lambda.ai

```

To copy file from local instance to lambda:
```bash
aws s3 cp isl_gospel_videos/ s3://<bucket-name>/isl_gospel_videos/ --recursive --endpoint-url https://files.us-east-3.lambda.ai
```


To copy file from lambda to local instance :
```bash
aws s3 cp  s3://<bucket-name>/isl_gospel_videos/ isl_gospel_videos/ --recursive --endpoint-url https://files.us-east-3.lambda.ai
```
s3 sync is another option:
```bash
aws s3 sync  s3://<bucket-name>/isl_gospel_videos/ isl_gospel_videos/  --endpoint-url https://files.us-east-3.lambda.ai

```

## Challenges faced
Original plan was to use a lambda instance for the GPU compute. But it was not successful.
- Only two regions, us-east-2 and us-east-3 provides this storage facility (as on Aug 2025).
- The lambda instances which needs to mount these storage should also reside in the same region.
- The us-east-3 region, had GH200 GPU machines. But they were ARM64 based archs. The `onnxruntime-gpu` package, which dwpose uses, did not have an official whl for this arch.
- Upon using `onnxruntime` instead, only CPU was being used.
- Next option was to complie and build whl for `onnxruntime-gpu` on the instance itself. But later, these instances became too busy/unavailable. And could not continue the plan.

