# AWS EC2 Setup for FSL FAST

Use AWS for MRI preprocessing when local disk space is too small for FSL.

## Recommended Instance

For the first HCP/OASIS preprocessing demo:

```text
AMI: Ubuntu Server 22.04 LTS
Instance type: t3.large or t3.xlarge
Storage: 100-150 GB gp3 EBS
Region: us-east-1
GPU: not required for FSL FAST
```

Avoid Ubuntu 26.04 for this project. The FSL conda installer can fail during the micromamba environment creation step on that image. Ubuntu 22.04 LTS is the safest choice.

Stop or terminate the instance when finished to avoid charges. Delete any extra EBS volumes/snapshots you do not need.

## Launch Steps

1. Open AWS Console.
2. Go to EC2.
3. Click **Launch instance**.
4. Name it `alzheimers-mri-fsl`.
5. Choose **Ubuntu Server 22.04 LTS**.
6. Choose `t3.large` for a cheap first run.
7. Create or choose an SSH key pair.
8. Set storage to 100-150 GB.
9. Allow SSH from your IP only.
10. Launch.

## Connect

Replace the key path and IP:

```bash
ssh -i ~/Downloads/your-key.pem ubuntu@EC2_PUBLIC_IP
```

If macOS complains about key permissions:

```bash
chmod 400 ~/Downloads/your-key.pem
```

## Bootstrap the Instance

Copy the bootstrap script to the instance:

```bash
scp -i ~/Downloads/your-key.pem scripts/aws/bootstrap_ubuntu_fsl.sh ubuntu@EC2_PUBLIC_IP:~
```

Run it:

```bash
ssh -i ~/Downloads/your-key.pem ubuntu@EC2_PUBLIC_IP
bash ~/bootstrap_ubuntu_fsl.sh
```

## Configure HCP S3 Credentials on EC2

On the EC2 instance, set the BALSA/HCP S3 credentials locally. Do not commit these values.

```bash
aws configure
```

Use:

```text
AWS Access Key ID: from BALSA
AWS Secret Access Key: from BALSA
Default region: us-east-1
Default output format: json
```

## Run HCP FAST Demo

Copy and run the pipeline script:

```bash
scp -i ~/Downloads/your-key.pem scripts/aws/run_hcp_100206_fast.sh ubuntu@EC2_PUBLIC_IP:~
ssh -i ~/Downloads/your-key.pem ubuntu@EC2_PUBLIC_IP
bash ~/run_hcp_100206_fast.sh
```

## Download Results Back

From your Mac:

```bash
mkdir -p data/processed/hcp_ya/fast
scp -i ~/Downloads/your-key.pem -r ubuntu@EC2_PUBLIC_IP:~/alzheimers-mri-reproduction/data/processed/hcp_ya/fast/100206 data/processed/hcp_ya/fast/
```

## Security Cleanup

After the run:

1. Reset the BALSA/HCP S3 key.
2. Stop or terminate the EC2 instance.
3. Delete EBS volumes/snapshots you no longer need.
