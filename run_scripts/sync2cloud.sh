module load aws-cli
aws --endpoint-url https://s3.abci.ai s3 sync mlruns s3://experiment-data-j2020-1/mlruns
aws --endpoint-url https://s3.abci.ai s3 sync Test s3://experiment-data-j2020-1/Test
