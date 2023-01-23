# Before running tests, ensure that a test bucket called "test" in the Minio Store is available:
* Create a Read-Write bucket "test"
* Upload data/sentiment1.csv
* Create a RabbitMQ virtual host "qcirbflj"
* Create a Loadbalancer for the RabbitMQ service:
  * kubectl expose svc/rabbitmq-headless --name rabbitmq-external --port=5672 --target-port=5672 --type=LoadBalancer
  * (On AWS, increase the newly created LoadBalancer endpoint's timeout to 3600 seconds)
* Deploy the configmap dependency:
```
kubectl delete configmap test-ml-model || true
kubectl create configmap test-ml-model --from-env-file=.env
kubectl rollout restart deployment/skipper
kubectl rollout restart deployment/scdf-server
```

* Prepare the environment for the pipelines:
```
cd /path/to/scdf/model
scripts/prepare-pipeline-environment.sh
cd -
```

* In GitHub, create Secrets under Settings -> Secrets:
  * Create a Secret named CREATE_PIPELINE_CMD with the content of scripts/commands/create-scdf-ml-pipeline.txt.
  * Create a Secret named UPDATE_PIPELINE_CMD with the content of scripts/commands/update-scdf-ml-pipeline.txt.
