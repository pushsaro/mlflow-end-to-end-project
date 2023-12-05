# MLflow Project steps

This repository contains an end-to-end machine learning project using MLflow, Git, CI/CD with Jenkins, Docker, and Gunicorn.

# Create a new environment for project

...
conda create -p venv python==3.7 -y
...

# Setup and Run

1. Clone the repository:

    ...bash
    git clone https://github.com/pushsaro/mlflow-end-to-end-project.git
    cd mlflow-end-to-end-project
    ...

2. Build and run the MLflow server:

    ...bash
    docker-compose up -d
    ...

    The MLflow UI will be available at [http://localhost:8000](http://localhost:8000).

3. Train the model:

    ...bash
    python model_training/train.py
    ...

4. Deploy the model:

    ...bash
    python model_inference/predict.py --model_path /path/to/your/model
    ...

5. View the results in the MLflow UI.

# Additional Information

# Software and tools requirements

1. [Github Account](https://github.com)
   
2. [Amazon Web Services Account](https://aws.amazon.com/free/?trk=16847e0c-46fb-467d-91ee-6e259e339665&sc_channel=ps&s_kwcid=AL!4422!10!72086958325164!72087482393523&ef_id=de197eca60e313e469e91ba207a0345a:G:s&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all)
   
3. [VS Code IDE](https://code.visualstudio.com/)
   
4. [Docker Hub](https://hub.docker.com)

5. [MLFlow](https://mlflow.org/)

# Docker setup in EC2 commands to be executed

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
