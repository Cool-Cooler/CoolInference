# CoolInference
This is a project to build a docker image for object detection using Flask RESTful API with Docker-Compose.

## Requirements

To build this project you will need [Docker][Docker Install] and [Docker Compose][Docker Compose Install].

## Deploy and Run Locally

After cloning this repository, you can type the following command to start the simple app:

```sh
make install
```

Then simply visit [localhost:5000][App] !

## Deploying in Amazon ECR
This project is configured with an "GitHub CI" action to build and push the docker image to Amazon ECR. Set the repository secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` with AWS access credentials to setup this automatic workflow.  

[Docker Install]:  https://docs.docker.com/install/
[Docker Compose Install]: https://docs.docker.com/compose/install/
[App]: http://127.0.0.1:5000
