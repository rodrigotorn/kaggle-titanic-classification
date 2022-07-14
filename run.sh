#!/bin/bash

docker build --tag titanic-classification .

docker run --rm \
	-v $PWD:/home/kaggle-titanic-classification \
	titanic-classification
