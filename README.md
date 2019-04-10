# Identifying Unclear Questions in Community Question Answering Websites

This repository provides the resources developed within the following paper:

> J. Trienes and K. Balog. **Identifying Unclear Questions in Community Question Answering Websites**. In proceedings of the 41st European conference on Advances in Information Retrieval (ECIR '19), pages 276--289, 2019. [DOI:10.1007/978-3-030-15712-8_18](http://doi.org/10.1007/978-3-030-15712-8_18)

These resources allow to reproduce the results presented in [the paper](https://arxiv.org/abs/1901.06168).

**You can get the authors' version of the paper from this link: [paper](https://arxiv.org/abs/1901.06168)**


### Abstract

> *Thousands of complex natural language questions are submitted to community question answering websites on a daily basis, rendering them as one of the most important information sources these days. However, oftentimes submitted questions are unclear and cannot be answered without further clarification questions by expert community members. This study is the first to investigate the complex task of classifying a question as clear or unclear, i.e., if it requires further clarification. We construct a novel dataset and propose a classification approach that is based on the notion of similar questions. This approach is compared to state-of-the-art text classification baselines. Our main finding is that the similar questions approach is a viable alternative that can be used as a stepping stone towards the development of supportive user interfaces for question formulation.*

## Computational Environment

We provide the computational environment we used throughout our experiments. A local installation of Elasticsearch and MongoDB is required. A pre-configured docker setup can be started as follows:

```sh
docker-compose up -d

# MongoDB is running on localhost:27017
# Elasticsearch is running on localhost:9200
# express-mongo is running on localhost:8081
```

For all Python code (except the CNN code), use the versions specified in `environment.yml`.

```sh
conda env create -f environment.yml
source activate stackexchange
```

For the CNN experiments, a slightly different environment is used:

```sh
conda env create -f environment-cnn.yml
source activate stackexchange-cnn
```

If you are not using Docker or conda, make sure to have the correct software versions as per the `docker-compose.yml` and `environment[-cnn].yml`.

In case you intend to use an external MongoDB or Elasticsearch instance, you have to define the following environment variables:

```sh
export MONGO_URI="mongodb://<user>:<pass>@<host>:<port>/<database>"
export ELASTIC_HOST="<the host>"
export ELASTIC_PORT="<the port>"
```

Finally, you may have to add the root of this repository to your `PYTHONPATH` if you are getting import errors while executing the Python scripts.

```sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Dataset

There are two options to obtain the dataset we used in our experiments:

   1. Download our preprocessed `.csv` files which include question annotation (clear/unclear), extracted clarification questions, and our training/testing splits.
   2. Generate the data for a most recent Stack Exchange dump (see [Archive.org](https://archive.org/details/stackexchange)).

The first version is to be preferred if you want to reproduce our experimental results. The second version might be useful if you want to experiment with a Stack Exchange community that we haven't used.

The raw data we used has been [released under cc-by-sa 3.0](https://archive.org/details/stackexchange):

> All user content contributed to the Stack Exchange network is cc-by-sa 3.0 licensed, intended to be shared and remixed.
License: http://creativecommons.org/licenses/by-sa/3.0/

### Option 1: Download our data

Download data from [here](https://gustav1.ux.uis.no/downloads/ecir2019-qac/ecir2019-qac-data.zip) and extract into `data` folder. It should look as follows:

```sh
> tree data
data
├── clarq                     # clarification question for each unclear question
│   ├── askubuntu.csv
│   ├── debug.csv
│   ├── stackoverflow.csv
│   ├── stats.csv
│   ├── superuser.csv
│   └── unix.csv
└── labeled
    ├── askubuntu.csv
    ├── askubuntu_test.csv    # test ID's + labels
    ├── askubuntu_train.csv   # train ID's + labels
    ...
```

### Option 2: Generate data for new Stack Exchange community

Visit `localhost:8081` and create a new database (we use `stackexchange`). Afterwards, download data for a community of your choice.

```sh
# replace <community> with the desired name
cd data/
curl -L -O https://archive.org/download/stackexchange/<community>.stackexchange.com.7z
7z e <community>.stackexchange.com.7z -o<community>.stackexchange.com/
cd ..
```

Then, adapt the data generation script under `scripts/generate_data.sh` to point to the folder you just downloaded. An example for the "unix" community:

```sh
> head -n 3 scripts/generate_data.sh

community="unix"
xml_dir=data/"$community".stackexchange.com/
out_dir=data
```

Finally, start the data generation. The script imports the raw XML Stack Exchange dump into MongoDB, denormalize the schema and annotates the data. Finally, the data is exported as CSV files for use in our analysis and model training.

```sh
# This takes about 30 minutes for the Cross Validated community (130,000 questions)
# For Stack Overflow (17,000,000 questions), it will take multiple hours
./scripts/generate_data.sh
```

### Dataset Statistics

A summary of each dataset (see Table 3 in paper) can be generated.

```sh
python qac/dataset/data_analysis.py <community> data/labeled/<community>.csv

# output can be found under output/reports/<community>
```

## Baseline Models

Each model can be executed individually. When training completes, the model is tested and performance is reported. Additionally, the predictions on the testing data are written to `output/predictions/<community>/<model_id>_test.csv` for subsequent analysis.

*All scripts below follow the convention that data is under `data/labeled` and `data/clarq`.*

```sh
# Set the community name.
COMMUNITY=stats # Cross Validated

# Random
python qac/baseline/baseline_dummy.py "$COMMUNITY" baseline_random --strategy uniform

# Majority
python qac/baseline/baseline_dummy.py "$COMMUNITY" baseline_majority --strategy most_frequent

# BoW LR (n=1)
# set --ngram_range 3 for BoW LR (n=3)
python qac/baseline/baseline_lr_fixed_n.py "$COMMUNITY" baseline_lr_1ngram_c1 --ngram_range 1
```

## Convolutional Neural Network

See [this page](./qac/baseline/) for instructions on how to execute the CNN models and the hardware we used.

## Similar Questions Model (SimQ)

We decouple the similar questions retrieval and feature generation from actual model training. First, similar questions are retrieved for every question in the dataset and saved in TREC format under `models/simq`. Afterwards, features are generated based on this retrieval run and saved under `models/simq-features`. This will not take longer than 30 minutes for the Cross Validated community. For Stack Overflow, you can expect a runtime of about *1.5 weeks* and the retrieval results can get rather large (32GB for 11.7m Stack Overflow queries).

```sh
COMMUNITY=stats # Cross Validated
SIMQ_RUN=60stop0body

# Retrieve similar questions
python qac/simq/simq_retrieval.py "$COMMUNITY" $SIMQ_RUN --strategy "constrained"

# Compute features
python qac/simq/simq_features.py "$COMMUNITY" $SIMQ_RUN --n_jobs 1
```

Execute models. This should not take longer than a few minutes.

```sh
# SimQ Majority
python qac/simq/simq_majority.py "$COMMUNITY" "simq_${SIMQ_RUN}_majority" $SIMQ_RUN
# CQ Global
python qac/simq/simq_threshold_classifier.py "$COMMUNITY" "$SIMQ_RUN" feat_unclear_global_cos
# CQ Individual
python qac/simq/simq_threshold_classifier.py "$COMMUNITY" "$SIMQ_RUN" feat_unclear_individual_cos
# CQ Weighted
python qac/simq/simq_threshold_classifier.py "$COMMUNITY" "$SIMQ_RUN" feat_unclear_individual_cos_weighted
# SimQ ML
python qac/simq/simq_ml.py "$COMMUNITY" "simq_${SIMQ_RUN}_ml_all" "$SIMQ_RUN" --feature_group all
```

## Bulk Scripts

For convenience, the `scripts/` folder contains a number of scripts which execute the above models for multiple communities. Also, `scripts/evaluation.sh` evaluates every run for every community (i.e., `output/predictions/**/*_test.csv`) and writes a summary Excel and CSV file under `output/evaluation` for further analysis.

## Tests and Pylint

Some parts of the code have been unit-tested:

```
pytest qac/
```

Also, pylint is configured:

```
pylint qac/
```

## Citation

If you use the resources presented in this repository, please cite:

```
@inproceedings{Trienes:2019:IUQ,
 author =    {Trienes, Jan and Balog, Krisztian},
 title =     {Identifying Unclear Questions in Community Question Answering Websites},
 booktitle = {Proceedings of the 41st European conference on Advances in Information Retrieval},
 series =    {ECIR '19},
 year =      {2019},
 pages =     {276--289},
 doi =       {10.1007/978-3-030-15712-8_18},
 publisher = {Springer}
}
```

## Contact
If you have any question, please contact Jan Trienes at jan.trienes@gmail.com.
