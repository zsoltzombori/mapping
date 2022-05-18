# Learning Mapping Rules for Ontology to Relational Alignment

This codebase has two parts. There is a **dataset extraction** module, that is responsible for extracting datasets from the [RODI benchmark](https://www.cs.ox.ac.uk/isg/tools/RODI/). The datasets are saved as Tensorflow dataset objects. The second part is the **learning** module, provided in [learn/](learn/), which is descibed in [learn/README.md](learn/README.md). The two parts interact only via the generated datasets. In the following, we describe the **dataset extraction** part.

### Dataset Extraction

Extraction requires that you have a running [PostgreSQL](https://www.postgresql.org) database instance and the datasets provided with RODI are loaded into the dataset. See the linked page for instruction about how to set up PostgreSQL. Assuming that you have a running instance under user `user` which listens at localhost port `5432` and contains a database `rodi`, create a file named `database.ini` containing the following lines:

```
[postgresql]
host=localhost
database=rodi
user=user
port=5432
```

The source databases -- published in [https://www.cs.ox.ac.uk/isg/tools/RODI/](https://www.cs.ox.ac.uk/isg/tools/RODI/) are also available within this repository at [RODI/data](RODI/data) in SQL dump files. In order to load a dump file into the database, run e.g.:

`psql -U user -d rodi -p 5432 -f RODI/data/cmt_structured/dump.sql`

Once the PostgreSQL server is running and contains the necessary databases, `generator.py` performs the dataset extraction. To run the code, use the provided [learn/environment.yml](learn/environment.yml) file to reconstruct the required python environment. 




We consider the problem
Learning mapping rules for relational to ontology alignment
