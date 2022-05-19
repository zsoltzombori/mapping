## Extraction Module

Extraction requires that you have a running [PostgreSQL](https://www.postgresql.org) database instance and the datasets provided with RODI are loaded into the dataset. See the linked page for instruction about how to set up PostgreSQL. Assuming that you have a running instance under user `user` which listens at localhost port `5432` and contains a database `rodi`, create a file named `database.ini` containing the following lines:

```
[postgresql]
host=localhost
database=rodi
user=user
port=5432
```

The source databases -- published in [RODI](https://www.cs.ox.ac.uk/isg/tools/RODI/) are also available within this repository inside the `RODI` folder in SQL dump files. You can find the RODI license at [RODI/LICENSE](RODI/LICENSE).

In order to load a dump file (e.g. [RODI/data/cmt_structured/dump.sql](RODI/data/cmt_structured/dump.sql)) into the database, run:

`psql -U user -d rodi -p 5432 -f RODI/data/cmt_structured/dump.sql`

Once the PostgreSQL server is running and contains the necessary databases, `extract.py` performs the dataset extraction. To run the code, use the provided [../learn/environment.yml](../learn/environment.yml) file to reconstruct the required python environment. The datasets are saved as Tensorflow dataset objects. 


```
$ python extract.py -h
usage: extract.py [-h] --schema SCHEMA [--pos_size POS_SIZE] [--filterlist FILTERLIST] [--sampling SAMPLING] [--outdir OUTDIR]

optional arguments:
  -h, --help            show this help message and exit
  --schema SCHEMA       Database schema: one of the subdirectories of RODI/data
  --pos_size POS_SIZE   Number of positive samples per predicate
  --filterlist FILTERLIST
                        Comma separated list of predicates names. If missing, all predicates in the queries are considered.
  --sampling SAMPLING   Negative sampling: uniform/realistic
  --outdir OUTDIR       Output directory
```

### Commands used to generate datasets for CMT experiments (Section 5.1):

```
python extract.py --schema cmt_naive --pos_size 100 --sampling uniform --outdir outdata
python extract.py --schema cmt_renamed --pos_size 100 --sampling uniform --outdir outdata
python extract.py --schema cmt_structured --pos_size 100 --sampling uniform --outdir outdata
python extract.py --schema cmt_structured_ci --pos_size 100 --sampling uniform --outdir outdata
python extract.py --schema cmt_naive_ci --pos_size 100 --sampling uniform --outdir outdata
python extract.py --schema cmt_denormalized --pos_size 100 --sampling uniform --outdir outdata
python extract.py --schema cmt_mixed --pos_size 100 --sampling uniform --outdir outdata
```

### Commands used to generate datasets for NPD experiments (Section 5.2):

```
python extract.py --schema npd --pos_size 100 --sampling uniform --outdir outdata
python extract.py --schema npd --pos_size 1000 --sampling uniform --outdir npddata1000 --filterlist Agent,AppraisalWellbore,Area,AwardArea,BAA,BAAArea,BAATransfer,Block,ChangeOfCompanyNameTransfer,Company,CompanyReserve,ConcreteStructureFacility,Condeep3ShaftsFacility,Condeep4ShaftsFacility,CondensatePipeline,DSTForWellbore,DevelopmentWellbore,Discovery,DiscoveryArea,DiscoveryReserve
```
