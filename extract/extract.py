import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--schema', type=str, required=True, help="Database schema: one of the subdirectories of RODI/data")
parser.add_argument('--pos_size', type=int, default=100, help="Number of positive samples per predicate")
parser.add_argument('--filterlist', type=str, default=None, help="Comma separated list of predicates names. If missing, all predicates in the queries are considered.")
parser.add_argument('--sampling', type=str, default="uniform", help="Negative sampling: uniform/realistic")
parser.add_argument('--outdir', type=str, default="outdata", help="Output directory")
args = parser.parse_args()
print("\nArguments:")
for arg in vars(args):
  print("   ", arg, getattr(args, arg))
print("\n")
  
import mappingProblem
import supervision

def generate(schema, true_mapping, pos_size, outdir, filterlist=None, sampling="uniform"):
    print("SCHEMA: ", schema)
    ontology = "RODI/data/{}/ontology.ttl".format(schema)
    query_dir = "RODI/data/{}/queries".format(schema)
    datapath = "{}/{}/{}".format(outdir, schema, schema)
    problem = mappingProblem.MappingProblem(schema, ontology, true_mapping)
    problem.add_query_dir(query_dir)
    
    t0 = time.time()
    problem.generate_data(samplesize=pos_size, path=datapath, filterlist=filterlist, sampling=sampling)
    #     problem.generate_data_neg_uniform(samplesize=pos_size, path=datapath, filterlist=filterlist)
    t1 = time.time()
    print("Data generation for schema {} took {:.3f} sec".format(schema, t1 - t0))


schema2supervision = {
    "cmt_renamed": supervision.cmt_renamed_mapping,
    "cmt_structured": supervision.cmt_structured_mapping,
    "cmt_structured_ci": supervision.cmt_structured_ci_mapping,
    "cmt_naive": supervision.cmt_naive_mapping,
    "cmt_naive_ci": supervision.cmt_naive_ci_mapping,
    "cmt_denormalized": supervision.cmt_denormalized_mapping,
    "cmt_mixed": supervision.cmt_mixed_mapping,
}

if args.schema in schema2supervision:
    true_mapping = schema2supervision[args.schema]
else:
    true_mapping = None

if args.filterlist is not None:
    args.filterlist = args.filterlist.split(',')
    
generate(args.schema, true_mapping, args.pos_size, args.outdir, args.filterlist, args.sampling)
