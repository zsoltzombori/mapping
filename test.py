import rdflib
import psycopg2
import time
import tensorflow as tf

import util
import mappingProblem
import supervision

from rule import Rule, Variable, Constant
import train
import data

#schema = "conference_structured"
#schema = "conference_renamed"
#schema = "npd_user_tests"
POS_SIZE=100

true_mapping = supervision.cmt_mapping
true_schema = supervision.cmt_schema

schemas = (
    "cmt_renamed",
    "cmt_structured",
    "cmt_structured_ci",
    "cmt_naive",
    "cmt_naive_ci",
    "cmt_denormalized",
    "cmt_mixed",
)

for schema in schemas:
    print("SCHEMA: ", schema)
    ontology = "RODI/data/{}/ontology.ttl".format(schema)
    query_dir = "RODI/data/{}/queries".format(schema)
    datapath = "outdata/{}".format(schema)
    problem = mappingProblem.MappingProblem(schema, ontology, true_mapping, true_schema)
    t0 = time.time()
    problem.generate_data(samplesize=POS_SIZE, path=datapath)
    t1 = time.time()
    print("Data generation for schema {} took {:.3f} sec".format(schema, t1 - t0))


# generator = data.MyGenerator(dim=32, batch_size=1024)

# for predicate in true_mapping:
#     print("Predicate: ", predicate)
#     query = true_mapping[predicate]

#     # todo arity 2
#     rules = []        
#     rules.append(Rule([[predicate, Variable(1)], [predicate+"_pred1a", Variable(1)]], problem.cursor, schema, problem.preds))
#     # rules.append(Rule([[predicate, Variable(1)], [predicate+"_pred2a", Variable(1)]], problem.cursor, schema, problem.preds))
#     # rules.append(Rule([[predicate, Variable(1)], [predicate+"_pred3a", Variable(1)]], problem.cursor, schema, problem.preds))
#     # rules.append(Rule([[predicate, Variable(1)], [predicate+"_pred4a", Variable(1)]], problem.cursor, schema, problem.preds))
#     # rules.append(Rule([[predicate, Variable(1)], [predicate+"_pred5a", Variable(1)]], problem.cursor, schema, problem.preds))
#     # rules.append(Rule([[predicate, Variable(1)], [predicate+"_pred2a", Variable(1), Constant(predicate+"_const2a")]], problem.cursor, schema, problem.preds))

#     models = []

#     supervision = util.create_supervision(problem.cursor, schema, predicate, query, 100, problem.constants)
#     mapping_stats = {}
#     t0 = time.time()
#     print("Collecting support...")
#     for s, ispositive in supervision:
#         d_input = ["SOS"] + s + ["EOP","EOS"]
#         d_input = [str(x) for x in d_input]
#         d_input = " ".join(d_input)
#         mappings_list = []
#         for r in rules:
#             mappings, targets = r.get_support(s)
#             for t in targets:
#                 d_output = [str(x) for x in t]
#                 d_output = " ".join(d_output)
#                 dataset_elements.append([d_input, d_output, str(ispositive)])
#             if len(mappings) == 0:
#                 continue
#             generator.add_mappings(mappings, ispositive)
#             mappings_list.append(mappings)
#             for m in mappings:
#                 for aux in m:
#                     pred = m[aux][0]
#                     predname = util.pred2name(pred)
#                     if predname not in mapping_stats:
#                         mapping_stats[predname] = {True:0, False:0}
#                     mapping_stats[predname][ispositive] += 1
#         if len(mappings_list) == 0:
#             continue

#     # models.append(train.MappingLayer(mappings_list, emap, fact=s, positive=ispositive))
#     t1 = time.time()
#     print("Collecting support took {} sec.".format(t1 - t0))
#     print(mapping_stats)

#     # print("Training...")
#     # train.train(generator, epochs=100, lr=0.01, Tmin=0.01)
#     # xxx
#     # print("Avg eval loss: ", train.eval(models))
#     # t2 = time.time()
#     # print("Training took {} sec.".format(t2 - t1))

# dataset = tf.data.Dataset.from_tensor_slices(dataset_elements)

# xxx

# problem.add_query_dir(query_dir)

# success = 0
# for query in problem.queries:
#     print("\n**********************")
#     print(query.filename)
#     print("sparql query: ", query.sparql_query)
#     # qres = problem.graph.query(query.sparql_query, initNs = problem.namespaces)
#     # for row in qres:
#     #     print(f"{row}")
    
#     print("\nsql query: ", query.sql_query)
#     problem.cursor.execute(query.sql_query)
#     result = problem.cursor.fetchall()

#     sql_candidates = problem.sparql2sql(query)
#     curr_succ = False
#     for sql2 in sql_candidates:
#         print("\n transformed query: ", sql2)
#         problem.cursor.execute(sql2)
#         result2 = problem.cursor.fetchall()

#         if result == result2:
#             success+=1
#             curr_succ = True
#             print("!!!!PASSED!!!!")
#             break
#     if not curr_succ:
#         print("!!!!FAILED!!!!")
#         # util.print_result(result)
#         # util.print_result(result2)
#         # assert False
# print("Passed {} out of {}".format(success, len(problem.queries)))
        
        





# print("All triples with 'acceptedBy' in subject position")
# result = list(g[rdflib.URIRef("http://cmt#acceptedBy")])
# for i, r in enumerate(result):
#     print(i, r)

# print("All triples with 'acceptedBy' in predicate position")
# result = list(g[:rdflib.URIRef("http://cmt#acceptedBy")])
# for i, r in enumerate(result):
#     print(i, r)

# print("Entities with range 'Bid'")
# result = list(g[:rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#range'): rdflib.URIRef("http://cmt#Bid")])
# for i, r in enumerate(result):
#     print(i, r)

# print("persons")
# result = list(g[:rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'): rdflib.URIRef("http://cmt#Person")])
# for i, r in enumerate(result):
#     print(i, r)

# qres = g.query("""SELECT ?s WHERE { ?s a <c:> }""")
# for row in qres:
#     print(f"{row.s}")





# ##########################






# xxx

# try connecting to the database
# conn = None
# try:
#     params = config()
#     conn = psycopg2.connect(**params)
#     cursor = conn.cursor()

#     inspect(cur)
#     xxx

#     # execute a statement
#     print('PostgreSQL database version:')
#     cur.execute('SELECT version()')

#     # display the PostgreSQL database server version
#     db_version = cur.fetchone()
#     print(db_version)
       
#     # close the communication with the PostgreSQL
#     cur.close()
# except (Exception, psycopg2.DatabaseError) as error:
#     print(error)
# finally:
#     if conn is not None:
#         conn.close()
#         print('Database connection closed.')
        

# # restore a database from dump

# try:
#     params = config()
#     conn = psycopg2.connect(**params)
#     cur = conn.cursor()

#     cur.execute("pg_restore {}".format(dump_file))
       
#     # close the communication with the PostgreSQL
#     cur.close()
# except (Exception, psycopg2.DatabaseError) as error:
#     print(error)
# finally:
#     if conn is not None:
#         conn.close()
#         print('Database connection closed.')
