from configparser import ConfigParser
import psycopg2
import rdflib
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pandas as pd
import random

# parse db configurations from file
def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db

def init_db():
    params = config()
    conn = psycopg2.connect(**params)
    cursor = conn.cursor()
    return cursor

# print table
def print_result(table):
    print("Found {} rows:".format(len(table)))
    for t in table:
        print(t)

def inspect_schema(cursor, schema):
    s = "select table_name, column_name, data_type from information_schema.columns where table_schema='{}' ORDER BY table_name".format(schema)
    cursor.execute(s)
    result = cursor.fetchall()
    print("\nTables for schema '{}':".format(schema))
    print_result(result)

def table_names(cursor, schema):
    s = "select table_name from information_schema.columns where table_schema='{}'".format(schema)
    cursor.execute(s)
    result = cursor.fetchall()
    return [r[0] for r in result]

def attributes(cursor, schema):
    s = "select table_name, column_name, data_type from information_schema.columns where table_schema='{}'".format(schema)
    cursor.execute(s)
    result = cursor.fetchall()
    attributes = DictOfList()
    for r in result:
        attributes.add(r[0], [r[1], r[2]])
    return attributes

def db_types(db_attributes):
    types = []
    for table in db_attributes.keys():
        for n, t in db_attributes.get(table):
            types.append(t)
    types = list(set(types))
    return types

def inspect_database(cursor):
    s = "SELECT * FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';"
    cursor.execute(s)
    result = cursor.fetchall()
    print("\nAll tables for all schemas:")
    schema = ""
    for row in result:
        if row[0] != schema:
            schema = row[0]
            print(schema)
        print("    {}".format(row[1]))

def schema_constants(cursor, schema, allowed_types=None):
    s = "select table_name, column_name, data_type from information_schema.columns where table_schema='{}'".format(schema)
    cursor.execute(s)
    result = cursor.fetchall()
    constants = []
    for r in result:
        table = r[0]
        column = r[1]
        dtype = r[2]
        if allowed_types is not None and dtype not in allowed_types:
            continue
        s = "select distinct \"{}\" from \"{}\";".format(column, table)
        cursor.execute(s)
        result2 = cursor.fetchall()
        result2 = [r[0] for r in result2]
        result2 = list(set(result2))
        constants += result2
    constants = list(set(constants))
    return constants

def sparql_result_to_list(qres, colname):
    result = []
    for row in qres:
        if isinstance(row[colname], rdflib.term.URIRef): # TODO UNDERSTAND WHAT TO DO WITH BNode-s (blank nodes)
            result.append(row[colname].n3())
    return result


class DictOfList:
    def __init__(self):
        self.data = {}

    def add(self, key, value):
        if key in self.data:
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def keys(self):
        return list(self.data.keys())

    def get(self, key):
        return self.data[key]

    def values(self):
        v = []
        for k in self.data:
            v += self.data[k]
        return v

# TODO explore other alternatives than character level bleu score
def compare_strings(reference, candidate):
    reference = list(reference)
    candidate = list(candidate)
    return sentence_bleu([reference], candidate)
    

def top_candidates(reference, candidates, threshold=0.01):
    
    reference0 = list(reference.lower())
    candidates0 = [list(c[0].lower()) for c in candidates]
    if len(reference0) < 4:
        weights = [1/len(reference0) for _ in range(len(reference0))]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
    sim_scores = [sentence_bleu([reference], c, weights=weights) for c in candidates0]
    sim_scores2 = [ (sim_scores[i],i) for i in range(len(sim_scores)) ]
    sim_scores2.sort(reverse=True)
    sorted_scores, perm = zip(*sim_scores2)
    sorted_candidates = [candidates[p][0] for p in perm]
    best = []
    best_scores = []
    
    last_score = 1.0
    for score, candidate in zip(sorted_scores, sorted_candidates):
        if score < last_score * threshold:
            return best, best_scores
        else:
            best.append(candidate)
            best_scores.append(score)
            last_score = score
    return best, best_scores

# TODO alternatives
def compose_scores_and(scores):
    return np.prod(scores)


    
def groupby_max(rows, max_index):
    if len(rows) == 0:
        return rows
    cols = np.max([len(r) for r in rows])
    groupby_indices = list(range(cols))
    groupby_indices.remove(max_index)
    df = pd.DataFrame(rows).groupby(groupby_indices, dropna=False).max().reset_index()
    df = df[range(cols)]
    result = df.stack().groupby(level=0).apply(list).tolist()
    return result
    

# collect all table/column pairs and table/column1/column2 triples
def db2preds(cursor, schema):
    cursor.execute("select table_name, column_name, data_type from information_schema.columns where table_schema=%s", (schema,))
    unaries = []
    binaries = []
    result = cursor.fetchall()
    for r1 in result:
        unaries.append(r1)
        for r2 in result:
            if r1[0] == r2[0] and r1[1] != r2[1]:
                binaries.append((r1, r2))
    return {"unary": unaries, "binary": binaries}

def create_supervision(cursor, schema, predicate, query, size, constants):
    cursor.execute(query)
    result = cursor.fetchall()
    result = list(set(result))
    assert len(result) > 0
    arity = len(result[0])
    supervision = []

    for i in range(size):
        if i >= len(result):
            break
        positive = [predicate] + list(result[i])
        while True:
            tup = random.sample(constants, arity)
            if tuple(tup) not in result:
                negative = [predicate] + tup
                break
        supervision.append((positive, True))
        supervision.append((negative, False))
    return supervision


