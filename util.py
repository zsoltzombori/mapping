from configparser import ConfigParser
import psycopg2
import rdflib
from nltk.translate.bleu_score import sentence_bleu

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
    s = "select table_name, column_name from information_schema.columns where table_schema='{}'".format(schema)
    cursor.execute(s)
    result = cursor.fetchall()
    print("\nTables for schema '{}':".format(schema))
    print_result(result)

def table_names(cursor, schema):
    s = "select table_name from information_schema.columns where table_schema='{}'".format(schema)
    cursor.execute(s)
    result = cursor.fetchall()
    return [r[0] for r in result]

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
        return self.data.keys()

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
    reference0 = list(reference)
    candidates0 = [list(c) for c in candidates]
    if len(reference0) < 4:
        weights = [1/len(reference0) for _ in range(len(reference0))]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
    sim_scores = [sentence_bleu([reference], c, weights=weights) for c in candidates0]
    sim_scores2 = [ (sim_scores[i],i) for i in range(len(sim_scores)) ]
    sim_scores2.sort(reverse=True)
    sorted_scores, perm = zip(*sim_scores2)
    sorted_candidates = [candidates[p] for p in perm]
    result = []
    last_score = 1.0
    for score, candidate in zip(sorted_scores, sorted_candidates):
        if score < last_score * threshold:
            return result
        else:
            result.append(candidate)
            last_score = score
    return result
