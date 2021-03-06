from configparser import ConfigParser
import psycopg2
import rdflib
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pandas as pd
import random
import datetime
import decimal
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
    constants = {}
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
        result2 = list(filter(None,set(result2)))

        if dtype not in constants:
            constants[dtype] = []
        
        constants[dtype] += result2
    for dtype in constants:
        constants[dtype] = list(set(constants[dtype]))
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
# also return a dict that maps all tables to column/type pair list
def db2preds(cursor, schema, allowed_types=None):
    cursor.execute("select table_name, column_name, data_type from information_schema.columns where table_schema=%s", (schema,))
    tables = {}
    unaries = {}
    binaries = {}
    result = cursor.fetchall()
    for r1 in result:
        table1 = r1[0]
        column1 = r1[1]
        dtype1 = r1[2]

        if table1 not in tables:
            tables[table1] = []
        tables[table1].append((column1, dtype1))
        
        if allowed_types is not None and dtype1 not in allowed_types:
            continue
        if dtype1 not in unaries:
            unaries[dtype1] = []
        unaries[dtype1].append(r1)
        
        for r2 in result:
            table2 = r2[0]
            column2 = r2[1]
            dtype2 = r2[2]
            if allowed_types is not None and dtype2 not in allowed_types:
                continue
            if table1 == table2 and column1 != column2:
                if (dtype1,dtype2) not in binaries:
                    binaries[(dtype1, dtype2)] = []                
                binaries[(dtype1, dtype2)].append((r1, r2))
    return {"unary": unaries, "binary": binaries, "table": tables}

def eval_query(cursor, query):
    try:
        cursor.execute(query)
    except:
        print("Erroneous query, skipping it: ", query)
        return []
    
    result = cursor.fetchall()
    result = list(set(result))

    # remove null values
    result = list(filter((lambda r: None not in r), result))

    return result

def add_negative_facts(fact_dict, samplesize, db_constants=None):
    if db_constants is None:
        constants = []
        for p in fact_dict:
            tuples = fact_dict[p]
            for t in tuples:
                constants += list(t)
        constants = list(set(constants))
        print("Constants: ", len(constants))
    else:
        sql_types = db_constants.keys()
        supervision_types = {}
        for p in fact_dict:
            supervision_types[p] = [get_matching_types(r, sql_types) for r in fact_dict[p][0]]


    result = {}
    for p in fact_dict:
        tuples = fact_dict[p]
        random.shuffle(tuples)
        pos_tuples = tuples[:samplesize]
        pos_size = len(pos_tuples)

        arity = len(tuples[0])
        neg_tuples = []
        attempts = 10000
        while (len(neg_tuples) < pos_size) and (attempts > 0):
            attempts -= 1
            if db_constants is None:
                nt = random.choices(constants, k=arity)
            else:
                nt = []
                for types in supervision_types[p]:
                    t = random.choice(types)
                    nt.append(random.choice(db_constants[t]))
            nt = tuple(nt)
            if (nt not in tuples) and (nt not in neg_tuples):
                neg_tuples.append(nt)
        result[p] = {"pos":pos_tuples, "neg":neg_tuples}
    return result

def get_support(predicate, tuples, rules):
    targets = {}
    for t in tuples:
        targets_for_t = []
        atom = [predicate] + list(t)
        for r in rules:
            mappings_curr, targets_curr = r.get_support(atom)
            targets_for_t += targets_curr
        if len(targets_for_t) > 0:
            targets[tuple(atom)] = targets_for_t
    return targets

        
def create_supervision(cursor, predicate, query, constants, rules, pos_size, neg_support_limit=100):

    # first collect positive facts and all their proofs
    try:
        cursor.execute(query)
    except:
        print("Erroneous query, skipping it: ", query)
        return [], [], [], []
    
    result = cursor.fetchall()
    result = list(set(result))
    print("ROWS FOUND: ", len(result))

    # remove null values
    result_filtered = []
    for r in result:
        if None not in r:
            result_filtered.append(r)
    result = result_filtered
    random.shuffle(result)
    
    pos_size = min(pos_size, len(result))
    pos_tuples = result[:pos_size]
    pos_mappings = []
    pos_targets = {}
    for pt in pos_tuples: # collect proofs for each fact
        atom = [predicate] + list(pt)
        targets_for_pt = []
        for r in rules:
            pos_mappings_curr, pos_targets_curr = r.get_support(atom)
            targets_for_pt += pos_targets_curr
            pos_mappings += pos_mappings_curr

        if len(targets_for_pt) > 0:
            pos_targets[tuple(atom)] = targets_for_pt


    # create matching number of negative proofs
    pos_mappings_size = len(pos_mappings)
    if pos_mappings_size == 0:
        print("No positives found for predicate", predicate)
        return [], [], [], []

    arity = len(result[0])
    sql_types = constants.keys()
    supervision_types = [get_matching_types(r, sql_types) for r in result[0]]

    neg_mappings = []
    neg_targets = {}
    neg_tuples = []
    attempts = 1000
    # while (len(neg_mappings) < pos_mappings_size) and attempts > 0:
    while (len(neg_tuples) < len(pos_tuples)) and attempts > 0:
        attempts -= 1
        # generate a random tuple of matching type
        nt = []
        for types in supervision_types:
            t = random.choice(types)
            nt.append(random.choice(constants[t]))
        nt = tuple(nt)
        if nt in result:
            continue
        if nt in neg_tuples:
            continue
        else:
            # collect proofs for each negative fact
            neg_tuples.append(nt)
            atom = tuple([predicate] + list(nt))
            # print("neg fact: ", atom)
            neg_targets[atom] = []
            for r in rules:
                neg_mappings_curr, neg_targets_curr = r.get_support([predicate] + list(nt))
                neg_mappings_curr = neg_mappings_curr[:neg_support_limit]
                neg_targets_curr = neg_targets_curr[:neg_support_limit]
                neg_targets[atom] += neg_targets_curr
                neg_mappings += neg_mappings_curr

            if len(neg_targets[atom]) == 0:
                del neg_targets[atom]

    print("Proofs generated for predicate {}: {}/{} positives, {}/{} negatives".format(predicate, len(pos_targets), len(pos_mappings), len(neg_targets), len(neg_mappings)))
    return pos_mappings, pos_targets, neg_mappings, neg_targets

def pred2name(pred):
    # return "|".join(pred)
    return " ".join(pred) + " PREDEND"

def visualise_mapping_dict(mapping_dict):
    pred_dict = {}
    for isPositive in (True, False):
        mappings = mapping_dict[isPositive]
        for m in mappings:
            for pred in m:
                pname = pred2name(m[pred][0])
                if pname not in pred_dict:
                    pred_dict[pname] = {True:0, False:0}
                pred_dict[pname][isPositive] += 1
    print("\nPredicates in the mappings:")
    for p in pred_dict:
        print("   {}: {}".format(p, pred_dict[p]))
    print("")

def tokenize(mystring):
    return mystring.replace("/", " / ")


def type_match(sql_type, obj):
    if sql_type == "NULL":
        result = obj is None
    elif sql_type == "boolean":
        result = isinstance(obj, bool)
    elif sql_type in ("real", "double"):
        result = isinstance(obj, float)
    elif sql_type in ("smallint", "integer", "bigint"):
        result = isinstance(obj, int) and not isinstance(obj, bool)
    elif sql_type == "numeric":
        result = isinstance(obj, decimal.Decimal)
    elif sql_type in ("varchar", "text", "character varying", "character"):
        result = isinstance(obj, str)
    elif sql_type == "date":
        result = isinstance(obj, datetime.date)
    elif sql_type in ("time", "timetz"):
        result = isinstance(obj, datetime.time)
    elif sql_type in ("datetime", "datetimetz"):
        result = isinstance(obj, datetime.datetime)
    elif sql_type == "interval":
        result = isinstance(obj, datetime.timedelta)
    elif sql_type == "ARRAY":
        result = isinstance(obj, list)
    else:
        result = True
    return result

def get_matching_types(x, sql_types):
    result = []
    for sql_type in sql_types:
        if type_match(sql_type, x):
            result.append(sql_type)
    return result
