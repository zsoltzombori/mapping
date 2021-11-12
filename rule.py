import datetime
import decimal

import util

class Variable:
    def __init__(self, number):
        self.n = number

class Constant:
    def __init__(self, number):
        self.n = number

# an atom is a list
# the first element is the predicate
# subsequent elements are constants or Variables 
class Rule:
    def __init__(self, spec, cursor, preds):
        self.head = spec[0]
        self.body = spec[1:]
        self.cursor = cursor
        self.preds = preds

    # return all the mappings that yield a proof of fact
    def get_support(self, fact):
        if len(fact) != len(self.head):
            return [], []
        if fact[0] != self.head[0]:
            return [], []

        subst = {}
        target = ["SOS", self.head[0]]
        for h_arg, f_arg in zip(self.head[1:], fact[1:]):
            if isinstance(h_arg, Variable):
                subst[h_arg.n] = f_arg
                target.append("var_{}".format(h_arg.n))
            elif h_arg != f_arg:
                return [], []
        target.append("EOH")
        
        return self.candidate_mappings(self.body, subst, {}, target)


    def unifying_facts(self, atom):
        args = atom[1:]
        if len(args) == 1: # unary predicate
            assert not isinstance(args[0], (Variable, Constant)), "Unary argument should not be variable/constant"
            targets = self.preds["unary"]
            unary = True
        elif len(args) == 2: #binary predicates
            assert (not isinstance(args[0], (Variable, Constant))) or (not isinstance(args[1], (Variable, Constant))), "Both arguments should not be variable/constant"
            targets = self.preds["binary"]
            unary = False
        else:
            assert False, "Only unary/binary predicates are supported"

        temp = []
        for t in targets:
            temp += targets[t]
        targets = temp

        matches = []
        if unary:
            for t in targets:
                table0, column0, type0 = t
                sql = "SELECT DISTINCT \"{}\" from \"{}\" where \"{}\"=%s".format(column0, table0, column0)
                if util.type_match(type0, args[0]):
                    self.cursor.execute(sql, (args[0],))
                    result = self.cursor.fetchall()
                    for r in result:
                        if r[0] is not None:
                            matches.append( (atom[0], (table0, column0), (r[0],)))
        else:
            for t in targets:
                ((table0, column0, type0), (table1, column1, type1)) = t
                assert table0 == table1
                sql = "SELECT DISTINCT \"{}\", \"{}\" from \"{}\"".format(column0, column1, table0)

                result = None
                if isinstance(args[0], (Variable, Constant)):
                    if util.type_match(type1, args[1]):
                        self.cursor.execute(sql + " where \"{}\"=%s".format(column1),  (args[1],))
                        result = self.cursor.fetchall()
                elif isinstance(args[1], (Variable, Constant)):
                    if util.type_match(type0, args[0]):
                        self.cursor.execute(sql + " where \"{}\"=%s".format(column0),  (args[0],))
                        result = self.cursor.fetchall()
                elif util.type_match(type0, args[0]) and util.type_match(type1, args[1]):
                    self.cursor.execute(sql + " where \"{}\"=%s and \"{}\"=%s".format(column0, column1),  (args[0], args[1]))
                    result = self.cursor.fetchall()

                if result is not None:
                    for r in result:
                        if r[0] is not None and r[1] is not None:
                            matches.append((atom[0], (table0, column0, column1), (r[0], r[1])))

        return matches

    def candidate_mappings(self, body, subst, mapping, target):

        if len(body) == 0:
            return [mapping], [target + ["EOS"]]
        
        body_subst = [apply_subst(b, subst) for b in body]
        # TODO select the one with the least amount of variables
        B = body_subst[0][1]
        B2 = body[0]
        body2 = body[1:]

        mappings_out = []
        targets_out = []

        # collect all facts that unify with B
        # have the same arity as B and has the same constant in the same position
        # since Subst(B) contains some constants, Facts is much smaller than the dataset
        facts = self.unifying_facts(B)
        for f in facts:
            constants_match = True

            # we don't want an unary predicate that is just the projection of an existing binary predicate
            redundant_pred = False
            for m in mapping:
               p,v = mapping[m]
               if f[1] == p and f[2] == v:
                   redundant_pred = True
                   break
               if len(p) == 3:
                   if f[1] == (p[0], p[1]) and f[2][0] == v[0]:
                       redundant_pred = True
                       break
                   if f[1] == (p[0], p[2]) and f[2][0] == v[1]:
                       redundant_pred = True
                       break
            if redundant_pred:
                break
            
            #     Mapping2 is Mapping + {pred(B) -> pred(F)}
            mapping2 = mapping.copy()
            mapping2[f[0]] = (f[1], f[2])

            args = []
            for a, v in zip(B2[1:], f[2]):
                if isinstance(a, Variable):
                    args.append("var_{}".format(a.n))
                elif isinstance(a, Constant):
                    if isinstance(v, str):
                        v_parts = v.split()
                        v = "_".join(v_parts)
                    args.append(v)

            target2 = target + [util.pred2name(f[1])] + args + ["EOP"]

            #     Subs2 is Subst + mgu(Subst(B), F)
            subst2 = subst.copy() # shallow copy is fine here, I think
            for arg, value in zip(B[1:], f[2]):
                if isinstance(arg, Variable):
                    subst2[arg.n] = value
                if isinstance(arg, Constant):
                    if arg.n in mapping2:
                        if mapping2[arg.n] != value:
                            constants_match = False
                            break
                    else:
                        mapping2[arg.n] = (("const",), (value,))

            if constants_match:
                new_mappings, new_targets = self.candidate_mappings(body2, subst2, mapping2, target2)
                mappings_out += new_mappings
                targets_out += new_targets
        return mappings_out, targets_out

   
   #     score = embeddingScore(Mapping2)
   #     # if there are too many candidate facts in Facts, sample from them based on score
   #     # for negative samples, it is ok to use a threshold for score
   #     # for positive samples, farther candidates should have a chance too

def apply_subst(atom, subst):
    args = atom[1:]
    args2 = []
    varcnt = 0
    for a in args:
        if isinstance(a, Variable) and a.n in subst:
            a2 = subst[a.n]
        else:
            a2 = a
        if isinstance(a2, Variable):
            varcnt += 1
        args2.append(a2)
    return varcnt, atom[:1] + args2

