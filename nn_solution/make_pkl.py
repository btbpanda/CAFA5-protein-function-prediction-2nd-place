import argparse
import copy
import logging
import os

import joblib
import numpy as np
import yaml


class Graph:
    """
    Ontology class. One ontology == one namespace
    DAG is the adjacence matrix (sparse) which represent a Directed Acyclic Graph where
    DAG(i,j) == 1 means that the go term i is_a (or is part_of) j
    Parents that are in a different namespace are discarded
    """

    def __init__(self, namespace, terms_dict, ia_dict=None, orphans=False):
        """
        terms_dict = {term: {name: , namespace: , def: , alt_id: , rel:}}
        """
        self.namespace = namespace
        self.dag = []  # [[], ...] terms (rows, axis 0) x parents (columns, axis 1)
        self.terms_dict = {}  # {term: {index: , name: , namespace: , def: }  used to assign term indexes in the gt
        self.terms_list = []  # [{id: term, name:, namespace: , def:, adg: [], children: []}, ...]
        self.idxs = None  # Number of terms
        self.order = None
        self.toi = None
        self.ia = None

        rel_list = []
        for self.idxs, (term_id, term) in enumerate(terms_dict.items()):
            rel_list.extend([[term_id, rel, term['namespace']] for rel in term['rel']])
            self.terms_list.append({'id': term_id, 'name': term['name'], 'namespace': namespace, 'def': term['def'],
                                    'adj': [], 'children': []})
            self.terms_dict[term_id] = {'index': self.idxs, 'name': term['name'], 'namespace': namespace,
                                        'def': term['def']}
            for a_id in term['alt_id']:
                self.terms_dict[a_id] = copy.copy(self.terms_dict[term_id])
        self.idxs += 1

        self.dag = np.zeros((self.idxs, self.idxs), dtype='bool')

        # id1 term (row, axis 0), id2 parent (column, axis 1)
        for id1, id2, ns in rel_list:
            if self.terms_dict.get(id2):
                i = self.terms_dict[id1]['index']
                j = self.terms_dict[id2]['index']
                self.dag[i, j] = 1
                self.terms_list[i]['adj'].append(j)
                self.terms_list[j]['children'].append(i)
                logging.debug("i,j {},{} {},{}".format(i, j, id1, id2))
            else:
                logging.debug('Skipping branch to external namespace: {}'.format(id2))
        logging.debug("dag {}".format(self.dag))
        # Topological sorting
        self.top_sort()
        logging.debug("order sorted {}".format(self.order))

        if orphans:
            self.toi = np.arange(self.dag.shape[0])  # All terms, also those without parents
        else:
            self.toi = np.nonzero(self.dag.sum(axis=1) > 0)[0]  # Only terms with parents
        logging.debug("toi {}".format(self.toi))

        if ia_dict is not None:
            self.set_ia(ia_dict)

        return

    def top_sort(self):
        """
        Takes a sparse matrix representing a DAG and returns an array with nodes indexes in topological order
        https://en.wikipedia.org/wiki/Topological_sorting
        """
        indexes = []
        visited = 0
        (rows, cols) = self.dag.shape

        # create a vector containing the in-degree of each node
        in_degree = self.dag.sum(axis=0)
        # logging.debug("degree {}".format(in_degree))

        # find the nodes with in-degree 0 (leaves) and add them to the queue
        queue = np.nonzero(in_degree == 0)[0].tolist()
        # logging.debug("queue {}".format(queue))

        # for each element of the queue increment visits, add them to the list of ordered nodes
        # and decrease the in-degree of the neighbor nodes
        # and add them to the queue if they reach in-degree == 0
        while queue:
            visited += 1
            idx = queue.pop(0)
            indexes.append(idx)
            in_degree[idx] -= 1
            l = self.terms_list[idx]['adj']
            if len(l) > 0:
                for j in l:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)

        # if visited is equal to the number of nodes in the graph then the sorting is complete
        # otherwise the graph can't be sorted with topological order
        if visited == rows:
            self.order = indexes
        else:
            raise Exception("The sparse matrix doesn't represent an acyclic graph")

    def set_ia(self, ia_dict):
        self.ia = np.zeros(self.idxs, dtype='float')
        for term_id in self.terms_dict:
            if ia_dict.get(term_id):
                self.ia[self.terms_dict[term_id]['index']] = ia_dict.get(term_id)
            else:
                logging.debug('Missing IA for term: {}'.format(term_id))
        # Convert inf to zero
        np.nan_to_num(self.ia, copy=False, nan=0, posinf=0, neginf=0)
        self.toi = np.nonzero(self.ia > 0)[0]


def ia_parser(file):
    ia_dict = {}
    with open(file) as f:
        for line in f:
            if line:
                term, ia = line.strip().split()
                ia_dict[term] = float(ia)
    return ia_dict


def obo_parser(obo_file, valid_rel=("is_a", "part_of")):
    """
    Parse a OBO file and returns a list of ontologies, one for each namespace.
    Obsolete terms are excluded as well as external namespaces.
    """
    term_dict = {}
    term_id = None
    namespace = None
    name = None
    term_def = None
    alt_id = []
    rel = []
    obsolete = True
    with open(obo_file) as f:
        for line in f:
            line = line.strip().split(": ")
            if line and len(line) > 1:
                k = line[0]
                v = ": ".join(line[1:])
                if k == "id":
                    # Populate the dictionary with the previous entry
                    if term_id is not None and obsolete is False and namespace is not None:
                        term_dict.setdefault(namespace, {})[term_id] = {'name': name,
                                                                        'namespace': namespace,
                                                                        'def': term_def,
                                                                        'alt_id': alt_id,
                                                                        'rel': rel}
                    # Assign current term ID
                    term_id = v

                    # Reset optional fields
                    alt_id = []
                    rel = []
                    obsolete = False
                    namespace = None

                elif k == "alt_id":
                    alt_id.append(v)
                elif k == "name":
                    name = v
                elif k == "namespace" and v != 'external':
                    namespace = v
                elif k == "def":
                    term_def = v
                elif k == 'is_obsolete':
                    obsolete = True
                elif k == "is_a" and k in valid_rel:
                    s = v.split('!')[0].strip()
                    rel.append(s)
                elif k == "relationship" and v.startswith("part_of") and "part_of" in valid_rel:
                    s = v.split()[1].strip()
                    rel.append(s)

        # Last record
        if obsolete is False and namespace is not None:
            term_dict.setdefault(namespace, {})[term_id] = {'name': name,
                                                            'namespace': namespace,
                                                            'def': term_def,
                                                            'alt_id': alt_id,
                                                            'rel': rel}
    return term_dict


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)

if __name__ == '__main__':

    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    DATA_DIR = config['base_path']  # sys.argv[1]
    MODEL_DIR = os.path.join(config['base_path'], config['models_path'], 'nn_serg')  # sys.argv[6]
    OUT_DIR = MODEL_DIR

    th_step = 0.01
    tau_arr = np.arange(0.01, 1, th_step)
    # Consider terms without parents, e.g. the root(s), in the evaluation
    no_orphans = False
    # Parse and set information accretion (optional)
    ia_dict = ia_parser(os.path.join(DATA_DIR, 'IA.txt'))

    # Parse the OBO file and creates a different graph for each namespace
    ontologies = []
    obo_file = os.path.join(DATA_DIR, 'Train', 'go-basic.obo')
    for ns, terms_dict in obo_parser(obo_file).items():
        ontologies.append(Graph(ns, terms_dict, ia_dict, not no_orphans))

    Y_labels = np.load(os.path.join(OUT_DIR, 'Y_labels.npy'), allow_pickle=True)

    idxs = []
    i = 0
    for term in Y_labels:
        for o in range(3):
            if term in ontologies[o].terms_dict:
                idxs.append((o, ontologies[o].terms_dict[term]['index'], term, i))
                i += 1

    idxs = sorted(idxs)
    borders = [sum([x[0] == i for x in idxs]) for i in range(3)]
    names = [x[1] for x in idxs]
    label_idxs = [x[3] for x in idxs]
    idxs = [x[1] for x in idxs]

    Y_oof = np.load(os.path.join(OUT_DIR, 'Y_pred_oof_blend.npy'))
    Y_test = np.load(os.path.join(OUT_DIR, 'Y_submit.npy'))

    Y_oof = Y_oof[:, label_idxs]
    Y_test = Y_test[:, label_idxs]

    joblib.dump({
        'pred': Y_oof,
        'test_pred': Y_test,
        'idx': idxs,
        'names': names,
        'borders': borders
    }, os.path.join(OUT_DIR, config['public_models']['nn_serg']['source']))
    # pytorch-keras-etc-3-blend-cafa-metric-etc.pkl'))
