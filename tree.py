import numpy as np

class Tree:
    root = 'r'

    def __init__(self):
        self.parent = {}
        self.children = {}
        self.node_id = {}
        self.id_node = {}

        self.insert_node(self.root, None)
    
    @property
    def num_nodes(self):
        return len(self.node_id)
    
    @staticmethod
    def get_ancestors(node_path):
        ancestors = []
        for i, c in enumerate(node_path):
            if c == '_':
                ancestors.append(node_path[:i])
        ancestors.append(node_path)
        return ancestors
    
    def insert_node(self, node_str, par_str):
        if node_str in self.node_id:
            return
        nid = len(self.node_id)
        self.node_id[node_str] = nid
        self.id_node[nid] = node_str
        self.parent[node_str] = par_str
        self.children[node_str] = []
        if par_str is not None:
            self.children[par_str].append(node_str)
    
    def insert_seq(self, node_path):
        ancestors = self.get_ancestors(node_path)
        par = self.root
        for a in ancestors:
            self.insert_node(a, par)
            par = a
    
    def get_ancestor_ids(self, node_str):
        ids = []
        node = node_str
        while node is not None:
            ids.append(self.node_id[node])
            node = self.parent[node]
        return ids
    
    def precompute(self):
        self.init_levels()
        self.init_matrix()
    
    def init_matrix(self):
        n = len(self.node_id)
        self.leaf_matrix = np.zeros((n, n), dtype=np.float32)
        self.ancestor_matrix = np.zeros((n, n), dtype=np.float32)
        self.adj_matrix = np.zeros((n, n), dtype=np.float32)
        self.parent_matrix = np.zeros((n, n), dtype=np.float32)
        self.leaf_vector = np.zeros(n, dtype=np.float32)

        self._dfs(self.root, [])
    
    def _dfs(self, node_str, ancestors):
        nid = self.node_id[node_str]
        if len(ancestors):
            par = ancestors[-1]
            self.adj_matrix[par, nid] = 1
            self.adj_matrix[nid, par] = 1
            self.parent_matrix[nid, par] = 1
        ancestors = ancestors + [nid]
        self.ancestor_matrix[nid, ancestors] = 1
        if len(self.children[node_str]) == 0:  # leaf
            self.leaf_matrix[ancestors, nid] = 1
            self.leaf_vector[nid] = 1
        else:
            for ch in self.children[node_str]:
                self._dfs(ch, ancestors)
    
    def init_levels(self):
        self.levels = {}
        self._levels_rec(self.root, 0)
    
    def _levels_rec(self, node_str, depth):
        if depth not in self.levels:
            self.levels[depth] = []
        self.levels[depth].append(self.node_id[node_str])
        for ch in self.children[node_str]:
            self._levels_rec(ch, depth+1)


def main():
    tree = Tree()
    tree.insert_seq('food_2_1')
    tree.insert_seq('food_2_2')
    tree.insert_seq('hobbies_1')
    tree.insert_seq('hobbies_2')

    tree.precompute()

    print(tree.parent)
    print(tree.children)
    print(tree.node_id)
    print(tree.id_node)

    print(tree.leaf_matrix)
    print(tree.adj_matrix)
    print(tree.ancestor_matrix)

    # data = M5Data()
    # print(data.ts_data)
    # idx = data.tree.leaf_vector.astype(np.bool)
    # diff = data.ts_data[:, 0] - np.mean(data.ts_data[:, idx], axis=1)
    # diff = np.abs(diff)
    # print(np.sum(diff))
    # print(data.ts_data.dtype, data.ts_data.shape)

    # dataset = data.tf_dataset(True)
    # for d in dataset:
    #     feats = d[0]
    #     y_obs = d[1]
    #     nid = d[2]
    #     sw = d[3]
    #     print(feats[0].shape)
    #     print(feats[1][0].shape, feats[1][1].shape)
    #     print(y_obs.shape)
    #     print(nid.shape, sw.shape)
    #     break

    # for d in tqdm(data.train_gen()):
    #     pass

    # dataset = data.tf_dataset(True)
    # for d in tqdm(dataset):
    #     d[0]

    # print(data.weights)
    # print(np.sum(data.weights))
    # print(data.weights.shape)

if __name__ == "__main__":
    main()
