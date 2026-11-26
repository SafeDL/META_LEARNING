import numpy.random as ra


def add_info(root, roads, parent=None):
    """Attach parent links, failing counts, and selectable counts to tree nodes."""
    root.parent = parent
    if root.count == 1:
        root.fail_count = 1 if roads[root.id].is_failing else 0
        root.selectable_count = 1 if roads[root.id].is_selectable else 0
    else:
        add_info(root.left, roads, root)
        add_info(root.right, roads, root)
        root.fail_count = root.left.fail_count + root.right.fail_count
        root.selectable_count = root.left.selectable_count + root.right.selectable_count


def decrease_selectable_count(node):
    """Decrement selectable_count along the ancestor path."""
    node.selectable_count -= 1
    if node.parent is not None:
        decrease_selectable_count(node.parent)


def get_leafs_of_tree(root):
    """Return leaf nodes of the subtree rooted at root."""
    if root.count == 1:
        return [root]
    return get_leafs_of_tree(root.left) + get_leafs_of_tree(root.right)


class DETOUR:
    """This is the main class defining DETOUR test selection/prioritization
    procedures."""

    def __init__(self, executed_roads, not_executed_roads, road_clusterer, random_seed=0):
        """DETOUR expects a list of executed roads, a list of not executed roads (selectable roads)
        to select from/prioritize.
        In addition, it expects a clusterer derived from extending the HierarchicalClusterer class and
        a feature extractor object derived from a class that extends FeatureExtractor."""
        self.executed_roads = executed_roads
        self.not_executed_roads = not_executed_roads
        self.roads = executed_roads + not_executed_roads
        self.road_clusterer = road_clusterer
        ra.seed(random_seed)

    @staticmethod
    def get_distance(distance_matrix, n, i, j):
        """Given a distance matrix (in vector form) showing pairwise distance between n objects,
        return the distance between ith and jth objects."""
        return distance_matrix[n * i + j - ((i + 2) * (i + 1)) // 2]

    @staticmethod
    def choose_from_selectable_subtree(root, distance_matrix, n):
        """Given the root of a tree of nodes that represent selectable/failing tests
        with pairwise distances between all n nodes in the tree given in distance_matrix, return the node
        representing the selectable test in the tree that is closest to a failing test."""
        leafs = [node for node in get_leafs_of_tree(root)]
        selectables = [node for node in leafs if node.selectable_count == 1]
        failing_oracles = [node for node in leafs if node.fail_count == 1]
        tuples = [(f, s, DETOUR.get_distance(distance_matrix, n, f.id, s.id)) for f in failing_oracles for s in selectables]
        sorted_tuples = sorted(tuples, key=lambda e: e[2])
        return sorted_tuples[0][1]

    @staticmethod
    def retrieve(root, distance_matrix, n):
        """This method chooses a leaf node of a tree of nodes that represent selectable/failing
        tests with pairwise distances between all n nodes in the tree given in distance_matrix.
        The selection procedure ensures that the selected node represents a selectable test
        case that has desirable properties (being close to a test that is known to be failing)
        and diversity is promoted through exploration via probabilistic selection of
        branches to choose the node from."""

        if root.count == 1:
            return root

        left = root.left
        right = root.right

        # Short forms of boolean variables
        ls = left.selectable_count > 0
        rs = right.selectable_count > 0
        lf = left.fail_count > 0
        rf = right.fail_count > 0

        # Calculate fail ratios
        lfr = 0
        if lf:
            lfr = left.fail_count / (left.count - left.selectable_count)

        rfr = 0
        if rf:
            rfr = right.fail_count / (right.count - right.selectable_count)

        # If at least on of two subtrees (left or right) is (isFailing, isSelectable)
        # We choose one randomly with probability proportional to
        # (#failOracle/#totalOracle) * isSelectable
        if (lf and ls) or (rf and rs):
            left_value = lfr
            if not ls:
                left_value = 0

            right_value = rfr
            if not rs:
                right_value = 0

            probabilities = [left_value / (left_value + right_value),
                             right_value / (left_value + right_value)]

            # Select a branch probabilistically to promote diversity
            new_root = ra.choice([left, right], p=probabilities)

            # Choose the node from the selected branch
            return DETOUR.retrieve(new_root, distance_matrix, n)
        else:
            return DETOUR.choose_from_selectable_subtree(root, distance_matrix, n)

    @staticmethod
    def get_oracle_ids(roads):
        """This function retrieves indices (ids for the dendogram) of roads
        that serve eas oracles (executed tests that are not selectable)."""
        return [i for i in range(len(roads)) if not roads[i].is_selectable]

    @staticmethod
    def m_closest_oracle_ids(distance_matrix, n, oracle_ids, m, i):
        """This function provides the indices (ids for the dendogram) of m roads
        that are feature-wise closest to a road with a given id i."""
        m = min([len(oracle_ids), m])  # in case m is too large?
        tuples = [(oracle_id, DETOUR.get_distance(distance_matrix, n, i, oracle_id)) for oracle_id in oracle_ids]
        sorted_tuples = sorted(tuples, key=lambda e: e[1])
        return [rtuple[0] for rtuple in sorted_tuples][:m]

    @staticmethod
    def is_m_closest_oracle_all_passing(roads, distance_matrix, n, oracle_ids, m, i):
        """This function checks if the m roads that are feature-wise closest to
        given road (with id i) are all not-failing (i.e., passing)."""
        ids = DETOUR.m_closest_oracle_ids(distance_matrix, n, oracle_ids, m, i)
        count = 0
        for oid in ids:
            if not roads[oid].is_failing:
                count += 1
        return count >= m

    def select(self,
               min_select_ratio=0.05,
               max_select_ratio=0.4,
               m_closest_neighbor_count=4,
               w_selection_threshold=4):
        """This method uses Retrieve function to select a number of not-executed (selectable)
        roads based on given parameters:
        min_select_ratio (number of selected roads / total not-executed (selectable) road count is at least min_select_ratio)
        max_select_ratio (number of selected roads / total not-executed (selectable) road count is at most max_select_ratio)
        m_closest_neighbor_count (m)
        w_selection_threshold (w)
        such that selection is stopped early and for each of the last w selected tests,
        m closest executed tests are all passing. """
        root, distance_matrix = self.road_clusterer.cluster(self.roads)
        add_info(root, self.roads)
        selected_nodes = []

        selectable_count = root.selectable_count
        min_count = max([1, int(min_select_ratio * selectable_count)])
        max_count = max([1, int(max_select_ratio * selectable_count)])

        current_count = 0
        oracle_ids = DETOUR.get_oracle_ids(self.roads)
        n = len(self.roads)
        questionable_selectable_count = 0
        while True:
            if root.selectable_count == 0:
                break
            selected_node = DETOUR.retrieve(root, distance_matrix, len(self.roads))
            selected_nodes.append(selected_node)
            decrease_selectable_count(selected_node)
            current_count += 1

            if DETOUR.is_m_closest_oracle_all_passing(self.roads, distance_matrix, n, oracle_ids, m_closest_neighbor_count, selected_node.id):
                questionable_selectable_count += 1
            else:
                questionable_selectable_count = 0

            if current_count == max_count:
                break

            # Beyond this point, selected tests may be too far from
            # failing tests. Therefore, we stop selection.
            if current_count >= min_count and questionable_selectable_count >= w_selection_threshold:
                break


        return [self.roads[node.id] for node in selected_nodes]

    def prioritize(self, select_ratio):
        """This method uses Retrieve functuon to prioritize roads among select_ratio ration of
        those that are not-executed (selectable). It works by passing the request to select method
        by setting min_select_ratio and max_select ratio. When those are equal,
        windowed selection via parameters m_closest_neighbor_count and w_selection_threshold
        does not apply."""
        return self.select(min_select_ratio=select_ratio, max_select_ratio=select_ratio)