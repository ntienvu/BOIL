import operator
import numpy as np

class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        self.it_capacity = 1
        while self.it_capacity < capacity:
            self.it_capacity *= 2
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * self.it_capacity)
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
    
    
    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.it_capacity
        
        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update the leaf
        self.update(tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
            
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, a, b):
        v = np.random.uniform(a, b)
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.it_capacity
        if data_index >= self.capacity:
            return self.get_leaf(a,b)
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node

    @property
    def min_weight(self):
        valid = np.where(self.tree[self.it_capacity:self.it_capacity + self.capacity] > 0.0)
        return self.tree[self.it_capacity:self.it_capacity + self.capacity][valid].min()

    @property
    def max_weight(self):
        return np.max(self.tree[self.it_capacity:self.it_capacity + self.capacity])

class SumSegmentTree(object):
    def __init__(self, capacity):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = np.zeros((2 * self._capacity))

    def reset(self):
        self._value = np.zeros((2 * self._capacity))

    def sum(self):
        return self._value[1]

    def find_prefixsum_idx(self, prefixsum):
        #assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            idx *= 2
            if self._value[ idx] < prefixsum:
                prefixsum -= self._value[idx]
                idx += 1
        return idx - self._capacity

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        diff = val - self._value[idx]
        self.last_set.append((idx, self._value[idx], val, diff))
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] += diff
            idx //= 2

    def __getitem__(self, idx):
        #assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class MaxSegmentTree(object):
    def __init__(self, capacity):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = np.zeros((2 * self._capacity))

    def reset(self):
        self._value = np.zeros((2 * self._capacity))

    def max(self):
        return self._value[1]
    
    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        
        while idx > 1 and self._value[idx] == val:
            idx //= 2
            self._value[idx] = max(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )

    def __getitem__(self, idx):
        #assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

def test():
    # s = SumSegmentTree(4)
    # s[0] = 0.1
    # s[1] = 0.3
    # s[2] = 1.0
    # s[3] = 0.6
    # print(s._value)
    # print("SUM ", s.sum())
    # print("Items ", s[np.arange(4)])
    # print("0.05", s.find_prefixsum_idx(0.05))
    # print("0.15", s.find_prefixsum_idx(0.15))
    # print("0.45", s.find_prefixsum_idx(0.45))
    # print("1.45", s.find_prefixsum_idx(1.45))
    # print("2.05", s.find_prefixsum_idx(2.05))

    # m = MaxSegmentTree(8)
    # m[0] = 0.1
    # m[1] = 0.3
    # m[2] = 1.0
    # m[3] = 0.6
    # m[4] = 2.0
    # m[5] = 0.5
    # m[6] = 0.2
    # m[7] = 0.6
    # print(m._value)
    s = SumTree(10)
    s.add(.5,"a")
    print(s.get_leaf(0.4))
if __name__ == '__main__':
    test()