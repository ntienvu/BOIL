import sys

from collections import namedtuple
import numpy as np
from bayes_opt.test_functions.drl.segment_tree import SumSegmentTree, MaxSegmentTree, SumTree

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer(object):
    def __init__(self, capacity, batch):
        self.memory = []
        self._maxsize = capacity
        self._next_idx = 0
        self._batch_size = batch

    def __len__(self):
        return len(self.memory)

    def add(self, s, action, reward, s_n, done):
        if len(self.memory) < self._maxsize:
            self.memory.append(None)
        self.memory[self._next_idx] = Transition(s, action, reward, s_n, done)
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, beta = 0):
        idxes = list(np.random.randint( len(self), size=(self._batch_size,)))
        return (idxes, [1/self._batch_size] * self._batch_size, self._encode_samples(idxes))

    def _encode_samples(self, idx):
        return [self.memory[i] for i in idx]
        
    def update_priorities(self, idx, weights):
        pass

    def reset(self):
        self.memory = []
        self._next_idx = 0

class SumSegmentTreePERBuffer(ReplayBuffer):
    def __init__(self, size, batch, alpha, min_w = 1e-2):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(SumSegmentTreePERBuffer, self).__init__(size, batch)
        assert alpha >= 0, alpha
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self.min_w = min_w
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_max = MaxSegmentTree(it_capacity)

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum.last_set = []
        self._it_sum[idx] = self._it_max.max() + self.min_w
        self._it_max[idx] = self._it_sum[idx]

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum()
        for _ in range(batch_size):
            mass = np.random.rand() * p_total
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return np.array(res)

    def sample(self, beta):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights (0 - no corrections, 1 - full correction)
        """
        assert beta > 0

        idxes = self._sample_proportional(self._batch_size)
        weights = (self._it_sum[idxes] / self._it_sum.sum() * len(self.memory) ) ** (-beta)
        try:
            return (idxes, weights, self._encode_samples(idxes))
        except e:
            print("bug")
            pass

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to transitions at the sampled idxes denoted by variable `idxes`.
        """
        # assert len(idxes) == len(priorities)
        priorities = priorities ** self._alpha
        self._it_sum.last_set = []
        for idx, priority in zip(idxes, priorities):
            # assert (priority >= 0).all(), priority
            # assert (0 <= idx).all() and (idx < len(self.memory)).all()
            self._it_sum[idx] = priority + self.min_w
            self._it_max[idx] = priority + self.min_w

    def reset(self):
        super().reset()
        self._it_max.reset()
        self._it_sum.reset()

class SumTreePERBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity, batch, alpha, min_w= 0.01):
        # Making the tree 
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)
        self.batch_size = batch
        self.PER_e = min_w  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = alpha  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
            
    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def add(self, s, action, reward, s_n, done):
        # Find the max priority
        max_priority = self.tree.max_weight
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, Transition(s, action, reward, s_n, done))   # set the max p for new p

        
    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, beta):
        # Create a sample array that will contains the minibatch
        memory_b = []
        n = self.batch_size
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Calculating the max_weight
        p_min = self.tree.min_weight / self.tree.total_priority
        max_weight = (p_min * n) ** (-beta)
        
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(a,b)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i] = np.power(n * sampling_probabilities, -beta)/ max_weight
                                   
            b_idx[i]= index
            
            memory_b.append(data)
        return b_idx, b_ISWeights, memory_b
    
    """
    Update the priorities on the tree
    """
    def update_priorities(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
