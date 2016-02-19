"""Various gossip algorithms.

This module contains implementation of several gossip algorithms for computing
the sample mean [1], partials U-statistics [2] or complete U-statistics [2, 3].

Standard Gossip Averaging ([1]): Sample mean estimates are initialized to the
current observation. At each time step, selected nodes average their estimates.

U1-Gossip Algorithm ([2]): Sample mean estimates are initialized to 0. Each node
 also holds an auxiliary observation, initialized to the current observation. At
 each time step, selected nodes swap their auxiliary observations and every node
 updates its estimate.

U2-Gossip Algorithm ([2]). Sample mean estimates are initialized to 0. Each node
also holds two auxiliary observations, both initialized to the current
observation. At each time step, two edges are picked and selected nodes swap one
of their auxiliary observations; then, every node updates its estimate.

GoSta Algorithms ([3]). Sample mean estimates are initialized to 0. Each node
also holds an auxiliary observation, initialized to the current observation. At
each time step, selected nodes swap their auxiliary observations and every node
or only selected nodes update their estimates, depending on the setting
(synchronous/asynchronous).

[1] Randomized Gossip Algorithms. S. Boyd, A. Ghosh, B. Prabhakar, D. Shah. IEEE
Transactions on Information Theory, June 2006, 52(6):2508-2530.

[2] Gossip Algorithms for Computing U-statistics. K. Pelckmans,
J. A.K. Suykens. IFAC Workshop on Estimation and Control of Networked Systems,
2009.

[3] Extending Gossip Algorithms to Distributed Estimations of
U-statistics. I. Colin, A. Bellet, J. Salmon, S. Clemencon. Advances in Neural
Information Processing Systems, 2015.

"""

import numpy as np

class GossipSolver(object):
    """Interface for gossip solvers.

    This class serves as an interface for inherited gossip solvers. It should
    not be used directly.

    Args:
        x (numpy.array): Observations sample. If n is the sample size
            and d is the observations space dimension, then x.shape should be
            (n, d).
        h (Optional[Callable[[numpy.array], float]]): Transformation applied to
            the initial sample, usually real-valued. Default value is identity.
        edges_seq (Optional[Sequence[Tuple[int]]]): Sequence of edges picked at
            each iteration.
        n_iter_max (Optional[int]): Number of iterations. Default value is 1. If
            edges_seq is not None, will be overwritten by edges_seq length.
        is_asynchronous (Optional[bool]): Indicates whether or not the solver is
            performing in an synchronous (False) or asynchronous (True)
            setting. Default value is False.
        saving_step (Optional[int]): Interval at which the current estimates are
            stored in the historic. Default value is 1.
        target (Optional[float]): Target value of the gossip algorithm. Used for
            computing error.

    Attributes:
        init_values (numpy.array[float]): Values initially stored at each node.
        current_estimates (numpy.array[float]): Each node estimates of the target
            value at the current iteration.
        n_agents (int): Number of nodes in the network. Multisample solvers are
            not implemented yet so this also corresponds to the sample size.
        edges_seq (Sequence[Tuple[int]]): Sequence of edges picked at each
            iteration.
        n_iter_max (int): Number of iterations.
        saving_step (int): Interval at which current_estimates are stored in
            historic (see previous_estimates).
        previous_estimates (numpy.array[float]): Historic of the nodes estimates.
        is_asynchronous (bool): Indicates whether or not the solver is performing
            in an synchronous (False) or asynchronous (True) setting.
        current_iteration ([numpy.array[int]]): Number of times each node has
            been picked. If solver performs in synchronous setting, should be
            equal to global_iteration at every index.
        global_iteration (int): Number of edges picked so far.
        target (float): Target value of the gossip algorithm. Used for computing
            error.

    """

    def __init__ (self,
                  x,
                  h=lambda y: y,
                  edges_seq=None,
                  n_iter_max=1,
                  is_asynchronous=False,
                  saving_step=1,
                  target=None):

        self.init_values = h(x)
        self.current_estimates = self.init_values.copy()
        self.n_agents = x.shape[0]

        self.edges_seq = edges_seq
        if self.edges_seq is None:
            self.n_iter_max = n_iter_max
        else:
            self.n_iter_max = self.edges_seq.shape[0]

        self.saving_step = saving_step
        self.previous_estimates = np.zeros((self.n_iter_max // saving_step, )
                                           + (self.init_values.shape[0], ))

        self.is_asynchronous = is_asynchronous
        self.current_iteration = 0
        self.global_iteration = self.current_iteration
        if is_asynchronous:
            #In asynchronous setting, track the number of times each node has
            #been picked
            self.current_iteration *= np.ones(self.n_agents, dtype='float64')
            
        self.target = target
            
    def update_from_sequence(self):
        """Perform the gossip algorithm using the current edges sequence."""

        for e in self.edges_seq:
            self.update_from_edge(*e)

    def update_from_edge(self, edge):
        """Perform one step of the gossip algorithm using the provided edge.

        Args:
            i (int): First edge picked at current iteration.
            j (int): Second edge picked at current iteration.
        
        """
        
        pass

    def reset(self, new_seq=None):
        """Reset the algorithm and set a new edges sequence.

        Args:
            new_seq (Sequence[Tuple[int]])): New sequence of edges to be picked.

        Remarks: 
            The sequence length new_seq.shape[0] will overwrite self.n_iter_max.
            Other parameters will be preserved.

        """

        self.__init__(self.init_values,
                      h=lambda y: y,
                      edges_seq=new_seq,
                      n_iter_max=self.n_iter_max,
                      is_asynchronous=self.is_asynchronous,
                      saving_step=self.saving_step,
                      target=self.target)

    def save_if_necessary(self):
        """Save if current iteration is a multiple of the saving step.

        Add a copy of self.current_estimates to self.previous_estimates if
        self.global_iteration is a multiple of self.saving_step.

        """
        
        if (self.global_iteration - 1) % self.saving_step == 0:
            hist_index = (self.global_iteration - 1) // self.saving_step
            self.previous_estimates[hist_index] = self.current_estimates.copy()

    def get_gap_historic(self):
        """Return the relative distance between estimates and real values.

        Returns: 
            (Tuple[float]): Average and standard deviation of relative
            error at each saved iteration.

        """

        gap = np.linalg.norm(1 - self.previous_estimates / self.target, axis=1) \
              / np.sqrt(self.n_agents)
        std = np.std(np.abs(1 - self.previous_estimates / self.target), axis=1)
        return (gap, std)

        
class AveragingGossipSolver (GossipSolver):
    """Averaging gossip algorithm, as detailed in [1].

    Args:
        x (numpy.array): Observations sample. If n is the sample size and d is
            the observations space dimension, then x.shape should be (n, d).
        h (Optional[Callable[[numpy.array], float]]): Transformation
            applied to the initial sample, usually real-valued. Default value is
            identity. Argument type is identical to x type.
        edges_seq (Optional[Sequence[Tuple[int]]]): Sequence of edges picked at
            each iteration.
        n_iter_max (Optional[int]): Number of iterations. Default value is 1. If
            edges_seq is not None, will be overwritten by edges_seq.shape[0].
        is_asynchronous (Optional[bool]): Indicates whether or not the solver is
            performing in an synchronous (False) or asynchronous (True)
            setting. Default value is True.
        saving_step (Optional[int]): Interval at which the current estimates are
            stored in the historic. Default value is 1.
        target (Optional[float]): Target value of the gossip algorithm. Used for
            computing error.

    Remarks:
        This algorithm is by design a fully asynchronous gossip algorithm so
        is_asynchronous is set by default to True.

    """

    def __init__ (self,
                  x,
                  h=lambda y:y,
                  edges_seq=None,
                  n_iter_max=1,
                  is_asynchronous=True,
                  saving_step=1,
                  target=None):

        super(AveragingGossipSolver, self).__init__(x,
                                                    h=h,
                                                    edges_seq=edges_seq,
                                                    n_iter_max=n_iter_max,
                                                    is_asynchronous=is_asynchronous,
                                                    saving_step=saving_step,
                                                    target=target)

        if self.target is None:
            self.target = np.average(self.init_values, axis=0)
            
    def update_from_edge (self, i, j):
        """Perform one step of the gossip algorithm using the provided edge.
        
        Current estimates of nodes i and j are averaged.

        Args:
            i (int): First edge picked at the current iteration.
            j (int): Second edge picked at the current iteration.
        
        """

        self.current_estimates[[i, j]] = np.average(self.current_estimates[[i, j]])
        self.current_iteration[[i, j]] += 1

        self.global_iteration += 1
        self.save_if_necessary()

        
class U1GossipSolver (GossipSolver):
    """Gossip algorithm for computing partial sums, as detailed in [2].

    Args: 
        x (numpy.array): Observations sample. If n is the sample size and d is
            the observations space dimension, then x.shape should be (n, d).
        h (Callable[[numpy.array, numpy.array], numpy.array[float]]): Pairwise
            function associated to the degree-2 U-statistics. Arguments types
            are identical to x type. Returned numpy.array is a one dimensional
            array with size equals to arguments size over the first coordinate.
        edges_seq (Optional[Sequence[Tuple[int]]]): Sequence of edges picked at
            each iteration.
        n_iter_max (Optional[int]): Number of iterations. Default value is 1. If
            edges_seq is not None, will be overwritten by edges_seq.shape[0].
        is_asynchronous (Optional[bool]): Indicates whether or not the solver is
            performing in an synchronous (False) or asynchronous (True)
            setting. Default value is False.
        saving_step (Optional[int]): Interval at which the current estimates are
            stored in the historic. Default value is 1.
        target (Optional[float]): Target values of the gossip algorithm. Used
            for computing error. If not assigned, will be set to the partial sums
            values.
        h_gram (Optional[numpy.array[float]]): Gram matrix associated to the
            function h and the sample x: h_gram[i, j] = h(x[i], x[j]). If not
            assigned, will be computed from h and x.

    Attributes:
        ind_x (numpy.array[int]): Indexes of primary observation stored at each
            node.
        ind_y (numpy.array[int]): Indexes of auxiliary observation stored at each
            node.
        h_gram (numpy.array[float]): Gram matrix associated to the function h and
            the sample: h_gram[i, j] = h(init_values[i], init_values[j]). If not
            assigned, will be computed from h and x, when h is provided.

    Remarks:
        This algorithm is by design a synchronous gossip algorithm so
        is_asynchronous is set by default to False.

    """

    def __init__ (self,
                  x,
                  h,
                  edges_seq=None,
                  n_iter_max=1,
                  is_asynchronous=False,
                  saving_step=1,
                  target=None,
                  h_gram=None):

        super(U1GossipSolver, self).__init__(x,
                                             h=lambda y: y,
                                             edges_seq=edges_seq,
                                             n_iter_max=n_iter_max,
                                             is_asynchronous=is_asynchronous,
                                             saving_step=saving_step)

        self.current_estimates = np.zeros(self.n_agents)
        self.ind_x = np.arange(self.n_agents)
        self.ind_y = np.arange(self.n_agents)

        self.h_gram = h_gram
        if self.h_gram is None:
            self.h_gram = np.zeros((self.n_agents, self.n_agents))
            for i in range(self.n_agents):
                self.h_gram[i] = h(self.init_values[i],
                                   self.init_values[[self.ind_x]])
                
        self.target = target
        if self.target is None:
            self.target = np.average(self.h_gram, axis=1)

    def reset (self, new_seq=None):
        """Reset the algorithm and set a new edges sequence.

        Args:
            new_seq (Sequence[Tuple[int]]): New sequence of edges to be picked.

        Remarks: 
            The sequence length will overwrite self.n_iter_max. Other parameters
            will be preserved.

        """

        self.__init__(self.init_values,
                      h=lambda y: y,
                      edges_seq=new_seq,
                      n_iter_max=self.n_iter_max,
                      is_asynchronous=self.is_asynchronous,
                      saving_step=self.saving_step,
                      target=self.target,
                      h_gram=self.h_gram)
    
    def update_from_edge (self, i, j):
        """Perform one step of the gossip algorithm using the provided edge.
        
        Nodes i and j swap their auxialiary observations. Then, every node
        update its estimate.

        Args:
            i (int): First node picked.
            j (int): Second node picked.

        Remarks:
            i and j must be connected in the support network.

        """

        self.ind_y[[i, j]] = self.ind_y[[j, i]]

        if self.is_asynchronous:
            self.current_estimates[[i, j]] *= self.current_iteration[[i, j]]
            self.current_estimates[[i, j]] += self.h_gram[[self.ind_x[[i, j]], self.ind_y[[i, j]]]]
            self.current_estimates[[i, j]] /= self.current_iteration[[i, j]] + 1
            self.current_iteration[[i, j]] += 1
        else:
            self.current_estimates *= self.current_iteration
            self.current_estimates += self.h_gram[[self.ind_x, self.ind_y]]
            self.current_estimates /= self.current_iteration + 1
            self.current_iteration += 1

        self.global_iteration += 1
        self.save_if_necessary()
        

class U2GossipSolver (U1GossipSolver):
    """Gossip algorithm for computing $U$-statistics, as detailed in [2].

    Args:
        x (numpy.array): Observations sample. If n is the sample size and d is
            the observations space dimension, then x.shape should be (n, d).
        h (Callable[[numpy.array, numpy.array], numpy.array[float]]): Pairwise
            function associated to the degree-2 U-statistics. Arguments types
            are identical to x type. Returned numpy.array is a one dimensional
            array with size equals to arguments size over the first coordinate.
        edges_seq (Optional[Sequence[Tuple[int]]]): Sequence of edges picked at
            each iteration.
        edges_seq2 (Optional[Sequence[Tuple[int]]]): Second sequence of edges
            picked at each iteration.
        n_iter_max (Optional[int]): Number of iterations. Default value is 1. If
            edges_seq is not None, will be overwritten by edges_seq.shape[0].
        n_iter_max (Optional[int]): Number of iterations. Default value is 1. If
            edges_seq is not None, will be overwritten by edges_seq.shape[0].
        is_asynchronous (Optional[bool]): Indicates whether or not the solver is
            performing in an synchronous (False) or asynchronous (True)
            setting. Default value is False.
        saving_step (Optional[int]): Interval at which the current estimates are
            stored in the historic. Default value is 1.
        target (Optional[float]): Target values of the gossip algorithm. Used
            for computing error. If not assigned, will be set to the partial sums
            values.
        h_gram (Optional[numpy.array[float]]): Gram matrix associated to the
            function h and the sample x: h_gram[i, j] = h(x[i], x[j]). If not
            assigned, will be computed from h and x.

    Attributes:
        edges_seq2 (Sequence[Tuple[int]]): Second sequence of edges picked at
            each iteration.

    Remarks:
        This algorithm is by design a synchronous gossip algorithm so
        self.is_asynchronous is set by default to False.

    """

    def __init__ (self,
                  x,
                  h,
                  edges_seq=None,
                  edges_seq2=None,
                  n_iter_max=1,
                  is_asynchronous=False,
                  saving_step=1,
                  target=None,
                  h_gram=None):

        super(U2GossipSolver, self).__init__(x,
                                             h=h,
                                             edges_seq=edges_seq,
                                             n_iter_max=n_iter_max,
                                             is_asynchronous=is_asynchronous,
                                             saving_step=saving_step,
                                             target=target,
                                             h_gram=h_gram)

        self.edges_seq2 = edges_seq2
        self.target = np.average(self.target)

    def reset (self, new_seq=None, new_seq2=None):
        """Reset the algorithm and set a new edges sequence.

        Args:
            new_seq (Sequence[Tuple[int]]): New sequence of edges to be picked
                for primary observations swapping.
            new_seq2 (Sequence[Tuple[int]]): New sequence of edges to be picked
                for auxiliary observations swapping.

        Remarks:
            The sequence length will overwrite self.n_iter_max. Other parameters
            will be preserved.

        """

        self.__init__(self.init_values,
                      h=lambda y: y,
                      edges_seq=self.edges_seq,
                      edges_seq2=self.edges_seq2,
                      n_iter_max=self.n_iter_max,
                      is_asynchronous=self.is_asynchronous,
                      saving_step=self.saving_step,
                      target=self.target,
                      h_gram=self.h_gram)

        
    def update_from_edge (self, i, j, k, l):
        """Perform one step of the gossip algorithm using the provided edge.
        
        Nodes i and j swap their primary observations. Then, nodes k and l swap
        their auxiliary observations. Finally, every node update its estimate.

        Args:
            i (int): First node of the first edge picked.
            j (int): Second node of the first edge picked.
            k (int): First node of the second edge picked.
            l (int): Second node of the second edge picked.
        
        Remarks:
            (i, j) and (k, l) must be edges of the support network.

        """

        self.ind_x[[i, j]] = self.ind_x[[j, i]]
        self.ind_y[[k, l]] = self.ind_y[[l, k]]

        if self.is_asynchronous:
            self.current_estimates[[i, j]] *= self.current_iteration[[i, j]]
            self.current_estimates[[i, j]] += self.h_gram[[self.ind_x[[i, j]], self.ind_y[[i, j]]]]
            self.current_estimates[[i, j]] /= self.current_iteration[[i, j]] + 1
            self.current_iteration[[i, j]] += 1

            self.current_estimates[[k, l]] *= self.current_iteration[[k, l]]
            self.current_estimates[[k, l]] += self.h_gram[[self.ind_x[[k, l]], self.ind_y[[k, l]]]]
            self.current_estimates[[k, l]] /= self.current_iteration[[k, l]] + 1
            self.current_iteration[[k, l]] += 1
        else:
            self.current_estimates *= self.current_iteration
            self.current_estimates += self.h_gram[[self.ind_x, self.ind_y]]
            self.current_estimates /= self.current_iteration + 1
            self.current_iteration += 1

        self.global_iteration += 1
        self.save_if_necessary()

    def update_from_sequence(self):
        """Perform the gossip algorithm using the current edges sequence.
        
        The sequence self.edges_seq is used for swapping primary
        observations. The sequence self.edges_seq2 is used for swapping
        secondary observations.

        """

        for i in range(len(self.edges_seq)):
            self.update_from_edge(self.edges_seq[i][0], self.edges_seq[i][1],
                                  self.edges_seq2[i][0], self.edges_seq2[i][1])

            
class GoStaSolver (U1GossipSolver):
    """GoSta algorithm for computing $U$-statistics, as detailed in [3].

    Args: 
        x (numpy.array): Observations sample. If n is the sample size and d is
            the observations space dimension, then x.shape should be (n, d).
        h (Callable[[numpy.array, numpy.array], numpy.array[float]]): Pairwise
            function associated to the degree-2 U-statistics. Arguments types
            are identical to x type. Returned numpy.array is a one dimensional
            array with size equals to arguments size over the first coordinate.
        edges_seq (Optional[Sequence[Tuple[int]]]): Sequence of edges picked at
            each iteration.
        n_iter_max (Optional[int]): Number of iterations. Default value is 1. If
            edges_seq is not None, will be overwritten by edges_seq.shape[0].
        is_asynchronous (Optional[bool]): Indicates whether or not the solver is
            performing in an synchronous (False) or asynchronous (True)
            setting. Default value is False.
        saving_step (Optional[int]): Interval at which the current estimates are
            stored in the historic. Default value is 1.
        target (Optional[float]): Target values of the gossip algorithm. Used
            for computing error. If not assigned, will be set to the partial sums
            values.
        h_gram (Optional[numpy.array[float]]): Gram matrix associated to the
            function h and the sample x: h_gram[i, j] = h(x[i], x[j]). If not
            assigned, will be computed from h and x.
        asynchronous_weights (Optional[numpy.array[float]]): Weights used in the
            asynchronous setting. Useless in synchronous setting.

    Attributes:
        w (numpy.array[float]): Weights used in the asynchronous setting. Not
            used if is_asynchronous = False.

    Remarks:
        This algorithm can perform either in a synchronous or an asynchronous
        setting.

    """

    def __init__ (self,
                  x,
                  h,
                  edges_seq=None,
                  n_iter_max=1,
                  is_asynchronous=False,
                  saving_step=1,
                  target=None,
                  h_gram=None,
                  asynchronous_weights=None):

        super(GoStaSolver, self).__init__(x,
                                          h=h,
                                          edges_seq=edges_seq,
                                          n_iter_max=n_iter_max,
                                          is_asynchronous=is_asynchronous,
                                          saving_step=saving_step,
                                          target=target,
                                          h_gram=h_gram)

        self.target = target
        if self.target is None:
            self.target = np.average(self.h_gram)

        self.w = asynchronous_weights

    def reset (self, new_seq=None):
        """Reset the algorithm and set a new edges sequence.

        Args:
            new_seq (Sequence[Tuple]): New sequence of edges to be picked.

        Remarks:
            The sequence length will overwrite self.n_iter_max. Other parameters
            will be preserved.

        """

        self.__init__(self.init_values,
                      h=lambda y: y,
                      edges_seq=new_seq,
                      n_iter_max=self.n_iter_max,
                      is_asynchronous=self.is_asynchronous,
                      saving_step=self.saving_step,
                      target=self.target,
                      h_gram=self.h_gram,
                      asynchronous_weights=self.w)
            
    def update_from_edge (self, i, j):
        """Perform one step of the GoSta algorithm using the provided edge.
        
        Nodes i and j swap their auxialiary observations and average their
        estimates. Then, every node update its estimate in the synchronous
        setting, otherwise only i and j perform the update with specific
        weights.

        Args:
            i (int): First node picked.
            j (int): Second node picked.

        Remarks:
            i and j must be connected in the support network.

        """

        self.ind_y[[i, j]] = self.ind_y[[j, i]]
        self.current_estimates[[i, j]] = .5 * np.sum(self.current_estimates[[i, j]])

        if self.is_asynchronous:
            self.current_estimates[[i, j]] *= self.current_iteration[[i, j]]
            self.current_estimates[[i, j]] += self.w[[i,j]] \
                                              * self.h_gram[[self.ind_x[[i,j]],
                                                             self.ind_y[[i,j]]]]
            self.current_iteration[[i, j]] += self.w[[i, j]]
            self.current_estimates[[i, j]] /= self.current_iteration[[i, j]]
        else:
            self.current_estimates *= self.current_iteration
            self.current_estimates += self.h_gram[[self.ind_x, self.ind_y]]
            self.current_iteration += 1
            self.current_estimates /= self.current_iteration

        self.global_iteration += 1
        self.save_if_necessary()
