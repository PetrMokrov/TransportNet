# Attention: as shown on the table above
# nodes indexed from 0 to ...
# edges indexed from 0 to ...
import graph_tool.all as gt
import graph_tool.topology as gtt
import numpy as np
import math
from t_swsf import t_swsf, shorest_dists_equal
from copy import deepcopy, copy
from collections import defaultdict

class TransportGraph:
    def __init__(self, graph_table, nodes_number, links_number, maxpath_const = 3, recompute_method='t_swsf'):
        assert recompute_method in ['dijkstra', 't_swsf', 'compare']
        self.recompute_method = recompute_method
        self.nodes_number = nodes_number
        self.links_number = links_number
        self.max_path_length = maxpath_const * int(math.sqrt(self.links_number))
        
        self.graph = gt.Graph(directed=True)
        #nodes indexed from 0 to V-1
        vlist = self.graph.add_vertex(self.nodes_number)
        # let's create some property maps
        ep_freeflow_time = self.graph.new_edge_property("double")
        ep_capacity = self.graph.new_edge_property("double")
        
        #define data for edge properties
        self.capacities = np.array(graph_table[['capacity']], dtype = 'float64').flatten()
        self.freeflow_times = np.array(graph_table[['free_flow_time']], dtype = 'float64').flatten()  

        #adding edges to the graph
        self.inits = np.array(graph_table[['init_node']], dtype = 'int64').flatten()
        self.terms = np.array(graph_table[['term_node']], dtype = 'int64').flatten()
        for index in range(self.links_number):
            init = self.inits[index]
            term = self.terms[index]
            edge = self.graph.add_edge(self.graph.vertex(init),
                                       self.graph.vertex(term))
            ep_freeflow_time[edge] = self.freeflow_times[index]
            ep_capacity[edge] = self.capacities[index]
            
        #save properties to graph
        self.graph.edge_properties["freeflow_times"] = ep_freeflow_time
        self.graph.edge_properties["capacities"] = ep_capacity
        if self.recompute_method in ['t_swsf', 'compare']:
            # lazy parameters update
            #TODO: share single edge weights property between all source nodes!
            self._prev_info = {} # (shortest distances, shortest paths, curr_weights)

    @property
    def edges(self):
        return self.graph.get_edges([self.graph.edge_index])

    def successors(self, node):
        return self.graph.get_out_neighbors(node)

    def predecessors(self, node):
        return self.graph.get_in_neighbors(node)
        
    #source, target and index of an edge
    def in_edges(self, node):
        return self.graph.get_in_edges(node, [self.graph.edge_index])
    
    #source, target and index of an edge
    def out_edges(self, node):
        return self.graph.get_out_edges(node, [self.graph.edge_index])

    def shortest_distances(self, source, targets, times):

        if targets is None:
            targets = np.arange(self.nodes_number)

        def finalize_return(distances, pred_map):
            assert isinstance(targets, (list, np.ndarray))
            return np.asarray(distances.get_array()[targets]), pred_map.a

        def recompute_dijkstra():
            ep_time_map = self.graph.new_edge_property("double", vals = times)
            distances, pred_map = gtt.shortest_distance(
                g = self.graph, source = source,
                weights = ep_time_map, pred_map = True)
            return ep_time_map, distances, pred_map

        def recompute_t_swsf():
            np_times = np.asarray(times)
            _prev_dists, _prev_pred_map, _prev_lens = self._prev_info[source]
            np_prev_times = np.asarray(_prev_lens.get_array())
            #TODO: this part can be optimized
            times_diff = np_times - np_prev_times
            # print(max(times_diff))
            times_diff_prop = self.graph.new_edge_property('double', vals=times_diff)
            _source = self.graph.vertex(source) if isinstance(source, int) else source
            new_times, new_dists, new_pred_map = t_swsf(
                self.graph, _prev_lens, _source, 
                _prev_dists, _prev_pred_map, times_diff_prop)
            # print(np.max(np.abs(np.asarray(new_times.get_array()) - np_times)))
            assert np.max(np.abs(np.asarray(new_times.get_array()) - np_times)) < 1e-10
            return new_times, new_dists, new_pred_map

        if self.recompute_method == 'dijkstra':
            _, distances, pred_map = recompute_dijkstra()
            return finalize_return(distances, pred_map)

        if self.recompute_method == 't_swsf':
            if not source in self._prev_info:
                times_prop, distances, pred_map = recompute_dijkstra()
            else:
                times_prop, distances, pred_map = recompute_t_swsf()
            self._prev_info[source] = (copy(distances), copy(pred_map), times_prop)
            return finalize_return(distances, pred_map)

        if self.recompute_method == 'compare':
            _, gt_distances, gt_pred_map = recompute_dijkstra()
            if not source in self._prev_info:
                times_prop, distances, pred_map = recompute_dijkstra()
            else:
                times_prop, distances, pred_map = recompute_t_swsf()
            self._prev_info[source] = (copy(distances), copy(pred_map), times_prop)
            assert shorest_dists_equal(self.graph, distances, gt_distances)
            return finalize_return(distances, pred_map)


#    def nodes_number(self):
#        return self.nodes_number
    
#    def links_number(self):
#        return self.links_number
    
#    def capacities(self):
#        return self.capacities
    
#    def freeflow_times(self):
#        return self.freeflow_times