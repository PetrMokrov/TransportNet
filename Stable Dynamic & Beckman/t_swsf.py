import graph_tool
import graph_tool.all as gt
import graph_tool.topology as gtt
import numpy as np
import math
from copy import deepcopy, copy
import sortedcontainers
from sortedcontainers import SortedDict, SortedList
from tqdm import tqdm

class GraphPriorityQueue:

    def __init__(self):
        # priority queue initialization
        # dict for <vertex_number, vertex_priority> pairs
        self.vertices_dict = {}
        # sorted_list for <vertex_priority, vertex_number> pairs
        self.priority_dict = SortedList()

    def insert_or_update(self, vertex, priority):
        i_vertex = int(vertex)
        try:
            old_priority = self.vertices_dict[i_vertex]
            self.priority_dict.remove((old_priority, i_vertex))
        except KeyError as e:
            pass
        finally:
            self.vertices_dict[i_vertex] = priority
            self.priority_dict.add((priority, i_vertex))
            assert len(self.vertices_dict) == len(self.priority_dict)

    def pop_min(self):
        priority, i_vertex = self.priority_dict.pop(0)
        del self.vertices_dict[i_vertex]
        assert len(self.vertices_dict) == len(self.priority_dict)
        return i_vertex, priority

    def __len__(self):
        return len(self.vertices_dict)

    def is_empty(self):
        return len(self) == 0

def t_swsf(g, lens, s, dists, pred_map, delta_lens, infty=2147483647.0):
    '''
    Recomputes the shortest pathes from s to all other nodes
    For reference, see https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.5911&rep=rep1&type=pdf
    :Parameters:
    g : graph_tool.Graph : the graph
    lens : graph_tool.EdgePropertyMap : outdated edge's weights property
    s : graph_tool.Vertex : the source vertex
    dists : graph_tool.VertexPropertyMap : the minimal distances w.r.t. outdated weights
    pred_map : graph_tool.VertexPropertyMap : predecessor's tree of minimal pathes w.r.t. outdated weights
    delta_lens : graph_tool.EdgePropertyMap : edge's weights update
    :Returns:
    lens: graph_tool.EdgePropertyMap : updated edge's weights property
    Dists: graph_tool.VertexPropertyMap : the minimal distances w.r.t. updated weights
    pred_map: graph_tool.VertexPropertyMap : predecessor's tree of minimal pathes w.r.t. updated weights
    '''
    def min_with_v_upd(old_val, new_val, old_v, new_v):
        if old_val <= new_val:
            return old_val, old_v, False
        return new_val, new_v, True

    # con function
    def con(v, D, lens):
        '''
        :Parameters:
        v: graph_tool.Vertex
        D: graph_tool.VertexPropertyMap : min distances
        len_prop : edge's lengths
        :Returns:
        con, con_predecessor
        '''
        if v == s:
            return 0.0, s
        con_fin = 2 * infty #TODO: it is slightly crazy!
        v_pred = None
        for edge in v.in_edges():
            con_upd = D[edge.source()] + lens[edge]
            con_fin, v_pred, _ = min_with_v_upd(
                con_fin, D[edge.source()] + lens[edge], 
                v_pred, edge.source())
        assert v_pred is not None
        return con_fin, v_pred
    # priority queue
    Q = GraphPriorityQueue()
    # replication of 'dists'
    Dists = copy(dists)
    # Initialization phase
    for edge in g.edges():
        if delta_lens[edge] == 0.:
            pass
        lens[edge] += delta_lens[edge]
        v_pred, v = edge.source(), edge.target()
        if delta_lens[edge] > 0.:
            if v_pred == pred_map[v]:
                # 'edge' lies is on the minimum path
                dists[v], pred_map[v] = con(v, Dists, lens)
        else:
            dists[v], pred_map[v], _ = min_with_v_upd(
                dists[v], Dists[v_pred] + lens[edge], 
                pred_map[v], v_pred)
        # update Q
        if dists[v] != Dists[v]:
            Q.insert_or_update(v, min(dists[v], Dists[v]))
    # Main phase
    while not Q.is_empty():
        i_v, priority = Q.pop_min()
        v = g.vertex(i_v)
        if dists[v] < Dists[v]:
            Dists[v] = dists[v]
            for edge in v.out_edges():
                v_next = edge.target()
                dists[v_next], pred_map[v_next], _flag = min_with_v_upd(
                    dists[v_next], Dists[v] + lens[edge], 
                    pred_map[v_next], v)
                if _flag: 
                    Q.insert_or_update(v_next, min(dists[v_next], Dists[v_next]))
        elif dists[v] > Dists[v]:
            D_v_old = Dists[v]
            Dists[v] = infty
            dists[v], pred_map[v] = con(v, Dists, lens)
            Q.insert_or_update(v, dists[v])
            for edge in v.out_edges():
                v_next = edge.target()
                if D_v_old + lens[edge] == dists[v_next]:
                    dists[v_next], pred_map[v_next] = con(v_next, Dists, lens)
                    Q.insert_or_update(v_next, min(dists[v_next], Dists[v_next]))
    return lens, Dists, pred_map

def shorest_dists_equal(g, dist_1, dist_2, tol=1e-10):
    res = True
    for v in g.vertices():
        if abs(dist_1[v] - dist_2[v]) > tol:
            res = False
    return res