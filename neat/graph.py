from tarjan import tarjan


class Vertex:

    def __init__(self, i):
        self.i = i

    def __hash__(self):  # will allow vertex to be a map/set key
        return hash(id(self))

    def __str__(self):
        return str(self.i)


class Edge:

    def __init__(self, u, v):
        self.origin = u
        self.destination = v

    def endpoints(self):
        return self.origin, self.destination

    def opposite(self, v):
        if not isinstance(v, Vertex):
            raise TypeError('v must be a Vertex')
        return self.destination if v is self.origin else self.origin

    def __hash__(self):  # will allow edge to be a map/set key
        return hash((self.origin, self.destination))

    def __str__(self):
        return '({0},{1})'.format(str(self.origin.element()), str(self.destination.element()))


class Graph:
    def __init__(self):
        self.outgoing = {}
        self.incoming = {}

    @classmethod
    def from_genome(cls, genome) -> 'Graph':
        ret = cls()
        for neuron in genome.neurons:
            ret.insert_vertex(neuron.neuron_id)
        for link in genome.links:
            ret.insert_edge(ret.find_vertex(link.from_neuron_id), ret.find_vertex(link.to_neuron_id))

        return ret

    def find_vertex(self, i):
        for v in self.outgoing.keys():
            if v.i == i:
                return v
        return None

    def _validate_vertex(self, v):
        if not isinstance(v, Vertex):
            raise TypeError('Vertex expected')
        if self.outgoing.get(v) is None:
            raise ValueError('Vertex does not belong to this graph.')

    def vertex_count(self):
        return len(self.outgoing)

    def vertices(self):
        return self.outgoing.keys()

    def edge_count(self):
        total = sum(len(self.outgoing[v]) for v in self.outgoing)
        return total

    def edges(self):
        result = set()  # avoid double-reporting edges of undirected graph
        for secondary_map in self.outgoing.values():
            result.update(secondary_map.values())  # add edges to resulting set
        return result

    def get_edge(self, u, v):
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self.outgoing[u].get(v)  # returns None if v not adjacent

    def get_vertex(self, element):
        outdict = {}
        indict = {}
        for outkey, inkey in zip(self.outgoing.keys(), self.incoming.keys()):
            outdict[outkey.element()] = outkey
            indict[inkey.element()] = inkey

        if element in outdict.keys():
            return outdict[element]
        elif element in indict.keys():
            return indict[element]
        else:
            return None

    def degree(self, v, outgoing=True):
        self._validate_vertex(v)
        adj = self.outgoing if outgoing else self.incoming
        return len(adj[v])

    def incindent_edges(self, v, outgoing=True):
        adj = self.outgoing if outgoing else self.incoming
        for edge in adj[v].keys():
            yield edge

    def incindent_edge_count(self, v, outgoing=True):
        adj = self.outgoing if outgoing else self.incoming
        return len(adj[v].values())

    def contains_vertex(self, v):
        try:
            self._validate_vertex(v)
        except ValueError:
            return False
        return True

    def insert_vertex(self, id):
        v = Vertex(id)
        if self.outgoing.get(v) is not None:
            return v
        else:
            self.outgoing[v] = {}
            self.incoming[v] = {}
            return v

    def insert_edge(self, u, v):
        self._validate_vertex(u)
        self._validate_vertex(v)
        e = Edge(u, v)
        self.outgoing[u][v] = e
        self.incoming[v][u] = e

    def clear(self):
        self.incoming = {}
        self.outgoing = {}


    def is_cyclic_graph(self):
        graph_map = {}
        for v in self.vertices():
            graph_map[v.i] = [out.i for out in self.outgoing[v].keys()]
        scc = tarjan(graph_map)
        return len(scc) < len(self.vertices())
