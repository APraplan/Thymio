class nodeSet:
    """
    in nodeSet the nodes are stored,
    when a new node is introduced, it is connected to its neighbours if they are already in the set of node
    the current implementation doesnt scale very well but works fine for simple obstacles
    """
    def __init__(self):
        self.nodes = set()
    def addNode(self, node):
        for oldnode in self.nodes:
            if oldnode.intersection_of_neighbour1 == node.intersection:
                oldnode.addneighbour1(node)

            if oldnode.intersection_of_neighbour2 == node.intersection:
                oldnode.addneighbour2(node)

            if oldnode.intersection_of_neighbour3 == node.intersection:
                oldnode.addneighbour3(node)

            if oldnode.intersection_of_neighbour4 == node.intersection:
                oldnode.addneighbour4(node)

            if node.intersection_of_neighbour1 == oldnode.intersection:
                node.addneighbour1(oldnode)

            if node.intersection_of_neighbour2 == oldnode.intersection:
                node.addneighbour2(oldnode)

            if node.intersection_of_neighbour3 == oldnode.intersection:
                node.addneighbour3(oldnode)

            if node.intersection_of_neighbour4 == oldnode.intersection:
                node.addneighbour4(oldnode)
        self.nodes.add(node)

