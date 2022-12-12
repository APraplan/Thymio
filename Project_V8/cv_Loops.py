class Loops:
    " a loop is a series of intersections that are connected through lines in a clockwise manner"
    " it consists of edges and nodes refering to corners"
    " because of the lines intersecting, a loop might only be a part of an obstacle,"
    " or not even part of any obstacle at all"
    " loops function as a data structure "
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.corners = set()
        for node in self.nodes:
            self.corners.add(node.intersection.coordinate)
        for edge in self.edges:
            for point in edge.points:
                assert point in self.corners
