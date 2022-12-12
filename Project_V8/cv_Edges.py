class Edge:
    " an edge is defined by a set of two corners, corners are points, an edge is used to make an obstacle"
    def __init__(self, corner1, corner2):
        self.points = set()
        self.points.add(corner1)
        self.points.add(corner2)

    def view(self):
        """
        export the corners of an edge in a different, str, format
        example: "(0,0) & (10, 34)"
        """
        i = 0
        coords = [0,0]
        for point in self.points:
            coords[i] = point.getCoordinates()
            i+=1
        return str(coords[0])+'&'+str(coords[1])