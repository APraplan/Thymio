import cv_Loops as Loops
class LoopSet:
    " Loopset is a set with loops upon which I added some functions to create the loops"
    def __init__(self):
        self.loopSet = set()
################################################################################################################
    # creating loops
################################################################################################################
    def addClockwiseLoop(self, node):
        """
        from a node (intersection), try to make a clockwise loop
        in: node Object
        out: /
        append new loops to self.loops
        """
        # we try every possible previous node (determines starting direction)
        neighbours = [node.neighbour1, node.neighbour2, node.neighbour3, node.neighbour4]
        for prev in neighbours:
            if prev != None:
                current = node
                loopNodes = set()
                loopEdges = set()
                original = node
                while current not in loopNodes and current != None:
                    loopNodes.add(current)
                    next, edge = current.clockwiseNext(prev)
                    if current != None and edge != None:
                        loopEdges.add(edge)
                    prev = current
                    current = next

                if current != None and current in loopNodes: #so its a closed loop
                    if original != current:
                        print(loopNodes)
                        assert original == current #double check
                    #print('loop finished')
                    new_loop = Loops.Loops(loopNodes, loopEdges)
                    self.loopSet.add(new_loop)

                else:
                    pass


    ################################################################################################################
    # by trying every possible start direction from every possible start node we create duplicates -> remove these
    # they are a necessary evil as there is no single starting direction that would find all types of objects
    # this slows everything down :/
    ################################################################################################################
    def remove_duplicates(self):
        """
        from self.loopSet, remove all duplicate loops
        """
        # a loop is completely defined by its edges -> no: edge defined in two directions -> duplicates exist
        # a loop is completely defined by its corners -> yes: we are talking about minimal size and simple shape loops so far
        # a complex obstacle is NOT defined by its corners!
        version = 3
        if version == 3:
            def compare(loop1, loop2):
                same = True
                edges = loop1.edges
                for edge in loop2.edges:
                    match = False
                    for original_edge in edges:
                        if edge.points == original_edge.points:
                            match = True
                    if match == False:
                        same = False
                return same
            new_loops = set()
            for potential_new_loop in self.loopSet:
                new = True
                for already_approved_loop in new_loops:
                    same = compare(already_approved_loop, potential_new_loop)
                    if same == True:
                        new = False
                if new:
                    new_loops.add(potential_new_loop)
            self.loopSet = new_loops




