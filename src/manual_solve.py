#!/usr/bin/python

import os, sys
import json
import numpy as np
import re


### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

def solve_9d9215db(x):
    """
    Stephen Jeffery, 21249306
    https://github.com/sbjeffery/ARC
    
    
    The initial grid consists of four pixels in one quarter of the grid. Two of these pixels
    are the same colour. The four pixels are reflected in the x and y axis with the origin at the
    centre of the grid. The two points of the same colour are joined by alternating pixels of the
    same colour to form an inner square(s).
    
    This program uses numpy operations to complete the affine transformations to give the target shape.
    As the points can be independently tranformed to get to the target position, the main unit of 
    shape here is the individual point. Points are translated from (r,c) to (x,y) positions with an origin
    at the centre of the grid. Functions for translate, rotate, select_pairs and join_pairs complete the tranformations.
    
    All training and test grids are solved correctly.
    
    These grids are solved with just pure numpy operations, with no special modules.
    I wanted to see if the functions would have common features with set of grids the use point
    transformations rather than groups of points in shapes. I wanted to write functions that
    took a point or array of points and returned a tranformed point or array of points so
    that transformations could be joined in different orders.
    
    Stephen Jeffery 21249306, Nov 2021
    
    """
    ## Main imports
    from itertools import combinations
    import copy
    
    # imports from other file to allow testing in jupyterlab.
    import json
    import numpy as np
    import re
    import matplotlib.pyplot as plt
        
    
    def rows_cols (A):
        """
        Get the size of the initial array A and return number of rows and coloums.
        
        >>> test_me = np.array([[1,2,0], [3,0,0], [0,0,4]]) # make a small array for testing.
        >>> rows_cols (test_me)
        (3, 3)
                
        """
        rows, cols = A.shape[0], A.shape[0]
        return rows, cols


    
    # Every array has a number of pixels often grouped into a shape and with colors.
    # Here we only have one shape in the training image to find. We will treat these
    # as individual points rather than shapes as all points get the same transformations.
    # we want the coordinates of all the non-zero pixels.

    def get_coords (A):
        """ 
        takes the array A with a grid of color values.
        get coordinates(x, y), with (0,0) in top left of arrary as (row, col)
        returns array of (r,c) points and array of (x,y, color) points where color is a integer from 0 to 9.
        
        get_coords (test_me)
        (array([[0, 0],
        [1, 0],
        [0, 1],
        [2, 2]]),
 array([[0., 2., 1.],
        [0., 1., 2.],
        [1., 2., 3.],
        [2., 0., 4.]]))
        
        """
        rc_coords = np.flip(np.column_stack(np.where(A > 0)), axis = 1) # the array of points (r,c)
        rows, cols = rows_cols (A) # get size of the main training area

        n_points = rc_coords.shape[0]  # number of non zero points
        points = np.zeros((n_points, 3)) # new array to hold the tranformed points in x,y,color format

        for p in range(n_points):
            points[p][0] = rc_coords[p][1]  # x = col
            points[p][1] = rows -1 - rc_coords[p][0] # y = nrows - 1 - row
            points[p][2] = A[rc_coords[p][1], rc_coords[p][0]] # the color value at (x,y) = (c,r)
            
            # print (f'<get_coords> initial points {p} as (r, c, color)')
            # print (f' {p}, ({rc_coords[p][0]}, {rc_coords[p][1]}, {A[rc_coords[p][1], rc_coords[p][0]]} )')

        return (rc_coords, points)
    
    
    
    # We want to transform these to a new coordinate axis with a new origin at the centre of the grid.
    # these are all 19 x 19 grids, so point (9,9)
    
    def translate (p, tx, ty):
        """
        Translate takes a single point in  x,y coordinates and translates them via the vector (tx, ty, 1) so that the (x,y, color) points move to
        (x + tx), (y + ty), negative numbrs for tx and ty are allowed. We can use this to move to a new coordinate system.
        transform is the tranformation matrix for translation, so that we complete the Affine transformation with matrix multiplication.
        
        >>> translate (np.array([2, 5, 8,]), 3, 4)
        array([5., 9., 8.])
        
        """
        pxy = copy.deepcopy (p[0:2]) # just the x,y points
        color = p[2] # just the color

        pxy.resize((3,)) # points always size (3,)
        pxy[2] = 1 # add on a dummy 1 on the end

        transform = np.eye(3)
        transform [0,2], transform [1,2] = (tx, ty)

        tp = np.matmul(transform, pxy)
        tp [2] = color # add the color back in

        return tp

    
    
    
    def reflect (p, axis = 1, inplace = False):
        """
        The reflect function takes a single point in  x,y coordinates and translates them via the transformation matrix [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        so that the (x,y) points move to a new point reflected about the y-axis when axis = 1, and the x-ais reflection when axis = -1
        We can use this to move to a new coordinate system. transform is the tranformation matrix for this translation, 
        so that we complete the Affine transformation with matrix multiplication.
        If Inplace = Trure, we return two points the point tp the reflected point and point p the original.

        >>> reflect (np.array([-8.,  8., 1.]), axis = 1)
        array([8., 8., 1.])

        >>> reflect (np.array([-8.,  8.,  1.]), axis = -1)
        array([-8., -8.,  1.])

        >>> reflect (np.array([-8.,  6.,  5]), axis = -1, inplace = True)
        array([[-8., -6.,  5.],
               [-8.,  6.,  5.]])

        """
        pxy = copy.deepcopy (p)
        pxy.resize((3,)) # points always size (3,)
        
        # make the transform function, new point = T*p
        transform = np.eye(3)
        transform [0,0] = -1

        if axis == 1:
            transform *= axis
        else:
            transform *= np.array([axis, axis, 1])
        # print (f'Transform = {transform}')

        tp = np.matmul(transform, pxy)

        if inplace:
            tp = np.vstack((tp, p))

        return tp

        
    
    def common_color (points, color = False):
        """
        Inputs are the point array, of transformed or raw points, of the form (x,y,color) or (r,c,color) and returns
        A, the most common color and B, the point of that color.

        >>> p = np.int_([[ 1, 17,  8], [ 1, 15,  1], [ 3, 17,  1], [ 3, 15,  2]])
        >>> n, a, b = common_color(p)
        >>> print (n,a,b)
        1 [1] [array([[ 1, 15,  1],
           [ 3, 17,  1]])]

        """
        # get any pairs of colors in the list
        colors = [color for (_,_,color) in points]
        color_pairs = [int(c1) for (c1,c2) in list(combinations(colors,2)) if (c1 == c2)]
        B = [points[points[:,2] == A] for A in color_pairs]
        n_common_color = len(color_pairs)

        return n_common_color, color_pairs, B

   
    
    
    def select_pairs (A):
        """
        Select the pairs of points to join with a line.
        points is the list of points to select from, is an array of (n x 3) shape with n points of (x, y, color)
        the function is how we select the points, min(x), max (x), min(y), max(y)
        potentially sort and select top two.
        return a pair of points. point_pairs

        """
        # sort the x values and take the min and max pairs from the sorted list.
        Ax = A[A[:,0].argsort()]
        # max x values
        min_x_pair, max_x_pair = Ax[:2], Ax[-2:]

        # sort the y values and take the min and max pairs from the sorted list.
        Ay = A[A[:,1].argsort()]
        min_y_pair, max_y_pair = Ay[:2], Ay[-2:]

        if not (min_x_pair[0][0] == min_x_pair[1][0]) and (max_x_pair[0][0] == max_x_pair[1][0]):
            raise ValueError ('The x values in the provided points do not have a shared max/min x value pair')

        if not (min_y_pair[0][1] == min_y_pair[1][1]) and (max_y_pair[0][1] == max_y_pair[1][1]):
            raise ValueError ('The y values in the provided points do not have a shared max/min y value pair')

        return max_x_pair, min_x_pair, max_y_pair, min_y_pair
    
    
    
    def flatten(t):
        """ Takes a list of lists, t and flatterns the list.
        """
        return [item for sublist in t for item in sublist]

    def join_pairs( p1, p2, step, color = -1):
        """ Color = -1 uses the color of the points returns an error if points are not the same color.
        otherwise color is a value beween 0[0,9]
        p (x,y,color), step = 1 colors every points, step = 2 every other point.
        mod (p1 - p2 / steps) == 0 as there must be a whole number of steps to get to the new point.
        """
        # print (p1,p2)
        max_x = max(p1[0], p2[0])
        min_x = min(p1[0], p2[0])
        max_y = max(p1[1], p2[1])
        min_y = min(p1[1], p2[1])

        if color == -1:
            color = p1[2]

        points = [] # initiate for new points [x, y, color]

        # print (max_x, min_x, max_y, min_y)

        if (max_x == min_x):  # then we want to tranverse this x value from ymin to ymax with color p1[3] and step

            if (max_y- min_y) % step == 0:
                N = int((max_y- min_y) / step)
            else:
                raise ValueError (f'You can\'t fill with that step {step} between between {min_y} and {max_y},please select a new step value.')

            ys = list(np.linspace(min_y, max_y,N+1))
            # print (f' y= {ys}')
            new_points = [[min_x, y, p1[2]] for y in ys]
            points.append(new_points)

        elif (max_y == min_y):  # then we want to tranverse this x value from ymin to ymax with color p1[3] and step

            if (max_x- min_x) % step == 0:
                N = int((max_x- min_x) / step)
            else:
                raise ValueError (f'You can\'t fill with that step {step} between between {min_x} and {max_x},please select a new step value.')

            xs = list(np.linspace(min_x, max_x,N+1))
            # print (f' x= {xs}')
            new_points = [[x, min_y, p1[2]] for x in xs]
            points.append(new_points)

        else:
            raise ValueError ('The points do not form a vert of horizontal line so can not be filled.')

        return points

      
    
    ########   Main section   ###########
    
    # Take all the points with none zero color and translate their coordinates. These are the points with reference to a 
    # new origin at (tx, ty) so that this is now (0, 0)
    
    rows_cols (x) # get rows and cols of the initial grid.
    rc_coords, points = get_coords (x) # get the coords of each point.
    
    trans_pts = np.zeros((points.shape[0],3)) # make the zeros matrix for transformed points.
    tp = translate (points[0], -9 , -9) # move points to new origin
    tp = reflect (trans_pts[0], axis = 1, inplace = True) # reflect the points.
        

    # iterate over the matrix rows of points transforming each point to the new origin
    for idx, point in enumerate(points):
        # print (idx, point)
        trans_pts[idx] = translate(point, -9, -9)
    trans_pts

    # print ('points, the coordinates of each point:')
    # print (points)
    # print ('trans_pts, the transformed points at the new origin')
    # print (trans_pts)
    
    # reflect the these points about the x and then the y axis about the new origin

    reflect_pts = np.zeros((trans_pts.shape[0]*4,3)) # make the zeros matrix for transformed points. 4 x poits after double reflection.

    for idx, point in enumerate(trans_pts):
        reflect_pts[idx] = point
        reflect_pts[idx+trans_pts.shape[0]] = reflect (point, axis = 1, inplace = False) # reflect across y axis
        
        
    for idx, point in enumerate(reflect_pts[0:reflect_pts.shape[0]//2]):
        reflect_pts[idx+reflect_pts.shape[0]//2] = reflect (point, axis = -1, inplace = False)  # reflect across x axis

    # reflect_pts, are all the points from the initial grid reflected about both axis at the centre of the grid

    # get the points with a common color
    n_common_colors, color_pairs, color_points = common_color (points);
    edge_points = [reflect_pts[reflect_pts[:,2]==color, :] for color in color_pairs]
    # color_pairs, edge_points

    # get the pairs of points that are at the edge of the grid that we want to join.
    # we should do this for a loop of edge_points, in range(n_common_points)
    join_pts = []
    for edge in range(n_common_colors):
        max_x_pair, min_x_pair, max_y_pair, min_y_pair = select_pairs (edge_points[edge])
        print (n_common_colors, edge, max_x_pair, min_x_pair, max_y_pair, min_y_pair)
        # call join points here to join all the pairs
        joins = [(max_x_pair[0], max_x_pair[1], 2), (min_x_pair[0], min_x_pair[1], 2),
                 (max_y_pair[0], max_y_pair[1], 2), (min_y_pair[0], min_y_pair[1], 2)]
        jp = [flatten(join_pairs (*join)) for join in joins]
        join_pts.append(np.array (flatten (jp)))
        # join_pts, list of arrays of points for each color to stack

    # make the array of all the points with hstack
    all_trans_pts =  copy.deepcopy(reflect_pts)

    for pts in range(len(join_pts)):
        all_trans_pts =  np.vstack((all_trans_pts, join_pts[pts]))
    
    # transform the array back to the matrix coordinate system

    # create the empty array the same size as all the points.
    all_pts = np.zeros_like(all_trans_pts).astype(int)

    # translate the coordinate positions back to the orgin in the bottom left of the array.
    for idx, row in enumerate (all_trans_pts):
        all_pts[idx] = translate (row, 9 , 9)

    # print (all_pts)

    # open a blank final grid
    final_grid = np.zeros((19,19))

    # for each point, populate the array with the color given (x,y,color) to (r,c, color)
    for row in all_pts:
        final_grid [row[1], final_grid.shape[1] -1 - row[0]] = row[2]

    final_grid = final_grid.reshape((19,19))
    
    # print ('The final grid:')
    # print (final_grid)
    
        
    return final_grid

################################   
####  810b9b61              ####
################################

def solve_810b9b61(x):
    """
    Stephen Jeffery, 21249306
    https://github.com/sbjeffery/ARC
    
    The initial grid consists several shapes, shapes consiting of closed rings keep their
    initial colour (green) while open shapes are given a new colour (blue). Shapes that are closed against
    the grid edge, but not closed within the grid are trated as open shapes.
    
    This function uses the skimage momdule to identify the individual regions as shapes.
    I initially select a background point, and use this as a point to flood fill the grid.
    The flood fill checks for regions closed by the grid edge, and fills these also, we
    note that this only occurs for regions one square deep.
    Having completed the flood fill the points from closed shapes are adjacent to unfilled
    points. is_adjacent takes two points a returns True if they are adjacent.
    Closed points are selected by checking if each is_adjacent to a closed region.
    The closed regions are looped over and the colour changed.  
    
    All training and test grids are solved correctly.
       
    This program uses the skimage module to select and label regions & complete tranformations.
    I wanted to make more use of existing modules to identify regions and complete typical
    pixel operations like floodfill. Once masks are created with the skimage.measure.label
    function, we use numpy and filter using the masks to update the points.
    # https://scikit-image.org/docs/stable/user_guide/getting_started.html
    
    Stephen Jeffery 21249306, Nov 2021    
    """
    # imports for numpy and skimage which we'll use to manipulate the image.
    import numpy as np
    import matplotlib.pyplot as plt

    # https://scikit-image.org/docs/stable/user_guide/getting_started.html
    import skimage
    from skimage.morphology import flood_fill
    
    def select_bckgrd_seed_pt(A):
        """
        Takes the array like A, and selects a background point having an array entry of zero
        returns the tuple of (r, c) of this point which we use to seed a flood fill.
        """
        seed_pt = ()
        for col in range(A.shape[1]):
            for row in range(A.shape[0]):
                if A[(row,col)] == 0:
                    seed_pt = (row,col)
                    # print (f'bckgrd seed point = ({row} , {col})')
                    return seed_pt

        if seed_pt == ():
            raise ValueError ('Opps, we have a problem trying to find the seed point')

        
        
    def points_from_region(A, region = 0):
        """
        argumments are A an array like region with area labels.
        region, is the label number, usually a integer value.
        return the list of points (r,c) for each point in the region.
        """
        res_row, res_col = np.where(A == region)
        return list(zip(res_row, res_col))
    
    
    def is_adjacent (p1,p2):
        """
        Takes two points and checks to see if they are adjacent.
        returns true if they are adacent and false otherwise.
        p1 and p2 are tuples of (r, c) or (x, y) coordinates, but should both use the same coordinate system.

        >>> is_adjacent ((3,3), (4,4))
        True

        >>> is_adjacent ((3,3), (3,5))
        False

        """
        x1, y1 = p1 
        x2, y2 = p2

        return (abs(x1-x2) <= 1) and (abs(y1-y2)<=1)


    def select_closed(A, c, n_labels):
        """
        A is the array of region labels
        c is the label for the close points
        n_labels are the number of labels in A


        """
        selected = []
        closed_points = points_from_region(A, c)

        for region in range(2, n_labels):
            test_point = points_from_region(mask_label, region = region)[0] # just one test point is sufficient.

            for c_pt in closed_points:
                # print (test_point, c_pt)
                if is_adjacent(test_point, c_pt):
                    selected.append(region)

        return selected



    ## Main program section.
    
    # get the seed point for the flood_fill, which is a background 0 point in the array.
    seed_pt = select_bckgrd_seed_pt (x)
    # print (f'seed point = {seed_pt}')
    
    # flood fill from the seed point, leaving the closed regions unchanged
    mask = flood_fill(x, seed_pt, new_value = 2)
    print (f' mask = {mask}')
    
    # Check for edge cases where fill can't fill due to closed by edge.
    r_mask,c_mask = mask.shape
    closed_points = np.where (mask == 0)
    closed_coords = list(zip(closed_points[0], closed_points[1]))
    closed_coords = [(r,c) for (r, c) in closed_coords if (c == 0 or c == c_mask or r == r_mask or r == 0)]
    # it would be better to create fills at each of these potential points and merge them, but this works
    # for our specific set.
    for (r,c) in closed_coords:
        mask[r,c] = 2
    
    # apply labels over each region to act as a mask for the individual items.
    # there are n_labels differetn regions in the array. background is zero.
    mask_label, n_labels = skimage.measure.label(mask, return_num = True)
    # print (f'mask labels generated for {n_labels} labels')
    
    # select the closed regions, closed regions are adjacent to the points that did not get flood filled.
    
    target = x.copy() # make a copy of the training array so we can keep unchanged elements
    color = 3  # select the color we want to change the closed shapes to.

    for sel in (select_closed(mask_label, 0, n_labels)): # loop over the closed shape labels.
        # overwrite each closed shape with the new color
        change_col = mask_label * (mask_label == sel) * color/sel  
        np.place (change_col, change_col == 0, [1]) 
        # print (change_col)
        target = target * change_col

    x = target
    return x



#################################
####       36d67576          ####
#################################
def solve_36d67576(x):

    """
    Stephen Jeffery, 21249306
    https://github.com/sbjeffery/ARC
    
    The initial traiing grids consist of 3 or 4 shapes, one shapes is a pattern 
    of the complete shape. The other shapes in teh training grid are partial pattern
    shapes. The output shapes have the pattern transformed onto the targets by
    one of a variety of reflections, rotations or translations. The shapes have
    yellow, red, blue and green colours. In all shapes there is a single red pixel
    which defines the translation of the pattern to the target.
    
    This function uses the skimage momdule to identify the individual regions as shapes.
    Each shape region is then treated as if it were in a new dimension.
    Tranformations also use the skimage affine tranformation functions, and
    a separate rotate function.
    I defined a score function, where the score is the number of non-overlapping pixels
    between the target and the pattern shape.
    There are 8 possible combinations of tranformations, the potential tranformed
    points and score are calculated over each target shape and the tranformation with
    a score of 0 selected. Finally all transformed target shapes are joined together
    to make the final array.
    
    All training and test grids are solved correctly.
       
    This program uses the skimage module to select and label regions & complete tranformations.
    I thought this might be a more interesting example with the function selecting the required
    transformation from a finite set. Extended use of the sklearn functions for transformations
    helps make the individual functions much more readable.
    Given more time, I think I would have refacted the main functions t1 - t4 again,
    to combine them into one transform function - splitting them into the main translation
    methods appears more readable in this case where we only use each function 3 times.
    
    
    # https://scikit-image.org/docs/stable/user_guide/getting_started.html
    
    Stephen Jeffery 21249306, Nov 2021    
    """
    
    # https://scikit-image.org/docs/stable/user_guide/getting_started.html
    import skimage
    from skimage.transform import matrix_transform, SimilarityTransform, rotate, AffineTransform
        
    import math
    
    # https://scikit-image.org/docs/0.13.x/auto_examples/xx_applications/plot_geometric.html
    
    # https://scikit-image.org/docs/0.13.x/api/skimage.transform.html?highlight=rotate#skimage.transform.SimilarityTransform

    def scores (target_region, transformed_pts):
        """
        Calculates a score for which is the number of non-overlapping squares between the target and the transformed pattern.
        A score of zero is a perfect match, a score > the zero indicates how many points are unmatching.
        As we translate the red points on to each other the score should not always be > the (number of target points - 1)
        A score >= number of target points indicates an error in the tansformation.
        
        """
        target_score = n_reg_pts[target_region]
        intersection = len(set([(x,y) for [x,y] in transformed_pts]).intersection(set(target_coords[target_region])))
        # print (intersection, target_score)
        score = (target_score - intersection)
        return score  # a score of 0 is a match.
    
    
    # get the red points for the targets and the two transformations (target - pattern)

    def get_transform_xy (p_red):
        """
        Input the coords of a red point. Return the transformation tx, ty reququird to move the red points from the
        pattern to the target regions. Returned as a list of lists [[tx,ty],..]
        """
        
        targets_trfm = []
        for target_region in target_regions:
            # print (target_coords[target_region], np.shape(target_regions), np.shape((targets, np.where (targets[target_region] == 2))))
            # print ((targets[target_region]))
            # print (f' tx{target_region} = {(np.where (targets[target_region] == 2))[1][0]}')
            # print (f' ty{target_region} = {(np.where (targets[target_region] == 2))[1][0]}')
            reds = [(np.where (targets[target_region] == 2))[0][0], (np.where (targets[target_region] == 2))[1][0]]
            tx = reds[0] - p_red[0]
            ty = reds[1] - p_red[1]
            trfm = [tx,ty]    
            targets_trfm.append(trfm)

        return targets_trfm


    
    def tform (P, xt=0, yt=0, rotn = 0, matrix = None):
        """
        transform a point, P or an array of points, with translation xt, yt and rotation, counter clockwise in rads.
        so math.pi/2 is the 90 deg rotation. if you join translate and rotate it rotates first.
        returns the transformed coordinate

        >>> P1 = tform(pattern_pts[0], 1, 1, 0)
        array([[2., 3.]])

        >>> mirror = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> P2 = tform(P1, 1, 1, 0, matrix = mirror)
        array([[3., 4.]])

        >>> P3 = tform(P1, 0, 0, math.pi/2, 0)
        array([[-3.,  2.]])

        """
        tform_translate = SimilarityTransform(translation=(xt, yt), rotation = rotn)
        return tform_translate(P)
    
    mirror = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tform_reflect = AffineTransform(matrix = mirror) # is reflecting about the y-axis
        

    def t1 (points, tx, ty):
        """
        Translate point or array of points by tx, ty and return the transformed point or array.
        In this probelm we always tranform pattern points.
        """
        return tform(points ,tx , ty, 0) # transform only, P1 by tx, ty


    # t2 = [tform_reflect(pt) for pt in pattern_pts]
    def t2 (points):
        """
        reflection followed by transformation of the pattern points.
        returns transformed points.
        """ 
        pt_red = tform (tform_reflect(tform(pattern_red))) # completes a reflection of the red point.
        tx, ty = get_transform_xy(pt_red[0])[target_n] # for the target_n target region.
        #print (f' deltas = {tx}, {ty}')

        pt2 = t1 (tform_reflect(tform(points)), tx, ty ) # completes a reflection of all pattern

        #print (pattern_red, pt_red, pt2)

        return [(round(x),round(y)) for [x,y] in pt2]

    
    def t3 (points, rads):
        """
        rotation followed by translation of the pattern points.
        returns transformed points.
        rot, rotation is rad counter clockwise.
        """
        pt_red = tform(pattern_red, 0, 0, rads) # completes a rotation of the red point for the pattern.
        tx, ty = get_transform_xy(pt_red[0])[target_n] # get the tx, ty for the target_n target region.
        #print (f' deltas = {tx}, {ty}')

        pt_rot = tform(points, 0, 0, rads) # completes a rotation and tranlastion by tx, ty of all pattern points

        pt3 = t1(pt_rot, tx, ty)

        #print (pattern_red, pt_red)
        #print (points, pt_rot, pt3)

        return [(round(x),round(y)) for [x,y] in pt3]
    
    
    def t4 (points, rads):
        """
        rotation & reflection followed by translation of the pattern points.
        returns transformed points.
        rot, rotation is rad counter clockwise.
        """
        pt_red = tform(pattern_red, 0, 0, rads) # completes a rotation of the red point for the pattern.
        pt_red2 = tform (tform_reflect(tform(pt_red))) # completes a reflection of the red point.
        tx, ty = get_transform_xy(pt_red2[0])[target_n] # get the tx, ty for the target_n target region.
        #print (f' deltas = {tx}, {ty}')
        #print (pattern_red, pt_red)
        #print (f' pt_red2 point double transformed; {pt_red2}')

        pt_rot = tform(points, 0, 0, rads) # completes a rotation and tranlastion by tx, ty of all pattern points
        pt6 = t1 (tform_reflect(tform(pt_rot)), tx, ty ) # completes a reflection of all pattern

        #print (points, pt_rot, pt6)

        return [(round(x),round(y)) for [x,y] in pt6]

    

    
    
    
    ###########################
    ## Main section here
    
    input_grid = x

    # get_regions of connected points.
    # here's an array of all one color with the individual regions.
    mono = (input_grid > 0) * np.ones_like(input_grid)
    regions, n_regions = skimage.measure.label(mono, return_num = True) # mark the regions

    # How many points are in each region, the largest region is the pattern, other regions are targets.
    region_coords = []
    n_region_pts = []

    for region in range (n_regions):
        r,c = np.nonzero(regions == region+1)
        coords = list(zip(r, c))
        region_coords.append(coords) # is a list of tuples.
        n_region_pts.append ((region+1, len (coords)))

    # Sort the region sizes and get the pattern and list of other regions.
    # print (region_coords)
    n_region_pts.sort (key = lambda x:x[1], reverse = True)
    # print (n_region_pts[0]) # this is the pattern region, as it is the largest group, the other regions are the target regions.
    pattern_region = n_region_pts[0][0]
    # print (f'The pattern region is label = {pattern_region}')
    pattern_coords = region_coords[pattern_region-1]

    target_regions = [x for (x,_) in n_region_pts[1:]]
    # print (f'target regions are {target_regions}')

    target_coords = {r:region_coords[r-1] for r in target_regions} # dictionary of coordinates for each region
    
    # the movement of the red points defines the translation.
    # identify the red points in the image and get their coordinate.
    reds = (input_grid == 2) * np.ones_like(input_grid)
    red_regions = skimage.measure.label(reds) # mark the regions
    
    # the coordinates of the red points.
    for region in range(n_regions):
        r,c = np.nonzero(red_regions == region+1)
        coords = list(zip(r, c))
        
    # lets put the colors back with the pattern_coords coordinates.
    color_pattern = [(coord, input_grid[coord[0], coord[1]]) for coord in pattern_coords]
    # we want a list of lists of coords for each pattern to pass to the transform function as eacah target has different transfer functions.
    pattern_pts = [[coord[0], coord[1]] for coord in pattern_coords]
    
    # get the array of just the pattern points that we will transform, and the red points.
    pattern = input_grid * (regions == pattern_region)
    pattern_red = [(np.where (pattern == 2))[0][0], (np.where (pattern == 2))[1][0]]
    
    # Make the target arrays for each target region. We are aiming to translate the pattern to these targets.
    targ = []
    for target_region in target_regions:
        # print (target_region)
        targ.append(input_grid * (regions == target_region))

    targets = dict(zip(target_regions, targ))
    targets_trfm = get_transform_xy (pattern_red)
    
    # Main loop over the each target region returning the dict of {target_region:target grids}
    # The loop evaluates and scores each of the 8 possible transforms and selects the transform with score = 0.
    all_scores = {} # dict for the 8 possible function scores 
    target_grid = {} # dict of target_region: array of final points.   
    
    for target_region in target_regions: # for regions labeled from 1 with 1 = pattern so first target = 2 loop target_region in target_regions

        target_n = target_region - 2 # first target region, matched to target_region
        n_reg_pts = dict(n_region_pts) # gets a dict of {region : n_pts} for the target function.
        
        # tx, ty = targets_trfm [target_n][0], targets_trfm [target_n][1]
        
        # Transformation using t1, translate only by tx, ty gives pt1.
        pt1 = t1(pattern_pts, targets_trfm[target_n][0], targets_trfm[target_n][1])
        all_scores.update({'pt1': scores(target_region, pt1)})
        
        # translate (reflect (pattern_points))
        pt2 = t2(pattern_pts)        
        all_scores.update({'pt2': scores(target_region, pt2)})
        
        pt3 = t3 (pattern_pts, math.pi*3/2)
        all_scores.update({'pt3': scores(target_region, pt3)}) # for 270 deg rotation.
        
        pt4 = t3 (pattern_pts, math.pi/2)  # for 90 deg rotation.
        all_scores.update({'pt4': scores(target_region, pt4)})
        
        pt5 = t3 (pattern_pts, math.pi)  # for 180 deg rotation.
        all_scores.update({'pt5': scores(target_region, pt5)})
        
        pt6 = t4 (pattern_pts, math.pi*1/2) # for pt6 rotation, reflection and translate
        all_scores.update({'pt6': scores(target_region, pt6)})
        
        pt7 = t4 (pattern_pts, math.pi) # for pt7 rotation, reflect and translate
        all_scores.update({'pt7': scores(target_region, pt7)})
        
        pt8 = t4 (pattern_pts, math.pi * 3/2)  # for pt8 rotation, reflect and translate
        all_scores.update({'pt8': scores(target_region, pt8)})
        
        # select the transform where the score = 0
        select = { v:k for k,v in all_scores.items()}
        select[0] # the score will always be 0 for the selected transforms.

        # get the points from the selected transformation * add the colors back to the tranformed location.
        target_res = [(int(t1), int(t2), c) for [(t1,t2), (z, c)] in list(zip(eval(select[0]), color_pattern))]

        # open a blank grid for this target_region

        t_grid = np.zeros_like(pattern)

        # for each point, populate the array with the color given (r,c, color)
        for row in target_res:
            t_grid [row[0], row[1]] = row[2]

        target_grid.update ({target_region : t_grid})

    #Finally put the individual target arrays back together.

    final_grid = np.zeros_like(pattern)
    final_grid += pattern

    for n in target_regions:
        final_grid += target_grid[n]

    x = final_grid
    return x




#################
def solve_4347f46a(x):
    """ 
    Stephen Jeffery, 21249306
    https://github.com/sbjeffery/ARC
    
    The initial grid consists several rectangles of different colors in different grid sizes.
    The target arrays are the outlines of the rectangular shapes in the same colour
    on the same grid size.
    
    This function uses the skimage momdule to draw the outlines.
        
    All training and test grids are solved correctly.
    
    Initially I though this one looked more difficult, attached here just
    for the fun of doing it in one line. Easy once you've found the line!
    
    # https://scikit-image.org/docs/stable/user_guide/getting_started.html
    
    """

    from skimage.segmentation import find_boundaries
    return find_boundaries(x, mode = 'inner') * x


################

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

