# class Interval:
#     """
#     Representation of an intervals bound.
#     """

#     def __init__(self, lower_bound, upper_bound):
#         """
#         An interval of a feature.
#         :param lower_bound: The lower boundary of the feature.
#         :type lower_bound: `double` or `-np.inf`
#         :param upper_bound: The upper boundary of the feature.
#         :type upper_bound: `double` or `np.inf`
#         """
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound

#     def __repr__(self):
#         return "[{}, {}]".format(self.lower_bound, self.upper_bound)


class Box:
    """
    Representation of a box of intervals bounds.
    """

    def __init__(self, intervals=None):
        """
        A box of intervals.
        :param intervals: A dictionary of intervals with features as keys.
        :type intervals: `dict(feature: Interval)`
        """
        if intervals is None:
            self.intervals = dict()
        else:
            self.intervals = intervals

    def is_intersect(self, box):
        for key, value in box.intervals.items():
            if key not in self.intervals:
                continue
            else:
                lower_bound = max(self.intervals[key][0], value[0])
                upper_bound = min(self.intervals[key][1], value[1])

                if lower_bound >= upper_bound:
                    return False

        return True

    def intersect_with_box(self, box):
        """
        Get the intersection of two interval boxes. This function modifies this box instance.
        :param box: Interval box to intersect with this box.
        :type box: `Box`
        """
        for key, value in box.intervals.items():
            if key not in self.intervals:
                self.intervals[key] = value
            else:
                lower_bound = max(self.intervals[key][0], value[0])
                upper_bound = min(self.intervals[key][1], value[1])

                if lower_bound >= upper_bound:
                    lower_bound = upper_bound
                    self.intervals[key] = (lower_bound, upper_bound)
                    break

                self.intervals[key] = (lower_bound, upper_bound)

    def get_intersection(self, box):
        """
        Get the intersection of two interval boxes. This function creates a new box instance.
        :param box: Interval box to intersect with this box.
        :type box: `Box`
        """
        box_new = Box(intervals=self.intervals.copy())

        for key, value in box.intervals.items():
            if key not in box_new.intervals:
                box_new.intervals[key] = value
            else:
                lower_bound = max(box_new.intervals[key][0], value[0])
                upper_bound = min(box_new.intervals[key][1], value[1])

                if lower_bound >= upper_bound:
                    lower_bound = upper_bound
                    box_new.intervals[key] = (lower_bound, upper_bound)
                    return box_new

                box_new.intervals[key] = (lower_bound, upper_bound)

        return box_new

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.intervals)
