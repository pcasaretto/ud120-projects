#!/usr/bin/python

import numpy

def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """

    error = numpy.subtract(predictions, net_worths)
    max_value = numpy.percentile(error, 90)

    tuples = zip(ages, net_worths, error)
    print(tuples)

    return filter(lambda (a, n, error): error < max_value, tuples)
