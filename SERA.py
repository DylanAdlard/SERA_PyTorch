import collections
import math

import numpy as np
import scipy.integrate
import scipy.interpolate
import torch


class SERA(torch.autograd.Function):
    """A PyTorch compatible, Python implementation of the SERA loss function"""

    @staticmethod
    def foward(ctx, y_pred, y_true, control):
        """The forward Pass - returns the loss"""

        # defense
        assert (
            y_pred.shape == y_true.shape
        ), "the predicted and true arrays should be np arrays of equal length"
        assert isinstance(control, list) or isinstance(
            control, tuple
        ), "the control set must be either a list or tuple"
        assert (
            len(control) == 3
        ), "the control set should contain three floats or integers"

        # calculate the bounds of the skewness and outlier adjusted boxplot
        true_bounds = SERA._relevance_interval(y_true)

        # remove outliers from the true array and their corresponding //
        # predictions from the predicted array
        y_true, y_pred = SERA._filter_outliers(y_true, y_pred, true_bounds)

        # generate the 'relevance curve'
        relevance = SERA.interpolator(y_true, true_bounds, control)

        # save tensors for backwards pass
        y_true_tens, y_pred_tens = torch.tensor([y_true]), torch.tensor([y_pred])
        rel_tens, rel_contr = torch.tensor([relevance]), torch.tensor([control])
        ctx.save_for_backward(y_pred_tens, y_true_tens, rel_tens, rel_contr)

        return torch.tensor([SERA.sera(relevance, y_true, y_pred, control)])

    @staticmethod
    def backward(ctx, grad_output):
        """The backwards pass - returns the gradient on backpropagation"""

        # load in tensors
        y_pred, y_true, relevance, control = ctx.saved_tensors

        control = control[0].numpy()
        # determine minumum and maximum values of user-defined relevance for the control set
        min_control, max_control = np.min(control), np.max(control)

        # calculate the gradient at each predicted value
        grad_input = [
            scipy.integrate.quad(
                lambda t: SERA.dser(y_true, x, t, relevance), min_control, max_control
            )[0]
            for x in y_pred.numpy()[0]
        ]
        # convert gradient array to tensor
        grad_input = torch.tensor([grad_input])

        return grad_input, None, None

    @staticmethod
    def sera(relevance, y_true, y_pred, control):
        """calculates the sera loss value"""

        # determine minumum and maximum values of user-defined relevance for the control set
        min_control, max_control = np.min(control), np.max(control)

        # calculate the area under the ser curve (ie. the loss)
        sera = scipy.integrate.quad(
            lambda t: SERA.ser(t, relevance, y_true, y_pred), min_control, max_control
        )

        return sera[0]

    @staticmethod
    def ser(cutoff, relevance, y_true, y_pred):
        """calculate and return sum of squared error relevance"""

        # filter out true and predicted values below the relevance cutoff passed in during integration
        rel_y_true, rel_y_pred = [], []
        for i in range(len(relevance)):
            if relevance[i] > cutoff:
                rel_y_true.append(y_true[i])
                rel_y_pred.append(y_pred[i])

        # ser
        return np.sum((np.array(rel_y_pred) - np.array(rel_y_true)) ** 2)

    @staticmethod
    def dser(y_true, x, cutoff, relevance):
        """calculate and return differentiated sum of squared error relevance (wrt y_pred)"""

        y_true = y_true.numpy()[0]
        relevance = relevance.numpy()[0]

        # filter out true values below the relevnace cutoff passed in during integration
        rel_y_true = []
        for i in range(len(relevance)):
            if relevance[i] > cutoff:
                rel_y_true.append(y_true[i])

        # dser
        return 2 * np.sum(x - np.array(rel_y_true))

    @staticmethod
    def interpolator(array, true_bounds, control):
        """Calculates the interpolation function/curve using a set of 3 control
        points and their corresponding relevance. These control point relevance values
        can be manually tuned. The interpolation function is used to estimate
        relevance through the y_true array."""

        sorted_arr = np.sort(array)
        argsorted = np.argsort(array)

        # determine the x values for the control ponts (lower adj, median, upper_adj)
        control_x = SERA._control_points(sorted_arr, true_bounds)

        relevance_y = control

        # generate the interpolation function
        pchip = scipy.interpolate.PchipInterpolator(control_x, relevance_y)

        # calculate interpolated relevance for each true value
        y = pchip.__call__(sorted_arr)

        # preserve the original ordering of true values (so relevance arr
        # indexes match corresonding predicted values)
        Dict = {}
        for i in range(len(argsorted)):
            Dict[argsorted[i]] = y[i]
        od = collections.OrderedDict(sorted(Dict.items()))
        arr = []
        for v in od.values():
            arr.append(v)

        return np.array(arr)

    @staticmethod
    def _control_points(array, true_bounds):
        """Determines the x-values of the set of 3 control points for the interpolator."""

        array = np.sort(array)

        lower_bound, upper_bound = true_bounds[0], true_bounds[1]

        upper_adj = None

        # determine lower adjacent value = 1st control point
        for i in range(len(array)):
            if array[i] > lower_bound:
                lower_adj = array[i]
                break

        # determine upper adjacent value = 3rd control point
        for i in range(len(array)):
            if array[i] >= upper_bound:
                if array[i - 1]:
                    upper_adj = array[i - 1]
                    break

        if not upper_adj:
            upper_adj = np.max(array)

        interval = [i for i in array if (i > lower_bound) & (i < upper_bound)]
        # determine median of the interval = 2nd control point
        centrality_val = np.median(interval)

        return np.array([lower_adj, centrality_val, upper_adj])

    @staticmethod
    def _filter_outliers(y_true, y_pred, true_bounds):
        """remove outliers as determined by the boxplot and the medcouple"""

        filt_y_true, filt_y_pred = [], []
        for i in range(len(y_true)):
            if (y_true[i] >= true_bounds[0]) and (y_true[i] < true_bounds[1]):
                filt_y_true.append(y_true[i])
                filt_y_pred.append(y_pred[i])
            else:
                med = np.median(y_true)
                filt_y_true.append(med)
                filt_y_pred.append(med)

        return (filt_y_true, filt_y_pred)

    @staticmethod
    def _relevance_interval(array):
        """calculates the interval for which relevance is determined over"""

        # generate Tukey's boxplot
        boxplot = SERA._boxplot(array)

        # calculate the medcouple
        MC = SERA._medcouple(array)

        # calculate the bounds of the interval, as per the original paper
        if MC >= 0:
            lower_bound = boxplot["q1"] - (1.5 * math.exp(-4 * MC) * boxplot["iqr"])
            upper_bound = boxplot["q3"] + (1.5 * math.exp(3 * MC) * boxplot["iqr"])
        else:
            lower_bound = boxplot["q1"] - (1.5 * math.exp(-3 * MC) * boxplot["iqr"])
            upper_bound = boxplot["q3"] + (1.5 * math.exp(4 * MC) * boxplot["iqr"])

        return (lower_bound, upper_bound)

    def _medcouple(array):
        """calculate the medcouple [Byrs et al. 2004]"""

        y = np.asarray(array, dtype=np.double).ravel()
        y = np.sort(y)

        n = y.shape[0]
        if n % 2 == 0:
            median = (y[(n // 2) - 1] + y[n // 2]) / 2
        else:
            median = y[(n - 1) // 2]

        # z = range between value and median (half will be negative, the other half positive)
        # centres data around 0
        z = y - median
        lower = z[z <= 0.0]
        upper = z[z >= 0.0]
        # create axis with length 1 (None is an alias for np.newaxis) = matrix with shape (upper, 1)
        upper = upper[:, None]

        standardise = upper - lower

        # returns truth value array of lower AND upper == 0
        is_zero = np.logical_and(lower == 0.0, upper == 0.0)
        # convert zeros to inf
        standardise[is_zero] = np.inf
        spread = upper + lower
        h = spread / standardise

        num_ties = np.sum(lower == 0.0)
        if num_ties:
            # Replacements has -1 above the anti-diagonal, 0 on the anti-diagonal, and 1 below the anti-diagonal.
            #   Create matrix of ones with dim (num_ties, num_ties) and subtract identity matrix
            #   = 'inside-out' identity matrix
            replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
            # convert ones to -ones in a triangularised matrix
            # (np.triu(replacements) returns upper triangle matrix (bottom left corner zeroed))
            replacements -= 2 * np.triu(replacements)
            # convert diagonal to anti-diagonal
            replacements = np.fliplr(replacements)
            # replace upper right block
            h[:num_ties, -num_ties:] = replacements

        return np.median(h)

    @staticmethod
    def _boxplot(array):
        """Generate Tukey's boxplot and return the first and third quartile,
        as well as the interquartile range."""

        q3, q1 = np.percentile(array, [75, 25])
        iqr = q3 - q1

        return {"q1": q1, "q3": q3, "iqr": iqr}
