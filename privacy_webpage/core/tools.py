import numpy as np
import sys
from operator import itemgetter

sys.path.insert(0, "../../")

from core.probabilitybuckets_light import ProbabilityBuckets as PB

def delta_dist_events(dist1, dist2):
    # This computation of distinguishing events is numerically unstable and delta_dist_0 * n is an upper bound.
    # delta_dist = (1 - (1 - delta_dist_0)**n) 
    return np.sum(dist1[np.where(dist2 == 0)])

def include_dist_events(delta, delta_dist_0, r):
    delta_dist = delta_dist_0 * r
    return delta * (1 - delta_dist) + delta_dist

def delta_of_eps(eps, f_delta_of, upper_bound = None):
    f = lambda eps_1: exp(eps_1) - exp(eps) + f_delta_of(eps_1)
    res = minimize(f, [eps], bounds = [(eps, upper_bound)])
    return min(1, min(f_delta_of(eps),f(res.x[0])))

def eps_of_delta_for_ed(delta, ed, tolerance = None):
    ## somehow minimize does not work reliably; so, I switched to binary search
    # f = lambda eps: abs(delta - ed(eps))
    # opt_res = minimize(f, [self.initial_value], bounds=[(self.bound_low, None)], tol=tolerance)
    # return opt_res.x[0]
    if tolerance is None:
        tolerance = 10**(-3)
    tolerance = delta * tolerance
    eps = 0
    left_border = 0
    right_border = 10**-7
    current_value = ed(eps)
    current_distance = abs(current_value - delta)
    # go to the vicinity of the target and upper bound the right border
    max_eps = 10**7
    while current_value > delta:
        prior_value = copy.copy(current_value)
        current_value = ed(eps)
        if right_border >= max_eps:
            print("DEBUG: ed(12) = " + str(ed(12)))
            raise ValueError("There is no epsilon <= " + str(max_eps) + ", for which delta is <= the target delta " + str(delta) + ". Values: prior_value = " + str(prior_value) + ", current_value = " + str(current_value) + ", right_border = " + str(right_border))
        current_distance = abs(current_value - delta)
        right_border *= 2
        eps = right_border
    # binary search
    stop = False
    i = 0
    while not stop:
        i += 1
        # evaluate ed(eps)
        prior_value = current_value
        current_value = ed(eps)
        current_distance = current_value - delta

        if (tolerance >= current_value - delta >= - tolerance) or (current_value < delta and eps == 0): \
        # ed(eps) is within acceptable bounds or ed(eps) is too small but eps is already 0
            stop = True
        elif current_value < delta and eps != 0: # eps too large
            right_border = eps
            eps = (right_border + left_border) * 0.5
        elif current_value > delta: # eps too small
            left_border = eps
            eps = (right_border + left_border) * 0.5

        if i % 1000 == 0:
            print("[*] target(@" + str(i) + ") = " + str(delta))
            print("    eps = " + str(eps))
            print("    current_value = " + str(ed(eps)))
            print("    current distance = " + str(abs(current_value - delta)))
            print("    left border = " + str(left_border))
            print("    value at left border = " + str(ed(left_border)))
            print("    right border = " + str(right_border))
            print("    value at right border = " + str(ed(right_border)))
        if i > 10**4:
            print("[*] Aborting")
            exit(0)
    # print("Returning " + str(eps))
    return eps

def true_delta(dist1, dist2, e_eps = 1):
    delta = 0
    if len(dist1) == len(dist2):
        delta = np.sum(np.maximum(dist1 - e_eps * dist2, 0))
    else:
        raise ValueError( "Error: Distributions have different number of atomic events.")
    return delta

def sanitize_distributions(dist1,dist2):
    if dist1.shape != dist2.shape or dist1.ndim != 1:
        raise ValueError("The shape of dist1 and dist2 is not the same or the distributions are not arrays.")
    support_indices = []
    for x in range(len(dist1)):
        if dist2[x] != 0 and dist1[x] != 0:
            support_indices.append(x)
    return (np.asarray([dist1[index] for index in support_indices]), \
            np.asarray([dist2[index] for index in support_indices]))

def optimal_factor(dist1, dist2, number_of_buckets, x_range = 50, infty_budget = None):
    ## By default only tolerate factors for which the infinity bucket plus distinguishing events is <= delta(20).
    ## To find a better factor, we sometimes want to tolerate more than delta(20), e.g., for Gaussians.
    if infty_budget is None:
        infty_budget = true_delta(dist1, dist2, e_eps = np.exp(20))
    dist1 = np.asarray(dist1)
    dist2 = np.asarray(dist2)
    (dist1_sanitized, dist2_sanitized) = sanitize_distributions(dist1,dist2)
    ## We can compute the mu and sigma without calling PB.
    mu = np.sum(np.multiply(np.log(dist1_sanitized / dist2_sanitized), dist1_sanitized))
    sigma = np.sqrt(np.sum(np.multiply(np.power((np.log(dist1_sanitized / dist2_sanitized) - mu), 2), dist1_sanitized)))
    ## Look for the factor that causes the smallest error: delta / true_delta
    factor_candidate = lambda x: np.exp((mu + x * sigma) / float(number_of_buckets))
    module_params = {'dist1_array' : dist1_sanitized\
                    ,'dist2_array' : dist2_sanitized\
                    ,'number_of_buckets' : number_of_buckets\
                    ,'error_correction' : False\
                    }
    errors = []
    ## Try for different x from 1/x_range to x_range
    for x in np.linspace(1/float(x_range), x_range, 20, endpoint = True):
        module_params['factor'] = factor_candidate(x)
        instance = PB(**module_params)
        ## only consider those factors where the infinity bucket is smaller than 
        if instance.infty_bucket + instance.distinguishing_events <= infty_budget:
            errors.append((instance.delta_of_eps_upper_bound(0) / true_delta(dist1_sanitized, dist2_sanitized), x))
            # errors.append((instance.delta_of_eps_upper_bound(0) / true_delta(dist1, dist2), \
            #                 x, factor_candidate(x), instance.delta_of_eps_upper_bound(0), true_delta(dist1, dist2)))
        del instance
    # print([(x[0],x[2],x[3],x[4]) for x in errors])
    errors.sort(key = lambda x: x[0])
    if errors != []:
        x = errors[0][1]
    else:
        raise ValueError(   "\
No factor found. Infinity bucket + mass of distinguishing events > delta(20) (eps = 20).\
Try a higher number of buckets (keyword 'number_of_buckets' in the PB-constructor) \
or a wider range for x (keyword 'x_range', default is 50).")
    return factor_candidate(x)

def compute_initial_bounds(module_params):
    if 'verbose' in module_params:
        verbose = module_params['verbose']
    else:
        verbose = False
    instance = PB(**module_params)
    true_delta_value = true_delta(module_params['dist1_array'], module_params['dist2_array'])
    error = instance.delta_of_eps_upper_bound(0)/true_delta_value
    infty_bucket = instance.infty_bucket + instance.distinguishing_events
    # infty_bucket_ratio = infty_bucket/true_delta_value
    if verbose:
        print("[*] FACTOR:\t\t\t\t 1 + %.20f" % (module_params['factor'] - 1.0))
        print("[*] INFTY:\t\t\t\t %.20f" % infty_bucket)
        print("[*] ERROR:\t\t\t\t %.20f" % error)
    return error, infty_bucket, true_delta_value


def get_sufficient_big_factor(module_params, max_comps = None, target_delta = None):
    if 'verbose' in module_params:
        verbose = module_params['verbose']
    else:
        verbose = False

    if target_delta is not None and max_comps is None:
        print("Warning: target_delta is ignored, since max_comps is not given.")

    found_factor = False
    bounds = [(10**(-4), 1.0 + 10**(-2)), (10**(-3), 1.0 + 10**(-2)), (10**(-2), 1.0 + 10**(-2)), (10**(-2), 1.0 + 7*10**(-2)), (1.5 * 10**(-2), 1.0 + 8 * 10**(-2)), (1.7 * 10**(-1), 1.0 + 10**(-4)), (1.7 * 10**(-1), 1.0 + 10**(-3)), (1.7 * 10**(-1), 1.0 + 10**(-2)), (1.7 * 10**(-1), 1.0 + 1.8 * 10**(-1))]
    if max_comps is not None:
        bounds = [(" ",b) for b in [1.0 + 10**(-2), 1.0 + 5 * 10**(-2), 1.0 + 1.8 * 10**(-1)]]
    i = 0
    offset_list = []
    alternating_bound = 6 # how often the direction of the factor search has to change before the search stops, used to avoid divergence

    # initially compute bounds to determine the initial error
    offset = 10**(-5)
    module_params['factor'] = 1.0 + offset
    candidate_offset = offset
    error, infty_bucket, true_delta_value = compute_initial_bounds(module_params)
    candidate_error = error
    offset_list.append((offset, error, infty_bucket, true_delta_value))

    while(not found_factor and i < len(bounds)):
        offset_upwards = False # in which direction is the offset currently increased
        infty_bucket_ratio_bound = bounds[i][0] if max_comps is None\
                                 else (2 ** (- np.log2(max_comps) - 3) * 1.55 ** np.log2(max_comps))
                                     # efficient way of computing 0.1 * 1.5**max_comps/ 2.2**max_comps
        if target_delta is not None and max_comps is not None:
            infty_condition = lambda x, y: x * 2.25**np.log2(max_comps) <= target_delta
        else:
            infty_condition = lambda x, y: x/y <= infty_bucket_ratio_bound
        error_bound = bounds[i][1]
        error_condition = lambda x: x <= error_bound
        offset_condition = lambda x, y, z: error_condition(x) and infty_condition(y, z)
        if verbose:
            print("Trying to find a factor for")
            print("[*] BOUND for INFTY BUCKET RATIO:\t " + str(infty_bucket_ratio_bound))
            print("[*] BOUND for ERROR:\t\t\t " + str(error_bound))
        # check whether previous runs already had a suitable candidate. If so, start with that candidate.
        candidates = sorted([offset_data for offset_data in offset_list \
                    if offset_condition(offset_data[1], offset_data[2], offset_data[3])], key=itemgetter(1))
        alternating = alternating_bound
        if candidates != []:
            found_factor = True
            candidate_offset = candidates[0][0]
            candidate_error = candidates[0][1]

        rounds = 35
        while(alternating > 0 and rounds > 0):
            rounds -= 1
            # if the infty condition does not hold increase the factor and end this loop iteration
            if not infty_condition(infty_bucket, true_delta_value):
                offset = offset * 1.5
                alternating = (alternating - 1) if not offset_upwards else alternating
                if verbose:
                    print("infty_condition violated -> increasing the offset")
                offset_upwards = True
            # if the infty_condition holds but error condition does not hold, decrease the factor and end this loop iteration
            elif not error_condition(error):
                offset = offset/2.0
                alternating = (alternating - 1) if offset_upwards else alternating
                if verbose:
                    print("infty_condition satisfied but error_condition violated -> decreasing the offset")
                offset_upwards = False
            # if both conditions hold
            else:
                # the factor is within acceptable bounds
                found_factor = True
                # if the current error is worse than or equal to the previous error,
                if error < candidate_error:
                #     # try a larger factor
                #     offset = offset * 1.5
                #     alternating = alternating - 1 if not offset_upwards else alternating
                #     offset_upwards = True
                # else: # if the current error is better than the previous error
                    # choose this offset as a new candidate
                    candidate_offset = offset
                    candidate_error = error
                # try to find a tighter factor by decreasing the factor
                offset = offset/2.0
                alternating = alternating - 1 if offset_upwards else alternating
                offset_upwards = False
            module_params['factor'] = 1.0 + offset

            error, infty_bucket, true_delta_value = compute_initial_bounds(module_params)
            offset_list.append((offset, error, infty_bucket, true_delta_value))

        if verbose:
            print("alternating = " + str(alternating) + " and rounds = " + str(rounds))
        i = i + 1
        if not found_factor:
            print("No factor found for infty-bound = " + str(infty_bucket_ratio_bound) + " and error-bound = " + str(error_bound) + ".")
    if not found_factor:
        print(sorted([offset_data for offset_data in offset_list],key=itemgetter(1)))
        print("NO POSSIBLE FACTOR. ABORT.")
        sys.exit(1)
    # now offset_list is not empty
    if verbose:
        print(sorted([offset_data for offset_data in offset_list],key=itemgetter(1)))
    candidates = sorted([offset_data for offset_data in offset_list \
                    if offset_condition(offset_data[1], offset_data[2], offset_data[3])],key=itemgetter(1))
    assert(candidates != [])
    print("[*] Found factor " + str(1 + candidates[0][0]) \
            + "\n \t with error " + str(candidates[0][1]) \
            + "\n \t and infty ratio " + str(candidates[0][2]))
    # return the offset of the candidate with the lowest error and with acceptable infty
    return 1.0 + candidates[0][0]


if __name__ == '__main__':
    dist1 = np.linspace(0.1,0.2,100)
    dist1 /= np.sum(dist1)
    dist2 = np.linspace(0.2,0.1,100)
    dist2 /= np.sum(dist2)
    print(true_delta(dist1,dist2))
    print(optimal_factor(dist1, dist2, 1000000))