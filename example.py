# written by David Sommer (david.sommer at inf.ethz.ch)

import numpy as np
from core.probabilitybuckets_light import ProbabilityBuckets
from matplotlib import pyplot as plt

####
# Privacy Buckets is based on the following publication
####

# [*] S. Meiser, E. Mohammadi,
# [*] "Tight on Budget? Tight Bounds for r-Fold Approximate Differential Privacy",
# [*] Proceedings of the 25th ACM Conference on Computer and Communications Security (CCS), 2018

# We examine Extended Randomized Repsonse distributions defined as follows
delta=0.0001
eps_rr = 0.01
expeps = np.exp(eps_rr)
norm = (1. + expeps + delta)

distribution_A = np.array([0, 1./norm, expeps/norm, delta])
distribution_B = np.array([delta, expeps/norm, 1./norm, 0])

# Important: the individual input distributions neede each to sum up to 1 exactly!
distribution_A = distribution_A/np.sum(distribution_A)
distribution_B = distribution_B/np.sum(distribution_B)


### Choosing the number of buckets.
#
# We distribute the elements of distribution_A in the buckets according to the privacy loss L_A/B.
# Higher number-of_buckets allows more finegrade composition. The runtime if composition is quadratic in the number of buckets.
# A good value to start with is
number_of_buckets = 100000
# It is more beneficial to adapt the factor (see below) than the number-of-buckets.
# Once you understand how it influcences the accuracy, change it to any value you like and your hardware supports.


### Choosing the factor
#
# The factor is the average additive distance between the privacy-losses that are put in neighbouring buckets.
# You want to put most of you probablity mass of distribution_A in the buckets, as the rest gets put into the
# infitity_bucket and the minus_n-bucket. Therefore for an o
#
#       L_A/B(o) = log(factor) * number_of_buckets / 2
#
# then the mass Pr[ L_A/B(o) > mass_infinity_bucket, o <- distribution_A] will be put into the infinity bucket.
# The infinity-bucket gros exponentially with the number of compositions. Chose the factor according to the
# probability mass you want to tolerate in the inifity bucket. For this example, the minimal factor should be
#
#       log(factor) > eps
#
# as for randomized response, there is no privacy loss L_A/B greater than epsilon (excluding delta/infinity-bucket).
# We set the factor to
factor = 1 + 1e-4


# Initialize privacy buckets.
privacybuckets = ProbabilityBuckets(
        number_of_buckets = number_of_buckets,
        factor = factor,
        dist1_array = distribution_A,  # distribution A
        dist2_array = distribution_B,  # distribution B
        caching_directory = "./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
        )

# Now we evaluate how the distributon looks after 2**k independent compositions
k = 13
# input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
privacybuckets_composed = privacybuckets.compose(2**k)

# Print status summary
privacybuckets_composed.print_state()


# Now we build the delta(eps) graphs from the computed distribution.
eps_vector =  np.linspace(0,3,100)
upper_bound = [privacybuckets_composed.delta_of_eps_upper_bound(eps) for eps in eps_vector]
lower_bound = [privacybuckets_composed.delta_of_eps_lower_bound(eps) for eps in eps_vector]

plt.plot(eps_vector, upper_bound, label="upper_bound")
plt.plot(eps_vector, lower_bound, label="lower_bound")
plt.legend()
plt.title("Extended Randomized response with eps={:e}, delta={:f} after {:d} compositions".format(eps_rr, delta, 2**k))
plt.xlabel("eps")
plt.ylabel("delta")
plt.ticklabel_format(useOffset=False)  # Hotfix for the behaviour of my current matplotlib version
plt.show()


# abusing internals, we can look at the bucket distribution
plt.plot(privacybuckets_composed.bucket_distribution)
plt.title("bucket distribution")
plt.xlabel("bucket number")
plt.ylabel("mass")
plt.show()
