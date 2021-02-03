# written by David Sommer (david.sommer at inf.ethz.ch in 2021)

####
# Privacy Buckets is based on the following publication
####

# [*] S. Meiser, E. Mohammadi,
# [*] "Tight on Budget? Tight Bounds for r-Fold Approximate Differential Privacy",
# [*] Proceedings of the 25th ACM Conference on Computer and Communications Security (CCS), 2018

# and its ADP / PDP bucket construction implementation is based on

# [*] Sommer, David M., Sebastian Meiser, and Esfandiar Mohammadi.
# [*] "Privacy loss classes: The central limit theorem in differential privacy",
# [*] Proceedings on Privacy Enhancing Technologies 2019.2 (2019)

import numpy as np
import scipy.stats as st

from matplotlib import pyplot as plt

from core.probabilitybuckets_light import ProbabilityBuckets, ProbabilityBuckets_fromDelta

# which differntial privacy type to use
DP_TYPE = 'adp'
# DP_TYPE = 'pdp'

# We examine Gaussian additive Noise defined as follows
sensitivity = 1
sigma = 5

truncation_width = 5 * sigma
events = int(1e5)

# this is a helper variable
noise_events = int(events * ( (2 * truncation_width - sensitivity) / (2 * truncation_width + sensitivity)))

x = np.linspace(start=-truncation_width, stop=truncation_width, num=noise_events)

noise = st.norm.pdf(x, loc=0, scale=sigma)

distribution_A = np.zeros(events)
distribution_B = np.zeros(events)

distribution_A[:noise_events] = noise
distribution_B[-noise_events:] = noise

# Important: the individual input distributions neede each to sum up to 1 exactly!
distribution_A = distribution_A / np.sum(distribution_A)
distribution_B = distribution_B / np.sum(distribution_B)

### Choosing the number of buckets.
#
# We distribute the elements of distribution_A in the buckets according to the privacy loss L_A/B.
# Higher number-of_buckets allows more finegrade composition. The runtime if composition is quadratic in the number of buckets.
# A good value to start this example with with is
number_of_buckets = 2500
# It is more beneficial to adapt the factor (see below) than the number-of-buckets.
# Once you understand its influcences on the accuracy, change it to any value you like and your hardware supports.


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
# probability mass you want to tolerate in the inifity bucket. For this example, we set it magically to
factor = 1 + 1e-3

# Initialize privacy buckets.
kwargs = {'number_of_buckets': number_of_buckets,
          'factor': factor,
          'caching_directory': "./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
          'free_infty_budget': 10**(-20),  # how much we can put in the infty bucket before first squaring
          'error_correction': False,  # True is not supported.
          }

pb = ProbabilityBuckets(dist1_array=distribution_A,  # distribution A
                        dist2_array=distribution_B,  # distribution B
                        **kwargs)

# Print status summary
pb.print_state()

# the delta we are going to reconstruct the privacy loss distribution from.
if DP_TYPE == 'adp':
    delta_func = pb.delta_ADP
if DP_TYPE == 'pdp':
    delta_func = pb.delta_PDP

print("[*] reconstructing..")

pb_reconstructed = ProbabilityBuckets_fromDelta(delta_func=delta_func, DP_type=DP_TYPE, **kwargs)

pb_reconstructed.print_state()


# compare ADP deltas:
eps = (np.arange(pb.number_of_buckets) - pb.number_of_buckets // 2 ) * pb.log_factor  # epsilons of individual buckets
plt.plot(eps, [ delta_func(e) for e in eps ], label='original delta')
if DP_TYPE == 'adp':
    plt.plot(eps, [ pb_reconstructed.delta_ADP(e) for e in eps ], linestyle=':', label='reconstructed delta')
if DP_TYPE == 'pdp':
    plt.plot(eps, [ pb_reconstructed.delta_PDP(e) for e in eps ], linestyle=':', label='reconstructed delta')
plt.ylabel("delta(epsilon)")
plt.xlabel("epsilon")
plt.legend()
plt.show()

# NOTE: the reconstructed bucket distribution might contain numerical errrors while still leading to (almost) the same
#       delta curve!
plt.semilogy(eps, pb.bucket_distribution, alpha=0.5, label='orignial distribution')
plt.semilogy(eps, pb_reconstructed.bucket_distribution, alpha=0.5, label='reconstructed distribution (with (almost) identical delta curve)')
plt.ylabel("bucket content [probability mass]")
plt.xlabel("privacy loss corresponding to bucket")
plt.legend()
plt.show()
