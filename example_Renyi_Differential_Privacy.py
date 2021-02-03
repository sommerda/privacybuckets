# written by David Sommer (david.sommer at inf.ethz.ch)

import numpy as np
import scipy.stats as st
from core.probabilitybuckets_light import ProbabilityBuckets
from matplotlib import pyplot as plt

####
# Privacy Buckets is based on the following publication
####

# [*] S. Meiser, E. Mohammadi,
# [*] "Tight on Budget? Tight Bounds for r-Fold Approximate Differential Privacy",
# [*] Proceedings of the 25th ACM Conference on Computer and Communications Security (CCS), 2018

# and its Renyi Differential Privacy (RDP) implementation is based on

# [*] Sommer, David M., Sebastian Meiser, and Esfandiar Mohammadi.
# [*] "Privacy loss classes: The central limit theorem in differential privacy",
# [*] Proceedings on Privacy Enhancing Technologies 2019.2 (2019)


# We examine Gaussian additive Noise defined as follows
sensitivity = 1
sigma = 10

x = np.linspace(start=-15*sigma, stop=15*sigma, num=int(1e5))

# as the current implementation of RDP does not allow distinguishing events ("a delta"),
# the two input distributions differ slightly as we have to cut the tails differently.
distribution_A = st.norm.pdf(x, loc=0, scale=sigma)
distribution_B = st.norm.pdf(x, loc=sensitivity, scale=sigma)

# Important: the individual input distributions neede each to sum up to 1 exactly!
distribution_A = distribution_A/np.sum(distribution_A)
distribution_B = distribution_B/np.sum(distribution_B)

# For this kind of noise, we know the analytic expression of the Renyi-moments
RDP_Gaussian_analytic = lambda alpha, sensitivity, sigma: alpha * (sensitivity**2) / (2 * sigma**2)
# For details, see Lemma 2.4 in
# Bun et al. "Concentrated differential privacy: Simplifications, extensions, and lower bounds." Theory of Cryptography Conference.


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
# probability mass you want to tolerate in the inifity bucket. For this example, we set it magically to
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

# Print status summary
privacybuckets.print_state()


# Now we build the RDP(alpha) graphs from the computed distribution.
alpha_vec =  np.linspace(1, 100, 200)
upper_bound = [privacybuckets.renyi_divergence_upper_bound(alpha) for alpha in alpha_vec]
analytic = [RDP_Gaussian_analytic(alpha, sensitivity, sigma) for alpha in alpha_vec]

plt.plot(alpha_vec, upper_bound, label="RDP upper_bound", alpha=0.5)
plt.plot(alpha_vec, analytic, label="RDP analytic", alpha=0.5)
plt.legend()
plt.title("RDP Gaussian Noise with sensivity {} and sigma {}".format(sensitivity, sigma))
plt.xlabel("alpha")
plt.ylabel("RDP")
plt.ticklabel_format(useOffset=False)  # Hotfix for the behaviour of my current matplotlib version
plt.show()


# abusing internals, we can look at the bucket distribution
plt.plot(privacybuckets.bucket_distribution)
plt.title("bucket distribution")
plt.xlabel("bucket number")
plt.ylabel("mass")
plt.show()
