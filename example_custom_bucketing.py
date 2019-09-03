# written by David Sommer (david.sommer at inf.ethz.ch) in 2019

import numpy as np
import scipy.stats
from core.probabilitybuckets_light import ProbabilityBuckets
from matplotlib import pyplot as plt

####
# Privacy Buckets is based on the following publication
####

# [*] S. Meiser, E. Mohammadi,
# [*] "Tight on Budget? Tight Bounds for r-Fold Approximate Differential Privacy",
# [*] Proceedings of the 25th ACM Conference on Computer and Communications Security (CCS), 2018


# In this example, we implement custom bucketing for binomial distribution (k, n, p=0.5), shifted by one draw:
#   k_vector = np.arange(n+1)
#   distribution_A = np.zeros(n+2)
#   distribution_B = np.zeros(n+2)
#   distribution_A[:-1] = scipy.stats.binom.pmf(k_vector,n,p=0.5)
#   distribution_B[1:] = scipy.stats.binom.pmf(k_vector,n,p=0.5)
#
# We compute the optimal privacy loss for an event k as
#        L_A/B(k) = log [ (n over k) * 0.5**k * 0.5**(n-k) ] / [ (n over k-1) * 0.5**(k-1) * 0.5**(n-(k-1)) ]
#                 = log [ (n - k + 1) / k ]
#
# The largest privacy loss is L_A/B(n) = log(n) [ and L_A/B(0) = -log(n) ]. Plug this in the condition for the factor
#
#       L_A/B(n) = ln(n) < log(factor) * number_of_buckets / 2
#
# gives
#       factor > exp( log(n) * 2 / number_of_buckets )



class ProbabilityBuckets_Binom(ProbabilityBuckets):
    def __init__(self, n, **kwargs):

        self.n = np.int32(n)

        # we want to fit all possible loss values in our bucket_distribution.
        # Using the derivation above, we overapproximate the factor with n -> n+1 to avoid numerical issues
        # ( L_A/B(n) might fall beyond the last bucket)
        kwargs['factor'] = np.exp(np.log(n+1) * 2 / kwargs['number_of_buckets']  )

        # Tell the parent __init__() method that we create our own bucket_distribution.
        kwargs['skip_bucketing'] = True

        # Our custom create_bucket_distribution() method does not set up the error correction
        kwargs['error_correction'] = False

        super(ProbabilityBuckets_Binom, self).__init__(**kwargs)

        self.create_bucket_distribution()

        # Cacheing setup needs to be called after the buckets have been filled as the caching utilized a hash over the bucket distribution
        self.cacheing_setup()

    def create_bucket_distribution(self):
        self.bucket_distribution = np.zeros(self.number_of_buckets, dtype=np.float64)
        self.error_correction = False

        # k = 0 gives distringuishing event
        k = np.arange(1, self.n + 1)
        privacy_loss_AB = np.log( (self.n - k + 1) / k )
        indices = privacy_loss_AB / self.log_factor + self.number_of_buckets // 2
        indices = np.ceil(indices).astype(int)

        distr1 = scipy.stats.binom.pmf(k, self.n, p=0.5)

         # fill buckets
        for i,  a, in zip(indices, distr1 ):
            # i = int(np.ceil(i))
            if i >= self.number_of_buckets:
                assert False  # should not happen in this implementation
                self.infty_bucket += a
                continue
            if i < 0:
                assert False  # should not happen in this implementation
                self.bucket_distribution[0] += a
                continue
            self.bucket_distribution[i] += a

        # the infinity-bucket is zero by design
        self.infty_bucket = np.float64(0.0)

        # k = 0 gives distringuishing event, Pr[k = 0] = 0.5**(n-0)*0.5**0 = 0.5**n
        self.distinguishing_events = np.float64(0.5)**self.n

        self.one_index = int(self.number_of_buckets // 2)   # this is a shortcut to the 0-bucket where L_A/B ~ 1
        self.u = np.int64(1)   # for error correction. Actually not needd


n = 5000  # the number of draws in a binomial distribution

# Initialize privacy buckets.
privacybuckets = ProbabilityBuckets_Binom(
        n = n,  # The n of the binominal distribution
        number_of_buckets=100000,   # number of buckets. The more the better as the resolution gets more finegraind. But you pay with performance
        factor = None,   # We compute the optimal one in our custom implementation
        caching_directory = "./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before first squaring
        error_correction=None,  # error correction. We set it to False in our custom implementation internally
        )


privacybuckets.print_state()


#
# analytical privacy loss distribution, see 'Sommer et al.  "Privacy loss classes: The central limit theorem in differential privacy." Proceedings on Privacy Enhancing Technologies. 2019'
#

# the k's for which we do not hit distinguishing events, i.e., where distribution_B != 0
k_vec = np.arange(1, n + 1)
privacy_loss_AB = np.log((n - k_vec + 1) / k_vec)

distribution_A = scipy.stats.binom.pmf(k_vec, n, p=0.5)

plt.plot(privacy_loss_AB, distribution_A, label='analytic solution')
plt.plot( ( np.arange(privacybuckets.number_of_buckets) - privacybuckets.one_index) * privacybuckets.log_factor, privacybuckets.bucket_distribution, label='numeric solution')
plt.xlabel("privacy loss")
plt.ylabel("probability mass")
plt.legend()

print("distinguishing events (containing only the probability mass from k=0): {} (which should be 0.5**{} but might be 0 due to numerical precision".format(str(scipy.stats.binom.pmf(0,n,p=0.5)), n))

plt.show()


#
# Composition
#

# Now we evaluate how the distributon looks after 2**k independent compositions
k = 5
# input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
privacybuckets_composed = privacybuckets.compose(2**k)

# Print status summary
privacybuckets_composed.print_state()


# Now we build the delta(eps) graphs from the computed distribution.
eps_vector =  np.linspace(0,3,100)
upper_bound = [privacybuckets_composed.delta_of_eps_upper_bound(eps) for eps in eps_vector]

plt.plot(eps_vector, upper_bound, label="upper_bound")
plt.legend()
plt.title("Binomial(n={},p=0.5) distribution after {:d} compositions".format(n, 2**k))
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
