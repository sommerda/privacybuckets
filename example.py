import numpy as np
from core.probabilitybuckets_light import ProbabilityBuckets
from matplotlib import pyplot as plt 

####
# Privacy Buckets is based on the following publication
####

# [*] S. Meiser, E. Mohammadi, "Tight on Budget? Tight Bounds for r-Fold Approximate Differential Privacy", Proceedings of the 25th ACM Conference on Computer and Communications Security (CCS), 2018  -- to appear

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


# Initialize privacy buckets. 
privacybuckets = ProbabilityBuckets(
        number_of_buckets=100000,   # number of buckets. The more the better as the resolution gets more finegraind
        factor = 1 + 1e-4,   # depends on the number_of_buckets and is the multiplicative constant between two buckets.
        dist1_array = distribution_A,  # distribution A
        dist2_array = distribution_B,  # distribution B
        caching_directory = "./pb-cache",  # chaching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before squaring
        error_correction=True,  # error correction. See publication for details
        )


# Now we evaluate how the distributon looks after 2**k independent compositions
k = 13
privacybuckets_composed = privacybuckets.compose(2**k) # input can be arbitrary positive integer, but exponents of 2 are numerically the most stable  


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
