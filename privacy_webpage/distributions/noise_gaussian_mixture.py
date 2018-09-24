# Written by David Sommer (david.sommer at inf.ethz.ch)
import numpy as np
from scipy.stats import norm

try:
    from .noise_virtual import Virtual_Noise_1D, Virtual_Noise_1D_mechanism
except ModuleNotFoundError:
    from noise_virtual import Virtual_Noise_1D, Virtual_Noise_1D_mechanism


class Gaussian_Mixture_Noise_1D_mechanism(Virtual_Noise_1D_mechanism):
    """
    No distinguishing events!

    """
    def __init__(self, mean_diff, q, truncation_at=10, granularity=100, scale=1):
        super(Gaussian_Mixture_Noise_1D_mechanism, self).__init__(mean_diff, eps=0, truncation_at=truncation_at, granularity=granularity, scale=scale, target_distribution=norm)

        noise = Virtual_Noise_1D(eps=0, sensitivity=None, truncation_at = truncation_at + mean_diff, granularity=granularity, scale=scale, target_distribution=norm)

        shift = mean_diff * granularity

        self.y1 = np.zeros(noise.number_of_events - shift)
        self.y1 = noise.distribution[shift:noise.number_of_events - shift]

        self.y2 = np.zeros(noise.number_of_events - shift)
        self.y2 = q * noise.distribution[:noise.number_of_events - 2*shift] + (1-q) * self.y1

        self.x = np.linspace(-truncation_at, truncation_at + mean_diff, num=shift + noise.number_of_events, endpoint=True)


if __name__ == "__main__":
    truncation_at = 10
    granularity = 3

    # the exact delta(eps) graph for arbitrary number of compositions n
    mech = Gaussian_Mixture_Noise_1D_mechanism(mean_diff=1, scale=4)
    n = 100
    eps_vector = np.linspace(0, 2, 100)
    deltas = [ mech.delta_of_n(eps, n) for eps in eps_vector ]
    plt.plot(eps_vector, deltas)
    plt.show()
