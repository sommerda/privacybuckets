PLTSHOW = False
# written by David Sommer (david.sommer at inf.ethz.ch), and Esfandiar Mohammadi (mohammadi at inf.ethz.ch)
import sys
import os
import csv
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
if not PLTSHOW:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import base64
import binascii
sys.path.insert(0, "../")
from distributions.noise_gaussian import Gaussian_Noise_1D_mechanism
from distributions.noise_laplace import Laplace_Noise_1D_mechanism
from core.probabilitybuckets_light import ProbabilityBuckets
#from bounds.renyi_privacy import reny_delta_of_eps_efficient
from bounds.concentrated_dp import ConcentratedDP
from core.tools import delta_dist_events, include_dist_events, get_sufficient_big_factor, true_delta

def constructPB(module_params, n):
    plots = {\
        'pbup': \
        {'name' : "PB upper bound",\
        'color' : "brown",\
        'linestyle' : "solid"},\
        'pblow' : \
        {'name' : "PB lower bound",\
        'color' : "brown",\
        'linestyle' : "dotted"},\
        'cdp' : \
        {'name' : "CDP",\
        'color' : "blue",\
        'linestyle' : "dashed"},\
        'rdp': \
        {'name' : "RDP",\
        'color' : "red",\
        'linestyle' : "dashdot"}\
    }

    tmp_module_params = {
                'error_correction' : False,
                'free_infty_budget' : 10**(-20),
                'caching_directory' : "./pb-cache",
                'error_correction' : True,
    }
    tmp_module_params = dict(tmp_module_params, **module_params)

    if 'factor' not in tmp_module_params:
        # factor_offset = 0.00001
        # tmp_module_params2 = tmp_module_params
        # while True:
        #     tmp_module_params2['factor'] = factor_offset + 1
        #     tmppb = ProbabilityBuckets(**tmp_module_params2)
        #     if tmppb.infty_bucket <= 10**(-20):
        #         factor_offset *= 0.5
        #         print(factor_offset + 1)
        #         print(tmppb.infty_bucket)
        #         del tmppb
        #     else:
        #         break
        # tmp_module_params['factor'] = factor_offset + 1
        tmp_module_params['factor'] = get_sufficient_big_factor(tmp_module_params, max_comps = n, target_delta = 3**np.log2(n) * delta_dist_events(tmp_module_params['dist1_array'],tmp_module_params['dist2_array']))

    verbose = False
    if module_params['verbose']:
        print(tmp_module_params)

    pb = ProbabilityBuckets(**tmp_module_params)
    pbn = pb.compose(n)
    moment = lambda p, lam: np.log(tmp_module_params['factor']) * np.sum(np.multiply( np.power(np.arange(len(p)) - len(p)/2, lam), p))
    central_moment = lambda p, mean, lam: np.log(tmp_module_params['factor']) * np.sum(np.multiply( np.power(np.arange(len(p)) - len(p)/2 - mean, lam), p))
    pbn_mean = moment(pbn.bucket_distribution, 1)
    pbn_sigma = (central_moment(pbn.bucket_distribution, pbn_mean, 2))**(1/2.0)
    print(pbn_mean)
    print(pbn_sigma)
    granularity = 2000
    eps_vector = np.linspace(0, pbn_mean + 1 * pbn_sigma, granularity, endpoint=True)

    plots['pbup']['data'] = [pbn.delta_of_eps_upper_bound(eps) for eps in eps_vector]

    xlast = 1
    count = 0
    end_point = eps_vector[-1]
    for index in range(len(plots['pbup']['data'])):
        x = plots['pbup']['data'][index]
        if xlast/x < 1.05:
            count += 1
        else:
            count = 0
        if count >= granularity//30 or eps_vector[index]>1:
            end_point = eps_vector[index]
            break
        xlast = x

    eps_vector = np.linspace(0, end_point, 100, endpoint=True)


    # PB
    plots['pbup']['data'] = [pbn.delta_of_eps_upper_bound(eps) for eps in eps_vector]
    plots['pblow']['data'] = [pbn.delta_of_eps_lower_bound(eps) for eps in eps_vector]

    # # Renyi-DP
    # plots['rdp']['data'] = reny_delta_of_eps_efficient(eps_vector, n, pb.bucket_distribution, pb.log_factor, None)

    return plots, eps_vector


def convert_plots_to_hexstring(filename, plots, eps_vector, title):
    filetype = filename.split(".")[-1]
    filename = filename + str(np.random.randint(10000)) + "." + filetype
    for _, plot in plots.items():
        if 'data' in plot:
            plt.plot(eps_vector, plot['data'], linestyle = plot['linestyle'], color = plot['color'], alpha = 0.5, label = plot['name'])
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$\delta(\varepsilon)$')
    plt.savefig(filename, bbox_inches='tight', dpi = 200)
    if PLTSHOW:
        plt.show()
    else:
        plt.gcf().clear()

        with open(filename, 'rb') as f:
            b64string = base64.b64encode(f.read())
            #b64string = binascii.hexlify(f.read())

        os.remove(filename)

    return 'data:image/' + filetype + ';base64, ' + b64string.decode('utf-8')



def executeGaussian(sigma, n, number_of_buckets = 10**5, truncation_at = None):
    mean_diff = 1 # has to be an integer, as it is used for computing length of arrays in the implementation
    if truncation_at is None:
        truncation_at = sigma * 10
    gran = 10**(6)//truncation_at + 1 # granularity of the mechanism, has to be an integer, as it is used for computing length of arrays in the implementation
    mechs = Gaussian_Noise_1D_mechanism(mean_diff, eps=1, sigma=sigma, truncation_at=truncation_at, granularity=gran)

    granularity					= (number_of_buckets - 2)//40 # granularity of the factor
    factor 						= np.exp(1/np.float64(granularity * sigma))

    module_params = {
                'number_of_buckets' : number_of_buckets,
                'dist1_array' : mechs.y1,
                'dist2_array' : mechs.y2,
                'factor' : factor,
                'verbose' : False,
    }

    plots, eps_vector = constructPB(module_params, n)

    mean_diff_cdp = mean_diff/np.float64(sigma)
    sigma_cdp = 1
    delta_dist_events_0 = delta_dist_events(mechs.y1, mechs.y2)

    # zCDP
    zcdp = ConcentratedDP.from_gaussians(mean_diff_cdp, sigma_cdp)
    assert np.log2(n) == int(np.log2(n))
    zcdp.self_compose_iterated_squaring(int(np.log2(n)))
    plots['cdp']['data'] = [ include_dist_events(zcdp.delta_of_eps(eps), delta_dist_events_0, n)\
                 for eps in eps_vector ]

    image_1 = convert_plots_to_hexstring('gauss.png', plots, eps_vector, 'Gaussian mechanism with ' + r'$\sigma = $' + str(sigma) + " and " + str(n) + " compositions")
    return image_1

def executeLaplace(sigma, n, number_of_buckets = 10**5, truncation_at = None):
    mean_diff = 1 # has to be an integer, as it is used for computing length of arrays in the implementation
    if truncation_at is None:
        truncation_at = sigma * 10
    gran = 10**(6)//truncation_at + 1 # granularity of the mechanism, has to be an integer, as it is used for computing length of arrays in the implementation
    mechs = Laplace_Noise_1D_mechanism(mean_diff, eps=1, scale=sigma, truncation_at=truncation_at, granularity=gran)

    granularity					= (number_of_buckets - 2)//40 # granularity of the factor
    factor 						= np.exp(1/np.float64(granularity * sigma))

    module_params = {
                'number_of_buckets' : number_of_buckets,
                'dist1_array' : mechs.y1,
                'dist2_array' : mechs.y2,
                'factor' : factor,
                'verbose' : False,
    }

    plots, eps_vector = constructPB(module_params, n)

    # mean_diff_cdp = mean_diff/np.float64(sigma)
    # sigma_cdp = 1
    # delta_dist_events_0 = delta_dist_events(mechs.y1, mechs.y2)
    #
    # # zCDP
    # zcdp = ConcentratedDP.from_gaussians(mean_diff, sigma)
    # assert np.log2(n) == int(np.log2(n))
    # zcdp.self_compose_iterated_squaring(int(np.log2(n)))
    # plots['cdp']['data'] = [ include_dist_events(zcdp.delta_of_eps(eps), delta_dist_events_0, n)\
    #              for eps in eps_vector ]

    image_1 = convert_plots_to_hexstring('laplace.png', plots, eps_vector, 'Laplace mechanism with ' + r'$\lambda = $' + str(sigma) + " and " + str(n) + " compositions")
    return image_1

def executeHistogram(dist1, dist2, n, number_of_buckets = 10**5):
    dist1_tmp /= np.float(dist1)
    dist2_tmp /= np.float(dist2)
    print(true_delta(dist1_tmp,dist2_tmp))
    module_params = {
                'number_of_buckets' : number_of_buckets,
                'dist1_array' : dist1_tmp,
                'dist2_array' : dist2_tmp,
                'verbose' : False,
    }

    plots, eps_vector = constructPB(module_params, n)

    image_1 = convert_plots_to_hexstring('histogram.png', plots, eps_vector, 'Custom distribution pair after ' + str(n) + " compositions")
    return image_1

if __name__ == '__main__':
    # executeGaussian(100, 2**4)
    # executeLaplace(100, 2**4)
    # executeHistogram(np.linspace(0,100,10000),np.linspace(1,101,10000), 2**4)
    a = (np.array([2,10,20,30,40,0,70,32,30,50], dtype = np.float64))
    b = (np.array([0,11,20,32,40,2,70,30,30,50], dtype = np.float64))
    a_full = np.convolve(np.convolve(a, b, mode="full"), a, mode="full")
    b_full = np.convolve(np.convolve(b, a, mode="full"), b, mode="full")
    a_full /= np.float64(np.sum(a_full))
    b_full /= np.float64(np.sum(b_full))
    executeHistogram(a_full, b_full, 2**8)
