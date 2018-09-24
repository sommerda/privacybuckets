# written by David Sommer (david.sommer at inf.ethz.ch), and Esfandiar Mohammadi (mohammadi at inf.ethz.ch)

import sys
import io
import numpy as np
import base64
from collections import OrderedDict

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter

sys.path.insert(0, "../")
from distributions.noise_gaussian import Gaussian_Noise_1D_mechanism
from distributions.noise_laplace import Laplace_Noise_1D_mechanism
from distributions.noise_gaussian_mixture import Gaussian_Mixture_Noise_1D_mechanism
from core.probabilitybuckets_light import ProbabilityBuckets
# from bounds.renyi_privacy import reny_delta_of_eps_efficient
# from bounds.concentrated_dp import ConcentratedDP
from core.tools import delta_dist_events, get_sufficient_big_factor, true_delta

BACKGROUNDCOLOR = '#E6E6E8'


class PBError(ValueError):
    pass


def constructPB(module_params, n, endpoint = None):
    module_params = module_params.copy()

    # Construct numerical adp bounds

    module_params.update({
                'free_infty_budget': 10**(-20),
                'caching_directory': "./pb-cache",
                'error_correction': True,
    })

    if 'factor' not in module_params:
        module_params['factor'] = get_sufficient_big_factor(module_params, max_comps = n, target_delta = 3**np.log2(n) * delta_dist_events(module_params['dist1_array'],module_params['dist2_array']))

    pb = ProbabilityBuckets(**module_params)
    pbn = pb.compose(n)

    # Prepare the data for plots

    plots = {
        'pbup': {
            'name': "PB upper bound",
            'color': "brown",
            'linestyle': "solid"
            },
        'pblow': {
            'name': "PB lower bound",
            'color': "blue",
            'linestyle': "dashed"
            },
        'cdp': {
            'name': "CDP",
            'color': "green",
            'linestyle': "dotted"
            },
        'rdp': {
            'name': "RDP",
            'color': "red",
            'linestyle': "dashdot"
            }
    }

    central_moment = lambda p, mean, lam: np.log(module_params['factor']) * np.sum(np.multiply( np.power(np.arange(len(p)) - len(p)/2 - mean, lam), p))

    pbn_mean = central_moment(pbn.bucket_distribution, 0, 1)
    pbn_sigma = (central_moment(pbn.bucket_distribution, pbn_mean, 2))**(1/2.0)

    eps_granularity = 2000
    eps_vector = np.linspace(0, pbn_mean + 1 * pbn_sigma, eps_granularity, endpoint=True)

    # What are you doing here?
    if endpoint is None:
        plots['pbup']['ydata'] = [ pbn.delta_of_eps_upper_bound(eps) for eps in eps_vector ]

        xlast = 1
        count = 0
        for index in range(len(plots['pbup']['ydata'])):
            x = plots['pbup']['ydata'][index]
            if xlast/x < 1.005:
                count += 1
            else:
                count = 0
            if count >= eps_granularity//30 or eps_vector[index]>1:
                endpoint = eps_vector[index]
                break
            xlast = x

    eps_vector = np.linspace(0, endpoint, 100, endpoint=True)

    # PB
    plots['pbup']['ydata'] = np.asarray([pbn.delta_of_eps_upper_bound(eps) for eps in eps_vector])
    plots['pblow']['ydata'] = np.asarray([pbn.delta_of_eps_lower_bound(eps) for eps in eps_vector])

    # # Renyi-DP
    # plots['rdp']['ydata'] = reny_delta_of_eps_efficient(eps_vector, n, pb.bucket_distribution, pb.log_factor, None)

    return plots, eps_vector, pb, pbn


def convert_plots_to_hexstring(figures, eps_vector, titles, filetype='png'):
    plt.figure(figsize=(8,8))
    plt.title(titles[0])
    plt.subplots_adjust(hspace=0.6)
    for i, (title, plots) in enumerate(figures.items()):
        i += 1
        plt.subplot(3,1,i)
        plt.title(title)
        logymin = 10**100
        logymax = -200
        for index, plot in plots['dict'].items():
            if 'ydata' in plot:
                plt.semilogy(plot['xdata'], plot['ydata'], linestyle = plot['linestyle'],
                            color = plot['color'], alpha = 0.5, label = plot['name'])
                sanitizedy = plot['ydata'][np.nonzero(plot['ydata'])]
                logymin = min(logymin, np.log10(np.min(sanitizedy)))
                logymax = max(logymax, np.log10(np.max(sanitizedy)))
        yticks = np.logspace(logymin, logymax, num = 5)
        plt.yticks(yticks)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.2E}'))
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.xlabel(plots['x axis'])
        plt.ylabel(plots['y axis'])
        plt.legend()

    f = io.BytesIO()
    plt.savefig(f, format=filetype, bbox_inches='tight', dpi = 200, facecolor=BACKGROUNDCOLOR)

    f.seek(0)
    b64string = base64.b64encode(f.read())

    return 'data:image/' + filetype + ';base64, ' + b64string.decode('utf-8')


def construct_image(module_params, n, titles, dual = False):

    # Compute privacy buckets #

    pb_plot_data, eps_vector, pb, pbn = constructPB(module_params, n)

    # we need to compute the same with switched  dist_1 <-> dist_2 if the probelm is not symetric
    if dual:
        dist1_tmp = module_params['dist1_array']
        module_params['dist1_array'] = module_params['dist2_array']
        module_params['dist2_array'] = dist1_tmp
        pb_plot_data_dual, eps_vector, _, _ = constructPB(module_params, n, endpoint = eps_vector[-1])
        pb_plot_data['pbup']['ydata'] = np.maximum(pb_plot_data['pbup']['ydata'], pb_plot_data_dual['pbup']['ydata'])
        pb_plot_data['pblow']['ydata'] = np.maximum(pb_plot_data['pblow']['ydata'], pb_plot_data_dual['pblow']['ydata'])

    # Event space plots #

    real_plots = {}

    real_plots['dist1'] = {
            'ydata': module_params['dist1_array'],
            'xdata': range(len(module_params['dist1_array'])),
            'name': 'distribution 1',
            'color': 'blue',
            'linestyle': 'solid'
            }
    real_plots['dist2'] = {
            'ydata': module_params['dist2_array'],
            'xdata': range(len(module_params['dist2_array'])),
            'name': 'distribution 2',
            'color': 'brown',
            'linestyle': 'dashed'
            }

    # Privacy Loss distribution plots #

    pld_plots = {}

    pld_plots['initial'] = {
            'ydata': pb.bucket_distribution,
            'xdata': (np.arange(len(pb.bucket_distribution))- pb.one_index ) * pb.log_factor,
            'name': 'initial',
            'color': 'blue',
            'linestyle': 'solid',
    }

    pld_plots['final'] = {
            'ydata': pbn.bucket_distribution,
            'xdata': (np.arange(len(pbn.bucket_distribution))- pbn.one_index ) * pbn.log_factor,
            'name': 'after composition',
            'color': 'brown',
            'linestyle': 'dashed',
    }

    # ADP plots #

    adp_plots = {}

    adp_plots['pbup'] = {
        'xdata': eps_vector,
        'name': 'upper bound',
        'color': 'green',
        'linestyle': 'solid',
        'ydata': pb_plot_data['pbup']['ydata']
        }
    adp_plots['pblow'] = {
        'xdata': eps_vector,
        'name': 'upper bound',
        'color': 'green',
        'linestyle': 'dashed',
        'ydata': pb_plot_data['pblow']['ydata']
        }

    # if False:
    #     # zCDP
    #     mean_diff_cdp = mean_diff/np.float64(sigma)
    #     sigma_cdp = 1
    #     delta_dist_events_0 = delta_dist_events(mechs.y1, mechs.y2)
    #
    #     zcdp = ConcentratedDP.from_gaussians(mean_diff_cdp, sigma_cdp)
    #     assert np.log2(n) == int(np.log2(n))
    #     zcdp.self_compose_iterated_squaring(math.ceil(np.log2(n)))
    #     adp_plots['cdp']['ydata'] = [ include_dist_events(zcdp.delta_of_eps(eps), delta_dist_events_0, n)\
    #                  for eps in eps_vector ]

    # Defining the plot #

    figures = OrderedDict()

    figures['Event space'] = {
            'dict': real_plots,
            'x axis': 'event',
            'y axis': 'Pr[event]'
            }

    figures['Privacy loss distribution'] = {
            'dict': pld_plots,
            'x axis': 'ε',
            'y axis': 'Loss(ε)'
            }

    figures['ADP epsilon-delta graph'] = {
            'dict': adp_plots,
            'x axis': r'$\varepsilon$',
            'y axis': r'$\delta(\varepsilon)$'
            }

    image_1 = convert_plots_to_hexstring(figures, eps_vector, titles)
    return image_1


# #
# Different Histogram types
# #


def executeGeneric(mechanism, n, title, scale, number_of_buckets = 10**5, truncation_at = None):
    mean_diff = 1
    if truncation_at is None:
        truncation_at = int(scale * 10)

    gran = 10**(6)/truncation_at + 1
    mechs = mechanism(mean_diff=mean_diff, eps=1, scale=scale, truncation_at=truncation_at, granularity=gran)

    # some magic based on experience
    fac_gran = (number_of_buckets - 2)//40
    factor = np.exp(1/np.float64(fac_gran * scale))

    module_params = {
                'number_of_buckets': number_of_buckets,
                'dist1_array': mechs.y1,
                'dist2_array': mechs.y2,
                'factor': factor,
    }

    image_1 = construct_image(module_params, n, title)
    return image_1


def executeGaussian(scale, n, number_of_buckets = 10**5, truncation_at = None):
    title = [r'Gaussian mechanism with $\sigma = {}$ and {} compositions'.format(scale, n)]
    return executeGeneric(Gaussian_Noise_1D_mechanism, n, title=title, scale=scale, number_of_buckets=number_of_buckets, truncation_at=truncation_at)


def executeLaplace(scale, n, number_of_buckets = 10**5, truncation_at = None):
    title = [r'Laplace mechanism with $\lambda = {}$ and {} compositions'.format(scale, n)]
    return executeGeneric(Laplace_Noise_1D_mechanism, n, title=title, scale=scale, number_of_buckets=number_of_buckets, truncation_at=truncation_at)


def executeSGD(q, scale, n, number_of_buckets = 10**5, truncation_at = None):
    mechanism = lambda **kwargs: Gaussian_Mixture_Noise_1D_mechanism(q=q, **kwargs)
    title = [r'Stochastic-Gradient-Descent with $q = {}$, $\sigma = {}$, and ${} compostitions'.format(q, scale, n)]
    return executeGeneric(mechanism, n, title=title, scale=scale, number_of_buckets=number_of_buckets, truncation_at=truncation_at)


def executeHistogram(dist1, dist2, n, number_of_buckets = 10**5):
    dist1 = np.float64(dist1.splitlines())
    dist2 = np.float64(dist2.splitlines())
    if dist1.shape != dist2.shape:
        raise PBError("After parsing, the two csv files do not result in two distributions with the same support. Please ensure that both CSV files have the same amount of lines.")
    if False:
        dist1_tmp = np.convolve(np.convolve(dist1, dist2, mode="full"), dist1, mode="full")
        dist2_tmp = np.convolve(np.convolve(dist2, dist1, mode="full"), dist2, mode="full")
        dist1 = dist1_tmp
        dist2 = dist2_tmp

    dist1_tmp = dist1/np.sum(dist1)
    dist2_tmp = dist2/np.sum(dist2)
    print(true_delta(dist1_tmp,dist2_tmp))
    module_params = {
                'number_of_buckets': number_of_buckets,
                'dist1_array': dist1_tmp,
                'dist2_array': dist2_tmp,
    }

    image_1 = construct_image(module_params, n, 'histogram.png', ['Custom distribution pair after ' + str(n) + " compositions"], dual = True)
    return image_1

if __name__ == '__main__':
    PLTSHOW = True
    # executeLaplace(100, 2**4)
    # executeHistogram(np.linspace(0,100,10000),np.linspace(1,101,10000), 2**4)
    # a = (np.array([2,10,20,30,40,0,70,32,30,50], dtype = np.float64))
    # b = (np.array([0,11,20,32,40,2,70,30,30,50], dtype = np.float64))
    # a_full = np.convolve(np.convolve(a, b, mode="full"), a, mode="full")
    # b_full = np.convolve(np.convolve(b, a, mode="full"), b, mode="full")
    # a_full /= np.float64(np.sum(a_full))
    # b_full /= np.float64(np.sum(b_full))
    # file1 = 'new1.txt'
    # file2 = 'new2.txt'
    # with open(file1, 'r') as myfile:
    #     a_string = myfile.read()
    # # print(a_string)
    # with open(file2, 'r') as myfile:
    #     b_string = myfile.read()
    # print(b_string)
    # b_string = b'1\r\n123\r\n5\r\n12\r\n3\r\n4\r\n11\r\n32\r\n3\r\n3\r\n3\r\n5\r\n5\r\n4\r\n4\r\n4\r\n43\r\n34\r\n0\r\n34'
    # a_string = b'2\r\n110\r\n0\r\n7\r\n4\r\n4\r\n10\r\n34\r\n3\r\n3\r\n4\r\n4\r\n4\r\n4\r\n3\r\n4\r\n43\r\n34\r\n4\r\n3'
    try:
        # image = executeHistogram(a_string, b_string, 2**2)
        # image = executeGaussian(100, 2**4)
        image = executeGaussian(1, 2**5)

        if PLTSHOW:
            plt.switch_backend('qt')
        else:
            with open('tmp.base64', 'w') as myfile:
                myfile.write(image)
    except PBError as w:
        print("[*] CSV Error: "+str(w))
