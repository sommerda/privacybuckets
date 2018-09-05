from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect
from django.shortcuts import render

import sys
sys.path.insert(0, "../")
from interfaces import executeGaussian, executeLaplace, executeHistogram


@csrf_protect
def main_site(request):
    return render(request, "index.html", {})


def result(request):
    if request.method == 'POST':
        print(request.POST)
        try:
            da_type = request.POST['type']
            if da_type == 'Gaussian':
                n_gaussian = int(request.POST['n_gauss'])
                sigma = int(request.POST['sigma'])
                print("Gaussian",sigma, n_gaussian)
                image_1 = executeGaussian(sigma, n_gaussian)
            elif da_type == 'Laplace':
                n_lap = int(request.POST['n_lap'])
                scale = int(request.POST['scale'])
                print("laplace",scale, n_lap)
                image_1 = executeLaplace(scale, n_lap)
            elif da_type == 'Custom':
                print("custom")
                distr_1 = request.FILES['distr_1'].read()
                distr_2 = request.FILES['distr_2'].read()
                n_cust = int(request.POST['n_cust'])
                print("custom", distr_1, distr_2)
                image_1 = executeHistogram(distr_1, distr_2, n_cust)
            else:
                return HttpResponse('Invalid type!')
        except:
            return HttpResponse('A Error happened')

        ret_dict = {'image_1': image_1}
        return render(request, "result.html", ret_dict)
    return HttpResponse('No data received. Use the index page to submit.')
