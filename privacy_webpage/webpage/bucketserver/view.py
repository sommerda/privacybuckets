from django.views.decorators.csrf import csrf_protect
from django.shortcuts import render


@csrf_protect
def main_site(request):
    return render(request, "index.html", {})