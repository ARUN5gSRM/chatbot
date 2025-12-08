# cb_app/sub_views/crawl_view.py
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from cb_app.logic.crawler_logic import crawl_site

@login_required
@require_http_methods(["GET", "POST"])
def crawl_site_view(request):
    result = None

    if request.method == "POST":
        url = request.POST.get("url", "").strip()
        max_pages = int(request.POST.get("max_pages", 5))
        delay = float(request.POST.get("delay", 0.5))

        if url:
            result = crawl_site(url, max_pages=max_pages, delay=delay)

    return render(request, "cb_app/crawl_site.html", {
        "result": result,
    })
