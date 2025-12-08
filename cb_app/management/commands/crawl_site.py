# cb_app/management/commands/crawl_site.py
from django.core.management.base import BaseCommand
from cb_app.logic.crawler_logic import crawl_site


class Command(BaseCommand):
    help = "Crawl a website and store paragraphs + embeddings + FAISS vectors."

    def add_arguments(self, parser):
        parser.add_argument("start_url", type=str)
        parser.add_argument("--max-pages", type=int, default=20)
        parser.add_argument("--delay", type=float, default=0.5)

    def handle(self, *args, **opts):
        res = crawl_site(
            start_url=opts["start_url"],
            max_pages=opts["max_pages"],
            delay=opts["delay"]
        )

        self.stdout.write(self.style.SUCCESS(
            f"Crawled {res['pages_crawled']} pages, created {res['paragraphs_created']} paragraphs."
        ))

        if res["errors"]:
            self.stdout.write(self.style.WARNING("Errors:"))
            for e in res["errors"]:
                self.stdout.write(f" - {e}")
