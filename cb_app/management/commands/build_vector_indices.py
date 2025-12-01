# cb_app/management/commands/build_vector_indices.py
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from cb_app.logic.index_manager import faiss_manager
from cb_app.models import Ticket, PDFChunk

class Command(BaseCommand):
    help = "Build in-memory FAISS indices from Postgres-stored embeddings. Usage: manage.py build_vector_indices [--names tickets,pdf] [--clear]"

    def add_arguments(self, parser):
        parser.add_argument(
            "--names",
            type=str,
            help="Comma-separated namespaces to build. Allowed values: tickets,pdf_chunks (default: both)."
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear existing indices for the selected namespaces before building."
        )

    def handle(self, *args, **options):
        names_opt = options.get("names")
        clear_flag = options.get("clear", False)

        all_names = {
            "tickets": {
                "fetch_fn": lambda: list(Ticket.objects.filter(embedding__isnull=False).values_list("id", "embedding")),
                "ns": "tickets"
            },
            "pdf_chunks": {
                "fetch_fn": lambda: list(PDFChunk.objects.filter(embedding__isnull=False).values_list("id", "embedding")),
                "ns": "pdf_chunks"
            }
        }

        selected = []
        if names_opt:
            requested = [n.strip() for n in names_opt.split(",") if n.strip()]
            for r in requested:
                if r not in all_names:
                    raise CommandError(f"Unknown namespace '{r}'. Allowed: {', '.join(all_names.keys())}")
                selected.append(r)
        else:
            selected = list(all_names.keys())

        self.stdout.write(self.style.NOTICE(f"Build vector indices for namespaces: {', '.join(selected)}"))
        for name in selected:
            info = all_names[name]
            ns = info["ns"]

            if clear_flag:
                if ns in faiss_manager.indices:
                    faiss_manager.indices.pop(ns, None)
                    self.stdout.write(self.style.SUCCESS(f"Cleared existing index for namespace '{ns}'"))

            # Build in a DB transaction read-only block to ensure consistency of reads
            try:
                with transaction.atomic():
                    fetch_fn = info["fetch_fn"]
                    items = fetch_fn()
                    count = len(items)
                    self.stdout.write(f"Found {count} rows with embeddings for namespace '{ns}'. Building index...")
                    # Use faiss_manager.build_from_db which accepts a fetch function that returns (id, embedding)
                    faiss_manager.build_from_db(ns, fetch_fn)
                    built = faiss_manager.get(ns).index.ntotal
                    self.stdout.write(self.style.SUCCESS(f"Built index for '{ns}' with {built} vectors."))
            except Exception as e:
                raise CommandError(f"Failed to build index for '{ns}': {e}")

        self.stdout.write(self.style.SUCCESS("All requested indices built successfully."))
