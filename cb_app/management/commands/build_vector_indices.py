# cb_app/management/commands/build_vector_indices.py

from django.core.management.base import BaseCommand
from cb_app.models import Ticket, PDFChunk
from cb_app.logic.index_manager import faiss_manager

class Command(BaseCommand):
    help = "Rebuild FAISS vector indices for tickets and pdf chunks (safe 768-d version)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--names",
            type=str,
            default="tickets,pdf_chunks",
            help="Comma-separated namespaces to rebuild (default: tickets,pdf_chunks)",
        )

    def handle(self, *args, **options):
        namespaces = [n.strip() for n in options["names"].split(",")]

        for ns in namespaces:
            self.stdout.write(f"\nRebuilding namespace: {ns}")

            if ns == "tickets":
                items = list(
                    Ticket.objects.filter(embedding__isnull=False)
                    .values_list("id", "embedding")
                )
            elif ns == "pdf_chunks":
                items = list(
                    PDFChunk.objects.filter(embedding__isnull=False)
                    .values_list("id", "embedding")
                )
            else:
                self.stdout.write(self.style.ERROR(f"Unknown namespace: {ns}"))
                continue

            # Clear old index safely
            faiss_manager.safe_pop(ns)

            # Rebuild safely
            faiss_manager.safe_build_from_db_if_empty(ns, lambda: items)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Successfully rebuilt FAISS index for namespace '{ns}' "
                    f"with {len(items)} vectors."
                )
            )
