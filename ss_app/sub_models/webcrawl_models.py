# ss_app/sub_models/webcrawl_models.py
from django.db import models
from django.contrib.postgres.fields import ArrayField


class Page(models.Model):
    url = models.URLField(unique=True)
    title = models.CharField(max_length=500)

    class Meta:
        db_table = "web_pages"

    def __str__(self):
        return self.title


class Paragraph(models.Model):
    page = models.ForeignKey(Page, on_delete=models.CASCADE, related_name="paragraphs")
    text = models.TextField()
    order = models.IntegerField()

    # store 768-dim embedding from your global embedder
    embedding = ArrayField(models.FloatField(), size=768, null=True, blank=True)

    class Meta:
        db_table = "web_paragraphs"
        ordering = ("page_id", "order")

    def __str__(self):
        return f"{self.page.title} [{self.order}]"
