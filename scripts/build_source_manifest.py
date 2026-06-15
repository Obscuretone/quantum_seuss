import argparse
import json
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


DEFAULT_DSE_LIBRARY = "https://www.drseussenterprises.com/dr-seuss-library"
DEFAULT_SEUSSVILLE_BOOKS = "https://www.seussville.com/books/"
DEFAULT_OVERDRIVE_SEARCH = "https://phoenix.overdrive.com/search/creatorId?query=356015&sortBy=newlyadded"


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.current_href = None
        self.current_text = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return

        attributes = dict(attrs)
        self.current_href = attributes.get("href")
        self.current_text = []

    def handle_data(self, data):
        if self.current_href:
            self.current_text.append(data)

    def handle_endtag(self, tag):
        if tag != "a" or not self.current_href:
            return

        title = " ".join(" ".join(self.current_text).split())
        if title:
            self.links.append({"title": title, "url": self.current_href})

        self.current_href = None
        self.current_text = []


def read_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "quantum-seuss-source-manifest/1.0"})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8", errors="ignore")


def collect_dse_sources(url):
    parser = LinkParser()
    parser.feed(read_url(url))
    sources = []
    seen = set()

    for link in parser.links:
        if "penguinrandomhouse.com/books/" not in link["url"]:
            continue

        key = (link["title"], link["url"])
        if key in seen:
            continue

        seen.add(key)
        sources.append(
            {
                "title": link["title"],
                "source_page": link["url"],
                "source_type": "official_publisher_page",
                "text_url": None,
                "status": "needs_authorized_text_export",
                "notes": "Official Dr. Seuss Enterprises library link to publisher page.",
            }
        )

    return sources


def build_manifest(args):
    manifest = {
        "notes": [
            "This manifest records official source/acquisition pages, not unauthorized full-text downloads.",
            "Fill text_url only with URLs or exports covered by your license.",
            "Run scripts/prepare_corpus.py --manifest licensed_urls.txt --clean after obtaining authorized text files.",
        ],
        "catalog_sources": [
            {
                "name": "Dr. Seuss Enterprises Library",
                "url": args.dse_library,
                "purpose": "Official title and publisher-link source.",
            },
            {
                "name": "Seussville Books",
                "url": args.seussville_books,
                "purpose": "Official consumer-facing catalog.",
            },
            {
                "name": "OverDrive Search",
                "url": args.overdrive_search,
                "purpose": "Library ebook availability reference; access may require library/vendor credentials.",
            },
        ],
        "sources": collect_dse_sources(args.dse_library),
    }

    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest['sources'])} source records to {args.out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build an official source manifest for licensed corpus work.")
    parser.add_argument("--dse-library", default=DEFAULT_DSE_LIBRARY)
    parser.add_argument("--seussville-books", default=DEFAULT_SEUSSVILLE_BOOKS)
    parser.add_argument("--overdrive-search", default=DEFAULT_OVERDRIVE_SEARCH)
    parser.add_argument("--out", type=Path, default=Path("sources.json"))
    return parser.parse_args()


def main():
    build_manifest(parse_args())


if __name__ == "__main__":
    main()
