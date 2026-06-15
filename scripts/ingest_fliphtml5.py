import argparse
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image


def normalize_book_url(url):
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    if not path.endswith("/"):
        path += "/"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def slug_from_url(url):
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    return "-".join(parts[-2:]) if len(parts) >= 2 else "fliphtml5-book"


def request_url(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": "quantum-seuss-fliphtml5-ingest/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_page_image(base_url, page_number, timeout):
    image_url = f"{base_url}files/large/{page_number}.webp"
    try:
        return request_url(image_url, timeout)
    except Exception:
        return None


def ocr_image(image_bytes, language):
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "page.png"
        output_base = Path(temp_dir) / "page"

        with tempfile.NamedTemporaryFile(suffix=".webp") as source:
            source.write(image_bytes)
            source.flush()
            Image.open(source.name).save(image_path)

        subprocess.run(
            ["tesseract", str(image_path), str(output_base), "-l", language],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return output_base.with_suffix(".txt").read_text(encoding="utf-8", errors="ignore")


def iter_urls(args):
    if args.url:
        yield args.url

    if args.manifest:
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            url = line.strip()
            if url and not url.startswith("#"):
                yield url


def ingest_url(args, url):
    base_url = normalize_book_url(url)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.name or slug_from_url(base_url)}.txt"
    if args.clean and target.exists():
        target.unlink()

    image_dir = args.images_dir / slug_from_url(base_url)
    if args.keep_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    misses = 0
    page_number = args.start_page
    extracted_pages = 0

    while page_number <= args.max_pages and misses < args.max_misses:
        image_bytes = download_page_image(base_url, page_number, args.timeout)
        if image_bytes is None:
            misses += 1
            page_number += 1
            continue

        misses = 0
        if args.keep_images:
            (image_dir / f"{page_number:04d}.webp").write_bytes(image_bytes)

        text = ocr_image(image_bytes, args.language).strip()
        if text:
            prefix = f"\n\n[page {page_number}]\n" if args.page_markers else "\n\n"
            with target.open("a", encoding="utf-8") as output:
                output.write(f"{prefix}{text}")
            extracted_pages += 1

        print(f"ocr page {page_number}: {len(text.split())} words", flush=True)
        page_number += 1

    if not target.exists():
        raise SystemExit("error: no OCR text extracted")

    target.write_text(target.read_text(encoding="utf-8").strip() + "\n", encoding="utf-8")
    print(f"wrote {target} from {extracted_pages} pages", flush=True)


def ingest(args):
    urls = list(iter_urls(args))
    if not urls:
        raise SystemExit("error: provide a FlipHTML5 URL or --manifest")
    if args.name and len(urls) > 1:
        raise SystemExit("error: --name can only be used with a single URL")

    for url in urls:
        ingest_url(args, url)


def parse_args():
    parser = argparse.ArgumentParser(description="OCR an authorized FlipHTML5 book into corpus text.")
    parser.add_argument("url", nargs="?", help="FlipHTML5 book URL, e.g. https://online.fliphtml5.com/piaso/fncp/")
    parser.add_argument("--manifest", type=Path, help="Newline-delimited list of authorized FlipHTML5 URLs.")
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    parser.add_argument("--name", help="Output basename without .txt")
    parser.add_argument("--images-dir", type=Path, default=Path("corpus_raw/fliphtml5_images"))
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--max-misses", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--clean", action="store_true", help="Replace the output file first.")
    parser.add_argument("--page-markers", action="store_true", help="Include [page N] markers in output.")
    return parser.parse_args()


def main():
    ingest(parse_args())


if __name__ == "__main__":
    main()
