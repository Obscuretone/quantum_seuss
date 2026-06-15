import argparse
import html
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf", ".epub"}
SKIPPED_FILENAMES = {"readme.md", "readme.txt"}


def slugify(value):
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "document"


def normalize_text(text):
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def strip_html(text):
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return normalize_text(text)


def extract_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise SystemExit(
            "error: PDF extraction requires pypdf. Run `pip install -r requirements.txt`."
        ) from error

    reader = PdfReader(path)
    return normalize_text("\n\n".join(page.extract_text() or "" for page in reader.pages))


def extract_epub(path):
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:
        raise SystemExit(
            "error: EPUB extraction requires beautifulsoup4. Run `pip install -r requirements.txt`."
        ) from error

    chunks = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.lower().endswith((".xhtml", ".html", ".htm")):
                document = archive.read(name).decode("utf-8", errors="ignore")
                soup = BeautifulSoup(document, "html.parser")
                chunks.append(soup.get_text(" "))

    return normalize_text("\n\n".join(chunks))


def extract_text(path):
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix in {".html", ".htm"}:
        return strip_html(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".epub":
        return extract_epub(path)

    raise ValueError(f"unsupported file type: {path}")


def iter_input_files(path):
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        return

    for file_path in sorted(path.rglob("*")):
        if file_path.name.lower() in SKIPPED_FILENAMES:
            continue
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file_path


def download_manifest(manifest, download_dir):
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for line in manifest.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue

        parsed = urllib.parse.urlparse(url)
        filename = Path(urllib.parse.unquote(parsed.path)).name
        if not filename:
            filename = slugify(parsed.netloc)

        target = download_dir / filename
        with urllib.request.urlopen(url) as response:
            target.write_bytes(response.read())
        downloaded.append(target)
        print(f"downloaded {url} -> {target}")

    return downloaded


def prepare_corpus(inputs, output_dir, clean):
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for input_path in inputs:
        for source in iter_input_files(input_path):
            text = extract_text(source)
            if not text.strip():
                print(f"skipped empty text from {source}")
                continue

            target = output_dir / f"{slugify(source.stem)}.txt"
            target.write_text(text, encoding="utf-8")
            written.append(target)
            print(f"wrote {target}")

    if not written:
        raise SystemExit("error: no supported corpus files were found")

    print(f"prepared {len(written)} text files in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare licensed/local source files for LLM training."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("corpus_raw")],
        help="Files or directories containing .txt, .md, .html, .pdf, or .epub sources.",
    )
    parser.add_argument("--manifest", type=Path, help="Optional newline-delimited URL list.")
    parser.add_argument("--download-dir", type=Path, default=Path("corpus_raw/downloads"))
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    parser.add_argument("--clean", action="store_true", help="Replace the output directory first.")
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = list(args.inputs)

    if args.manifest:
        inputs.extend(download_manifest(args.manifest, args.download_dir))

    prepare_corpus(inputs, args.out, args.clean)


if __name__ == "__main__":
    main()
