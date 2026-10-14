import pymupdf
from pathlib import Path


def parse_pdf(file_path):
    """
    Extract text and metadata from a PDF document.
    """

    document = pymupdf.open(file_path)

    parsed_blocks = []

    for page_number, page in enumerate(document, start=1):

        blocks = page.get_text("blocks")

        for block in blocks:

            text = block[4].strip()

            if not text:
                continue

            parsed_blocks.append({
                "text": text,
                "page": page_number,
                "source": Path(file_path).name,
                "content_type": "text"
            })

    document.close()

    return parsed_blocks


def validate_parsing(parsed_blocks):
    """
    Check whether useful text was extracted.
    """

    if not parsed_blocks:
        return {
            "success": False,
            "message": "No text extracted from the PDF."
        }

    return {
        "success": True,
        "message": "PDF parsing completed successfully.",
        "total_blocks": len(parsed_blocks)
    }