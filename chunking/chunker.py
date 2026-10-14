from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def fixed_chunking(parsed_blocks):
    """
    Split PDF text into fixed-size chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = []

    for block in parsed_blocks:
        text_chunks = splitter.split_text(block["text"])

        for chunk in text_chunks:
            chunks.append({
                "text": chunk,
                "page": block["page"],
                "source": block["source"],
                "content_type": block["content_type"]
            })

    return chunks


def structural_chunking(parsed_blocks):
    """
    Split PDF text based on policy headings.
    """

    chunks = []

    if not parsed_blocks:
        return chunks

    current_heading = "Introduction"
    current_text = []
    current_page = parsed_blocks[0]["page"]
    source = parsed_blocks[0]["source"]

    heading_pattern = re.compile(
        r"^\d+\.\s+.+Policy$",
        re.IGNORECASE
    )

    def save_chunk():
        text = "\n".join(current_text).strip()

        if text:
            chunks.append({
                "text": text,
                "heading": current_heading,
                "page": current_page,
                "source": source,
                "content_type": "text"
            })

    for block in parsed_blocks:
        text = block["text"].strip()

        if not text:
            continue

        if heading_pattern.match(text):
            save_chunk()
            current_heading = text
            current_text = []
            current_page = block["page"]
        else:
            current_text.append(text)

    save_chunk()

    return chunks


def dynamic_chunking(parsed_blocks, max_chunk_size=500):
    """
    Group text blocks dynamically based on character count.
    """

    chunks = []

    if not parsed_blocks:
        return chunks

    current_text = []
    current_length = 0
    current_page = parsed_blocks[0]["page"]
    source = parsed_blocks[0]["source"]

    for block in parsed_blocks:
        text = block["text"].strip()

        if not text:
            continue

        # Start a new chunk when adding text exceeds the limit.
        if current_text and current_length + len(text) + 1 > max_chunk_size:
            chunks.append({
                "text": "\n".join(current_text),
                "page": current_page,
                "source": source,
                "content_type": "text"
            })

            current_text = []
            current_length = 0

        current_text.append(text)
        current_length += len(text) + 1
        current_page = block["page"]

    # Save the remaining text.
    if current_text:
        chunks.append({
            "text": "\n".join(current_text),
            "page": current_page,
            "source": source,
            "content_type": "text"
        })

    return chunks
