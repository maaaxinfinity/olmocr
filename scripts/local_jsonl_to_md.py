#!/usr/bin/env python3
import argparse
import json
import os
import glob
import logging
import re

# Basic logging setup
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Read local JSONL files, extract text, and write to local .md files.")
    parser.add_argument(
        "jsonl_dir",
        help="Directory containing the input JSONL files.",
    )
    parser.add_argument(
        "output_dir",
        help="Local directory to store output .md files.",
    )
    return parser.parse_args()

def sanitize_for_xml(text):
    """
    Removes characters that are invalid in XML 1.0, except for tab, newline, and carriage return.
    Also removes null bytes.
    """
    if not isinstance(text, str):
        return text # Return as is if not a string

    # Remove null bytes explicitly
    text = text.replace('\x00', '')

    # Regex to match invalid XML characters (control characters excluding \t, \n, \r)
    # Reference: https://www.w3.org/TR/xml/#charsets
    # Valid range: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    # We remove characters in the ranges #x0-#x8, #xB-#xC, #xE-#x1F, #x7F-#x84, #x86-#x9F
    # Simplified approach: remove most control chars except tab, newline, cr
    invalid_xml_chars_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F]')
    return invalid_xml_chars_re.sub('', text)

def main():
    args = parse_args()

    # Ensure local output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory set to: {args.output_dir}")

    # Find all JSONL files in the input directory
    jsonl_files = glob.glob(os.path.join(args.jsonl_dir, "*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} JSONL files in {args.jsonl_dir}")

    if not jsonl_files:
        print("No JSONL files found in the specified directory.")
        return

    for jsonl_path in jsonl_files:
        logger.info(f"Processing JSONL file: {jsonl_path}")
        output_md_content = ""
        source_file_base = os.path.splitext(os.path.basename(jsonl_path))[0].replace('_output', '') # Try to get original base name

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f_in:
                for i, line in enumerate(f_in):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        text_content = record.get("text", "")
                        metadata = record.get("metadata", {})
                        # Use source file from metadata if available for better naming
                        source_file_meta = metadata.get("Source-File")
                        if source_file_meta:
                             source_file_base = os.path.splitext(os.path.basename(source_file_meta))[0]
                             source_file_base = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in source_file_base) # Sanitize

                        sanitized_text = sanitize_for_xml(text_content)

                        # Append page break or separator? Maybe just double newline.
                        output_md_content += sanitized_text + "\n\n"

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON line {i+1} in {jsonl_path}")
                        continue

            if output_md_content.strip():
                # Use the derived base name for the output markdown file
                output_filename = f"{source_file_base}.md"
                output_path = os.path.join(args.output_dir, output_filename)

                try:
                    with open(output_path, "w", encoding="utf-8") as f_out:
                        f_out.write(output_md_content.strip())
                    logger.info(f"Successfully wrote Markdown to {output_path}")
                except IOError as e:
                     logger.error(f"Failed to write Markdown file {output_path}: {e}")

        except Exception as e:
            logger.error(f"Failed to process file {jsonl_path}: {e}")

    logger.info("Done processing all JSONL files.")

if __name__ == "__main__":
    main()