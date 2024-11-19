import logging
import argparse
import boto3
import signal
import os
import sys
import time
import subprocess
import hashlib
import json
import base64
import atexit
import asyncio
import httpx
import datetime
import tempfile
import random
import re
import torch

from tqdm import tqdm
from io import BytesIO
from PIL import Image
from pypdf import PdfReader
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Set
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from pdelfin.s3_queue import S3WorkQueue, WorkItem
from pdelfin.s3_utils import expand_s3_glob, get_s3_bytes, get_s3_bytes_with_backoff, parse_s3_path, download_zstd_csv, upload_zstd_csv, download_directory
from pdelfin.data.renderpdf import render_pdf_to_base64png
from pdelfin.prompts import build_finetuning_prompt, PageResponse
from pdelfin.prompts.anchor import get_anchor_text
from pdelfin.check import check_poppler_version
from pdelfin.metrics import MetricsKeeper, WorkerTracker
from pdelfin.version import VERSION

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

sglang_logger = logging.getLogger("sglang")
sglang_logger.propagate = False

file_handler = logging.FileHandler('beakerpipeline-debug.log', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
sglang_logger.addHandler(file_handler)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global s3 clients fo the whole script, we have two separate ones in case your workspace and your pdfs are in different accounts
workspace_s3 = boto3.client('s3')
pdf_s3 = boto3.client('s3')

# Global variables for token statistics
metrics = MetricsKeeper(window=60*5)
tracker = WorkerTracker()

# Process pool for offloading cpu bound work, like calculating anchor texts
process_pool = ProcessPoolExecutor()

SGLANG_SERVER_PORT = 30024

@dataclass(frozen=True)
class PageResult:
    s3_path: str
    page_num: int
    response: PageResponse

    input_tokens: int
    output_tokens: int


async def build_page_query(local_pdf_path: str, page: int, target_longest_image_dim: int, target_anchor_text_len: int, image_rotation: int=0) -> dict:
    MAX_TOKENS = 3000
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"

    # Allow the page rendering to process in the background while we get the anchor text (which blocks the main thread)
    image_base64 = asyncio.to_thread(render_pdf_to_base64png, local_pdf_path, page, target_longest_image_dim=target_longest_image_dim)

    # GET ANCHOR TEXT IS NOT THREAD SAFE!! Ahhhh..... don't try to do it
    # and it's also CPU bound, so it needs to run in a process pool
    loop = asyncio.get_running_loop()
    anchor_text = loop.run_in_executor(process_pool, partial(get_anchor_text, pdf_engine="pdfreport", target_length=target_anchor_text_len), local_pdf_path, page)

    image_base64, anchor_text = await asyncio.gather(image_base64, anchor_text)
    if image_rotation != 0:
        image_bytes = base64.b64decode(image_base64)
        with Image.open(BytesIO(image_bytes)) as img:
            rotated_img = img.rotate(-image_rotation, expand=True)

            # Save the rotated image to a bytes buffer
            buffered = BytesIO()
            rotated_img.save(buffered, format="PNG")

        # Encode the rotated image back to base64
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_finetuning_prompt(anchor_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.8
    }


async def process_page(args, session: httpx.AsyncClient, worker_id: int, pdf_s3_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    COMPLETION_URL = f"http://localhost:{SGLANG_SERVER_PORT}/v1/chat/completions"
    MAX_RETRIES = args.max_page_retries
    
    exponential_backoffs = 0
    local_anchor_text_len = args.target_anchor_text_len
    local_image_rotation = 0
    attempt = 0
    await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "started")

    while attempt < MAX_RETRIES:
        query = await build_page_query(
            pdf_local_path,
            page_num,
            args.target_longest_image_dim,
            local_anchor_text_len,
            image_rotation=local_image_rotation
        )

        try:
            response = await session.post(COMPLETION_URL, json=query)
            if response.status_code == 400:
                raise ValueError(f"Got BadRequestError from server: {response.text}, skipping this response")
            elif response.status_code == 500:
                raise ValueError(f"Got InternalServerError from server: {response.text}, skipping this response")
            else:
                response.raise_for_status()

            base_response_data = response.json()

            if base_response_data["usage"]["total_tokens"] > args.model_max_context:
                local_anchor_text_len = max(1, local_anchor_text_len // 2)
                logger.info(f"Reducing anchor text len to {local_anchor_text_len} for {pdf_s3_path}-{page_num}")
                raise ValueError(f"Response exceeded model_max_context, cannot use this response")
            
            metrics.add_metrics(sglang_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                                sglang_output_tokens=base_response_data["usage"].get("completion_tokens", 0))

            model_response_json = json.loads(base_response_data["choices"][0]["message"]["content"])
            page_response = PageResponse(**model_response_json)

            if not page_response.is_rotation_valid and attempt < MAX_RETRIES - 1:
                logger.info(f"Got invalid_page rotation for {pdf_s3_path}-{page_num} attempt {attempt}, retrying with {page_response.rotation_correction} rotation")
                local_image_rotation = page_response.rotation_correction
                raise ValueError(f"invalid_page rotation for {pdf_s3_path}-{page_num}")

            await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "finished")
            return PageResult(
                pdf_s3_path,
                page_num,
                page_response,
                input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                output_tokens=base_response_data["usage"].get("completion_tokens", 0)
            )
        except (httpx.TimeoutException, asyncio.TimeoutError) as e:
            logger.warning(f"Client error on attempt {attempt} for {pdf_s3_path}-{page_num}: {e}")
            
            # Now we want to do exponential backoff, and not count this as an actual page retry
            # Page retrys are supposed to be for fixing bad results from the model, but actual requests to sglang 
            # are supposed to work. Probably this means that the server is just restarting
            sleep_delay = 10 * (2 ** exponential_backoffs)
            exponential_backoffs += 1
            logger.info(f"Sleeping for {sleep_delay} seconds on {pdf_s3_path}-{page_num} to allow server restart")
            await asyncio.sleep(sleep_delay)
        except asyncio.CancelledError:
            logger.info(f"Process page {pdf_s3_path}-{page_num} cancelled")
            await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "cancelled")
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt} for {pdf_s3_path}-{page_num}: {e}")
            attempt += 1
        except ValueError as e:
            logger.warning(f"ValueError on attempt {attempt} for {pdf_s3_path}-{page_num}: {type(e)} - {e}")
            attempt += 1
        except Exception as e:
            logger.exception(f"Unexpected error on attempt {attempt} for {pdf_s3_path}-{page_num}: {type(e)} - {e}")
            attempt += 1

    logger.error(f"Failed to process {pdf_s3_path}-{page_num} after {MAX_RETRIES} attempts.")
    await tracker.track_work(worker_id, f"{pdf_s3_path}-{page_num}", "errored")
    raise ValueError(f"Could not process {pdf_s3_path}-{page_num} after {MAX_RETRIES} attempts")


async def process_pdf(args, session: httpx.AsyncClient, worker_id: int, pdf_s3_path: str):
    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
        # TODO Switch to aioboto3 or something
        data = await asyncio.to_thread(lambda: get_s3_bytes_with_backoff(pdf_s3, pdf_s3_path))
        tf.write(data)
        tf.flush()

        try:
            reader = PdfReader(tf.name)
            num_pages = reader.get_num_pages()
        except:
            logger.exception(f"Could not count number of pages for {pdf_s3_path}, aborting document")
            return None

        logger.info(f"Got {num_pages} pages to do for {pdf_s3_path} in worker {worker_id}")

        # List to hold the tasks for processing each page
        page_tasks = []
        page_results = []

        try:
            async with asyncio.TaskGroup() as tg:
                for page_num in range(1, num_pages + 1):
                    task = tg.create_task(process_page(args, session, worker_id, pdf_s3_path, tf.name, page_num))
                    page_tasks.append(task)

            # Collect the results from the entire task group, assuming no exceptions
            page_results = [task.result() for task in page_tasks]
            return build_dolma_document(pdf_s3_path, page_results)
        except Exception as e:
            logger.exception(f"Exception in process_pdf for {pdf_s3_path}: {e}")
            # You can't build a dolma doc with even 1 failed page, so just get out of here
            # However, you don't want to propagate an exception higher up and cancel the entire work_group
            return None


def build_dolma_document(pdf_s3_path, page_results):
    # Build the document text and page spans
    document_text = ""
    pdf_page_spans = []
    current_char_pos = 0

    for index, page_result in enumerate(page_results):
        if page_result.response.natural_text is not None:
            content = page_result.response.natural_text + ("\n" if index < len(page_results) - 1 else "")
        else:
            content = ""

        start_pos = current_char_pos
        document_text += content
        current_char_pos = len(document_text)
        pdf_page_spans.append([start_pos, current_char_pos, page_result.page_num])

    if not document_text:
        logger.info(f"No document text for {pdf_s3_path}")
        return None  # Return None if the document text is empty

    # Build the Dolma document
    metadata = {
        "Source-File": pdf_s3_path,
        "pdf-total-pages": len(page_results),
        "total-input-tokens": sum(page.input_tokens for page in page_results),
        "total-output-tokens": sum(page.output_tokens for page in page_results)
    }

    id_ = hashlib.sha1(document_text.encode()).hexdigest()

    dolma_doc = {
        "id": id_,
        "text": document_text,
        "source": "pdelfin",
        "added": datetime.datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "attributes": {
            "pdf_page_numbers": pdf_page_spans
        }
    }
    return dolma_doc


async def worker(args, work_queue: S3WorkQueue, semaphore, worker_id):
    while True:
        # Wait until allowed to proceed
        await semaphore.acquire()

        work_item = await work_queue.get_work()

        if work_item is None:
            logger.info(f"Worker {worker_id} exiting due to empty queue")
            semaphore.release()
            break

        logger.info(f"Worker {worker_id} processing work item {work_item.hash}")
        await tracker.clear_work(worker_id)

        try:    
            async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=1000)) as session:
                async with asyncio.TaskGroup() as tg:      
                    dolma_tasks = [tg.create_task(process_pdf(args, session, worker_id, pdf)) for pdf in work_item.s3_work_paths]
                    logger.info(f"Created all tasks for {work_item.hash}")

                logger.info(f"Finished TaskGroup for worker on {work_item.hash}")

            logger.info(f"Closed ClientSession for {work_item.hash}")

            dolma_docs = []
            for task in dolma_tasks:
                try:
                    result = task.result()
                except:
                    # some dolma doc creations may have failed
                    pass

                if result is not None:
                    dolma_docs.append(result)
            
            logger.info(f"Got {len(dolma_docs)} docs for {work_item.hash}")

            # Write the Dolma documents to a local temporary file in JSONL format
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tf:
                for doc in dolma_docs:
                    tf.write(json.dumps(doc))
                    tf.write('\n')
                tf.flush()

                # Define the output S3 path using the work_hash
                output_s3_path = os.path.join(args.workspace, 'results', f'output_{work_item.hash}.jsonl')

                bucket, key = parse_s3_path(output_s3_path)
                workspace_s3.upload_file(tf.name, bucket, key)

            # Update finished token counts from successful documents
            metrics.add_metrics(finished_input_tokens=sum(doc["metadata"]["total-input-tokens"] for doc in dolma_docs),
                                finished_output_tokens=sum(doc["metadata"]["total-output-tokens"] for doc in dolma_docs))
  
            # Update last batch time
            last_batch_time = time.perf_counter()
        except Exception as e:
            logger.exception(f"Exception occurred while processing work_hash {work_item.hash}: {e}")
        finally:
            await work_queue.mark_done(work_item)
            semaphore.release()


async def sglang_server_task(args, semaphore):
    model_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'pdelfin', 'model')
    download_directory(args.model, model_cache_dir)

    # Check the rope config and make sure it's got the proper key
    with open(os.path.join(model_cache_dir, "config.json"), "r") as cfin:
        config_data = json.load(cfin)

    if "rope_type" in config_data["rope_scaling"]:
        del config_data["rope_scaling"]["rope_type"]
        config_data["rope_scaling"]["type"] = "mrope"

        with open(os.path.join(model_cache_dir, "config.json"), "w") as cfout:
            json.dump(config_data, cfout)

    # Check GPU memory, lower mem devices need a bit less KV cache space because the VLM takes additional memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    mem_fraction_arg = ["--mem-fraction-static", "0.80"] if gpu_memory < 60 else []

    cmd = [
        "python3",
        "-m", "sglang.launch_server",
        "--model-path", model_cache_dir,
        "--chat-template", args.model_chat_template,
        # "--context-length", str(args.model_max_context),  # Commented out due to crashes
        "--port", str(SGLANG_SERVER_PORT),
        "--log-level-http", "warning",
    ]
    cmd.extend(mem_fraction_arg)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Ensure the subprocess is terminated on exit
    def _kill_proc():
        proc.terminate()

    atexit.register(_kill_proc)

    # Shared variables between tasks
    last_running_req, last_queue_req = 0, 0
    server_printed_ready_message = False
    last_semaphore_release = time.time()

    async def process_line(line):
        nonlocal last_running_req, last_queue_req, last_semaphore_release, server_printed_ready_message
        sglang_logger.info(line)

        if "Detected errors during sampling" in line:
            logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
            sys.exit(1)

        if not server_printed_ready_message and "The server is fired up and ready to roll!" in line:
            server_printed_ready_message = True
            last_semaphore_release = time.time()

        match = re.search(r'#running-req: (\d+)', line)
        if match:
            last_running_req = int(match.group(1))

        match = re.search(r'#queue-req: (\d+)', line)
        if match:
            last_queue_req = int(match.group(1))
            logger.info(f"sglang running req: {last_running_req} queue req: {last_queue_req}")

    async def read_stream(stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            line = line.decode('utf-8').rstrip()
            await process_line(line)

    async def timeout_task():
        nonlocal last_running_req, last_queue_req, last_semaphore_release
        try:
            while True:
                await asyncio.sleep(1)
                if server_printed_ready_message and last_queue_req == 0 and time.time() - last_semaphore_release > 30 and semaphore.locked():
                    semaphore.release()
                    last_semaphore_release = time.time()
                    logger.info("Semaphore released, allowing a worker to proceed.")
        except asyncio.CancelledError:
            pass  # Clean up if the task is cancelled

    # Start tasks to read stdout, stderr, and handle timeout logic
    stdout_task = asyncio.create_task(read_stream(proc.stdout))
    stderr_task = asyncio.create_task(read_stream(proc.stderr))
    timeout_task = asyncio.create_task(timeout_task())

    await proc.wait()
    timeout_task.cancel()
    await asyncio.gather(stdout_task, stderr_task, timeout_task, return_exceptions=True)


async def sglang_server_host(args, semaphore):
    while True:
        await sglang_server_task(args, semaphore)
        logger.warning("SGLang server task ended")


async def sglang_server_ready():
    max_attempts = 300
    delay_sec = 1
    url = f'http://localhost:{SGLANG_SERVER_PORT}/v1/models'

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)

                if response.status_code == 200:
                    logger.info("sglang server is ready.")
                    return
                else:
                    logger.info(f"Attempt {attempt}: Unexpected status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Attempt {attempt}: {e}")

        await asyncio.sleep(delay_sec)

    raise Exception("sglang server did not become ready after waiting.")


async def metrics_reporter(work_queue):
    while True:
        # Leading newlines preserve table formatting in logs
        logger.info(f"Queue remaining: {work_queue.size}")
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)


def submit_beaker_job(args):
    from beaker import (
        Beaker,
        Constraints,
        DataMount,
        DataSource,
        EnvVar,
        ExperimentSpec,
        ImageSource,
        Priority,
        ResultSpec,
        SecretNotFound,
        TaskContext,
        TaskResources,
        TaskSpec,
    )
    
    b = Beaker.from_env(default_workspace=args.beaker_workspace)
    account = b.account.whoami()
    owner = account.name
    beaker_image = f"jakep/pdelfin-inference-{VERSION}"

    task_name = f"pdelfin-{os.path.basename(args.workspace.rstrip('/'))}"

    # Take out --beaker flag so the workers will just run things
    args_list = [arg for arg in sys.argv[1:] if arg != "--beaker"]

    # Take out the --pdfs [arg] or --pdfs=[arg], since the queue is populated locally
    args_list = [arg for i, arg in enumerate(args_list) if not (arg.startswith("--pdfs") or (i > 0 and args_list[i-1] == "--pdfs"))]

    try:
        b.secret.get(f"{owner}-WEKA_ACCESS_KEY_ID", args.beaker_workspace)
        b.secret.get(f"{owner}-WEKA_SECRET_ACCESS_KEY", args.beaker_workspace)
        b.secret.get(f"{owner}-AWS_CREDENTIALS_FILE", args.beaker_workspace)
    except SecretNotFound:
        print(f"Expected beaker secrets for accessing Weka and S3 are not found. Are you okay to write those to your beaker workspace {args.beaker_workspace}? [y/n]")

        if input().strip().lower() != "y":
            print("Exiting...")
            sys.exit(1)

        b.secret.write(f"{owner}-WEKA_ACCESS_KEY_ID", os.environ.get("WEKA_ACCESS_KEY_ID", ""), args.beaker_workspace)
        b.secret.write(f"{owner}-WEKA_SECRET_ACCESS_KEY", os.environ.get("WEKA_SECRET_ACCESS_KEY", ""), args.beaker_workspace)
        b.secret.write(f"{owner}-AWS_CREDENTIALS_FILE", open(os.path.join(os.path.expanduser('~'), '.aws', 'credentials')).read(), args.beaker_workspace)

    try:
        b.secret.get(f"OE_DATA_GCS_SA_KEY", args.beaker_workspace)
    except SecretNotFound:
        print("Input the olmo-gcs SA key if you would like to load weights from gcs (end with a double newline):")
        lines = []
        prev_empty = False
        for line in iter(input, None):
            if not line and prev_empty:
                break
            prev_empty = not line
            lines.append(line)
        gcs_sa_key = "\n".join(lines[:-1]).strip()  # Remove the last empty line
        if gcs_sa_key:
            b.secret.write(f"OE_DATA_GCS_SA_KEY", gcs_sa_key, args.beaker_workspace)

    # Create the experiment spec
    experiment_spec = ExperimentSpec(
        budget="ai2/oe-data",
        description=task_name,
        tasks=[
            TaskSpec(
                name=task_name,
                propagate_failure=False,
                propagate_preemption=False,
                replicas=args.beaker_gpus,
                context=TaskContext(
                    priority=Priority(args.beaker_priority),
                    preemptible=True,
                ),
                image=ImageSource(beaker=beaker_image),
                command=["python", "-m", "pdelfin.beakerpipeline"] + args_list,
                env_vars=[
                    EnvVar(name="BEAKER_JOB_NAME", value=task_name),
                    EnvVar(name="OWNER", value=owner),
                    EnvVar(name="WEKA_ACCESS_KEY_ID", secret=f"{owner}-WEKA_ACCESS_KEY_ID"),
                    EnvVar(name="WEKA_SECRET_ACCESS_KEY", secret=f"{owner}-WEKA_SECRET_ACCESS_KEY"),
                    EnvVar(name="AWS_CREDENTIALS_FILE", secret=f"{owner}-AWS_CREDENTIALS_FILE"),
                    EnvVar(name="GOOGLE_APPLICATION_CREDENTIALS_FILE", secret=f"OE_DATA_GCS_SA_KEY"),
                ],
                resources=TaskResources(gpu_count=1),
                constraints=Constraints(cluster=args.beaker_cluster if isinstance(args.beaker_cluster, list) else [args.beaker_cluster]),
                result=ResultSpec(path="/noop-results"),
            )
        ],
    )
    
    experiment_data = b.experiment.create(spec=experiment_spec, workspace=args.beaker_workspace)
    
    print(f"Experiment URL: https://beaker.org/ex/{experiment_data.id}")


def print_stats(args):    
    # Get total work items and completed items
    index_file_s3_path = os.path.join(args.workspace, "work_index_list.csv.zstd")
    output_glob = os.path.join(args.workspace, "results", "*.jsonl")
    
    work_queue_lines = download_zstd_csv(workspace_s3, index_file_s3_path)
    done_work_items = expand_s3_glob(workspace_s3, output_glob)
    
    total_items = len([line for line in work_queue_lines if line.strip()])
    completed_items = len(done_work_items)
    
    print(f"\nWork Items Status:")
    print(f"Total work items: {total_items:,}")
    print(f"Completed items: {completed_items:,}")
    print(f"Remaining items: {total_items - completed_items:,}")
    
    def process_output_file(s3_path):
        try:
            data = get_s3_bytes(workspace_s3, s3_path)
            doc_count = 0
            total_input_tokens = 0
            total_output_tokens = 0
            total_pages = 0
            processed_paths = set()
            
            for line in data.decode('utf-8').splitlines():
                if line.strip():
                    doc = json.loads(line)
                    doc_count += 1
                    total_input_tokens += doc["metadata"]["total-input-tokens"]
                    total_output_tokens += doc["metadata"]["total-output-tokens"]
                    total_pages += doc["metadata"]["pdf-total-pages"]
                    processed_paths.add(doc["metadata"]["Source-File"])
                    
            return doc_count, total_input_tokens, total_output_tokens, total_pages, processed_paths
        except Exception as e:
            logger.warning(f"Error processing {s3_path}: {e}")
            return 0, 0, 0, 0, set()
    
    print("\nProcessing output files...")
    docs_total = 0
    input_tokens_total = 0
    output_tokens_total = 0
    pages_total = 0
    all_processed_paths = set()
    original_paths = set()
    
    # First collect all original PDF paths
    for line in work_queue_lines:
        if line.strip():
            paths = line.strip().split(',')
            original_paths.update(paths[1:])
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_output_file, item): item for item in done_work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            doc_count, input_tokens, output_tokens, pages, processed_paths = future.result()
            docs_total += doc_count
            input_tokens_total += input_tokens
            output_tokens_total += output_tokens
            pages_total += pages
            all_processed_paths.update(processed_paths)
    
    skipped_paths = original_paths - all_processed_paths
    
    print(f"\nResults:")
    print(f"Total documents processed: {docs_total:,}")
    print(f"Total documents skipped: {len(skipped_paths):,}")
    print(f"Total pages processed: {pages_total:,}")
    
    print(f"\nTotal output tokens: {output_tokens_total:,}")

    print(f"\nAverage pages per doc: {pages_total/max(1,docs_total):,.1f}")
    print(f"Average output tokens per doc: {output_tokens_total/max(1,docs_total):,.1f}")
    print(f"Average output tokens per page: {output_tokens_total/max(1,pages_total):,.1f}")


async def main():
    parser = argparse.ArgumentParser(description='Manager for running millions of PDFs through a batch inference pipeline')
    parser.add_argument('workspace', help='The S3 path where work will be done e.g., s3://bucket/prefix/')
    parser.add_argument('--pdfs', help='Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths', default=None)
    parser.add_argument('--workspace_profile', help='S3 configuration profile for accessing the workspace', default=None)
    parser.add_argument('--pdf_profile', help='S3 configuration profile for accessing the raw pdf documents', default=None)
    parser.add_argument('--pages_per_group', type=int, default=500, help='Aiming for this many pdf pages per work item group')
    parser.add_argument('--max_page_retries', type=int, default=8, help='Max number of times we will retry rendering a page')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers to run at a time')
    parser.add_argument('--stats', action='store_true', help='Instead of running any job, reports some statistics about the current workspace')

    # Model parameters
    parser.add_argument('--model', help='List of paths where you can find the model to convert this pdf. You can specify several different paths here, and the script will try to use the one which is fastest to access',
                         default=["weka://oe-data-default/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/best_bf16/",
                                  "gs://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/",
                                  "s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"])
    parser.add_argument('--model_max_context', type=int, default="8192", help="Maximum context length that the model was fine tuned under")
    parser.add_argument('--model_chat_template', type=str, default="qwen2-vl", help="Chat template to pass to sglang server")
    parser.add_argument('--target_longest_image_dim', type=int, help='Dimension on longest side to use for rendering the pdf pages', default=1024)
    parser.add_argument('--target_anchor_text_len', type=int, help='Maximum amount of anchor text to use (characters)', default=6000)

    # Beaker/job running stuff
    parser.add_argument('--beaker', action='store_true', help='Submit this job to beaker instead of running locally')
    parser.add_argument('--beaker_workspace', help='Beaker workspace to submit to', default='ai2/pdelfin')
    parser.add_argument('--beaker_cluster', help='Beaker clusters you want to run on', default=["ai2/jupiter-cirrascale-2", "ai2/pluto-cirrascale", "ai2/neptune-cirrascale", "ai2/saturn-cirrascale", "ai2/augusta-google-1"])
    parser.add_argument('--beaker_gpus', type=int, default=1, help="Number of gpu replicas to run")
    parser.add_argument('--beaker_priority', type=str, default="normal", help="Beaker priority level for the job")
    args = parser.parse_args()

    global workspace_s3, pdf_s3

    # setup the job to work in beaker environment, load secrets, adjust logging, etc.
    if "BEAKER_JOB_NAME" in os.environ:
        sglang_logger.addHandler(console_handler)
        cred_path = os.path.join(os.path.expanduser('~'), '.aws', 'credentials')
        os.makedirs(os.path.dirname(cred_path), exist_ok=True)
        with open(cred_path, "w") as f:
            f.write(os.environ.get("AWS_CREDENTIALS_FILE"))
        cred_path = os.path.join(os.path.expanduser('~'), '.gcs', 'credentials')
        os.makedirs(os.path.dirname(cred_path), exist_ok=True)
        with open(cred_path, "w") as f:
            f.write(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_FILE"))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        workspace_s3 = boto3.client('s3')
        pdf_s3 = boto3.client('s3')

    if args.workspace_profile:
        workspace_session = boto3.Session(profile_name=args.workspace_profile)
        workspace_s3 = workspace_session.client("s3")

    if args.pdf_profile:
        pdf_session = boto3.Session(profile_name=args.pdf_profile)
        pdf_s3 = pdf_session.client("s3")

    check_poppler_version()

    # Create work queue
    work_queue = S3WorkQueue(workspace_s3, args.workspace)

    if args.pdfs:
        logger.info("Got --pdfs argument, going to add to the work queue")

        # Expand s3 paths
        if args.pdfs.startswith("s3://"):
            logger.info(f"Expanding s3 glob at {args.pdfs}")
            s3_work_paths = expand_s3_glob(pdf_s3, args.pdfs)
        elif os.path.exists(args.pdfs):
            logger.info(f"Loading file at {args.pdfs}")
            with open(args.pdfs, "r") as f:
                s3_work_paths = list(filter(None, (line.strip() for line in f)))
        else:
            raise ValueError("pdfs argument needs to be either an s3 glob search path, or a local file contains pdf paths (one per line)")

        s3_work_paths = set(s3_work_paths)
        logger.info(f"Found {len(s3_work_paths):,} total pdf paths to add")

        # Estimate average pages per pdf
        sample_size = min(100, len(s3_work_paths))
        sampled_pdfs = random.sample(list(s3_work_paths), sample_size)
        page_counts = []

        for pdf in tqdm(sampled_pdfs, desc="Sampling PDFs to calculate optimal length"):
            try:
                # Download the PDF to a temp file
                with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
                    s3_bucket, s3_key = parse_s3_path(pdf)
                    pdf_s3.download_fileobj(s3_bucket, s3_key, tmp_file)
                    tmp_file.flush()
                    reader = PdfReader(tmp_file.name)
                    page_counts.append(len(reader.pages))
            except Exception as e:
                logger.warning(f"Failed to read {pdf}: {e}")

        if page_counts:
            avg_pages_per_pdf = sum(page_counts) / len(page_counts)
        else:
            logger.warning("Could not read any PDFs to estimate average page count.")
            avg_pages_per_pdf = 10  # Default to 10 pages per PDF if sampling fails

        items_per_group = max(1, int(args.pages_per_group / avg_pages_per_pdf))
        logger.info(f"Calculated items_per_group: {items_per_group} based on average pages per PDF: {avg_pages_per_pdf:.2f}")

        # Now call populate_queue
        await work_queue.populate_queue(s3_work_paths, items_per_group)

    if args.stats:
        print_stats(args)
        return

    if args.beaker:
        submit_beaker_job(args)
        return

    logger.info(f"Starting pipeline with PID {os.getpid()}")

    # Initialize the work queue
    await work_queue.initialize_queue()

    # Create a semaphore to control worker access
    # We only allow one worker to move forward with requests, until the server has no more requests in its queue
    # This lets us get full utilization by having many workers, but also to be outputting dolma docs as soon as possible
    # As soon as one worker is no longer saturating the gpu, the next one can start sending requests
    semaphore = asyncio.Semaphore(1)

    sglang_server = asyncio.create_task(sglang_server_host(args, semaphore))

    await sglang_server_ready()

    metrics_task = asyncio.create_task(metrics_reporter(work_queue))

    # Create worker tasks to process the queue concurrently.
    worker_tasks = []
    for i in range(args.workers):
        task = asyncio.create_task(worker(args, work_queue, semaphore, worker_id=i))
        worker_tasks.append(task)

    # Wait for all worker tasks to finish
    await asyncio.gather(*worker_tasks)

    # Wait for server to stop
    process_pool.shutdown(wait=False)

    sglang_server.cancel()
    metrics_task.cancel()
    logger.info("Work done")


if __name__ == "__main__":
    asyncio.run(main())

    # TODO
    # - Add logging of failed pages and have the stats function read them
    # - Sglang commit a fix for the context length issue
    # - pypdf fix for the 'v' error
    # - aiohttp repro and bug report
    # - Get a solid benchmark on the stream vs non stream approach
    