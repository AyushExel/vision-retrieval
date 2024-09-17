import base64
import io
import os
import pyarrow as pa
import pandas as pd
from typing import Optional

import lancedb
import numpy as np
import PIL
import PIL.Image
import requests
import torch
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import (
    process_images,
    process_queries,
)
from tqdm import tqdm
from colpali_engine.utils.image_utils import get_base64_image
from pdf2image import convert_from_path
from pypdf import PdfReader
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


def base64_to_pil(base64_str: str) -> PIL.Image.Image:
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    image = PIL.Image.open(io.BytesIO(image_data))
    return image


def download_pdf(url: str, save_directory: str = "."):
    response = requests.get(url)
    if response.status_code == 200:
        # Check for Content-Disposition header to get the filename
        if "Content-Disposition" in response.headers:
            # Extract filename from header if available
            filename = response.headers.get("Content-Disposition").split("filename=")[-1].strip('"')
        else:
            # Fallback: Use the last part of the URL as filename
            filename = os.path.basename(url)
            # Ensure the file has a .pdf extension
            if not filename.endswith(".pdf"):
                filename += ".pdf"

        # Save the file to the specified directory
        file_path = os.path.join(save_directory, filename)
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"PDF downloaded and saved as {file_path}")
        return file_path
    else:
        raise Exception(f"Failed to download PDF: Status code {response.status_code}")


def get_pdf_images(pdf_path):
    reader = PdfReader(pdf_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)

    images = convert_from_path(pdf_path)
    assert len(images) == len(page_texts)
    return (images, page_texts)


def get_model_colpali(base_model_id: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "vidore/colpali"
    if base_model_id is None:
        base_model_id = "google/paligemma-3b-mix-448"
    model = ColPali.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map=device).eval()
    model.load_adapter(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def get_pdf_embedding(pdf_path: str, model, processor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    page_images, page_texts = get_pdf_images(pdf_path=pdf_path)
    page_embeddings = []
    batch_size = 2
    dataloader = DataLoader(
        page_images,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    i = 0
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        page_embeddings = list(torch.unbind(embeddings_doc.to("cpu")))
        for page_embedding in page_embeddings:
            document = {
                "name": pdf_path,
                "page_idx": i,
                "page_image": page_images[i],
                "page_text": page_texts[i],
                "page_embedding": page_embedding,
            }
            i += 1
        
            yield document


def get_query_embedding(query: str, model, processor):
    dummy_image = PIL.Image.new("RGB", (448, 448), (255, 255, 255))
    dataloader = DataLoader(
        [query],
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_queries(processor, x, dummy_image),
    )

    qs = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    q = {"query": query, "embeddings": qs[0]}
    return q


def embedd_docs(docs_path, model, processor):
    """
    Returns embeddings for all pages in a document.

    Args:
        docs_path (str): Path to the PDF document.
        model (ColPali): Model instance.
        processor (AutoProcessor): Processor instance.
        mode (str): Mode for embedding. Can be "batch" or "streaming".

    """
    docs_to_store_pages = []
    if os.path.isdir(docs_path):
        docs = os.listdir(docs_path)
        docs_path = [os.path.join(docs_path, doc) for doc in docs if not os.path.isdir(doc)]

    for pdf_path in docs_path:
        print(pdf_path)
        pdf_doc = get_pdf_embedding(pdf_path=pdf_path, model=model, processor=processor)
        for batch in tqdm(pdf_doc):
                yield batch


def create_db(docs_storage, table_name: str = "demo", db_path: str = "lancedb"):
    batch = []
    def _gen():
        for x in docs_storage:
            page_embedding_flatten = x["page_embedding"].float().numpy().flatten().tolist()
            #yield pa.RecordBatch.from_arrays(
            #    [
            #        pa.array([x["name"]], pa.string()),
            #        pa.array([x["page_text"]], pa.string()),
            #        pa.array([get_base64_image(x["page_image"])], pa.string()),
            #        pa.array([x["page_idx"]], pa.int64()),
            #        pa.array([page_embedding_flatten], pa.list_(pa.float32(), 131840)),
            #    ],
            #    [
            #        "name",
            #        "page_text",
            #        "image",
            #        "page_idx",
            #        "page_embedding_flatten"
            #    ]
            #)
            yield [{
                "name": x["name"],
                "page_text": x["page_text"],
                "image": get_base64_image(x["page_image"]),
                "page_idx": x["page_idx"],
                "page_embedding_flatten": page_embedding_flatten,
                "page_embedding_shape": list(x["page_embedding"].shape)
            }]
        

    db = lancedb.connect(db_path)
    data = next(_gen())[0]
    schema = pa.schema([
                pa.field("name", pa.string()),
                pa.field("page_text", pa.string()),
                pa.field("image", pa.string()),
                pa.field("page_idx", pa.int64()),
                pa.field("page_embedding_shape", pa.list_(pa.int64())),
                pa.field("page_embedding_flatten", pa.list_(pa.float32(), len(data["page_embedding_flatten"]))),
            ])
    data = _gen()
    table = db.create_table(table_name, schema=schema, data=_gen(), mode="overwrite")
    return table

def flatten_and_zero_pad(tensor, desired_length):
    """Flattens a PyTorch tensor and zero-pads it to a desired length.

    Args:
        tensor: The input PyTorch tensor.
        desired_length: The desired length of the flattened tensor.

    Returns:
        The flattened and zero-padded tensor.
    """

    # Flatten the tensor
    flattened_tensor = tensor.view(-1)

    # Calculate the padding length
    padding_length = desired_length - flattened_tensor.size(0)

    # Check if padding is needed
    if padding_length > 0:
        # Zero-pad the tensor
        padded_tensor = torch.cat([flattened_tensor, torch.zeros(padding_length, dtype=tensor.dtype)], dim=0)
    else:
        # Truncate the tensor if it's already too long
        padded_tensor = flattened_tensor[:desired_length]

    return padded_tensor


def search(query: str, table_name: str, model, processor, db_path: str = "lancedb", top_k: int = 3, fts=False, vector=False, limit=None, where=None):
    qs = get_query_embedding(query=query, model=model, processor=processor)
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    try:
        table.create_fts_index("page_text")
    except Exception:
        pass
    # Search over all dataset
    if vector and fts:
        raise ValueError("can't filter using both fts and vector")
        
    if fts:
        limit = limit or 100
        r = table.search(query, query_type="fts").limit(limit)
    elif vector:
        limit = limit or 100
        vec_q = flatten_and_zero_pad(qs["embeddings"],table.to_pandas()["page_embedding_flatten"][0].shape[0])
        r = table.search(vec_q.float().numpy(), query_type="vector").limit(limit)
    else:
        r = table.search().limit(limit)
    if where:
        r = r.where(where)
    
    r = r.to_list()
    
    def process_patch_embeddings(x):
        patches = np.reshape(x['page_embedding_flatten'], x['page_embedding_shape'])
        return torch.from_numpy(patches).to(torch.bfloat16)
    
    all_pages_embeddings = [process_patch_embeddings(x) for x in r]
    
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate_colbert([qs["embeddings"]], all_pages_embeddings)

    top_k_indices = torch.topk(scores, k=top_k, dim=1).indices

    results = []
    for idx in top_k_indices[0]:
        page = r[idx]
        pil_image = base64_to_pil(page["image"])
        result = {"name": page["name"], "page_idx": page["page_idx"], "pil_image": pil_image}
        results.append(result)
    return results


def get_model_phi_vision(model_id: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_id is None:
        model_id = "microsoft/Phi-3.5-vision-instruct"
    # Note: set _attn_implementation='eager' if you don't have flash_attn installed
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
        # _attn_implementation='flash_attention_2'
        _attn_implementation="eager",
    )
    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
    return model, processor


def run_vision_inference(input_images: PIL.Image, prompt: str, model, processor):
    images = []
    placeholder = ""

    # Note: if OOM, you might consider reduce number of frames in this example.
    for i in range(len(input_images)):
        images.append(input_images[i])
        placeholder += f"<|image_{i + 1}|>\n"

    messages = [
        {"role": "user", "content": f"{placeholder} {prompt}"},
    ]

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 512,
        "temperature": 0.2,
        "do_sample": True,
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    # remove input tokens
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response
