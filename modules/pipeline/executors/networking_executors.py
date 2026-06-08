from http.server import BaseHTTPRequestHandler, HTTPServer
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import requests
from typing import Callable, Optional, Literal
import json

from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.storage.encoding_util import PipelineDataEncoderDecoder
from modules.pipeline.linking_steps import (
    LinkingStep, ParallelizationType, BatchLinkingStep, LinkingStepResult
)
import warnings

#TODO future work: utilities for easy deployment of processing steps in a microservice architecture


def start_as_http_server(
        processing_step : LinkingStep, 
        server : Optional[Callable[[BaseHTTPRequestHandler], HTTPServer]] = None,
        pipeline_data_endec : Optional[PipelineDataEncoderDecoder] = None,
        n_workers : int | Literal["auto"] = 1
    ):
    warnings.warn("http server mode is experimental")
    # capture self in local variable for use in handler class
    pipeline_data_endec = pipeline_data_endec or PipelineDataEncoderDecoder()
    if processing_step.parallelization_type != ParallelizationType.SIMPLE and n_workers != 1:
        warnings.warn(f"Processing step {processing_step.name} cannot be parallelized, so n_workers must be 1")
        n_workers = 1

    class ProcessingStepHTTPHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def do_GET(self):
            if self.path in ["/status", "/info", "/", ""]:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                status = {
                    "name": processing_step.name,
                    "parallelization_type": processing_step.parallelization_type.value
                }
                self.wfile.write(json.dumps(status).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/apply":
                try:
                    with self.rfile as f:
                        request_data = json.load(f)
                    linking_address = pipeline_data_endec.decode_from_dict(request_data["address"], LinkedAddress)
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.end_headers()
                    return
                processed_address, step_result = processing_step.apply(linking_address)
                response_data = {
                    "address": pipeline_data_endec.encode_as_dict(processed_address),
                    "result": pipeline_data_endec.encode_as_dict(step_result)
                }
                encoded = pipeline_data_endec.encode_as_dict(response_data)
                with self.wfile as f:
                    json.dump(encoded, f)
                self.send_response(200)
                self.end_headers()
            elif self.path == "/apply_batch" and processing_step.parallelization_type == ParallelizationType.BATCH:
                try:
                    with self.rfile as f:
                        request_data = json.load(f)
                    linking_addresses = [pipeline_data_endec.decode_from_dict(addr, LinkedAddress) for addr in request_data]
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.end_headers()
                    return
                processed_addresses_results = processing_step.batch_apply(linking_addresses)
                response_data = [
                    {
                        "address": pipeline_data_endec.encode_as_dict(addr),
                        "result": pipeline_data_endec.encode_as_dict(result)
                    } for addr, result in processed_addresses_results
                ]
                with self.wfile as f:
                    json.dump(encoded, f)
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()
    if server is None:
        server = lambda handler: HTTPServer(("localhost", 0), handler)
    httpd = server(ProcessingStepHTTPHandler)
    print(f"Starting HTTP server for processing step {processing_step.name} on {httpd.server_address}")
    processing_step.initialize()
    try:            
        httpd.serve_forever()
    except:
        print(f"Shutting down HTTP server for processing step {processing_step.name}")
        httpd.shutdown()
        httpd.server_close()
        processing_step.finalize()

def start_as_http_client(
        processing_step : LinkingStep, 
        server_address : str | list[str], 
        pipeline_data_endec : Optional[PipelineDataEncoderDecoder] = None
    ):
    warnings.warn("http client mode is experimental")
    import requests
    pipeline_data_endec = pipeline_data_endec or PipelineDataEncoderDecoder()
    if isinstance(server_address, str):
        server_addresses = [server_address]
    else:
        server_addresses = server_address
    if processing_step.parallelization_type == ParallelizationType.BATCH:
        class ProcessingStepHTTPClient(BatchLinkingStep):
            name = processing_step.name
            batch_size = processing_step.batch_size

            def initialize(self):
                for server in server_addresses:
                    response = requests.get(f"http://{server}/status")
                    if response.status_code == 200:
                        status = response.json()
                        if status["name"] != processing_step.name:
                            raise ValueError(f"Processing step name mismatch in server {server}: expected {processing_step.name}, got {status['name']}")
                    if status["parallelization_type"] != processing_step.parallelization_type.value:
                        raise ValueError(f"Processing step parallelization type mismatch in server {server}: expected {processing_step.parallelization_type.value}, got {status['parallelization_type']}")
                    if status.get("batch_size") != processing_step.batch_size:
                        raise ValueError(f"Processing step batch size mismatch in server {server}: expected {processing_step.batch_size}, got {status.get('batch_size')}")
                    
            def batch_apply(self, addresses: list[LinkedAddress]) -> list[tuple[LinkedAddress, LinkingStepResult]]:
                request_data = [pipeline_data_endec.encode_as_dict(addr) for addr in addresses]
                response = requests.post(f"http://{server_address}/apply_batch", json=request_data)
                if response.status_code == 200:
                    response_data = response.json()
                    results = []
                    for item in response_data:
                        processed_address = pipeline_data_endec.decode_from_dict(item["address"], LinkedAddress)
                        step_result = pipeline_data_endec.decode_from_dict(item["result"], LinkingStepResult)
                        results.append((processed_address, step_result))
                    return results
                else:
                    raise ConnectionError(f"Failed to connect to processing step server at {server_address}, status code: {response.status_code}")
    else:
        class ProcessingStepHTTPClient(LinkingStep):
            name = processing_step.name
            def apply(self, address: LinkedAddress) -> tuple[LinkedAddress, LinkingStepResult]:
                request_data = {
                    "address": pipeline_data_endec.encode_as_dict(address)
                }
                response = requests.post(f"http://{server_address}/apply", json=request_data)
                if response.status_code == 200:
                    response_data = response.json()
                    processed_address = pipeline_data_endec.decode_from_dict(response_data["address"], LinkedAddress)
                    step_result = pipeline_data_endec.decode_from_dict(response_data["result"], LinkingStepResult)
                    return processed_address, step_result
                else:
                    raise ConnectionError(f"Failed to connect to processing step server at {server_address}, status code: {response.status_code}")
    return ProcessingStepHTTPClient()