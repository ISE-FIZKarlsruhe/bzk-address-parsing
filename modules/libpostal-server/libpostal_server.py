"""
Heavy AI assitance on the creation of this script
"""

import json
import logging
import time
from argparse import ArgumentParser

from bottle import Bottle, HTTPResponse, request, run, HTTPError
from postal.parser import parse_address
from postal.expand import expand_address

logging.basicConfig(level=logging.INFO)

app = Bottle()


@app.get("/health")
def health():
    return HTTPResponse(status=204)


def _parse_request_body():
    raw_body = request.body.read() or b""
    body_text = raw_body.decode("utf-8", errors="replace") if raw_body else ""
    try:
        return json.loads(body_text)
    except json.JSONDecodeError as exc:
        logging.error(f"Bad request - JSON decode error: {exc}")
        raise HTTPError(status=400, body="Invalid JSON in request body", exception=exc)


@app.route("/", method=["GET", "POST"])
@app.route("/parse_addresses", method=["GET", "POST"])
def parse_addresses():
    input_addresses = _parse_request_body()
    expand_first = request.query.get("expandFirst", "false").lower() == "true"
    def parse_function(addr):
        if expand_first:
            expanded = expand_address(addr)
            if isinstance(expanded, list) and len(expanded) > 0:
                addr = expanded[0]
            elif isinstance(expanded, str):
                addr = expanded
            else:
                logging.warning(f"Unexpected result from expand_address: {expanded}")
        return parse_address(addr)
        
    results = [parse_function(addr) for addr in input_addresses]
    body = json.dumps(results, ensure_ascii=False)
    return HTTPResponse(
        body=body,
        status=200,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

@app.route("/expand_addresses", method=["GET", "POST"])
def expand_addresses():
    input_addresses = _parse_request_body()

    results = [expand_address(addr) for addr in input_addresses]
    body = json.dumps(results, ensure_ascii=False)
    return HTTPResponse(
        body=body,
        status=200,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )


@app.hook("before_request")
def log_request_start():
    request.environ["request_start_time"] = time.time()


@app.hook("after_request")
def log_request_end():
    start_time = request.environ.get("request_start_time")
    duration_ms = None
    if start_time is not None:
        duration_ms = (time.time() - start_time) * 1000
    duration_text = f"{duration_ms:.1f}ms" if duration_ms is not None else "n/a"
    logging.info(
        "%s %s %s",
        request.method,
        request.path,
        duration_text,
    )

@app.error(404)
def not_found(_):
    return HTTPResponse(status=404)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Libpostal HTTP Server")
    arg_parser.add_argument("--host", type=str, default="0.0.0.0")
    arg_parser.add_argument("--port", type=int, default=7272)
    args = arg_parser.parse_args()
    logging.info(f"Listening on http://{args.host}:{args.port}")
    run(app, host=args.host, port=args.port)
