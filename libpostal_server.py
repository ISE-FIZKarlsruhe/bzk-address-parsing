import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from argparse import ArgumentParser
import logging
from postal.parser import parse_address
logging.basicConfig(level=logging.INFO)

class LibpostalServer(BaseHTTPRequestHandler):
    """
    Wrapper server for libpostal API, allowing the library to be accessed on a Docker container via HTTP requests.
    """
    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/health":
            self.send_response(204)
            self.end_headers()
            return
        raw_body = self._read_body()
        body_text = raw_body.decode("utf-8", errors="replace") if raw_body else ""
        try:
            input_addresses = json.loads(body_text)
        except json.JSONDecodeError as e:
            logging.error(f"Bad request - JSON decode error: {e}")
            self.send_response(400)
            self.end_headers()
            return
        
        results = None
        if parsed_path.path in ["/", "/parse_addresses"]:
            results = [parse_address(addr) for addr in input_addresses]
        else:
            logging.error(f"Method not implemented yet: {parsed_path.path}")
            self.send_response(404)
            self.end_headers()
            return

        body = json.dumps(results, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, format, *args):
        logging.info("%s - - [%s] %s\n" %
                     (self.client_address[0],
                      self.log_date_time_string(),
                      format % args))


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Libpostal HTTP Server")
    arg_parser.add_argument("--host", type=str, default="0.0.0.0")
    arg_parser.add_argument("--port", type=int, default=7272)
    args = arg_parser.parse_args()
    httpd = HTTPServer((args.host, args.port), LibpostalServer)
    logging.info(f"Listening on http://{args.host}:{args.port}")
    httpd.serve_forever()
