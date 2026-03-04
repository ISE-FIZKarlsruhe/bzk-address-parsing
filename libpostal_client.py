import http.client
import json
import urllib.parse
import time
from utils import ParsedAddressResultBuilder

LIBPOSTAL_LABEL_MAPPING = {
    "house_number": "HouseNumber",
    "road": "StreetName",
    "city": "City",
    "state": "State",
    "country": "Country",
    "postcode": "PostalCode"
}

class LibpostalClient:
    def __init__(self, url: str = "http://localhost:7272", 
                 label_mapping = LIBPOSTAL_LABEL_MAPPING, expand_before_parsing = False,
                 start_container_if_unavailable: bool = True):
        self.url = url
        parsed_url = urllib.parse.urlparse(url)
        self.host = parsed_url.hostname
        self.port = parsed_url.port
        self.label_mapping = label_mapping or {}
        self.expand_before_parsing = expand_before_parsing
        self.auto_started = False
        if start_container_if_unavailable:
            if not self.start_container_if_needed():
                raise ConnectionError(f"Could not connect to libpostal server at {self.url}, and failed to start docker container.")
    
    def _transform_results(self, parsed_addresses : list[list[list[str]]], addresses: list[str]) -> list[dict]:
        results = []
        for parsed, addr in zip(parsed_addresses, addresses):
            result_builder = ParsedAddressResultBuilder(addr)
            for part, label in parsed:
                if label in self.label_mapping:
                    label = self.label_mapping[label]
                result_builder.add_part(label, part)
            results.append(result_builder.build())
        return results
    
    def _handle_request(self, addresses):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=3)
        headers = {'Content-Type': 'application/json'}
        if isinstance(addresses, str):
            addresses = [addresses]
        elif not isinstance(addresses, list):
            addresses = list(addresses)
        body = json.dumps(addresses)
        path = "/parse_addresses"
        if self.expand_before_parsing:
            path += "?expandFirst=true"
        conn.request("GET", path, body, headers)
        response = conn.getresponse()
        data = response.read()
        
        results = json.loads(data.decode('utf-8'))
        conn.close()

        return self._transform_results(results, addresses)
    
    def _health_check(self) -> bool:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=3)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            conn.close()
            return response.status == 204
        except Exception:
            return False
    
    def _start_docker_container(self) -> bool:
        print("Attempting to start libpostal-server docker-compose service...")
        print("This may take a long time on first run since the docker image needs to be built.")
        import subprocess
        import shutil
        compose_program = shutil.which("docker-compose") or shutil.which("podman-compose")
        if compose_program is None: raise Exception("Cannot start libpostal-server service because neither 'docker-compose' nor 'podman-compose' is available.")
        result = subprocess.run(
            [compose_program, "-f", "docker-compose.yml", "up", "-d", "libpostal-server"],
            capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to start libpostal-server docker container (exit code {result.returncode}):")
            print(result.stdout)
            print(result.stderr)
            return False
        for _ in range(10):
                time.sleep(1)
                if self._health_check():
                    print("Libpostal server is now available.")
                    self.auto_started = True
                    return True
        return False
    
    def start_container_if_needed(self):
        if not self._health_check():
            return self._start_docker_container()
        else: return True

    def parse_addresses(self, addresses: list[str]):
        try:
            return self._handle_request(addresses)
        except Exception as e:
            if self._health_check():
                raise e
            else:
                print(f"Libpostal server not reachable at {self.url}.")
                if not self._start_docker_container():
                    raise e
            return self._handle_request(addresses)
    
    def close(self):
        if self.auto_started:
            print("Stopping auto-started libpostal-server docker container...")
            import subprocess
            subprocess.run(["docker-compose", "-f", "docker-compose.yml", "down", "libpostal-server"])