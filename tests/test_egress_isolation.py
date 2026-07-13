"""Unit tests for HyperAgents meta-agent container egress isolation.

Daemon-free: exercises the pure allowlist/mode/kwargs helpers, a real localhost
round-trip through the CONNECT proxy (allow vs deny), and that ``build_container``
swaps host networking for the internal network + proxy env when isolated.
"""
import socket
import sys
import threading
import time
import types

import pytest

# Ensure the docker submodules utils.docker_utils imports exist. The test env
# may have a partial `docker` package (errors but no models/types), so stub any
# missing pieces rather than only stubbing when docker is wholly absent.
def _ensure_mod(name):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


_docker = _ensure_mod("docker")
try:
    from docker.models.containers import Container  # noqa: F401
except Exception:
    _models = _ensure_mod("docker.models")
    _containers = _ensure_mod("docker.models.containers")
    _containers.Container = object
    _models.containers = _containers
    _docker.models = _models
try:
    from docker.types import Mount, DeviceRequest  # noqa: F401
except Exception:
    _dtypes = _ensure_mod("docker.types")
    if not hasattr(_dtypes, "Mount"):
        _dtypes.Mount = object
    if not hasattr(_dtypes, "DeviceRequest"):
        _dtypes.DeviceRequest = type("DeviceRequest", (object,), {"__init__": lambda self, *a, **k: None})
    _docker.types = _dtypes
if not hasattr(_docker, "errors"):
    _errors = _ensure_mod("docker.errors")
    _errors.NotFound = type("NotFound", (Exception,), {})
    _errors.APIError = type("APIError", (Exception,), {})
    _docker.errors = _errors

from utils import egress
from utils import egress_proxy


# --- pure helpers ---------------------------------------------------------

def test_egress_open_default_is_isolated():
    assert egress.egress_open({}) is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_egress_open_truthy(val):
    assert egress.egress_open({"KCSI_HA_EGRESS_OPEN": val}) is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off"])
def test_egress_open_falsy(val):
    assert egress.egress_open({"KCSI_HA_EGRESS_OPEN": val}) is False


def test_derive_allowlist_always_includes_pypi():
    allow = egress.derive_allowlist({"ANTHROPIC_API_KEY": "x"})
    assert "pypi.org" in allow and "files.pythonhosted.org" in allow


def test_derive_allowlist_provider_hosts_from_creds():
    assert "api.anthropic.com" in egress.derive_allowlist({"ANTHROPIC_API_KEY": "x"})
    assert "api.openai.com" in egress.derive_allowlist({"OPENAI_API_KEY": "x"})


def test_derive_allowlist_never_includes_github():
    allow = egress.derive_allowlist({"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"})
    assert "github.com" not in allow


def test_derive_allowlist_extra_hosts():
    allow = egress.derive_allowlist({"ANTHROPIC_API_KEY": "x", "KCSI_HA_EGRESS_ALLOW": "a.com, b.com"})
    assert "a.com" in allow and "b.com" in allow


def test_derive_allowlist_fallback_when_no_creds():
    allow = egress.derive_allowlist({})
    assert "api.anthropic.com" in allow and "api.openai.com" in allow


def test_agent_run_kwargs_open_is_empty():
    assert egress.agent_run_kwargs(None) == {}


def test_agent_run_kwargs_isolated():
    infra = egress.EgressInfra("int", "ext", "proxyhost", 8080)
    kw = egress.agent_run_kwargs(infra)
    assert kw["network"] == "int"
    assert kw["dns"] == ["0.0.0.0"]
    assert kw["environment"]["HTTPS_PROXY"] == "http://proxyhost:8080"


# --- proxy allowlist logic ------------------------------------------------

def test_host_allowed_exact_and_subdomain():
    allow = {"api.anthropic.com", "amazonaws.com"}
    assert egress_proxy.host_allowed("api.anthropic.com", allow)
    assert egress_proxy.host_allowed("bedrock.us-east-1.amazonaws.com", allow)
    assert not egress_proxy.host_allowed("github.com", allow)


# --- real CONNECT round-trip through the proxy (no Docker) -----------------

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_proxy_connect_allowed_and_denied(monkeypatch):
    proxy_port = _free_port()
    echo_port = _free_port()
    monkeypatch.setattr(egress_proxy, "PORT", proxy_port)
    monkeypatch.setattr(egress_proxy, "ALLOW", {"localhost"})

    threading.Thread(target=egress_proxy.main, daemon=True).start()
    time.sleep(0.3)

    c = socket.create_connection(("127.0.0.1", proxy_port), timeout=5)
    c.sendall(b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com\r\n\r\n")
    assert b"403" in c.recv(1024)
    c.close()

    # echo server for the allowed tunnel
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", echo_port))
    srv.listen(1)

    def serve():
        try:
            conn, _ = srv.accept()
            conn.sendall(conn.recv(1024))
            conn.close()
        except OSError:
            pass

    threading.Thread(target=serve, daemon=True).start()

    c = socket.create_connection(("127.0.0.1", proxy_port), timeout=5)
    c.sendall(f"CONNECT localhost:{echo_port} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
    assert b"200" in c.recv(1024)
    c.sendall(b"ping")
    assert c.recv(1024) == b"ping"
    c.close()
    srv.close()


# --- build_container wiring -----------------------------------------------

class _FakeContainer:
    def start(self):
        pass


class _FakeContainers:
    def __init__(self):
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return _FakeContainer()

    def get(self, name):
        raise sys.modules["docker"].errors.NotFound("nope")


class _FakeImages:
    def list(self):
        return [types.SimpleNamespace(tags=["hyperagents"])]

    def build(self, **kwargs):
        return (types.SimpleNamespace(tags=["hyperagents"]), [])


class _FakeClient:
    def __init__(self):
        self.containers = _FakeContainers()
        self.images = _FakeImages()

    def info(self):
        return {}


def _setup_logger():
    from utils.docker_utils import setup_logger
    setup_logger("/tmp/ha_egress_test.log")


def test_build_container_isolated_drops_host_networking():
    _setup_logger()
    from utils.docker_utils import build_container
    client = _FakeClient()
    infra = egress.EgressInfra("ha-egress-int-1", "ha-egress-ext-1", "ha-egress-proxy-1", 8080)
    build_container(client, "./", "hyperagents", "hyperagents-gl-container-1", egress=infra)
    kw = client.containers.run_kwargs
    assert kw["network"] == "ha-egress-int-1"
    assert kw["dns"] == ["0.0.0.0"]
    assert kw["environment"]["HTTPS_PROXY"] == "http://ha-egress-proxy-1:8080"
    assert "network_mode" not in kw


def test_build_container_open_keeps_host_networking():
    _setup_logger()
    from utils.docker_utils import build_container
    client = _FakeClient()
    build_container(client, "./", "hyperagents", "hyperagents-gl-container-2", egress=None)
    kw = client.containers.run_kwargs
    assert kw.get("network_mode") == "host"
    assert "network" not in kw
