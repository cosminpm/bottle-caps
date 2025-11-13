from starlette.requests import Request


def fake_request(path: str) -> Request:
    return Request({
        "type": "http",
        "path": path
    })
