from __future__ import annotations

from django.templatetags.static import static
from jinja2 import Environment


def url_for(name: str, **kwargs: str) -> str:
    if name == "static":
        path = kwargs.get("path", "").lstrip("/")
        return static(path)
    return "#"


def environment(**options) -> Environment:
    env = Environment(**options)
    env.globals.update(url_for=url_for)
    return env

