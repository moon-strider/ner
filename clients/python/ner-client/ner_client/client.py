import ssl
from typing import Any

import httpx
from attrs import define, evolve, field


@define
class Client:
    raise_on_unexpected_status: bool = field(default=False, kw_only=True)
    _base_url: str = field(alias="base_url")
    _cookies: dict[str, str] = field(factory=dict, kw_only=True, alias="cookies")
    _headers: dict[str, str] = field(factory=dict, kw_only=True, alias="headers")
    _timeout: httpx.Timeout | None = field(default=None, kw_only=True, alias="timeout")
    _verify_ssl: str | bool | ssl.SSLContext = field(default=True, kw_only=True, alias="verify_ssl")
    _follow_redirects: bool = field(default=False, kw_only=True, alias="follow_redirects")
    _httpx_args: dict[str, Any] = field(factory=dict, kw_only=True, alias="httpx_args")
    _client: httpx.Client | None = field(default=None, init=False)
    _async_client: httpx.AsyncClient | None = field(default=None, init=False)

    def with_headers(self, headers: dict[str, str]) -> "Client":
        if self._client is not None:
            self._client.headers.update(headers)
        if self._async_client is not None:
            self._async_client.headers.update(headers)
        return evolve(self, headers={**self._headers, **headers})

    def with_cookies(self, cookies: dict[str, str]) -> "Client":
        if self._client is not None:
            self._client.cookies.update(cookies)
        if self._async_client is not None:
            self._async_client.cookies.update(cookies)
        return evolve(self, cookies={**self._cookies, **cookies})

    def with_timeout(self, timeout: httpx.Timeout) -> "Client":
        if self._client is not None:
            self._client.timeout = timeout
        if self._async_client is not None:
            self._async_client.timeout = timeout
        return evolve(self, timeout=timeout)

    def set_httpx_client(self, client: httpx.Client) -> "Client":
        self._client = client
        return self

    def get_httpx_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                cookies=self._cookies,
                headers=self._headers,
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=self._follow_redirects,
                **self._httpx_args,
            )
        return self._client

    def __enter__(self) -> "Client":
        self.get_httpx_client().__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.get_httpx_client().__exit__(*args, **kwargs)

    def set_async_httpx_client(self, async_client: httpx.AsyncClient) -> "Client":
        self._async_client = async_client
        return self

    def get_async_httpx_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                cookies=self._cookies,
                headers=self._headers,
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=self._follow_redirects,
                **self._httpx_args,
            )
        return self._async_client

    async def __aenter__(self) -> "Client":
        await self.get_async_httpx_client().__aenter__()
        return self

    async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
        await self.get_async_httpx_client().__aexit__(*args, **kwargs)


@define
class AuthenticatedClient:
    raise_on_unexpected_status: bool = field(default=False, kw_only=True)
    _base_url: str = field(alias="base_url")
    _cookies: dict[str, str] = field(factory=dict, kw_only=True, alias="cookies")
    _headers: dict[str, str] = field(factory=dict, kw_only=True, alias="headers")
    _timeout: httpx.Timeout | None = field(default=None, kw_only=True, alias="timeout")
    _verify_ssl: str | bool | ssl.SSLContext = field(default=True, kw_only=True, alias="verify_ssl")
    _follow_redirects: bool = field(default=False, kw_only=True, alias="follow_redirects")
    _httpx_args: dict[str, Any] = field(factory=dict, kw_only=True, alias="httpx_args")
    _client: httpx.Client | None = field(default=None, init=False)
    _async_client: httpx.AsyncClient | None = field(default=None, init=False)
    token: str
    prefix: str = "Bearer"
    auth_header_name: str = "Authorization"

    def with_headers(self, headers: dict[str, str]) -> "AuthenticatedClient":
        if self._client is not None:
            self._client.headers.update(headers)
        if self._async_client is not None:
            self._async_client.headers.update(headers)
        return evolve(self, headers={**self._headers, **headers})

    def with_cookies(self, cookies: dict[str, str]) -> "AuthenticatedClient":
        if self._client is not None:
            self._client.cookies.update(cookies)
        if self._async_client is not None:
            self._async_client.cookies.update(cookies)
        return evolve(self, cookies={**self._cookies, **cookies})

    def with_timeout(self, timeout: httpx.Timeout) -> "AuthenticatedClient":
        if self._client is not None:
            self._client.timeout = timeout
        if self._async_client is not None:
            self._async_client.timeout = timeout
        return evolve(self, timeout=timeout)

    def set_httpx_client(self, client: httpx.Client) -> "AuthenticatedClient":
        self._client = client
        return self

    def get_httpx_client(self) -> httpx.Client:
        if self._client is None:
            self._headers[self.auth_header_name] = (
                f"{self.prefix} {self.token}" if self.prefix else self.token
            )
            self._client = httpx.Client(
                base_url=self._base_url,
                cookies=self._cookies,
                headers=self._headers,
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=self._follow_redirects,
                **self._httpx_args,
            )
        return self._client

    def __enter__(self) -> "AuthenticatedClient":
        self.get_httpx_client().__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.get_httpx_client().__exit__(*args, **kwargs)

    def set_async_httpx_client(self, async_client: httpx.AsyncClient) -> "AuthenticatedClient":
        self._async_client = async_client
        return self

    def get_async_httpx_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._headers[self.auth_header_name] = (
                f"{self.prefix} {self.token}" if self.prefix else self.token
            )
            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                cookies=self._cookies,
                headers=self._headers,
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=self._follow_redirects,
                **self._httpx_args,
            )
        return self._async_client

    async def __aenter__(self) -> "AuthenticatedClient":
        await self.get_async_httpx_client().__aenter__()
        return self

    async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
        await self.get_async_httpx_client().__aexit__(*args, **kwargs)
