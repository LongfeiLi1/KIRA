from runtime.paths import CODE_ROOT
'One authorized connectivity probe using existing Aliyun settings in memory.\n\nNo persistent credentials, SDK installation, retries or model substitutions.\nThis adapter is independent of the deterministic controller and AutoSizer.\n'
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError
import json, re, time

@dataclass(frozen=True)
class ProviderSettings:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    enabled: bool = True

    def validate(self):
        u = urlsplit(self.base_url)
        if u.scheme != 'https' or u.hostname != 'dashscope.aliyuncs.com' or u.port not in (None, 443) or (u.path.rstrip('/') != '/compatible-mode/v1') or u.query or u.fragment or u.username or u.password:
            raise ValueError('Provider endpoint is outside the authorized existing Aliyun service')
        if not self.enabled:
            raise ValueError('Existing Aliyun provider is disabled')
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError('Existing key is missing')
        if not re.fullmatch('qwen[a-zA-Z0-9._:/-]{0,100}', self.model):
            raise ValueError('Existing Qwen model identifier is missing or unsupported')

class _NoRedirect(HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

class QwenConnectivityAdapter:

    def __init__(self, settings, opener=None):
        settings.validate()
        self._settings = settings
        self._opener = opener or build_opener(_NoRedirect())
        self.attempted_calls = 0

    @classmethod
    def from_existing_config(cls):
        import os
        return cls(ProviderSettings(os.environ.get('KIRA_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'), os.environ.get('KIRA_MODEL', 'qwen3.8-flash'), os.environ.get('KIRA_API_KEY', '')))

    def probe_once(self):
        if self.attempted_calls:
            raise RuntimeError('One connectivity request already attempted; automatic paid retries are forbidden')
        self.attempted_calls += 1
        s = self._settings
        body = {'model': s.model, 'messages': [{'role': 'user', 'content': 'Reply with exactly OK.'}], 'max_tokens': 8, 'temperature': 0, 'enable_thinking': False, 'stream': False}
        req = Request(s.base_url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + s.api_key}, method='POST')
        result = {'provider': 'aliyun_bailian', 'endpoint_hostname': 'dashscope.aliyuncs.com', 'model_id': s.model, 'attempted_calls': 1, 'automatic_retries': 0, 'prompt': 'Reply with exactly OK.', 'max_completion_tokens': 8, 'thinking_requested': False, 'private_task_material_sent': False, 'historical_budget_is_current_balance': False}
        start = time.monotonic()
        try:
            with self._opener.open(req, timeout=20) as response:
                raw = response.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError('Oversized response')
                data = json.loads(raw)
            text = data['choices'][0]['message']['content']
            if not isinstance(text, str):
                raise ValueError('Missing text response')
            clean = text.replace(s.api_key, '[REDACTED]')[:128]
            usage = data.get('usage', {})
            usage = {k: v for k, v in usage.items() if k in {'prompt_tokens', 'completion_tokens', 'total_tokens'} and isinstance(v, int) and (not isinstance(v, bool))}
            result.update(status='success', response_text=clean, usage=usage, reply_matches_probe=clean.strip() == 'OK')
        except HTTPError as exc:
            result.update(status='http_error', http_status=exc.code, error_detail='Response body deliberately not logged')
        except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            result.update(status='failed', error_type=type(exc).__name__, error_detail='No raw exception, request headers, configuration or response body logged')
        result['wall_seconds'] = time.monotonic() - start
        return result
